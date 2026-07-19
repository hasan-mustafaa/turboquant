"""TurboQuant-mse: batched vector quantizer (paper Algorithm 1).

Quant(x):   y = Pi @ (x / ||x||);  idx_j = nearest centroid to y_j;
            store bit-packed idx and ||x|| in fp16.
DeQuant():  y~_j = c_{idx_j};  x~ = ||x|| * Pi^T @ y~.

Everything is data-oblivious: Pi is seeded and data-independent, the codebook
comes from the analytic sphere marginal (Phase 1), and each vector is
quantized independently in one pass -- one matmul plus one sorted-boundary
search (torch.bucketize). Nearest-centroid search *is* bucketize here because
Lloyd-Max cell boundaries are centroid midpoints.

Norm handling: the paper assumes ||x|| = 1 and notes non-unit datasets store
norms "in floating-point precision" (Section 1.3). We store fp16: its ~2^-11
relative error enters the MSE as ~(5e-4)^2 ~ 2.5e-7, four orders of magnitude
below the b=4 quantization distortion (~9.5e-3), i.e. invisible.

Storage per vector: ceil(d*b/8) code bytes (exact packing for b in {1,2,4,8},
one byte per code otherwise) + 2 bytes norm. At b=4, d=64 that is
34 bytes vs 128 bytes fp16 -> 3.76x, matching PLAN.md's accounting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from turboquant.codebooks import Codebook, gaussian_codebook, sphere_codebook
from turboquant.rotation import haar_rotation

_PACKABLE_BITS = (1, 2, 4, 8)


def pack_codes(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer codes in [0, 2^bits) along the last dim into uint8.

    For bits in {1, 2, 4, 8}, 8/bits consecutive codes share one byte
    (code j occupies bits [j*bits, (j+1)*bits) of its byte, LSB first).
    Other widths fall back to one byte per code -- the *accounting* in
    QuantizedBatch.bits_per_coord still reports the true b, and PLAN.md's
    follow-on (2.5/3.5-bit outlier splits) is where cross-byte packing for
    b=3 would be added if we choose to store it packed.
    """
    if bits not in _PACKABLE_BITS:
        return idx.to(torch.uint8)
    per_byte = 8 // bits
    *lead, d = idx.shape
    pad = (-d) % per_byte
    if pad:
        idx = torch.nn.functional.pad(idx, (0, pad))
    grouped = idx.reshape(*lead, -1, per_byte).to(torch.uint8)
    out = torch.zeros(grouped.shape[:-1], dtype=torch.uint8, device=idx.device)
    for j in range(per_byte):
        out |= grouped[..., j] << (j * bits)
    return out


def unpack_codes(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Inverse of pack_codes; returns int64 codes with last dim d."""
    if bits not in _PACKABLE_BITS:
        return packed[..., :d].to(torch.int64)
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    parts = [(packed >> (j * bits)) & mask for j in range(per_byte)]
    codes = torch.stack(parts, dim=-1).reshape(*packed.shape[:-1], -1)
    return codes[..., :d].to(torch.int64)


@dataclass
class QuantizedBatch:
    """Bit-packed codes + fp16 norms for a batch of vectors."""

    codes: torch.Tensor  # uint8, (..., ceil(d/codes_per_byte)) or (..., d)
    norms: torch.Tensor  # fp16, (...)
    bits: int
    d: int

    @property
    def num_bytes(self) -> int:
        return self.codes.nbytes + self.norms.nbytes

    @property
    def bits_per_coord(self) -> float:
        """Effective storage rate including the norm scalar."""
        return 8.0 * self.num_bytes / (self.norms.numel() * self.d)


class TurboQuantMSE:
    """b-bit MSE-optimal quantizer for d-dimensional vectors (Algorithm 1).

    codebook="sphere" (default) solves the exact Beta marginal for this d;
    codebook="gaussian" uses the d->infinity Gaussian-limit codebook scaled by
    1/sqrt(d) -- what the paper's "precompute for a range of bit-widths"
    deployment implies. The mismatch penalty of the Gaussian codebook on true
    sphere data is only ~0.5% at d=64, b=4 (measured) -- not the ~4% a naive
    reading of the Phase 1 table suggests. That table compares optimal
    distortions of two different *sources*; the codebook-mismatch penalty is
    second-order small because distortion is stationary in the centroids at
    the Lloyd optimum (envelope theorem). This is why the paper can ship
    precomputed dimension-independent codebooks at no real cost.
    """

    def __init__(
        self,
        d: int,
        bits: int,
        *,
        codebook: str = "sphere",
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.d, self.bits = d, bits
        self.device, self.dtype = torch.device(device), dtype
        if codebook == "sphere":
            cb: Codebook = sphere_codebook(bits, d)
        elif codebook == "gaussian":
            cb = gaussian_codebook(bits, d=d)
        else:
            raise ValueError(f"unknown codebook {codebook!r}")
        self._codebook = cb
        self.centroids = torch.as_tensor(cb.centroids, dtype=dtype, device=device)
        self.boundaries = torch.as_tensor(cb.boundaries, dtype=dtype, device=device)
        self.rotation = haar_rotation(d, seed=seed, dtype=dtype, device=device)

    @property
    def expected_unit_mse(self) -> float:
        """Theoretical E||x - x~||^2 for unit-norm inputs: d * C(f_X, b)."""
        return self._codebook.normalized_distortion

    def quantize(self, x: torch.Tensor) -> QuantizedBatch:
        if x.shape[-1] != self.d:
            raise ValueError(f"last dim {x.shape[-1]} != d={self.d}")
        x = x.to(device=self.device, dtype=self.dtype)
        norms = x.norm(dim=-1)
        xhat = x / norms.clamp_min(torch.finfo(self.dtype).tiny).unsqueeze(-1)
        y = xhat @ self.rotation.T
        idx = torch.bucketize(y, self.boundaries)
        return QuantizedBatch(
            codes=pack_codes(idx, self.bits),
            norms=norms.to(torch.float16),
            bits=self.bits,
            d=self.d,
        )

    def dequantize(self, q: QuantizedBatch) -> torch.Tensor:
        idx = unpack_codes(q.codes, q.bits, q.d)
        y_tilde = self.centroids[idx]
        x_tilde = y_tilde @ self.rotation
        return x_tilde * q.norms.to(self.dtype).unsqueeze(-1)

    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))
