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

Storage per vector: ceil(d*b/8) code bytes (exact bit-level packing for every
b in 1..8) + 2 bytes norm. At b=4, d=64 that is 34 bytes vs 128 bytes fp16
-> 3.76x, matching PLAN.md's accounting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from turboquant.codebooks import Codebook, gaussian_codebook, sphere_codebook
from turboquant.rotation import haar_rotation

def pack_codes(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer codes in [0, 2^bits) along the last dim into uint8.

    Bit-level layout: code j occupies bit positions [j*bits, (j+1)*bits) of a
    little-endian bit stream, so every width b in 1..8 packs to exactly
    ceil(d*b/8) bytes -- including the cross-byte widths (3, 5, 6, 7) needed
    for the paper's mixed-precision outlier-split configs.
    """
    *lead, d = idx.shape
    total_bits = d * bits
    n_bytes = (total_bits + 7) // 8
    shifts = torch.arange(bits, device=idx.device)
    stream = ((idx.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)
    stream = stream.reshape(*lead, total_bits)
    pad = n_bytes * 8 - total_bits
    if pad:
        stream = torch.nn.functional.pad(stream, (0, pad))
    grouped = stream.reshape(*lead, n_bytes, 8)
    out = torch.zeros(grouped.shape[:-1], dtype=torch.uint8, device=idx.device)
    for j in range(8):
        out |= grouped[..., j] << j
    return out


def unpack_codes(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Inverse of pack_codes; returns int64 codes with last dim d."""
    stream = torch.stack(
        [(packed >> j) & 1 for j in range(8)], dim=-1
    ).reshape(*packed.shape[:-1], -1)[..., : d * bits].to(torch.int64)
    per_code = stream.reshape(*packed.shape[:-1], d, bits)
    codes = torch.zeros(per_code.shape[:-1], dtype=torch.int64,
                        device=packed.device)
    for j in range(bits):
        codes |= per_code[..., j] << j
    return codes


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

    def cat(self, other: "QuantizedBatch") -> "QuantizedBatch":
        """Concatenate along the token dimension (codes dim -2, norms dim -1).
        Batches must come from the same quantizer configuration."""
        if (self.bits, self.d) != (other.bits, other.d):
            raise ValueError("cannot cat batches from different quantizers")
        return QuantizedBatch(
            codes=torch.cat([self.codes, other.codes], dim=-2),
            norms=torch.cat([self.norms, other.norms], dim=-1),
            bits=self.bits, d=self.d,
        )


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
