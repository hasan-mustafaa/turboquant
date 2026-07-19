"""TurboQuant-prod: unbiased inner-product quantizer (paper Algorithm 2).

The mse quantizer is a shrinkage estimator -- E<y, x~> = (1 - Dmse(b))<y, x>
(measured in Phase 3; the paper proves the b=1 case, factor 2/pi). For
applications needing unbiased inner products (NN search, attention-logit
estimation), Algorithm 2 composes two stages at a total budget of b bits per
coordinate:

  1. TurboQuant-mse at b-1 bits  ->  x~_mse, residual r = x - x~_mse.
     (b=1 degenerates to a pure QJL quantizer: x~_mse = 0, r = x.)
  2. QJL (Zandieh, Daliri, Han -- arXiv:2406.03482) on the residual: store
     sign(S r) with S ~ N(0,1)^{d x d} i.i.d., plus ||r|| in fp16.

  DeQuant: x~ = x~_mse + sqrt(pi/2)/d * ||r|| * S^T sign(S r).

Unbiasedness (paper Thm 2): conditioned on the mse stage,
E_S[sqrt(pi/2)/d * S^T sign(S r)] = r / ||r|| * ||r|| = r exactly, so
E[x~] = x~_mse + r = x. The distortion obeys
Dprod <= (pi/2d) * Dmse(b-1) * ||y||^2 -- i.e. {1.57, 0.57, 0.18, 0.054}/d
for b = 1..4 as an upper bound; the paper's finer values {1.57, 0.56, 0.18,
0.047}/d sit slightly below it because the pi/2 variance bound is not tight
for all query directions.

Storage per vector: (b-1)*d bits of mse codes + d sign bits + two fp16
scalars (||x|| and ||r||) = b*d + 32 bits, i.e. b + 32/d effective bits per
coordinate. sign(0) is mapped to +1 so the code stream is exactly d bits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from turboquant.quantizer import (
    QuantizedBatch,
    TurboQuantMSE,
    pack_codes,
    unpack_codes,
)


def _qjl_matrix(
    d: int,
    seed: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randn(d, d, generator=gen, dtype=torch.float32)
    return s.to(device=device, dtype=dtype)


@dataclass
class ProdQuantizedBatch:
    """Two-stage codes for a batch of vectors: mse stage + QJL residual."""

    mse: QuantizedBatch | None  # None when bits == 1 (pure QJL)
    sign_codes: torch.Tensor  # uint8, (..., ceil(d/8)): packed sign(S r)
    r_norms: torch.Tensor  # fp16, (...): residual norms
    x_norms: torch.Tensor  # fp16, (...): input norms
    bits: int
    d: int

    @property
    def num_bytes(self) -> int:
        n = self.sign_codes.nbytes + self.r_norms.nbytes + self.x_norms.nbytes
        if self.mse is not None:
            # the mse stage's own norm entries are the unit-direction norms
            # (identically 1.0) -- exclude them from accounting, they carry
            # no information and an integrated implementation drops them.
            n += self.mse.codes.nbytes
        return n

    @property
    def bits_per_coord(self) -> float:
        return 8.0 * self.num_bytes / (self.x_norms.numel() * self.d)


class TurboQuantProd:
    """b-bit unbiased inner-product quantizer (Algorithm 2).

    API mirrors TurboQuantMSE: quantize -> ProdQuantizedBatch -> dequantize.
    The estimator property to test is E<y, dequantize(quantize(x))> = <y, x>
    for any fixed y -- see tests/test_qjl.py.
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
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.d, self.bits = d, bits
        self.device, self.dtype = torch.device(device), dtype
        self.mse: TurboQuantMSE | None = (
            TurboQuantMSE(d, bits - 1, codebook=codebook, seed=seed,
                          device=device, dtype=dtype)
            if bits > 1 else None
        )
        # independent randomness for the QJL stage
        self.S = _qjl_matrix(d, seed + 1_000_003, dtype, device)

    @property
    def expected_dprod_ub(self) -> float:
        """Theorem 2 upper bound on d * Dprod for unit x, y: (pi/2)*Dmse(b-1),
        with Dmse(0) = 1 for the pure-QJL case."""
        dmse = self.mse.expected_unit_mse if self.mse is not None else 1.0
        return math.pi / 2.0 * dmse

    def quantize(self, x: torch.Tensor) -> ProdQuantizedBatch:
        if x.shape[-1] != self.d:
            raise ValueError(f"last dim {x.shape[-1]} != d={self.d}")
        x = x.to(device=self.device, dtype=self.dtype)
        x_norms = x.norm(dim=-1)
        xhat = x / x_norms.clamp_min(torch.finfo(self.dtype).tiny).unsqueeze(-1)

        if self.mse is not None:
            mse_q = self.mse.quantize(xhat)
            r = xhat - self.mse.dequantize(mse_q)
        else:
            mse_q = None
            r = xhat
        r_norms = r.norm(dim=-1)
        signs01 = (r @ self.S.T >= 0).to(torch.int64)  # sign(0) -> +1
        return ProdQuantizedBatch(
            mse=mse_q,
            sign_codes=pack_codes(signs01, 1),
            r_norms=r_norms.to(torch.float16),
            x_norms=x_norms.to(torch.float16),
            bits=self.bits,
            d=self.d,
        )

    def dequantize(self, q: ProdQuantizedBatch) -> torch.Tensor:
        if q.bits != self.bits or q.d != self.d:
            raise ValueError(
                f"batch from a (bits={q.bits}, d={q.d}) quantizer cannot be "
                f"decoded by this (bits={self.bits}, d={self.d}) instance")
        signs = unpack_codes(q.sign_codes, 1, self.d).to(self.dtype) * 2.0 - 1.0
        scale = math.sqrt(math.pi / 2.0) / self.d
        r_est = scale * q.r_norms.to(self.dtype).unsqueeze(-1) * (signs @ self.S)
        xhat = r_est if q.mse is None else self.mse.dequantize(q.mse) + r_est
        return xhat * q.x_norms.to(self.dtype).unsqueeze(-1)

    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))
