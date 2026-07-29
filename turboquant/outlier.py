"""Outlier-channel split: the paper's mixed-precision KV recipe (Section 4.3).

Every TurboQuant KV result in the paper's Table 1 uses this, not uniform
quantization: "splitting channels into outlier and non-outlier sets, and
applying two independent instances of TurboQuant to each, allocating higher
bit precision to outliers". Discrepancy flagged, not forced: the paper's
printed config is "(32*3+96*2)/128=2.5" -- verbatim -- but that arithmetic
equals 2.25, not 2.5. Either the label is wrong (they ran 2.25 effective
bits) or the formula is misprinted (32*4+96*2 = 320/128 would give a true
2.5). Both interpretations are implemented and evaluated here; see README.
The paper does not specify the selection rule;
following the outlier-KV literature it cites (RotateKV, and KVQuant-style
magnitude ranking), we rank channels by a scale statistic computed on a
sample and freeze the partition.

Mechanically: the d coordinates are partitioned *before* rotation (a Haar
rotation mixes all channels, so per-channel bit allocation is only possible
by splitting first), and each sub-vector gets its own rotation + Lloyd-Max
codebook + fp16 norm. Effective rate:

    bits/coord = (n_out * b_out + (d - n_out) * b_reg) / d   (+ 2 fp16 norms)

Connection to the Phase 4 findings: the shared-constant component of keys
lives in a few large channels; granting those channels more bits is the
channel-space analogue of the frozen-mu vector centering in kv_cache.py. The
two compose -- in the KV cache the split (when enabled) is applied to the
*centered* keys, so selection ranks channels by residual scale, not by the
constant offset that centering already removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from turboquant.quantizer import QuantizedBatch, TurboQuantMSE


def select_outlier_channels(sample: torch.Tensor, n_outliers: int) -> torch.Tensor:
    """Rank channels of a (..., d) sample by per-channel RMS (computed over
    all leading dims) and return the sorted indices of the top `n_outliers`.

    RMS rather than mean-absolute so that a few extreme tokens count more --
    matching why these channels hurt quantization in the first place.
    """
    d = sample.shape[-1]
    if not 0 < n_outliers < d:
        raise ValueError(f"n_outliers must be in (0, {d})")
    rms = sample.reshape(-1, d).to(torch.float32).pow(2).mean(0).sqrt()
    return torch.sort(rms.topk(n_outliers).indices).values


@dataclass
class SplitQuantizedBatch:
    """Codes for a channel-partitioned batch: outlier + regular sub-vectors."""

    out: QuantizedBatch
    reg: QuantizedBatch
    d: int

    @property
    def num_bytes(self) -> int:
        return self.out.num_bytes + self.reg.num_bytes

    @property
    def bits_per_coord(self) -> float:
        return 8.0 * self.num_bytes / (self.out.norms.numel() * self.d)

    def cat(self, other: "SplitQuantizedBatch") -> "SplitQuantizedBatch":
        return SplitQuantizedBatch(
            out=QuantizedBatch(
                codes=torch.cat([self.out.codes, other.out.codes], dim=-2),
                norms=torch.cat([self.out.norms, other.out.norms], dim=-1),
                bits=self.out.bits, d=self.out.d),
            reg=QuantizedBatch(
                codes=torch.cat([self.reg.codes, other.reg.codes], dim=-2),
                norms=torch.cat([self.reg.norms, other.reg.norms], dim=-1),
                bits=self.reg.bits, d=self.reg.d),
            d=self.d,
        )


class OutlierSplitQuantizer:
    """Two independent TurboQuantMSE instances over a frozen channel
    partition. API mirrors TurboQuantMSE (quantize/dequantize/roundtrip)."""

    def __init__(
        self,
        d: int,
        outlier_idx: torch.Tensor,
        bits_outlier: int,
        bits_regular: int,
        *,
        codebook: str = "sphere",
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        outlier_idx = torch.as_tensor(outlier_idx, dtype=torch.long)
        if outlier_idx.unique().numel() != outlier_idx.numel():
            raise ValueError("outlier_idx contains duplicates")
        if not 0 < outlier_idx.numel() < d:
            raise ValueError("need 0 < n_outliers < d")
        self.d = d
        mask = torch.zeros(d, dtype=torch.bool)
        mask[outlier_idx] = True
        self.idx_out = outlier_idx.to(device)
        self.idx_reg = torch.nonzero(~mask, as_tuple=True)[0].to(device)
        n_out = self.idx_out.numel()
        self.q_out = TurboQuantMSE(n_out, bits_outlier, codebook=codebook,
                                   seed=seed + 1, device=device, dtype=dtype)
        self.q_reg = TurboQuantMSE(d - n_out, bits_regular, codebook=codebook,
                                   seed=seed + 2, device=device, dtype=dtype)
        self.device, self.dtype = torch.device(device), dtype

    @property
    def effective_bits(self) -> float:
        """Code bits per coordinate, excluding the two fp16 norms."""
        n_out = self.idx_out.numel()
        return (n_out * self.q_out.bits
                + (self.d - n_out) * self.q_reg.bits) / self.d

    def quantize(self, x: torch.Tensor) -> SplitQuantizedBatch:
        if x.shape[-1] != self.d:
            raise ValueError(f"last dim {x.shape[-1]} != d={self.d}")
        x = x.to(device=self.device, dtype=self.dtype)
        return SplitQuantizedBatch(
            out=self.q_out.quantize(x[..., self.idx_out]),
            reg=self.q_reg.quantize(x[..., self.idx_reg]),
            d=self.d,
        )

    def dequantize(self, q: SplitQuantizedBatch) -> torch.Tensor:
        part_out = self.q_out.dequantize(q.out)
        part_reg = self.q_reg.dequantize(q.reg)
        y = torch.empty(*part_out.shape[:-1], self.d,
                        device=self.device, dtype=self.dtype)
        y[..., self.idx_out] = part_out
        y[..., self.idx_reg] = part_reg
        return y

    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))
