"""Stretch-phase validation: TurboQuant-prod (Algorithm 2) vs Theorem 2.

Ground truth:
- Unbiasedness: E<y, x~> = <y, x> for any fixed y -- the regression slope of
  <y, x~> on <y, x> must be 1 (contrast the mse variant's 1 - Dmse shrinkage,
  pinned in test_quantizer.py::test_shrinkage_identity).
- Distortion: d * Dprod <= (pi/2) * Dmse(b-1)  (paper values ~{1.57, 0.56,
  0.18, 0.047} for b = 1..4; the bound gives {1.571, 0.571, 0.185, 0.054} --
  the paper's finer numbers sit below the bound because pi/2 is not tight
  for every query direction).
- b=1 degenerates to pure QJL (no mse stage).

Pure math + torch; no model, no network. Runtime dominated by Monte-Carlo
sampling (~30 s CPU).
"""

import math

import pytest
import torch

from turboquant.qjl import TurboQuantProd
from turboquant.quantizer import TurboQuantMSE

D = 256
N_X = 20_000
N_Y = 64


def _unit(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True)


class TestTurboQuantProd:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_unbiased_inner_products(self, bits):
        """Regression slope of <y, x~> on <y, x> == 1 within MC error."""
        x = _unit(N_X, D, seed=bits)
        y = _unit(N_Y, D, seed=100 + bits)
        tq = TurboQuantProd(D, bits, seed=7)
        ips = (y @ x.T).flatten()
        ipd = (y @ tq.roundtrip(x).T).flatten()
        slope = float((ips * ipd).sum() / (ips * ips).sum())
        assert slope == pytest.approx(1.0, abs=5e-3), slope
        # and strictly less shrunk than the biased mse variant at the same b
        mse_slope = 1.0 - TurboQuantMSE(D, bits, seed=7).expected_unit_mse
        assert abs(slope - 1.0) < (1.0 - mse_slope) / 4

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_dprod_within_theorem2(self, bits):
        x = _unit(N_X, D, seed=bits)
        y = _unit(N_Y, D, seed=200 + bits)
        tq = TurboQuantProd(D, bits, seed=8)
        err2 = ((y @ (tq.roundtrip(x) - x).T) ** 2).mean().item()
        d_dprod = D * err2
        assert d_dprod <= tq.expected_dprod_ub * 1.05, (d_dprod, tq.expected_dprod_ub)
        # not vacuously small either: within a factor ~2 below the bound
        assert d_dprod >= tq.expected_dprod_ub * 0.5

    def test_b1_is_pure_qjl(self):
        tq = TurboQuantProd(D, 1, seed=3)
        assert tq.mse is None
        q = tq.quantize(_unit(10, D, seed=0))
        assert q.mse is None
        assert q.sign_codes.shape[-1] == math.ceil(D / 8)

    def test_storage_accounting(self):
        n, bits = 100, 4
        tq = TurboQuantProd(D, bits, seed=1)
        q = tq.quantize(_unit(n, D, seed=1))
        # (b-1)*d/8 mse code bytes + d/8 sign bytes + 2x fp16 scalars
        expected = n * ((bits - 1) * D // 8 + D // 8 + 4)
        assert q.num_bytes == expected
        assert q.bits_per_coord == pytest.approx(bits + 32 / D)

    def test_roundtrip_norm_handling(self):
        g = torch.Generator().manual_seed(5)
        x = torch.randn(1_000, D, generator=g) * 7.5
        tq = TurboQuantProd(D, 4, seed=2)
        rel = ((x - tq.roundtrip(x)).pow(2).sum(-1) / x.pow(2).sum(-1)).mean()
        # prod trades a little MSE for unbiasedness: bounded by the b-1 mse
        # distortion plus the QJL residual variance (roughly 2x mse-at-b)
        mse_b1 = TurboQuantMSE(D, 3, seed=2).expected_unit_mse
        assert rel.item() < 2.5 * mse_b1

    def test_deterministic_given_seed(self):
        x = _unit(50, D, seed=9)
        a = TurboQuantProd(D, 3, seed=11).roundtrip(x)
        b = TurboQuantProd(D, 3, seed=11).roundtrip(x)
        c = TurboQuantProd(D, 3, seed=12).roundtrip(x)
        assert torch.equal(a, b)
        assert not torch.allclose(a, c)
