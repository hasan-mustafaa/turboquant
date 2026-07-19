"""Phase 2 validation: the batched quantizer against Phase 1 theory.

The load-bearing test is Monte-Carlo MSE vs the analytically-solved
d * C(f_X, b): it exercises rotation, encoding, packing, decoding, and norm
handling end to end, and its expected value is a number we derived
independently of this code path (and cross-checked against Max 1960).
Haar-ness gets its own statistical test with a *fixed* input vector, because
TurboQuant's guarantee is worst-case per-vector: it relies on Pi @ x being
uniform on the sphere for every fixed x, not just for random x.
"""

import math

import numpy as np
import pytest
import torch
from scipy.stats import beta as beta_dist
from scipy.stats import kstest

from turboquant.codebooks import sphere_codebook
from turboquant.quantizer import TurboQuantMSE, pack_codes, unpack_codes
from turboquant.rotation import haar_rotation

torch.manual_seed(0)


class TestRotation:
    def test_orthogonal(self):
        q = haar_rotation(128, seed=3, dtype=torch.float64)
        eye = q @ q.T
        assert torch.allclose(eye, torch.eye(128, dtype=torch.float64), atol=1e-12)

    def test_seed_determinism(self):
        a = haar_rotation(64, seed=7)
        b = haar_rotation(64, seed=7)
        c = haar_rotation(64, seed=8)
        assert torch.equal(a, b)
        assert not torch.allclose(a, c)

    def test_haar_marginal_fixed_input(self):
        """For a FIXED x, coordinates of Pi @ x pooled over independent
        rotations must follow the Lemma 1 Beta marginal. The un-sign-fixed
        LAPACK Q fails this for structured x -- this is the regression test
        for the Mezzadri correction."""
        d = 64
        x = torch.ones(d, dtype=torch.float64) / math.sqrt(d)  # structured x
        samples = []
        for seed in range(300):
            pi = haar_rotation(d, seed=seed, dtype=torch.float64)
            samples.append((pi @ x).numpy())
        pooled = np.concatenate(samples)  # 19200 coordinate samples
        a = (d - 1) / 2.0
        stat = kstest(pooled, lambda t: beta_dist.cdf((t + 1.0) / 2.0, a, a))
        assert stat.pvalue > 1e-3, f"KS p={stat.pvalue:.2e}: not Haar"


class TestPacking:
    @pytest.mark.parametrize("bits", list(range(1, 9)))
    def test_roundtrip_exact(self, bits):
        """Every width 1..8 packs to exactly ceil(d*b/8) bytes -- including
        the cross-byte widths (3, 5, 6, 7) used by outlier-split configs."""
        rng = torch.Generator().manual_seed(bits)
        for d in (64, 128, 1536, 61):  # 61: exercises stream padding
            idx = torch.randint(0, 2**bits, (17, d), generator=rng)
            packed = pack_codes(idx, bits)
            assert packed.dtype == torch.uint8
            assert packed.shape[-1] == math.ceil(d * bits / 8)
            assert torch.equal(unpack_codes(packed, bits, d), idx)


class TestQuantizerMSE:
    @pytest.mark.parametrize("d,n", [(64, 40_000), (128, 20_000), (1536, 3_000)])
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_matches_theory_random_inputs(self, bits, d, n):
        """Empirical MSE within ~1.5% of the analytic d * C(f_X, b)."""
        gen = torch.Generator().manual_seed(d * 10 + bits)
        x = torch.randn(n, d, generator=gen)
        x = x / x.norm(dim=-1, keepdim=True)
        tq = TurboQuantMSE(d, bits, seed=1)
        mse = (x - tq.roundtrip(x)).pow(2).sum(-1).mean().item()
        assert mse == pytest.approx(tq.expected_unit_mse, rel=0.015)

    @pytest.mark.parametrize("bits", [1, 4])
    def test_matches_theory_fixed_worst_case_input(self, bits):
        """The paper's guarantee is E over the quantizer's randomness for any
        fixed x. Check a structured x (basis-vector pair) over many seeds."""
        d = 64
        x = torch.zeros(1, d)
        x[0, 0] = x[0, 1] = 1.0 / math.sqrt(2.0)
        errs = []
        for seed in range(200):
            tq = TurboQuantMSE(d, bits, seed=seed)
            errs.append((x - tq.roundtrip(x)).pow(2).sum().item())
        mse = float(np.mean(errs))
        expected = sphere_codebook(bits, d).normalized_distortion
        # 200 seeds x 64 coords -> looser MC tolerance than the pooled test
        assert mse == pytest.approx(expected, rel=0.05)

    def test_scaling_4x_per_bit(self):
        d = 128
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(5_000, d, generator=gen)
        x = x / x.norm(dim=-1, keepdim=True)
        mses = []
        for bits in (2, 3, 4, 5):
            tq = TurboQuantMSE(d, bits, seed=2)
            mses.append((x - tq.roundtrip(x)).pow(2).sum(-1).mean().item())
        ratios = [mses[i] / mses[i + 1] for i in range(len(mses) - 1)]
        assert all(3.2 < r < 4.6 for r in ratios), ratios

    def test_norm_handling(self):
        """Non-unit vectors: relative MSE == unit-vector MSE; fp16 norm
        storage adds nothing measurable."""
        d, bits = 64, 4
        gen = torch.Generator().manual_seed(5)
        x = torch.randn(10_000, d, generator=gen)
        u = torch.rand(10_000, generator=gen)
        scales = torch.exp(math.log(0.1) + u * (math.log(100.0) - math.log(0.1)))
        x = scales.unsqueeze(-1) * x / x.norm(dim=-1, keepdim=True)
        tq = TurboQuantMSE(d, bits, seed=3)
        rel = ((x - tq.roundtrip(x)).pow(2).sum(-1) / x.pow(2).sum(-1)).mean().item()
        assert rel == pytest.approx(tq.expected_unit_mse, rel=0.02)

    def test_zero_vector_safe(self):
        tq = TurboQuantMSE(64, 4)
        x = torch.zeros(3, 64)
        assert torch.equal(tq.roundtrip(x), x)

    def test_storage_accounting(self):
        d, bits, n = 64, 4, 100
        tq = TurboQuantMSE(d, bits)
        q = tq.quantize(torch.randn(n, d))
        assert q.codes.nbytes == n * d * bits // 8
        assert q.num_bytes == n * (d * bits // 8 + 2)  # + fp16 norm
        # 4.25 effective bits/coord at d=64 -> 16/4.25 = 3.76x vs fp16
        assert q.bits_per_coord == pytest.approx(4.25)

    @pytest.mark.parametrize("bits", [1, 2, 4])
    def test_shrinkage_identity(self, bits):
        """The mse quantizer is a shrinkage estimator: E<x, x~> = 1 - Dmse.

        Proof sketch: E<x,x~> = sum_j E[y_j y~_j] = d * sum_i p_i c_i^2
        (centroid condition), and d * (Var - sum p c^2) = Dmse, with
        d * Var = 1. At b=1 this reduces to the paper's 2/pi bias factor:
        1 - (1 - 2/pi) = 2/pi. The paper proves b=1 via the Gaussian sign
        identity; this form gives every bit-width at once, and predicts the
        inner-product bias TurboQuant-prod exists to remove.
        """
        d = 256
        gen = torch.Generator().manual_seed(bits)
        x = torch.randn(30_000, d, generator=gen)
        x = x / x.norm(dim=-1, keepdim=True)
        tq = TurboQuantMSE(d, bits, seed=6)
        alpha = (x * tq.roundtrip(x)).sum(-1).mean().item()
        assert alpha == pytest.approx(1.0 - tq.expected_unit_mse, abs=2e-3)

    def test_gaussian_codebook_slightly_worse_at_small_d(self):
        """The paper's deployment shortcut (precomputed Gaussian-limit
        codebooks) costs well under 1% MSE at d=64 vs the exact Beta codebook:
        distortion is stationary in the centroids at the Lloyd optimum, so a
        codebook perturbation of size eps costs only O(eps^2)."""
        d, bits = 64, 4
        gen = torch.Generator().manual_seed(9)
        x = torch.randn(30_000, d, generator=gen)
        x = x / x.norm(dim=-1, keepdim=True)
        mse_sphere = (x - TurboQuantMSE(d, bits, seed=4).roundtrip(x)).pow(2).sum(-1).mean()
        mse_gauss = (x - TurboQuantMSE(d, bits, seed=4, codebook="gaussian")
                     .roundtrip(x)).pow(2).sum(-1).mean()
        assert mse_sphere.item() < mse_gauss.item()
        assert mse_gauss.item() / mse_sphere.item() < 1.10
