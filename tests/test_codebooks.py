"""Phase 1 validation: the codebook engine against the paper's numbers.

Ground truth used here:
- Theorem 1 (paper): Dmse ~= 0.36, 0.117, 0.03, 0.009 for b = 1..4, with
  bounds 4^-b <= Dmse <= (sqrt(3)pi/2) 4^-b. The b=3 value is a loose rounding;
  the canonical Lloyd-Max Gaussian value is 0.0345 (Max, 1960).
- Closed forms: b=1 distortion is exactly 1 - 2/pi, centroids +/-sqrt(2/pi).
- An independent Monte-Carlo estimate of the distortion (samples never touch
  the solver), guarding against a self-consistent-but-wrong solver.
"""

import math

import numpy as np
import pytest

from turboquant.codebooks import (
    PAPER_DMSE,
    gaussian_codebook,
    panter_dite_upper_bound,
    shannon_lower_bound,
    sphere_codebook,
)

# Canonical Lloyd-Max results for a unit-variance Gaussian (Max 1960; also
# quoted in the paper: +/-sqrt(2/pi) and {+/-0.453, +/-1.51} for b = 1, 2).
CANONICAL_DMSE = {1: 0.363380, 2: 0.117482, 3: 0.034539, 4: 0.009497}
CANONICAL_C1 = math.sqrt(2.0 / math.pi)  # 0.797885
CANONICAL_C2 = (0.45278, 1.51042)


class TestGaussianCodebook:
    def test_b1_closed_form(self):
        cb = gaussian_codebook(1)
        # E[X | X > 0] = sqrt(2/pi) for N(0,1); distortion 1 - 2/pi, exactly.
        np.testing.assert_allclose(
            cb.centroids, [-CANONICAL_C1, CANONICAL_C1], rtol=1e-10
        )
        assert cb.distortion == pytest.approx(1.0 - 2.0 / math.pi, rel=1e-10)

    def test_b2_centroids(self):
        cb = gaussian_codebook(2)
        lo, hi = CANONICAL_C2
        np.testing.assert_allclose(
            cb.centroids, [-hi, -lo, lo, hi], atol=5e-5
        )

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_distortion_table(self, bits):
        # The real gate: canonical Lloyd-Max Gaussian values (Max, 1960).
        cb = gaussian_codebook(bits)
        assert cb.distortion == pytest.approx(CANONICAL_DMSE[bits], abs=5e-5)

    def test_paper_reported_values(self):
        """The paper prints ~0.36, 0.117, 0.03, 0.009. b=1,2 agree at printed
        precision; b=3,4 are truncations of the true optima 0.0345, 0.0095
        (documented discrepancy, PLAN.md section 0c). The honest invariant:
        the paper's printed value never exceeds the true optimum, and the gap
        stays under one unit of its coarsest printed digit."""
        for bits, paper in PAPER_DMSE.items():
            d = gaussian_codebook(bits).distortion
            assert paper <= d + 5e-4
            assert d - paper < 5e-3

    @pytest.mark.parametrize("bits", range(1, 9))
    def test_between_theoretical_bounds(self, bits):
        cb = gaussian_codebook(bits)
        assert cb.distortion >= shannon_lower_bound(bits)
        assert cb.distortion <= panter_dite_upper_bound(bits)

    def test_monotone_in_bits(self):
        d = [gaussian_codebook(b).distortion for b in range(1, 9)]
        assert all(d[i + 1] < d[i] for i in range(len(d) - 1))
        # High-resolution theory: each extra bit divides MSE by ~4.
        ratios = [d[i] / d[i + 1] for i in range(3, 7)]
        assert all(3.5 < r < 4.5 for r in ratios)

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_monte_carlo_cross_check(self, bits):
        rng = np.random.default_rng(0)
        z = rng.standard_normal(5_000_000)
        cb = gaussian_codebook(bits)
        err = z - cb.decode(cb.encode(z))
        assert float(np.mean(err**2)) == pytest.approx(cb.distortion, rel=5e-3)

    def test_probs_sum_to_one(self):
        for b in (1, 4, 8):
            assert gaussian_codebook(b).probs.sum() == pytest.approx(1.0, abs=1e-12)

    def test_scaled_codebook(self):
        d = 64
        unit = gaussian_codebook(4)
        scaled = gaussian_codebook(4, d=d)
        np.testing.assert_allclose(
            scaled.centroids, unit.centroids / math.sqrt(d), rtol=1e-12
        )
        assert scaled.distortion == pytest.approx(unit.distortion / d, rel=1e-12)
        assert scaled.normalized_distortion == pytest.approx(unit.distortion, rel=1e-12)

    def test_entropy_b4(self):
        # Paper Section 3.1: index entropy at b=4 is ~3.8 bits (~5% headroom).
        assert 3.70 < gaussian_codebook(4).entropy_bits < 3.85


class TestSphereCodebook:
    @pytest.mark.parametrize("d", [64, 128, 1536])
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_close_below_gaussian_limit(self, bits, d):
        """The exact Beta marginal is slightly lighter-tailed than its Gaussian
        limit at equal variance 1/d, so its optimal distortion is (a hair)
        smaller, converging up to the Gaussian value as d grows."""
        s = sphere_codebook(bits, d).normalized_distortion
        g = gaussian_codebook(bits).distortion
        assert s <= g * (1.0 + 1e-9)
        tol = 0.05 if d <= 128 else 0.005
        assert s == pytest.approx(g, rel=tol)

    def test_convergence_in_d(self):
        gaps = [
            gaussian_codebook(4).distortion
            - sphere_codebook(4, d).normalized_distortion
            for d in (16, 64, 256, 1024)
        ]
        assert all(g > 0 for g in gaps)
        assert all(gaps[i + 1] < gaps[i] for i in range(len(gaps) - 1))

    @pytest.mark.parametrize("bits", [1, 2, 4])
    def test_monte_carlo_cross_check(self, bits):
        """Sample true sphere coordinates x_1 = 2*Beta(a,a) - 1 and compare the
        empirical distortion with the solver's analytic value."""
        d = 64
        rng = np.random.default_rng(1)
        x = 2.0 * rng.beta((d - 1) / 2.0, (d - 1) / 2.0, size=5_000_000) - 1.0
        cb = sphere_codebook(bits, d)
        err = x - cb.decode(cb.encode(x))
        assert float(np.mean(err**2)) == pytest.approx(cb.distortion, rel=5e-3)

    def test_b1_centroids_match_paper(self):
        # Paper Section 3.1: for moderately high d, b=1 centroids ~ +/-sqrt(2/pi)/sqrt(d).
        d = 1536
        cb = sphere_codebook(1, d)
        np.testing.assert_allclose(
            cb.centroids,
            [-CANONICAL_C1 / math.sqrt(d), CANONICAL_C1 / math.sqrt(d)],
            rtol=2e-3,
        )

    def test_variance_identity(self):
        # sum_i p_i c_i^2 + D = Var(X) = 1/d, exactly, at the fixed point.
        d, bits = 128, 3
        cb = sphere_codebook(bits, d)
        total = float(np.sum(cb.probs * cb.centroids**2)) + cb.distortion
        assert total == pytest.approx(1.0 / d, rel=1e-10)
