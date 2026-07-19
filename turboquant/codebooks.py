"""Lloyd-Max optimal scalar quantizer codebooks (paper Eq. (4)).

TurboQuant rotates a unit vector x by a Haar-random orthogonal matrix, making
y = Pi @ x uniform on the sphere S^{d-1}. Each coordinate of y then has the
known marginal density (paper Lemma 1)

    f_X(t) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - t^2)^{(d-3)/2}

on [-1, 1] -- a symmetric Beta((d-1)/2, (d-1)/2) affinely mapped to [-1, 1],
with variance exactly 1/d, converging to N(0, 1/d) as d -> infinity. The
codebook is the optimal fixed-rate scalar quantizer for that density: the
solution of a continuous 1-D k-means problem, found by Lloyd-Max iteration.

Both densities here are log-concave (Beta requires d >= 3), so the optimal
scalar quantizer is unique (Fleischer/Trushkin) and Lloyd iteration converges
to the global optimum.

Conventions:
- `gaussian_codebook(bits)` solves for N(0, 1). Its `distortion` equals the
  paper's *normalized* MSE d * C(f_X, b) -- i.e. the per-unit-vector MSE
  targets ~0.36 / 0.117 / 0.0345 / 0.0095 for b = 1..4. Pass `d` to get the
  codebook rescaled by 1/sqrt(d) for actual use (distortion scales by 1/d).
- `sphere_codebook(bits, d)` solves the exact Beta marginal for dimension d.
  Its `distortion` is C(f_X, b); multiply by d (`normalized_distortion`) to
  compare against the paper's numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.special import betainc, betaincinv, gammaln, ndtr, ndtri

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _phi(x: np.ndarray) -> np.ndarray:
    """Standard normal pdf; exact 0 at +/-inf."""
    return np.exp(-0.5 * x * x) / _SQRT_2PI

# Paper Theorem 1 reports Dmse ~= 0.36, 0.117, 0.03, 0.009 for b = 1..4.
# The canonical Lloyd-Max values for a unit-variance Gaussian (Max, 1960) are
# 0.3634, 0.1175, 0.0345, 0.0095; the paper's "0.03" at b = 3 is a loose
# rounding of 0.0345 (its own bounds 4^-3 = 0.0156 and 0.0425 bracket both).
PAPER_DMSE = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}


def shannon_lower_bound(bits: float) -> float:
    """Theorem 3: no b-bit quantizer beats 4^-b on unit vectors."""
    return 4.0 ** (-bits)


def panter_dite_upper_bound(bits: float) -> float:
    """Theorem 1: high-resolution bound (sqrt(3)*pi/2) * 4^-b ~= 2.72 * 4^-b."""
    return math.sqrt(3.0) * math.pi / 2.0 * 4.0 ** (-bits)


@dataclass(frozen=True)
class Codebook:
    """Optimal fixed-rate scalar quantizer for one coordinate distribution."""

    bits: int
    kind: str  # "gaussian" (limit d -> inf) or "sphere" (exact Beta marginal)
    d: int | None  # None => unit-variance Gaussian, not tied to a dimension
    centroids: np.ndarray = field(repr=False)  # (2^bits,), ascending
    boundaries: np.ndarray = field(repr=False)  # (2^bits - 1,) cell edges
    probs: np.ndarray = field(repr=False)  # (2^bits,) cell probabilities
    distortion: float  # E[(X - c_idx(X))^2] under the source density
    iterations: int

    @property
    def normalized_distortion(self) -> float:
        """Per-unit-vector MSE, d * C(f_X, b): the paper's Dmse scale."""
        if self.kind == "gaussian" and self.d is None:
            return self.distortion  # unit variance == already normalized
        assert self.d is not None
        return self.distortion * self.d

    @property
    def entropy_bits(self) -> float:
        """Entropy of the code-index distribution (paper: ~3.8 at b=4)."""
        p = self.probs[self.probs > 0]
        return float(-(p * np.log2(p)).sum())

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Nearest-centroid indices; equal to argmin_k |x - c_k| since
        boundaries are centroid midpoints."""
        return np.searchsorted(self.boundaries, x)

    def decode(self, idx: np.ndarray) -> np.ndarray:
        return self.centroids[idx]


def _lloyd_iterate(
    centroids: np.ndarray,
    cell_prob,
    cell_partial_mean,
    var: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Generic scalar Lloyd-Max fixed point.

    cell_prob(lo, hi) -> P(lo < X <= hi); cell_partial_mean(lo, hi) ->
    E[X * 1{lo < X <= hi}], both vectorized over cell-edge arrays. The centroid
    update c_i = E[X | X in cell_i] is exact (no quadrature, no sampling), so
    the fixed point is limited only by float64 and the tolerance.

    Convergence is linear with a rate that degrades as 2^bits grows: at
    bits=8 the centroid delta floors at ~1.5e-11 (float64 chatter) around
    100k iterations, while the distortion -- a quadratic function of centroid
    error near the optimum -- is stable to 12 digits well before that. Hence
    the default tol of 1e-10: tighter is unreachable, and unnecessary.
    """
    c = centroids
    it = 0
    for it in range(1, max_iter + 1):
        t = 0.5 * (c[:-1] + c[1:])
        lo = np.concatenate(([-np.inf], t))
        hi = np.concatenate((t, [np.inf]))
        p = cell_prob(lo, hi)
        m = cell_partial_mean(lo, hi)
        c_new = m / p
        delta = float(np.max(np.abs(c_new - c)))
        c = c_new
        if delta < tol:
            break
    t = 0.5 * (c[:-1] + c[1:])
    lo = np.concatenate(([-np.inf], t))
    hi = np.concatenate((t, [np.inf]))
    p = cell_prob(lo, hi)
    # At the centroid condition c_i = E[X | cell_i], the distortion collapses
    # to E[X^2] - sum_i p_i c_i^2 (law of total variance).
    distortion = float(var - np.sum(p * c * c))
    return c, t, p, distortion, it


@lru_cache(maxsize=64)
def gaussian_codebook(
    bits: int,
    d: int | None = None,
    tol: float = 1e-10,
    max_iter: int = 150_000,
) -> Codebook:
    """Optimal 2^bits-level quantizer for N(0,1), optionally rescaled to
    N(0, 1/d) -- the d -> infinity limit of the sphere marginal.

    Uses closed-form Gaussian cell statistics: P = Phi(hi) - Phi(lo) and
    E[X * 1{lo<X<=hi}] = phi(lo) - phi(hi).
    """
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    n = 2**bits

    def prob(lo, hi):
        return ndtr(hi) - ndtr(lo)

    def pmean(lo, hi):
        return _phi(lo) - _phi(hi)

    c0 = ndtri((np.arange(n) + 0.5) / n)  # quantile initialization
    c, t, p, dist, iters = _lloyd_iterate(c0, prob, pmean, 1.0, tol, max_iter)
    if d is not None:
        s = 1.0 / math.sqrt(d)
        c, t, dist = c * s, t * s, dist / d
    return Codebook(
        bits=bits, kind="gaussian", d=d, centroids=c, boundaries=t, probs=p,
        distortion=dist, iterations=iters,
    )


@lru_cache(maxsize=64)
def sphere_codebook(
    bits: int,
    d: int,
    tol: float = 1e-10,
    max_iter: int = 150_000,
) -> Codebook:
    """(Memoized in-process AND on disk. A KV cache instantiates one
    quantizer per layer per run -- without the lru cache, a b=8 solve of
    ~100 s repeated across 24 layers turned a 45-second perplexity config
    into a 45-minute one; the disk cache removes the remaining once-per-
    process solve. Set TURBOQUANT_CACHE_DIR to relocate, or
    TURBOQUANT_NO_DISK_CACHE=1 to disable.)"""
    cached = _disk_load(bits, d, tol)
    if cached is not None:
        return cached
    cb = _sphere_codebook_impl(bits, d, tol, max_iter)
    _disk_save(cb, tol)
    return cb


def _disk_cache_path(bits: int, d: int, tol: float):
    import os
    from pathlib import Path

    if os.environ.get("TURBOQUANT_NO_DISK_CACHE"):
        return None
    root = Path(os.environ.get(
        "TURBOQUANT_CACHE_DIR",
        Path.home() / ".cache" / "turboquant"))
    return root / f"sphere_b{bits}_d{d}_tol{tol:.0e}.npz"


def _disk_load(bits: int, d: int, tol: float) -> Codebook | None:
    path = _disk_cache_path(bits, d, tol)
    if path is None or not path.exists():
        return None
    try:
        z = np.load(path)
        return Codebook(
            bits=bits, kind="sphere", d=d,
            centroids=z["centroids"], boundaries=z["boundaries"],
            probs=z["probs"], distortion=float(z["distortion"]),
            iterations=int(z["iterations"]),
        )
    except Exception:
        return None  # corrupt cache entry: fall through to a fresh solve


def _disk_save(cb: Codebook, tol: float) -> None:
    path = _disk_cache_path(cb.bits, cb.d, tol)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, centroids=cb.centroids, boundaries=cb.boundaries,
                 probs=cb.probs, distortion=cb.distortion,
                 iterations=cb.iterations)
    except OSError:
        pass  # read-only filesystem: cache is an optimization, not a need


def _sphere_codebook_impl(
    bits: int,
    d: int,
    tol: float,
    max_iter: int,
) -> Codebook:
    """Optimal quantizer for the exact coordinate marginal of a uniform point
    on S^{d-1}: f_X(t) = C_d * (1 - t^2)^{(d-3)/2} on [-1, 1].

    Cell probabilities come from the regularized incomplete Beta function via
    X = 2B - 1, B ~ Beta((d-1)/2, (d-1)/2). The partial mean has the exact
    antiderivative  int t f_X(t) dt = -C_d (1 - t^2)^{(d-1)/2} / (d - 1),
    so no numerical quadrature is needed anywhere.
    """
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if d < 3:
        raise ValueError("d must be >= 3 (log-concavity of the marginal)")
    n = 2**bits
    a = (d - 1) / 2.0
    log_cd = gammaln(d / 2.0) - gammaln((d - 1) / 2.0) - 0.5 * math.log(math.pi)
    cd = math.exp(log_cd)

    def prob(lo, hi):
        lo_ = np.clip(lo, -1.0, 1.0)
        hi_ = np.clip(hi, -1.0, 1.0)
        return betainc(a, a, 0.5 * (hi_ + 1.0)) - betainc(a, a, 0.5 * (lo_ + 1.0))

    def antideriv(t):
        t_ = np.clip(t, -1.0, 1.0)
        # -C_d/(d-1) * (1 - t^2)^{(d-1)/2}, computed in log space for stability
        with np.errstate(divide="ignore"):
            log1mt2 = np.log1p(-t_ * t_)
        return -cd / (d - 1.0) * np.exp(0.5 * (d - 1.0) * log1mt2)

    def pmean(lo, hi):
        return antideriv(hi) - antideriv(lo)

    c0 = 2.0 * betaincinv(a, a, (np.arange(n) + 0.5) / n) - 1.0
    var = 1.0 / d  # exact: coordinates of a unit vector, exchangeable
    c, t, p, dist, iters = _lloyd_iterate(c0, prob, pmean, var, tol, max_iter)
    return Codebook(
        bits=bits, kind="sphere", d=d, centroids=c, boundaries=t, probs=p,
        distortion=dist, iterations=iters,
    )


def _main() -> None:
    print("Normalized MSE d*C(f_X,b) vs paper (Theorem 1) and bounds")
    print(
        f"{'b':>2} {'gaussian':>12} {'sphere d=64':>12} {'sphere d=1536':>14} "
        f"{'paper':>7} {'LB 4^-b':>10} {'UB 2.72*4^-b':>13} {'iters':>6}"
    )
    for b in range(1, 9):
        g = gaussian_codebook(b)
        s64 = sphere_codebook(b, 64)
        s1536 = sphere_codebook(b, 1536)
        paper = PAPER_DMSE.get(b)
        print(
            f"{b:>2} {g.distortion:>12.6f} {s64.normalized_distortion:>12.6f} "
            f"{s1536.normalized_distortion:>14.6f} "
            f"{paper if paper is not None else '-':>7} "
            f"{shannon_lower_bound(b):>10.6f} {panter_dite_upper_bound(b):>13.6f} "
            f"{g.iterations:>6}"
        )
    g1, g2, g4 = gaussian_codebook(1), gaussian_codebook(2), gaussian_codebook(4)
    print(f"\nb=1 centroids (unit var): {g1.centroids}  [paper: +/-sqrt(2/pi) = "
          f"{math.sqrt(2 / math.pi):.6f}]")
    print(f"b=2 centroids (unit var): {g2.centroids}  [paper: +/-0.453, +/-1.51]")
    print(f"b=4 index entropy: {g4.entropy_bits:.4f} bits  [paper: ~3.8]")


if __name__ == "__main__":
    _main()
