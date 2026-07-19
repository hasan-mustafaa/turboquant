"""Haar-random orthogonal rotations (paper Algorithm 1, line 2).

The paper generates the rotation "by applying QR decomposition on a random
matrix with i.i.d. Normal entries". One subtlety the paper leaves implicit:
the raw Q from LAPACK's QR is *not* Haar-distributed. QR is only unique up to
the signs of R's diagonal, and LAPACK's Householder convention picks signs in
a data-dependent way that biases Q (visible, e.g., in eigenvalue-spacing
statistics). Multiplying each column of Q by sign(R_jj) restores uniqueness
(R with positive diagonal) and yields exactly Haar measure (Mezzadri 2007,
"How to generate random matrices from the classical compact groups").

Haar-ness is not cosmetic here: TurboQuant's guarantee is worst-case over
inputs, and it holds only because Pi @ x is uniform on the sphere for *every*
fixed x. A non-Haar rotation breaks Lemma 1's marginal for adversarial x.
"""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=256)
def _haar_cpu_f64(d: int, seed: int) -> torch.Tensor:
    """Memoized master copy (CPU float64). A KV cache instantiates one
    rotation per layer per run; the QR itself is cheap at head_dim scale but
    re-sampling hundreds of times per sweep adds up at d ~ 1536. Treat the
    returned tensor as immutable."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(d, d, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    s = torch.sign(torch.diagonal(r))
    s = torch.where(s == 0, torch.ones_like(s), s)
    return q * s  # column-wise sign fix -> Haar


def haar_rotation(
    d: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Sample a (d, d) Haar-distributed orthogonal matrix, deterministically
    from `seed`. Computed in float64 on CPU for orthogonality to machine
    precision, then cast to the requested dtype/device."""
    return _haar_cpu_f64(d, seed).to(device=device, dtype=dtype)
