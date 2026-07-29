"""Shared --bits token grammar for the model-quality experiments (03, 04).

Each token names one TurboQuantCache configuration:

    "4"        uniform K+V at 4 bits                  -> b=4
    "8:4"      asymmetric, keys 8 / values 4          -> K8V4
    "2+16x4:8" outlier split on keys: 16 channels at  -> K2+16x4V8
               4 bits, the rest at 2 bits; values 8

The outlier grammar is <bits_k>+<n_outliers>x<bits_outlier>, keys only
(values are nearly free to quantize -- README Phase 4). At head_dim 64 the
paper's 1:3 channel ratio is 16 outliers of 64; "2+16x3" is the paper's
printed 3&2 pattern (2.25 effective key bits), "2+16x4" its label-faithful
4&2 reading (2.5) -- see the arithmetic-discrepancy note in
turboquant/outlier.py.
"""

from __future__ import annotations

import re
from typing import Callable

from turboquant.kv_cache import TurboQuantCache

_OUTLIER = re.compile(r"^(\d+)\+(\d+)x(\d+)$")


def parse_bits_token(token: str) -> tuple[str, Callable[[], TurboQuantCache]]:
    """Return (display name, cache factory) for one --bits token."""
    if ":" in token:
        k_part, v_part = token.split(":", 1)
        bv = int(v_part)
    else:
        k_part, bv = token, None

    m = _OUTLIER.match(k_part)
    if m:
        bk, n_out, b_out = (int(g) for g in m.groups())
        name = f"K{bk}+{n_out}x{b_out}V{bv if bv is not None else bk}"
        kwargs = dict(bits_k=bk, bits_v=bv if bv is not None else bk,
                      outlier_channels=n_out, bits_k_outlier=b_out)
        return name, lambda kw=kwargs: TurboQuantCache(**kw)
    if bv is not None:
        bk = int(k_part)
        return (f"K{bk}V{bv}",
                lambda bk=bk, bv=bv: TurboQuantCache(bits_k=bk, bits_v=bv))
    b = int(k_part)
    return f"b={b}", lambda b=b: TurboQuantCache(bits=b)
