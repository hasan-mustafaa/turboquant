"""TurboQuant: online, data-oblivious vector quantization (arXiv:2504.19874).

Implemented from the paper. Phase 1 exposes the Lloyd-Max codebook engine;
later phases add the batched quantizer, KV-cache integration, and the
inner-product (QJL-residual) variant.
"""

from turboquant.codebooks import (
    Codebook,
    gaussian_codebook,
    sphere_codebook,
    panter_dite_upper_bound,
    shannon_lower_bound,
    PAPER_DMSE,
)

__all__ = [
    "Codebook",
    "gaussian_codebook",
    "sphere_codebook",
    "panter_dite_upper_bound",
    "shannon_lower_bound",
    "PAPER_DMSE",
]
