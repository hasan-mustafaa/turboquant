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
from turboquant.quantizer import QuantizedBatch, TurboQuantMSE
from turboquant.rotation import haar_rotation

__all__ = [
    "Codebook",
    "gaussian_codebook",
    "sphere_codebook",
    "panter_dite_upper_bound",
    "shannon_lower_bound",
    "PAPER_DMSE",
    "QuantizedBatch",
    "TurboQuantMSE",
    "haar_rotation",
]

# TurboQuantCache / TurboQuantProd are imported from their modules directly
# (turboquant.kv_cache, turboquant.qjl) to keep `import turboquant` free of
# the transformers dependency.
