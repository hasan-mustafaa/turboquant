# TurboQuant — from-the-paper implementation

A rigorous, standalone implementation of **TurboQuant** (Zandieh, Daliri,
Hadian, Mirrokni — *TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate*, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)),
built phase by phase and validated against the paper's reported numbers at
every step. Target application: online 4-bit KV-cache quantization of
Llama-3.2-1B. No wrappers around existing quantization libraries — the
algorithm is implemented from the math in the paper.

**The algorithm in one paragraph.** A unit vector is multiplied by a
Haar-random orthogonal matrix Π, making it uniform on the sphere; each
coordinate of the rotated vector then has a *known* marginal density — a
symmetric Beta((d−1)/2, (d−1)/2) mapped to [−1, 1], → N(0, 1/d) for large d
(paper, Lemma 1). Each coordinate is quantized independently with the optimal
Lloyd-Max scalar quantizer for that density (a continuous 1-D k-means,
paper Eq. 4), and reconstruction is a table lookup followed by Πᵀ. Because the
codebook depends only on the rotation-induced distribution — never on data —
the scheme is **data-oblivious** (no calibration set, no training) and
**online** (each vector quantized independently on arrival, one matmul + one
nearest-centroid search), with worst-case per-vector guarantees:
4⁻ᵇ ≤ D_mse ≤ (√3π/2)·4⁻ᵇ ≈ 2.72·4⁻ᵇ at b bits per coordinate
(paper, Theorems 1 & 3).

## Status

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Lloyd-Max codebook engine (exact Beta + Gaussian limit) | ✅ validated |
| 2 | Batched TurboQuant-mse quantizer (PyTorch, bit-packing) | — |
| 3 | Distortion-rate validation on DBpedia-OpenAI embeddings (paper §4.1) | — |
| 4 | KV-cache integration (Llama-3.2-1B, `transformers` Cache) | — |
| 5 | Long-context evaluation (needle-in-a-haystack, bit-width sweep) | — |
| 6 | Memory accounting + throughput (MPS/CPU + RunPod CUDA) | — |
| — | Stretch: TurboQuant-prod (QJL residual, unbiased inner products) | — |

See [PLAN.md](PLAN.md) for the full technical plan, the distortion-rate math,
and per-phase pass/fail criteria.

## Phase 1 results — codebook engine vs the paper

Normalized MSE `d·C(f_X, b)` (per-unit-vector distortion), solved by exact
Lloyd-Max iteration (closed-form cell statistics, no quadrature):

| b | Gaussian limit | exact Beta d=64 | exact Beta d=1536 | paper (Thm 1) | LB 4⁻ᵇ | UB √3π/2·4⁻ᵇ |
|---|---|---|---|---|---|---|
| 1 | 0.363380 | 0.358387 | 0.363173 | ≈0.36 | 0.25 | 0.680 |
| 2 | 0.117482 | 0.114526 | 0.117358 | ≈0.117 | 0.0625 | 0.170 |
| 3 | 0.034548 | 0.033391 | 0.034499 | ≈0.03 | 0.0156 | 0.0425 |
| 4 | 0.009501 | 0.009132 | 0.009485 | ≈0.009 | 0.0039 | 0.0106 |

- b=1 centroids: ±0.797885 = ±√(2/π); b=2: {±0.45278, ±1.51042} — both match
  the values printed in the paper (§3.1).
- b=4 code-index entropy: 3.765 bits (paper: "approximately 3.8", the ~5%
  entropy-coding headroom the authors deliberately leave on the table).
- Every distortion sits inside the paper's [4⁻ᵇ, 2.72·4⁻ᵇ] band; the exact
  Beta marginal is slightly lighter-tailed than its Gaussian limit, so its
  optimum is marginally lower, converging up to the Gaussian value as d→∞.
- Discrepancy flagged, not forced: the paper's printed "0.03" (b=3) and
  "0.009" (b=4) are truncations of the canonical Lloyd-Max optima 0.0345 and
  0.0095 (Max, 1960) — the paper's own bounds bracket both, and our tests
  assert the precise values.

Reproduce:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q       # 41 assertions against the paper
.venv/bin/python -m turboquant.codebooks   # prints the table above
```

## Source

This is an independent implementation of:

> Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni.
> **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.**
> arXiv:2504.19874, 2025. https://arxiv.org/abs/2504.19874

Related work referenced by the implementation: QJL
([arXiv:2406.03482](https://arxiv.org/abs/2406.03482), the 1-bit residual
quantizer used by TurboQuant-prod) and PolarQuant
([arXiv:2502.02617](https://arxiv.org/abs/2502.02617), a baseline in the
paper's KV-cache experiments).
