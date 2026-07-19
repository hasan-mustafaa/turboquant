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
| 2 | Batched TurboQuant-mse quantizer (PyTorch, bit-packing) | ✅ validated |
| 3 | Distortion-rate validation on DBpedia-OpenAI embeddings (paper §4.1) | ✅ validated |
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

## Phase 2 results — batched quantizer (Algorithm 1) vs theory

End-to-end Monte-Carlo MSE (Haar rotation → bucketize encode → uint8
bit-packing → decode → back-rotation) against the analytic d·C(f_X, b),
on random unit vectors:

| d | b=1 | b=2 | b=3 | b=4 | worst ratio to analytic |
|---|---|---|---|---|---|
| 64 (Llama-3.2-1B head_dim) | 0.358236 | 0.114444 | 0.033353 | 0.009127 | 0.9989 |
| 128 | 0.360905 | 0.115994 | 0.033968 | 0.009316 | 1.0002 |
| 1536 (OpenAI embeddings) | 0.363290 | 0.117353 | 0.034515 | 0.009492 | 1.0007 |

- Every empirical value is within **0.1%** of the analytically-solved optimum,
  and the guarantee also holds for *fixed* worst-case inputs averaged over
  rotations (tested) — the sense in which the paper's Theorem 1 is stated.
- **Haar-ness is load-bearing and tested**: raw LAPACK QR output is *not*
  Haar-distributed; the column sign-fix (Mezzadri 2007) is required, and a
  KS test on rotated fixed vectors against the Lemma 1 marginal guards it.
- **Finding — the paper's precomputed-codebook shortcut is essentially free:**
  using the d→∞ Gaussian-limit codebook on true d=64 sphere data costs only
  +0.47% MSE at b=4 (not the ~4% the source-distortion gap suggests), because
  distortion is stationary in the centroids at the Lloyd optimum — codebook
  perturbations enter only at second order (envelope theorem).
- Storage: 4.25 bits/coord at d=64 (**3.76×** vs fp16), 4.125 at d=128
  (**3.88×**), norm overhead included.
- Throughput (M2, batch 100k, d=128, b=4): 1.5M vec/s CPU, 3.0M vec/s MPS,
  with 100.000% CPU/MPS code agreement.

## Phase 3 results — the paper's §4.1 experiment on its actual dataset

100k DBpedia entities × OpenAI text-embedding-3-large-1536 (the dataset behind
the paper's Fig. 3), 1k held-out queries, streamed sample:

| b | D_mse measured | D_mse analytic | ratio | shrinkage α measured | α = 1−D_mse predicted |
|---|---|---|---|---|---|
| 1 | 0.363406 | 0.363173 | 1.0006 | 0.6344 | 0.6368 (= 2/π) |
| 2 | 0.117485 | 0.117358 | 1.0011 | 0.8812 | 0.8826 |
| 3 | 0.034542 | 0.034499 | 1.0012 | 0.9650 | 0.9655 |
| 4 | 0.009498 | 0.009485 | 1.0014 | 0.9904 | 0.9905 |
| 5 | 0.002504 | 0.002500 | 1.0015 | 0.9974 | 0.9975 |

([Fig. 3 reproduction](results/fig3_reproduction.png) ·
[bias histograms](results/bias_histograms.png))

- **Data-obliviousness, demonstrated**: real-embedding distortion matches the
  worst-case analytic value to ≤0.15% at every bit-width — the guarantee is a
  property of the rotation, not the data, exactly as the theory says.
- **A shrinkage law the paper only states for b=1.** The paper proves
  E⟨y, x̃⟩ = (2/π)⟨y, x⟩ at one bit via the Gaussian sign identity. The Lloyd
  centroid condition gives the general law for free:
  E[X·X̃] = Σᵢ pᵢcᵢ², hence **α_b = 1 − D_mse(b)** at every bit-width —
  2/π = 1 − (1 − 2/π) is the b=1 special case. Measured α matches to ≤0.4%
  (the residual ~0.2% deficit is single-rotation realization noise: the
  identity holds in expectation over Π, and one fixed Π's empirical α
  fluctuates by O(1/√(d·2ᵇ))). This is precisely the bias TurboQuant-prod's
  QJL residual stage exists to remove — and why the mse variant's
  inner-product error (d·D_prod = 2.55 at b=1) is *worse* than the paper's
  unbiased prod variant (1.57): the gap is the bias² term D²·⟨y,x⟩².

Reproduce:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q       # 68 assertions against the paper
.venv/bin/python -m turboquant.codebooks   # Phase 1 table
.venv/bin/python experiments/00_quantizer_validation.py  # Phase 2 report
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
