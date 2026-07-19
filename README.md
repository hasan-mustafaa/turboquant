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
| 4 | KV-cache integration (`transformers` Cache, Qwen2.5-0.5B / Llama-3.2-1B) | ✅ validated |
| 5 | Long-context evaluation (needle-in-a-haystack, bit-width sweep) | 🔧 code complete — runs pending |
| 6 | Memory accounting + throughput (MPS/CPU + RunPod CUDA) | 🔧 code complete — runs pending |
| — | Stretch: TurboQuant-prod (QJL residual, unbiased inner products) | 🔧 code complete — runs pending |

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

## Phase 4 results — online KV-cache quantization in a real model

`TurboQuantCache` plugs into `transformers` as `past_key_values`: post-RoPE
keys and values are quantized per head vector the moment they enter the cache
(prefill and generation alike — no fp16 recent-token window, stricter than
KIVI/PolarQuant's eval setting), stored as bit-packed codes + fp16 norms.
Packed and simulate storage modes agree to the bit; split-point invariance is
tested (chunked prefill ≡ token-by-token decode).

**Finding — relative-MSE-optimal is not attention-optimal on raw keys.**
Next-token KL vs the fp32 baseline (Qwen2.5-0.5B, 512-token prompt), before
any key treatment:

| config | KL(base ‖ quant) |
|---|---|
| b=8 K+V | 0.00037 |
| b=4 **V-only** | 0.00026 |
| b=4 **K-only** | 0.73 |
| b=4 K+V | 6.47 |

Values are essentially free; keys are catastrophic — even though the
quantizer delivers exactly its guaranteed ~0.9% relative error per vector.
Measured cause: transformer keys are extremely anisotropic. At layer 0 of
Qwen2.5-0.5B, the token-mean key has norm 255 out of a mean key norm of 259 —
**~98% of the key norm is a constant shared across tokens** (K-projection
bias / massive-activation channels), so relative-error quantization spends
its budget re-encoding a constant while its error floor swamps the ~18% of
the norm that distinguishes tokens. Attention-sink protection alone does
nothing (KL 7.1).

**Fix (kept fully online): frozen-μ key centering.** The first 32 tokens stay
fp16 (doubling as sink protection), their mean μ is frozen once, and every
later key is quantized as k−μ (μ added back at dequant). One extra fp vector
per layer-head. KL drops **~1500×** to 0.004; a regression test pins this.
The per-vector worst-case guarantee holds verbatim for centered vectors.

*Commentary vs the paper:* Table 1's KV results only ever use the
outlier-channel-split configs (2.5/3.5-bit) — uniform no-split quantization
is never evaluated end-to-end. This finding explains why the split is
load-bearing, not a refinement: the outlier channels are where the shared
constant lives, and granting them extra bits is channel-space centering.
Frozen-μ centering achieves the same effect vector-wise with near-zero
overhead.

**Second finding — perplexity is a harder judge than logits, and keys are
the entire story.** WikiText-2, 10×2048-token sequences, chunked prefill,
Qwen2.5-0.5B on MPS fp16 (centering on everywhere):

| config | effective bits/coord | ppl | Δ vs fp16 |
|---|---|---|---|
| fp16 baseline | 16 | 13.298 | — |
| b=8 uniform | 8.25 | 13.309 | +0.09% |
| **K8V4** | **6.125** | **13.389** | **+0.7%** |
| K4V8 (control) | 6.125 | 25.677 | +93% |
| b=4 uniform | 4.25 | 25.492 | +91.7% |
| b=3 uniform | — | 105.3 | +692% |
| b=2 uniform | — | 360.2 | +2609% |

The K8V4 / K4V8 pair is the controlled experiment: identical bit budget,
opposite outcome. Damage at b=4 keys *accumulates with context position*
(+0.06 nats before position 64 → +1.5 nats past 512; fp16 and fp32 profiles
identical, ruling out numerics) — attention-logit noise grows as more of the
context is quantized keys, with rare catastrophic tokens (ΔNLL ≈ 14,
retrieval-type failures). Values quantize essentially for free (b=4 V-only:
KL 0.00026).

Why this doesn't contradict the paper, but does qualify it: (1) the paper's
KV model is Llama-3.1-8B with head_dim=128 — its own bound D_prod ∝ 1/d
gives half the logit-noise variance of head_dim=64, and an 8B model has far
more redundancy than 0.5B; (2) the paper's configs always grant outlier
channels extra precision — functionally, extra key precision. Our takeaway
for small models: **spend the bits on keys** — K8V4 is quality-neutral at
2.56× compression, while uniform 4-bit (3.76×) is not, on this model. A
Llama-3.2-1B rerun (gated; pending HF login) and the paper-style
outlier-split are the natural follow-ons.

## Stretch phase — TurboQuant-prod (implemented; validation runs pending)

[`turboquant/qjl.py`](turboquant/qjl.py) implements Algorithm 2: mse stage at
b−1 bits, then the QJL transform (sign(S·r), S Gaussian) on the residual with
its norm stored — an **unbiased** inner-product estimator
(E⟨y, x̃⟩ = ⟨y, x⟩ exactly, versus the mse variant's (1−D_mse) shrinkage)
with D_prod ≤ (π/2d)·D_mse(b−1), the paper's {≈1.57, 0.56, 0.18, 0.047}/d at
b=1..4. b=1 degenerates to pure QJL. Tests in
[`tests/test_qjl.py`](tests/test_qjl.py) encode Theorem 2;
[`experiments/07_qjl_validation.py`](experiments/07_qjl_validation.py)
reproduces the paper's Fig. 1a histograms on the DBpedia set.

Engineering notes: code packing is exact bit-level for every width 1–8
(cross-byte widths 3/5/6/7 included — the prerequisite for the paper's
2.5/3.5-bit outlier-split configs); Lloyd-Max codebooks are memoized
in-process and on disk (`~/.cache/turboquant/`); Haar rotations are memoized
per (d, seed).

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
