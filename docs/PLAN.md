# TurboQuant-mse Implementation Plan

> Status 2026-07-19 (end of implementation): Phases 1-4 complete, validated,
> pushed. Phase 4 findings: frozen-mu online key centering required (raw
> 4-bit keys catastrophic on Qwen2.5-0.5B — ~98% of key norm is a shared
> constant), and keys-need-precision (K8V4 @ 6.125 bits quality-neutral;
> K4V8 at the same budget reproduces the uniform-b=4 blowup). Phases 5-6 and
> the QJL stretch: code complete, all runs deferred — see the local,
> gitignored docs/RUNBOOK.md for the exact commands, expected numbers, and RunPod
> hosting steps. Llama-3.2-1B runs need `hf auth login` first.

Target: rigorous from-the-paper implementation of TurboQuant-mse (Zandieh, Daliri,
Hadian, Mirrokni — "TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate", arXiv:2504.19874), applied to KV-cache quantization of a small
open LLM at 4-bit, validated against the paper's numbers phase by phase.
TurboQuant-prod (QJL residual) is a stretch goal.

---

## 0. Corrections from reading the paper (important for interview defense)

**(a) TurboQuant-mse is NOT polar-coordinate quantization.** PolarQuant
(Han, Kacham, Karbasi, Mirrokni, Zandieh, arXiv:2502.02617) is *prior work by an
overlapping author group* that quantizes pairs of coordinates in polar form
(radius + angle); it appears in TurboQuant's experiments only as a *baseline*
(Fig. 4). TurboQuant-mse is: **Haar-random orthogonal rotation → per-coordinate
optimal Lloyd-Max scalar quantization against the analytically-known Beta
marginal of a uniform point on the sphere.** No polar transform anywhere.
Calling it "the PolarQuant-based variant" in an interview would be wrong.

**(b) The "up to 8× attention-logit speedup on H100s" claim is not in arXiv v1**
(the only version on arXiv as of 2026-07-18; checked the submission history).
All experiments in v1 ran on a single A100. The speed claim v1 actually makes is
Table 2: *quantization/indexing* time is ~0.001–0.002 s vs. 37–494 s for PQ and
597–3957 s for RabitQ (i.e., "indexing time virtually zero"), because encoding is
one matmul + one nearest-centroid lookup. If the ICLR 2026 camera-ready added
H100 logit-kernel numbers, that's a different artifact than this PDF. Either way
an M2 Mac cannot reproduce H100 kernel numbers — Phase 6 defines honest
substitutes.

**(c) The paper's b=3 MSE "≈0.03" is a loose rounding.** The canonical
Lloyd-Max distortion for a unit-variance Gaussian at 8 levels is **0.034539**
(Max, 1960). The paper's own upper bound √3π/2·4⁻³ = 0.0425 and lower bound
4⁻³ = 0.0156 bracket both. Our validation targets use the precise values
(0.3634, 0.1175, 0.03454, 0.009497) with the paper's rounded values
(0.36, 0.117, 0.03, 0.009) quoted alongside. If our numerically-solved codebook
lands at ≈0.0345 for b=3, that is *correct*, not a bug.

**(d) Memory-reduction bookkeeping.** "4–6×" corresponds to the paper's
2.5-bit and 3.5-bit *mixed-precision outlier-split* configs (16/3.5 ≈ 4.6×,
16/2.5 = 6.4×, before per-vector norm overhead; paper claims ">4.5×" /
">5×" net). A uniform 4-bit config with one fp16 norm per (token, head,
K-or-V) at head_dim 64 gives (4·64+16)/(16·64) = **3.76×** vs fp16. We will
report that exact figure, not "4–6×", unless we implement the outlier split.

---

## 1. The algorithm, precisely

### 1.1 Setup (data-independent, done once)

- Sample G ∈ R^{d×d} with i.i.d. N(0,1) entries; QR-decompose G = QR; set
  **Π = Q · diag(sign(diag(R)))**. The sign correction is required to make Π
  Haar-distributed (raw torch/numpy QR is *not* Haar; a classic implementation
  pitfall — Mezzadri 2007).
- Build the codebook: for y = Πx with ‖x‖₂ = 1, y is uniform on S^{d-1}, and
  each coordinate has marginal density (paper Lemma 1)

      f_X(t) = Γ(d/2) / (√π Γ((d−1)/2)) · (1 − t²)^{(d−3)/2},  t ∈ [−1,1]

  i.e. a symmetric Beta((d−1)/2, (d−1)/2) affinely mapped to [−1,1], with
  Var = 1/d exactly, converging to N(0, 1/d) as d → ∞.
- Solve the continuous 1-D k-means (Lloyd-Max) problem, Eq. (4) of the paper:

      C(f_X, b) = min_{c_1 ≤ … ≤ c_{2^b}}  Σ_i ∫_{(c_{i−1}+c_i)/2}^{(c_i+c_{i+1})/2} (t − c_i)² f_X(t) dt

  Fixed-point conditions: boundaries t_i = (c_i + c_{i+1})/2 (nearest-neighbor
  optimality) and c_i = E[X | X ∈ cell_i] (centroid optimality). Because f_X is
  log-concave (for d ≥ 3), the optimal fixed-rate scalar quantizer is *unique*
  (Fleischer/Trushkin), so Lloyd iteration converges to the global optimum.
- Scaling shortcut: if X = σZ then codebook(X) = σ·codebook(Z) and
  C scales by σ². So solve once for N(0,1) and scale by σ = 1/√d, or solve the
  exact Beta for the actual d (we'll do both and compare — at d = 64 the Beta
  is slightly lighter-tailed than the Gaussian, so distortion is marginally
  lower).

Known anchors in the Gaussian limit (these are what "correct" looks like):
- b=1 centroids: ±√(2/π)·σ ≈ ±0.7979σ  (paper: ±√(2/π)/√d)
- b=2 centroids: {±0.4528, ±1.5104}·σ   (paper: {±0.453, ±1.51}/√d)
- Normalized distortions d·C (= per-unit-vector MSE), Max (1960):

  | b | Lloyd-Max (Gaussian) | paper says | UB √3π/2·4⁻ᵇ | LB 4⁻ᵇ |
  |---|---------------------|------------|--------------|--------|
  | 1 | 1 − 2/π = 0.36338   | ≈0.36      | 0.6802       | 0.25   |
  | 2 | 0.117482            | ≈0.117     | 0.17004      | 0.0625 |
  | 3 | 0.034539            | ≈0.03      | 0.042510     | 0.015625 |
  | 4 | 0.0094966           | ≈0.009     | 0.010627     | 0.0039063 |

### 1.2 Quantize / dequantize (per vector, online)

    Quant(x):   y = Πx;  idx_j = argmin_k |y_j − c_k|  (b-bit codes, bit-packed)
    DeQuant(idx): ỹ_j = c_{idx_j};  x̃ = Πᵀỹ

Orthogonality gives ‖x − x̃‖₂ = ‖y − ỹ‖₂, so
D_mse = Σ_j E|y_j − c_{idx_j}|² = d·C(f_X, b) — the upper-bound proof
(Theorem 1) needs **only the marginals + linearity of expectation**, not
coordinate independence. Near-independence of coordinates in high d
(Diaconis–Freedman-type results) is what justifies *optimality* — it's why
per-coordinate scalar coding can't be substantially beaten by joint coding of
the rotated vector, so the scheme lands within a constant of the
information-theoretic bound.

For general (non-unit) x: store γ = ‖x‖₂ in fp16, quantize x/γ, rescale at
dequant. MSE then scales as γ²·d·C.

### 1.3 The bounds and where the constants come from

- **Upper bound (Theorem 1):** Panter-Dite high-resolution formula
  C(f, b) ≤ (1/12)(∫ f^{1/3})³ · 4⁻ᵇ. For N(0, σ²) the constant evaluates to
  (√3π/2)σ², giving D_mse ≤ (√3π/2)·4⁻ᵇ ≈ 2.7207·4⁻ᵇ for unit vectors.
- **Lower bound (Theorem 3):** Yao's minimax principle reduces worst-case
  randomized quantization to deterministic quantization of a uniform-on-sphere
  source; Shannon's lower bound with h(x) = log₂(A_d) and Stirling gives
  D(B) ≥ 2^{−2B/d} = 4⁻ᵇ (up to 1−O(1/d)).
- **The gap ≈ 2.72 (≈4.35 dB)** is the classic high-rate penalty of fixed-rate
  *scalar* quantization vs. the Shannon distortion-rate function for a Gaussian
  source: ≈1.53 dB of it is the space-filling loss (cubic cells vs. optimal
  lattices, Gersho), the rest is the fixed-rate product-code constraint. At
  b=1 the actual ratio is only 0.3634/0.25 ≈ 1.45.
- **Bias (motivates the prod variant):** E[x̃] ≠ x. At b=1 the scheme is
  exactly sign(Πx) with dequant √(2/(πd))·Πᵀsign(Πx), and
  E⟨y, x̃⟩ = (2/π)⟨y, x⟩ — a 2/π multiplicative shrinkage (paper §3.2, via the
  Gaussian identity E[s·sign(sᵀx)] = √(2/π)·x). The bias decays as b grows.
  TurboQuant-prod fixes it: run mse at b−1 bits, then QJL (sign(Sr) with
  S ~ N(0,1)^{d×d}) on the residual r, store ‖r‖; the QJL estimator is unbiased
  with Var ≤ (π/2d)‖r‖²‖y‖², giving D_prod ≤ (π/2d)·D_mse(b−1) ≈
  {1.57, 0.56, 0.18, 0.047}/d at b = 1..4. (Stretch goal.)
- Side note worth knowing: entropy-coding the code indices would save ~5% at
  b=4 (cell-probability entropy ≈ 3.8 bits); the authors deliberately skip it
  for speed. We skip it too.

### 1.4 What "data-oblivious" and "online" mean here, formally

- **Data-oblivious:** the quantizer's randomness (Π) is sampled *independent of
  the data*; the codebook is derived from the analytically-known marginal
  induced by the rotation, not from data. Guarantees are worst-case per-vector:
  for *any* fixed x, in expectation over the algorithm's internal randomness.
  No calibration set, no k-means on data, no Hessian-based tuning (contrast:
  PQ/OPQ, GPTQ/AWQ, KIVI's per-group scales are data-dependent).
- **Online:** each vector is quantized immediately and independently on
  arrival, in one pass, with O(d²) work (one matvec + nearest-centroid). Design
  consequences: nothing in the pipeline may depend on future tokens or dataset
  statistics; no re-fitting as the stream grows; the paper even quantizes
  tokens *during* generation (unlike KIVI/PolarQuant which keep recent tokens
  in fp16). This is exactly why it suits KV caches.
- The rotation identity ⟨q, x̃⟩ = ⟨Πq, ỹ⟩ means attention logits can in
  principle be computed directly in code space (rotate the query once, then
  codebook-lookup dot products) — that's where custom kernels would earn the
  hardware speedups; our first pass dequantizes and calls SDPA.

---

## 2. Phases

### Phase 1 — Lloyd-Max codebook engine (no torch needed; numpy/scipy)
**Build:** `codebooks.py`: Lloyd-Max solver for (i) N(0,1) via exact
conditional-mean updates using φ/Φ, (ii) the exact Beta marginal for given d
via numerical integration; codebook cache for b ∈ {1..8}; cell-probability and
distortion computation.
**Pass/fail:**
- b=1 centroids = ±0.79788 (√(2/π)), b=2 = {±0.45278, ±1.51042} to 4 decimals;
- Gaussian distortions match 0.36338 / 0.117482 / 0.034539 / 0.0094966 to 4
  significant figures; each ≤ Panter-Dite bound and ≥ Shannon LB;
- exact-Beta(d=64,128) distortion ≤ Gaussian value and within ~2% of it;
- Lloyd iterations monotonically decrease cost.
**Risks:** tail integration accuracy (use φ/Φ closed forms, not naive
quadrature); slow fixed-point convergence at b=8 (accelerate or tolerate);
forgetting the 1/√d scaling when exporting codebooks.

### Phase 2 — TurboQuant-mse quantizer, batched PyTorch
**Build:** `rotation.py` (seeded Haar sampling with the QR sign fix, one Π per
head-dim, reproducible), `quantizer.py` (encode → packed uint8 codes via
`torch.bucketize` against cell boundaries; decode; per-vector fp16 norms;
CPU/MPS; fp32 accumulation for the rotation matmuls).
**Pass/fail:** on 10⁵ random unit vectors, d ∈ {64, 128, 1536}, b = 1..4:
empirical mean ‖x−x̃‖² within Monte-Carlo error (±~1%) of d·C(f_X,b) from
Phase 1; MSE·4ᵇ approximately flat in b; pack/unpack bit-exact roundtrip;
Haar-ness sanity check (e.g., first-coordinate distribution of Πe₁ matches
f_X, mean of Π entries ≈ 0).
**Risks:** non-Haar rotation (skewed marginals → distortion off by a few %);
bucketize boundary off-by-one at cell edges; MPS fp16 matmul precision
polluting the MSE measurement (validate in fp32 on CPU first).

### Phase 3 — Empirical validation suite (reproduce Fig. 3 + bias study)
**Build:** `experiments/01_distortion_rate.py`: MSE and inner-product error vs
b ∈ {1..5} on (i) synthetic unit vectors, (ii) the paper's actual dataset:
DBpedia-OpenAI text-embedding-3-large 1536-dim (Qdrant HF dataset), sampled to
100k train + 1k query points exactly as in §4.1, streamed to avoid the full
multi-GB download; plotted against 4⁻ᵇ and √3π/2·4⁻ᵇ. `experiments/02_bias.py`: histogram of
⟨y,x̃⟩−⟨y,x⟩; verify the b=1 multiplicative bias ≈ 2/π = 0.6366 and its decay
with b.
**Pass/fail:** curves sit between the bounds and track 4⁻ᵇ slope (log-linear
fit slope ≈ −2b ln2, i.e., ratio ≈ 4× per bit); measured b=1 shrinkage within
~2% of 2/π. Real-data MSE ≈ synthetic MSE (data-obliviousness in action).
**Risks:** none serious — this phase is cheap insurance before touching a
model; a discrepancy here means Phase 2 has a bug.

### Phase 4 — KV-cache integration on a small model
**Build:** `kv_cache.py`: a `transformers` Cache subclass that stores K and V
as packed TurboQuant codes + fp16 norms (K quantized post-RoPE, per head, one
shared Π per head_dim), dequantizing per layer on read; wired into
Llama-3.2-1B-Instruct (head_dim 64; gated — needs your HF token) or fallback
Qwen2.5-0.5B/1.5B-Instruct (ungated; the 1.5B has head_dim 128, matching the
paper's geometry, but is tighter on 8 GB RAM). fp16 weights on MPS.
**Pass/fail (staged sanity ladder):**
- b=8: next-token logits ≈ fp16 baseline (KL per token ~0), perplexity delta
  ≈ 0 — proves plumbing correctness independent of quantization strength;
- b=4: WikiText-2 perplexity within a small delta of fp16 (paper says 3.5-bit
  is quality-neutral on an 8B model; 1B models are more sensitive — expect
  ≤ a few % Δppl; we *report* the number rather than force a target);
- greedy generations share long common prefixes with baseline.
**Risks:** cache API churn across transformers versions; RoPE ordering mistakes
(quantizing pre-RoPE keys by accident); attention-sink tokens — the first few
tokens matter disproportionately (ablation: keep first 4 tokens fp16 and
report both); per-channel key outliers hurting more at head_dim 64 (this is
what the paper's outlier split addresses — out of scope at 4-bit, in scope if
2-bit results look bad).

### Phase 5 — Long-context evaluation
**Build:** `experiments/04_needle.py`: needle-in-a-haystack-lite at 1k–8k
contexts (8 GB RAM ceiling; the paper ran 4k–104k on an 8B model — we scale the
protocol, not the exact grid), plus a b ∈ {2, 3, 4} × {K-only, V-only, K+V}
perplexity sweep.
**Pass/fail:** b=4 needle recall equal to fp16 baseline recall (paper's
TurboQuant matched full precision exactly at ≈4× compression); monotone
degradation as b drops; sweep table produced for README.
**Risks:** the 1B model's own needle recall may be imperfect at 8k — the
criterion is *quantized == unquantized baseline*, never absolute recall.

### Phase 6 — Memory & throughput measurement (honest scope)
**Build:** `experiments/05_memory_bench.py`: exact byte accounting of the cache
(codes + norms + shared Π) vs fp16 at several context lengths — analytic figure
at uniform 4-bit, head_dim 64: **3.76×** (3.88× at head_dim 128); measured
tensor sizes must match the formula. Quant/dequant throughput (vectors/s, GB/s)
on MPS and CPU; end-to-end tokens/s vs fp16 cache.
Additionally: a RunPod benchmark script (user has cloud GPU access) to run the
same quant/dequant throughput + end-to-end generation benchmark on an
A100/H100-class card, giving an honest CUDA data point comparable to the
paper's A100 setup (Table 2 encoding-time claim).
**Pass/fail:** measured bytes match the analytic formula; throughput reported
on MPS, CPU, and one CUDA card.
**Explicit non-goals:** the 8× logit-speedup figure — it is not in arXiv v1
(see §0b) and would require custom fused attention kernels, not benchmarking. End-to-end
tokens/s may be *slower* than fp16 due to Python-level dequant per layer; we
report it and explain why custom fused kernels are where the paper's speed
story lives. What we *can* honestly claim: paper-matching distortion at every
bit-width, quality-neutral 4-bit KV at ~3.8× memory reduction, and near-zero
quantization latency (Table 2's actual v1 claim).

### Stretch — TurboQuant-prod (QJL residual)
Algorithm 2: mse at b−1 bits → r = x − x̃ → sign(Sr), store ‖r‖; dequant adds
√(π/2)/d · ‖r‖ · Sᵀqjl. Validate: unbiasedness (mean inner-product error ≈ 0
at every b, reproducing Fig. 1) and D_prod ≈ {1.57, 0.56, 0.18, 0.047}/d at
b = 1..4. Natural demo: recall@k retrieval on GloVe vs the mse variant.

---

## 3. Tooling

- **Python 3.12, PyTorch (MPS)** — already installed (2.10 nightly). Plain
  PyTorch throughout; no custom CUDA/Metal kernels needed for a correctness-
  first pass (encode is a matmul + bucketize; decode is a gather + matmul).
- **numpy + scipy** — Lloyd-Max solver (`scipy.stats.norm` for φ/Φ closed-form
  conditional means; quadrature for the exact Beta).
- **transformers + datasets** (+ `accelerate`) — model loading, WikiText-2,
  needle harness. **huggingface_hub login needed for Llama 3.2** (gated);
  Qwen2.5 is the ungated fallback.
- **matplotlib** for Fig. 3-style plots; **pytest** for the validation suite.
- 8 GB RAM budget: 1B fp16 ≈ 2.5 GB weights; contexts ≤ 8k; quantized cache at
  4k ≈ 34 MB (vs 128 MB fp16) — comfortably feasible.

## 4. Repo structure

    turboquant/
    ├── README.md               # math summary, results tables vs paper, plots
    ├── LICENSE                 # Apache 2.0
    ├── pyproject.toml
    ├── docs/
    │   ├── PLAN.md              # this file
    │   └── RUNBOOK.md           # gitignored: deferred run commands + RunPod steps
    ├── turboquant/
    │   ├── __init__.py
    │   ├── codebooks.py        # Phase 1: Lloyd-Max (Gaussian + exact Beta)
    │   ├── rotation.py         # Phase 2: seeded Haar rotations (QR sign fix)
    │   ├── quantizer.py        # Phase 2: TurboQuantMSE encode/decode + packing
    │   ├── kv_cache.py         # Phase 4: transformers Cache integration
    │   └── qjl.py              # Stretch: TurboQuantProd
    ├── tests/
    │   ├── test_codebooks.py   # anchors: centroids + distortion table
    │   ├── test_quantizer.py   # MC distortion, packing roundtrip, Haar sanity
    │   ├── test_kv_cache.py    # b=8 == fp16 logits sanity
    │   └── test_qjl.py         # Theorem 2: unbiasedness, distortion bound
    ├── experiments/
    │   ├── 00_quantizer_validation.py
    │   ├── 01_distortion_rate.py
    │   ├── 02_bias.py
    │   ├── 03_perplexity.py
    │   ├── 04_needle.py
    │   ├── 05_memory_bench.py
    │   ├── 06_cuda_bench.py    # RunPod: paper Table 2 protocol
    │   └── 07_qjl_validation.py
    └── results/                # json + png, committed (small)

Interviewer-skim logic: README leads with the distortion-rate table
(ours vs paper vs bounds) and one Fig. 3-style plot; tests encode the paper's
numbers as assertions; experiments are one-command reproducible.

## 5. Decisions (confirmed 2026-07-19)

1. **Model:** Llama-3.2-1B-Instruct (user has HF access).
2. **Phase 3 dataset:** DBpedia-OpenAI-1536 — the paper's exact §4.1 dataset,
   sampled/streamed to 100k train + 1k query.
3. **CUDA:** user has RunPod access; Phase 6 includes an A100/H100-class
   benchmark script alongside the local MPS/CPU numbers.
4. **Uniform 4-bit first;** the 2.5/3.5-bit outlier-split mixed-precision
   configs are a follow-on after Phase 5.
5. **Repo:** GitHub `hasan-mustafaa/turboquant`, one commit (or more) pushed
   per phase as it completes.
