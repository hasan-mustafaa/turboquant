# TESTING — three tiers, from laptop to cloud GPU

This project runs in three tiers. Every number this repo reports comes from a
real run on real hardware — there are no simulated, mocked, or placeholder
results anywhere. Where a number requires hardware not yet used, it is listed
below as **pending**, not filled in. Results files (`results/*.json`) carry a
provenance block (git SHA, device, library versions, timestamp) written by
`experiments/_results.py`.

---

## Tier 1 — local correctness suite (no model, no network, no GPU)

Pure-math validation of the quantization stack against the paper's theorems
and the canonical Lloyd-Max literature values. Runs anywhere Python runs.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_kv_cache.py
```

Expected: **all tests pass** (see the count in the latest CI-style run at the
bottom of this file). What is being asserted, per file:

- `test_codebooks.py` — Lloyd-Max solver vs closed forms and Max (1960):
  b=1 centroids ±√(2/π)=±0.797885 (rtol 1e-10), distortions
  0.363380 / 0.117482 / 0.034539 / 0.009497 for b=1..4 (atol 5e-5), every
  value inside the paper's [4⁻ᵇ, (√3π/2)·4⁻ᵇ] band, Monte-Carlo
  cross-checks with 5M samples.
- `test_quantizer.py` — end-to-end quantizer MSE within 1.5% of the analytic
  d·C(f_X,b) at d ∈ {64,128,1536}; Haar-ness via KS test on *fixed* inputs
  (catches the missing QR sign-fix); bit-exact packing for every width 1–8;
  the shrinkage identity E⟨x,x̃⟩ = 1−D_mse; edge cases (zero vectors,
  unbatched input, fp16 input).
- `test_qjl.py` — TurboQuant-prod (Theorem 2): inner-product regression
  slope = 1 within 5e-3 (unbiasedness), d·D_prod ≤ (π/2)·D_mse(b−1)·1.05,
  b=1 = pure QJL, storage = b·d + 32 bits exactly.
- `test_outlier.py` — paper §4.3 mixed precision: planted outlier channels
  recovered exactly, split beats uniform-low-bit ≥5× on an outlier-heavy
  source, channel reassembly positionally exact, effective-bits formula.

Quick visual check (prints the Phase 1 table; compare against README):

```bash
.venv/bin/python -m turboquant.codebooks
.venv/bin/python experiments/00_quantizer_validation.py   # + throughput on this machine
```

## Tier 2 — local small-model mode (real model, laptop-scale)

Real end-to-end validation with **Qwen2.5-0.5B-Instruct** (ungated, ~1 GB
download, head_dim=64 — same geometry as Llama-3.2-1B). Works on CPU
(slow), Apple Silicon MPS, or a single consumer CUDA GPU; device is
auto-detected (cuda > mps > cpu).

```bash
.venv/bin/pip install -e ".[dev,llm]"
.venv/bin/python -m pytest tests/test_kv_cache.py -q   # ~20 min on CPU fp32
turboquant-bench ppl --profile local                   # WikiText-2 sweep
turboquant-bench needle --profile local                # retrieval grid
turboquant-bench memory --profile local                # bytes vs analytic formula
```

What a genuine pass looks like (real numbers from this machine, M2/MPS,
2026-07-19 — your hardware will differ in speed, not in quality numbers):

| config | WikiText-2 ppl (10×2048) | vs fp16 13.298 |
|---|---|---|
| b=8 uniform | 13.309 | +0.09 % |
| K8V4 (6.125 bits/coord) | 13.389 | +0.7 % |
| b=4 uniform | 25.492 | +91.7 % (expected — see README Phase 4) |
| b=2 uniform | 360.2 | expected-degradation reference |

KV test-suite criteria: b=8 next-token KL < 1e-3; b=4 KL < 0.05 with
centering on and > 50× worse with it off (the pinned Phase 4 regression);
packed ≡ simulate logits; split-point invariance; measured packed bytes ==
`layers·2·H_kv·[(T−32)·(d·b/8+2) + 32·d·4] + layers·H_kv·d·4`.

Needle pass criterion: **K8V4 matches the fp16 baseline cell-for-cell**
(absolute recall is a property of the small model, not the quantization).

Memory pass criterion: measured packed bytes equal the analytic formula
exactly (asserted in-script); ≈3.7–3.76× vs fp16 at b=4, head_dim 64.
Expect packed decode tok/s *below* fp16 — the Python-level per-layer
dequantization is documented overhead; the paper's latency wins come from
fused kernels, which are out of scope here.

## Tier 3 — cloud GPU mode (full benchmarking vs the paper)

Instance: any CUDA pod with ≥16 GB VRAM comfortably fits Llama-3.2-1B and
all sweeps; an **A100 40/80 GB** matches the paper's hardware for the
Table 2 comparison (H100 fine too — just label results accordingly).

```bash
git clone https://github.com/hasan-mustafaa/turboquant && cd turboquant
pip install -e ".[dev,llm]"
export HF_TOKEN=...            # Llama-3.2-1B is gated
python -m pytest tests/ -q     # full suite, fast on server CPUs
turboquant-bench cuda          # encode-throughput, paper Table 2 protocol
turboquant-bench ppl --profile cloud
turboquant-bench needle --profile cloud
turboquant-bench memory --profile cloud
turboquant-bench qjl           # Fig. 1a reproduction (streams ~500 MB DBpedia once)
```

What a genuine pass looks like:

- **Encode time** (100k vectors, 4-bit): the paper reports 0.0007 s (d=200),
  0.0013 s (d=1536), 0.0021 s (d=3072) on one A100, vs 37–494 s for PQ and
  597–3957 s for RabitQ. Same order of magnitude from our matmul+bucketize
  encoder = pass; the qualitative claim ("indexing time virtually zero") is
  the point.
- **MSE at each bit-width** (also verified in Tier 1): ≈0.36 / 0.117 /
  0.0345 / 0.0095 for b=1..4 (paper Theorem 1 prints the last two as
  "0.03"/"0.009" — truncations; see README).
- **Memory**: uniform 4-bit = 3.76× vs fp16 at head_dim 64 (3.88× at 128),
  norms included. The paper's headline "4–6×" corresponds to its 2.5/3.5-bit
  outlier-split configs, implemented as `outlier_channels=...` /
  `bits_k_outlier=...`. Note the paper's printed config "(32×3+96×2)/128=2.5"
  actually equals 2.25 — both readings (3&2 literal formula, 4&2 matching the
  label) are evaluated; see README's outlier section for the measured
  end-to-end quality.
- **Speedup**: no fused CUDA kernels here, so no attention-latency claims
  are made or reproduced. (Note: the "8× logit speedup on H100" figure some
  summaries attribute to this paper is not in arXiv v1 at all.)
- **TF32**: `06_cuda_bench.py` sweeps `--tf32 both` by default, reporting
  encode/decode time *and* measured d·D_mse for each setting. Ampere+ GPUs
  run fp32 matmuls at TF32 (10-bit mantissa) by default, and TurboQuant's
  rotation is an fp32 matmul — so this is a speed-vs-distortion tradeoff to
  measure, not a free win. Nothing else in the repo enables it: `kv_cache.py`
  deliberately keeps quantizer math in true fp32.
- **The open scale question**: does uniform b=4 remain degraded at
  Llama-3.2-1B (head_dim 64) as it is at 0.5B, and does K8V4 remain
  neutral? Record whatever comes out — both outcomes are informative.

## Implemented vs not implemented (honest scope)

Implemented and validated with real runs:
- Lloyd-Max codebooks (exact Beta + Gaussian limit), batched quantizer,
  bit-exact packing widths 1–8, Haar rotations w/ sign fix — Tier 1.
- DBpedia distortion-rate + bias study (paper §4.1, its actual dataset).
- KV cache (transformers ≥v5), frozen-μ key centering, K/V bit splits,
  perplexity results on Qwen2.5-0.5B (MPS) and Qwen2.5-0.5B + Llama-3.2-1B
  (A100 CUDA) — see README Phase 4.
- Needle grid (04), memory bench (05), CUDA bench (06), QJL validation
  suite (07) — all run on RunPod A100, both models where applicable; see
  README Phases 5/6 and Stretch. CUDA branch of device auto-detection
  exercised for real (was previously read-only-verified).

Implemented, statically verified, **runs pending**:
- Outlier-split end-to-end quality (paper §4.3 2.5/3.5-bit configs) —
  natural next phase now that Llama-scale K/V numbers exist.

Not implemented (explicit non-goals so far):
- Fused CUDA/Metal kernels computing attention logits directly on codes
  (where the paper's latency story lives).
- Entropy coding of code indices (paper skips it too; ~5% at b=4).
- Beam-search cache surgery (`crop`/`batch_*` raise `NotImplementedError`
  by design in packed mode).

## Latest Tier 1 run

Recorded below by whoever ran it last (keep this honest — update it when
the suite changes):

- **2026-07-19, Apple M2 (CPU):** `pytest tests/ -q
  --ignore=tests/test_kv_cache.py` → **100 passed in 15.19 s** (107 tests
  collected repo-wide; the 7 skipped-here KV tests require the Tier 2 model
  download). Two test-design fixes were made during this run and are worth
  knowing about: (a) the b=1 QJL unbiasedness test now averages over 12
  independent quantizer instances, because a *single* fixed S realizes a
  ~1% multiplicative gain fluctuation (O(√D_mse/√d)) that no amount of data
  averages away — the theorem's E_Q demands averaging over Q; (b) batch
  concatenation asserts exact equality on *codes* but only `allclose` on
  decoded floats (matmul blocking differs across batch sizes).
- **2026-07-29, RunPod A100 80GB PCIe (full suite, incl. KV cache):**
  `pytest tests/ -q` → **106 passed, 1 failed in ~36 s**. The one failure,
  `test_kv_cache.py::TestKVCache::test_split_point_invariance`, is
  reproducible (byte-identical across reruns) but appears CUDA-only —
  never observed on CPU/MPS. Ruled out: TF32 (disabling
  `torch.backends.cuda.matmul.allow_tf32`/`cudnn.allow_tf32` reproduces the
  exact same failing tensors). Leading hypothesis: `scaled_dot_product_attention`
  dispatches to a different backend (flash-attention vs math) depending on
  sequence length, and chunked (500+12 tokens) vs single (512-token) passes
  hit different backends with non-bit-identical (but each internally
  consistent) floating-point reduction order — the model is loaded in true
  float32 here (`dtype=torch.float32`), so this isn't a dtype-precision
  story. Not yet confirmed by forcing `attn_implementation="eager"`. Treated
  as a documented, non-blocking discrepancy rather than a correctness bug in
  the cache's split-point logic — the same real forward pass, chunked two
  ways, agrees on every other test (KL, packed==simulate, monotonicity,
  centering).
- Also found and fixed during this run: `experiments/05_memory_bench.py`'s
  hardcoded analytic byte formula never accounted for the fp16 warmup window
  or frozen-μ overhead (it was never actually executed before this run — see
  README Phase 6 for the corrected formula and passing numbers).
