#!/bin/bash
# One clean run: the full Tier 2/3 suite (docs/TESTING.md) in a single go.
#
#   bash experiments/run_all.sh            # both models if HF_TOKEN is set
#   QWEN_ONLY=1 bash experiments/run_all.sh   # skip the gated Llama runs
#
# On a CUDA box everything auto-detects the GPU. On macOS, wrap the call in
# `caffeinate -im` -- a sleeping laptop once stalled a 30-min sweep for 8h.
#
# Outlier-split configs use the --bits grammar from experiments/_configs.py:
# <bits_k>+<n_outliers>x<bits_outlier>, scaled to head_dim 64 at the paper's
# 25% outlier ratio (16 of 64 channels). Effective key bits:
#   2+16x3 -> 2.25  the paper's printed formula, "(32*3+96*2)/128"
#   2+16x4 -> 2.50  the paper's *label* for that same config (see README)
#   3+16x5 -> 3.50  the paper's second config
set -e
cd "$(dirname "$0")/.."
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python"

LLAMA="meta-llama/Llama-3.2-1B-Instruct"
PPL_BITS=(8 8:4 4 3 2 "2+16x3:4" "2+16x4:4" "3+16x5:4")
NEEDLE_BITS=("8:4" 4 2 "2+16x3:4" "2+16x4:4" "3+16x5:4")
NEEDLE_LEN=(1024 2048 4096 8192 16384)

run_llama=1
[ -n "$QWEN_ONLY" ] && run_llama=0

echo "=== [1/6] test suite ==="
$PY -m pytest tests/ -q

echo "=== [2/6] perplexity: Qwen2.5-0.5B ==="
$PY -u experiments/03_perplexity.py --seqs 40 --bits "${PPL_BITS[@]}"

echo "=== [3/6] needle grid: Qwen2.5-0.5B ==="
$PY -u experiments/04_needle.py --lengths "${NEEDLE_LEN[@]}" \
    --bits "${NEEDLE_BITS[@]}"

if [ "$run_llama" = 1 ]; then
    echo "=== [4/6] perplexity + needle: Llama-3.2-1B (gated) ==="
    $PY -u experiments/03_perplexity.py --model "$LLAMA" --seqs 40 \
        --bits "${PPL_BITS[@]}"
    $PY -u experiments/04_needle.py --model "$LLAMA" \
        --lengths "${NEEDLE_LEN[@]}" --bits "${NEEDLE_BITS[@]}"
else
    echo "=== [4/6] Llama runs skipped (QWEN_ONLY set) ==="
fi

echo "=== [5/6] memory + decode throughput ==="
$PY -u experiments/05_memory_bench.py
[ "$run_llama" = 1 ] && $PY -u experiments/05_memory_bench.py --model "$LLAMA"

echo "=== [6/6] encode throughput (TF32 sweep) + QJL validation ==="
$PY -u experiments/06_cuda_bench.py
$PY -u experiments/07_qjl_validation.py

echo "=== complete -- results in results/*.json ==="
ls -la results/
