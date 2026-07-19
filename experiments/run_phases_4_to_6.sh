#!/bin/bash
# Chained Phase 4-6 evaluation runs (portable: macOS or a RunPod/Linux CUDA
# box). On macOS, wrap in caffeinate so a sleeping laptop cannot stall the
# pipeline (a 30-min sweep once sat frozen for 8 hours of machine sleep):
#   caffeinate -im bash experiments/run_phases_4_to_6.sh
# On RunPod, just:
#   bash experiments/run_phases_4_to_6.sh
set -e
cd "$(dirname "$0")/.."
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python"

echo "=== [1/3] Phase 4: perplexity sweep ==="
$PY -u experiments/03_perplexity.py --seqs 10

echo "=== [2/3] Phase 5: needle-in-a-haystack grid ==="
$PY -u experiments/04_needle.py --lengths 1024 2048 4096

echo "=== [3/3] Phase 6: memory + throughput bench ==="
$PY -u experiments/05_memory_bench.py

echo "=== chain complete ==="
