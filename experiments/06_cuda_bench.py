"""Phase 6b (RunPod): CUDA encode-throughput benchmark, paper Table 2 protocol.

The paper's arXiv-v1 speed claim is Table 2: quantizing 100k vectors at 4 bits
takes ~0.0007 s (d=200), ~0.0013 s (d=1536), ~0.0021 s (d=3072) on a single
A100 -- versus 37-494 s for PQ and 597-3957 s for RabitQ, i.e. "indexing time
virtually zero". This script reproduces that measurement for our
implementation (encode = one matmul + bucketize; steady-state, codebook and
rotation amortized), plus decode and a KV-shaped microbenchmark.

RunPod usage (any PyTorch CUDA image, e.g. runpod/pytorch):
    git clone https://github.com/hasan-mustafaa/turboquant && cd turboquant
    pip install -e .
    python experiments/06_cuda_bench.py
Optionally set HF_TOKEN and add --ppl to also run the Phase 4 perplexity
sweep with meta-llama/Llama-3.2-1B-Instruct on CUDA.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device import default_device, synchronize

from turboquant.quantizer import TurboQuantMSE

RESULTS = Path(__file__).resolve().parent.parent / "results"


def bench(fn, sync, warmup=3, iters=10) -> float:
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dims", type=int, nargs="*", default=[200, 1536, 3072])
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--ppl", action="store_true",
                    help="also run Llama-3.2-1B perplexity (needs HF_TOKEN)")
    args = ap.parse_args()

    dev = args.device
    sync = lambda: synchronize(dev)
    if dev == "cuda":
        print(torch.cuda.get_device_name())

    report = {"device": dev, "n": args.n, "bits": args.bits, "encode_s": {},
              "decode_s": {}}
    print(f"\nEncode time, {args.n} vectors at b={args.bits} "
          f"(paper Table 2, A100: 0.0007 / 0.0013 / 0.0021 s):")
    for d in args.dims:
        tq = TurboQuantMSE(d, args.bits, seed=1, device=dev)
        x = torch.randn(args.n, d, device=dev)
        x = x / x.norm(dim=-1, keepdim=True)
        q = tq.quantize(x)
        t_enc = bench(lambda: tq.quantize(x), sync)
        t_dec = bench(lambda: tq.dequantize(q), sync)
        report["encode_s"][d] = t_enc
        report["decode_s"][d] = t_dec
        gb = x.nbytes / 1e9
        print(f"  d={d:>5}: encode {t_enc:.5f} s ({gb / t_enc:6.1f} GB/s fp32)"
              f"   decode {t_dec:.5f} s")

    # KV-shaped: one generation step's worth of history dequantization
    print("\nKV-shaped dequantize (1 layer, 8 KV heads, d=128, b=4):")
    for ctx in (4096, 32768, 131072):
        tq = TurboQuantMSE(128, 4, seed=1, device=dev)
        x = torch.randn(8 * ctx, 128, device=dev)
        x = x / x.norm(dim=-1, keepdim=True)
        q = tq.quantize(x)
        t = bench(lambda: tq.dequantize(q), sync)
        report["decode_s"][f"kv_ctx_{ctx}"] = t
        print(f"  ctx={ctx:>7,}: {t * 1e3:8.3f} ms")

    RESULTS.mkdir(exist_ok=True)
    name = f"cuda_bench_{dev}.json"
    (RESULTS / name).write_text(json.dumps(report, indent=2))
    print(f"saved {RESULTS / name}")

    if args.ppl:
        import subprocess
        subprocess.run([
            sys.executable, str(Path(__file__).parent / "03_perplexity.py"),
            "--model", "meta-llama/Llama-3.2-1B-Instruct",
            "--device", dev, "--seqs", "40",
        ], check=True)


if __name__ == "__main__":
    main()
