"""Phase 6b (RunPod): CUDA encode-throughput benchmark, paper Table 2 protocol.

The paper's arXiv-v1 speed claim is Table 2: quantizing 100k vectors at 4 bits
takes ~0.0007 s (d=200), ~0.0013 s (d=1536), ~0.0021 s (d=3072) on a single
A100 -- versus 37-494 s for PQ and 597-3957 s for RabitQ, i.e. "indexing time
virtually zero". This script reproduces that measurement for our
implementation (encode = one matmul + bucketize; steady-state, codebook and
rotation amortized), plus decode and a KV-shaped microbenchmark.

TF32. Ampere-and-newer NVIDIA GPUs silently run fp32 matmuls through
tensor cores at TF32 precision (10-bit mantissa, ~fp16) unless told not to,
and TurboQuant's rotation *is* an fp32 matmul -- so this is a real
speed-vs-distortion knob, not a free win. The default `--tf32 both` sweeps
it in one invocation and reports encode/decode time *and* measured
d.D_mse against the analytic optimum for each setting, so the tradeoff is
visible rather than assumed. (Everything else in the repo keeps TF32 off:
kv_cache.py deliberately does quantizer math in true fp32.)

RunPod usage (any PyTorch CUDA image, e.g. runpod/pytorch):
    git clone https://github.com/hasan-mustafaa/turboquant && cd turboquant
    pip install -e .
    python experiments/06_cuda_bench.py
Optionally set HF_TOKEN and add --ppl to also run the Phase 4 perplexity
sweep with meta-llama/Llama-3.2-1B-Instruct on CUDA.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device import default_device, synchronize
from _results import save_results

from turboquant.quantizer import TurboQuantMSE


def bench(fn, sync, warmup=3, iters=10) -> float:
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


def set_tf32(enabled: bool) -> None:
    """Toggle tensor-core TF32 for fp32 matmuls (Ampere+; no-op elsewhere)."""
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def unit_vectors(n: int, d: int, device: str, seed: int) -> torch.Tensor:
    """Identical inputs across TF32 settings -- generated on CPU because MPS
    has no device generator, then moved."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=gen).to(device)
    return x / x.norm(dim=-1, keepdim=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dims", type=int, nargs="*", default=[200, 1536, 3072])
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--kv-ctx", type=int, nargs="*",
                    default=[4096, 32768, 131072],
                    help="context lengths for the KV-shaped dequantize bench")
    ap.add_argument("--tf32", choices=["off", "on", "both"], default="both",
                    help="TF32 tensor-core matmuls: sweep both by default "
                         "(CUDA/Ampere+ only; ignored on other devices)")
    ap.add_argument("--ppl", action="store_true",
                    help="also run Llama-3.2-1B perplexity (needs HF_TOKEN)")
    args = ap.parse_args()

    dev = args.device
    sync = lambda: synchronize(dev)
    if dev == "cuda":
        print(torch.cuda.get_device_name())

    # TF32 exists only on CUDA; elsewhere there is nothing to toggle and the
    # single "off" pass is plain fp32.
    modes = (["off", "on"] if args.tf32 == "both" else [args.tf32])
    if dev != "cuda":
        modes = ["off"]
        print(f"({dev}: TF32 is CUDA-only, running plain fp32)")

    report = {"device": dev, "n": args.n, "bits": args.bits, "modes": {}}
    for mode in modes:
        set_tf32(mode == "on")
        m = report["modes"][f"tf32_{mode}"] = {
            "encode_s": {}, "decode_s": {}, "d_mse": {}, "d_mse_analytic": {}}
        print(f"\n[TF32 {mode}] encode time, {args.n} vectors at b={args.bits} "
              f"(paper Table 2, A100: 0.0007 / 0.0013 / 0.0021 s):")
        for d in args.dims:
            tq = TurboQuantMSE(d, args.bits, seed=1, device=dev)
            x = unit_vectors(args.n, d, dev, seed=20250729 + d)
            q = tq.quantize(x)
            mse = (x - tq.dequantize(q)).pow(2).sum(-1).mean().item()
            t_enc = bench(lambda: tq.quantize(x), sync)
            t_dec = bench(lambda: tq.dequantize(q), sync)
            m["encode_s"][d] = t_enc
            m["decode_s"][d] = t_dec
            m["d_mse"][d] = mse
            m["d_mse_analytic"][d] = tq.expected_unit_mse
            gb = x.nbytes / 1e9
            print(f"  d={d:>5}: encode {t_enc:.5f} s ({gb / t_enc:6.1f} GB/s)"
                  f"   decode {t_dec:.5f} s"
                  f"   d*D_mse {mse:.6f} (analytic {tq.expected_unit_mse:.6f},"
                  f" {mse / tq.expected_unit_mse:.4f}x)")

        # KV-shaped: one generation step's worth of history dequantization
        print(f"[TF32 {mode}] KV-shaped dequantize "
              f"(1 layer, 8 KV heads, d=128, b=4):")
        for ctx in args.kv_ctx:
            tq = TurboQuantMSE(128, 4, seed=1, device=dev)
            x = unit_vectors(8 * ctx, 128, dev, seed=1000 + ctx)
            q = tq.quantize(x)
            t = bench(lambda: tq.dequantize(q), sync)
            m["decode_s"][f"kv_ctx_{ctx}"] = t
            print(f"  ctx={ctx:>7,}: {t * 1e3:8.3f} ms")
    set_tf32(False)  # leave the process in the repo's default fp32 state

    if len(modes) == 2:
        off, on = (report["modes"][f"tf32_{k}"] for k in ("off", "on"))
        print("\nTF32 tradeoff (on vs off) -- speedup, and what it costs in "
              "distortion:")
        print(f"{'d':>6} {'encode':>9} {'decode':>9} {'d*D_mse off':>13}"
              f" {'d*D_mse on':>12} {'penalty':>9}")
        for d in args.dims:
            print(f"{d:>6} {off['encode_s'][d] / on['encode_s'][d]:>8.2f}x"
                  f" {off['decode_s'][d] / on['decode_s'][d]:>8.2f}x"
                  f" {off['d_mse'][d]:>13.6f} {on['d_mse'][d]:>12.6f}"
                  f" {on['d_mse'][d] / off['d_mse'][d]:>8.3f}x")

    out = save_results(f"cuda_bench_{dev}", report, device=dev)
    print(f"\nsaved {out}")

    if args.ppl:
        import subprocess
        subprocess.run([
            sys.executable, str(Path(__file__).parent / "03_perplexity.py"),
            "--model", "meta-llama/Llama-3.2-1B-Instruct",
            "--device", dev, "--seqs", "40",
        ], check=True)


if __name__ == "__main__":
    main()
