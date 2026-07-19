"""Phase 6a (local): measured KV-cache memory + quantizer throughput.

Memory: run real forward passes with packed storage and compare measured
device bytes against both the fp16 DynamicCache and the analytic formula
    bytes/token/layer = 2 * H_kv * (head_dim * b / 8 + 2).
Decode speed: tokens/s for fp16 vs simulate vs packed cache. The packed path
dequantizes the whole history per layer per step in Python -- expect it to be
*slower* than fp16; the honest speed story on this hardware is encode
throughput and memory, not end-to-end latency (the paper's latency wins come
from fused kernels operating directly on codes; see PLAN.md Phase 6).

Run: .venv/bin/python experiments/05_memory_bench.py [--model M] [--device mps]
"""

import argparse
import json
import time
from pathlib import Path

import torch

from turboquant.kv_cache import TurboQuantCache

RESULTS = Path(__file__).resolve().parent.parent / "results"


@torch.no_grad()
def prefill(model, ids, device, cache):
    for i in range(0, ids.shape[1], 512):
        out = model(ids[:, i:i + 512].to(device), past_key_values=cache,
                    use_cache=True)
        cache = out.past_key_values
    return cache, out.logits[:, -1]


@torch.no_grad()
def decode_speed(model, ids, device, cache_factory, steps=48) -> float:
    cache, logits = prefill(model, ids, device, cache_factory())
    cur = logits.argmax(-1, keepdim=True)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        out = model(cur, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
    if device == "mps":
        torch.mps.synchronize()
    return steps / (time.perf_counter() - t0)


def dynamic_cache_bytes(cache) -> int:
    return sum(l.keys.nbytes + l.values.nbytes for l in cache.layers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--lengths", type=int, nargs="*", default=[512, 2048, 4096])
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    dtype = torch.float16 if args.device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(
        args.device).eval()
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    text = "\n\n".join(t for t in load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
    all_ids = tok(text, return_tensors="pt").input_ids

    report = {"model": args.model, "memory": {}, "decode_tok_s": {}}
    print(f"{cfg.num_hidden_layers} layers, {cfg.num_key_value_heads} KV heads, "
          f"head_dim {head_dim}\n")
    print(f"{'ctx':>6} {'fp16 cache':>12} {'packed b=4':>12} {'analytic':>12} {'ratio':>7}")
    for L in args.lengths:
        ids = all_ids[:, :L]
        cache, _ = prefill(model, ids, args.device, None)
        fp16_b = dynamic_cache_bytes(cache)
        # fp16 baseline stores in model dtype; normalize to 2 bytes/elt for CPU runs
        fp16_b = fp16_b if dtype == torch.float16 else fp16_b // 2
        qcache, _ = prefill(model, ids, args.device, TurboQuantCache(bits=4, packed=True))
        q_b = qcache.quantized_bytes
        analytic = cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * L * (
            head_dim * 4 // 8 + 2)
        print(f"{L:>6} {fp16_b:>12,} {q_b:>12,} {analytic:>12,} "
              f"{fp16_b / q_b:>6.2f}x")
        assert q_b == analytic, "measured bytes != analytic formula"
        report["memory"][L] = {"fp16": fp16_b, "packed": q_b}

    print("\ndecode speed at ctx=2048 (greedy, 48 steps):")
    ids = all_ids[:, :2048]
    for name, factory in [
        ("fp16", lambda: None),
        ("b=4 simulate", lambda: TurboQuantCache(bits=4)),
        ("b=4 packed", lambda: TurboQuantCache(bits=4, packed=True)),
    ]:
        tps = decode_speed(model, ids, args.device, factory)
        report["decode_tok_s"][name] = tps
        print(f"  {name:>13}: {tps:6.2f} tok/s")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "memory_bench.json").write_text(json.dumps(report, indent=2))
    print(f"saved {RESULTS / 'memory_bench.json'}")


if __name__ == "__main__":
    main()
