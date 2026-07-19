"""Phase 4: language-model quality under online KV-cache quantization.

Chunked-prefill perplexity on WikiText-2 with the TurboQuant cache. Chunking
does not change semantics: every K/V vector is quantized the moment it enters
the cache (before any attention reads it), so a token's representation is
identical whether it arrived in a 512-token chunk or one-by-one -- this is
verified by test_incremental_decode_consistent. The fp16 baseline runs the
same chunked path with a DynamicCache for an apples-to-apples comparison.

Note the eval regime is *stricter* than most KV-quant papers: KIVI and
PolarQuant keep a recent fp16 window; here every token including the newest
is quantized (the paper's own streaming setup, Section 4.3).

Run:  .venv/bin/python experiments/03_perplexity.py [--model M] [--ctx 2048]
      [--seqs 20] [--device mps]
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from turboquant.kv_cache import TurboQuantCache

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_wikitext_tokens(tok, n_tokens: int) -> torch.Tensor:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return ids[:n_tokens]


@torch.no_grad()
def perplexity(model, ids, ctx, chunk, device, cache_factory) -> float:
    total_nll, total_tok = 0.0, 0
    n_seqs = (len(ids) - 1) // ctx
    for s in range(n_seqs):
        seq = ids[s * ctx: (s + 1) * ctx + 1]
        inp, tgt = seq[:-1], seq[1:]
        cache = cache_factory()
        for i in range(0, len(inp), chunk):
            out = model(inp[i:i + chunk].unsqueeze(0).to(device),
                        past_key_values=cache, use_cache=True)
            logits = out.logits[0].float()
            cache = out.past_key_values
            total_nll += F.cross_entropy(
                logits, tgt[i:i + chunk].to(device), reduction="sum").item()
            total_tok += logits.shape[0]
    return math.exp(total_nll / total_tok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seqs", type=int, default=20)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--bits", type=str, nargs="*",
                    default=["8", "8:4", "4", "3", "2"],
                    help="ints for uniform K+V bits, or K:V pairs like 8:4")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if args.device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(
        args.device).eval()
    ids = load_wikitext_tokens(tok, args.ctx * args.seqs + 1)
    print(f"{args.model} on {args.device} ({dtype}), "
          f"{args.seqs} x {args.ctx}-token sequences, chunk={args.chunk}")

    results = {}
    t0 = time.time()
    base = perplexity(model, ids, args.ctx, args.chunk, args.device,
                      cache_factory=lambda: None)
    results["fp16"] = base
    print(f"  fp16 baseline: ppl {base:.4f}  [{time.time() - t0:.0f}s]")
    for b in args.bits:
        if ":" in b:
            bk, bv = (int(x) for x in b.split(":"))
            name = f"K{bk}V{bv}"
            factory = lambda bk=bk, bv=bv: TurboQuantCache(bits_k=bk, bits_v=bv)
        else:
            name = f"b={b}"
            factory = lambda b=int(b): TurboQuantCache(bits=b)
        t0 = time.time()
        ppl = perplexity(model, ids, args.ctx, args.chunk, args.device,
                         cache_factory=factory)
        results[name] = ppl
        print(f"  TurboQuant {name} (K+V): ppl {ppl:.4f}  "
              f"(+{100 * (ppl / base - 1):.2f}%)  [{time.time() - t0:.0f}s]")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "perplexity.json"
    out.write_text(json.dumps(
        {"model": args.model, "ctx": args.ctx, "seqs": args.seqs,
         "device": args.device, "ppl": results}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
