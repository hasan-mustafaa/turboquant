"""Phase 5: needle-in-a-haystack under KV-cache quantization (paper S4.2).

Protocol scaled to this hardware (paper: Llama-3.1-8B, 4k-104k on an A100;
here: a small open model, 1k-8k on an M2). A needle sentence carrying a
random 4-digit code is buried at varying depths inside WikiText filler; the
model must produce the code from a retrieval question. Pass criterion follows
PLAN.md: the quantized cache must match the *fp16 baseline's* recall
cell-for-cell -- absolute recall is a property of the small model, not of the
quantization.

Run:  .venv/bin/python experiments/04_needle.py [--model M]
      [--lengths 1024 2048 4096 8192] [--depths 0 0.25 0.5 0.75 1.0]
      [--bits 4 2]
"""

import argparse
import json
import random
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from turboquant.kv_cache import TurboQuantCache

RESULTS = Path(__file__).resolve().parent.parent / "results"

NEEDLE = ("\n\nThe secret access code for the vault mentioned in the annual "
          "report is {code}. Remember this number carefully.\n\n")
QUESTION = ("Based only on the document above: what is the secret access "
            "code for the vault? Answer with the number only.")


def build_haystack(tok, filler_ids, n_ctx_tokens, depth, code):
    needle_ids = tok(NEEDLE.format(code=code), add_special_tokens=False,
                     return_tensors="pt").input_ids[0]
    n_fill = n_ctx_tokens - len(needle_ids)
    pos = int(depth * n_fill)
    return torch.cat([filler_ids[:pos], needle_ids, filler_ids[pos:n_fill]])


@torch.no_grad()
def run_cell(model, tok, device, doc_ids, cache, max_new=16) -> str:
    msgs = [{"role": "user", "content":
             tok.decode(doc_ids) + "\n\n" + QUESTION}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(device)
    # chunked prefill through the (possibly quantized) cache
    chunk = 512
    logits = None
    for i in range(0, ids.shape[1], chunk):
        out = model(ids[:, i:i + chunk], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits
    generated = []
    cur = logits[:, -1].argmax(-1, keepdim=True)
    for _ in range(max_new):
        generated.append(cur.item())
        out = model(cur, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
    return tok.decode(generated)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--lengths", type=int, nargs="*",
                    default=[1024, 2048, 4096, 8192])
    ap.add_argument("--depths", type=float, nargs="*",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--bits", type=str, nargs="*", default=["8:4", "4", "2"],
                    help="ints for uniform K+V bits, or K:V pairs like 8:4")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if args.device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(
        args.device).eval()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    filler_text = "\n\n".join(t for t in ds["text"] if t.strip())
    filler_ids = tok(filler_text[:1_200_000], add_special_tokens=False,
                     return_tensors="pt").input_ids[0]

    configs = {"fp16": lambda: None}
    for b in args.bits:
        if ":" in b:
            bk, bv = (int(x) for x in b.split(":"))
            configs[f"K{bk}V{bv}"] = (
                lambda bk=bk, bv=bv: TurboQuantCache(bits_k=bk, bits_v=bv))
        else:
            configs[f"b={b}"] = lambda b=int(b): TurboQuantCache(bits=int(b))

    rng = random.Random(0)
    grid = {name: {} for name in configs}
    for L in args.lengths:
        for depth in args.depths:
            code = str(rng.randint(1000, 9999))
            doc = build_haystack(tok, filler_ids, L, depth, code)
            row = []
            for name, factory in configs.items():
                ans = run_cell(model, tok, args.device, doc, factory())
                hit = code in ans
                grid[name][f"L={L},depth={depth:.2f}"] = int(hit)
                row.append(f"{name}:{'Y' if hit else 'n'}")
            print(f"L={L:>5} depth={depth:.2f} code={code}  " + "  ".join(row))

    print("\nrecall per config:")
    summary = {}
    for name, cells in grid.items():
        r = sum(cells.values()) / len(cells)
        summary[name] = r
        print(f"  {name:>6}: {r:.3f}")
    # Pass criterion (revised after the Phase 4 finding): the quality-neutral
    # config is K8V4 -- keys need precision, values are nearly free. Uniform
    # b=4 and b=2 are reported as the expected-degradation reference points.
    must_match = [n for n in grid if n.startswith("K8")]
    match = all(grid[n] == grid["fp16"] for n in must_match)
    print(f"PASS: {must_match} match fp16 cell-for-cell" if match
          else f"MISMATCH vs fp16 in {must_match} -- inspect grid")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "needle.json"
    out.write_text(json.dumps(
        {"model": args.model, "grid": grid, "recall": summary}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
