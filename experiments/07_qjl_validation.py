"""Stretch phase: TurboQuant-prod validation (paper Fig. 1a + Theorem 2).

Reproduces the paper's Section 4.1 comparison on the DBpedia OpenAI-1536
dataset: error histograms per bit-width for the *unbiased* prod variant
(Fig. 1a analogue -- centered at zero at every b, unlike the mse variant's
shifted histograms from experiments/02_bias.py), plus d*Dprod against the
Theorem 2 envelope and the paper's reported values {1.57, 0.56, 0.18,
0.047}/d.

Run: .venv/bin/python experiments/07_qjl_validation.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import load_dbpedia_openai

from turboquant.qjl import TurboQuantProd

RESULTS = Path(__file__).resolve().parent.parent / "results"
BITS = (1, 2, 3, 4)
PAPER_DPROD = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}


def main() -> None:
    train_np, query_np = load_dbpedia_openai()
    x = torch.from_numpy(train_np[:20_000])
    y = torch.from_numpy(query_np)
    d = x.shape[1]

    fig, axes = plt.subplots(1, len(BITS), figsize=(3.1 * len(BITS), 3.0),
                             sharex=True)
    print(f"{'b':>2} {'mean err':>10} {'d*Dprod':>9} {'paper':>7} {'UB':>7}")
    dprods = []
    for ax, b in zip(axes, BITS):
        tq = TurboQuantProd(d, b, seed=1)
        err = ((y @ (tq.roundtrip(x) - x).T)).flatten()
        dp = d * float((err**2).mean())
        dprods.append(dp)
        print(f"{b:>2} {float(err.mean()):>+10.6f} {dp:>9.4f} "
              f"{PAPER_DPROD[b]:>7.3f} {tq.expected_dprod_ub:>7.4f}")
        ax.hist(err.numpy(), bins=120, range=(-0.1, 0.1), color="#2e7d32")
        ax.axvline(0.0, color="k", lw=0.8)
        ax.set_title(f"b={b}, mean={float(err.mean()):+.5f}")
        ax.set_xlabel("IP distortion")
    axes[0].set_ylabel("frequency")
    fig.suptitle("TurboQuant-prod: unbiased at every bit-width (cf. paper Fig. 1a)")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "qjl_histograms.png", dpi=160)
    print(f"saved {RESULTS / 'qjl_histograms.png'}")

    ok = all(dp <= TurboQuantProd(d, b, seed=1).expected_dprod_ub * 1.05
             for b, dp in zip(BITS, dprods))
    print("PASS: d*Dprod within Theorem 2 bound at all bit-widths" if ok
          else "FAIL: Dprod exceeds Theorem 2 bound")


if __name__ == "__main__":
    main()
