"""Phase 3a: reproduce the paper's Fig. 3 on its own dataset.

MSE and inner-product distortion of TurboQuant-mse across bit-widths, on
100k DBpedia OpenAI-1536 embeddings with 1k query vectors (Section 4.1 setup),
plotted against the theoretical envelope:

  MSE:        4^-b  <=  Dmse  <=  (sqrt(3)pi/2) 4^-b        (Thms 1, 3)
  inner-prod: 4^-b/d <= Dprod                                (Thm 3)

The mse variant's inner-product error also carries a *bias* term
(1 - Dmse)<y,x> - <y,x> = -Dmse <y,x>, so at low bit-widths its Dprod curve
sits above what variance alone would give -- the paper's Fig. 3a shows the
same effect (TurboQuant-prod, the unbiased variant, is the stretch phase).

Run: .venv/bin/python experiments/01_distortion_rate.py
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

from turboquant.codebooks import panter_dite_upper_bound, shannon_lower_bound
from turboquant.quantizer import TurboQuantMSE

RESULTS = Path(__file__).resolve().parent.parent / "results"
BITS = (1, 2, 3, 4, 5)


def main() -> None:
    print("Loading DBpedia OpenAI-1536 (paper Section 4.1 sample)...")
    train_np, query_np = load_dbpedia_openai()
    x = torch.from_numpy(train_np)
    y = torch.from_numpy(query_np)
    n, d = x.shape
    print(f"  train {tuple(x.shape)}, query {tuple(y.shape)}")

    mse, dprod, bias = [], [], []
    for b in BITS:
        tq = TurboQuantMSE(d, b, seed=1)
        xt = tq.roundtrip(x)
        m = (x - xt).pow(2).sum(-1).mean().item()
        # inner-product error over all (query, train) pairs, chunked
        se_sum, ip_orig_sum, ip_deq_sum, count = 0.0, 0.0, 0.0, 0
        for i in range(0, n, 20_000):
            ips = y @ x[i:i + 20_000].T
            ipd = y @ xt[i:i + 20_000].T
            se_sum += (ips - ipd).pow(2).sum().item()
            ip_orig_sum += ips.sum().item()
            ip_deq_sum += ipd.sum().item()
            count += ips.numel()
        dp = se_sum / count
        alpha = ip_deq_sum / ip_orig_sum  # multiplicative shrinkage estimate
        mse.append(m)
        dprod.append(dp)
        bias.append(alpha)
        print(f"  b={b}: Dmse={m:.6f} (pred {tq.expected_unit_mse:.6f}, "
              f"ratio {m / tq.expected_unit_mse:.4f})  "
              f"d*Dprod={d * dp:.4f}  shrinkage={alpha:.4f} "
              f"(pred {1 - tq.expected_unit_mse:.4f})")

    bits = np.array(BITS, dtype=float)
    lb = np.array([shannon_lower_bound(b) for b in bits])
    ub = np.array([panter_dite_upper_bound(b) for b in bits])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.semilogy(bits, dprod, "o-", label="TurboQuant-mse (measured)")
    ax.semilogy(bits, lb / d, "k--", label=r"lower bound $4^{-b}/d$")
    ax.semilogy(bits, np.pi ** 2 * np.sqrt(3) / (2 * d) * 4.0 ** -bits, "k:",
                label=r"prod upper bound $\frac{\sqrt{3}\pi^2}{2d}4^{-b}$")
    ax.set_xlabel("Bitwidth (b)"); ax.set_ylabel(r"Inner product error ($D_{prod}$)")
    ax.set_xticks(bits); ax.legend(); ax.set_title("(a) inner-prod error, d=1536")
    ax = axes[1]
    ax.semilogy(bits, mse, "o-", label="TurboQuant-mse (measured)")
    ax.semilogy(bits, lb, "k--", label=r"lower bound $4^{-b}$")
    ax.semilogy(bits, ub, "k:", label=r"upper bound $\frac{\sqrt{3}\pi}{2}4^{-b}$")
    ax.set_xlabel("Bitwidth (b)"); ax.set_ylabel(r"Mean squared error ($D_{mse}$)")
    ax.set_xticks(bits); ax.legend(); ax.set_title("(b) MSE, DBpedia OpenAI-1536")
    fig.suptitle("Fig. 3 reproduction (paper Section 4.1 dataset)")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "fig3_reproduction.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")

    ok = all(l <= m <= u for m, l, u in zip(mse, lb, ub))
    print("PASS: all Dmse within [4^-b, 2.72*4^-b]" if ok else
          "FAIL: Dmse outside theoretical band")


if __name__ == "__main__":
    main()
