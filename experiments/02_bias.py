"""Phase 3b: the inner-product bias of TurboQuant-mse (paper Fig. 1b / S3.2).

The paper proves the b=1 case: E<y, x~> = (2/pi) <y, x>. The general law,
which falls out of the Lloyd centroid condition (see
tests/test_quantizer.py::test_shrinkage_identity), is

    E<y, x~> = (1 - Dmse(b)) <y, x>    i.e. shrinkage alpha_b = 1 - Dmse(b),

predicting alpha ~= 0.6366 (=2/pi), 0.8826, 0.9655, 0.9905 for b = 1..4.
This script measures alpha on the DBpedia data by regressing <y,x~> on <y,x>
over all (query, train) pairs, and draws the error histograms per bit-width
(the analogue of the paper's Fig. 1b for the mse variant).

Run: .venv/bin/python experiments/02_bias.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import load_dbpedia_openai

from turboquant.quantizer import TurboQuantMSE

RESULTS = Path(__file__).resolve().parent.parent / "results"
BITS = (1, 2, 3, 4)


def main() -> None:
    train_np, query_np = load_dbpedia_openai()
    x = torch.from_numpy(train_np[:20_000])
    y = torch.from_numpy(query_np)
    d = x.shape[1]

    fig, axes = plt.subplots(1, len(BITS), figsize=(3.1 * len(BITS), 3.0),
                             sharex=True)
    print(f"{'b':>2} {'alpha (measured)':>17} {'1 - Dmse (pred)':>16} {'2/pi':>7}")
    for ax, b in zip(axes, BITS):
        tq = TurboQuantMSE(d, b, seed=1)
        xt = tq.roundtrip(x)
        ips = (y @ x.T).flatten()
        ipd = (y @ xt.T).flatten()
        alpha = float((ips * ipd).sum() / (ips * ips).sum())  # LS slope
        pred = 1.0 - tq.expected_unit_mse
        extra = f"{2 / torch.pi:>7.4f}" if b == 1 else f"{'-':>7}"
        print(f"{b:>2} {alpha:>17.4f} {pred:>16.4f} {extra}")
        err = (ipd - ips).numpy()
        ax.hist(err, bins=120, range=(-0.1, 0.1), color="#4472c4")
        ax.axvline(0.0, color="k", lw=0.8)
        ax.set_title(f"b={b}, mean={err.mean():+.4f}")
        ax.set_xlabel("IP distortion")
    axes[0].set_ylabel("frequency")
    fig.suptitle("TurboQuant-mse inner-product error (biased; cf. paper Fig. 1b)")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "bias_histograms.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
