"""Phase 2 validation report: empirical quantizer MSE vs analytic theory.

Prints, for d in {64, 128, 1536} and b in {1..4}:
- Monte-Carlo end-to-end MSE on random unit vectors,
- the analytic prediction d * C(f_X, b) from the Phase 1 solver,
- their ratio (pass criterion: within ~1%),
plus the exact-Beta vs Gaussian-limit codebook comparison at d = 64, encode/
decode throughput on CPU and any available accelerator (CUDA or MPS), and
storage accounting.

Run: .venv/bin/python experiments/00_quantizer_validation.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device import default_device, synchronize

from turboquant.quantizer import TurboQuantMSE

torch.manual_seed(0)


def unit_vectors(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True)


def main() -> None:
    print("Empirical MSE (Monte-Carlo) vs analytic d*C(f_X,b) [exact-Beta codebook]")
    print(f"{'d':>5} {'b':>2} {'empirical':>11} {'analytic':>11} {'ratio':>7}")
    for d, n in ((64, 200_000), (128, 100_000), (1536, 10_000)):
        x = unit_vectors(n, d, seed=d)
        for b in (1, 2, 3, 4):
            tq = TurboQuantMSE(d, b, seed=1)
            mse = (x - tq.roundtrip(x)).pow(2).sum(-1).mean().item()
            print(f"{d:>5} {b:>2} {mse:>11.6f} {tq.expected_unit_mse:>11.6f} "
                  f"{mse / tq.expected_unit_mse:>7.4f}")

    print("\nCodebook choice at d=64 (paper deploys precomputed Gaussian-limit tables):")
    x = unit_vectors(200_000, 64, seed=64)
    for b in (1, 2, 3, 4):
        m_s = (x - TurboQuantMSE(64, b, seed=1).roundtrip(x)).pow(2).sum(-1).mean().item()
        m_g = (x - TurboQuantMSE(64, b, seed=1, codebook="gaussian")
               .roundtrip(x)).pow(2).sum(-1).mean().item()
        print(f"  b={b}: exact-Beta {m_s:.6f}  Gaussian-limit {m_g:.6f}  "
              f"(+{100 * (m_g / m_s - 1):.2f}%)")

    print("\nStorage (b=4, fp16 norms):")
    for d in (64, 128):
        q = TurboQuantMSE(d, 4).quantize(unit_vectors(1000, d, seed=0))
        print(f"  d={d}: {q.bits_per_coord:.3f} bits/coord -> "
              f"{16.0 / q.bits_per_coord:.2f}x vs fp16")

    print("\nThroughput (b=4, d=128, batch 100k, best of 3):")
    x = unit_vectors(100_000, 128, seed=7)
    accel = default_device()
    devices = ["cpu"] + ([accel] if accel != "cpu" else [])
    for dev in devices:
        tq = TurboQuantMSE(128, 4, seed=1, device=dev)
        xd = x.to(dev)
        tq.quantize(xd)  # warmup
        tt = []
        for _ in range(3):
            t0 = time.perf_counter()
            q = tq.quantize(xd)
            _ = tq.dequantize(q)
            synchronize(dev)
            tt.append(time.perf_counter() - t0)
        best = min(tt)
        gbps = x.nbytes / best / 1e9
        print(f"  {dev:>4}: {best * 1e3:8.2f} ms roundtrip "
              f"({1e-6 * len(x) / best:6.1f} M vec/s, {gbps:5.2f} GB/s fp32 in)")
        if dev != "cpu":
            q_cpu = TurboQuantMSE(128, 4, seed=1).quantize(x)
            match = (q.codes.cpu() == q_cpu.codes).float().mean().item()
            print(f"        {dev.upper()}/CPU code agreement: {100 * match:.3f}%")


if __name__ == "__main__":
    main()
