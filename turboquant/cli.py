"""turboquant-bench: one entry point for every validation/benchmark run.

Two execution profiles, matching the two environments this project targets:

  --profile local   small ungated model (Qwen2.5-0.5B-Instruct), short
                    sweeps -- runnable on a laptop or single consumer GPU.
  --profile cloud   gated Llama-3.2-1B-Instruct (export HF_TOKEN first),
                    full-length sweeps -- intended for a rented A100/H100.

Device (cuda > mps > cpu) is auto-detected inside each experiment; profiles
only choose models and sweep sizes. Any extra arguments are forwarded to the
underlying script and override the profile's defaults (argparse last-wins):

  turboquant-bench ppl --profile cloud --seqs 80
  turboquant-bench needle --profile local
  turboquant-bench cuda                 # paper Table 2 protocol
  turboquant-bench all --profile local  # everything runnable in that profile

Runs from a repository checkout (experiments/ lives next to the package);
results land in results/*.json with provenance metadata.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"

# command -> (script, takes --model flag)
_COMMANDS: dict[str, tuple[str, bool]] = {
    "validate": ("00_quantizer_validation.py", False),
    "distortion": ("01_distortion_rate.py", False),
    "bias": ("02_bias.py", False),
    "ppl": ("03_perplexity.py", True),
    "needle": ("04_needle.py", True),
    "memory": ("05_memory_bench.py", True),
    "cuda": ("06_cuda_bench.py", False),
    "qjl": ("07_qjl_validation.py", False),
}

_PROFILES: dict[str, dict[str, list[str]]] = {
    "local": {
        "model": ["--model", "Qwen/Qwen2.5-0.5B-Instruct"],
        "ppl": ["--seqs", "10"],
        "needle": ["--lengths", "1024", "2048", "4096"],
    },
    "cloud": {
        "model": ["--model", "meta-llama/Llama-3.2-1B-Instruct"],
        "ppl": ["--seqs", "40"],
        "needle": ["--lengths", "1024", "2048", "4096", "8192", "16384"],
    },
}

# `all` in profile order; cuda only makes sense with a CUDA device but will
# honestly report whatever device it finds.
_ALL = ["validate", "distortion", "bias", "qjl", "ppl", "needle", "memory", "cuda"]


def _run(command: str, profile: str, extra: list[str]) -> int:
    script, takes_model = _COMMANDS[command]
    path = _EXPERIMENTS / script
    if not path.exists():
        sys.exit(f"error: {path} not found -- turboquant-bench must run from "
                 "a repository checkout (pip install -e .)")
    args = [sys.executable, "-u", str(path)]
    prof = _PROFILES[profile]
    if takes_model:
        args += prof["model"]
    args += prof.get(command, [])
    args += extra  # last-wins overrides
    print(f"$ {' '.join(args[1:])}", flush=True)
    return subprocess.run(args).returncode


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="turboquant-bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("command", choices=[*_COMMANDS, "all"])
    ap.add_argument("--profile", choices=list(_PROFILES), default="local")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="forwarded to the underlying experiment script")
    ns = ap.parse_args()
    commands = _ALL if ns.command == "all" else [ns.command]
    for cmd in commands:
        rc = _run(cmd, ns.profile, ns.rest)
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
