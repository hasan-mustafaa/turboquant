"""Structured results output with provenance metadata.

Every benchmark writes through save_results() so each results/*.json records
where its numbers came from: git commit, hardware/device, library versions,
and a timestamp. This is the enforcement mechanism for the project's
no-fake-numbers rule -- a results file without real provenance is not a
results file.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def _provenance(device: str | None) -> dict:
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        if device == "cuda" and torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name()
    except ImportError:
        pass
    try:
        import transformers

        meta["transformers"] = transformers.__version__
    except ImportError:
        pass
    if device is not None:
        meta["device"] = device
    return meta


def model_slug(model_id: str) -> str:
    """Short family tag for results filenames: 'qwen', 'llama', or a
    sanitized fallback. Model-quality benchmarks suffix their output with
    this so runs against different models never overwrite each other
    (which silently happened once before this existed)."""
    low = model_id.lower()
    for family in ("qwen", "llama"):
        if family in low:
            return family
    return low.rsplit("/", 1)[-1].replace(".", "").replace("-", "_")


def save_results(name: str, payload: dict, device: str | None = None) -> Path:
    """Write results/<name>.json with a provenance block; returns the path."""
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(
        {"meta": _provenance(device), **payload}, indent=2))
    return out
