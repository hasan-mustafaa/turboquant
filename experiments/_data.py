"""Dataset loader for the paper's Section 4.1 setup: DBpedia entities encoded
with OpenAI text-embedding-3-large at 1536 dims, 100k train + 1k query sample.

Streams only the needed rows from the 1M-row HF dataset and caches a local
.npz (data/ is gitignored), so the multi-GB full download never happens.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATASET = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_dbpedia_openai(
    n_train: int = 100_000, n_query: int = 1_000
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (train, query) float32 arrays, unit-normalized rows."""
    cache = DATA_DIR / f"dbpedia_openai1536_{n_train}_{n_query}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["train"], z["query"]

    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train", streaming=True)
    first = next(iter(ds))
    emb_key = next(k for k, v in first.items()
                   if isinstance(v, list) and len(v) >= 512)
    rows = np.empty((n_train + n_query, len(first[emb_key])), dtype=np.float32)
    for i, row in enumerate(ds.take(n_train + n_query)):
        rows[i] = row[emb_key]
        if (i + 1) % 20_000 == 0:
            print(f"  streamed {i + 1}/{n_train + n_query} rows")
    # OpenAI embeddings arrive unit-norm; renormalize anyway so the unit-sphere
    # theory applies exactly (float32 storage drifts norms by ~1e-4).
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    train, query = rows[:n_train], rows[n_train:]
    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cache, train=train, query=query)
    print(f"  cached to {cache}")
    return train, query
