"""Shared helpers for writing expert data splits."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import jsonlines


def dedup_by_prompt(samples: List[Dict]) -> List[Dict]:
    """Drop exact duplicates on the normalized user prompt."""
    seen = set()
    kept = []
    for s in samples:
        key = " ".join(m["content"] for m in s["messages"] if m["role"] == "user").lower().split()
        key = " ".join(key)
        if key in seen:
            continue
        seen.add(key)
        kept.append(s)
    return kept


def write_splits(samples: List[Dict], output_path: Path, max_samples: int,
                 val_fraction: float = 0.05, seed: int = 42) -> Dict:
    """Dedup, shuffle, cap, split into train/validation, and write stats.json."""
    n_raw = len(samples)
    samples = dedup_by_prompt(samples)
    n_dedup = len(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)
    samples = samples[:max_samples]
    n_val = max(1, int(len(samples) * val_fraction)) if samples else 0
    splits = {"validation": samples[:n_val], "train": samples[n_val:]}

    output_path.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with jsonlines.open(output_path / f"{name}.jsonl", "w") as f:
            for s in rows:
                f.write({**s, "split": name})

    def chars(s):
        return sum(len(m["content"]) for m in s["messages"])

    stats = {
        "raw": n_raw,
        "after_dedup": n_dedup,
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "sources": dict(Counter(s["source"] for s in samples)),
        "mean_chars": round(sum(map(chars, samples)) / len(samples), 1) if samples else 0,
        "max_chars": max(map(chars, samples)) if samples else 0,
        "seed": seed,
    }
    (output_path / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Saved {stats['train']} train / {stats['validation']} validation to {output_path}")
    return stats
