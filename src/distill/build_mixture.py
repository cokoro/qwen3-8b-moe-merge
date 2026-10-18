"""Combine per-domain distillation files into one balanced student training set.

    python -m src.distill.build_mixture --per_domain 1200 --out data/distill
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/distill")
    p.add_argument("--per_domain", type=int, default=1200)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/distill")
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for d in ["swe", "hle", "asi", "terminal"]:
        f = Path(args.src) / f"{d}.jsonl"
        if not f.exists():
            print(f"missing {f}, skipping")
            continue
        dom = [json.loads(l) for l in open(f) if l.strip()]
        rng.shuffle(dom)
        rows += dom[: args.per_domain]
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_fraction))
    out = Path(args.out)
    for name, part in [("validation", rows[:n_val]), ("train", rows[n_val:])]:
        with open(out / f"{name}.jsonl", "w") as fh:
            for r in part:
                fh.write(json.dumps({**r, "split": name}) + "\n")
    stats = {"train": len(rows) - n_val, "validation": n_val,
             "by_source": dict(Counter(r["source"] for r in rows)), "by_check": dict(Counter(r["check"] for r in rows))}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
