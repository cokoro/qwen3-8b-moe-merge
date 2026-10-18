"""Compute bootstrap 95% confidence intervals for each model/benchmark accuracy from detailed per-instance results."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


CORRECTNESS_FIELD = {
    "swe": "resolved",
    "hle": "is_correct",
    "asi": "success",
    "terminal": "correct",
}


def detect_benchmark_key(name: str) -> Optional[str]:
    for key in CORRECTNESS_FIELD:
        if key in name.lower():
            return key
    return None


def load_correctness(jsonl_path: Path, bench_key: str) -> List[int]:
    field = CORRECTNESS_FIELD[bench_key]
    values = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instance = json.loads(line)
            val = instance.get(field)
            if val is None:
                continue  # skip nulls/unverified (swe resolved can be null)
            values.append(int(bool(val)))
    return values


def bootstrap_ci(values: List[int], n_bootstrap: int, rng: np.random.Generator) -> Dict:
    arr = np.array(values, dtype=float)
    n = len(arr)
    point_estimate = float(arr.mean())
    boot_means = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    return {
        "point_estimate": point_estimate,
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute bootstrap 95% CIs for model/benchmark accuracy")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="results/confidence_intervals.json")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rng = np.random.default_rng(args.seed)
    records = []

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for bench_dir in sorted(model_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            jsonl_path = bench_dir / "detailed_results.jsonl"
            if not jsonl_path.exists():
                continue
            bench_key = detect_benchmark_key(bench_dir.name) or detect_benchmark_key(model_dir.name)
            if bench_key is None:
                print(f"Warning: could not detect benchmark for {bench_dir}, skipping")
                continue
            try:
                values = load_correctness(jsonl_path, bench_key)
            except Exception as exc:
                print(f"Warning: failed to load {jsonl_path}: {exc}")
                continue
            if not values:
                continue
            ci = bootstrap_ci(values, args.n_bootstrap, rng)
            records.append({
                "model": model_dir.name,
                "benchmark": bench_dir.name,
                "n_instances": len(values),
                "n_bootstrap": args.n_bootstrap,
                **ci,
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    print(f"Saved {len(records)} records to {output_path}\n")

    if records:
        w = [30, 25, 8, 8, 8, 8]
        print(f"{'Model':<{w[0]}} {'Benchmark':<{w[1]}} {'N':>{w[2]}} {'Point':>{w[3]}} {'Lo':>{w[4]}} {'Hi':>{w[5]}}")
        print("-" * (sum(w) + len(w) - 1))
        for r in records:
            print(
                f"{r['model']:<{w[0]}} {r['benchmark']:<{w[1]}} "
                f"{r['n_instances']:>{w[2]}} {r['point_estimate']:>{w[3]}.3f} "
                f"{r['ci_lower']:>{w[4]}.3f} {r['ci_upper']:>{w[5]}.3f}"
            )


if __name__ == "__main__":
    main()
