"""Summarize GPU hours and costs across training, evaluation, and merging from training_metrics.json files and results directory."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

EXPERTS = ["swe", "hle", "asi", "terminal"]


def load_training_hours(checkpoints_dir: Path, expert: str) -> Optional[float]:
    metrics_path = checkpoints_dir / f"expert_{expert}" / "training_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text())
        if isinstance(data, list):
            return sum(entry.get("train_runtime", 0) for entry in data) / 3600
        return data.get("train_runtime", 0) / 3600
    except Exception as exc:
        print(f"Warning: could not load {metrics_path}: {exc}")
        return None


def count_evaluated_models(results_dir: Path) -> int:
    if not results_dir.exists():
        return 0
    return sum(1 for p in results_dir.iterdir() if p.is_dir())


def main():
    parser = argparse.ArgumentParser(description="Summarize GPU hours and costs across training, evaluation, and merging")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--gpu_rate", type=float, default=1.5)
    parser.add_argument("--output", default="results/cost_summary.json")
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    results_dir = Path(args.results_dir)
    rate = args.gpu_rate

    training_details: List[Dict] = []
    total_train_hours = 0.0

    for expert in EXPERTS:
        hours = load_training_hours(checkpoints_dir, expert)
        training_details.append({"expert": expert, "hours": hours if hours is not None else 0.0})
        if hours is not None:
            total_train_hours += hours

    n_models = count_evaluated_models(results_dir)
    eval_hours = n_models * 4 * 2.0
    merging_hours = 0.2

    train_cost = total_train_hours * rate
    eval_cost = eval_hours * rate
    merge_cost = merging_hours * rate
    total_hours = total_train_hours + eval_hours + merging_hours
    total_cost = total_hours * rate

    summary = {
        "training": {"hours": round(total_train_hours, 2), "cost": round(train_cost, 2), "details": training_details},
        "evaluation": {"hours": round(eval_hours, 2), "cost": round(eval_cost, 2), "n_models": n_models},
        "merging": {"hours": merging_hours, "cost": round(merge_cost, 2)},
        "total": {"hours": round(total_hours, 2), "cost": round(total_cost, 2), "gpu_rate": rate},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved cost summary to {output_path}\n")

    col = [18, 10, 10]
    sep = "-" * (sum(col) + len(col) - 1)
    print(f"{'Phase':<{col[0]}} {'Hours':>{col[1]}} {'Cost ($)':>{col[2]}}")
    print(sep)
    for d in training_details:
        h = d["hours"]
        print(f"  train/{d['expert']:<{col[0]-9}} {h:>{col[1]}.2f} {h*rate:>{col[2]}.2f}")
    print(f"{'Training':<{col[0]}} {total_train_hours:>{col[1]}.2f} {train_cost:>{col[2]}.2f}")
    print(f"{'Evaluation':<{col[0]}} {eval_hours:>{col[1]}.2f} {eval_cost:>{col[2]}.2f}")
    print(f"{'Merging':<{col[0]}} {merging_hours:>{col[1]}.2f} {merge_cost:>{col[2]}.2f}")
    print(sep)
    print(f"{'Total':<{col[0]}} {total_hours:>{col[1]}.2f} {total_cost:>{col[2]}.2f}")
    print(f"\nGPU rate: ${rate}/hr  |  evaluated model dirs: {n_models}")


if __name__ == "__main__":
    main()
