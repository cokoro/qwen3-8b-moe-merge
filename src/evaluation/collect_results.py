"""Collect and summarize evaluation results from all models and benchmarks."""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_result(result_file: Path) -> Optional[Dict]:
    if not result_file.exists():
        return None
    with open(result_file) as f:
        return json.load(f)


# Primary metric per benchmark for the summary pivot.
PRIMARY_METRIC = {
    "swe": "resolved_rate",
    "hle": "accuracy",
    "asi": "score",
    "terminal": "pass_rate",
}


def collect_all_results(results_dir: str) -> pd.DataFrame:
    results_path = Path(results_dir)
    all_results: List[Dict] = []

    for model_dir in sorted(results_path.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):  # _smoke etc.
            continue
        model_name = model_dir.name
        for benchmark_dir in model_dir.iterdir():
            if not benchmark_dir.is_dir():
                continue
            benchmark_name = benchmark_dir.name
            result = load_result(benchmark_dir / "results.json")
            if result:
                all_results.append({
                    "model": model_name,
                    "benchmark": benchmark_name,
                    **result,
                })

    if not all_results:
        print("No results found!")
        return pd.DataFrame()
    return pd.DataFrame(all_results)


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to model x benchmark with the primary metric for each benchmark."""
    rows: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        benchmark = row["benchmark"]
        metric = PRIMARY_METRIC.get(benchmark, "accuracy")
        model = row["model"]
        value = row.get(metric)
        rows.setdefault(model, {})[benchmark] = value
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    parser = argparse.ArgumentParser(description="Collect evaluation results into a summary")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="results/summary.json")
    args = parser.parse_args()

    df = collect_all_results(args.results_dir)
    if df.empty:
        return

    summary_table = create_summary_table(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "results": df.to_dict(orient="records"),
        "summary_table": summary_table.to_dict(),
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str))

    csv_path = output_path.with_suffix(".csv")
    summary_table.to_csv(csv_path)

    print("\nSummary table (primary metric per benchmark):")
    print(summary_table.to_string())
    print(f"\nSaved to {output_path} and {csv_path}")


if __name__ == "__main__":
    main()
