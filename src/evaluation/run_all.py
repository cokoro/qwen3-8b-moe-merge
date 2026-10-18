"""Run all 4 benchmark evaluations for a single model via subprocess calls."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_benchmark(benchmark: str, model_path: str, output_dir: str, base_model: str = None):
    module_map = {
        "swe": "src.evaluation.run_swe",
        "hle": "src.evaluation.run_hle",
        "asi": "src.evaluation.run_asi",
        "terminal": "src.evaluation.run_terminal",
    }
    module = module_map.get(benchmark)
    if not module:
        print(f"Unknown benchmark: {benchmark}")
        return False

    cmd = [sys.executable, "-m", module, "--model", model_path, "--output", str(Path(output_dir) / benchmark)]
    if base_model:
        cmd += ["--base_model", base_model]

    print(f"\n{'='*50}")
    print(f"Running {benchmark}...")
    print(f"{'='*50}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all benchmark evaluations for a model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["swe", "hle", "asi", "terminal"])
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running All Benchmarks")
    print(f"Model: {args.model}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    print(f"{'='*60}")

    statuses = {}
    for benchmark in args.benchmarks:
        try:
            ok = run_benchmark(benchmark, args.model, args.output, args.base_model)
            statuses[benchmark] = "completed" if ok else "failed"
        except Exception as exc:
            print(f"Error running {benchmark}: {exc}")
            statuses[benchmark] = f"error: {exc}"

    print(f"\n{'='*60}")
    print("Summary:")
    for b, s in statuses.items():
        print(f"  {b:12s}: {s}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
