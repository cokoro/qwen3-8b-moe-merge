"""Analyze failure cases for merged models across benchmarks; categorize failures and recommend next experiments."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BENCHMARKS = ["swe", "hle", "asi", "terminal"]

CORRECTNESS_FIELD = {
    "swe": "resolved",
    "hle": "is_correct",
    "asi": "success",
    "terminal": "correct",
}


def detect_benchmark_key(name: str) -> Optional[str]:
    for key in BENCHMARKS:
        if key in name.lower():
            return key
    return None


def truncate(text: str, max_chars: int = 300) -> str:
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def categorize_failure(instance: Dict, bench_key: str) -> str:
    response = str(instance.get("generated", instance.get("response", instance.get("output", ""))))
    tokens = response.split()
    if len(tokens) < 50:
        return "incomplete"
    # heuristic: if the response is very short relative to expected format, it's likely a parse fail
    expected_keywords = {
        "swe": ["diff", "patch", "---", "+++"],
        "hle": ["answer", "therefore", "because", "since"],
        "asi": ["action", "tool", "step", "plan"],
        "terminal": ["$", "bash", "sh", "echo", "cat", "ls"],
    }
    keywords = expected_keywords.get(bench_key, [])
    if keywords and not any(kw in response.lower() for kw in keywords):
        return "parsing_error"
    return "logic_error"


def load_failures(jsonl_path: Path, bench_key: str, max_samples: int) -> Tuple[Dict[str, int], List[Dict]]:
    field = CORRECTNESS_FIELD[bench_key]
    counts: Dict[str, int] = {"logic_error": 0, "parsing_error": 0, "incomplete": 0}
    samples: List[Dict] = []

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inst = json.loads(line)
            val = inst.get(field)
            if val is None or bool(val):
                continue  # skip passing or null (unverified)

            category = categorize_failure(inst, bench_key)
            counts[category] += 1

            if len(samples) < max_samples:
                input_text = inst.get("input", inst.get("problem_statement", inst.get("prompt", "")))
                generated = inst.get("generated", inst.get("response", inst.get("output", "")))
                expected = inst.get("expected", inst.get("patch", inst.get("answer", "")))
                samples.append({
                    "model": jsonl_path.parent.parent.name,
                    "benchmark": bench_key,
                    "input": truncate(input_text),
                    "generated": truncate(generated),
                    "expected": truncate(expected),
                    "category": category,
                })

    return counts, samples


def generate_recommendations(all_counts: Dict[str, Dict[str, int]]) -> List[str]:
    total: Dict[str, int] = {"logic_error": 0, "parsing_error": 0, "incomplete": 0}
    for bench_counts in all_counts.values():
        for cat, n in bench_counts.items():
            total[cat] += n

    dominant = sorted(total, key=lambda k: total[k], reverse=True)
    recs = []
    category_recs = {
        "logic_error": (
            "Augment training data with more chain-of-thought reasoning examples "
            "and verified step-by-step solutions to reduce logic errors."
        ),
        "parsing_error": (
            "Add format-constrained fine-tuning examples that reinforce expected output structure "
            "(patches, JSON tool calls, shell commands) to address parsing failures."
        ),
        "incomplete": (
            "Increase max_new_tokens during evaluation and consider longer-context training samples "
            "to fix truncated responses."
        ),
    }
    for cat in dominant[:3]:
        if total[cat] > 0:
            recs.append(category_recs[cat])
    return recs


def main():
    parser = argparse.ArgumentParser(
        description="Analyze failure cases for merged models across benchmarks"
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="results/failure_analysis.json")
    parser.add_argument("--max_samples", type=int, default=20)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    analyses: List[Dict] = []
    all_samples: List[Dict] = []
    aggregate_counts: Dict[str, Dict[str, int]] = {}

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if "merged" not in model_dir.name.lower():
            continue

        model_analysis: Dict = {"model": model_dir.name, "benchmarks": {}}

        for bench_key in BENCHMARKS:
            bench_dir = model_dir / bench_key
            if not bench_dir.exists():
                # also try substring match
                matches = [d for d in model_dir.iterdir() if d.is_dir() and bench_key in d.name.lower()]
                if not matches:
                    continue
                bench_dir = matches[0]

            jsonl_path = bench_dir / "detailed_results.jsonl"
            if not jsonl_path.exists():
                continue

            try:
                counts, samples = load_failures(jsonl_path, bench_key, args.max_samples)
            except Exception as exc:
                print(f"Warning: failed to load {jsonl_path}: {exc}")
                continue

            total_failures = sum(counts.values())
            model_analysis["benchmarks"][bench_key] = {
                "total_failures": total_failures,
                "categories": counts,
            }
            all_samples.extend(samples)

            key = f"{model_dir.name}/{bench_key}"
            aggregate_counts[key] = counts

        analyses.append(model_analysis)

    recommendations = generate_recommendations(aggregate_counts)
    output = {"analyses": analyses, "sample_failures": all_samples, "next_experiments": recommendations}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved failure analysis to {output_path}\n")

    for analysis in analyses:
        print(f"Model: {analysis['model']}")
        for bench, stats in analysis["benchmarks"].items():
            cats = stats["categories"]
            print(
                f"  {bench:12s}  total={stats['total_failures']:4d}  "
                f"logic={cats['logic_error']:4d}  "
                f"parse={cats['parsing_error']:4d}  "
                f"incomplete={cats['incomplete']:4d}"
            )
        print()

    if recommendations:
        print("Recommended next experiments:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()
