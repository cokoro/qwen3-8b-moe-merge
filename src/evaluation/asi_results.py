"""Normalize an ASI-Bench local score report into results.json + detailed_results.jsonl.

``score`` is the mean of final_score / max_score over instances (0-1); an instance
counts as ``is_correct`` when it passes its hard gates and scores > 0.

    python -m src.evaluation.asi_results --report results/base/asi/local_score_seed31415.json \
        --output results/base/asi
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    rows = []
    for r in report.get("results", []):
        frac = (r.get("final_score") or 0.0) / (r.get("max_score") or 100.0)
        rows.append({
            "instance_id": r["instance_id"],
            "task_id": r["task_id"],
            "prompt_level": r.get("prompt_level"),
            "score": frac,
            "hard_gates_passed": r.get("hard_gates_passed"),
            "is_correct": bool(r.get("hard_gates_passed")) and frac > 0,
            "scorer_internal_error": r.get("scorer_internal_error"),
            "gate_messages": [g.get("message") for g in r.get("gate_results", []) if not g.get("passed")],
        })
    n = len(rows)
    results = {
        "benchmark": "asi_bench_seed31415",
        "protocol": "direct_llm single-turn; public references, local non-official scoring",
        "metric": "score",
        "n_tasks": n,
        "score": (sum(r["score"] for r in rows) / n) if n else None,
        "hard_gate_pass_rate": (sum(bool(r["hard_gates_passed"]) for r in rows) / n) if n else None,
        "n_scorer_errors": report.get("scorer_error_count"),
        "official": report.get("official", False),
    }
    out = Path(args.output)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    with open(out / "detailed_results.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
