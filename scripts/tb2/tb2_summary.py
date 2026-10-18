"""Merge Harbor job dirs into one Terminal-Bench 2.0 result (same row fields as src/evaluation/harbor_results.py).

A task can appear in several jobs (split lanes, reruns after an environment fix); the trial that finished last wins.
The oracle jobs (reference solutions) define which tasks are solvable on this host, so the pass rate is reported
both on all 89 tasks (official denominator; errors count as failures) and on the oracle-solvable subset.

    python3 tb2_summary.py --model-jobs jobs/qwen3-8b-base-* --oracle-jobs jobs/oracle-* --output results/qwen3-8b-base
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

DATASET = Path("/root/gpufree-data/datasets/terminal-bench-2.0")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def load_trials(job_dirs):
    """Latest finished trial per task across job dirs."""
    best = {}
    for job in job_dirs:
        for res in Path(job).glob("*/result.json"):
            d = json.loads(res.read_text())
            if not d.get("finished_at"):
                continue
            rewards = (d.get("verifier_result") or {}).get("rewards") or {}
            exc = d.get("exception_info") or {}
            agent = d.get("agent_result") or {}
            row = {
                "instance_id": d.get("task_name"),
                "trial": d.get("trial_name"),
                "job": Path(job).name,
                "reward": rewards.get("reward"),
                "is_correct": rewards.get("reward") is not None and float(rewards["reward"]) >= 1.0,
                "exception": exc.get("exception_type") if exc else None,
                "n_input_tokens": agent.get("n_input_tokens"),
                "n_output_tokens": agent.get("n_output_tokens"),
                "started_at": d.get("started_at"),
                "finished_at": d.get("finished_at"),
            }
            prev = best.get(row["instance_id"])
            if prev is None or row["finished_at"] > prev["finished_at"]:
                best[row["instance_id"]] = row
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-jobs", nargs="+", required=True)
    ap.add_argument("--oracle-jobs", nargs="*", default=[])
    ap.add_argument("--model-name", default="Qwen3-8B")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tasks = sorted(p.parent.name for p in DATASET.glob("*/task.toml"))
    model = load_trials(args.model_jobs)
    oracle = load_trials(args.oracle_jobs) if args.oracle_jobs else {}

    rows = []
    for t in tasks:
        r = model.get(t) or {"instance_id": t, "reward": None, "is_correct": False, "exception": "NotRun"}
        if oracle:
            o = oracle.get(t)
            r["oracle_reward"] = o["reward"] if o else None
            r["oracle_solvable"] = bool(o and o["is_correct"])
        rows.append(r)

    n = len(rows)
    k = sum(r["is_correct"] for r in rows)
    lo, hi = wilson(k, n)
    results = {
        "benchmark": "terminal_bench_2.0",
        "dataset": "harborframework/terminal-bench-2.0@f2e8c75e23add71613117eecc9498f53bcd7e04e",
        "model": args.model_name,
        "agent": "terminus-2",
        "metric": "pass_rate",
        "n_tasks": n,
        "n_resolved": k,
        "pass_rate": k / n if n else None,
        "pass_rate_95ci_wilson": [lo, hi],
        "n_errored": sum(r["exception"] is not None for r in rows),
        "exceptions": dict(Counter(r["exception"] for r in rows if r["exception"])),
        "n_not_run": sum(r["exception"] == "NotRun" for r in rows),
        "model_jobs": [Path(j).name for j in args.model_jobs],
    }
    if oracle:
        solv = [r for r in rows if r["oracle_solvable"]]
        ks = sum(r["is_correct"] for r in solv)
        results.update({
            "oracle_jobs": [Path(j).name for j in args.oracle_jobs],
            "n_oracle_solvable": len(solv),
            "n_resolved_on_oracle_solvable": ks,
            "pass_rate_on_oracle_solvable": ks / len(solv) if solv else None,
            "pass_rate_on_oracle_solvable_95ci_wilson": list(wilson(ks, len(solv))),
            "oracle_unsolvable_tasks": [r["instance_id"] for r in rows if not r["oracle_solvable"]],
            "resolved_but_oracle_unsolvable": [r["instance_id"] for r in rows if r["is_correct"] and not r["oracle_solvable"]],
        })
    results["resolved_tasks"] = [r["instance_id"] for r in rows if r["is_correct"]]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    with open(out / "detailed_results.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
