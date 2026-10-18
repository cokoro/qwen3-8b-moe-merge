"""Normalize a Harbor job directory into results.json + detailed_results.jsonl.

Each trial's ``result.json`` holds ``verifier_result.rewards.reward`` (1.0 = solved)
and ``exception_info`` (agent/environment errors). Errored trials count as failures
in the pass rate and are reported separately so infra problems stay visible.

    python -m src.evaluation.harbor_results --job_dir results/base/terminal/jobs/tb2 \
        --benchmark terminal_bench_2.0 --output results/base/terminal
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_dir", required=True, nargs="+",
                        help="one or more job dirs (e.g. an interrupted job and its resume); "
                             "for a task seen in several, the finished trial from the last dir wins")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", default="pass_rate")
    args = parser.parse_args()

    jobs = [Path(j) for j in args.job_dir]
    by_task = {}
    for job in jobs:
        for res in sorted(job.glob("*/result.json")):
            d = json.loads(res.read_text())
            if d.get("finished_at"):
                by_task[d.get("task_name")] = d
    rows = []
    for d in by_task.values():
        rewards = (d.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")
        exc = d.get("exception_info") or {}
        agent = d.get("agent_result") or {}
        rows.append({
            "instance_id": d.get("task_name"),
            "trial": d.get("trial_name"),
            "reward": reward,
            "is_correct": reward is not None and float(reward) >= 1.0,
            "exception": exc.get("exception_type") if exc else None,
            "n_input_tokens": agent.get("n_input_tokens"),
            "n_output_tokens": agent.get("n_output_tokens"),
            "started_at": d.get("started_at"),
            "finished_at": d.get("finished_at"),
        })

    job_result = json.loads((jobs[-1] / "result.json").read_text()) if (jobs[-1] / "result.json").exists() else {}
    n = len(rows)
    results = {
        "benchmark": args.benchmark,
        "metric": args.metric,
        "n_tasks": n,
        "n_resolved": sum(r["is_correct"] for r in rows),
        args.metric: (sum(r["is_correct"] for r in rows) / n) if n else None,
        "n_errored": sum(r["exception"] is not None for r in rows),
        "exceptions": dict(Counter(r["exception"] for r in rows if r["exception"])),
        "started_at": job_result.get("started_at"),
        "finished_at": job_result.get("finished_at"),
        "job_dirs": [str(j) for j in jobs],
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    with open(out / "detailed_results.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
