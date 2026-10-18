"""Score an ASI-Bench run one task at a time, each under a wall-clock timeout.

``asibench score`` scores a whole run in one process; a single scorer that hangs (e.g.
waiting on an unavailable VLM judge) blocks every result. Here each task directory is
scored in its own ``asibench score`` call on a temporary run dir that contains only that
task, with ``--timeout`` seconds. A timed-out task gets final_score 0 and is flagged
``scorer_timeout`` -- identical treatment for every model. The per-task reports are
merged into one local_score_seed31415.json (same schema) for src.evaluation.asi_results.

    python -m src.evaluation.asi_score_per_task --run results/base/asi/run --output results/base/asi
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ASI = ".venv-asi/bin/asibench"


def score_task(task_dir: Path, run_dir: Path, args) -> list:
    with tempfile.TemporaryDirectory(prefix="asi_score_") as tmp:
        tmp = Path(tmp)
        run = tmp / "run"
        run.mkdir()
        os.symlink(task_dir.resolve(), run / task_dir.name)
        for f in run_dir.iterdir():  # run-level metadata files
            if f.is_file():
                shutil.copy(f, run / f.name)
        report = tmp / "score.json"
        env = dict(os.environ, ASIBENCH_JUDGE_API_BASE=args.judge_api_base, ASIBENCH_JUDGE_API_PROTOCOL="openai",
                   ASIBENCH_JUDGE_API_KEY_ENV="NO_JUDGE_KEY", NO_JUDGE_KEY="none")
        try:
            subprocess.run([ASI, "score", "--repo", "seed31415", "--results-dir", str(run),
                            "--instances-dir", args.instances, "--tasks-dir", args.tasks_dir,
                            "--output", str(report)], env=env, timeout=args.timeout,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            pass
        if report.exists():
            return json.loads(report.read_text()).get("results", [])
    inst = next((p.name.split("__b")[0] for p in task_dir.glob("*__b*.json")), task_dir.name)
    return [{"task_id": task_dir.name, "instance_id": inst, "prompt_level": None, "final_score": 0.0,
             "max_score": 100.0, "hard_gates_passed": False, "scorer_internal_error": True,
             "gate_results": [{"passed": False, "message": f"scorer_timeout ({args.timeout}s)"}],
             "evaluation_status": "scorer_timeout"}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--instances", default="data/eval/asi_seed31415")
    p.add_argument("--tasks_dir", default="third_party/ASI-Bench/tasks")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--judge_api_base", default="http://127.0.0.1:9/v1")
    args = p.parse_args()

    run_dir = Path(args.run)
    # task dirs hold <instance>__b<level>.json results; skip helper dirs such as instances/
    tasks = sorted(d for d in run_dir.iterdir() if d.is_dir() and any(d.glob("*__b*.json")))
    with ThreadPoolExecutor(args.parallel) as pool:
        per_task = list(pool.map(lambda t: score_task(t, run_dir, args), tasks))
    results = [r for rs in per_task for r in rs]
    n_timeout = sum(r.get("evaluation_status") == "scorer_timeout" for r in results)
    total = sum(r.get("final_score") or 0 for r in results)
    total_max = sum(r.get("max_score") or 100 for r in results)
    report = {"repo": "seed31415", "official": False, "score_scope": "public_local_per_task",
              "instance_count": len(results), "scorer_error_count": sum(bool(r.get("scorer_internal_error")) for r in results),
              "scorer_timeouts": n_timeout, "per_task_timeout_s": args.timeout,
              "total_score": total, "total_max_score": total_max,
              "mean_percent": 100 * total / total_max if total_max else 0.0, "results": results}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "local_score_seed31415.json").write_text(json.dumps(report, indent=2))
    print(f"{run_dir}: {len(results)} instances, mean {report['mean_percent']:.2f}%, timeouts {n_timeout}")


if __name__ == "__main__":
    main()
