"""Assemble logs/<run_id>/ + logs/resources.csv from the raw logs of both hosts.

Inputs (DGX checkout): logs/*.log, results/experiment_times.csv (timed_run.sh), checkpoints/*/training_metrics.json,
checkpoints/expert_mopd/mopd_log.jsonl, and the x86 host's logs/ + results/experiment_times.csv copied to x86/.
Output layout per run_id:
  logs/<run_id>/metadata.json   host, GPU, start/end, seconds, exit status, command log names, config refs
  logs/<run_id>/train.log | eval.log   the raw log (training or evaluation)
  logs/<run_id>/metrics.jsonl   per-step metrics (training runs)
  logs/<run_id>/errors.log      error / retry / timeout lines grepped from the raw log ("none" if empty)
logs/resources.csv: run_id, host, gpu, gpu_count, seconds, gpu_hours, peak_mem_gb, est_cost_usd

    python scripts/collect_logs.py --x86 x86 --dgx_rate 1.5 --x86_rate 1.0
"""

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

ERR = re.compile(r"(error|exception|traceback|timeout|timed out|retry|failed|oom|killed)", re.I)


def read_times(path, host):
    rows = []
    if not Path(path).exists():
        return rows
    for line in open(path):
        p = line.strip().split(",")
        if len(p) >= 5:
            rows.append({"run_id": p[0], "start": p[1], "end": p[2], "seconds": int(p[3]), "status": p[4], "host": host})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x86", default="x86", help="copy of the x86 host's swe-eval dir (logs/, results/)")
    ap.add_argument("--out", default="logs_submission")
    ap.add_argument("--dgx_rate", type=float, default=1.5, help="USD per GPU-hour (owned DGX Spark, amortised estimate)")
    ap.add_argument("--x86_rate", type=float, default=1.0, help="USD per GPU-hour (rented L40S container)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hosts = {"dgx": {"gpu": "NVIDIA GB10 (DGX Spark)", "logs": Path("logs"), "rate": args.dgx_rate},
             "x86": {"gpu": "NVIDIA L40S 48GB", "logs": Path(args.x86) / "logs", "rate": args.x86_rate}}
    runs = read_times("results/experiment_times.csv", "dgx") + read_times(Path(args.x86) / "results/experiment_times.csv", "x86")
    # keep the last attempt of each run_id (earlier attempts stay visible in errors via status)
    last = {}
    for r in runs:
        last.setdefault(r["run_id"], []).append(r)

    res_rows = []
    for run_id, attempts in sorted(last.items()):
        r = attempts[-1]
        h = hosts[r["host"]]
        d = out / run_id
        d.mkdir(exist_ok=True)
        kind = "train" if run_id.startswith(("train_", "merge_", "fold_")) else "eval"
        # raw log: timed_run name -> the log file the scheduler wrote
        cands = [h["logs"] / f"{run_id}.log", h["logs"] / f"{run_id.replace('eval_', 'eval_')}.log"]
        m = re.match(r"(swe|terminal)_(.+)", run_id)
        if m:
            cands.append(h["logs"] / f"eval_{m.group(2)}_{m.group(1)}.log")
        if run_id.startswith("train_"):
            cands.append(h["logs"] / f"{run_id}.log")
        raw = next((c for c in cands if c.exists()), None)
        errors = []
        if raw:
            shutil.copy(raw, d / f"{kind}.log")
            errors = [l.rstrip() for l in open(raw, errors="ignore") if ERR.search(l) and "it/s" not in l][:400]
        (d / "errors.log").write_text("\n".join(errors) + "\n" if errors else "none (no error/retry/timeout lines)\n")
        peak = None
        if run_id.startswith("train_"):
            name = run_id[len("train_"):]
            tm = Path("checkpoints") / f"expert_{name}" / "training_metrics.json"
            if tm.exists():
                t = json.loads(tm.read_text())
                peak = t.get("peak_memory_gb")
                hist = t.get("log_history") or []
                with open(d / "metrics.jsonl", "w") as fh:
                    for rec in hist:
                        fh.write(json.dumps(rec) + "\n")
            ml = Path("checkpoints") / f"expert_{name}" / "mopd_log.jsonl"
            if ml.exists():
                shutil.copy(ml, d / "metrics.jsonl")
        gpu_h = r["seconds"] / 3600
        meta = {"run_id": run_id, "host": r["host"], "gpu": h["gpu"], "gpu_count": 1, "start": r["start"], "end": r["end"],
                "seconds": r["seconds"], "exit_status": r["status"], "attempts": len(attempts),
                "raw_log": str(raw) if raw else None, "seed": 42,
                "configs": ["configs/training.yaml", "configs/merging.yaml", "configs/stage2.yaml", "configs/eval/benchmarks.yaml"]}
        (d / "metadata.json").write_text(json.dumps(meta, indent=2))
        res_rows.append({"run_id": run_id, "host": r["host"], "gpu": h["gpu"], "gpu_count": 1, "seconds": r["seconds"],
                         "gpu_hours": round(gpu_h, 3), "peak_mem_gb": round(peak, 1) if peak else "",
                         "est_cost_usd": round(gpu_h * h["rate"], 2), "exit_status": r["status"], "attempts": len(attempts)})
    with open(out / "resources.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res_rows[0].keys()))
        w.writeheader(); w.writerows(res_rows)
    tot = {h: sum(r["gpu_hours"] for r in res_rows if r["host"] == h) for h in hosts}
    print(f"{len(res_rows)} runs -> {out}; GPU-hours by host (runs overlap in wall time): {tot}")


if __name__ == "__main__":
    main()
