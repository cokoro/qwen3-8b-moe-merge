"""Per-task diagnostics for a terminus-2 TB2 run (latest finished trial per task), to tell model failures from
agent-config or pipeline failures.

    python3 tb2_diagnose.py --jobs jobs/qwen3-8b-base-* --output results/qwen3-8b-base/diagnostics.jsonl

Per task: turns, how the episode ended, JSON-parse errors, repeated command batches / repeated reasoning,
prompt-token growth per turn, context summarizations, and wall-time split between LLM calls and command waits.
"""

import argparse
import difflib
import json
import tomllib
from datetime import datetime
from pathlib import Path

DATASET = Path("/root/gpufree-data/datasets/terminal-bench-2.0")


def ts(s):
    return datetime.fromisoformat(s)


def latest_trials(job_globs):
    best = {}
    for j in job_globs:
        for res in Path(j).glob("*/result.json"):
            d = json.loads(res.read_text())
            if d.get("finished_at") and (d["task_name"] not in best or d["finished_at"] > best[d["task_name"]][0]["finished_at"]):
                best[d["task_name"]] = (d, res.parent)
    return best


def analyse(d, trial_dir):
    task = d["task_name"]
    cfg = tomllib.loads((DATASET / task / "task.toml").read_text())
    traj = json.loads((trial_dir / "agent" / "trajectory.json").read_text())
    steps = traj["steps"]
    agent = [s for s in steps if s.get("source") == "agent"]
    batches, msgs, prompt_tok, parse_err, waits, gaps = [], [], [], 0, [], []
    complete_calls = 0
    for i, s in enumerate(agent):
        calls = s.get("tool_calls") or []
        keys = tuple((c.get("arguments") or {}).get("keystrokes", "") for c in calls if c.get("function_name") == "bash_command")
        complete_calls += sum(c.get("function_name") == "mark_task_complete" for c in calls)
        batches.append(keys)
        msgs.append((s.get("message") or "").strip())
        prompt_tok.append((s.get("metrics") or {}).get("prompt_tokens") or 0)
        obs = " ".join(r.get("content") or "" for r in (s.get("observation") or {}).get("results", []))
        parse_err += "parsing errors" in obs or "No valid JSON" in obs
        waits.append(sum(float((c.get("arguments") or {}).get("duration") or 0) for c in calls))
        if i:
            gaps.append((ts(s["timestamp"]) - ts(agent[i - 1]["timestamp"])).total_seconds())
    n = len(agent)
    nonempty = [b for b in batches if any(k.strip() for k in b)]
    dup_batches = sum(1 for i, b in enumerate(nonempty) if b in nonempty[:i])
    run = best_run = 1 if nonempty else 0
    for a, b in zip(nonempty, nonempty[1:]):
        run = run + 1 if a == b else 1
        best_run = max(best_run, run)
    near_dup_msgs = sum(
        1 for i in range(1, n)
        if msgs[i] and max(difflib.SequenceMatcher(None, msgs[i][:600], m[:600]).quick_ratio() for m in msgs[max(0, i - 5):i]) > 0.9
    )
    summaries = sum(1 for s in steps if s.get("source") == "system" and "summarization" in (s.get("message") or ""))
    exc = (d.get("exception_info") or {}).get("exception_type")
    wall = (ts(d["agent_execution"]["finished_at"]) - ts(d["agent_execution"]["started_at"])).total_seconds() if d.get("agent_execution") else None
    if exc == "AgentTimeoutError":
        end = "agent_timeout"
    elif complete_calls:
        end = "declared_complete"
    elif exc:
        end = exc
    else:
        end = "other"
    return {
        "task": task,
        "difficulty": cfg.get("metadata", {}).get("difficulty"),
        "category": cfg.get("metadata", {}).get("category"),
        "reward": ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
        "exception": exc,
        "end": end,
        "turns": n,
        "agent_timeout_s": cfg["agent"]["timeout_sec"],
        "agent_wall_s": round(wall) if wall else None,
        "parse_error_turns": parse_err,
        "mark_task_complete_calls": complete_calls,
        "empty_command_turns": n - len(nonempty),
        "duplicate_command_batches": dup_batches,
        "max_consecutive_identical_batches": best_run,
        "near_duplicate_reasoning_turns": near_dup_msgs,
        "context_summarizations": summaries,
        "prompt_tokens_first": prompt_tok[0] if prompt_tok else None,
        "prompt_tokens_max": max(prompt_tok) if prompt_tok else None,
        "prompt_tokens_last": prompt_tok[-1] if prompt_tok else None,
        "mean_turn_gap_s": round(sum(gaps) / len(gaps), 1) if gaps else None,
        "mean_requested_wait_s": round(sum(waits) / n, 1) if n else None,
        "first_analysis": msgs[0][:400] if msgs else "",
        "last_analysis": msgs[-1][:400] if msgs else "",
        "commands": [list(b) for b in batches[:3]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = [analyse(d, p) for _, (d, p) in sorted(latest_trials(args.jobs).items())]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
