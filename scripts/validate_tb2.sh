#!/usr/bin/env bash
# Run Terminal-Bench 2.0 reference solutions (oracle) on this machine and keep the tasks
# that pass -> data/eval/tb2_valid_tasks.txt. Tasks that fail here would score 0 for
# every model for environment reasons (e.g. arm64 vs x86), so they are excluded and listed.
set -uo pipefail
JOBS=results/_validation/tb2
.venv-harness/bin/harbor run -p third_party/terminal-bench -a oracle \
  -n "${TB_ORACLE_CONCURRENCY:-6}" -o "$JOBS" --job-name oracle -q -y
python - <<'PY'
import glob, json
ok, bad = [], {}
for f in glob.glob("results/_validation/tb2/oracle/*/result.json"):
    d = json.load(open(f))
    r = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    exc = (d.get("exception_info") or {}).get("exception_type") if d.get("exception_info") else None
    (ok.append(d["task_name"]) if r is not None and float(r) >= 1.0 else bad.__setitem__(d["task_name"], exc or f"reward={r}"))
open("data/eval/tb2_valid_tasks.txt", "w").write("\n".join(sorted(ok)) + "\n")
json.dump({"valid": sorted(ok), "excluded": bad}, open("data/eval/tb2_validation.json", "w"), indent=2)
print(f"TB2 oracle: {len(ok)} valid, {len(bad)} excluded")
PY
