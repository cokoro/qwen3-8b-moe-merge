#!/usr/bin/env bash
# SWE-bench Pro + Terminal-Bench 2.0 on an x86 host without Docker (Apptainer/Singularity).
# Both ship linux/amd64-only images, so they cannot be graded on the aarch64 DGX.
# Run from the project root on that host. Steps:
#   1. wait until no other Terminal-Bench harbor job is running on the host
#   2. stop that job's sglang server (frees the GPU)
#   3. pick a seeded HARD-51 subset that fits the disk budget, validate it with the
#      oracle (gold patch) and keep only tasks the gold patch resolves
#   4. per PLAN item "model:bench,bench": build weights on CPU (merged / folded expert)
#      -> serve -> run its benchmarks concurrently -> stop -> delete the built weights
#      (disk holds one extra model at a time). Base TB2 comes from the earlier run on
#      this host (job qwen3-8b-base-main), so base only needs SWE.
set -uo pipefail
export PATH="${SGLANG_BIN:-/root/gpufree-data/tb2/sglangenv/bin}:$PWD/.venv-harness/bin:$PATH"   # python = sglang env (torch); harbor = .venv-harness
export LITELLM_LOCAL_MODEL_COST_MAP=True
SIF_CACHE="${SIF_CACHE:-/root/gpufree-data/sif-cache-swe}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/apptainer-swe}"
mkdir -p "$SIF_CACHE" "$APPTAINER_TMPDIR" logs data/eval
export SWE_ENV_ARGS="-e singularity --ek singularity_image_cache_dir=$SIF_CACHE"
export SWE_CONCURRENCY="${SWE_CONCURRENCY:-8}" PYTHON="${PYTHON:-python}"
PLAN="${PLAN:-merged_ties:swe,terminal distilled:swe,terminal base:swe expert_swe:swe expert_terminal:terminal merged_linear:swe,terminal}"
export TB_CONCURRENCY="${TB_CONCURRENCY:-10}"
PORT="${PORT:-30000}"
export MODEL_BASE_URL="http://127.0.0.1:$PORT/v1"
SBP=third_party/SWE-bench_Pro-os

# 1-2. wait for the other Terminal-Bench job on this host: start when its base job is
# marked finished, or after IDLE_MIN minutes without any TB2 harbor process.
# The base run was interrupted by a host migration and resumed as qwen3-8b-base-r2-main
# (remaining tasks); together they cover the 74 tasks.
BASE_JOBS="${BASE_JOBS:-/root/gpufree-data/tb2/jobs/qwen3-8b-base-main /root/gpufree-data/tb2/jobs/qwen3-8b-base-r2-main}"
BASE_JOB="${BASE_JOBS##* }/result.json"
base_tb2_finished() { python3 -c "import json,sys; sys.exit(0 if json.load(open('$BASE_JOB')).get('finished_at') else 1)" 2>/dev/null; }
echo "$(date -Is) waiting for other harbor TB2 jobs to finish"
IDLE=0
while :; do
  if pgrep -f "harbor run -p /root/gpufree-data/datasets/tb2" >/dev/null; then IDLE=0; else IDLE=$((IDLE + 2)); fi
  if base_tb2_finished && [ "$IDLE" -ge 4 ]; then echo "base TB2 job finished"; break; fi
  if [ "$IDLE" -ge "${IDLE_MIN:-30}" ]; then echo "no TB2 activity for ${IDLE_MIN:-30} min"; break; fi
  sleep 120
done
for pid in $(pgrep -f "sglang.launch_server"); do echo "stopping sglang $pid"; kill "$pid"; done
sleep 20
if ! base_tb2_finished; then
  echo "base TB2 job unfinished -> base also runs TB2 here (same settings)"
  PLAN=$(echo "$PLAN" | sed 's/\bbase:swe\b/base:swe,terminal/')
fi
echo "PLAN: $PLAN"

# 3. subset + oracle validation
if [ ! -f data/eval/swe_valid_ids.txt ]; then
  [ -f data/eval/swe_subset_ids.txt ] || python3 -m src.evaluation.swe_subset --ids "$SBP/v2/hard51_ids.txt" \
      --budget_gb "${SWE_BUDGET_GB:-35}" --out data/eval/swe_subset_ids.txt
  IDS=$(sed 's/^/-i /' data/eval/swe_subset_ids.txt)
  # shellcheck disable=SC2086
  harbor run -p "$SBP/v2/tasks" $IDS -a oracle $SWE_ENV_ARGS -n "$SWE_CONCURRENCY" \
    -o results/_validation/swe --job-name oracle -q -y > logs/validate_swe.log 2>&1
  python3 - <<'PY'
import glob, json
ok, bad = [], {}
for f in glob.glob("results/_validation/swe/oracle/*/result.json"):
    d = json.load(open(f))
    r = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    exc = (d.get("exception_info") or {}).get("exception_type") if d.get("exception_info") else None
    full = d["task_name"].split("/")[-1]
    (ok.append(full) if r is not None and float(r) >= 1.0 else bad.__setitem__(full, exc or f"reward={r}"))
open("data/eval/swe_valid_ids.txt", "w").write("\n".join(sorted(ok)) + "\n")
json.dump({"valid": sorted(ok), "excluded": bad}, open("data/eval/swe_validation.json", "w"), indent=2)
print(f"SWE oracle: {len(ok)} valid, {len(bad)} excluded")
PY
fi
export SWE_IDS=data/eval/swe_valid_ids.txt
echo "$(date -Is) $(wc -l < $SWE_IDS) validated SWE tasks"

build_model() {  # $1 = model name -> prints weights dir
  case "$1" in
    base) echo models/Qwen3-8B ;;
    expert_swe) MERGE_DEVICE=cpu python -m src.merging.fold_expert --adapter adapters/expert_swe \
                  --output built/expert_swe >&2 && echo built/expert_swe ;;
    merged_linear) MERGE_DEVICE=cpu python -m src.merging.merge_linear --config configs/merging_x86.yaml \
                  --output built/merged_linear >&2 && echo built/merged_linear ;;
    distilled) [ -d adapters/expert_distill ] || { echo "distilled adapter not here yet" >&2; return 1; }
               MERGE_DEVICE=cpu python -m src.merging.fold_expert --adapter adapters/expert_distill \
                  --output built/distilled >&2 && echo built/distilled ;;
    merged_ties) MERGE_DEVICE=cpu python -m src.merging.merge_ties --config configs/merging_x86.yaml \
                  --density 0.2 --output built/merged_ties >&2 && echo built/merged_ties ;;
  esac
}

# base TB2 from the earlier run on this host (same settings as bench_terminal_x86.sh)
if [ ! -f results/base/terminal/results.json ] && base_tb2_finished; then
  # shellcheck disable=SC2086
  python -m src.evaluation.harbor_results --job_dir $BASE_JOBS \
    --benchmark terminal_bench_2.0 --metric pass_rate --output results/base/terminal
fi

for ITEM in $PLAN; do
  M="${ITEM%%:*}"
  TODO=""
  for B in $(echo "${ITEM#*:}" | tr ',' ' '); do
    [ -f "results/$M/$B/results.json" ] && echo "skip $M/$B (done)" || TODO="$TODO $B"
  done
  [ -z "$TODO" ] && continue
  echo "=== $(date -Is) $M ->$TODO"
  DIR=$(build_model "$M") || { echo "build $M failed"; continue; }
  MEM_FRACTION=0.85 bash scripts/serve_model.sh "$DIR" "$M" "$PORT" > "logs/serve_$M.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 240); do curl -sf "$MODEL_BASE_URL/models" | grep -q "\"$M\"" && break; sleep 5; done
  PIDS=""
  for B in $TODO; do
    SCRIPT=scripts/bench_$B.sh; [ "$B" = terminal ] && SCRIPT=scripts/bench_terminal_x86.sh
    bash scripts/timed_run.sh "${B}_$M" bash "$SCRIPT" "$M" > "logs/eval_${M}_$B.log" 2>&1 &
    PIDS="$PIDS $!:$B"
  done
  for P in $PIDS; do
    wait "${P%%:*}" && echo "$(date -Is) $M/${P#*:} done" || echo "$(date -Is) $M/${P#*:} FAILED (logs/eval_${M}_${P#*:}.log)"
  done
  kill "$SERVER"; wait "$SERVER" 2>/dev/null; sleep 15
  [ "$DIR" != models/Qwen3-8B ] && rm -rf "$DIR"
done
echo "$(date -Is) x86 plan complete"
