#!/usr/bin/env bash
# Phase 2 on the x86 host, against the running multi-LoRA server (port 30000):
#   - SWE-bench Pro subset: start only when the DGX-built SIFs are all in the cache
#     (GHCR is ~0.25 MB/s from this host), then oracle validation + base / expert_swe /
#     merged_linear / mopd runs concurrently
#   - MOPD student: hot-load the adapter when it arrives, then TB2 (images cached) + SWE
set -uo pipefail
source /root/gpufree-data/tb2/env.sh
export PATH="/root/gpufree-data/tb2/bin:/root/gpufree-data/tb2/sglangenv/bin:$PWD/.venv-harness/bin:$PATH"
export PYTHON=python MODEL_BASE_URL=http://127.0.0.1:30000/v1
SIF_CACHE=/root/gpufree-data/sif-cache-swe
export SWE_ENV_ARGS="-e singularity --ek singularity_image_cache_dir=$SIF_CACHE"
export SWE_IDS=data/eval/swe_subset_ids.txt SWE_CONCURRENCY="${SWE_CONCURRENCY:-4}" TB_CONCURRENCY="${TB_CONCURRENCY:-6}"
export SWE_TASKS_DIR=data/eval/swe_tasks_singularity
SBP=third_party/SWE-bench_Pro-os

# patched task copies (agent phase network "public": singularity cannot enforce no-network)
rm -rf "$SWE_TASKS_DIR"; mkdir -p "$SWE_TASKS_DIR"
while read -r id; do
  [ -n "$id" ] || continue
  cp -r "$SBP/v2/tasks/$id" "$SWE_TASKS_DIR/"
  python - "$SWE_TASKS_DIR/$id/task.toml" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r'(\[agent\][^\[]*?network_mode\s*=\s*)"no-network"', r'\1"public"', s, flags=re.S)
open(p, "w").write(s)
PY
done < "$SWE_IDS"

run() {  # model bench served_name
  local M=$1 B=$2 S=$3
  [ -f "results/$M/$B/results.json" ] && { echo "skip $M/$B"; return; }
  local SCRIPT=scripts/bench_$B.sh; [ "$B" = terminal ] && SCRIPT=scripts/bench_terminal_x86.sh
  SERVED_MODEL="$S" bash scripts/timed_run.sh "${B}_$M" bash "$SCRIPT" "$M" > "logs/eval_${M}_$B.log" 2>&1 \
    && echo "$(date -Is) $M/$B done" || echo "$(date -Is) $M/$B FAILED"
}

mopd_ready() { [ -f adapters/expert_mopd/READY ]; }
( until mopd_ready; do sleep 60; done
  curl -sf -X POST "http://127.0.0.1:30000/load_lora_adapter" -H "Content-Type: application/json" \
    -d '{"lora_name": "mopd", "lora_path": "adapters/expert_mopd"}' && echo "$(date -Is) mopd adapter loaded"
  run mopd terminal base:mopd ) &

until [ -f "$SIF_CACHE/READY" ]; do sleep 60; done
echo "$(date -Is) SWE SIFs ready: $(ls $SIF_CACHE/*.sif | wc -l)"
( IDS=$(sed 's/^/-i /' $SWE_IDS)
  # shellcheck disable=SC2086
  harbor run -p "$SWE_TASKS_DIR" $IDS -a oracle $SWE_ENV_ARGS -n 4 -o results/_validation/swe --job-name oracle -q -y \
    > logs/validate_swe.log 2>&1
  python -m src.evaluation.harbor_results --job_dir results/_validation/swe/oracle --benchmark swe_oracle \
    --output results/_validation/swe > /dev/null && echo "$(date -Is) SWE oracle validation done" ) &
run base swe base &
run expert_swe swe base:expert_swe &
run merged_linear swe base:merged_linear &
( until mopd_ready; do sleep 60; done; sleep 120; run mopd swe base:mopd ) &
wait
echo "$(date -Is) x86 phase 2 complete"
