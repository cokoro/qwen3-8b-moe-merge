#!/usr/bin/env bash
# x86 side of the merged_ties TB2/SWE run: identical harness settings to phase 2/3, but the model is
# served from the DGX through the reverse tunnel at 127.0.0.1:30002 (see scripts/serve_ties_remote.sh).
# Replaces run_x86_phase3.sh (which would wait for phase 2 and then take over the L40S).
set -uo pipefail
cd /root/gpufree-data/projects/swe-eval
source /root/gpufree-data/tb2/env.sh
export PATH="/root/gpufree-data/tb2/bin:/root/gpufree-data/tb2/sglangenv/bin:$PWD/.venv-harness/bin:$PATH"
export PYTHON=python MODEL_BASE_URL=http://127.0.0.1:30002/v1
SIF_CACHE=/root/gpufree-data/sif-cache-swe
export SWE_ENV_ARGS="-e singularity --ek singularity_image_cache_dir=$SIF_CACHE"
export SWE_IDS=data/eval/swe_subset_ids.txt SWE_TASKS_DIR=data/eval/swe_tasks_singularity
export SWE_CONCURRENCY=4 TB_CONCURRENCY="${TB_CONCURRENCY:-12}"

for p in $(ps -ww -eo pid,args | awk '$2 == "bash" && $3 == "scripts/run_x86_phase3.sh" {print $1}'); do kill "$p"; done
until curl -sf "$MODEL_BASE_URL/models" | grep -q merged_ties; do sleep 20; done
echo "$(date -Is) merged_ties reachable through tunnel"
for B in terminal swe; do
  SCRIPT=scripts/bench_$B.sh; [ "$B" = terminal ] && SCRIPT=scripts/bench_terminal_x86.sh
  ( bash scripts/timed_run.sh "${B}_merged_ties" bash "$SCRIPT" merged_ties > "logs/eval_merged_ties_$B.log" 2>&1 \
      && echo "$(date -Is) merged_ties/$B done" || echo "$(date -Is) merged_ties/$B FAILED" ) &
done
wait
echo "$(date -Is) ties remote complete"
