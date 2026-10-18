#!/usr/bin/env bash
# x86 side of the SWE v2 rerun: model served from the DGX (YaRN 64K context) through a reverse tunnel at
# 127.0.0.1:30012; harness = bench_swe_v2.sh (fixed patch capture). Same 6 tasks, agent config and step limit.
#   bash scripts/run_swe_v2_x86.sh <served_name>
set -uo pipefail
cd /root/gpufree-data/projects/swe-eval
source /root/gpufree-data/tb2/env.sh
export PATH="/root/gpufree-data/tb2/bin:/root/gpufree-data/tb2/sglangenv/bin:$PWD/.venv-harness/bin:$PATH"
M="$1"
export PYTHON=python MODEL_BASE_URL=http://127.0.0.1:30012/v1 RESULTS_DIR="results_v2/$M"
export SWE_ENV_ARGS="-e singularity --ek singularity_image_cache_dir=/root/gpufree-data/sif-cache-swe"
export SWE_IDS=data/eval/swe_subset_ids.txt SWE_TASKS_DIR=data/eval/swe_tasks_singularity SWE_CONCURRENCY=3
until curl -sf "$MODEL_BASE_URL/models" | grep -q "\"$M\""; do sleep 20; done
echo "$(date -Is) $M reachable"
bash scripts/timed_run.sh "swe_v2_$M" bash scripts/bench_swe_v2.sh "$M" > "logs/eval_v2_${M}_swe.log" 2>&1
echo "$(date -Is) swe_v2 $M done rc=$?"
