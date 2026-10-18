#!/usr/bin/env bash
# expert_swe's agent-phase harbor process finished its job (result.json at 04:34) but hung in shutdown
# retries; kill it and run the remaining regrade + scoring steps of bench_swe.sh with the same settings.
cd /root/gpufree-data/projects/swe-eval
source /root/gpufree-data/tb2/env.sh
export PATH="/root/gpufree-data/tb2/bin:/root/gpufree-data/tb2/sglangenv/bin:$PWD/.venv-harness/bin:$PATH"
export PYTHON=python MODEL_BASE_URL=http://127.0.0.1:30000/v1
SIF_CACHE=/root/gpufree-data/sif-cache-swe
SWE_ENV_ARGS="-e singularity --ek singularity_image_cache_dir=$SIF_CACHE"
SBP=third_party/SWE-bench_Pro-os; R=results/expert_swe; JOBS=$R/swe/jobs
for p in $(ps -ww -eo pid,args | awk '/harbor run/ && /base:expert_swe/ && !/awk/ {print $1}'); do kill $p; done
sleep 20
for p in $(ps -ww -eo pid,args | awk '/harbor run/ && /base:expert_swe/ && !/awk/ {print $1}'); do kill -9 $p; done
IDS=$(sed 's/^/-i /' data/eval/swe_subset_ids.txt)
export PYTHONPATH="$SBP/v2/tooling" OPENAI_API_BASE="$MODEL_BASE_URL" OPENAI_API_KEY=EMPTY MSWEA_COST_TRACKING=ignore_errors
harbor run -p data/eval/swe_tasks_singularity $IDS -a patch_replay:PatchReplayAgent -m replay \
  --ak "source_job=$JOBS/sbp-hard51" -n 4 --timeout-multiplier 1 $SWE_ENV_ARGS -o "$JOBS" --job-name sbp-hard51-regrade -q -y
python -m src.evaluation.harbor_results --job_dir "$JOBS/sbp-hard51-regrade" \
  --benchmark swe_bench_pro_v2_hard51 --metric resolved_rate --output "$R/swe"
echo "$(date -Is) expert_swe swe finished manually"
