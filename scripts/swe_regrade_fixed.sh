#!/usr/bin/env bash
# Re-grade the SWE agent patches after restoring the trailing newline that the harness stripped when capturing
# `git diff` (all non-empty model.patch files end mid-line -> git apply "corrupt patch" -> partial fuzzy apply).
# Copies only; the original results are untouched. Output: /root/gpufree-data/review_swe_fix/<model>/
cd /root/gpufree-data/projects/swe-eval
source /root/gpufree-data/tb2/env.sh
export PATH="/root/gpufree-data/tb2/bin:/root/gpufree-data/tb2/sglangenv/bin:$PWD/.venv-harness/bin:$PATH"
SBP=third_party/SWE-bench_Pro-os
export PYTHONPATH="$SBP/v2/tooling" OPENAI_API_KEY=EMPTY
ENV_ARGS="-e singularity --ek singularity_image_cache_dir=/root/gpufree-data/sif-cache-swe"
IDS=$(sed 's/^/-i /' data/eval/swe_subset_ids.txt)
OUT=/root/gpufree-data/review_swe_fix
for m in base mopd; do
  mkdir -p $OUT/$m/jobs
  rm -rf $OUT/$m/jobs/sbp-hard51 $OUT/$m/jobs/regrade-fixed
  cp -r results/$m/swe/jobs/sbp-hard51 $OUT/$m/jobs/
  for p in $OUT/$m/jobs/sbp-hard51/*/agent/model.patch; do
    [ -s "$p" ] && [ "$(tail -c 1 "$p" | od -An -c | tr -d ' ')" != '\n' ] && printf '\n' >> "$p" && echo "fixed $p"
  done
  harbor run -p data/eval/swe_tasks_singularity $IDS -a patch_replay:PatchReplayAgent -m replay \
    --ak "source_job=$OUT/$m/jobs/sbp-hard51" -n 4 --timeout-multiplier 1 $ENV_ARGS -o $OUT/$m/jobs --job-name regrade-fixed -q -y
  python -m src.evaluation.harbor_results --job_dir $OUT/$m/jobs/regrade-fixed \
    --benchmark swe_bench_pro_v2_hard51 --metric resolved_rate --output $OUT/$m/swe
  echo "$(date -Is) $m regrade-fixed done"
done
