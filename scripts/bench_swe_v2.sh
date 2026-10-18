#!/usr/bin/env bash
# v2 rerun (fixed patch capture). SWE-bench Pro V2 HARD-51 under the repo's locked protocol: host-side mini-swe-agent
# (sandbox has no network), then the authoritative re-grade of each captured diff on a
# pristine image (patch_replay). Images are linux/amd64 only: run on an x86 host
# (qemu on the Spark breaks the verifier). SWE_ENV_ARGS selects the sandbox backend,
# e.g. "-e singularity --ek singularity_image_cache_dir=/path" on hosts without Docker.
set -euo pipefail
M="$1"
R="${RESULTS_DIR:-results/$M}"
SERVED="${SERVED_MODEL:-$M}"   # e.g. base:expert_swe when served as a LoRA on the base
SBP=third_party/SWE-bench_Pro-os
JOBS="$R/swe/jobs"
IDS=$(sed 's/^/-i /' "${SWE_IDS:-$SBP/v2/hard51_ids.txt}")
# v2: fixed host_mini_swe (patch saved on agent errors + trailing newline) shadows the original
export PYTHONPATH="$SBP/v2/tooling_v2:$SBP/v2/tooling" OPENAI_API_BASE="$MODEL_BASE_URL" OPENAI_API_KEY=EMPTY \
  MSWEA_COST_TRACKING=ignore_errors
# shellcheck disable=SC2086
"${HARBOR:-.venv-harness/bin/harbor}" run -p "${SWE_TASKS_DIR:-$SBP/v2/tasks}" $IDS -a host_mini_swe:HostMiniSweAgent \
  -m "openai/$SERVED" --ak "config_file=$SBP/v2/tooling/configs/mini_toolcall.yaml" \
  --ak "step_limit=${SWE_STEP_LIMIT:-100}" -n "${SWE_CONCURRENCY:-4}" \
  --timeout-multiplier "${SWE_TIMEOUT_MULT:-1}" ${SWE_ENV_ARGS:-} -o "$JOBS" --job-name sbp-hard51 -q -y
# shellcheck disable=SC2086
"${HARBOR:-.venv-harness/bin/harbor}" run -p "${SWE_TASKS_DIR:-$SBP/v2/tasks}" $IDS -a patch_replay:PatchReplayAgent -m replay \
  --ak "source_job=$JOBS/sbp-hard51" -n "${SWE_CONCURRENCY:-4}" \
  --timeout-multiplier "${SWE_TIMEOUT_MULT:-1}" ${SWE_ENV_ARGS:-} -o "$JOBS" --job-name sbp-hard51-regrade -q -y
"${PYTHON:-python}" -m src.evaluation.harbor_results --job_dir "$JOBS/sbp-hard51-regrade" \
  --benchmark swe_bench_pro_v2_hard51 --metric resolved_rate --output "$R/swe"
