#!/usr/bin/env bash
# Terminal-Bench 2.0 on the x86 host (Apptainer/Singularity, no Docker). Mirrors the
# settings of the base-model run already done on this host (job qwen3-8b-base-main):
# same dataset mirror, terminus-2, temperature 0.7, same model_info, and the same 15
# network-dependent tasks excluded -> 74 tasks.
set -euo pipefail
M="$1"
R="${RESULTS_DIR:-results/$M}"
SERVED="${SERVED_MODEL:-$M}"   # e.g. base:expert_swe when served as a LoRA on the base
JOBS="$R/terminal/jobs"
TB2_DATASET="${TB2_DATASET:-/root/gpufree-data/datasets/tb2-mirror}"
TB2_SIF_CACHE="${TB2_SIF_CACHE:-/root/gpufree-data/sif-cache}"
EXCLUDE=""
for t in configure-git-webserver nginx-request-logging pypi-server install-windows-3.11 git-multibranch \
         qemu-alpine-ssh torch-pipeline-parallelism headless-terminal kv-store-grpc qemu-startup \
         torch-tensor-parallelism hf-model-inference mailman make-doom-for-mips sanitize-git-repo; do
  EXCLUDE="$EXCLUDE -x $t"
done
# shellcheck disable=SC2086
OPENAI_API_KEY=EMPTY "${HARBOR:-.venv-harness/bin/harbor}" run -p "$TB2_DATASET" \
  -a terminus-2 -m "openai/$SERVED" --ak "api_base=$MODEL_BASE_URL" --ak "temperature=0.7" \
  --ak 'model_info={"max_input_tokens": 28672, "max_output_tokens": 4096, "input_cost_per_token": 0, "output_cost_per_token": 0}' \
  -e singularity --ek "singularity_image_cache_dir=$TB2_SIF_CACHE" \
  -o "$JOBS" --job-name tb2 -y -q $EXCLUDE -n "${TB_CONCURRENCY:-12}"
"${PYTHON:-python}" -m src.evaluation.harbor_results --job_dir "$JOBS/tb2" \
  --benchmark terminal_bench_2.0 --metric pass_rate --output "$R/terminal"
