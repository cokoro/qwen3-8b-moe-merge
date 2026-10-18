#!/usr/bin/env bash
# HLE via the official centerforaisafety/hle scripts (text-only, seeded subset).
# Data: official test parquet from the public ModelScope mirror (data/eval/hle_source).
# Judge: official o3-mini when OPENAI_API_KEY is set and HLE_JUDGE_BASE_URL is unset;
# otherwise defaults to Qwen3.5-397B-A17B on ModelScope API-Inference (MODELSCOPE_API_KEY
# in .env), recorded in results.json as a non-official judge.
set -euo pipefail
M="$1"
R="${RESULTS_DIR:-results/$M}"
[ -f .env ] && { set -a; . ./.env; set +a; }
if [ -z "${HLE_JUDGE_BASE_URL:-}" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${MODELSCOPE_API_KEY:-}" ]; then
  export HLE_JUDGE_BASE_URL=https://api-inference.modelscope.cn/v1
  export HLE_JUDGE_MODEL="${HLE_JUDGE_MODEL:-Qwen/Qwen3.5-397B-A17B}"
  export HLE_JUDGE_API_KEY_ENV=MODELSCOPE_API_KEY
fi
python -m src.evaluation.hle_official --model "$M" --base_url "$MODEL_BASE_URL" \
  --output "$R/hle" --n "${HLE_N:-500}"
