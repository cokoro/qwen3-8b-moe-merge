#!/usr/bin/env bash
# v2: --sandbox task = each task runs in its declared runtime environment (runtime.packages, e.g. matplotlib);
# v1 used the default --sandbox none, where generated code ran in a Python missing declared packages.
# ASI-Bench seed31415 (public references, local non-official scoring) with the
# built-in single-turn direct_llm agent. ASI_LEVELS / ASI_TASKS narrow the run.
set -euo pipefail
M="$1"
R="${RESULTS_DIR:-results/$M}"
ASI=.venv-asi/bin/asibench
# 13/60 tasks contain a VLM-judge rubric item (openai/gpt-5.5). No VLM judge is available
# here, so the judge endpoint points at a closed local port: those items fail fast and
# score 0 for every model alike (documented in the report).
export ASIBENCH_JUDGE_API_BASE="${ASIBENCH_JUDGE_API_BASE:-http://127.0.0.1:9/v1}" \
       ASIBENCH_JUDGE_API_PROTOCOL="${ASIBENCH_JUDGE_API_PROTOCOL:-openai}" \
       ASIBENCH_JUDGE_API_KEY_ENV=NO_JUDGE_KEY NO_JUDGE_KEY=none
INST=data/eval/asi_seed31415
[ -d "$INST" ] || $ASI task pull --repo seed31415 --output-dir "$INST"
RUN="$R/asi/run"
$ASI run --instances-dir "$INST" --tasks-dir third_party/ASI-Bench/tasks --tasks "${ASI_TASKS:-all}" \
  --agent direct_llm --sandbox task \
  --agent-config "{\"model\": \"${SERVED_MODEL:-$M}\", \"api_base\": \"$MODEL_BASE_URL\", \"api_protocol\": \"openai\", \"api_key\": \"EMPTY\"}" \
  --prompt-levels "${ASI_LEVELS:-b1}" --parallel "${ASI_PARALLEL:-12}" \
  --timeout "${ASI_TIMEOUT:-1800}" --seed 31415 --output-dir "$RUN"
# score task by task with a per-task timeout (a hanging scorer cannot block the run)
python -m src.evaluation.asi_score_per_task --run "$RUN" --output "$R/asi" --timeout "${ASI_SCORE_TIMEOUT:-300}"
python -m src.evaluation.asi_results --report "$R/asi/local_score_seed31415.json" --output "$R/asi"
