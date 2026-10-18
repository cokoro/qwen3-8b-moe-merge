#!/usr/bin/env bash
# Terminal-Bench 2.0 (89 tasks) with Harbor's terminus-2 agent; the LLM loop runs on
# the host and talks to the local server. TB_ARGS can add filters, e.g. "-l 10".
set -euo pipefail
M="$1"
R="${RESULTS_DIR:-results/$M}"
JOBS="$R/terminal/jobs"
# Only tasks whose oracle solution passes on this machine (see scripts/validate_tb2.sh)
TASK_FILTER=""
if [ -f "${TB_TASKS_FILE:-data/eval/tb2_valid_tasks.txt}" ]; then
  TASK_FILTER=$(sed 's/^/-i /' "${TB_TASKS_FILE:-data/eval/tb2_valid_tasks.txt}")
fi
OPENAI_API_KEY=EMPTY .venv-harness/bin/harbor run -p third_party/terminal-bench \
  -a terminus-2 -m "openai/$M" --ak "api_base=$MODEL_BASE_URL" \
  --ak "temperature=${TB_TEMPERATURE:-0.7}" \
  --ak 'model_info={"max_input_tokens": 28672, "max_output_tokens": 4096, "input_cost_per_token": 0, "output_cost_per_token": 0}' \
  -n "${TB_CONCURRENCY:-3}" -o "$JOBS" --job-name tb2 -q -y $TASK_FILTER ${TB_ARGS:-}
python -m src.evaluation.harbor_results --job_dir "$JOBS/tb2" \
  --benchmark terminal_bench_2.0 --metric pass_rate --output "$R/terminal"
