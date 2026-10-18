#!/usr/bin/env bash
# Re-run only the local scoring step of ASI-Bench for models whose run is complete.
#   bash scripts/rescore_asi.sh base merged_linear
set -uo pipefail
export ASIBENCH_JUDGE_API_BASE="${ASIBENCH_JUDGE_API_BASE:-http://127.0.0.1:9/v1}" ASIBENCH_JUDGE_API_PROTOCOL=openai ASIBENCH_JUDGE_API_KEY_ENV=NO_JUDGE_KEY NO_JUDGE_KEY=none
for M in "$@"; do
  .venv-asi/bin/asibench score --repo seed31415 --results-dir "results/$M/asi/run" \
    --instances-dir data/eval/asi_seed31415 --tasks-dir third_party/ASI-Bench/tasks \
    --parallel 8 --output "results/$M/asi/local_score_seed31415.json" > "logs/rescore_asi_$M.log" 2>&1 \
  && python -m src.evaluation.asi_results --report "results/$M/asi/local_score_seed31415.json" --output "results/$M/asi" \
  && echo "$M rescored" || echo "$M rescore FAILED"
done
