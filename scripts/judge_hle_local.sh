#!/usr/bin/env bash
# Local-judge pass over every HLE prediction file that has not been judged yet (Qwen3-8B judge on :30000).
cd ~/moe-merge; source ~/miniconda3/etc/profile.d/conda.sh; conda activate moe-merge
export HLE_JUDGE_BASE_URL=http://127.0.0.1:30000/v1 HLE_JUDGE_MODEL=judge-qwen3-8b HLE_JUDGE_API_KEY_ENV=JUDGE_KEY JUDGE_KEY=EMPTY \
       HLE_JUDGE_MIN_INTERVAL=0 HLE_JUDGE_WORKERS=48 HLE_JUDGE_RETRIES=2 HLE_JUDGE_PASSES=1 HLE_JUDGE_TIMEOUT=90
for M in base merged_linear merged_ties expert_hle mopd; do
  d=results/$M/hle
  [ -f $d/hle_$M.json ] && [ ! -f $d/results.json ] || continue
  echo "$(date -Is) judging $M"
  HLE_SKIP_JUDGE=0 python -m src.evaluation.hle_official --model $M --output $d --judge_only \
    --judge judge-qwen3-8b --judge_base_url $HLE_JUDGE_BASE_URL > logs/judge_hle_$M.log 2>&1 && echo "$(date -Is) $M judged" || echo "$(date -Is) $M judge FAILED"
done
echo "$(date -Is) all HLE judged"
