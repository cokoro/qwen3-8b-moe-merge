#!/usr/bin/env bash
# Judge the repeat-run HLE predictions with the same local judge once generation and the main judge loop are done.
cd ~/moe-merge; source ~/miniconda3/etc/profile.d/conda.sh; conda activate moe-merge
export HLE_JUDGE_BASE_URL=http://127.0.0.1:30000/v1 HLE_JUDGE_MODEL=judge-qwen3-8b HLE_JUDGE_API_KEY_ENV=JUDGE_KEY JUDGE_KEY=EMPTY \
       HLE_JUDGE_MIN_INTERVAL=0 HLE_JUDGE_WORKERS=48 HLE_JUDGE_RETRIES=2 HLE_JUDGE_PASSES=1 HLE_JUDGE_TIMEOUT=90
until grep -q "all HLE judged" logs/judge_loop.log; do sleep 30; done
for M in base merged_linear mopd; do
  d=results_rep2/$M/hle
  until [ -f $d/hle_$M.json ] || grep -q "rep2 $M FAILED" logs/hle_rep2.log; do sleep 30; done
  [ -f $d/hle_$M.json ] && [ ! -f $d/results.json ] || continue
  echo "$(date -Is) judging rep2 $M"
  HLE_SKIP_JUDGE=0 python -m src.evaluation.hle_official --model $M --output $d --judge_only \
    --judge judge-qwen3-8b --judge_base_url $HLE_JUDGE_BASE_URL > logs/judge_hle_rep2_$M.log 2>&1 && echo "$(date -Is) rep2 $M judged" || echo "$(date -Is) rep2 $M judge FAILED"
done
echo "$(date -Is) rep2 all judged"
