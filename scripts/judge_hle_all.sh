#!/usr/bin/env bash
# Judge all saved HLE predictions with a local judge (fallback when the hosted judge has
# no quota). Starts after DGX lane A frees port 30000, judges every results/*/hle that has
# predictions but no results.json, and keeps polling until the distillation plan is done.
# Judge = local Qwen3-8B (non-official; recorded in results.json).
set -uo pipefail
PORT="${JUDGE_PORT:-30000}"
JUDGE_NAME="${JUDGE_NAME:-judge-qwen3-8b}"
export HLE_JUDGE_BASE_URL="http://127.0.0.1:$PORT/v1" HLE_JUDGE_MODEL="$JUDGE_NAME" \
       HLE_JUDGE_API_KEY_ENV=JUDGE_KEY JUDGE_KEY=EMPTY HLE_JUDGE_MIN_INTERVAL=0 HLE_JUDGE_WORKERS=48 \
       HLE_JUDGE_RETRIES=2 HLE_JUDGE_PASSES=1   # local judge: a degenerate generation is a judge failure, not worth 3x5 timeouts
until grep -q "lane :$PORT complete" logs/lane_a.log 2>/dev/null; do sleep 60; done
MEM_FRACTION="${MEM_FRACTION:-0.4}" bash scripts/serve_model.sh models/Qwen3-8B "$JUDGE_NAME" "$PORT" \
  > logs/serve_judge.log 2>&1 &
SERVER=$!
for _ in $(seq 1 360); do curl -sf "$HLE_JUDGE_BASE_URL/models" | grep -q "$JUDGE_NAME" && break; sleep 5; done
echo "$(date -Is) judge up"
while :; do
  for d in results/*/hle; do
    M=$(basename "$(dirname "$d")")
    [[ $M == _* ]] && continue
    [ -f "$d/hle_$M.json" ] && [ ! -f "$d/results.json" ] || continue
    echo "$(date -Is) judging $M"
    HLE_SKIP_JUDGE=0 python -m src.evaluation.hle_official --model "$M" --output "$d" --judge_only \
      --judge "$JUDGE_NAME" --judge_base_url "$HLE_JUDGE_BASE_URL" > "logs/judge_hle_$M.log" 2>&1 \
      && echo "$(date -Is) $M judged" || echo "$(date -Is) $M judge FAILED"
  done
  grep -q "mopd plan done" logs/run_mopd_plan.log 2>/dev/null && break
  sleep 120
done
kill "$SERVER"; wait "$SERVER" 2>/dev/null
echo "$(date -Is) all HLE judged"
