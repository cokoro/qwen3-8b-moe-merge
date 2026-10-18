#!/usr/bin/env bash
# Multi-teacher distillation (alternative to weight merging), DGX lane B after its evals:
#   1. each expert generates answers for prompts of its own domain (sglang, port 30001)
#   2. keep only answers that pass the domain check (src/distill/generate.py)
#   3. balanced mixture -> train one student LoRA on the same Qwen3-8B base
#   4. evaluate the student ("distilled") on HLE + ASI; TB2/SWE run on the x86 host
set -uo pipefail
PORT="${PORT:-30001}"
export MODEL_BASE_URL="http://127.0.0.1:$PORT/v1"
export MEM_FRACTION="${MEM_FRACTION:-0.4}"
mkdir -p logs data/distill

until grep -q "lane :$PORT complete" logs/lane_b.log 2>/dev/null; do sleep 60; done
echo "$(date -Is) lane B free, generating distillation data"

declare -A N=([hle]=1500 [asi]=1500 [terminal]=1500 [swe]=600)
declare -A MAXTOK=([hle]=1024 [asi]=512 [terminal]=512 [swe]=3072)
for D in hle asi terminal swe; do
  [ -f "data/distill/$D.jsonl" ] && { echo "skip $D (exists)"; continue; }
  T="expert_$D"
  bash scripts/serve_model.sh "checkpoints/full/$T" "$T" "$PORT" > "logs/serve_distill_$T.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 360); do curl -sf "$MODEL_BASE_URL/models" | grep -q "\"$T\"" && break; sleep 5; done
  bash scripts/timed_run.sh "distill_gen_$D" python -m src.distill.generate --domain "$D" --teacher "$T" \
    --base_url "$MODEL_BASE_URL" --n_prompts "${N[$D]}" --max_tokens "${MAXTOK[$D]}" \
    --out "data/distill/$D.jsonl" > "logs/distill_gen_$D.log" 2>&1 || echo "generation $D FAILED"
  tail -1 "logs/distill_gen_$D.log"
  kill "$SERVER"; wait "$SERVER" 2>/dev/null; sleep 15
done

python -m src.distill.build_mixture --per_domain "${PER_DOMAIN:-1200}" --out data/distill
[ -f checkpoints/expert_distill/training_metrics.json ] || \
  bash scripts/timed_run.sh train_distill python -m src.training.train_expert --expert distill \
    > logs/train_distill.log 2>&1 || { echo "student training FAILED"; exit 1; }
echo "$(date -Is) student trained"

PLAN="distilled:hle,asi" PORT="$PORT" bash scripts/evaluate_all.sh > logs/lane_distill.log 2>&1
echo "$(date -Is) distill plan done"
