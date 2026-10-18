#!/usr/bin/env bash
# Evaluate models through one inference stack: for each model -> serve it with sglang ->
# run its benchmarks *concurrently* against that server -> stop.
#
#   PLAN="base:terminal,hle,asi expert_hle:hle" PORT=30000 bash scripts/evaluate_all.sh
# Without PLAN: every model in $MODELS x every benchmark in $BENCHMARKS.
# A (model, benchmark) cell whose results.json exists is skipped, so lanes can be restarted.
set -uo pipefail

PORT="${PORT:-30000}"
export MODEL_BASE_URL="http://127.0.0.1:$PORT/v1"
MEM_FRACTION="${MEM_FRACTION:-0.55}"
if [ -z "${PLAN:-}" ]; then
  MODELS="${MODELS:-base expert_swe expert_hle expert_asi expert_terminal merged_linear merged_ties}"
  BENCHMARKS="${BENCHMARKS:-hle terminal swe asi}"
  PLAN=""
  for M in $MODELS; do PLAN="$PLAN $M:$(echo $BENCHMARKS | tr ' ' ',')"; done
fi
mkdir -p logs results

model_dir() {
  case "$1" in
    base) echo models/Qwen3-8B ;;
    expert_*) echo "checkpoints/full/$1" ;;
    distilled) echo checkpoints/full/expert_distill ;;
    mopd) echo checkpoints/full/expert_mopd ;;
    mopd14b) echo checkpoints/full/expert_mopd14b ;;
    *) echo "checkpoints/$1" ;;
  esac
}

wait_ready() {
  for _ in $(seq 1 360); do
    curl -sf "$MODEL_BASE_URL/models" | grep -q "\"$1\"" && return 0
    sleep 5
  done
  return 1
}

if curl -sf "$MODEL_BASE_URL/models" >/dev/null; then
  echo "Port $PORT already serves a model; stop it first." >&2
  exit 1
fi

for ITEM in $PLAN; do
  M="${ITEM%%:*}"
  BENCHES=$(echo "${ITEM#*:}" | tr ',' ' ')
  TODO=""
  for B in $BENCHES; do
    [ -f "results/$M/$B/results.json" ] && echo "skip $M/$B (done)" || TODO="$TODO $B"
  done
  [ -z "$TODO" ] && continue

  DIR=$(model_dir "$M")
  if [[ ( $M == expert_* || $M == distilled || $M == mopd || $M == mopd14b ) && ! -d $DIR ]]; then
    ADAPTER="checkpoints/$M"; [ "$M" = distilled ] && ADAPTER=checkpoints/expert_distill
    [ "$M" = mopd ] && ADAPTER=checkpoints/expert_mopd
    FOLD_CFG=configs/merging.yaml
    [ "$M" = mopd14b ] && { ADAPTER=checkpoints/expert_mopd14b; FOLD_CFG=configs/merging_14b.yaml; }
    python -m src.merging.fold_expert --adapter "$ADAPTER" --output "$DIR" --config "$FOLD_CFG" \
      || { echo "fold $M failed"; continue; }
  fi
  [ -d "$DIR" ] || { echo "missing $DIR, skipping $M"; continue; }

  echo "=== $(date -Is) $M ($DIR) on :$PORT ->$TODO"
  MEM_FRACTION="$MEM_FRACTION" bash scripts/serve_model.sh "$DIR" "$M" "$PORT" > "logs/serve_$M.log" 2>&1 &
  SERVER=$!
  if ! wait_ready "$M"; then
    echo "server for $M did not come up (logs/serve_$M.log)"
    kill "$SERVER" 2>/dev/null; continue
  fi
  PIDS=""
  for B in $TODO; do
    bash scripts/timed_run.sh "eval_${M}_${B}" bash "scripts/bench_$B.sh" "$M" \
      > "logs/eval_${M}_${B}.log" 2>&1 &
    PIDS="$PIDS $!:$B"
  done
  for P in $PIDS; do
    wait "${P%%:*}" && echo "$(date -Is) $M/${P#*:} done" || echo "$(date -Is) $M/${P#*:} FAILED (logs/eval_${M}_${P#*:}.log)"
  done
  kill "$SERVER"; wait "$SERVER" 2>/dev/null
  sleep 15
done
echo "$(date -Is) lane :$PORT complete"
