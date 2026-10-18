#!/usr/bin/env bash
# Serve one full model directory behind an OpenAI-compatible API (sglang).
#   bash scripts/serve_model.sh checkpoints/merged_ties merged_ties [port]
# Every benchmark harness talks to http://127.0.0.1:$PORT/v1 with model name $NAME,
# so all models are evaluated through the identical inference stack.
set -euo pipefail
MODEL_DIR="$1"
NAME="$2"
PORT="${3:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.55}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
HOST="${HOST:-0.0.0.0}"   # 0.0.0.0 so Docker task containers can reach it via the bridge gateway
# Experts are trained in Qwen3 non-thinking format, so every model is served with a
# template whose default is enable_thinking=false (a request can still opt in).
CHAT_TEMPLATE="${CHAT_TEMPLATE:-configs/chat_templates/qwen3_nothink.jinja}"

mkdir -p logs
exec python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name "$NAME" \
  --host "$HOST" --port "$PORT" \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION" \
  --tool-call-parser qwen25 \
  --chat-template "$CHAT_TEMPLATE" \
  --random-seed 42 "${@:4}"   # extra sglang flags (e.g. YaRN override) pass through
