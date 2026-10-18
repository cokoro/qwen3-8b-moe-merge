#!/bin/bash
# Terminal-Bench 2.0 (89 tasks, harborframework/terminal-bench-2.0@f2e8c75) with terminus-2 + the local model server.
# Agent settings mirror scripts/bench_terminal.sh (the DGX pipeline) so numbers are comparable across machines;
# only the sandbox differs (-e singularity, see README).
# usage: bash run_tb2.sh <job-name> [extra harbor args...]   e.g. -i fix-git  -n 8
#
# Optional overrides (defaults = the original runs):
#   TB_CONTEXT=40960        server context; max_input_tokens = TB_CONTEXT - 4096 (default 32768 -> 28672)
#   TB_TOP_P=0.8 TB_PRESENCE_PENALTY=1.5 TB_TOP_K=20
#                           Qwen3 non-thinking sampling recommendation (presence_penalty against endless repetition);
#                           unset = server defaults (generation_config.json: top_p 0.95, top_k 20, no presence penalty)
set -e
source /root/gpufree-data/tb2/env.sh
JOB=${1:?job name}; shift
MODEL_NAME=${MODEL_NAME:-Qwen3-8B}
TB_CONTEXT=${TB_CONTEXT:-32768}
MAX_INPUT=$((TB_CONTEXT - 4096))
CALL_KW=$(python3 - <<EOF
import json
kw = {}
if "${TB_TOP_P:-}": kw["top_p"] = float("${TB_TOP_P:-0}")
if "${TB_PRESENCE_PENALTY:-}": kw["presence_penalty"] = float("${TB_PRESENCE_PENALTY:-0}")
if "${TB_TOP_K:-}": kw["extra_body"] = {"top_k": int("${TB_TOP_K:-0}")}
print(json.dumps(kw) if kw else "")
EOF
)
EXTRA=()
[ -n "$CALL_KW" ] && EXTRA+=(--ak "llm_call_kwargs=$CALL_KW")
cd /root/gpufree-data/tb2
OPENAI_API_KEY=EMPTY harbor run \
  -p "$TB2_DATASET" \
  -a terminus-2 \
  -m "openai/$MODEL_NAME" \
  --ak api_base=http://127.0.0.1:30000/v1 \
  --ak "temperature=${TB_TEMPERATURE:-0.7}" \
  --ak "model_info={\"max_input_tokens\": $MAX_INPUT, \"max_output_tokens\": 4096, \"input_cost_per_token\": 0, \"output_cost_per_token\": 0}" \
  "${EXTRA[@]}" \
  -e singularity \
  --ek singularity_image_cache_dir="$SIF_CACHE" \
  -o /root/gpufree-data/tb2/jobs \
  --job-name "$JOB" \
  -y "$@"
