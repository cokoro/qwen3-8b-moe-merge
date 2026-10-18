#!/bin/bash
# Serve a model on 127.0.0.1:30000 through the project's own scripts/serve_model.sh (sglang, Qwen3 no-think
# template, context 32768, seed 42) so every model is evaluated through the identical inference stack.
#   bash serve_sglang.sh [model_dir] [served_name]      (defaults: models/Qwen3-8B Qwen3-8B)
cd /root/gpufree-data/projects/moe-merge
export PATH=/root/gpufree-data/tb2/sglangenv/bin:$PATH
export HOST=127.0.0.1           # containers share the host network; no need to listen on 0.0.0.0
export MEM_FRACTION=${MEM_FRACTION:-0.85}   # dedicated 48GB L40S (DGX uses 0.55 of unified memory)
exec bash scripts/serve_model.sh "${1:-models/Qwen3-8B}" "${2:-Qwen3-8B}" 30000
