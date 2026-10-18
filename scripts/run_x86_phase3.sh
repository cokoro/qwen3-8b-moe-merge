#!/usr/bin/env bash
# Phase 3 on the x86 host: merged_ties is not expressible as a low-rank LoRA, so it gets its
# own bf16 server after phase 2 frees the GPU. Weights are rebuilt on CPU from the 4 expert
# adapters (identical recipe to the DGX checkpoint), then TB2 + SWE run concurrently.
set -uo pipefail
source /root/gpufree-data/tb2/env.sh
export PATH="/root/gpufree-data/tb2/bin:/root/gpufree-data/tb2/sglangenv/bin:$PWD/.venv-harness/bin:$PATH"
export PYTHON=python MODEL_BASE_URL=http://127.0.0.1:30000/v1
SIF_CACHE=/root/gpufree-data/sif-cache-swe
export SWE_ENV_ARGS="-e singularity --ek singularity_image_cache_dir=$SIF_CACHE"
export SWE_IDS=data/eval/swe_subset_ids.txt SWE_TASKS_DIR=data/eval/swe_tasks_singularity
export SWE_CONCURRENCY=4 TB_CONCURRENCY="${TB_CONCURRENCY:-12}"

until grep -q "x86 phase 2 complete" logs/run_x86_phase2.log 2>/dev/null; do sleep 60; done
echo "$(date -Is) phase 2 done; switching server to merged_ties"
for pid in $(pgrep -f "sglang.launch_server"); do kill "$pid"; done
[ -f built/merged_ties/merge_manifest.json ] || MERGE_DEVICE=cpu python -m src.merging.merge_ties \
  --config configs/merging_x86.yaml --density 0.2 --output built/merged_ties > logs/build_merged_ties.log 2>&1
sleep 15
python -m sglang.launch_server --model-path built/merged_ties --served-model-name merged_ties \
  --host 0.0.0.0 --port 30000 --context-length 32768 --mem-fraction-static 0.85 \
  --tool-call-parser qwen25 --chat-template configs/chat_templates/qwen3_nothink.jinja --random-seed 42 \
  > logs/serve_merged_ties.log 2>&1 &
SERVER=$!
for _ in $(seq 1 240); do curl -sf "$MODEL_BASE_URL/models" | grep -q merged_ties && break; sleep 5; done
echo "$(date -Is) merged_ties server up"
for B in terminal swe; do
  SCRIPT=scripts/bench_$B.sh; [ "$B" = terminal ] && SCRIPT=scripts/bench_terminal_x86.sh
  ( bash scripts/timed_run.sh "${B}_merged_ties" bash "$SCRIPT" merged_ties > "logs/eval_merged_ties_$B.log" 2>&1 \
      && echo "$(date -Is) merged_ties/$B done" || echo "$(date -Is) merged_ties/$B FAILED" ) &
done
wait
kill "$SERVER"
echo "$(date -Is) x86 phase 3 complete"
