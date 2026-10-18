#!/usr/bin/env bash
# Final x86 evaluation (SWE-bench Pro subset + Terminal-Bench 2.0) for everything that
# still needs them, served from ONE bf16 sglang server: the Qwen3-8B base plus LoRA
# adapters (experts, merged_linear as an exact rank-64 LoRA, MOPD student). All jobs run
# concurrently against that server; requests pick the adapter via model "base:<name>".
#   base            : SWE   (TB2 already done on this host, imported)
#   expert_swe      : SWE   (own benchmark)
#   expert_terminal : TB2   (own benchmark)
#   merged_linear   : SWE + TB2
#   mopd            : SWE + TB2 (adapter arrives from the DGX when training ends)
set -uo pipefail
export PATH="/root/gpufree-data/tb2/sglangenv/bin:$PWD/.venv-harness/bin:$PATH"
# host setup from the earlier TB2 runs on this pod: singularity shim (no PID namespace in an
# unprivileged k8s pod, containers as uid 2000 in per-trial cgroups), local pypi/github mirrors,
# apptainer cache/tmp on the data volume
source /root/gpufree-data/tb2/env.sh
export PATH="/root/gpufree-data/tb2/bin:/root/gpufree-data/tb2/sglangenv/bin:$PWD/.venv-harness/bin:$PATH"
export PYTHON=python
SIF_CACHE=/root/gpufree-data/sif-cache-swe
mkdir -p "$SIF_CACHE" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" logs data/eval
export SWE_ENV_ARGS="-e singularity --ek singularity_image_cache_dir=$SIF_CACHE"
export SWE_IDS=data/eval/swe_subset_ids.txt SWE_CONCURRENCY="${SWE_CONCURRENCY:-3}" TB_CONCURRENCY="${TB_CONCURRENCY:-6}"
export MODEL_BASE_URL=http://127.0.0.1:30000/v1
SBP=third_party/SWE-bench_Pro-os
# SWE-bench Pro tasks request network_mode="no-network" for the agent phase, which the
# singularity backend cannot enforce (harbor rejects the task). Copy the subset's task dirs
# with the agent phase set to "public" -- a documented deviation (the agent loop itself runs
# on the host; only commands it executes inside the sandbox could reach the network).
export SWE_TASKS_DIR=data/eval/swe_tasks_singularity
if [ ! -d "$SWE_TASKS_DIR" ]; then
  mkdir -p "$SWE_TASKS_DIR"
  while read -r id; do
    [ -n "$id" ] || continue
    cp -r "$SBP/v2/tasks/$id" "$SWE_TASKS_DIR/"
    python - "$SWE_TASKS_DIR/$id/task.toml" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r'(\[agent\][^\[]*?network_mode\s*=\s*)"no-network"', r'\1"public"', s, flags=re.S)
open(p, "w").write(s)
PY
  done < "$SWE_IDS"
fi

for pid in $(pgrep -f "sglang.launch_server"); do kill "$pid"; done; sleep 20

# base TB2 = earlier run on this host (interrupted job + its resume cover all 74 tasks)
[ -f results/base/terminal/results.json ] || python -m src.evaluation.harbor_results \
  --job_dir /root/gpufree-data/tb2/jobs/qwen3-8b-base-main /root/gpufree-data/tb2/jobs/qwen3-8b-base-r2-main \
  --benchmark terminal_bench_2.0 --metric pass_rate --output results/base/terminal

[ -f adapters/merged_linear/adapter_model.safetensors ] || \
  python -m src.merging.linear_as_lora --config configs/merging_x86.yaml --output adapters/merged_linear

python -m sglang.launch_server --model-path models/Qwen3-8B --served-model-name base \
  --host 0.0.0.0 --port 30000 --context-length 32768 --mem-fraction-static 0.85 \
  --tool-call-parser qwen25 --chat-template configs/chat_templates/qwen3_nothink.jinja --random-seed 42 \
  --enable-lora --max-lora-rank 64 --max-loras-per-batch 6 \
  --lora-paths expert_swe=adapters/expert_swe expert_terminal=adapters/expert_terminal \
               merged_linear=adapters/merged_linear > logs/serve_lora.log 2>&1 &
SERVER=$!
for _ in $(seq 1 240); do curl -sf "$MODEL_BASE_URL/models" >/dev/null && break; sleep 5; done
echo "$(date -Is) server up"

# oracle (gold patch) validation of the SWE subset, in parallel with the model runs
if [ ! -f data/eval/swe_validation.json ]; then
  ( IDS=$(sed 's/^/-i /' $SWE_IDS)
    # shellcheck disable=SC2086
    harbor run -p "$SWE_TASKS_DIR" $IDS -a oracle $SWE_ENV_ARGS -n 4 -o results/_validation/swe --job-name oracle -q -y \
      > logs/validate_swe.log 2>&1
    python -m src.evaluation.harbor_results --job_dir results/_validation/swe/oracle --benchmark swe_oracle \
      --output results/_validation/swe > /dev/null
    cp results/_validation/swe/detailed_results.jsonl data/eval/swe_validation.json ) &
fi

run() {  # model bench served_name
  local M=$1 B=$2 S=$3
  [ -f "results/$M/$B/results.json" ] && { echo "skip $M/$B"; return; }
  local SCRIPT=scripts/bench_$B.sh; [ "$B" = terminal ] && SCRIPT=scripts/bench_terminal_x86.sh
  SERVED_MODEL="$S" bash scripts/timed_run.sh "${B}_$M" bash "$SCRIPT" "$M" > "logs/eval_${M}_$B.log" 2>&1 \
    && echo "$(date -Is) $M/$B done" || echo "$(date -Is) $M/$B FAILED"
}
run base swe base &
run expert_swe swe base:expert_swe &
run expert_terminal terminal base:expert_terminal &
run merged_linear swe base:merged_linear &
run merged_linear terminal base:merged_linear &

# MOPD student: load the adapter into the running server as soon as it arrives
( until [ -f adapters/expert_mopd/adapter_model.safetensors ] && [ -f adapters/expert_mopd/READY ]; do sleep 60; done
  curl -sf -X POST "http://127.0.0.1:30000/load_lora_adapter" -H "Content-Type: application/json" \
    -d '{"lora_name": "mopd", "lora_path": "adapters/expert_mopd"}' && echo "$(date -Is) mopd adapter loaded"
  run mopd swe base:mopd &
  run mopd terminal base:mopd &
  wait ) &
wait
kill "$SERVER"
echo "$(date -Is) x86 lora plan complete"
