#!/usr/bin/env bash
# ASI v2 rerun (official per-task runtime environments) for base, expert_asi, merged_linear (rank-64 LoRA form) and
# mopd from one multi-LoRA server; merged_ties (full weights) runs afterwards from its own server.
set -uo pipefail
cd ~/moe-merge
source ~/miniconda3/etc/profile.d/conda.sh; conda activate moe-merge
. ~/.config/proxy.env 2>/dev/null   # pip/uv needs network to build the task environments
port_pids() { ps -ww -eo pid,args | awk -v p="--port $1" '/sglang.launch_server/ && index($0,p) && !/awk/ {print $1}'; }
stop_port() { for p in $(port_pids $1); do kill $p; done; sleep 20; for p in $(port_pids $1); do pkill -9 -P $p; kill -9 $p; done; }

MEM_FRACTION=0.3 bash scripts/serve_model.sh models/Qwen3-8B base 30021 --enable-lora --max-lora-rank 64 \
  --lora-paths expert_asi=checkpoints/expert_asi merged_linear=adapters/merged_linear mopd=checkpoints/expert_mopd \
  > logs/serve_asi_v2_lora.log 2>&1 &
for _ in $(seq 1 240); do curl -sf http://127.0.0.1:30021/v1/models | grep -q expert_asi && break; sleep 5; done
echo "$(date -Is) lora server up"
export MODEL_BASE_URL=http://127.0.0.1:30021/v1 ASI_PARALLEL=6
for pair in base:base expert_asi:expert_asi merged_linear:merged_linear mopd:mopd; do
  M=${pair%%:*}
  S=base; [ "$M" != base ] && S="base:$M"
  ( RESULTS_DIR=results_v2/$M SERVED_MODEL=$S bash scripts/timed_run.sh "asi_v2_$M" bash scripts/bench_asi_v2.sh "$M" > logs/eval_v2_${M}_asi.log 2>&1
    echo "$(date -Is) asi_v2 $M done rc=$?" ) &
  sleep 60   # stagger task-environment creation
done
wait
stop_port 30021
echo "$(date -Is) lora models done; waiting for merged_ties SWE v2 to release memory"
until grep -q "swe v2 all done" logs/swe_v2_loop.log; do sleep 60; done
MEM_FRACTION=0.3 bash scripts/serve_model.sh checkpoints/merged_ties merged_ties 30022 > logs/serve_asi_v2_ties.log 2>&1 &
for _ in $(seq 1 240); do curl -sf http://127.0.0.1:30022/v1/models | grep -q merged_ties && break; sleep 5; done
MODEL_BASE_URL=http://127.0.0.1:30022/v1 ASI_PARALLEL=12 RESULTS_DIR=results_v2/merged_ties \
  bash scripts/timed_run.sh asi_v2_merged_ties bash scripts/bench_asi_v2.sh merged_ties > logs/eval_v2_merged_ties_asi.log 2>&1
echo "$(date -Is) asi_v2 merged_ties done rc=$?"
stop_port 30022
echo "$(date -Is) asi v2 all done"
