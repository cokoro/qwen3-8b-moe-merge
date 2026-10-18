#!/usr/bin/env bash
# DGX side of the merged_ties TB2/SWE run. merged_ties cannot share the x86 multi-LoRA server (TIES is not
# low-rank) and the L40S holds only one bf16 8B server, so merged_ties is served from the DGX and reached
# from the x86 host through an SSH reverse tunnel (x86 127.0.0.1:30002 -> DGX 127.0.0.1:30001).
# Runs after the HLE repeat run of merged_linear; the mopd repeat run is skipped to free the GPU.
set -uo pipefail
cd ~/moe-merge
source ~/miniconda3/etc/profile.d/conda.sh; conda activate moe-merge
X86="${X86:?set X86=user@x86-host}"; X86_PORT="${X86_PORT:?set X86_PORT}"; KEY="${KEY:-$HOME/.ssh/l40s_transfer}"

until grep -qE "rep2 merged_linear (generated|FAILED)" logs/hle_rep2.log; do sleep 20; done
for p in $(ps -ww -eo pid,args | awk '$2 == "bash" && $3 == "/tmp/hle_rep2.sh" {print $1}'); do kill "$p"; done
sleep 5
port_pids() { ps -ww -eo pid,args | awk '/sglang.launch_server/ && /--port 30001/ && !/awk/ {print $1}'; }
for p in $(port_pids); do kill "$p"; done
sleep 20
for p in $(port_pids); do pkill -9 -P "$p"; kill -9 "$p"; done   # sglang can ignore SIGTERM while busy
echo "$(date -Is) rep2 mopd FAILED (skipped: GPU reassigned to merged_ties TB2/SWE)" >> logs/hle_rep2.log
sleep 30

MEM_FRACTION=0.4 bash scripts/serve_model.sh checkpoints/merged_ties merged_ties 30001 > logs/serve_merged_ties_remote.log 2>&1 &
for _ in $(seq 1 240); do curl -sf http://127.0.0.1:30001/v1/models | grep -q merged_ties && break; sleep 5; done
echo "$(date -Is) merged_ties server up on DGX:30001"

# keep the reverse tunnel alive until the x86 side reports completion
while ! ssh -i "$KEY" -p "$X86_PORT" -o BatchMode=yes "$X86" \
      'grep -q "ties remote complete" /root/gpufree-data/projects/swe-eval/logs/run_x86_ties_remote.log 2>/dev/null'; do
  ssh -i "$KEY" -p "$X86_PORT" -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      -o ExitOnForwardFailure=yes -N -R 30002:127.0.0.1:30001 "$X86" &
  T=$!
  while kill -0 "$T" 2>/dev/null; do
    sleep 60
    ssh -i "$KEY" -p "$X86_PORT" -o BatchMode=yes "$X86" \
      'grep -q "ties remote complete" /root/gpufree-data/projects/swe-eval/logs/run_x86_ties_remote.log 2>/dev/null' && kill "$T"
  done
  echo "$(date -Is) tunnel exited"; sleep 5
done
for p in $(ps -ww -eo pid,args | awk '/sglang.launch_server/ && /--port 30001/ && !/awk/ {print $1}'); do kill "$p"; done
echo "$(date -Is) merged_ties remote serving done"
