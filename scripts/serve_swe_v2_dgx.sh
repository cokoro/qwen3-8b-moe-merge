#!/usr/bin/env bash
# DGX side of the SWE v2 rerun: serve each model in turn with a 64K context (YaRN factor 2 over the native 32K,
# identical for every model) and expose it to the x86 host via a reverse tunnel (x86:30012 -> DGX:30011).
# The x86 runner for each model is started over ssh and the next model is served when it reports done.
#   X86=user@host X86_PORT=port bash scripts/serve_swe_v2_dgx.sh base=models/Qwen3-8B merged_ties=checkpoints/merged_ties
set -uo pipefail
cd ~/moe-merge
source ~/miniconda3/etc/profile.d/conda.sh; conda activate moe-merge
X86="${X86:?}"; X86_PORT="${X86_PORT:?}"; KEY="${KEY:-$HOME/.ssh/l40s_transfer}"
SSH="ssh -i $KEY -p $X86_PORT -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
port_pids() { ps -ww -eo pid,args | awk '/sglang.launch_server/ && /--port 30011/ && !/awk/ {print $1}'; }
for spec in "$@"; do
  M=${spec%%=*}; DIR=${spec#*=}
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 CONTEXT_LENGTH=65536 MEM_FRACTION=0.25 bash scripts/serve_model.sh "$DIR" "$M" 30011 \
    --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768,"rope_theta":1000000.0}}' \
    > logs/serve_swe_v2_$M.log 2>&1 &
  for _ in $(seq 1 240); do curl -sf http://127.0.0.1:30011/v1/models | grep -q "\"$M\"" && break; sleep 5; done
  echo "$(date -Is) $M served (64K, YaRN x2)"
  rssh() { for _ in $(seq 1 10); do $SSH "$X86" "$1" && return 0; sleep 15; done; return 1; }
  # x86 runner in the background (one short ssh, retried: the x86 sshd drops connections under load)
  rssh "cd /root/gpufree-data/projects/swe-eval && (setsid nohup bash scripts/run_swe_v2_x86.sh $M > logs/run_swe_v2_$M.log 2>&1 < /dev/null &)"
  # reverse tunnel, re-established whenever it drops, until the runner reports done
  ( while true; do $SSH -N -o ExitOnForwardFailure=yes -R 30012:127.0.0.1:30011 "$X86"; sleep 10; done ) &
  T=$!
  until rssh "grep -q 'swe_v2 $M done' /root/gpufree-data/projects/swe-eval/logs/run_swe_v2_$M.log"; do sleep 120; done
  echo "$(date -Is) $M x86 run finished"
  pkill -P $T;   kill $T 2>/dev/null
  for p in $(port_pids); do kill $p; done; sleep 20
  for p in $(port_pids); do pkill -9 -P $p; kill -9 $p; done; sleep 10
done
echo "$(date -Is) swe v2 all done"
