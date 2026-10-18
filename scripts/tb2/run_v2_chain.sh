#!/bin/bash
# TB2 rerun with the fixed inference settings, base then merged_ties, unattended:
#   context 40960 (max_input_tokens 36864), Qwen3 non-thinking sampling top_p 0.8 / top_k 20 / presence_penalty 1.5,
#   temperature 0.7, same agent / tasks / lanes / concurrency (8 + 3 lanes) as the original base run.
set -u
cd /root/gpufree-data/tb2
export TB_CONTEXT=40960 TB_TOP_P=0.8 TB_TOP_K=20 TB_PRESENCE_PENALTY=1.5
log() { echo "$(date -Is) $*"; }

stop_server() {  # by listening port, not by command-line pattern (pgrep -f would match this script's callers)
  local pids; pids=$(ss -ltnp 'sport = :30000' | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
  [ -n "$pids" ] && kill $pids
  for _ in $(seq 1 60); do ss -ltn 'sport = :30000' | grep -q 30000 || break; sleep 2; done
  tmux kill-session -t model 2>/dev/null; sleep 5
}
serve() {  # model_dir served_name
  stop_server
  tmux new-session -d -s model "CONTEXT_LENGTH=$TB_CONTEXT bash serve_sglang.sh $1 $2 2>&1 | tee model_server_$2.log"
  for _ in $(seq 1 120); do curl -sf 127.0.0.1:30000/v1/models | grep -q "\"$2\"" && { log "server up: $2"; return 0; }; sleep 5; done
  log "server FAILED to start: $2"; return 1
}
run_model() {  # prefix served_name
  MODEL_NAME=$2 bash run_split.sh "$1" model 8 all
  sleep 30
  while tmux ls 2>/dev/null | grep -q "^$1_"; do sleep 60; done
  log "jobs done: $1"
}

serve models/Qwen3-8B Qwen3-8B && run_model v2-base Qwen3-8B
serve /root/gpufree-data/projects/moe-merge-dgx/checkpoints/merged_ties merged_ties && run_model v2-merged_ties merged_ties
stop_server
log "CHAIN DONE"
