#!/bin/bash
# Run all 89 TB2 tasks as concurrent harbor jobs, because containers share the host network namespace:
#   <prefix>-net{a,b,c} : 17 tasks that bind host ports or act on other processes, in 3 serial lanes (lanes.sh)
#   <prefix>-main       : the other 72 tasks, parallel (-n $N, default 8: 8+3 agents fit in the sglang KV cache)
# usage: bash run_split.sh <prefix> oracle|model [N] [all|main|net]
set -e
PREFIX=${1:?prefix}; MODE=${2:?oracle|model}; N=${3:-8}; PART=${4:-all}
cd /root/gpufree-data/tb2
source env.sh
source lanes.sh   # LANE_A / LANE_B / LANE_C (conflicting tasks share a lane)

if [ "$MODE" = oracle ]; then
  CMD=(harbor run -p "$TB2_DATASET" -a oracle -e singularity --ek singularity_image_cache_dir="$SIF_CACHE" -o jobs -y -q)
else
  CMD=(env MODEL_NAME="${MODEL_NAME:-Qwen3-8B}" bash run_tb2.sh)   # explicit: tmux sessions do not inherit the caller env
fi
launch() {  # session job_name args...
  local s=$1 j=$2; shift 2
  if [ "$MODE" = oracle ]; then all=("${CMD[@]}" --job-name "$j" "$@"); else all=("${CMD[@]}" "$j" -q "$@"); fi
  tmux new-session -d -s "$s" "$(printf '%q ' "${all[@]}") 2>&1 | grep --line-buffered -v LiteLLM | tee logs_$j.log"
  echo "started $s -> jobs/$j"
}
flags() { local f=$1; shift; for t in "$@"; do printf -- "%s\n%s\n" "$f" "$t"; done; }
mapfile -t INC_A < <(flags -i $LANE_A)
mapfile -t INC_B < <(flags -i $LANE_B)
mapfile -t INC_C < <(flags -i $LANE_C)
mapfile -t EXC   < <(flags -x $LANE_A $LANE_B $LANE_C)

if [ "$PART" != main ]; then
  launch "${PREFIX}_neta" "${PREFIX}-neta" "${INC_A[@]}" -n 1
  launch "${PREFIX}_netb" "${PREFIX}-netb" "${INC_B[@]}" -n 1
  launch "${PREFIX}_netc" "${PREFIX}-netc" "${INC_C[@]}" -n 1
fi
[ "$PART" = net ] || launch "${PREFIX}_main" "${PREFIX}-main" "${EXC[@]}" -n "$N"
