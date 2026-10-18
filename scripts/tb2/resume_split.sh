#!/bin/bash
# Resume an interrupted run_split.sh model run: for each lane, start a new job "<prefix>-<round>-<lane>" with only
# the tasks that have no finished trial in any "<prefix>-*" job yet. tb2_summary.py merges all rounds afterwards.
# usage: bash resume_split.sh <prefix> <round> [N]      e.g. bash resume_split.sh qwen3-8b-base r2 8
#   re-export the same TB_* settings as the interrupted round, e.g.
#   TB_CONTEXT=40960 TB_TOP_P=0.8 TB_TOP_K=20 TB_PRESENCE_PENALTY=1.5 MODEL_NAME=merged_ties bash resume_split.sh v2-merged_ties r2 8
set -e
PREFIX=${1:?prefix}; ROUND=${2:?round}; N=${3:-8}
cd /root/gpufree-data/tb2
source env.sh
source lanes.sh   # LANE_A / LANE_B / LANE_C (conflicting tasks share a lane)
NET="$LANE_A $LANE_B $LANE_C"

DONE=$(python3 - "$PREFIX" <<'EOF'
import glob, json, sys
done = set()
for f in glob.glob(f"jobs/{sys.argv[1]}-*/*/result.json"):
    d = json.load(open(f))
    if d.get("finished_at"):
        done.add(d["task_name"])
print(" ".join(sorted(done)))
EOF
)
ALL=$(ls -d "$TB2_DATASET"/*/ | xargs -n1 basename)
todo() { for t in $1; do case " $DONE " in *" $t "*) ;; *) echo "$t";; esac; done; }
MAIN=$(for t in $ALL; do case " $NET " in *" $t "*) ;; *) echo "$t";; esac; done)

launch() {  # lane n tasks...
  local lane=$1 n=$2; shift 2
  [ $# -gt 0 ] || { echo "$lane: nothing left"; return; }
  local j="${PREFIX}-${ROUND}-${lane}" inc=()
  for t in "$@"; do inc+=(-i "$t"); done
  # explicit: tmux sessions do not inherit the caller env (TB_* = inference settings, must match the original round)
  local cmd=(env MODEL_NAME="${MODEL_NAME:-Qwen3-8B}")
  for v in TB_CONTEXT TB_TEMPERATURE TB_TOP_P TB_TOP_K TB_PRESENCE_PENALTY; do [ -n "${!v:-}" ] && cmd+=("$v=${!v}"); done
  cmd+=(bash run_tb2.sh "$j" -q "${inc[@]}" -n "$n")
  tmux new-session -d -s "${PREFIX}_${ROUND}_${lane}" "$(printf '%q ' "${cmd[@]}") 2>&1 | grep --line-buffered -v LiteLLM | tee logs_$j.log"
  echo "started $j ($# tasks)"
}
launch neta 1 $(todo "$LANE_A")
launch netb 1 $(todo "$LANE_B")
launch netc 1 $(todo "$LANE_C")
launch main "$N" $(todo "$MAIN")
