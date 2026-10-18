#!/usr/bin/env bash
# MOPD on DGX lane B (port 30001): take the lane over once merged_linear's ASI score is
# written, train the MOPD student, then evaluate it (and expert_asi, deferred) on HLE+ASI.
set -uo pipefail
PORT=30001
mkdir -p logs
until [ -f results/merged_linear/asi/results.json ]; do sleep 30; done
# stop lane B's evaluate_all (identified by PORT=30001 in its environment) and its server
for pid in $(pgrep -f "bash scripts/evaluate_all.sh"); do
  tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -qx "PORT=$PORT" && { echo "stop lane B $pid"; kill "$pid"; }
done
for pid in $(pgrep -f "sglang.launch_server.*--port $PORT"); do kill "$pid"; done
for pid in $(pgrep -f "bench_asi.sh expert_asi") $(pgrep -f "results/expert_asi/asi"); do kill "$pid"; done
sleep 30
rm -rf results/expert_asi/asi   # partial run from lane B, redone below
echo "$(date -Is) lane B taken over for MOPD"

[ -f checkpoints/expert_mopd/training_metrics.json ] && grep -q '"final": true' checkpoints/expert_mopd/training_metrics.json || \
  bash scripts/timed_run.sh train_mopd python -m src.distill.mopd \
    --teachers hle=checkpoints/expert_hle asi=checkpoints/expert_asi \
    --output checkpoints/expert_mopd --max_minutes "${MOPD_MINUTES:-150}" \
    --batch_per_domain "${MOPD_BATCH:-16}" --max_new_tokens "${MOPD_MAX_NEW:-320}" > logs/train_mopd.log 2>&1 \
  || { echo "MOPD training FAILED"; exit 1; }
echo "$(date -Is) MOPD student trained"
# expert_asi's finished ASI run was set aside while lane B was taken over; put it back
if [ -f results_hold/expert_asi_asi/results.json ] && [ ! -d results/expert_asi/asi ]; then
  mkdir -p results/expert_asi && mv results_hold/expert_asi_asi results/expert_asi/asi
fi

PLAN="mopd:hle,asi expert_asi:asi" PORT=$PORT MEM_FRACTION=0.4 bash scripts/evaluate_all.sh > logs/lane_mopd.log 2>&1
echo "$(date -Is) mopd plan done"
