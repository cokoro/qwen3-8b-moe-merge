#!/usr/bin/env bash
# Extra experiment for the "small experts -> bigger student" idea: MOPD with the two 8B
# expert teachers (expert_hle, expert_asi) and a Qwen3-14B student. Runs after the main
# plan (8B MOPD + its evals) so the required results are never at risk.
set -uo pipefail
mkdir -p logs
until grep -q "mopd plan done" logs/run_mopd_plan.log 2>/dev/null && grep -q "lane :30000 complete" logs/lane_a.log 2>/dev/null; do sleep 120; done
until grep -q DOWNLOAD_DONE logs/download_14b.log 2>/dev/null; do sleep 60; done
until ! pgrep -f "scripts/judge_hle_all.sh" >/dev/null; do sleep 60; done   # judge done with the main models
echo "$(date -Is) starting MOPD-14B"

[ -f checkpoints/expert_mopd14b/training_metrics.json ] && grep -q '"final": true' checkpoints/expert_mopd14b/training_metrics.json || \
  bash scripts/timed_run.sh train_mopd14b python -m src.distill.mopd --student_base models/Qwen3-14B \
    --teachers hle=checkpoints/expert_hle asi=checkpoints/expert_asi --output checkpoints/expert_mopd14b \
    --batch_per_domain 6 --max_minutes "${MOPD14B_MINUTES:-120}" > logs/train_mopd14b.log 2>&1 \
  || { echo "MOPD-14B training FAILED"; exit 1; }
echo "$(date -Is) MOPD-14B trained"

PLAN="mopd14b:hle,asi" PORT=30001 MEM_FRACTION=0.6 bash scripts/evaluate_all.sh > logs/lane_mopd14b.log 2>&1
bash scripts/judge_hle_all.sh > logs/judge_hle_night.log 2>&1
echo "$(date -Is) night plan done"
