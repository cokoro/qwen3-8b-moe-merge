#!/usr/bin/env bash
# Train the 4 experts sequentially; each run is timed into results/experiment_times.csv
# and logged to logs/train_<expert>.log. Set EXPERTS="swe hle" to train a subset.
set -uo pipefail

EXPERTS="${EXPERTS:-swe hle asi terminal}"
mkdir -p logs results

for EXPERT in $EXPERTS; do
  if [ -f "checkpoints/expert_$EXPERT/training_metrics.json" ]; then
    echo "expert_$EXPERT already trained, skipping"
    continue
  fi
  echo "Training $EXPERT expert..."
  bash scripts/timed_run.sh "train_$EXPERT" \
    python -m src.training.train_expert --expert "$EXPERT" > "logs/train_$EXPERT.log" 2>&1 \
    || echo "expert_$EXPERT FAILED (see logs/train_$EXPERT.log)"
done

echo "All experts done!"
