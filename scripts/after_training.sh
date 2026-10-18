#!/usr/bin/env bash
# Wait for train_all.sh to finish, then merge (linear + TIES) and fold each expert
# into a standalone checkpoint for serving. Safe to re-run: finished steps are skipped.
set -uo pipefail
while pgrep -f "scripts/train_all.sh" >/dev/null; do sleep 60; done

for E in swe hle asi terminal; do
  [ -f "checkpoints/expert_$E/training_metrics.json" ] || { echo "expert_$E missing; not merging"; exit 1; }
done

[ -f checkpoints/merged_linear/merge_manifest.json ] || \
  bash scripts/timed_run.sh merge_linear python -m src.merging.merge_linear --output checkpoints/merged_linear
[ -f checkpoints/merged_ties/merge_manifest.json ] || \
  bash scripts/timed_run.sh merge_ties python -m src.merging.merge_ties --output checkpoints/merged_ties --density 0.2
for M in merged_linear merged_ties; do
  python -m src.merging.verify_merged --model_path "checkpoints/$M" > "logs/verify_$M.log" 2>&1 \
    && echo "$M verified" || echo "$M verification FAILED (logs/verify_$M.log)"
done
for E in swe hle asi terminal; do
  [ -f "checkpoints/full/expert_$E/merge_manifest.json" ] || \
    bash scripts/timed_run.sh "fold_$E" python -m src.merging.fold_expert \
      --adapter "checkpoints/expert_$E" --output "checkpoints/full/expert_$E"
done
echo "after_training done"
