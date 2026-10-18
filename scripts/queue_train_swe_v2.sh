#!/usr/bin/env bash
# Train the v2 SWE expert (mini-swe-agent tool-call trajectories) after the terminal v2 expert finishes,
# so the two trainings do not share the GB10. Same v2 settings: 8192 tokens, 120 steps.
cd ~/moe-merge; source ~/miniconda3/etc/profile.d/conda.sh; conda activate moe-merge
until [ -f data/swe_v2/READY ] && grep -q "END train_terminal_v2" logs/train_terminal_v2.log 2>/dev/null; do sleep 60; done
bash scripts/timed_run.sh train_swe_v2 python -m src.training.train_expert --expert swe --config configs/training_v2.yaml \
  --experiment_config configs/experiment_v2.yaml --output_dir checkpoints/expert_swe_v2 --report_to none > logs/train_swe_v2.log 2>&1
echo "$(date -Is) train_swe_v2 exit $?"
