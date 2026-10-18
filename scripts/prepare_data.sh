#!/usr/bin/env bash
set -euo pipefail

echo "Preparing training data for all experts..."

echo "Preparing SWE data..."
python -m src.data.prepare_swe --output data/swe --max_samples 5000

echo "Preparing HLE data..."
python -m src.data.prepare_hle --output data/hle --max_samples 3000

echo "Preparing ASI data..."
python -m src.data.prepare_asi --output data/asi --max_samples 4000

echo "Preparing Terminal data..."
python -m src.data.prepare_terminal --output data/terminal --max_samples 2000 \
    --seta_dir "${SETA_DIR:-$HOME/model-merging/seed-expanded-20261016/terminal-seeds/tasks}"

echo "Auditing for test set contamination..."
python -m src.data.audit_contamination --train_dir data --build_references

echo "Data preparation complete!"
