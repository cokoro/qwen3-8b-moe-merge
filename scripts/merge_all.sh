#!/usr/bin/env bash
set -euo pipefail

echo "Merging all expert models..."

echo "Performing linear merge..."
python -m src.merging.merge_linear --output checkpoints/merged_linear

echo "Performing TIES merge..."
python -m src.merging.merge_ties --output checkpoints/merged_ties --density 0.2

echo "Verifying merged_linear..."
python -m src.merging.verify_merged \
    --model_path checkpoints/merged_linear

echo "Verifying merged_ties..."
python -m src.merging.verify_merged \
    --model_path checkpoints/merged_ties

echo "Merging complete!"
