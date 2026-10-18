#!/usr/bin/env bash
set -euo pipefail

echo "Setting up environment for MOE-Merge experiment..."

mkdir -p data/{swe,hle,asi,terminal}
mkdir -p checkpoints results logs

pip install -r requirements.txt

# flash-attention is optional but recommended for throughput
pip install flash-attn --no-build-isolation || echo "Warning: flash-attn install failed, continuing without it"

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo "Environment setup complete!"

# Base weights: link the sha256-verified Qwen3-8B snapshot (or download it)
mkdir -p models
if [ ! -e models/Qwen3-8B ]; then
  if [ -d "$HOME/model-merging/runs/qwen3-8b-smoke-20261016/verified-model" ]; then
    ln -s "$HOME/model-merging/runs/qwen3-8b-smoke-20261016/verified-model" models/Qwen3-8B
  else
    huggingface-cli download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir models/Qwen3-8B
  fi
fi
