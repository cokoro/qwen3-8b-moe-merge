"""Merge multiple LoRA expert adapters into a single model via weighted averaging.

Averaging the folded expert models is equivalent to adding the weighted
average of the LoRA deltas to the base (all non-LoRA weights are identical),
so the merge is done per target module in delta space. See lora_delta.py.
Math is float32, weights are saved as bfloat16.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.merging.lora_delta import merge_into_base


def main():
    parser = argparse.ArgumentParser(description="Linear weighted-average merge of expert LoRA adapters")
    parser.add_argument("--config", default="configs/merging.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default=None,
                        help="Comma-separated 4 floats overriding config weights")
    parser.add_argument("--base_model", default=None,
                        help="Override base model name from config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    base_model_name = args.base_model or config["base_model"]
    experts = config["experts"]

    if args.weights:
        weights = [float(x) for x in args.weights.split(",")]
        if len(weights) != len(experts):
            raise ValueError(f"--weights has {len(weights)} values, expected {len(experts)}")
    else:
        weights = [e.get("weight", 1.0 / len(experts)) for e in experts]

    if config.get("linear", {}).get("normalize_weights", True):
        total = sum(weights)
        weights = [w / total for w in weights]

    print(f"Linear merge: base={base_model_name}")
    for e, w in zip(experts, weights):
        print(f"  expert={e['name']:10s} adapter={e['adapter_path']} weight={w:.4f}")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    start = time.time()
    start_iso = datetime.now().isoformat()

    def combine(deltas):
        return sum(w * d for w, d in zip(weights, deltas))

    stats = merge_into_base(
        base_model_name, config.get("base_revision"), experts, combine, output_path
    )

    elapsed = time.time() - start
    manifest = {
        "method": "linear",
        "base_model": base_model_name,
        "base_revision": config.get("base_revision"),
        "weights": weights,
        **stats,
        "expert_names": [e["name"] for e in experts],
        "source_adapter_paths": [e["adapter_path"] for e in experts],
        "timestamp": start_iso,
        "elapsed_seconds": elapsed,
    }
    (output_path / "merge_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Merge complete in {elapsed:.1f}s. Manifest at {output_path}/merge_manifest.json")


if __name__ == "__main__":
    main()
