"""Fold a single LoRA expert into the base and save a standalone model.

Used so every evaluated model (base, experts, merged) is a full checkpoint that
the inference server can load the same way.

    python -m src.merging.fold_expert --adapter checkpoints/expert_swe --output checkpoints/full/expert_swe
"""

import argparse
import json
import time
from pathlib import Path

import yaml

from src.merging.lora_delta import merge_into_base


def main():
    parser = argparse.ArgumentParser(description="Fold one LoRA adapter into the base model")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/merging.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    start = time.time()
    stats = merge_into_base(
        config["base_model"], config.get("base_revision"),
        [{"name": Path(args.adapter).name, "adapter_path": args.adapter}],
        lambda deltas: deltas[0], Path(args.output),
    )
    manifest = {"method": "fold", "base_model": config["base_model"],
                "base_revision": config.get("base_revision"), "source_adapter_paths": [args.adapter],
                "elapsed_seconds": time.time() - start, **stats}
    (Path(args.output) / "merge_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Folded {args.adapter} -> {args.output}")


if __name__ == "__main__":
    main()
