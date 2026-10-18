"""Merge multiple LoRA expert adapters with TIES-Merging.

TIES-Merging (Trim, Elect, Sign):
1. For each expert, the *task vector* is its LoRA delta
   ``delta_i = (B_i @ A_i) * alpha / r`` (identical to folded expert minus base).
2. Trim: per-parameter tensor, keep only the top-``density`` fraction by
   absolute magnitude; zero the rest.
3. Elect (sign consensus): for each parameter index across experts, take the
   majority sign weighted by total magnitude. Only updates that agree with the
   majority sign are retained.
4. Sum the retained, sign-consistent task vectors, rescale by the number of
   contributing experts at each index, and add to the base weights.

Done per target module in float32 on the GPU (see lora_delta.py); non-LoRA
weights are untouched because they are identical across experts.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.merging.lora_delta import merge_into_base


def trim_tensor(t: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the top-``density`` fraction of entries by |value|, zero the rest."""
    if density >= 1.0 or t.numel() == 0:
        return t
    k_keep = max(1, int(t.numel() * density))
    threshold = torch.topk(t.abs().flatten(), k_keep, largest=True, sorted=False).values.min()
    return t * (t.abs() >= threshold)


def ties_combine(deltas, density, majority_sign_method="total"):
    """TIES over one module's expert deltas: trim, elect sign, disjoint mean."""
    stacked = torch.stack([trim_tensor(d, density) for d in deltas], dim=0)
    signs = stacked.sign()
    if majority_sign_method == "frequency":
        vote = (signs > 0).sum(dim=0) - (signs < 0).sum(dim=0)
    else:  # "total": sign of the summed trimmed values
        vote = stacked.sum(dim=0)
    majority_sign = torch.where(vote >= 0, 1.0, -1.0)
    agree = signs == majority_sign.unsqueeze(0)
    n_contrib = agree.sum(dim=0).clamp(min=1)
    return (stacked * agree).sum(dim=0) / n_contrib


def main():
    parser = argparse.ArgumentParser(description="TIES-Merging of expert LoRA adapters")
    parser.add_argument("--config", default="configs/merging.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--density", type=float, default=0.2,
                        help="Fraction of parameters to keep per tensor after trimming")
    parser.add_argument("--majority_sign_method", choices=["total", "frequency"], default="total")
    parser.add_argument("--base_model", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    base_model_name = args.base_model or config["base_model"]
    experts = config["experts"]
    density = args.density

    print(f"TIES merge: base={base_model_name} density={density} sign={args.majority_sign_method}")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    start = time.time()
    start_iso = datetime.now().isoformat()

    for e in experts:
        print(f"  expert={e['name']:10s} adapter={e['adapter_path']}")

    stats = merge_into_base(
        base_model_name,
        config.get("base_revision"),
        experts,
        lambda deltas: ties_combine(deltas, density, args.majority_sign_method),
        output_path,
    )

    elapsed = time.time() - start
    manifest = {
        "method": "ties",
        "base_model": base_model_name,
        "base_revision": config.get("base_revision"),
        "density": density,
        **stats,
        "majority_sign_method": args.majority_sign_method,
        "expert_names": [e["name"] for e in experts],
        "source_adapter_paths": [e["adapter_path"] for e in experts],
        "timestamp": start_iso,
        "elapsed_seconds": elapsed,
    }
    (output_path / "merge_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"TIES merge complete in {elapsed:.1f}s. Manifest at {output_path}/merge_manifest.json")


if __name__ == "__main__":
    main()
