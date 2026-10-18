"""Shared helpers for merging LoRA experts directly in delta space.

For a LoRA adapter, the update to a target weight is
``delta_W = (lora_B @ lora_A) * (lora_alpha / r)``. Every expert shares the
same frozen base, so "fold each expert into a full model, then combine the
state dicts" is equivalent to ``W_base + combine(delta_W_1..n)`` on the LoRA
target weights only. Working per-module keeps peak memory at roughly one base
model plus a few delta matrices, instead of n full model copies.
"""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# (lora_A, lora_B, scale) for one target module
LoraFactors = Tuple[torch.Tensor, torch.Tensor, float]


def load_lora_factors(adapter_path: str) -> Dict[str, LoraFactors]:
    """Return {base_weight_name: (A, B, scale)} for a saved PEFT LoRA adapter."""
    adapter_dir = Path(adapter_path)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("use_dora") or cfg.get("use_rslora"):
        raise ValueError(f"{adapter_path}: DoRA/rsLoRA adapters are not supported")
    default_scale = cfg["lora_alpha"] / cfg["r"]
    rank_pattern = cfg.get("rank_pattern") or {}
    alpha_pattern = cfg.get("alpha_pattern") or {}
    if rank_pattern or alpha_pattern:
        raise ValueError(f"{adapter_path}: per-module rank/alpha patterns are not supported")

    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    factors: Dict[str, LoraFactors] = {}
    for key, a in tensors.items():
        if ".lora_A." not in key:
            continue
        b = tensors[key.replace(".lora_A.", ".lora_B.")]
        # base_model.model.<module>.lora_A.weight -> <module>.weight
        module = key.split(".lora_A.")[0]
        if module.startswith("base_model.model."):
            module = module[len("base_model.model."):]
        factors[f"{module}.weight"] = (a, b, default_scale)
    if not factors:
        raise ValueError(f"{adapter_path}: no LoRA weights found")
    return factors


def lora_delta(factors: LoraFactors, device: torch.device) -> torch.Tensor:
    """Materialize delta_W in float32 on ``device``."""
    a, b, scale = factors
    return (b.to(device, torch.float32) @ a.to(device, torch.float32)) * scale


def load_all_experts(experts: List[dict]) -> List[Dict[str, LoraFactors]]:
    all_factors = []
    for e in experts:
        path = e["adapter_path"]
        if not Path(path).exists():
            raise FileNotFoundError(f"Expert adapter not found: {path}. Train the experts before merging.")
        print(f"Loading LoRA factors for expert '{e['name']}' from {path}")
        all_factors.append(load_lora_factors(path))
    keys = set(all_factors[0])
    for e, f in zip(experts, all_factors):
        if set(f) != keys:
            raise ValueError(f"Expert '{e['name']}' targets different modules than '{experts[0]['name']}'")
    return all_factors


def merge_into_base(
    base_model_name: str,
    base_revision: str,
    experts: List[dict],
    combine: Callable[[List[torch.Tensor]], torch.Tensor],
    output_path: Path,
) -> Dict[str, int]:
    """Load the base, add ``combine(deltas)`` to each LoRA target weight, save.

    Returns simple stats about the merge.
    """
    # MERGE_DEVICE=cpu when the GPU is busy serving (math is identical, just slower)
    device = torch.device(os.environ.get("MERGE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_factors = load_all_experts(experts)

    print(f"Loading base model {base_model_name}@{base_revision} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, revision=base_revision, dtype=torch.bfloat16,
        device_map={"": device.index or 0} if device.type == "cuda" else None,
    )
    params = dict(model.named_parameters())

    n_modules = 0
    with torch.no_grad():
        for name in sorted(all_factors[0]):
            if name not in params:
                raise KeyError(f"LoRA target {name} not found in base model")
            deltas = [lora_delta(f[name], device) for f in all_factors]
            merged = combine(deltas)
            p = params[name]
            p.copy_((p.to(torch.float32) + merged).to(p.dtype))
            n_modules += 1
            del deltas, merged

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_path} ...")
    model.save_pretrained(output_path, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(base_model_name, revision=base_revision).save_pretrained(output_path)
    total_bytes = sum(f.stat().st_size for f in output_path.glob("*.safetensors"))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"merged_modules": n_modules, "checkpoint_bytes": total_bytes}
