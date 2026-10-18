"""Express the linear merge of n rank-r LoRA experts *exactly* as one rank-(n*r) LoRA.

    sum_i w_i * s_i * B_i A_i  ==  [w_1 s_1 B_1 | ... | w_n s_n B_n] @ [A_1; ...; A_n]

With lora_alpha = r_new the adapter scale is 1, so the stacked factors reproduce the
merged delta of src/merging/merge_linear.py (up to bf16 storage rounding). This lets
an inference server serve the merged model as a LoRA on the shared base, next to the
experts, without materialising another full checkpoint.

    python -m src.merging.linear_as_lora --config configs/merging_x86.yaml --output adapters/merged_linear
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/merging.yaml")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    experts = cfg["experts"]
    weights = [e.get("weight", 1.0 / len(experts)) for e in experts]
    total = sum(weights)
    weights = [w / total for w in weights]

    adapters, scales, ref_cfg = [], [], None
    for e in experts:
        d = Path(e["adapter_path"])
        c = json.loads((d / "adapter_config.json").read_text())
        ref_cfg = ref_cfg or c
        scales.append(c["lora_alpha"] / c["r"])
        adapters.append(load_file(str(d / "adapter_model.safetensors")))

    keys_a = sorted(k for k in adapters[0] if ".lora_A." in k)
    out = {}
    for ka in keys_a:
        kb = ka.replace(".lora_A.", ".lora_B.")
        out[ka] = torch.cat([ad[ka].float() for ad in adapters], dim=0).to(torch.bfloat16)
        out[kb] = torch.cat([w * s * ad[kb].float() for ad, w, s in zip(adapters, weights, scales)],
                            dim=1).to(torch.bfloat16)
    new_r = sum(ad[keys_a[0]].shape[0] for ad in adapters)

    o = Path(args.output)
    o.mkdir(parents=True, exist_ok=True)
    save_file(out, str(o / "adapter_model.safetensors"))
    new_cfg = dict(ref_cfg, r=new_r, lora_alpha=new_r, lora_dropout=0.0)
    new_cfg.pop("rank_pattern", None)
    new_cfg.pop("alpha_pattern", None)
    (o / "adapter_config.json").write_text(json.dumps(new_cfg, indent=2))
    (o / "merge_manifest.json").write_text(json.dumps({
        "method": "linear_as_lora", "weights": weights, "expert_names": [e["name"] for e in experts],
        "source_adapter_paths": [e["adapter_path"] for e in experts], "rank": new_r}, indent=2))

    # exactness check on one module: stacked product vs weighted sum of expert deltas
    ka = keys_a[0]
    kb = ka.replace(".lora_A.", ".lora_B.")
    ref = sum(w * s * ad[kb].float() @ ad[ka].float() for ad, w, s in zip(adapters, weights, scales))
    got = out[kb].float() @ out[ka].float()
    rel = ((got - ref).abs().max() / ref.abs().max()).item()
    print(f"rank-{new_r} LoRA written to {o}; max rel. error vs exact merge on {ka}: {rel:.2e}")


if __name__ == "__main__":
    main()
