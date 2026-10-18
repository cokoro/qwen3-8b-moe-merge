"""Check delta-space merging against PEFT's own merge_and_unload.

Merging n copies of one adapter with weights summing to 1 (linear), or with
TIES at density 1.0, must reproduce that single adapter folded into the base.

    python -m tests.test_merge_equivalence --adapter checkpoints/smoke/expert_swe
"""

import argparse
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from src.merging.lora_delta import load_lora_factors, merge_into_base
from src.merging.merge_ties import ties_combine


def compare(merged_dir: Path, reference: dict, names):
    """Return (max abs diff, fraction of entries off by more than 1 bf16 ulp)."""
    worst, n_bad, n_total = 0.0, 0, 0
    for shard in merged_dir.glob("*.safetensors"):
        with safe_open(str(shard), "pt") as f:
            for name in f.keys():
                if name not in names:
                    continue
                a, b = f.get_tensor(name).float(), reference[name].float()
                diff = (a - b).abs()
                ulp = b.abs().clamp(min=1e-30) * 2.0 ** -7
                worst = max(worst, diff.max().item())
                n_bad += (diff > ulp * 1.01).sum().item()
                n_total += diff.numel()
    return worst, n_bad / max(n_total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base_model", default="models/Qwen3-8B")
    args = parser.parse_args()

    names = set(load_lora_factors(args.adapter))
    print(f"{len(names)} LoRA target weights")

    ref = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map={"": 0})
    ref = PeftModel.from_pretrained(ref, args.adapter).merge_and_unload()
    reference = {k: v.detach().cpu() for k, v in ref.state_dict().items() if k in names}
    base_ref = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    base_delta = max((reference[k].float() - base_ref.state_dict()[k].float()).abs().max().item() for k in names)
    del ref, base_ref
    torch.cuda.empty_cache()

    experts = [{"name": f"copy{i}", "adapter_path": args.adapter} for i in range(2)]
    cases = {
        "linear(0.5,0.5)": lambda ds: 0.5 * ds[0] + 0.5 * ds[1],
        "ties(density=1.0)": lambda ds: ties_combine(ds, 1.0),
    }
    ok = True
    for label, combine in cases.items():
        with tempfile.TemporaryDirectory(dir="checkpoints") as tmp:
            merge_into_base(args.base_model, None, experts, combine, Path(tmp))
            diff, frac_bad = compare(Path(tmp), reference, names)
        # fp32-vs-bf16 accumulation order may flip the last bf16 bit; nothing beyond 1 ulp
        passed = frac_bad == 0.0
        ok &= passed
        print(f"{label:20s} max|merged-peft|={diff:.3e} max|delta|={base_delta:.3e} "
              f">1ulp={frac_bad:.2e}  {'PASS' if passed else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
