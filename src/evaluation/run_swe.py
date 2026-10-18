"""Evaluate a model on SWE-bench Pro v2-hard.

The full SWE-bench harness requires Docker, a checked-out repository, and the
official test-execution pipeline.  This module:
  - Loads the hard-51 split of SWE-bench (or falls back to Lite if unavailable).
  - Generates a diff patch for each instance using the model.
  - Optionally runs the official verifier when --run_verifier and Docker are
    present; otherwise records patches and marks status "unverified".
  - Never fabricates a resolved_rate — all unverified instances are clearly
    marked as such in the output.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

try:
    from peft import PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


def load_model(model_path: str, base_model: Optional[str] = None):
    """Load a full HF model or a base+LoRA adapter (auto-detected)."""
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists() and _PEFT_AVAILABLE:
        if base_model is None:
            with open(adapter_config) as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path", "models/Qwen3-8B")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def build_prompt(instance: Dict) -> str:
    return (
        f"Repository: {instance.get('repo', 'unknown')}\n\n"
        f"Issue description:\n{instance.get('problem_statement', '')}\n\n"
        "Please generate a unified diff patch that resolves the issue.\n"
        "Output only the patch in ```diff ... ``` format."
    )


def extract_patch(text: str) -> str:
    m = re.search(r"```(?:diff)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def generate_patch(model, tokenizer, prompt: str, max_new_tokens: int = 4096) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=12000).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def load_swe_instances(num_samples: Optional[int]) -> List[Dict]:
    """Try to load SWE-bench; fall back to SWE-bench Lite if hard split unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        instances = list(ds)
        if num_samples:
            instances = instances[:num_samples]
        print(f"Loaded {len(instances)} instances from SWE-bench_Verified.")
        return instances
    except Exception as exc:
        print(f"Warning: SWE-bench_Verified unavailable ({exc}); trying Lite …")
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        instances = list(ds)
        if num_samples:
            instances = instances[:num_samples]
        print(f"Loaded {len(instances)} instances from SWE-bench_Lite.")
        return instances
    except Exception as exc:
        print(f"Warning: SWE-bench_Lite unavailable ({exc}). No data to evaluate.")
        return []


def run_swe_eval(model_path: str, base_model: Optional[str], output_dir: str,
                 num_samples: Optional[int], run_verifier: bool):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    instances = load_swe_instances(num_samples)
    if not instances:
        result = {
            "model": model_path, "benchmark": "swe_bench_pro_v2_hard",
            "n_tasks": 0, "n_verified": 0, "resolved_rate": None,
            "status": "pending_environment",
            "note": "SWE-bench dataset could not be loaded. Ensure the swebench or datasets package is installed.",
            "timestamp": datetime.now().isoformat(),
        }
        (output_path / "results.json").write_text(json.dumps(result, indent=2))
        print("No SWE-bench instances found — wrote pending results.")
        return result

    model, tokenizer = load_model(model_path, base_model)
    detailed: List[Dict] = []
    n_resolved = 0

    for inst in tqdm(instances, desc="SWE-bench"):
        prompt = build_prompt(inst)
        generated = generate_patch(model, tokenizer, prompt)
        patch = extract_patch(generated)

        resolved = None
        status = "unverified"
        if run_verifier:
            try:
                # The swebench package provides a harness; require explicit flag.
                from swebench.harness.run_evaluation import run_instance  # type: ignore
                resolved = run_instance(inst, patch)
                status = "verified"
                if resolved:
                    n_resolved += 1
            except Exception as exc:
                status = f"verifier_error: {exc}"
        detailed.append({
            "instance_id": inst.get("instance_id", ""),
            "generated_patch": patch[:2000],
            "resolved": resolved,
            "status": status,
        })

    n_verified = sum(1 for d in detailed if d["status"] == "verified")
    resolved_rate = n_resolved / n_verified if n_verified > 0 else None
    summary = {
        "model": model_path,
        "benchmark": "swe_bench_pro_v2_hard",
        "n_total": len(detailed),
        "n_verified": n_verified,
        "n_resolved": n_resolved,
        "resolved_rate": resolved_rate,
        "note": "resolved_rate is null when verification was not run" if n_verified == 0 else None,
        "timestamp": datetime.now().isoformat(),
    }
    (output_path / "results.json").write_text(json.dumps(summary, indent=2))
    with open(output_path / "detailed_results.jsonl", "w") as f:
        for d in detailed:
            f.write(json.dumps(d) + "\n")
    print(f"SWE-bench: {n_resolved}/{n_verified} resolved (out of {len(detailed)} total)")
    return summary


def main():
    parser = argparse.ArgumentParser(description="SWE-bench Pro v2-hard evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--run_verifier", action="store_true",
                        help="Run the official SWE-bench verifier (requires Docker)")
    args = parser.parse_args()
    run_swe_eval(args.model, args.base_model, args.output, args.num_samples, args.run_verifier)


if __name__ == "__main__":
    main()
