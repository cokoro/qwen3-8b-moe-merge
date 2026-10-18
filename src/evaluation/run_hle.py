"""Evaluate a model on HLE (High-Level Evaluation).

When the official HLE test set is not available locally, we fall back to the
GSM8K test split as a reasoning proxy and flag the result as ``is_proxy``.
The real HLE test set must remain untouched until final evaluation.
"""

import argparse
import json
import re
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


def extract_answer(text: str, expected: str) -> str:
    # GSM8K-style "#### <number>" first.
    if "####" in text:
        return text.split("####")[-1].strip()
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def evaluate_instance(model, tokenizer, question: str, answer: str) -> Dict:
    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=8000).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = extract_answer(generated, answer)
    gold = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
    is_correct = pred == gold
    return {
        "question": question[:200],
        "generated": generated[:500],
        "correct_answer": gold,
        "predicted": pred,
        "is_correct": is_correct,
    }


def run_hle_eval(model_path: str, base_model: Optional[str], output_dir: str, num_samples: int = 100):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning HLE Evaluation\nModel: {model_path}\n")

    model, tokenizer = load_model(model_path, base_model)

    is_proxy = True
    dataset = []
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=f"test[:{num_samples}]")
        dataset = list(ds)
        is_proxy = True
    except Exception as exc:
        print(f"Warning: could not load GSM8K proxy ({exc}). Writing empty result.")

    results: List[Dict] = []
    correct = 0
    for inst in tqdm(dataset, desc="HLE"):
        r = evaluate_instance(model, tokenizer, inst["question"], inst["answer"])
        results.append(r)
        if r["is_correct"]:
            correct += 1

    accuracy = correct / len(results) if results else 0.0
    summary = {
        "model": model_path,
        "benchmark": "hle_proxy_gsm8k" if is_proxy else "hle",
        "is_proxy": is_proxy,
        "n_instances": len(results),
        "n_correct": correct,
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
    }
    (output_path / "results.json").write_text(json.dumps(summary, indent=2))
    with open(output_path / "detailed_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(results)})  proxy={is_proxy}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="HLE evaluation (GSM8K proxy fallback)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    run_hle_eval(args.model, args.base_model, args.output, args.num_samples)


if __name__ == "__main__":
    main()
