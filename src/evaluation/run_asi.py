"""Evaluate a model on ASI-Bench (agent/tool-use tasks).

Uses a fixed proxy task set — each task has a description and a list of
required keywords that the model's plan must contain. Success requires all
keywords present in the generated text (case-insensitive).

Note: this is a small proxy task set pending integration with the official
ASI-Bench evaluation harness.
"""

import argparse
import json
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

TASKS = [
    {"description": "Find all Python files in the current directory and count total lines of code.",
     "steps": ["find", ".py", "wc"]},
    {"description": "Search for a recent paper on arxiv about transformer models and summarize it.",
     "steps": ["search", "arxiv", "transformer"]},
    {"description": "List all running processes and find the one using the most CPU.",
     "steps": ["ps", "cpu", "sort"]},
    {"description": "Create a compressed backup of all configuration files.",
     "steps": ["backup", "config", "tar"]},
    {"description": "Monitor disk usage and report any partition over 80% full.",
     "steps": ["disk", "usage", "df"]},
    {"description": "Fetch JSON data from an API endpoint and extract a nested field.",
     "steps": ["curl", "json", "jq"]},
    {"description": "Rename all .txt files in a directory to .md.",
     "steps": ["rename", ".txt", ".md"]},
    {"description": "Schedule a cron job to run a script every day at midnight.",
     "steps": ["cron", "midnight", "script"]},
    {"description": "Compare two directories and list files that differ.",
     "steps": ["diff", "directory", "compare"]},
    {"description": "Set up a Python virtual environment and install dependencies from requirements.txt.",
     "steps": ["venv", "pip", "requirements"]},
]


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


def evaluate_task(model, tokenizer, task: Dict) -> Dict:
    messages = [{"role": "user", "content": task["description"]}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=8000).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    text_lower = generated.lower()
    success = all(kw.lower() in text_lower for kw in task["steps"])
    return {
        "task": task["description"][:200],
        "generated": generated[:1000],
        "expected_steps": task["steps"],
        "success": success,
    }


def run_asi_eval(model_path: str, base_model: Optional[str], output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning ASI-Bench Evaluation\nModel: {model_path}\n")
    model, tokenizer = load_model(model_path, base_model)

    results: List[Dict] = []
    successful = 0
    for task in tqdm(TASKS, desc="ASI-Bench"):
        r = evaluate_task(model, tokenizer, task)
        results.append(r)
        if r["success"]:
            successful += 1

    success_rate = successful / len(results) if results else 0.0
    summary = {
        "model": model_path,
        "benchmark": "asi_bench",
        "n_tasks": len(results),
        "n_successful": successful,
        "success_rate": success_rate,
        "timestamp": datetime.now().isoformat(),
    }
    (output_path / "results.json").write_text(json.dumps(summary, indent=2))
    with open(output_path / "detailed_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Success Rate: {success_rate:.2%} ({successful}/{len(results)})")
    return summary


def main():
    parser = argparse.ArgumentParser(description="ASI-Bench evaluation (proxy task set)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_asi_eval(args.model, args.base_model, args.output)


if __name__ == "__main__":
    main()
