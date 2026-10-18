"""Evaluate a model on Terminal-Bench.

Uses a fixed proxy task set of shell commands. The model's response is parsed
for shell commands (```bash blocks or `$ ` prefixed lines) and checked against
expected commands via case-insensitive substring match.

Note: this is a proxy task set pending integration with the official
Terminal-Bench harness.
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

TASKS = [
    {"description": "List all files in the current directory sorted by size.",
     "expected_commands": ["ls -lhS", "ls -l --sort=size"]},
    {"description": "Find all Python files modified in the last 24 hours.",
     "expected_commands": ["find . -name '*.py' -mtime -1"]},
    {"description": "Count the number of lines in all text files.",
     "expected_commands": ["find . -name '*.txt' -exec wc -l {} +", "wc -l *.txt"]},
    {"description": "Show the top 10 largest files in the current directory.",
     "expected_commands": ["du -ah | sort -rh | head -10", "find . -type f -exec du -h {} + | sort -rh | head -10"]},
    {"description": "Create a tarball backup of the current directory.",
     "expected_commands": ["tar -czf backup.tar.gz .", "tar -czvf backup.tar.gz ."]},
    {"description": "Find all processes using more than 50% CPU.",
     "expected_commands": ["ps aux | awk '$3 > 50'", "top -b -n 1"]},
    {"description": "Replace all occurrences of 'foo' with 'bar' in all Python files.",
     "expected_commands": ["find . -name '*.py' -exec sed -i 's/foo/bar/g' {} +", "sed -i 's/foo/bar/g'"]},
    {"description": "Show disk usage of all directories sorted by size.",
     "expected_commands": ["du -h --max-depth=1 | sort -h", "du -sh * | sort -h"]},
    {"description": "Kill all Python processes.",
     "expected_commands": ["pkill python", "killall python"]},
    {"description": "Monitor network connections in real-time.",
     "expected_commands": ["netstat -tuln", "ss -tuln", "watch netstat -tuln"]},
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


def extract_commands(text: str) -> List[str]:
    """Pull shell commands out of a model response."""
    code_blocks = re.findall(r"```(?:bash|shell|sh)?\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        cmds = []
        for block in code_blocks:
            for line in block.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    cmds.append(line)
        return cmds
    dollar = re.findall(r"\$\s+(.+?)(?:\n|$)", text)
    if dollar:
        return [c.strip() for c in dollar]
    return []


def evaluate_task(model, tokenizer, task: Dict) -> Dict:
    messages = [{"role": "user", "content": task["description"]}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4000).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    generated_commands = extract_commands(generated)

    correct = False
    for expected in task["expected_commands"]:
        for gen in generated_commands:
            if expected.lower() in gen.lower() or gen.lower() in expected.lower():
                correct = True
                break
        if correct:
            break

    return {
        "task": task["description"],
        "expected": task["expected_commands"],
        "generated": generated_commands,
        "full_response": generated[:500],
        "correct": correct,
    }


def run_terminal_eval(model_path: str, base_model: Optional[str], output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning Terminal-Bench Evaluation\nModel: {model_path}\n")
    model, tokenizer = load_model(model_path, base_model)

    results: List[Dict] = []
    correct = 0
    for task in tqdm(TASKS, desc="Terminal-Bench"):
        r = evaluate_task(model, tokenizer, task)
        results.append(r)
        if r["correct"]:
            correct += 1

    accuracy = correct / len(results) if results else 0.0
    summary = {
        "model": model_path,
        "benchmark": "terminal_bench",
        "n_tasks": len(results),
        "n_correct": correct,
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
    }
    (output_path / "results.json").write_text(json.dumps(summary, indent=2))
    with open(output_path / "detailed_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(results)})")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Terminal-Bench evaluation (proxy task set)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_terminal_eval(args.model, args.base_model, args.output)


if __name__ == "__main__":
    main()
