"""Prepare SWE-bench training data."""

import json
import argparse
import hashlib
import random
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset

from src.data.common import write_splits


def create_swe_training_sample(instance: Dict) -> Dict:
    """Convert a SWE-bench-style instance to conversation training format."""
    context = f"Repository: {instance['repo']}\nIssue: {instance['problem_statement']}"
    if instance.get("hints_text"):
        context += f"\n\nHints: {instance['hints_text']}"
    patch = instance.get("patch", "")
    return {
        "id": instance["instance_id"],
        "messages": [
            {"role": "user", "content": context},
            {"role": "assistant", "content": f"Here is the patch to fix this issue:\n\n```diff\n{patch}\n```"},
        ],
        "source": "swebench_train",
        "split": "train",
        "repo": instance["repo"],
        "base_commit": instance.get("base_commit", ""),
    }


def collect_swebench_train(max_samples: int, max_chars: int = 12000) -> List[Dict]:
    """Issue -> gold patch pairs from the SWE-bench *train* split.

    The train split covers different repositories/instances from the SWE-bench
    test sets; SWE-bench Pro uses its own repositories. Overlap with the target
    evaluation is checked separately by audit_contamination.
    Long samples are skipped so they fit max_seq_length without truncating the patch.
    """
    examples = []
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench", split="train")
        for inst in dataset:
            if not inst.get("patch") or len(inst["problem_statement"]) + len(inst["patch"]) > max_chars:
                continue
            inst = dict(inst, hints_text="")  # hints can leak the fix discussion; keep prompt = issue only
            examples.append(create_swe_training_sample(inst))
            if len(examples) >= max_samples * 2:
                break
    except Exception as exc:
        print(f"Warning: could not load SWE-bench train split: {exc}")
    return examples


def collect_synthetic_repair_data(max_samples: int) -> List[Dict]:
    """Collect code-repair style training data from public datasets.

    NOTE: We deliberately do NOT use actual SWE-bench test instances here.
    The test set must remain untouched until final evaluation.
    """
    examples = []
    try:
        dataset = load_dataset("code_search_net", "python", split=f"train[:{max_samples}]")
        for item in dataset:
            doc = item.get("func_documentation_string", "")
            if len(doc) < 50:
                continue
            examples.append({
                "id": f"synthetic_{hashlib.md5(item['whole_func_string'].encode()).hexdigest()[:8]}",
                "messages": [
                    {"role": "user", "content": f"Write a Python function that: {doc}"},
                    {"role": "assistant", "content": item["whole_func_string"]},
                ],
                "source": "code_search_net_synthetic",
                "split": "train",
            })
    except Exception as exc:
        print(f"Warning: could not load CodeSearchNet: {exc}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare SWE expert training data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--include_synthetic", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    samples: List[Dict] = collect_swebench_train(args.max_samples)
    print(f"SWE-bench train samples: {len(samples)}")
    if args.include_synthetic:
        print("Collecting synthetic repair data…")
        samples.extend(collect_synthetic_repair_data(args.max_samples))

    write_splits(samples, output_path, args.max_samples)

if __name__ == "__main__":
    main()
