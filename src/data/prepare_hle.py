"""Prepare HLE (high-level reasoning) expert training data.

NOTE: This module uses GSM8K as a proxy/synthetic data source.
The real HLE test set must remain completely untouched until final evaluation.
Do NOT load or inspect HLE test instances here.
"""

import json
import argparse
import hashlib
import random
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset

from src.data.common import write_splits


def create_hle_training_sample(item: Dict, source: str) -> Dict:
    """Convert a reasoning dataset item to conversation training format."""
    question = item.get("question", "")
    answer = item.get("answer", "")
    sample_id = hashlib.md5(question.encode()).hexdigest()[:8]
    return {
        "id": f"hle_{source}_{sample_id}",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "source": f"hle_proxy_{source}",
        "split": "train",
    }


def collect_gsm8k_data(max_samples: int) -> List[Dict]:
    """Load GSM8K and convert to HLE-style chain-of-thought training examples.

    GSM8K is used as a proxy for high-level reasoning training data.
    The real HLE test set must stay untouched until final evaluation.
    """
    examples = []
    try:
        dataset = load_dataset("openai/gsm8k", "main", split=f"train[:{max_samples}]")
        for item in dataset:
            question = item.get("question", "")
            answer = item.get("answer", "")
            if not question or not answer:
                continue
            examples.append(create_hle_training_sample(
                {"question": question, "answer": answer},
                source="gsm8k",
            ))
    except Exception as exc:
        print(f"Warning: could not load GSM8K: {exc}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare HLE expert training data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=3000)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Collecting GSM8K proxy data for HLE expert…")
    samples: List[Dict] = collect_gsm8k_data(args.max_samples)

    write_splits(samples, output_path, args.max_samples)

if __name__ == "__main__":
    main()
