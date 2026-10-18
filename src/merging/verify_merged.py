"""Smoke-test a merged model (or merged LoRA adapter) by loading it and running fixed inputs.

This is a sanity check that the merged checkpoint loads and generates coherent
text, not a benchmark. Run after every merge to catch a broken checkpoint
before spending GPU hours on full evaluation.
"""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SMOKE_PROMPTS = [
    "Explain what a diff is and write a small Python function that returns the patch for a one-line change.",
    "If a train travels 120 miles in 2 hours, then slows to half speed, how long to travel another 90 miles?",
    "Write the shell command to find all Python files modified in the last 24 hours.",
]


def load_model(model_path: str, base_model: str = None, adapter_path: str = None):
    """Load a full model, or a base model + LoRA adapter folded into weights."""
    if adapter_path:
        base_name = base_model or "models/Qwen3-8B"
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test a merged model")
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    if args.adapter_path and args.model_path:
        print("Pass either --adapter_path OR --model_path, not both.")
        sys.exit(2)

    if args.adapter_path:
        model, tokenizer = load_model(None, args.base_model, args.adapter_path)
        print(f"Loaded base={args.base_model} + adapter={args.adapter_path}")
    elif args.model_path:
        model, tokenizer = load_model(args.model_path)
        print(f"Loaded full model={args.model_path}")
    else:
        print("Provide --adapter_path (with --base_model) or --model_path.")
        sys.exit(2)

    all_ok = True
    for prompt in SMOKE_PROMPTS:
        text = generate(model, tokenizer, prompt, args.max_new_tokens)
        print(f"\n--- PROMPT ---\n{prompt}\n--- GENERATED ---\n{text}\n")
        if not text.strip():
            all_ok = False

    if all_ok:
        print("VERIFICATION PASSED")
        sys.exit(0)
    else:
        print("VERIFICATION FAILED: one or more generations were empty.")
        sys.exit(1)


if __name__ == "__main__":
    main()
