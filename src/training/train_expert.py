"""Train a single bf16 LoRA expert on Qwen3-8B.

Supports four expert domains: swe, hle, asi, terminal.
Run as:
    python src/training/train_expert.py --expert swe
    python src/training/train_expert.py --expert hle --report_to none
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def load_expert_dataset(data_path: str, max_samples: Optional[int] = None) -> Dataset:
    """Read a JSONL file of chat samples into conversational prompt/completion form.

    The final assistant turn is the completion; everything before it is the
    prompt. With completion_only_loss the loss is computed on the answer only.
    """
    prompts: List[list] = []
    completions: List[list] = []
    with open(data_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line).get("messages", [])
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                continue
            prompts.append(messages[:-1])
            completions.append(messages[-1:])
            if max_samples and len(prompts) >= max_samples:
                break
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


def print_trainable_params(model: Any) -> None:
    """Print the number of trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a single LoRA expert")
    parser.add_argument(
        "--expert",
        required=True,
        choices=["swe", "hle", "asi", "terminal", "distill"],
        help="Which expert domain to train",
    )
    parser.add_argument(
        "--config",
        default="configs/training.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--experiment_config",
        default="configs/experiment.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap training samples (smoke tests)")
    parser.add_argument(
        "--report_to",
        default=None,
        help="Override report_to (e.g. 'none' to disable wandb)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load configs, build model, train, and save the LoRA adapter."""
    args = parse_args()

    with open(args.config) as fh:
        train_cfg = yaml.safe_load(fh)
    with open(args.experiment_config) as fh:
        exp_cfg = yaml.safe_load(fh)

    t_cfg = train_cfg.get("training", {})
    lora_cfg_vals = train_cfg.get("lora", {})
    expert_cfg = exp_cfg["experts"][args.expert]

    seed = exp_cfg.get("seed", t_cfg.get("seed", 42))
    set_seed(seed)

    base_model_id: str = exp_cfg["base_model"]
    base_revision: Optional[str] = exp_cfg.get("base_revision")
    print(f"Loading base model: {base_model_id}@{base_revision}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        revision=base_revision,
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation=exp_cfg.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, revision=base_revision, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=lora_cfg_vals.get("r", 16),
        lora_alpha=lora_cfg_vals.get("lora_alpha", 32),
        lora_dropout=lora_cfg_vals.get("lora_dropout", 0.05),
        bias=lora_cfg_vals.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg_vals.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model = get_peft_model(model, lora_config)
    print_trainable_params(model)

    # Load train / validation data
    data_path = expert_cfg["data_path"]
    val_path = Path(data_path).with_name("validation.jsonl")
    max_seq_length: int = t_cfg.get("max_seq_length", 4096)
    print(f"Loading expert data from: {data_path}")
    train_dataset = load_expert_dataset(data_path, args.max_train_samples)
    eval_dataset = None
    if val_path.exists():
        eval_dataset = load_expert_dataset(str(val_path), t_cfg.get("max_eval_samples", 200))
        print(f"Validation samples: {len(eval_dataset)} ({val_path})")
    print(f"Training samples: {len(train_dataset)}")

    output_dir_base = args.output_dir or t_cfg.get("output_dir", "checkpoints")
    output_dir = str(Path(output_dir_base) / f"expert_{args.expert}")
    report_to: str = args.report_to or t_cfg.get("report_to", "none")
    max_steps: int = args.max_steps or t_cfg.get("max_steps", 500)
    eval_strategy = t_cfg.get("eval_strategy", "steps") if eval_dataset is not None else "no"

    from trl import SFTConfig, SFTTrainer

    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=t_cfg.get("num_train_epochs", 1),
        max_steps=max_steps,
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=t_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 8),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=t_cfg.get("learning_rate", 2e-4),
        weight_decay=t_cfg.get("weight_decay", 0.01),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=t_cfg.get("logging_steps", 10),
        save_steps=t_cfg.get("save_steps", 100),
        eval_strategy=eval_strategy,
        eval_steps=t_cfg.get("eval_steps", 100),
        save_total_limit=t_cfg.get("save_total_limit", 3),
        bf16=t_cfg.get("bf16", True),
        max_grad_norm=t_cfg.get("max_grad_norm", 0.3),
        optim=t_cfg.get("optim", "adamw_torch_fused"),
        seed=seed,
        report_to=report_to,
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 2),
        max_length=max_seq_length,
        packing=t_cfg.get("packing", False),
        completion_only_loss=t_cfg.get("completion_only_loss", True),
        include_num_input_tokens_seen=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    start_iso = datetime.now(timezone.utc).isoformat()
    wall_start = time.monotonic()
    print(f"Starting training for expert '{args.expert}' — {start_iso}")
    train_result = trainer.train()
    wall_elapsed = time.monotonic() - wall_start
    end_iso = datetime.now(timezone.utc).isoformat()

    # Save adapter and tokenizer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"Adapter saved to {output_path}")

    # Collect metrics
    tm = train_result.metrics
    final_eval: Dict[str, Any] = trainer.evaluate() if eval_dataset is not None else {}
    tokens_seen = getattr(trainer.state, "num_input_tokens_seen", None)
    adapter_bytes = sum(f.stat().st_size for f in output_path.glob("adapter_model*"))
    peak_mem: Optional[float] = None
    gpu_count: Optional[int] = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        gpu_count = torch.cuda.device_count()

    metrics: Dict[str, Any] = {
        "expert": args.expert,
        "base_model": base_model_id,
        "base_revision": base_revision,
        "method": train_cfg.get("method", "lora"),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "max_seq_length": max_seq_length,
        "train_loss": tm.get("train_loss"),
        "final_eval_loss": final_eval.get("eval_loss"),
        "num_input_tokens_seen": tokens_seen,
        "tokens_per_second": (tokens_seen / wall_elapsed) if tokens_seen else None,
        "adapter_bytes": adapter_bytes,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "log_history": trainer.state.log_history,
        "seed": seed,
        "max_steps": max_steps,
        "train_runtime": tm.get("train_runtime"),
        "train_samples_per_second": tm.get("train_samples_per_second"),
        "total_flos": tm.get("total_flos"),
        "peak_memory_gb": peak_mem,
        "elapsed_seconds": round(wall_elapsed, 2),
        "timestamp": end_iso,
        "gpu_count": gpu_count,
        "start_iso": start_iso,
        "end_iso": end_iso,
    }
    metrics_path = output_path / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics written to {metrics_path}")

    print(
        f"\nTraining complete — expert={args.expert}, "
        f"steps={max_steps}, elapsed={wall_elapsed:.0f}s, "
        f"peak_gpu={f'{peak_mem:.2f} GB' if peak_mem is not None else 'n/a'}"
    )


if __name__ == "__main__":
    main()
