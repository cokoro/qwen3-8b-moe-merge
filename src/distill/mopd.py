"""MOPD: multi-teacher on-policy distillation into one student LoRA.

Alternative to weight merging: instead of averaging expert parameters, the student
samples its own answers (on-policy) and each answer is supervised token by token by
the expert of that prompt's domain, minimising the reverse KL

    L = mean_t  KL( p_student(. | x_<t) || p_teacher_d(. | x_<t) )

computed exactly over the vocabulary on the student-sampled response tokens.

All models share the frozen Qwen3-8B base, so one model holds every LoRA: the trainable
``student`` adapter and the frozen teacher adapters (one per domain). Teachers are
evaluated with ``set_adapter(<domain>)`` under no_grad; the student with gradients.

    python -m src.distill.mopd --teachers hle=checkpoints/expert_hle asi=checkpoints/expert_asi \
        --output checkpoints/expert_mopd --max_minutes 150
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(domain: str, n: int, seed: int):
    rows = [json.loads(l) for l in open(f"data/{domain}/train.jsonl") if l.strip()]
    random.Random(seed).shuffle(rows)
    return [r["messages"][:-1] for r in rows[:n]]


def reverse_kl(student_logits, teacher_logits, mask, chunk=128):
    """Sum over masked positions of KL(p_s || p_t), computed in fp32 in sequence chunks."""
    total = student_logits.new_zeros((), dtype=torch.float32)
    for s in range(0, student_logits.shape[1], chunk):
        ls = F.log_softmax(student_logits[:, s:s + chunk].float(), dim=-1)
        lt = F.log_softmax(teacher_logits[:, s:s + chunk].float(), dim=-1)
        kl = (ls.exp() * (ls - lt)).sum(-1)
        total = total + (kl * mask[:, s:s + chunk]).sum()
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teachers", nargs="+", required=True, help="domain=adapter_path")
    p.add_argument("--experiment_config", default="configs/experiment.yaml")
    p.add_argument("--output", default="checkpoints/expert_mopd")
    p.add_argument("--prompts_per_domain", type=int, default=2000)
    p.add_argument("--batch_per_domain", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=1.0, help="on-policy sampling temperature")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--max_minutes", type=float, default=150)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--student_base", default=None,
                   help="base for the student (default: the teachers' base). A larger model from the same "
                        "tokenizer family gives small-experts -> bigger-student distillation.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    exp = yaml.safe_load(open(args.experiment_config))
    lcfg = yaml.safe_load(open("configs/training.yaml"))["lora"]
    teachers = dict(t.split("=", 1) for t in args.teachers)
    domains = list(teachers)

    student_base = args.student_base or exp["base_model"]
    tok = AutoTokenizer.from_pretrained(student_base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.student_base:
        assert AutoTokenizer.from_pretrained(exp["base_model"]).get_vocab() == tok.get_vocab(), \
            "student and teachers must share the tokenizer for token-level KL"
    base = AutoModelForCausalLM.from_pretrained(exp["base_model"], dtype=torch.bfloat16,
                                                device_map={"": 0}, attn_implementation="sdpa")
    first = domains[0]
    tmodel = PeftModel.from_pretrained(base, teachers[first], adapter_name=first)
    for d in domains[1:]:
        tmodel.load_adapter(teachers[d], adapter_name=d)
    student_cfg = LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["lora_alpha"], lora_dropout=0.0, bias="none",
        target_modules=lcfg["target_modules"], task_type="CAUSAL_LM")
    if args.student_base:
        tmodel.eval()
        for prm in tmodel.parameters():
            prm.requires_grad = False
        sbase = AutoModelForCausalLM.from_pretrained(student_base, dtype=torch.bfloat16,
                                                     device_map={"": 0}, attn_implementation="sdpa")
        smodel = get_peft_model(sbase, student_cfg, adapter_name="student")
    else:  # shared base: one model holds teachers + student
        tmodel.add_adapter("student", student_cfg)
        smodel = tmodel
    model = smodel
    model.set_adapter("student")
    for n, prm in model.named_parameters():
        prm.requires_grad = ".student." in n
    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    print(f"student_base={student_base} teachers={teachers} trainable params={sum(t.numel() for t in trainable):,}")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    pools = {d: load_prompts(d, args.prompts_per_domain, args.seed) for d in domains}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "mopd_log.jsonl", "a")
    start = time.time()

    def save(step, final=False):
        tmp = out / "_save"
        model.save_pretrained(tmp, selected_adapters=["student"])
        src = tmp / "student" if (tmp / "student").exists() else tmp
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy(src / f, out / f)
        shutil.rmtree(tmp, ignore_errors=True)
        tok.save_pretrained(out)
        (out / "training_metrics.json").write_text(json.dumps({
            "method": "mopd", "teachers": teachers, "steps": step, "elapsed_seconds": time.time() - start,
            "batch_per_domain": args.batch_per_domain, "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature, "lr": args.lr, "base_model": student_base,
            "teacher_base_model": exp["base_model"],
            "base_revision": None if args.student_base else exp.get("base_revision"),
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9, "final": final}, indent=2))

    step = 0
    while step < args.max_steps and (time.time() - start) / 60 < args.max_minutes:
        t0 = time.time()
        # ---- on-policy rollouts with the current student
        model.set_adapter("student")
        model.eval()
        batch = []  # (domain, prompt_ids, response_ids)
        for d in domains:
            msgs = random.sample(pools[d], args.batch_per_domain)
            texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for m in msgs]
            enc = tok(texts, return_tensors="pt", padding=True).to(0)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=1.0,
                                     max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
                                     use_cache=True)
            plen = enc["input_ids"].shape[1]
            for i in range(len(msgs)):
                prompt_ids = enc["input_ids"][i][enc["attention_mask"][i].bool()]
                resp = gen[i, plen:]
                stop = (resp == tok.eos_token_id) | (resp == tok.pad_token_id)
                end = int(stop.nonzero()[0]) + 1 if stop.any() else resp.shape[0]
                batch.append((d, prompt_ids, resp[:end]))
        t_gen = time.time() - t0

        # ---- teacher-supervised reverse KL on the student's own tokens
        model.train()
        opt.zero_grad(set_to_none=True)
        stats = {}
        n_tokens = sum(len(r) for _, _, r in batch)
        for d in domains:
            items = [(pi, r) for dd, pi, r in batch if dd == d]
            seqs = [torch.cat([pi, r]) for pi, r in items]
            L = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), L), tok.pad_token_id, device=0)
            att = torch.zeros((len(seqs), L), device=0, dtype=torch.long)
            mask = torch.zeros((len(seqs), L - 1), device=0)
            for i, ((pi, r), s) in enumerate(zip(items, seqs)):
                ids[i, :len(s)] = s
                att[i, :len(s)] = 1
                mask[i, len(pi) - 1:len(s) - 1] = 1  # positions predicting response tokens
            with torch.no_grad():
                tmodel.eval()  # no LoRA dropout on the teacher
                tmodel.set_adapter(d)
                t_logits = tmodel(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1]
            model.set_adapter("student")
            model.train()
            for n, prm in model.named_parameters():  # set_adapter may reset trainability
                prm.requires_grad = ".student." in n
            s_logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1]
            kl_sum = reverse_kl(s_logits, t_logits, mask)
            (kl_sum / max(n_tokens, 1)).backward()
            stats[d] = {"kl_per_token": kl_sum.item() / max(mask.sum().item(), 1),
                        "mean_resp_len": mask.sum().item() / len(seqs)}
            del t_logits, s_logits
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1
        rec = {"step": step, "elapsed_min": round((time.time() - start) / 60, 2),
               "gen_s": round(t_gen, 1), "step_s": round(time.time() - t0, 1), **stats}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if step % args.save_every == 0:
            save(step)
    save(step, final=True)
    print(f"MOPD done: {step} steps in {(time.time() - start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
