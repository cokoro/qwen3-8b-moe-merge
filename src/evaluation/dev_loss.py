"""Dev-set selection of the final single model: completion-only loss of each candidate on the four experts'
held-out validation splits (data/<expert>/validation.jsonl, never used for training or checkpoint choice).
Score = equal-weight mean over the four domains of the per-token loss; lowest wins. No benchmark data used.

    python -m src.evaluation.dev_loss --models base=models/Qwen3-8B merged_linear=checkpoints/merged_linear \
        merged_ties=checkpoints/merged_ties mopd=checkpoints/full/expert_mopd --out analysis/dev_selection.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DOMAINS = ["swe", "hle", "asi", "terminal"]


def encode(tok, template, messages, max_len):
    prompt = tok.apply_chat_template(messages[:-1], chat_template=template, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(messages, chat_template=template, tokenize=False)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    f = tok(full, add_special_tokens=False)["input_ids"]
    n = 0
    while n < min(len(p), len(f)) and p[n] == f[n]:
        n += 1  # completion starts after the common prefix
    f = f[:max_len]
    labels = [-100] * min(n, len(f)) + f[n:]
    return f, labels


@torch.no_grad()
def domain_loss(model, tok, template, path, n_max, max_len):
    tot, cnt = 0.0, 0
    for i, line in enumerate(open(path)):
        if i >= n_max:
            break
        ids, labels = encode(tok, template, json.loads(line)["messages"], max_len)
        if sum(l != -100 for l in labels) < 2:
            continue
        x = torch.tensor([ids], device=model.device)
        y = torch.tensor([labels], device=model.device)
        logits = model(input_ids=x).logits[:, :-1].float()
        tgt = y[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                 ignore_index=-100, reduction="sum")
        tot += loss.item()
        cnt += int((tgt != -100).sum())
    return tot / max(cnt, 1), cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=100, help="validation samples per domain (first n, fixed order)")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--template", default="configs/chat_templates/qwen3_nothink.jinja")
    ap.add_argument("--tokenizer", default="models/Qwen3-8B")
    ap.add_argument("--out", default="analysis/dev_selection.json")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    template = Path(args.template).read_text()
    out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        if name in out:
            continue
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        res = {d: domain_loss(model, tok, template, Path(args.data) / d / "validation.jsonl", args.n, args.max_len)
               for d in DOMAINS}
        out[name] = {"path": path, **{d: {"loss": l, "tokens": c} for d, (l, c) in res.items()},
                     "mean_loss": sum(l for l, _ in res.values()) / len(DOMAINS)}
        print(name, {d: round(v[0], 4) for d, v in res.items()}, "mean", round(out[name]["mean_loss"], 4), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        del model
        torch.cuda.empty_cache()
    cands = {k: v["mean_loss"] for k, v in out.items() if k != "base"}
    if cands:
        best = min(cands, key=cands.get)
        out["_selected"] = {"model": best, "rule": "lowest equal-weight mean dev loss over 4 domains, base excluded"}
        Path(args.out).write_text(json.dumps(out, indent=2))
        print("selected", best)


if __name__ == "__main__":
    main()
