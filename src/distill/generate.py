"""Multi-teacher distillation data: each expert answers prompts from its own domain,
and only answers that pass a domain check are kept for training one student.

The idea (see report): instead of merging expert *weights*, keep what the experts
reliably agree on / get right and distill that into a single model.

Domain checks (no human-written answer is copied; the reference is only used to verify):
  hle       final number matches the reference ("#### N")             [verified]
  asi       "Answer: X" letter matches the reference                  [verified]
  terminal  bash command equals the reference after whitespace normalisation, or
            both teacher samples agree exactly                         [verified / consensus]
  swe       a unified diff touching at least one file the reference patch touches
                                                                        [weakly verified]

    python -m src.distill.generate --domain hle --teacher expert_hle \
        --base_url http://127.0.0.1:30001/v1 --n_prompts 1500 --out data/distill/hle.jsonl
"""

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

from openai import AsyncOpenAI

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
BASH = re.compile(r"```(?:bash|sh)?\n(.*?)```", re.S)
DIFF_FILE = re.compile(r"^diff --git a/(\S+)", re.M)


def last_number(text: str):
    if "####" in text:
        text = text.split("####")[-1]
    nums = NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def answer_letter(text: str):
    m = re.findall(r"Answer:\s*([A-E])\b", text)
    return m[-1] if m else None


def bash_cmd(text: str):
    m = BASH.findall(text)
    return " ".join(m[-1].split()) if m else None


def check(domain: str, gold: str, outs: list):
    """Return (kept_output or None, tag)."""
    if domain == "hle":
        ref = last_number(gold)
        for o in outs:
            if ref is not None and last_number(o) == ref:
                return o, "verified"
    elif domain == "asi":
        ref = answer_letter(gold)
        for o in outs:
            if ref and answer_letter(o) == ref:
                return o, "verified"
    elif domain == "terminal":
        ref = bash_cmd(gold)
        cmds = [bash_cmd(o) for o in outs]
        for o, c in zip(outs, cmds):
            if c and ref and c == ref:
                return o, "verified"
        if len(cmds) >= 2 and cmds[0] and all(c == cmds[0] for c in cmds):
            return outs[0], "consensus"
    elif domain == "swe":
        ref_files = set(DIFF_FILE.findall(gold))
        for o in outs:
            if ref_files & set(DIFF_FILE.findall(o)):
                return o, "file_overlap"
    return None, "rejected"


async def main_async(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=900)
    rows = [json.loads(l) for l in open(f"data/{args.domain}/train.jsonl") if l.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n_prompts]
    sem = asyncio.Semaphore(args.concurrency)
    stats = {"verified": 0, "consensus": 0, "file_overlap": 0, "rejected": 0, "error": 0}

    async def one(row):
        prompt, gold = row["messages"][:-1], row["messages"][-1]["content"]
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=args.teacher, messages=prompt, n=args.samples, temperature=args.temperature,
                    top_p=0.95, max_tokens=args.max_tokens, seed=args.seed)
            except Exception as exc:  # keep going; count it
                stats["error"] += 1
                return None
        outs = [c.message.content or "" for c in r.choices if c.finish_reason == "stop"]
        kept, tag = check(args.domain, gold, outs)
        stats[tag] += 1
        if kept is None:
            return None
        return {"id": f"distill_{args.domain}_{row['id']}", "messages": prompt + [{"role": "assistant", "content": kept}],
                "source": f"distill_{args.teacher}", "check": tag, "split": "train"}

    results = await asyncio.gather(*(one(r) for r in rows))
    kept = [r for r in results if r]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    report = {"domain": args.domain, "teacher": args.teacher, "n_prompts": len(rows), "samples": args.samples,
              "temperature": args.temperature, "kept": len(kept), **stats}
    Path(args.out).with_suffix(".stats.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=["hle", "asi", "terminal", "swe"])
    p.add_argument("--teacher", required=True, help="served model name of the expert")
    p.add_argument("--base_url", default="http://127.0.0.1:30001/v1")
    p.add_argument("--n_prompts", type=int, default=1500)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
