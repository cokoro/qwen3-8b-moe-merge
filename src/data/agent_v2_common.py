"""Shared helpers for the v2 multi-turn agent datasets (terminal_v2, swe_v2).

Every trajectory is a chat in the exact format the evaluation agent sends to the
model. ``train_expert.py`` trains only on the last assistant message, so a
trajectory with assistant turns t_1..t_n becomes up to ``MAX_TURNS`` samples:
prompt = all messages before t_k, completion = t_k (turns picked uniformly,
always including the final one). Samples are capped at ``MAX_TOKENS`` tokens
(Qwen3-8B tokenizer + qwen3_nothink chat template); older observations are
middle-truncated to fit, otherwise the sample is dropped.
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

MAX_TURNS = 4
MAX_TOKENS = 8192
TRUNC_MARK = "...[truncated]..."
NGRAM = 13


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

class TokenCounter:
    def __init__(self, model_dir: Path, template_path: Path):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.template = template_path.read_text()

    def render(self, messages: List[Dict], add_generation_prompt: bool = False) -> str:
        return self.tok.apply_chat_template(messages, chat_template=self.template, tokenize=False,
                                            add_generation_prompt=add_generation_prompt)

    def count(self, messages: List[Dict]) -> int:
        return len(self.tok(self.render(messages), add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------
# turn selection + length control
# ---------------------------------------------------------------------------

def pick_turns(candidates: List[int], k: int = MAX_TURNS) -> List[int]:
    """Up to k indices spread uniformly over ``candidates`` (first and last included)."""
    n = len(candidates)
    if n <= k:
        return list(candidates)
    picks = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [candidates[i] for i in picks]


def middle_truncate(text: str, keep: int) -> str:
    if len(text) <= 2 * keep + len(TRUNC_MARK) + 2:
        return text
    return text[:keep] + "\n" + TRUNC_MARK + "\n" + text[-keep:]


def fit_sample(messages: List[Dict], observation_idx: List[int], counter: TokenCounter,
               max_tokens: int = MAX_TOKENS) -> Tuple[Optional[List[Dict]], int, int]:
    """Truncate older observations (oldest first) until the sample fits.

    ``observation_idx`` are the indices of tool/terminal outputs in ``messages``;
    the most recent one (the observation the completion reacts to) is never cut.
    Returns (messages or None, n_tokens, n_truncated_messages).
    """
    n = counter.count(messages)
    if n <= max_tokens:
        return messages, n, 0
    older = sorted(observation_idx)[:-1]
    msgs = [dict(m) for m in messages]
    cut: Set[int] = set()
    for keep in (600, 150):
        for i in older:
            new = middle_truncate(msgs[i]["content"], keep)
            if new == msgs[i]["content"]:
                continue
            cut.add(i)
            msgs[i] = {**msgs[i], "content": new}
            n = counter.count(msgs)
            if n <= max_tokens:
                return msgs, n, len(cut)
    return None, n, len(cut)


# ---------------------------------------------------------------------------
# n-gram contamination helpers
# ---------------------------------------------------------------------------

def tokenize_words(text: str) -> List[str]:
    return text.lower().split()


def ngrams(text: str, n: int = NGRAM) -> Set[Tuple[str, ...]]:
    toks = tokenize_words(text)
    return set(zip(*[toks[i:] for i in range(n)])) if len(toks) >= n else set()


class NgramIndex:
    """13-gram index over a reference set (benchmark items)."""

    def __init__(self, docs: Dict[str, str], n: int = NGRAM):
        self.n = n
        self.grams = {k: ngrams(v, n) for k, v in docs.items()}
        self.index: Dict[Tuple[str, ...], Set[str]] = {}
        for k, gs in self.grams.items():
            for g in gs:
                self.index.setdefault(g, set()).add(k)

    def max_containment(self, text: str) -> Tuple[float, Optional[str]]:
        """max over refs of max(|A∩B|/|A|, |A∩B|/|B|), A = ref grams, B = text grams."""
        grams = ngrams(text, self.n)
        if not grams:
            return 0.0, None
        hits: Counter = Counter()
        for g in grams:
            for k in self.index.get(g, ()):
                hits[k] += 1
        best, best_k = 0.0, None
        for k, h in hits.items():
            c = max(h / max(1, len(self.grams[k])), h / len(grams))
            if c > best:
                best, best_k = c, k
        return best, best_k


# ---------------------------------------------------------------------------
# split + write
# ---------------------------------------------------------------------------

def take_by_quota(pools: Dict[str, List[str]], quotas: Dict[str, int],
                  build: Callable[[str, str], List[Dict]], used: Set[str]) -> List[Dict]:
    """Walk each (pre-shuffled) trajectory pool, building samples until the
    source's sample quota is met. Trajectory ids in ``used`` are skipped and the
    ones consumed are added to it (keeps train/validation trajectory-disjoint)."""
    out: List[Dict] = []
    for src, quota in quotas.items():
        got = 0
        for tid in pools[src]:
            if got >= quota:
                break
            if tid in used:
                continue
            samples = build(src, tid)
            if not samples:
                continue
            used.add(tid)
            out.extend(samples)
            got += len(samples)
    return out


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def token_stats(values: List[int]) -> Dict:
    if not values:
        return {}
    v = sorted(values)

    def q(p):
        return v[min(len(v) - 1, int(p * len(v)))]

    return {"mean": round(sum(v) / len(v), 1), "min": v[0], "p50": q(0.5), "p90": q(0.9),
            "p99": q(0.99), "max": v[-1]}


def shuffle(xs: List, seed: int) -> List:
    xs = list(xs)
    random.Random(seed).shuffle(xs)
    return xs
