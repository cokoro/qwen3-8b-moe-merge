"""Prepare terminal_v2: multi-turn Terminus-2 trajectories for the Terminal-Bench 2.0 expert.

Terminal-Bench 2.0 is evaluated with harbor's ``terminus-2`` agent (JSON parser):
no system message, the first user message is ``terminus-json-plain.txt`` filled
with the task instruction and the initial screen, every assistant message is one
JSON object {analysis, plan, commands[{keystrokes, duration}], task_complete},
and every later user message is the raw terminal output (or harbor's
warning / completion-confirmation wrapper around it).

Sources (both generated with terminus-2 on non-TB tasks):
  * open-thoughts/OpenThoughts-Agent-v1-SFT (Apache-2.0): GLM-4.6 traces on
    nl2bash-style tasks and InferredBugs (C#/Java bug fixing).
  * nvidia/Nemotron-Terminal-Corpus (CC-BY-4.0), skill_based synthetic tasks:
    DeepSeek-V3.2 traces (``<think>`` blocks inside the content).

Normalisation to the evaluation format:
  * the first user message is rebuilt from harbor's current template (the
    sources used an older template that differs in one line);
  * assistant messages: ``<think>`` blocks and any prose around the JSON object
    are removed (the JSON substring is kept verbatim); the following
    observation's "Extra text detected" / "Extracted JSON from mixed content"
    warnings are dropped accordingly. Turns the agent answered with a parse
    error stay verbatim in the history and are never used as targets;
  * a turn is a training target only if harbor's own TerminusJSONPlainParser
    parses the cleaned JSON with no error and no warning;
  * trajectories must end with a target turn that has ``task_complete: true``.

Decontamination against all 89 Terminal-Bench 2.0 tasks
(third_party/terminal-bench/<task>/{instruction.md,solution/solve.sh}):
task-name match, instruction 13-gram containment >= 0.2, and solve.sh overlap
(a distinctive solve.sh command line typed verbatim, or 13-gram containment of
solve.sh in the trajectory's keystrokes >= 0.2).

Usage:
    python -m src.data.prepare_terminal_v2 [--output_dir data/terminal_v2]
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.data.agent_v2_common import (MAX_TOKENS, MAX_TURNS, NgramIndex, TokenCounter, fit_sample,
                                      ngrams, pick_turns, shuffle, take_by_quota, token_stats,
                                      write_jsonl)

ROOT = Path(__file__).resolve().parents[2]
HARBOR_T2 = ROOT / ".venv-harness/lib/python3.12/site-packages/harbor/agents/terminus_2"
RAW = ROOT / "data/_raw_v2"
TB2_DIR = ROOT / "third_party/terminal-bench"

OLD_LINE = "- Most bash commands should end with a newline (\\n) to cause them to execute"
NEW_LINE = "- You must end every command with a newline (\\n) or it will not execute."
STATE_SEP = "\n\nCurrent terminal state:\n"
WARN_HEAD = "Previous response had warnings:\nWARNINGS: "
ERR_HEAD = "Previous response had parsing errors:"
EXTRA_TEXT_WARNINGS = ("Extra text detected before JSON object", "Extra text detected after JSON object",
                       "AUTO-CORRECTED: Extracted JSON from mixed content")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

SOURCES = {
    # source -> fraction of the sample budget
    "nemotron_terminal_skill": 0.5,
    "ota_inferredbugs": 0.25,
    "ota_nl2bash": 0.25,
}
LICENSES = {
    "ota_nl2bash": "apache-2.0 (open-thoughts/OpenThoughts-Agent-v1-SFT)",
    "ota_inferredbugs": "apache-2.0 (open-thoughts/OpenThoughts-Agent-v1-SFT)",
    "nemotron_terminal_skill": "cc-by-4.0 (nvidia/Nemotron-Terminal-Corpus)",
}


def load_parser():
    spec = importlib.util.spec_from_file_location("terminus_json_plain_parser",
                                                  HARBOR_T2 / "terminus_json_plain_parser.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TerminusJSONPlainParser()


TEMPLATE = (HARBOR_T2 / "templates/terminus-json-plain.txt").read_text()
PREFIX = TEMPLATE.split("{instruction}")[0].replace("{{", "{").replace("}}", "}")
OLD_PREFIX = PREFIX.replace(NEW_LINE, OLD_LINE)
assert OLD_PREFIX != PREFIX


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_raw() -> Dict[str, Dict]:
    """trajectory id -> {source, task, conv}"""
    trajs: Dict[str, Dict] = {}
    df = pd.read_parquet(RAW / "ota_sft/data/train-00000-of-00001.parquet")
    for r in df.itertuples():
        src = "ota_inferredbugs" if r.task.startswith("inferredbugs") else "ota_nl2bash"
        trajs[f"ota::{r.trial_name}"] = {"source": src, "task": r.task, "conv": list(r.conversations)}
    for f in sorted(glob.glob(str(RAW / "nemotron_term/synthetic_tasks/skill_based/*/*/data_filtered.parquet"))):
        level, skill = Path(f).parts[-3], Path(f).parts[-2]
        df = pd.read_parquet(f)
        for r in df.itertuples():
            trajs[f"nemo::{level}/{skill}::{r.trial_name}"] = {
                "source": "nemotron_terminal_skill", "task": r.task, "conv": list(r.conversations)}
    return trajs


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

def split_first_user(content: str) -> Optional[Tuple[str, str]]:
    for pre in (PREFIX, OLD_PREFIX):
        if content.startswith(pre):
            body = content[len(pre):]
            if STATE_SEP not in body:
                return None
            instruction, state = body.split(STATE_SEP, 1)
            return instruction, state
    return None


def extract_json(text: str) -> Optional[str]:
    """First top-level JSON object in ``text`` that looks like a terminus response."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, end = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("commands" in obj or "analysis" in obj):
            return text[m.start():end]
    return None


def strip_extra_text_warnings(obs: str) -> str:
    if not obs.startswith(WARN_HEAD):
        return obs
    head, sep, rest = obs[len(WARN_HEAD):].partition("\n\n")
    if not sep:
        return obs
    lines = [l for l in head.split("\n") if not any(w in l for w in EXTRA_TEXT_WARNINGS)]
    if not lines:
        return rest
    return WARN_HEAD + "\n".join(lines) + "\n\n" + rest


def normalize(traj: Dict, parser) -> Optional[Dict]:
    """-> {messages, targets, instruction} or None if unusable."""
    conv = traj["conv"]
    if len(conv) < 2 or conv[0]["role"] != "user":
        return None
    roles = [m["role"] for m in conv]
    if any(r == roles[i - 1] for i, r in enumerate(roles) if i) or roles[-1] != "assistant":
        return None
    parts = split_first_user(conv[0]["content"])
    if parts is None:
        return None
    instruction, state = parts
    msgs = [{"role": "user", "content": TEMPLATE.format(instruction=instruction, terminal_state=state)}]
    targets = []
    last_complete = False
    for i in range(1, len(conv)):
        m = conv[i]
        if m["role"] == "user":
            msgs.append({"role": "user", "content": m["content"]})
            continue
        raw = THINK_RE.sub("", m["content"])
        if "</think>" in raw:  # unopened think block
            raw = raw.split("</think>", 1)[1]
        raw = raw.strip()
        nxt = conv[i + 1]["content"] if i + 1 < len(conv) else ""
        js = extract_json(raw)
        if nxt.startswith(ERR_HEAD) or js is None:
            msgs.append({"role": "assistant", "content": raw})
            last_complete = False
            continue
        if i + 1 < len(conv):
            conv[i + 1] = {**conv[i + 1], "content": strip_extra_text_warnings(nxt)}
        msgs.append({"role": "assistant", "content": js})
        res = parser.parse_response(js)
        ok = not res.error and not res.warning
        if ok:
            targets.append(len(msgs) - 1)
        last_complete = ok and res.is_task_complete
    if not last_complete or not targets or targets[-1] != len(msgs) - 1:
        return None
    return {"messages": msgs, "targets": targets, "instruction": instruction}


def keystrokes_text(msgs: List[Dict]) -> str:
    out = []
    for m in msgs:
        if m["role"] != "assistant":
            continue
        try:
            obj = json.loads(m["content"])
            out.extend(c.get("keystrokes", "") for c in obj.get("commands", []) if isinstance(c, dict))
        except (json.JSONDecodeError, AttributeError):
            pass
    return "\n".join(str(x) for x in out)


# ---------------------------------------------------------------------------
# decontamination
# ---------------------------------------------------------------------------

def norm_cmd(line: str) -> str:
    return " ".join(line.strip().split())


GENERIC_CMD = re.compile(r"^(set -|cd |mkdir |echo |cat |ls|pip3? install|apt(-get)? |python3? -m pip|"
                         r"export |source |chmod |rm |cp |mv |EOF|fi$|done$|then$|else$|}$|\)$)")


def load_tb2():
    tasks = {}
    for d in sorted(TB2_DIR.iterdir()):
        if (d / "instruction.md").exists():
            sol = d / "solution/solve.sh"
            tasks[d.name] = {"instruction": (d / "instruction.md").read_text(),
                             "solution": sol.read_text() if sol.exists() else ""}
    # distinctive solve.sh lines: >= 30 chars, not boilerplate, appear in exactly one task
    line_tasks: Dict[str, set] = {}
    for name, t in tasks.items():
        for line in t["solution"].splitlines():
            n = norm_cmd(line)
            if len(n) < 30 or n.startswith("#") or GENERIC_CMD.match(n):
                continue
            line_tasks.setdefault(n, set()).add(name)
    unique_lines = {l: next(iter(s)) for l, s in line_tasks.items() if len(s) == 1}
    return tasks, unique_lines


GENERIC_MIN_TASKS = 5


class Decontaminator:
    def __init__(self):
        self.tasks, self.unique_lines = load_tb2()
        self.names = set(self.tasks)
        self.instr_index = NgramIndex({k: v["instruction"] for k, v in self.tasks.items()})
        self.sol_index = NgramIndex({k: v["solution"] for k, v in self.tasks.items()})

    def drop_generic_lines(self, keys_by_task: Dict[str, str], min_tasks: int = GENERIC_MIN_TASKS) -> List[str]:
        """Distinctive-for-TB2 lines that are common idioms in the training corpus
        (typed in >= min_tasks distinct training tasks, e.g. ``import matplotlib.pyplot
        as plt``) carry no leakage signal; drop them before matching."""
        df = Counter()
        for keys in keys_by_task.values():
            for n in {norm_cmd(l) for l in keys.splitlines()}:
                if n in self.unique_lines:
                    df[n] += 1
        generic = sorted(l for l, c in df.items() if c >= min_tasks)
        for l in generic:
            del self.unique_lines[l]
        return generic

    def check(self, task_id: str, instruction: str, keys: str) -> Optional[Tuple[str, str]]:
        if task_id in self.names:
            return "task_name", task_id
        c, k = self.instr_index.max_containment(instruction)
        if c >= 0.2:
            return "instruction_13gram", f"{k}:{c:.2f}"
        for line in keys.splitlines():
            n = norm_cmd(line)
            if n in self.unique_lines:
                return "solution_unique_cmd", f"{self.unique_lines[n]}:{n[:80]}"
        if keys:
            c, k = self.sol_index.max_containment(keys)
            if c >= 0.2:
                return "solution_13gram", f"{k}:{c:.2f}"
        return None


# ---------------------------------------------------------------------------
# format check
# ---------------------------------------------------------------------------

def format_check(rows: List[Dict], counter: TokenCounter, parser, out: Path) -> Dict:
    """Render two samples (an intermediate and a final turn) to full chat text and
    machine-check the properties the eval agent relies on."""
    eval_first = ROOT / "results/_l40s_backup_20261018/tb2/jobs/qwen3-8b-base-r2-main/regex-log__75JBN32/agent/trajectory.json"
    checks = {
        "all_first_user_use_current_harbor_template": all(r["messages"][0]["content"].startswith(PREFIX) for r in rows),
        "no_system_message": all(m["role"] != "system" for r in rows for m in r["messages"]),
        "roles_alternate_user_assistant": all(r["messages"][i]["role"] != r["messages"][i - 1]["role"]
                                              for r in rows for i in range(1, len(r["messages"]))),
        "targets_parse_with_harbor_parser_no_error_no_warning": all(
            not (res.error or res.warning) for res in (parser.parse_response(r["messages"][-1]["content"]) for r in rows)),
    }
    if eval_first.exists():  # first prompt of a real TB2 eval run (base model, x86 harbor)
        msg = json.loads(eval_first.read_text())["steps"][0]["message"]
        checks["real_eval_first_prompt_has_same_template_prefix"] = msg.startswith(PREFIX)
    picks = [next(r for r in rows if r["turn"] == 2 and r["n_tokens"] < 2500),
             next(r for r in rows if r["turn"] == r["n_turns"] >= 3 and r["n_tokens"] < 3500)]
    texts = []
    for r in picks:
        full = counter.render(r["messages"])
        prompt = counter.render(r["messages"][:-1], add_generation_prompt=True)
        checks[f"{r['id']}_prompt_render_is_prefix_of_full"] = full.startswith(prompt)
        texts.append(f"===== {r['id']} ({r['source']}, turn {r['turn']}/{r['n_turns']}, {r['n_tokens']} tokens)\n{full}")
    (out / "format_check_samples.txt").write_text("\n\n".join(texts))
    return {"samples": [r["id"] for r in picks], "rendered": "format_check_samples.txt", "checks": checks,
            "conclusion": "Rendered text matches what terminus-2 sends through sglang with qwen3_nothink: "
                          "<|im_start|>user = harbor template (instruction + initial screen), assistant = bare "
                          "JSON object, later user turns = terminal output / harbor confirmation text, and the "
                          "completion turn is preceded by the same '<think>\\n\\n</think>\\n\\n' generation "
                          "prefix the server adds at inference."}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=str(ROOT / "data/terminal_v2"))
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    parser = load_parser()
    counter = TokenCounter(ROOT / "models/Qwen3-8B", ROOT / "configs/chat_templates/qwen3_nothink.jinja")
    deco = Decontaminator()
    raw = load_raw()
    print(f"raw trajectories: {len(raw)}  ({dict(Counter(t['source'] for t in raw.values()))})")

    normalised: Dict[str, Dict] = {}
    rejected = Counter()
    for tid, t in raw.items():
        n = normalize(t, parser)
        if n is None:
            rejected[t["source"]] += 1
            continue
        n["keys"] = keystrokes_text(n["messages"])
        normalised[tid] = n
    keys_by_task: Dict[str, str] = {}
    for tid, n in normalised.items():
        keys_by_task[raw[tid]["task"]] = keys_by_task.get(raw[tid]["task"], "") + "\n" + n["keys"]
    generic_lines = deco.drop_generic_lines(keys_by_task)
    print(f"distinctive TB2 solve.sh lines: {len(deco.unique_lines)} "
          f"(dropped {len(generic_lines)} corpus-common idioms)")

    norm: Dict[str, Dict] = {}
    contam = Counter()
    contam_examples: Dict[str, List[str]] = {}
    name_mentions = Counter()
    for tid, n in normalised.items():
        t = raw[tid]
        hit = deco.check(t["task"], n["instruction"], n.pop("keys"))
        if hit:
            contam[hit[0]] += 1
            contam_examples.setdefault(hit[0], [])
            if len(contam_examples[hit[0]]) < 10:
                contam_examples[hit[0]].append(f"{tid} -> {hit[1]}")
            continue
        low = n["instruction"].lower()
        for name in deco.names:
            if len(name) >= 8 and "-" in name and re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", low):
                name_mentions[name] += 1
        n["source"] = t["source"]
        norm[tid] = n
    print(f"normalised: {len(norm)}; rejected (format/not completed): {dict(rejected)}; "
          f"contaminated: {dict(contam)}")

    # dedup identical instructions (keep one trajectory per instruction)
    seen, dedup = set(), {}
    for tid in shuffle(sorted(norm), args.seed):
        key = " ".join(norm[tid]["instruction"].lower().split())
        if key in seen:
            continue
        seen.add(key)
        dedup[tid] = norm[tid]
    print(f"after instruction dedup: {len(dedup)}")

    length_drop = Counter()
    trunc_count = Counter()

    def build(src: str, tid: str) -> List[Dict]:
        n = dedup[tid]
        msgs = n["messages"]
        samples = []
        for k in pick_turns(n["targets"], MAX_TURNS):
            prefix = msgs[:k + 1]
            obs = [i for i, m in enumerate(prefix) if m["role"] == "user" and i > 0]
            fitted, ntok, ncut = fit_sample(prefix, obs, counter, MAX_TOKENS)
            if fitted is None:
                length_drop[src] += 1
                continue
            trunc_count["samples_truncated"] += bool(ncut)
            trunc_count["messages_truncated"] += ncut
            turn = sum(1 for m in prefix if m["role"] == "assistant")
            samples.append({"id": f"tv2_{hashlib.sha1(tid.encode()).hexdigest()[:10]}_t{turn:02d}",
                            "messages": fitted,
                            "source": src, "traj_id": tid, "turn": turn,
                            "n_turns": sum(1 for m in msgs if m["role"] == "assistant"),
                            "n_tokens": ntok})
        return samples

    # prefer trajectories with >= MAX_TURNS target turns (multi-turn supervision)
    pools = {s: [] for s in SOURCES}
    for tid in shuffle(sorted(dedup), args.seed):
        pools[dedup[tid]["source"]].append(tid)
    for s in pools:
        pools[s].sort(key=lambda t: len(dedup[t]["targets"]) < MAX_TURNS)  # stable: keeps shuffle order

    used: set = set()
    val = take_by_quota(pools, {s: round(f * args.n_val) for s, f in SOURCES.items()}, build, used)
    train = take_by_quota(pools, {s: round(f * args.n_train) for s, f in SOURCES.items()}, build, used)
    assert not ({s["traj_id"] for s in val} & {s["traj_id"] for s in train})

    for name, rows in (("train", train), ("validation", val)):
        rows = shuffle(rows, args.seed)
        write_jsonl(out / f"{name}.jsonl", [{**r, "split": name} for r in rows])

    stats = {
        "description": "Multi-turn terminus-2 (harbor, JSON parser) trajectories; one sample per "
                       "selected assistant turn (prompt = all previous messages, completion = that "
                       "turn), <= 4 turns per trajectory picked uniformly incl. the final "
                       "task_complete turn. No system message, matching terminus-2.",
        "eval_format": {
            "agent": "harbor terminus-2, parser=json",
            "first_user_template": str(HARBOR_T2.relative_to(ROOT) / "templates/terminus-json-plain.txt"),
            "assistant": "one JSON object {analysis, plan, commands[{keystrokes,duration}], task_complete}",
            "observations": "raw terminal output / harbor warning + completion-confirmation wrappers",
            "target_validation": "harbor TerminusJSONPlainParser: no error, no warning",
        },
        "sources": {s: {"license": LICENSES[s],
                        "train_samples": sum(r["source"] == s for r in train),
                        "validation_samples": sum(r["source"] == s for r in val),
                        "trajectories": len({r["traj_id"] for r in train + val if r["source"] == s})}
                    for s in SOURCES},
        "raw_trajectories": len(raw),
        "rejected_format_or_incomplete": dict(rejected),
        "after_normalisation_and_decontam": len(norm),
        "after_instruction_dedup": len(dedup),
        "train": len(train),
        "validation": len(val),
        "train_trajectories": len({r["traj_id"] for r in train}),
        "validation_trajectories": len({r["traj_id"] for r in val}),
        "tokens_train": token_stats([r["n_tokens"] for r in train]),
        "tokens_validation": token_stats([r["n_tokens"] for r in val]),
        "turn_position": dict(Counter("final" if r["turn"] == r["n_turns"] else "intermediate"
                                      for r in train)),
        "length_control": {"max_tokens": MAX_TOKENS, **dict(trunc_count),
                           "samples_dropped_too_long": dict(length_drop)},
        "decontamination": {
            "reference": "Terminal-Bench 2.0, 89 tasks (third_party/terminal-bench, identical task list to "
                         "the x86 tb2-mirror)",
            "criteria": ["task name == TB2 task name",
                         "instruction 13-gram containment >= 0.2 (max of both directions)",
                         "any distinctive TB2 solve.sh line (>=30 chars, non-boilerplate, unique to one "
                         "task) typed verbatim in keystrokes",
                         "13-gram containment of solve.sh vs trajectory keystrokes >= 0.2"],
            "distinctive_solution_lines": len(deco.unique_lines),
            "solution_lines_dropped_as_corpus_idioms": {
                "rule": f"typed in >= {GENERIC_MIN_TASKS} distinct training tasks",
                "count": len(generic_lines), "examples": generic_lines[:20]},
            "removed_trajectories": dict(contam),
            "removed_total": sum(contam.values()),
            "examples": contam_examples,
            "tb2_name_mentions_kept_for_review": dict(name_mentions.most_common(20)),
        },
        "tokenizer": "models/Qwen3-8B + configs/chat_templates/qwen3_nothink.jinja",
        "seed": args.seed,
    }
    stats["format_check"] = format_check(train, counter, parser, out)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: stats[k] for k in ("train", "validation", "tokens_train", "decontamination")},
                     indent=1)[:3000])


if __name__ == "__main__":
    main()
