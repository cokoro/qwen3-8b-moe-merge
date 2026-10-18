"""Prepare swe_v2: multi-turn mini-swe-agent (tool-call) trajectories for the SWE-bench Pro expert.

SWE-bench Pro v2 is evaluated with ``host_mini_swe`` + ``mini_toolcall.yaml``
behind sglang (qwen3_nothink template, qwen25 tool parser):
  * system = mini's system_template, plus the single ``bash`` tool that sglang
    renders into the system turn;
  * user = mini's instance_template with task = the SWE-bench Pro v2
    ``instruction.md`` wrapper around the PR description;
  * every assistant turn = reasoning text + exactly one ``bash`` tool call;
  * every tool result = mini's observation_template (JSON with returncode/output);
  * the run ends with the tool call ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``.

The trainer (``train_expert.py``) does not pass ``tools`` to the chat template, so
the tool block is stored inside the system message content exactly as the Qwen3
template renders it (rendering with tools=[bash] and rendering our system
content without tools give identical text; checked in ``format_check``).

Source: SWE-bench/SWE-smith-trajectories (MIT), ``tool`` split: SWE-agent +
Claude 3.7 Sonnet on SWE-smith task instances, only ``resolved`` trajectories.
SWE-agent actions are converted to one bash command each:
  bash cmd                        -> cmd (``/testbed`` -> ``/app``)
  str_replace_editor view file    -> ``cat -n file`` / ``nl -ba file | sed -n 'a,bp'``
                                     (same line format as SWE-agent's view)
  str_replace_editor view dir     -> ``find dir -maxdepth 2 -not -path '*/.*'``
                                     (the command SWE-agent itself runs)
  str_replace_editor create       -> ``cat <<'EOF' > file`` heredoc
  str_replace_editor str_replace  -> python3 heredoc doing an exactly-once
                                     replace and printing the same cat -n snippet
                                     SWE-agent showed (failed replaces become the
                                     script's AssertionError traceback, rc=1)
  first ``submit`` (review page)  -> removed together with its observation
  final ``submit``                -> ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``
  abbreviated large-file view     -> step removed (output cannot be reproduced)
  insert / undo_edit / timeouts / other editor errors / no tool call -> trajectory dropped
Return codes are not recorded by SWE-agent; they are inferred from the output
(0 by default, 1 for tracebacks / failed pytest runs / missing files, 127 for
"command not found").

Decontamination: drop every trajectory whose repository is one of the 11
repositories of the 642 SWE-bench Pro v2 tasks (HARD-51 is a subset), plus PR
description 13-gram containment >= 0.2 against all 642 instructions.

Usage:
    python -m src.data.prepare_swe_v2 [--output_dir data/swe_v2]
"""

import argparse
import glob
import hashlib
import json
import re
import shlex
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from jinja2 import StrictUndefined, Template

from src.data.agent_v2_common import (MAX_TOKENS, MAX_TURNS, NgramIndex, TokenCounter, fit_sample,
                                      pick_turns, shuffle, take_by_quota, token_stats, write_jsonl)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/_raw_v2/swe_smith/data"
PRO = ROOT / "third_party/SWE-bench_Pro-os/v2"
MINI_CFG = yaml.safe_load((PRO / "tooling/configs/mini_toolcall.yaml").read_text())
EVAL_TRAJ = ROOT / ("results/_l40s_backup_20261018/swe-eval/results/base/swe/jobs/sbp-hard51/"
                    "instance_flipt-io__flipt-e2bd19d__toZdQcJ/agent/mini-swe-agent.trajectory.json")

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
# platform.uname() of the x86 host that ran the SWE-bench Pro evals (host-side agent loop),
# as it appears in the <system_information> block of the eval trajectories.
UNAME = {"system": "Linux", "release": "5.15.0-168-generic",
         "version": "#178-Ubuntu SMP Fri Jan 9 19:05:03 UTC 2026", "machine": "x86_64"}
TASK_HEAD = ("A code repository is available in the `/app` directory. Your task is to implement the changes "
             "necessary to satisfy the following Pull Request (PR) description:\n\n<pr_description>\n")
_CONSTRAINT = ("* Do not reference, look up, or copy existing solutions, external PRs, or online workarounds. "
               "The implementation must be entirely your own independent work.\n")
# the 642 instructions use two spellings of the tail (495 without, 147 with a blank line after
# "Constraints:"; HARD-51 has 28 / 23); each trajectory gets one, in the same 495:147 ratio
TASK_TAILS = ("\n</pr_description>\n\nConstraints:\n" + _CONSTRAINT,
              "\n</pr_description>\n\nConstraints:\n\n" + _CONSTRAINT)
TAIL_WEIGHTS = (495, 147)
# minisweagent.models.utils.actions_toolcall.BASH_TOOL as dumped by sglang's protocol.Tool
# (Function.strict defaults to False, Tool.defer_loading is kept as null).
BASH_TOOL_SGLANG = {"type": "function",
                    "function": {"description": "Execute a bash command", "name": "bash",
                                 "parameters": {"type": "object",
                                                "properties": {"command": {"type": "string",
                                                                           "description": "The bash command to execute"}},
                                                "required": ["command"]},
                                 "strict": False},
                    "defer_loading": None}
FINAL_THOUGHT = "The fix is implemented and verified, so I will submit the changes."
LICENSE = "mit (SWE-bench/SWE-smith-trajectories)"

OBS_TEMPLATE = Template(MINI_CFG["model"]["observation_template"], undefined=StrictUndefined)
SYSTEM_TEMPLATE = Template(MINI_CFG["agent"]["system_template"], undefined=StrictUndefined)
INSTANCE_TEMPLATE = Template(MINI_CFG["agent"]["instance_template"], undefined=StrictUndefined)

VIEW_FILE = re.compile(r"^Here's the result of running `cat -n` on (.+?):\n", re.S)
VIEW_DIR = re.compile(r"^Here's the files and directories up to 2 levels deep in (.+?), excluding hidden items:\n")
SNIPPET = re.compile(r"has been edited\. Here's the result of running `cat -n` on a snippet of .+?:\n(.*)\n"
                     r"Review the changes and make sure they are as expected", re.S)
NO_OUTPUT = "Your command ran successfully and did not produce any output."


class Unconvertible(Exception):
    pass


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------

def render_observation(output: str, returncode: int) -> str:
    return OBS_TEMPLATE.render(output={"output": output, "returncode": returncode, "exception_info": ""})


def task_tail(key: str) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % sum(TAIL_WEIGHTS)
    return TASK_TAILS[0] if h < TAIL_WEIGHTS[0] else TASK_TAILS[1]


def first_user(pr_description: str, tail: str) -> str:
    return INSTANCE_TEMPLATE.render(task=TASK_HEAD + pr_description + tail, **UNAME)


def system_content(counter: TokenCounter) -> str:
    """mini's system prompt + the tool block sglang adds, as plain system content."""
    sys = SYSTEM_TEMPLATE.render()
    text = counter.tok.apply_chat_template([{"role": "system", "content": sys}], tools=[BASH_TOOL_SGLANG],
                                           chat_template=counter.template, tokenize=False)
    head, tail = "<|im_start|>system\n", "<|im_end|>\n"
    assert text.startswith(head) and text.endswith(tail)
    return text[len(head):-len(tail)]


def py_literal(s: str) -> str:
    if "'''" in s or s.endswith(("'", "\\")) or "\r" in s:
        return json.dumps(s, ensure_ascii=False)
    return "r'''" + s + "'''"


def heredoc_tag(body: str) -> str:
    lines = set(body.split("\n"))
    for tag in ("EOF", "PYEOF", "END_OF_FILE", "EOF_7f3a"):
        if tag not in lines:
            return tag
    raise Unconvertible("heredoc tag")


def str_replace_script(path: str, old: str, new: str) -> Tuple[str, int]:
    body = "\n".join([
        "from pathlib import Path",
        f"p = Path({path!r})",
        "s = p.read_text()",
        f"old = {py_literal(old)}",
        f"new = {py_literal(new)}",
        "n = s.count(old)",
        "assert n == 1, f'old_str found {n} times, expected exactly 1'",
        "i = s.split(old)[0].count('\\n')",
        "s = s.replace(old, new)",
        "p.write_text(s)",
        "lines = s.split('\\n')",
        "for k in range(max(0, i - 4), min(len(lines), i + 5 + new.count('\\n'))):",
        "    print(f'{k + 1:6}\\t{lines[k]}')",
    ])
    tag = heredoc_tag(body)
    assert_line = body[:body.index("assert n == 1")].count("\n") + 1
    return f"python3 - <<'{tag}'\n{body}\n{tag}", assert_line


def infer_returncode(out: str) -> int:
    if not out.strip():
        return 0
    if "command not found" in out:
        return 127
    if "Traceback (most recent call last):" in out:
        return 1
    if re.search(r"^=+ .*\b\d+ (failed|errors?)\b.*=+$", out, re.M):
        return 1
    if re.search(r"^(cat|ls|cd|grep|head|tail|sed|python3?|bash): .*(No such file or directory|can't open file)",
                 out, re.M):
        return 1
    return 0


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def obs_text(m: Dict) -> str:
    c = m["content"]
    if isinstance(c, list):
        c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
    c = c.replace("\r\n", "\n")
    if c.startswith("OBSERVATION:\n"):
        c = c[len("OBSERVATION:\n"):]
    return c


def fix_paths(s: str) -> str:
    return s.replace("/testbed", "/app")


APOLOGY = re.compile(r"^(You're (absolutely )?right[^.!\n]*[.!]\s*)?(I apologize[^.!\n]*[.!]\s*)?")


def clean_thought(s: str) -> str:
    """Paths -> /app, editor-tool name -> bash, and drop the apologies SWE-agent's
    format-error requeries provoked (those requeries do not exist here)."""
    s = fix_paths(s).replace("str_replace_editor", "bash").strip()
    return APOLOGY.sub("", s).strip()


def convert_step(name: str, args: Dict, obs: str) -> Optional[Tuple[str, str, int]]:
    """-> (command, output, returncode); None = drop this step; raises Unconvertible."""
    if name == "bash":
        cmd = args.get("command", "")
        if not cmd.strip():
            raise Unconvertible("empty bash")
        if "was cancelled because it took more than" in obs or "timed out" in obs[:300]:
            raise Unconvertible("timeout")
        out = "" if obs.strip() == NO_OUTPUT else obs
        return fix_paths(cmd), fix_paths(out), infer_returncode(out)
    if name != "str_replace_editor":
        raise Unconvertible(f"tool:{name}")
    sub, path = args.get("command"), fix_paths(args.get("path", ""))
    if not path:
        raise Unconvertible("no path")
    q = shlex.quote(path)
    if sub == "view":
        if "too large to display entirely" in obs:
            return None
        m = VIEW_FILE.match(obs)
        if m:
            rng = args.get("view_range")
            if rng:
                a, b = int(rng[0]), int(rng[1])
                cmd = f"nl -ba {q} | sed -n '{a},{'$' if b == -1 else b}p'"
            else:
                cmd = f"cat -n {q}"
            return cmd, fix_paths(obs[m.end():]), 0
        m = VIEW_DIR.match(obs)
        if m:
            return f"find {q} -maxdepth 2 -not -path '*/.*'", fix_paths(obs[m.end():]), 0
        if "does not exist" in obs and not args.get("view_range"):
            return f"cat -n {q}", f"cat: {path}: No such file or directory\n", 1
        raise Unconvertible("view error")
    if sub == "create":
        if not obs.startswith("File created successfully at:"):
            raise Unconvertible("create error")
        text = fix_paths(args.get("file_text", ""))
        if text.endswith("\n"):
            text = text[:-1]
        tag = heredoc_tag(text)
        return f"cat <<'{tag}' > {q}\n{text}\n{tag}", "", 0
    if sub == "str_replace":
        old, new = fix_paths(args.get("old_str", "")), fix_paths(args.get("new_str", "") or "")
        if not old:
            raise Unconvertible("empty old_str")
        cmd, assert_line = str_replace_script(path, old, new)
        m = SNIPPET.search(obs)
        if m:
            return cmd, fix_paths(m.group(1)) + "\n", 0
        if obs.startswith("No replacement was performed, old_str"):
            n = 0
        elif obs.startswith("No replacement was performed. Multiple occurrences"):
            lm = re.search(r"in lines \[([^\]]*)\]", obs)
            if not lm:
                raise Unconvertible("multi-occurrence parse")
            n = len([x for x in lm.group(1).split(",") if x.strip()])
        else:
            raise Unconvertible("str_replace error")
        out = ("Traceback (most recent call last):\n"
               f"  File \"<stdin>\", line {assert_line}, in <module>\n"
               f"AssertionError: old_str found {n} times, expected exactly 1\n")
        return cmd, out, 1
    raise Unconvertible(f"editor:{sub}")


def extract_pr(user_text: str) -> Optional[str]:
    m = re.search(r"<pr_description>\n(.*)\n</pr_description>", user_text, re.S)
    return fix_paths(m.group(1).strip()) if m else None


XML_CALL = re.compile(r"<function=([\w-]+)>(.*?)</function>", re.S)
XML_PARAM = re.compile(r"<parameter=([\w-]+)>(.*?)</parameter>", re.S)


def parse_action(m: Dict) -> Tuple[str, str, Dict]:
    """(thought, tool name, args) from a native tool call or SWE-agent's XML
    ``<function=..><parameter=..>`` format (both occur in the ``tool`` split)."""
    tcs = m.get("tool_calls") or []
    if len(tcs) == 1:
        fn = tcs[0]["function"]
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            raise Unconvertible("bad arguments")
        return m.get("content") or m.get("thought") or "", fn["name"], args
    if tcs:
        raise Unconvertible("multiple tool calls")
    content = m.get("content") or ""
    if isinstance(content, list):
        content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
    calls = XML_CALL.findall(content)
    if len(calls) != 1:
        raise Unconvertible("no tool call" if not calls else "multiple tool calls")
    name, body = calls[0]
    args: Dict = {}
    for k, v in XML_PARAM.findall(body):
        if v.startswith("\n"):
            v = v[1:]
        if v.endswith("\n"):
            v = v[:-1]
        args[k] = v
    if "view_range" in args:
        try:
            args["view_range"] = json.loads(args["view_range"])
        except json.JSONDecodeError:
            raise Unconvertible("bad view_range")
    if "insert_line" in args:
        args["insert_line"] = int(args["insert_line"])
    return content[:content.index("<function=")], name, args


def convert(raw_msgs: List[Dict], sys_content: str, tail: str = TASK_TAILS[0]) -> Dict:
    """SWE-agent tool trajectory -> {messages, targets, pr}; raises Unconvertible."""
    if len(raw_msgs) < 4 or raw_msgs[0]["role"] != "system" or raw_msgs[1]["role"] != "user":
        raise Unconvertible("shape")
    pr = extract_pr(obs_text(raw_msgs[1]))
    if not pr:
        raise Unconvertible("no pr_description")
    msgs = [{"role": "system", "content": sys_content}, {"role": "user", "content": first_user(pr, tail)}]
    targets: List[int] = []
    i, n = 2, len(raw_msgs)
    while i < n:
        m = raw_msgs[i]
        if m["role"] != "assistant":
            raise Unconvertible(f"unexpected role {m['role']}")
        thought, name, args = parse_action(m)
        thought = clean_thought(thought)
        nxt = raw_msgs[i + 1] if i + 1 < n else None
        if name == "submit":
            if nxt is None:  # final submit
                msgs.append({"role": "assistant", "content": thought or FINAL_THOUGHT,
                             "tool_calls": [tool_call(SUBMIT)]})
                targets.append(len(msgs) - 1)
                return {"messages": msgs, "targets": targets, "pr": pr}
            if nxt["role"] in ("tool", "user") and "Thank you for your work on this issue" in obs_text(nxt):
                i += 2  # SWE-agent review page: no counterpart in mini-swe-agent
                continue
            raise Unconvertible("submit mid-trajectory")
        if nxt is None or nxt["role"] not in ("tool", "user"):
            raise Unconvertible("missing observation")
        step = convert_step(name, args, obs_text(nxt))
        i += 2
        if step is None:
            continue
        cmd, out, rc = step
        msgs.append({"role": "assistant", "content": thought, "tool_calls": [tool_call(cmd)]})
        targets.append(len(msgs) - 1)
        msgs.append({"role": "tool", "content": render_observation(out, rc)})
    raise Unconvertible("no final submit")


def tool_call(cmd: str) -> Dict:
    return {"type": "function", "function": {"name": "bash", "arguments": {"command": cmd}}}


# ---------------------------------------------------------------------------
# decontamination
# ---------------------------------------------------------------------------

def pro_tasks() -> Dict[str, str]:
    return {d.name: (d / "instruction.md").read_text() for d in sorted((PRO / "tasks").iterdir())
            if (d / "instruction.md").exists()}


def pro_repo(task_name: str) -> str:
    m = re.match(r"instance_(.+?)__(.+?)-[0-9a-f]{40}", task_name)
    if not m:
        raise ValueError(task_name)
    return f"{m.group(1)}/{m.group(2)}".lower()


def smith_repo(instance_id: str) -> str:
    owner, _, rest = instance_id.partition("__")
    return f"{owner}/{rest.split('.')[0]}".lower()


# ---------------------------------------------------------------------------
# format check
# ---------------------------------------------------------------------------

def format_check(rows: List[Dict], counter: TokenCounter, sys_content: str, out: Path) -> Dict:
    checks: Dict[str, object] = {}
    plain_sys = SYSTEM_TEMPLATE.render()
    probe = [{"role": "user", "content": "x"}]
    checks["system_with_embedded_tools_equals_render_with_tools"] = (
        counter.render([{"role": "system", "content": sys_content}] + probe)
        == counter.tok.apply_chat_template([{"role": "system", "content": plain_sys}] + probe,
                                           tools=[BASH_TOOL_SGLANG], chat_template=counter.template,
                                           tokenize=False))
    if EVAL_TRAJ.exists():  # a real SWE-bench Pro eval run (base model, x86 host_mini_swe)
        ev = json.loads(EVAL_TRAJ.read_text())["messages"]
        task_dir = PRO / "tasks/instance_flipt-io__flipt-e2bd19dafa7166c96b082fb2a59eb54b4be0d778"
        instr = (task_dir / "instruction.md").read_text()
        checks["eval_system_prompt_matches"] = ev[0]["content"] == plain_sys
        checks["eval_first_user_matches_our_render"] = ev[1]["content"] == INSTANCE_TEMPLATE.render(task=instr, **UNAME)
        checks["instruction_wrapper_matches"] = instr.startswith(TASK_HEAD) and instr.endswith(TASK_TAILS)
        obs_ok = [m["content"] == render_observation(m["extra"]["raw_output"], m["extra"]["returncode"])
                  for m in ev if m["role"] == "tool" and not m["extra"].get("exception_info")]
        checks["eval_observations_rerendered_identically"] = f"{sum(obs_ok)}/{len(obs_ok)}"
    wrappers = [t for t in pro_tasks().values()]
    checks["pro_instructions_with_same_wrapper"] = f"{sum(t.startswith(TASK_HEAD) and t.endswith(TASK_TAILS) for t in wrappers)}/{len(wrappers)}"
    checks["train_tail_variants_no_blank_vs_blank"] = [
        sum(r["messages"][1]["content"].count(TASK_TAILS[i]) > 0 for r in rows) for i in (0, 1)]
    checks["every_assistant_has_exactly_one_bash_call"] = all(
        len(m.get("tool_calls", [])) == 1 and m["tool_calls"][0]["function"]["name"] == "bash"
        for r in rows for m in r["messages"] if m["role"] == "assistant")
    checks["final_turn_is_submit_alone"] = all(
        r["messages"][-1]["tool_calls"][0]["function"]["arguments"]["command"] == SUBMIT
        for r in rows if r["turn"] == r["n_turns"])
    inter = [r for r in rows if 1 < r["turn"] < r["n_turns"]]
    picks = [min(inter, key=lambda r: (r["n_tokens"], r["id"])),
             min((r for r in rows if r["turn"] == r["n_turns"]), key=lambda r: (r["n_tokens"], r["id"]))]
    texts = []
    for r in picks:
        full = counter.render(r["messages"])
        prompt = counter.render(r["messages"][:-1], add_generation_prompt=True)
        checks[f"{r['id']}_prompt_render_is_prefix_of_full"] = full.startswith(prompt)
        texts.append(f"===== {r['id']} (turn {r['turn']}/{r['n_turns']}, {r['n_tokens']} tokens)\n{full}")
    (out / "format_check_samples.txt").write_text("\n\n".join(texts))
    return {"samples": [r["id"] for r in picks], "rendered": "format_check_samples.txt", "checks": checks,
            "conclusion": "Rendered text matches what host_mini_swe + mini_toolcall.yaml send through sglang "
                          "(qwen3_nothink, qwen25 tool parser): system turn = mini system prompt + the bash "
                          "tool block, user = mini instance_template around the SWE-bench Pro instruction "
                          "wrapper, assistant = reasoning text + one <tool_call>{\"name\": \"bash\", ...}, "
                          "tool results = <tool_response> with mini's JSON observation, final turn = "
                          "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT alone."}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=str(ROOT / "data/swe_v2"))
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--per_repo_cap", type=int, default=30, help="max trajectories per repository")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    counter = TokenCounter(ROOT / "models/Qwen3-8B", ROOT / "configs/chat_templates/qwen3_nothink.jinja")
    sys_content = system_content(counter)

    tasks = pro_tasks()
    pro_repos = {pro_repo(t) for t in tasks}
    pro_repo_names = {r.split("/")[1] for r in pro_repos}
    hard51 = [l.strip() for l in (PRO / "hard51_ids.txt").read_text().splitlines() if l.strip()]
    assert all(pro_repo(h) in pro_repos for h in hard51)
    pr_index = NgramIndex(tasks)

    n_raw = n_resolved = n_dup = 0
    contam = Counter()
    contam_examples: Dict[str, List[str]] = {}
    unconv = Counter()
    by_instance: Dict[str, Dict] = {}
    for f in sorted(glob.glob(str(RAW / "tool-*.parquet"))):
        df = pd.read_parquet(f, columns=["instance_id", "resolved", "traj_id", "messages"])
        for r in df.itertuples():
            n_raw += 1
            if not r.resolved:
                continue
            n_resolved += 1
            repo = smith_repo(r.instance_id)
            if repo in pro_repos or repo.split("/")[1] in pro_repo_names:
                contam["repo"] += 1
                contam_examples.setdefault("repo", []).append(r.instance_id)
                continue
            if r.instance_id in by_instance:  # keep one trajectory per task instance
                n_dup += 1
                continue
            raw_msgs = json.loads(r.messages)
            try:
                conv = convert(raw_msgs, sys_content, task_tail(r.traj_id))
            except Unconvertible as e:
                unconv[str(e)] += 1
                continue
            c, k = pr_index.max_containment(conv["pr"])
            if c >= 0.2:
                contam["pr_13gram"] += 1
                contam_examples.setdefault("pr_13gram", []).append(f"{r.instance_id} -> {k}:{c:.2f}")
                continue
            conv.update(repo=repo, traj_id=r.traj_id)
            by_instance[r.instance_id] = conv
    print(f"raw {n_raw}, resolved {n_resolved}, converted (1/instance) {len(by_instance)}, "
          f"unconvertible {dict(unconv)}, contaminated {dict(contam)}")

    # pools: shuffle, cap per repo, prefer >= MAX_TURNS target turns
    order = shuffle(sorted(by_instance), args.seed)
    per_repo = Counter()
    pool = []
    for iid in order:
        repo = by_instance[iid]["repo"]
        if per_repo[repo] >= args.per_repo_cap:
            continue
        per_repo[repo] += 1
        pool.append(iid)
    pools = {"swe_smith": pool}

    length_drop = Counter()
    trunc = Counter()

    def build(src: str, iid: str) -> List[Dict]:
        conv = by_instance[iid]
        msgs = conv["messages"]
        samples = []
        n_turns = len(conv["targets"])
        for k in pick_turns(conv["targets"], MAX_TURNS):
            prefix = msgs[:k + 1]
            obs = [i for i, m in enumerate(prefix) if m["role"] == "tool"]
            fitted, ntok, ncut = fit_sample(prefix, obs, counter, MAX_TOKENS)
            if fitted is None:
                length_drop["samples"] += 1
                continue
            trunc["samples_truncated"] += bool(ncut)
            trunc["messages_truncated"] += ncut
            turn = sum(1 for m in prefix if m["role"] == "assistant")
            samples.append({"id": f"sv2_{hashlib.sha1(conv['traj_id'].encode()).hexdigest()[:10]}_t{turn:02d}",
                            "messages": fitted, "source": src, "traj_id": conv["traj_id"],
                            "instance_id": iid, "repo": conv["repo"], "turn": turn, "n_turns": n_turns,
                            "n_tokens": ntok})
        return samples

    used: set = set()
    val = take_by_quota(pools, {"swe_smith": args.n_val}, build, used)
    train = take_by_quota(pools, {"swe_smith": args.n_train}, build, used)
    assert not ({s["traj_id"] for s in val} & {s["traj_id"] for s in train})
    for name, rows in (("train", train), ("validation", val)):
        write_jsonl(out / f"{name}.jsonl", [{**r, "split": name} for r in shuffle(rows, args.seed)])

    stats = {
        "description": "Multi-turn mini-swe-agent (tool-call) trajectories converted from SWE-agent "
                       "SWE-smith trajectories; one sample per selected assistant turn (prompt = all "
                       "previous messages, completion = that turn), <= 4 turns per trajectory picked "
                       "uniformly incl. the final submit turn.",
        "eval_format": {
            "agent": "host_mini_swe + third_party/SWE-bench_Pro-os/v2/tooling/configs/mini_toolcall.yaml",
            "system": "mini system_template + sglang-rendered bash tool block (embedded in content because "
                      "train_expert.py does not pass tools)",
            "tool_schema_as_rendered": BASH_TOOL_SGLANG,
            "system_information": UNAME,
            "assistant": "reasoning text + exactly one bash tool call (tool_calls[].function.arguments is a dict)",
            "tool": "role=tool, mini observation_template JSON",
            "final": SUBMIT,
            "returncode": "inferred from output (SWE-agent does not log exit codes)",
        },
        "sources": {"swe_smith": {"license": LICENSE, "split": "tool (resolved only)",
                                  "train_samples": len(train), "validation_samples": len(val)}},
        "raw_trajectories": n_raw,
        "resolved_trajectories": n_resolved,
        "unconvertible_dropped": dict(unconv),
        "duplicate_instance_trajectories_skipped": n_dup,
        "converted_one_per_instance": len(by_instance),
        "per_repo_cap": args.per_repo_cap,
        "repos_used": len({r["repo"] for r in train + val}),
        "train": len(train),
        "validation": len(val),
        "train_trajectories": len({r["traj_id"] for r in train}),
        "validation_trajectories": len({r["traj_id"] for r in val}),
        "tokens_train": token_stats([r["n_tokens"] for r in train]),
        "tokens_validation": token_stats([r["n_tokens"] for r in val]),
        "turn_position": dict(Counter("final" if r["turn"] == r["n_turns"] else "intermediate" for r in train)),
        "length_control": {"max_tokens": MAX_TOKENS, **dict(trunc), "samples_dropped_too_long": length_drop["samples"]},
        "decontamination": {
            "reference": f"SWE-bench Pro v2, {len(tasks)} tasks in {len(pro_repos)} repos (HARD-51 ⊂ these repos)",
            "pro_repos": sorted(pro_repos),
            "criteria": ["repository (owner/repo or repo name) is a SWE-bench Pro v2 repository",
                         "PR description 13-gram containment >= 0.2 vs any of the 642 instructions"],
            "removed_trajectories": dict(contam),
            "removed_total": sum(contam.values()),
            "examples": {k: v[:10] for k, v in contam_examples.items()},
        },
        "tokenizer": "models/Qwen3-8B + configs/chat_templates/qwen3_nothink.jinja",
        "seed": args.seed,
    }
    stats["format_check"] = format_check(train, counter, sys_content, out)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: stats[k] for k in ("train", "validation", "tokens_train", "decontamination",
                                            "format_check")}, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
