"""Prepare Terminal-Bench expert training data.

Synthesizes shell command training examples from hand-authored templates and
optionally extends with a public dataset. Each example pairs a natural-language
terminal task description with the correct shell command(s) wrapped in a bash
code block.

NOTE: The real Terminal-Bench test set must remain completely untouched until
final evaluation. Do NOT load or inspect Terminal-Bench test instances here.
"""

import json
import argparse
import hashlib
import random
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset

from src.data.common import write_splits


# Hand-authored terminal task examples.
# Each entry: task description → shell command answer.
TEMPLATES: List[Dict] = [
    {
        "task": "Find all Python files in the current directory tree that are larger than 100 KB.",
        "command": "find . -name '*.py' -size +100k",
    },
    {
        "task": "Count the total number of lines across all .log files in /var/log.",
        "command": "find /var/log -name '*.log' -exec wc -l {} + | tail -1",
    },
    {
        "task": "Create a compressed tar archive of the /etc/nginx directory named nginx_backup.tar.gz.",
        "command": "tar -czf nginx_backup.tar.gz /etc/nginx",
    },
    {
        "task": "Show disk usage for each top-level directory under /home, sorted by size descending.",
        "command": "du -sh /home/* | sort -rh",
    },
    {
        "task": "List all running processes, filtering for those belonging to user www-data.",
        "command": "ps aux | grep www-data | grep -v grep",
    },
    {
        "task": "Recursively search for the string 'TODO' in all .py files under the src directory.",
        "command": "grep -rn 'TODO' src --include='*.py'",
    },
    {
        "task": "Show the 20 most recently modified files anywhere under /var/www.",
        "command": "find /var/www -type f -printf '%T@ %p\\n' | sort -rn | head -20 | awk '{print $2}'",
    },
    {
        "task": "Delete all files with a .tmp extension in the /tmp directory.",
        "command": "find /tmp -maxdepth 1 -name '*.tmp' -delete",
    },
    {
        "task": "Show the top 10 processes sorted by memory usage.",
        "command": "ps aux --sort=-%mem | head -11",
    },
    {
        "task": "Replace all occurrences of 'localhost' with '127.0.0.1' in config.ini in place.",
        "command": "sed -i 's/localhost/127.0.0.1/g' config.ini",
    },
    {
        "task": "Display the last 100 lines of /var/log/syslog and follow it in real time.",
        "command": "tail -n 100 -f /var/log/syslog",
    },
    {
        "task": "Print each unique IP address that appears in access.log, one per line.",
        "command": "awk '{print $1}' access.log | sort -u",
    },
    {
        "task": "Check which process is listening on port 8080.",
        "command": "lsof -i :8080 | grep LISTEN",
    },
    {
        "task": "Extract the contents of archive.tar.gz into the directory /opt/app.",
        "command": "tar -xzf archive.tar.gz -C /opt/app",
    },
    {
        "task": "Recursively change the owner of /var/www/html and all its contents to user 'deploy' and group 'www-data'.",
        "command": "chown -R deploy:www-data /var/www/html",
    },
]


def make_sample_from_template(template: Dict, idx: int) -> Dict:
    """Convert a hand-authored template into a training sample."""
    task = template["task"]
    command = template["command"]
    sample_id = f"term_template_{idx:04d}_{hashlib.md5(task.encode()).hexdigest()[:6]}"
    return {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": task},
            {"role": "assistant", "content": f"```bash\n{command}\n```"},
        ],
        "source": "terminal_handauthored",
        "split": "train",
    }


def collect_nl2bash_data(max_samples: int) -> List[Dict]:
    """NL -> bash pairs from westenfelder/NL2SH-ALFA (MIT), train config only."""
    examples = []
    try:
        dataset = load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")
        for item in dataset:
            prompt, command = (item.get("nl") or "").strip(), (item.get("bash") or "").strip()
            if not prompt or not command:
                continue
            examples.append({
                "id": f"term_nl2sh_{hashlib.md5(prompt.encode()).hexdigest()[:8]}",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"```bash\n{command}\n```"},
                ],
                "source": "terminal_nl2sh_alfa",
                "split": "train",
            })
            if len(examples) >= max_samples * 2:
                break
    except Exception as exc:
        print(f"Warning: could not load NL2SH-ALFA: {exc}")
    return examples


CANARY_MARKERS = ("BENCHMARK DATA SHOULD NEVER APPEAR", "canary GUID", "terminal-bench-canary")


def collect_seta_tasks(seta_dir: str, max_chars: int = 12000) -> List[Dict]:
    """Harbor task instruction -> oracle solve.sh from locally reviewed SETA-Env tasks.

    Only ``<seta_dir>/<task>/`` directories are read (the reviewed candidates;
    quarantined tasks live elsewhere). Any file containing a benchmark canary
    marker is skipped.
    """
    examples = []
    root = Path(seta_dir)
    if not root.is_dir():
        print(f"Warning: SETA task dir not found: {seta_dir}")
        return examples
    for task in sorted(p for p in root.iterdir() if p.is_dir()):
        instruction, solution = task / "instruction.md", task / "solution" / "solve.sh"
        if not instruction.exists() or not solution.exists():
            continue
        text, script = instruction.read_text(errors="ignore"), solution.read_text(errors="ignore")
        if any(m.lower() in (text + script).lower() for m in CANARY_MARKERS):
            print(f"Skipping {task.name}: canary marker")
            continue
        if len(text) + len(script) > max_chars:
            continue
        examples.append({
            "id": f"term_seta_{task.name}",
            "messages": [
                {"role": "user", "content": "Complete this task in a Linux terminal. "
                                            "Reply with a single bash script.\n\n" + text},
                {"role": "assistant", "content": f"```bash\n{script.strip()}\n```"},
            ],
            "source": "terminal_seta_env",
            "split": "train",
        })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare Terminal expert training data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--seta_dir", default=None,
                        help="Directory of reviewed SETA-Env Harbor tasks (instruction.md + solution/solve.sh)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Building hand-authored terminal task templates…")
    samples: List[Dict] = [make_sample_from_template(t, i) for i, t in enumerate(TEMPLATES)]

    if args.seta_dir:
        seta = collect_seta_tasks(args.seta_dir)
        print(f"SETA-Env tasks: {len(seta)}")
        samples.extend(seta)

    remaining = args.max_samples - len(samples)
    if remaining > 0:
        print(f"Collecting public NL-to-bash data (up to {remaining} items)…")
        samples.extend(collect_nl2bash_data(remaining))

    write_splits(samples, output_path, args.max_samples)

if __name__ == "__main__":
    main()
