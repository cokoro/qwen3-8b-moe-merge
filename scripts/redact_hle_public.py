"""Before publishing: remove HLE test content (gold answers, question text) from the result files, as the HLE
maintainers ask that benchmark examples not be posted online in plain text. Scoring is unaffected: summarize.py
and analyze_failures.py read gold answers from the local, gated data/eval/hle/test.parquet.

    python scripts/redact_hle_public.py results results_rep2 analysis
"""
import json
import sys
from pathlib import Path

R = "[redacted: HLE test content]"
for root in sys.argv[1:]:
    for p in Path(root).rglob("judged_hle_*.json"):
        d = json.loads(p.read_text())
        for v in d.values():
            jr = v.get("judge_response") or {}
            for k in ("correct_answer", "reasoning"):
                if k in jr:
                    jr[k] = R
        p.write_text(json.dumps(d, indent=1))
        print("redacted", p)
    for p in Path(root).rglob("failures.jsonl"):
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        for r in rows:
            if r.get("benchmark") == "hle":
                for k in ("question", "gold"):
                    if k in r:
                        r[k] = R
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print("redacted", p)
    for p in Path(root).rglob("failures.md"):
        out = []
        for line in p.read_text().splitlines():
            if line.startswith("   ") and "question:" in line:
                parts = [s for s in line.strip().split("; ") if not s.startswith(("question:", "gold:"))]
                line = "   " + "; ".join(parts) + "; question/gold: " + R
            out.append(line)
        p.write_text("\n".join(out) + "\n")
        print("redacted", p)
