"""Failure analysis for the merged models -> analysis/failures.md (+ failures.jsonl).

Every failed item of the analysed models is assigned one category from the raw results:
  HLE       domain_format_leak (bare code block instead of an answer) | no_answer_extracted | template_echo (answers the literal 'Answer: {your chosen answer}') |
            wrong_choice (MC) | wrong_answer (exact match), under the audited verdict of summarize.py;
            tagged merge_regression when base got the item right, judge_false_positive when the raw
            local judge had accepted it
  ASI       missing_outputs (required files not produced: code crashed / not runnable) |
            scorer_timeout | wrong_values (outputs produced but failed gates/score)
  Terminal  timeout (AgentTimeoutError) | env_tool_failure (environment/runtime errors) |
            task_not_achieved (agent finished, verifier failed)
  SWE       no_patch (empty diff) | patch_fails_tests | timeout | env_tool_failure
Then a stratified sample (per benchmark x category, seeded) of >= 24 cases is written
with short excerpts so each attribution can be checked against the raw files.

    python scripts/analyze_failures.py --results results --models merged_linear mopd
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []


def _summarize():
    import importlib.util
    spec = importlib.util.spec_from_file_location("summarize", Path(__file__).with_name("summarize.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hle_items(res, model):
    """Failures under the audited HLE verdict (same rule as summarize.py). A failure the raw local judge
    had accepted is tagged judge_false_positive."""
    S = _summarize()
    d = res / model / "hle"
    preds = d / f"hle_{model}.json"
    if not preds.exists():
        return []
    import pandas as pd
    ds = pd.read_parquet("data/eval/hle/test.parquet").set_index("id")
    p = json.loads(preds.read_text())
    judged = d / f"judged_hle_{model}.json.json"
    j = json.loads(judged.read_text()) if judged.exists() else {}
    audited = S.hle_audited(res, model) or S.hle_mc(res, model) or {}
    base_ok = S.hle_audited(res, "base") or {}
    out = []
    for qid in ds.index:
        if qid not in audited or audited[qid]:
            continue
        resp = (p.get(qid) or {}).get("response", "")
        jr = (j.get(qid) or {}).get("judge_response") or {}
        mc = ds.loc[qid, "answer_type"] == "multipleChoice"
        fa = S.final_answer(resp)
        if not fa and resp.lstrip().startswith("```"):
            cat = "domain_format_leak"  # answers a reasoning question with a bare shell/code block (terminal expert's format)
        elif not fa:
            cat = "no_answer_extracted"
        elif S.PLACEHOLDER.search(fa):
            cat = "template_echo"
        else:
            cat = "wrong_choice" if mc else "wrong_answer"
        out.append({"benchmark": "hle", "id": qid, "category": cat,
                    "merge_regression": bool(base_ok.get(qid)),
                    "judge_false_positive": jr.get("correct") == "yes",
                    "question": str(ds.loc[qid, "question"])[:160].replace("\n", " "),
                    "gold": str(ds.loc[qid, "answer"])[:60],
                    "model_answer": fa[:80],
                    "category_raw": str(ds.loc[qid, "category"])})
    return out


def asi_items(res, model):
    rep = res / model / "asi" / "local_score_seed31415.json"
    if not rep.exists():
        return []
    out = []
    for r in json.loads(rep.read_text())["results"]:
        frac = (r.get("final_score") or 0) / (r.get("max_score") or 100)
        if frac >= 0.5:
            continue
        msgs = " | ".join(g.get("message", "") for g in r.get("gate_results", []) if not g.get("passed"))
        if r.get("evaluation_status") == "scorer_timeout":
            cat = "scorer_timeout"
        elif "not found" in msgs.lower() or "missing" in msgs.lower():
            cat = "missing_outputs"
        else:
            cat = "wrong_values"
        out.append({"benchmark": "asi", "id": r["instance_id"], "category": cat,
                    "score": round(100 * frac, 1), "gate_messages": msgs[:200]})
    return out


def harbor_items(res, model, bench):
    out = []
    for r in load_jsonl(res / model / bench / "detailed_results.jsonl"):
        if r.get("is_correct"):
            continue
        exc = r.get("exception") or ""
        if "Timeout" in exc:
            cat = "timeout"
        elif exc:
            cat = "env_tool_failure"
        elif bench == "terminal":
            cat = "task_not_achieved"
        else:
            cat = "patch_fails_tests"
        item = {"benchmark": bench, "id": r["instance_id"], "category": cat, "exception": exc}
        if bench == "swe":
            jobs = res / model / "swe" / "jobs" / "sbp-hard51"
            trial = next(iter(sorted(jobs.glob(f"{r['instance_id'].split('/')[-1][:30]}*"))), None)
            patch = trial / "agent" / "model.patch" if trial else None
            n = patch.stat().st_size if patch and patch.exists() else 0
            item["patch_bytes"] = n
            if n == 0 and cat == "patch_fails_tests":
                item["category"] = "no_patch"
        out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", nargs="+", default=["merged_linear", "mopd"])
    ap.add_argument("--per_cell", type=int, default=2, help="sampled cases per benchmark x category x model")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    all_items, counts = [], defaultdict(Counter)
    for m in args.models:
        items = hle_items(res, m) + asi_items(res, m) + harbor_items(res, m, "terminal") + harbor_items(res, m, "swe")
        for it in items:
            it["model"] = m
            counts[(m, it["benchmark"])][it["category"]] += 1
            if it.get("merge_regression"):
                counts[(m, it["benchmark"])]["(of which merge_regression vs base)"] += 1
            if it.get("judge_false_positive"):
                counts[(m, it["benchmark"])]["(of which raw local judge said correct)"] += 1
        all_items += items
    with open(out / "failures.jsonl", "w") as fh:
        for it in all_items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    sample = []
    groups = defaultdict(list)
    for it in all_items:
        groups[(it["model"], it["benchmark"], it["category"])].append(it)
    for key in sorted(groups):
        g = groups[key]
        # prefer merge regressions on HLE: they are the evidence about merging itself
        g.sort(key=lambda x: (not x.get("merge_regression", False), rng.random()))
        sample += g[: args.per_cell]

    lines = ["# Failure analysis (merged models)", "",
             f"Models: {', '.join(args.models)}. Source: `results/<model>/<benchmark>/` raw per-item files; "
             "full list in `analysis/failures.jsonl`. Categories are assigned by the rules in "
             "`scripts/analyze_failures.py`; sampled cases below are stratified per model x benchmark x category (seed 42).", "",
             "## Category counts", "", "| model | benchmark | category | count |", "|---|---|---|---|"]
    for (m, b), c in sorted(counts.items()):
        for cat, n in c.most_common():
            lines.append(f"| {m} | {b} | {cat} | {n} |")
    lines += ["", f"## Sampled cases ({len(sample)})", ""]
    for i, it in enumerate(sample, 1):
        detail = {k: v for k, v in it.items() if k not in ("benchmark", "id", "category", "model")}
        lines.append(f"{i}. **[{it['model']} / {it['benchmark']}] `{it['id'][:70]}`** — *{it['category']}*  ")
        lines.append("   " + "; ".join(f"{k}: {str(v)[:160]}" for k, v in detail.items() if v not in ("", None)))
    (out / "failures.md").write_text("\n".join(lines) + "\n")
    print(f"{len(all_items)} failures, {len(sample)} sampled -> {out}/failures.md")


if __name__ == "__main__":
    main()
