"""Summarise every model x benchmark result into results/summary.csv and
analysis/uncertainty.json, from the per-task files written by the benchmark scripts.

Primary metric per benchmark (all on a 0-100 scale, fixed before evaluation):
  hle       accuracy on the 500-question text-only subset, judge AUDITED: multiple-choice items are
            scored by exact letter match (no judge), exact-answer items keep the local judge's verdict
            unless the judge extracted no answer. The raw local-judge accuracy is reported as hle_judge.
  hle_mc    exact-letter accuracy on the multiple-choice questions of that subset (no judge)
  asi       mean final_score / max_score over the 60 seed31415 instances
  terminal  pass rate over the 74 Terminal-Bench 2.0 tasks (single run)
  swe       resolved rate over the SWE-bench Pro v2 HARD-51 subset tasks whose gold patch
            passes on this host (oracle-validated)
Macro average = equal-weight mean of hle, asi, terminal, swe (hle_mc is supplementary).
Deltas vs base use a paired bootstrap over the same items (10k resamples, seed 42).

    python scripts/summarize.py --results results --out_csv results/summary.csv --out_json analysis/uncertainty.json
"""

import argparse
import csv
import json
import re
import warnings
from pathlib import Path

import numpy as np

BENCHES = ["hle", "asi", "terminal", "swe"]
MODELS = ["base", "expert_hle", "expert_asi", "expert_terminal", "expert_swe",
          "merged_linear", "merged_ties", "mopd"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []


ASI_ROOT = None  # set from --asi_results: ASI v2 (per-task runtime environments) replaces the invalid v1 run


def per_item(results: Path, model: str, bench: str, swe_valid=None):
    """Return {item_id: score in [0,1]} or None if the cell was not run."""
    if bench == "asi" and ASI_ROOT is not None and (ASI_ROOT / model / "asi" / "results.json").exists():
        results = ASI_ROOT
    if bench == "asi_v1":
        bench = "asi"
    d = results / model / ("hle" if bench == "hle_judge" else bench)
    if not (d / "results.json").exists():
        return None
    rows = load_jsonl(d / "detailed_results.jsonl")
    if bench == "hle":
        return hle_audited(results, model)
    if bench == "hle_judge":
        ids = json.loads(Path("data/eval/hle/question_ids.json").read_text())
        got = {r["id"]: float(r["is_correct"]) for r in rows}
        return {i: got.get(i, 0.0) for i in ids}  # unanswered/unjudged count as wrong
    if bench == "asi":
        return {r["instance_id"]: float(r["score"]) for r in rows}
    if bench == "terminal":
        return {r["instance_id"]: float(r["is_correct"]) for r in rows}
    if bench == "swe":
        out = {r["instance_id"].split("/")[-1]: float(r["is_correct"]) for r in rows}
        if swe_valid is not None:
            out = {k: v for k, v in out.items() if k in swe_valid}
        return out
    return None


NO_ANSWER = {"", "none", "null", "n/a", "na", "no answer", "unknown"}
PLACEHOLDER = re.compile(r"\{\s*your (chosen )?answer", re.I)


def _norm(t):
    return re.sub(r"[\s$\\{}()\[\],.:;`*]|\\text|\\mathrm", "", str(t)).lower()


def _num(t):
    """Plain number or a pure-arithmetic expression (digits, + - * / ^ parentheses) -> float, else None."""
    t = str(t).replace(",", "").replace("\\times", "*").replace("\\cdot", "*").strip().strip("$").strip()
    if not t or not re.fullmatch(r"[\d\s.+\-*/^()]+", t) or "**" in t or t.count("^") > 1:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = eval(t.replace("^", "**"), {"__builtins__": {}}, {})  # charset-restricted above
        return float(v)
    except Exception:
        return None


def final_answer(resp):
    m = re.findall(r"Answer\s*:\s*(.+)", resp or "")
    return m[-1].strip() if m else ""


def hle_audited(results: Path, model: str):
    """Local-judge verdicts with deterministic overrides. The 8B judge sees the gold answer and, when the
    model's response has no real answer (e.g. it echoes the template 'Answer: {your chosen answer}'), it
    often 'extracts' the gold answer and says yes. Rules, identical for every model:
      MC items      -> exact letter match on the last 'Answer:' line (no judge);
      other items   -> correct only if judge says yes AND the response has a non-placeholder final answer
                       AND the judge's extracted answer occurs in the response (normalised) AND, when both
                       gold and extracted answer are plain numbers, they are equal.
    Unanswered / unjudged items count as wrong."""
    d = results / model / "hle"
    judged = d / f"judged_hle_{model}.json.json"
    if not (d / "results.json").exists() or not judged.exists():
        return None
    import pandas as pd
    ds = pd.read_parquet("data/eval/hle/test.parquet").set_index("id")
    ids = json.loads(Path("data/eval/hle/question_ids.json").read_text())
    J = json.loads(judged.read_text())
    mc = hle_mc(results, model)
    out = {}
    for qid in ids:
        if ds.loc[qid, "answer_type"] == "multipleChoice":
            out[qid] = mc.get(qid, 0.0)
            continue
        v = J.get(qid) or {}
        jr = v.get("judge_response") or {}
        resp = v.get("response") or ""
        fa = final_answer(resp)
        ext = str(jr.get("model_answer") or "").strip()
        ok = (jr.get("correct") == "yes" and fa and not PLACEHOLDER.search(fa)
              and ext.strip(".").lower() not in NO_ANSWER and _norm(ext) in _norm(resp))
        g, e = _num(ds.loc[qid, "answer"]), _num(ext)
        if ok and g is not None and e is not None and g != e:
            ok = False
        out[qid] = float(bool(ok))
    return out


def hle_mc(results: Path, model: str):
    """Exact-letter accuracy on multiple-choice questions (judge-free)."""
    pred = results / model / "hle" / f"hle_{model}.json"
    if not pred.exists():
        return None
    import pandas as pd
    ds = pd.read_parquet("data/eval/hle/test.parquet").set_index("id")
    p = json.loads(pred.read_text())
    out = {}
    for qid in ds.index:
        if ds.loc[qid, "answer_type"] != "multipleChoice":
            continue
        resp = (p.get(qid) or {}).get("response", "")
        m = re.findall(r"Answer\s*:\s*\**\s*([A-Z])\b", resp)
        out[qid] = float(bool(m) and m[-1] == str(ds.loc[qid, "answer"]).strip()[:1])
    return out


def paired_bootstrap(a: dict, b: dict, n=10000, seed=42):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    x = np.array([a[k] for k in keys]); y = np.array([b[k] for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    diffs = (y[idx] - x[idx]).mean(axis=1) * 100
    return {"n": len(keys), "delta": float((y - x).mean() * 100),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_le_0": float((diffs <= 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--swe_validation", default="results/_validation/swe/detailed_results.jsonl")
    ap.add_argument("--out_csv", default="results/summary.csv")
    ap.add_argument("--out_json", default="analysis/uncertainty.json")
    ap.add_argument("--rep2", default="results_rep2", help="second HLE run of some models (repeat-run variance)")
    ap.add_argument("--asi_results", default="results_v2",
                    help="ASI v2 root (asibench --sandbox task); v1 ran without the declared task packages and is "
                         "kept only as the supplementary column asi_v1")
    args = ap.parse_args()
    res = Path(args.results)
    global ASI_ROOT
    ASI_ROOT = Path(args.asi_results) if Path(args.asi_results).exists() else None

    swe_valid = None
    if Path(args.swe_validation).exists():
        swe_valid = {r["instance_id"].split("/")[-1] for r in load_jsonl(args.swe_validation) if r.get("is_correct")}

    items = {m: {b: per_item(res, m, b, swe_valid) for b in BENCHES} for m in MODELS}
    for m in MODELS:
        items[m]["hle_mc"] = hle_mc(res, m)
        items[m]["hle_judge"] = per_item(res, m, "hle_judge") if items[m]["hle"] is not None else None
        items[m]["asi_v1"] = per_item(res, m, "asi_v1")

    def score(d):
        return None if not d else 100 * sum(d.values()) / len(d)

    rows, unc = [], {}
    for m in MODELS:
        r = {"model": m}
        for b in BENCHES + ["hle_mc", "hle_judge", "asi_v1"]:
            r[b] = score(items[m][b])
            r[f"{b}_n"] = len(items[m][b]) if items[m][b] else 0
            meta = ((ASI_ROOT if b == "asi" and ASI_ROOT is not None else res) / m / b / "results.json")
            if b in BENCHES and meta.exists():
                j = json.loads(meta.read_text())
                r[f"{b}_errored"] = j.get("n_errored", j.get("n_scorer_errors"))
        vals = [r[b] for b in BENCHES]
        r["macro4"] = None if any(v is None for v in vals) else sum(vals) / 4
        rows.append(r)
        if m != "base":
            unc[m] = {b: paired_bootstrap(items["base"][b], items[m][b])
                      for b in BENCHES + ["hle_mc", "hle_judge", "asi_v1"] if items["base"][b] and items[m][b]}
            # macro delta CI: bootstrap each benchmark independently, average the deltas
            if all(items["base"][b] and items[m][b] for b in BENCHES):
                rng = np.random.default_rng(42)
                draws = []
                for b in BENCHES:
                    keys = sorted(set(items["base"][b]) & set(items[m][b]))
                    x = np.array([items["base"][b][k] for k in keys]); y = np.array([items[m][b][k] for k in keys])
                    idx = rng.integers(0, len(keys), size=(10000, len(keys)))
                    draws.append((y[idx] - x[idx]).mean(axis=1) * 100)
                md = np.mean(draws, axis=0)
                unc[m]["macro4"] = {"delta": float(np.mean([unc[m][b]["delta"] for b in BENCHES])),
                                    "ci95": [float(np.percentile(md, 2.5)), float(np.percentile(md, 97.5))]}

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    cols = ["model"] + [c for b in BENCHES + ["hle_mc", "hle_judge", "asi_v1"] for c in (b, f"{b}_n")] + \
           [f"{b}_errored" for b in BENCHES] + ["macro4"]
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
    # repeat-run variance: same items, same settings, a second sampling run (HLE: base/merged_linear;
    # TB2: base/merged_linear, possibly partial -> compared on the tasks both runs finished)
    rep = {}
    for b in ["hle", "terminal"]:
        for m in MODELS:
            if not (Path(args.rep2) / m / b / "results.json").exists() or not items[m][b]:
                continue
            r1, r2 = items[m][b], per_item(Path(args.rep2), m, b)
            keys = sorted(set(r1) & set(r2 or {}))
            if not keys:
                continue
            a, c = 100 * np.mean([r1[k] for k in keys]), 100 * np.mean([r2[k] for k in keys])
            rep.setdefault(b, {})[m] = {"n": len(keys), "run1": a, "run2": c, "abs_diff": abs(a - c),
                                        "item_agreement": float(np.mean([r1[k] == r2[k] for k in keys]))}
            if b == "hle":
                rep[b][m]["hle_judge_run2"] = score(per_item(Path(args.rep2), m, "hle_judge"))
            print(f"repeat {b} {m}: n={len(keys)} run1 {a:.2f} run2 {c:.2f} agreement {rep[b][m]['item_agreement']:.3f}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"swe_oracle_valid": sorted(swe_valid) if swe_valid else None,
                                               "paired_vs_base": unc, "repeat_runs": rep}, indent=2))

    fmt = lambda v: "  -  " if v is None else f"{v:5.2f}"
    print(f"{'model':16s} " + " ".join(f"{b:>8s}" for b in BENCHES + ['hle_mc', 'hle_judge', 'asi_v1', 'macro4']))
    for r in rows:
        print(f"{r['model']:16s} " + " ".join(f"{fmt(r[b]):>8s}" for b in BENCHES + ['hle_mc', 'hle_judge', 'asi_v1', 'macro4']))


if __name__ == "__main__":
    main()
