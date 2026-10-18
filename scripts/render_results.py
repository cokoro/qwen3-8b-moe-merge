"""Render results/summary.csv + analysis/uncertainty.json into the markdown tables used in README §7
and report.pdf -> analysis/results_tables.md (no numbers are typed by hand).

    python scripts/render_results.py
"""

import argparse
import csv
import json
from pathlib import Path

BENCH = [("hle", "HLE"), ("asi", "ASI v2"), ("terminal", "TB2"), ("swe", "SWE-Pro")]
SUPP = [("hle_judge", "HLE 原始 judge"), ("hle_mc", "HLE 选择题"), ("asi_v1", "ASI v1（环境缺包，仅参考）")]


def f(v, nd=2):
    return "—" if v in (None, "") else f"{float(v):.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--uncertainty", default="analysis/uncertainty.json")
    ap.add_argument("--out", default="analysis/results_tables.md")
    args = ap.parse_args()
    rows = {r["model"]: r for r in csv.DictReader(open(args.summary))}
    unc = json.loads(Path(args.uncertainty).read_text())
    pv, rep = unc["paired_vs_base"], unc.get("repeat_runs", {})
    L = []

    L += ["**表 1：主指标（0–100）**。macro4 = 四项等权平均；「—」= 未评测。n = 题目数。", "",
          "| 模型 | " + " | ".join(f"{n} (n)" for _, n in BENCH) + " | **macro4** | " + " | ".join(n for _, n in SUPP) + " |",
          "|---|" + "---|" * (len(BENCH) + 1 + len(SUPP))]
    for m, r in rows.items():
        cells = [f"{f(r[b])} ({r[b + '_n']})" if r[b] else "—" for b, _ in BENCH]
        L.append(f"| {m} | " + " | ".join(cells) + f" | **{f(r['macro4'])}** | " + " | ".join(f(r[b]) for b, _ in SUPP) + " |")

    L += ["", "**表 2：相对 base 的变化（百分点），配对 bootstrap 95% CI（10k 次，seed 42）**。", "",
          "| 模型 | " + " | ".join(n for _, n in BENCH) + " | **macro4** | " + " | ".join(n for _, n in SUPP) + " |",
          "|---|" + "---|" * (len(BENCH) + 1 + len(SUPP))]
    for m, d in pv.items():
        def c(k):
            x = d.get(k)
            return "—" if not x else f"{x['delta']:+.2f} [{x['ci95'][0]:+.1f}, {x['ci95'][1]:+.1f}]"
        L.append(f"| {m} | " + " | ".join(c(b) for b, _ in BENCH) + f" | **{c('macro4')}** | " + " | ".join(c(b) for b, _ in SUPP) + " |")

    if rep:
        L += ["", "**表 3：重复运行波动**（相同题目、相同设置，第二次采样；TB2 第二次运行若未跑完，只在两次都完成的任务上比较）。", "",
              "| benchmark | 模型 | n | 第 1 次 | 第 2 次 | 差值 | 逐题一致率 |", "|---|---|---|---|---|---|---|"]
        for b, ms in rep.items():
            for m, x in ms.items():
                L.append(f"| {b} | {m} | {x['n']} | {x['run1']:.2f} | {x['run2']:.2f} | {x['abs_diff']:.2f} | {x['item_agreement']:.3f} |")

    L += ["", "**运行失败（errored）**：" + "；".join(
        f"{m}: " + ", ".join(f"{b}={r[b + '_errored']}" for b, _ in BENCH if r.get(b + "_errored") not in (None, ""))
        for m, r in rows.items() if any(r.get(b + "_errored") not in (None, "") for b, _ in BENCH))]
    if unc.get("swe_oracle_valid") is not None:
        L += ["", f"**SWE-Pro 计分任务**（gold patch 在本机通过）：{len(unc['swe_oracle_valid'])} 个："
              + ", ".join(f"`{t.replace('instance_', '')[:40]}`" for t in unc["swe_oracle_valid"])]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
