"""Rebuild the training curves from the raw logs (unsmoothed).

  curves/sft_loss.png       train loss (per logging step) and held-out validation loss of each
                            LoRA expert, x = optimizer step
  curves/stage2_metrics.png MOPD student: per-token reverse KL to the domain teacher and mean
                            sampled response length, per domain, x = MOPD step
  curves/source/*.csv       the exact points plotted

    python scripts/plot_curves.py --checkpoints checkpoints --out curves
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPERTS = ["swe", "hle", "asi", "terminal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--out", default="curves")
    args = ap.parse_args()
    ck, out = Path(args.checkpoints), Path(args.out)
    (out / "source").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    rows = []
    for ax, e in zip(axes, EXPERTS):
        f = ck / f"expert_{e}" / "training_metrics.json"
        if not f.exists():
            ax.set_title(f"expert_{e}: missing"); continue
        hist = json.loads(f.read_text())["log_history"]
        tr = [(h["step"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
        ev = sorted({(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h})
        ax.plot(*zip(*tr), label="train loss", lw=1)
        ax.plot(*zip(*ev), "o-", label="validation loss", lw=1.5)
        ax.set_title(f"expert_{e} (LoRA r16, run train_{e})"); ax.set_xlabel("optimizer step")
        ax.set_ylabel("token CE loss (completion only)"); ax.legend(fontsize=8)
        rows += [{"expert": e, "split": "train", "step": s, "loss": v} for s, v in tr]
        rows += [{"expert": e, "split": "validation", "step": s, "loss": v} for s, v in ev]
    fig.tight_layout(); fig.savefig(out / "sft_loss.png", dpi=130); plt.close(fig)
    with open(out / "source" / "sft_loss.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["expert", "split", "step", "loss"]); w.writeheader(); w.writerows(rows)

    for name in ["expert_mopd", "expert_mopd14b"]:
        log = ck / name / "mopd_log.jsonl"
        if not log.exists():
            continue
        recs = [json.loads(l) for l in open(log) if l.strip()]
        doms = [k for k in recs[0] if isinstance(recs[0][k], dict)]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
        rows = []
        for d in doms:
            st = [r["step"] for r in recs]
            kl = [r[d]["kl_per_token"] for r in recs]
            ln = [r[d]["mean_resp_len"] for r in recs]
            a1.plot(st, kl, label=f"{d} teacher", lw=1)
            a2.plot(st, ln, label=d, lw=1)
            rows += [{"domain": d, "step": s, "kl_per_token": k, "mean_resp_len": l} for s, k, l in zip(st, kl, ln)]
        a1.set_title(f"{name}: reverse KL(student||teacher) per response token"); a1.set_xlabel("MOPD step (32 rollouts)")
        a1.set_ylabel("nats / token"); a1.legend(fontsize=8)
        a2.set_title("mean sampled response length"); a2.set_xlabel("MOPD step"); a2.set_ylabel("tokens"); a2.legend(fontsize=8)
        fig.tight_layout()
        fname = "stage2_metrics.png" if name == "expert_mopd" else f"{name}_metrics.png"
        fig.savefig(out / fname, dpi=130); plt.close(fig)
        with open(out / "source" / f"{name}_metrics.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["domain", "step", "kl_per_token", "mean_resp_len"]); w.writeheader(); w.writerows(rows)
    print(f"curves written to {out}")


if __name__ == "__main__":
    main()
