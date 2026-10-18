"""Run HLE with the official centerforaisafety/hle scripts against a served model.

Steps:
1. Build a fixed evaluation set from the official ``cais/hle`` test split, read from a
   local parquet (``--source``; downloaded from the public ModelScope mirror
   ``cais/hle``, sha256 recorded) or from the gated HF repo as fallback: text-only questions (the base model has no vision input), optionally a
   seeded random subset of size ``--n``. The question IDs are written to
   ``<output>/question_ids.json`` so every model is scored on the same questions.
2. ``run_model_predictions.py`` (unchanged official script) against the model's
   OpenAI-compatible endpoint.
3. ``run_judge_results.py`` (unchanged official script) with the judge endpoint.
   The official judge is ``o3-mini-2025-01-31``; any other judge is recorded as a
   protocol deviation in results.json.

    python -m src.evaluation.hle_official --model merged_ties --base_url http://127.0.0.1:30000/v1 \
        --output results/merged_ties/hle --n 500
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HLE_DIR = Path("third_party/hle/hle_eval")
OFFICIAL_JUDGE = "o3-mini-2025-01-31"


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_eval_set(dataset_dir: Path, n: int, seed: int, source: Path) -> list:
    from datasets import load_dataset

    if (dataset_dir / "question_ids.json").exists():
        return json.loads((dataset_dir / "question_ids.json").read_text())
    if source.exists():
        ds = load_dataset("parquet", data_files=str(source), split="train")
        provenance = {"source": str(source), "sha256": sha256(source), "origin": "modelscope:cais/hle"}
    else:
        ds = load_dataset("cais/hle", split="test")
        provenance = {"source": "hf:cais/hle"}
    n_total = len(ds)
    ds = ds.filter(lambda q: not q["image"])
    n_text_only = len(ds)
    ids = sorted(ds["id"])
    if n and n < len(ids):
        ids = sorted(random.Random(seed).sample(ids, n))
    keep = set(ids)
    ds = ds.filter(lambda q: q["id"] in keep)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(dataset_dir / "test.parquet"))
    (dataset_dir / "question_ids.json").write_text(json.dumps(ids))
    (dataset_dir / "provenance.json").write_text(json.dumps(
        {**provenance, "n_total": n_total, "n_text_only": n_text_only, "n_selected": len(ids), "seed": seed, "text_only": True}, indent=2))
    print(f"HLE eval set: {len(ids)} text-only questions (seed={seed}) -> {dataset_dir}")
    return ids


def main():
    parser = argparse.ArgumentParser(description="Official HLE eval against a served model")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--base_url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=500, help="seeded subset of text-only questions (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_dir", default="data/eval/hle")
    parser.add_argument("--source", default="data/eval/hle_source/test.parquet")
    parser.add_argument("--prepare_only", action="store_true", help="build the eval set and exit")
    parser.add_argument("--judge_only", action="store_true", help="skip predictions (already saved), only judge")
    parser.add_argument("--skip_judge", action="store_true", default=os.environ.get("HLE_SKIP_JUDGE") == "1",
                        help="stop after predictions (judge later with --judge_only)")
    parser.add_argument("--max_completion_tokens", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=int(os.environ.get("HLE_WORKERS", "48")))
    parser.add_argument("--judge_workers", type=int, default=int(os.environ.get("HLE_JUDGE_WORKERS", "8")))
    parser.add_argument("--judge", default=os.environ.get("HLE_JUDGE_MODEL", OFFICIAL_JUDGE))
    parser.add_argument("--judge_base_url", default=os.environ.get("HLE_JUDGE_BASE_URL"))
    parser.add_argument("--judge_api_key_env", default=os.environ.get("HLE_JUDGE_API_KEY_ENV", "OPENAI_API_KEY"),
                        help="name of the env var holding the judge API key")
    args = parser.parse_args()

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir).resolve()
    ids = build_eval_set(dataset_dir, args.n, args.seed, Path(args.source).resolve())
    if args.prepare_only:
        return
    dataset_arg = str(dataset_dir)

    started = datetime.now().isoformat()
    # 1) predictions: official script writes hle_<model>.json into its cwd
    env = dict(os.environ, OPENAI_BASE_URL=args.base_url, OPENAI_API_KEY=os.environ.get("MODEL_API_KEY", "EMPTY"))
    if not args.judge_only:
        subprocess.run([sys.executable, str((HLE_DIR / "run_model_predictions.py").resolve()),
                        "--dataset", dataset_arg, "--model", args.model,
                        "--max_completion_tokens", str(args.max_completion_tokens),
                        "--num_workers", str(args.num_workers)], cwd=out, env=env, check=True)
    predictions = out / f"hle_{args.model}.json"
    if args.skip_judge:
        print(f"predictions saved to {predictions}; judging deferred (--judge_only later)")
        return

    # 2) judge
    judge_env = dict(os.environ)
    if args.judge_base_url:
        judge_env["OPENAI_BASE_URL"] = args.judge_base_url
    if os.environ.get(args.judge_api_key_env):
        judge_env["OPENAI_API_KEY"] = os.environ[args.judge_api_key_env]
    judge_log = out / "judge_stdout.txt"
    # Non-o3 judges go through the compat wrapper (JSON hint + retries); the official
    # judge script itself is imported unchanged.
    use_compat = args.judge != OFFICIAL_JUDGE or bool(args.judge_base_url)
    judge_cmd = ([sys.executable, "-m", "src.evaluation.hle_judge_compat"] if use_compat
                 else [sys.executable, str((HLE_DIR / "run_judge_results.py").resolve())])
    judge_cmd += ["--dataset", dataset_arg, "--predictions", str(predictions),
                  "--judge", args.judge, "--num_workers", str(args.judge_workers)]
    judge_env["PYTHONPATH"] = str(Path.cwd())
    with open(judge_log, "w") as fh:
        subprocess.run(judge_cmd, cwd=out, env=judge_env, check=True, stdout=fh, stderr=subprocess.STDOUT)
    text = judge_log.read_text()

    judged = json.loads((out / f"judged_{predictions.name}.json").read_text())
    per_item = [{"id": qid, "is_correct": judged[qid]["judge_response"]["correct"] == "yes"}
                for qid in ids if qid in judged]
    acc = re.search(r"Accuracy:\s*([\d.]+)%", text)
    results = {
        "benchmark": "hle",
        "protocol": "official scripts; text-only subset",
        "judge": args.judge,
        "official_judge": args.judge == OFFICIAL_JUDGE and not args.judge_base_url,
        "judge_base_url": args.judge_base_url,
        "judge_prompt_json_hint": use_compat,
        "n_judged": len(per_item),
        "n_questions": len(ids),
        "n_answered": len(per_item),
        "accuracy": (sum(r["is_correct"] for r in per_item) / len(ids)) if ids else None,
        "official_accuracy_line": acc.group(0) if acc else None,
        "started": started,
        "finished": datetime.now().isoformat(),
        "model": args.model,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    with open(out / "detailed_results.jsonl", "w") as fh:
        for r in per_item:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
