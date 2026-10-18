"""Audit expert training data for benchmark contamination.

For every expert's train.jsonl + validation.jsonl:
  - exact duplicates and near-duplicates inside the split (8-gram Jaccard >= 0.5,
    candidate pairs found through an inverted index instead of all O(n^2) pairs)
  - canary strings that benchmarks embed to mark "never train on this"

Against every official evaluation set (all experts vs. all benchmarks, because the
merged model inherits all training data):
  - 13-gram containment: for each test item, the fraction of its 13-grams that
    appear anywhere in the training text. An item is flagged when containment
    >= 0.5 (the GPT-3 / Llama decontamination style criterion).

Reference sets are built by ``--build_references`` from the local harness copies:
  hle      data/eval/hle_source/test.parquet (all 2500 questions)
  swe      third_party/SWE-bench_Pro-os/v2/tasks/*/instruction.md (642)
  terminal third_party/terminal-bench/*/instruction.md (89)
  asi      data/eval/asi_seed31415/*/prompt_b*.md (60 x 4 levels)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

EXPERTS = ["swe", "hle", "asi", "terminal"]
DUP_NGRAM = 8
NEAR_DUP_THRESHOLD = 0.5
TEST_NGRAM = 13
CONTAINMENT_THRESHOLD = 0.5
CANARY_MARKERS = (
    "BENCHMARK DATA SHOULD NEVER APPEAR",
    "canary GUID",
    "terminal-bench-canary",
)


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    return set(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else set()


def sample_text(sample: Dict) -> str:
    return " ".join(m.get("content", "") for m in sample.get("messages", []))


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# reference sets
# ---------------------------------------------------------------------------

def build_references(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    refs: Dict[str, List[Dict]] = defaultdict(list)

    hle = Path("data/eval/hle_source/test.parquet")
    if hle.exists():
        import pandas as pd

        for row in pd.read_parquet(hle, columns=["id", "question"]).itertuples():
            refs["hle"].append({"id": row.id, "text": row.question})
    for p in sorted(Path("third_party/SWE-bench_Pro-os/v2/tasks").glob("*/instruction.md")):
        refs["swe"].append({"id": p.parent.name, "text": p.read_text(errors="ignore")})
    for p in sorted(Path("third_party/terminal-bench").glob("*/instruction.md")):
        refs["terminal"].append({"id": p.parent.name, "text": p.read_text(errors="ignore")})
    for p in sorted(Path("data/eval/asi_seed31415").glob("*/prompt_b*.md")):
        refs["asi"].append({"id": f"{p.parent.name}/{p.stem}", "text": p.read_text(errors="ignore")})

    for name, rows in refs.items():
        with open(out_dir / f"{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"reference {name}: {len(rows)} items")


# ---------------------------------------------------------------------------
# audits
# ---------------------------------------------------------------------------

def near_duplicates(texts: List[str]) -> Tuple[int, int]:
    """Return (exact_duplicates, near_duplicate_pairs) using an 8-gram inverted index."""
    exact = len(texts) - len(set(texts))
    grams = [ngrams(tokenize(t), DUP_NGRAM) for t in texts]
    index: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for i, g in enumerate(grams):
        for gram in g:
            index[gram].append(i)
    shared: Dict[Tuple[int, int], int] = defaultdict(int)
    for ids in index.values():
        if len(ids) > 50:  # boilerplate n-gram shared by many samples; not evidence of duplication
            continue
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                shared[(ids[a], ids[b])] += 1
    pairs = 0
    for (i, j), inter in shared.items():
        union = len(grams[i]) + len(grams[j]) - inter
        if union and inter / union >= NEAR_DUP_THRESHOLD:
            pairs += 1
    return exact, pairs


def containment(train_texts: Iterable[str], refs: List[Dict]) -> List[Dict]:
    """Per reference item: fraction of its 13-grams found in the training text."""
    ref_grams = [ngrams(tokenize(r["text"]), TEST_NGRAM) for r in refs]
    wanted: Set[Tuple[str, ...]] = set().union(*ref_grams) if ref_grams else set()
    seen: Set[Tuple[str, ...]] = set()
    for t in train_texts:
        seen.update(g for g in ngrams(tokenize(t), TEST_NGRAM) if g in wanted)
    out = []
    for r, g in zip(refs, ref_grams):
        frac = len(g & seen) / len(g) if g else 0.0
        out.append({"id": r["id"], "containment": frac})
    return out


def main():
    parser = argparse.ArgumentParser(description="Benchmark contamination audit for expert training data")
    parser.add_argument("--train_dir", default="data")
    parser.add_argument("--reference_dir", default="data/eval/references")
    parser.add_argument("--build_references", action="store_true")
    parser.add_argument("--output", default="results/contamination_report.json")
    args = parser.parse_args()

    ref_dir = Path(args.reference_dir)
    if args.build_references or not ref_dir.exists():
        build_references(ref_dir)
    references = {p.stem: load_jsonl(p) for p in sorted(ref_dir.glob("*.jsonl"))}

    report = {"thresholds": {"near_dup_jaccard_8gram": NEAR_DUP_THRESHOLD,
                             "test_containment_13gram": CONTAINMENT_THRESHOLD},
              "references": {k: len(v) for k, v in references.items()},
              "experts": []}
    for expert in EXPERTS:
        samples = []
        for split in ("train", "validation"):
            samples += load_jsonl(Path(args.train_dir) / expert / f"{split}.jsonl")
        texts = [sample_text(s) for s in samples]
        print(f"\n{expert}: {len(texts)} train+validation samples")
        exact, near_pairs = near_duplicates(texts)
        canary_ids = [s.get("id") for s, t in zip(samples, texts)
                      if any(m.lower() in t.lower() for m in CANARY_MARKERS)]
        entry = {"expert": expert, "n_samples": len(texts), "exact_duplicates": exact,
                 "near_duplicate_pairs": near_pairs, "canary_hits": canary_ids, "vs_benchmarks": {}}
        for bench, refs in references.items():
            scores = containment(texts, refs)
            flagged = [s for s in scores if s["containment"] >= CONTAINMENT_THRESHOLD]
            entry["vs_benchmarks"][bench] = {
                "n_reference": len(refs),
                "n_flagged": len(flagged),
                "max_containment": round(max((s["containment"] for s in scores), default=0.0), 4),
                "mean_containment": round(sum(s["containment"] for s in scores) / len(scores), 6) if scores else 0.0,
                "flagged": sorted(flagged, key=lambda s: -s["containment"])[:20],
            }
            print(f"  vs {bench:9s}: flagged {len(flagged)}/{len(refs)}  "
                  f"max containment {entry['vs_benchmarks'][bench]['max_containment']:.3f}")
        print(f"  exact dups {exact}, near-dup pairs {near_pairs}, canary hits {len(canary_ids)}")
        report["experts"].append(entry)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
