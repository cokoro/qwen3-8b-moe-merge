"""Pick a seeded random subset of SWE-bench Pro HARD-51 that fits a disk budget.

Image sizes come from the GHCR manifests (compressed layer bytes, linux/amd64). Tasks
are taken in a seeded random order; the first pass guarantees one task per repository,
the second fills up to ``--budget_gb``. Writes the ids (one per line) and a JSON report.

    python -m src.evaluation.swe_subset --ids v2/hard51_ids.txt --budget_gb 35 --out data/eval/swe_subset_ids.txt
"""

import argparse
import json
import random
import urllib.request
from pathlib import Path

REPO = "scaleapi/swe-bench_pro-v2"
ACCEPT = ("application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json,"
          "application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json")


def image_bytes(tag: str, token: str) -> int:
    def get(ref, accept=ACCEPT):
        req = urllib.request.Request(f"https://ghcr.io/v2/{REPO}/manifests/{ref}",
                                     headers={"Authorization": f"Bearer {token}", "Accept": accept})
        return json.load(urllib.request.urlopen(req, timeout=30))
    m = get(tag)
    if "manifests" in m:
        d = [x for x in m["manifests"] if x.get("platform", {}).get("architecture") == "amd64"][0]
        m = get(d["digest"], d["mediaType"])
    return sum(layer["size"] for layer in m["layers"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True)
    parser.add_argument("--budget_gb", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ids = [line.strip() for line in open(args.ids) if line.strip()]
    token = json.load(urllib.request.urlopen(f"https://ghcr.io/token?scope=repository:{REPO}:pull"))["token"]
    sizes = {i: image_bytes(i, token) for i in ids}
    repo_of = {i: i.split("__")[1].split("-")[0] for i in ids}
    order = ids[:]
    random.Random(args.seed).shuffle(order)
    budget = args.budget_gb * 1e9
    chosen, used = [], 0
    for i in order:  # pass 1: one per repository
        if repo_of[i] not in {repo_of[c] for c in chosen} and used + sizes[i] <= budget:
            chosen.append(i); used += sizes[i]
    for i in order:  # pass 2: fill the budget
        if i not in chosen and used + sizes[i] <= budget:
            chosen.append(i); used += sizes[i]
    chosen.sort()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(chosen) + "\n")
    report = {"seed": args.seed, "budget_gb": args.budget_gb, "n_selected": len(chosen), "n_total": len(ids),
              "selected_gb": round(used / 1e9, 2), "total_gb": round(sum(sizes.values()) / 1e9, 2),
              "repos": sorted({repo_of[c] for c in chosen}), "sizes_gb": {i: round(sizes[i] / 1e9, 3) for i in chosen}}
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"selected {len(chosen)}/{len(ids)} tasks, {used/1e9:.1f} GB, repos={len(report['repos'])}")


if __name__ == "__main__":
    main()
