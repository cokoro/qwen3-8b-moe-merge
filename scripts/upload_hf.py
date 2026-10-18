"""Upload the trained checkpoints to the candidate's Hugging Face account as PUBLIC repos.

  <user>/qwen3-8b-moe-merge-experts : expert_{swe,hle,asi,terminal}/ (LoRA adapters)
  <user>/qwen3-8b-moe-merge-merged  : merged_linear/ merged_ties/ (full bf16 weights),
                                      merged_linear_lora64/ (exact rank-64 LoRA form), mopd/ (LoRA)
Each repo gets a model card (base revision, recipe, data, eval pointers, license). Prints the
commit revision of every upload so it can be pasted into models/README.md.

    HF_TOKEN=... python scripts/upload_hf.py --user <hf_user>
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

BASE = "Qwen/Qwen3-8B"
BASE_REV = "b968826d9c46dd6066d109eabc6255188de91218"

CARD = """---
license: apache-2.0
base_model: Qwen/Qwen3-8B
library_name: {library}
tags: [lora, model-merging, distillation, qwen3]
---
# {title}

Base: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) @ `{rev}` (bf16, non-thinking chat template
`qwen3_nothink.jinja` from the experiment repo). Part of a multi-expert training + merging experiment on
SWE-bench Pro v2-hard, HLE, ASI-Bench and Terminal-Bench 2.0.

Contents:
{contents}

Recipes, data sources/licenses, evaluation protocol and results: see the experiment repository
(models/README.md, data/README.md, configs/eval/benchmarks.yaml, results/summary.csv).

Load a LoRA subfolder:
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", revision="{rev}", dtype="bfloat16")
m = PeftModel.from_pretrained(m, "{repo}", subfolder="<subfolder>")
```
Full-weight subfolders load with `AutoModelForCausalLM.from_pretrained("{repo}", subfolder="<subfolder>")`.

Limitations: experts were trained on small/proxy data (GSM8K for HLE, CommonsenseQA templates for
ASI); scores on all four benchmarks are low for this 8B base. Not for production use.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--checkpoints", default="checkpoints")
    args = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    ck = Path(args.checkpoints)

    repos = {
        f"{args.user}/qwen3-8b-moe-merge-experts": (
            "LoRA experts", "peft",
            {f"expert_{e}": ck / f"expert_{e}" for e in ["swe", "hle", "asi", "terminal"]},
        ),
        f"{args.user}/qwen3-8b-moe-merge-merged": (
            "Merged models (linear, TIES) and MOPD student", "transformers",
            # final merged model first; merged_linear ships as its exact rank-64 LoRA form when available
            {"merged_ties": ck / "merged_ties", "mopd": ck / "expert_mopd",
             "merged_linear_lora64": Path("adapters/merged_linear"), "merged_linear": ck / "merged_linear"},
        ),
    }
    allow = ["*.json", "*.safetensors", "*.jinja", "*.txt", "tokenizer*", "merges.txt", "vocab.json", "*.jsonl"]
    for repo, (title, lib, parts) in repos.items():
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        contents = "\n".join(f"- `{k}/`" for k in parts)
        api.upload_file(path_or_fileobj=CARD.format(title=title, library=lib, rev=BASE_REV, contents=contents,
                                                    repo=repo).encode(), path_in_repo="README.md", repo_id=repo)
        for sub, path in parts.items():
            if not path.exists():
                print(f"skip missing {path}")
                continue
            info = api.upload_folder(folder_path=str(path), path_in_repo=sub, repo_id=repo,
                                     allow_patterns=allow, ignore_patterns=["checkpoint-*", "_save*"])
            print(f"{repo}/{sub}: {info.oid if hasattr(info, 'oid') else info}", flush=True)
        print(f"https://huggingface.co/{repo}  revision: {api.model_info(repo).sha}")


if __name__ == "__main__":
    main()
