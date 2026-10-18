# 模型清单、合版配方与加载方法

所有模型都基于同一个基座 **Qwen/Qwen3-8B @ `b968826d9c46dd6066d109eabc6255188de91218`**。5 个权重分片都核对过 sha256，见 `models/Qwen3-8B/verification.json`。

- tokenizer 和基座相同。
- chat template 用的是 `configs/chat_templates/qwen3_nothink.jinja`：在官方模板基础上，把 `enable_thinking` 的默认值改成 false。训练和评测都用 non-thinking 模式。
- 不做量化，全程 bf16。

| 模型 | 类型 | 本地路径 | Hugging Face（Public） | 配方 / run_id |
|---|---|---|---|---|
| base | 原始权重 | `models/Qwen3-8B` | https://huggingface.co/Qwen/Qwen3-8B（revision 同上） | — |
| expert_swe | LoRA r16 | `checkpoints/expert_swe` | https://huggingface.co/cokoro/qwen3-8b-moe-merge-experts，子目录 `expert_swe` | `train_swe` |
| expert_hle | LoRA r16 | `checkpoints/expert_hle` | 同上，子目录 `expert_hle` | `train_hle` |
| expert_asi | LoRA r16 | `checkpoints/expert_asi` | 同上，子目录 `expert_asi` | `train_asi` |
| expert_terminal | LoRA r16 | `checkpoints/expert_terminal` | 同上，子目录 `expert_terminal` | `train_terminal` |
| merged_linear | 合版（权重平均） | `checkpoints/merged_linear`（完整权重）；`adapters/merged_linear`（等价的 rank-64 LoRA） | https://huggingface.co/cokoro/qwen3-8b-moe-merge-merged，子目录 `merged_linear` | `merge_linear` |
| **merged_ties（最终合版模型）** | 合版（TIES，density 0.2） | `checkpoints/merged_ties` | 同上，子目录 `merged_ties` | `merge_ties` |
| mopd | 自设计合版：多教师 on-policy 蒸馏（LoRA r16） | `checkpoints/expert_mopd` | 同上，子目录 `mopd` | `train_mopd` |

> 两个仓库都是 Public，可以直接下载，不需要申请权限：
> - `cokoro/qwen3-8b-moe-merge-experts`，revision `57c57131393fe12196496c76fa0ba71b2978f283`：4 个专家 adapter
> - `cokoro/qwen3-8b-moe-merge-merged`，revision `64ae5b49dfb5cc46a32fd922052529e1c4c25228`：merged_ties、merged_linear（完整权重 + 等价的 rank-64 LoRA `merged_linear_lora64/`）、mopd
>
> 上面两个 revision 是上传权重时的 commit。之后只更新过模型卡（README），对应的最新 revision 分别是 `fa6e1e38d6f4d0db047e8267a703d5e5a34f4f42` 和 `71497d96aaf3b9b5263b85e81ca0f820e8cb6085`，权重文件完全相同。
>
> **最终合版模型是 merged_ties**，选择依据见 README §2。v2 专家（`expert_terminal_v2`、`expert_swe_v2`，见 README §12）已训练完成，但因为 x86 评测主机停用，没有评测结果，所以没有上传。

## 合版配方（merge recipe）

4 个专家都在同一个冻结基座上训练 LoRA，所以每个专家对某个目标权重 W 的增量可以写成 ΔW_i = B_i A_i · (α/r) = 2·B_i A_i（α=32，r=16）。各方法的区别只在于怎样把这些增量合起来。

- **merged_linear**：W = W₀ + Σ_i w_i ΔW_i，其中 w_i = 0.25（4 个专家等权）。这和「先把每个专家各自合进基座，再对 4 个完整模型做平均」是同一回事。实现：`src/merging/merge_linear.py`，配置：`configs/merging.yaml`。
  - 这个合版也可以**精确地**写成一个 rank-64 的 LoRA：B = [w_i s_i B_i] 沿秩方向拼接，A = [A_i] 纵向堆叠（`src/merging/linear_as_lora.py`）。存成 bf16 后，相对误差在 2e-3 以内。x86 机器上的评测就是用这种形式，由 sglang 多 LoRA 挂载。
- **merged_ties**：对每个目标张量分三步：
  1. 每个专家只保留 |ΔW_i| 最大的 20%；
  2. 按符号统计「多数派」：把裁剪后的增量求和，看正负；
  3. 只对和多数派同号的元素求平均（disjoint mean），再加回 W₀。

  实现：`src/merging/merge_ties.py`（`--density 0.2`）。
- **mopd**：学生 = 基座 + 一个新的 LoRA（r16，7 个投影层），教师 = expert_hle 和 expert_asi。具体做法：
  - 学生先对自己领域的 prompt 采样回答（temperature 1.0，最多 320 token）；
  - 再在这些回答上，最小化每个 token 在完整词表上的反向 KL(p_student ‖ p_teacher_d)。
  - 设置：每步每个领域 16 条，学习率 1e-4，共 152 步（150 分钟）。实现：`src/distill/mopd.py`，配置：`configs/stage2.yaml`。

Linear 和 TIES 两种合版与 PEFT 官方 `merge_and_unload` 的结果逐位一致，验证脚本是 `tests/test_merge_equivalence.py`。

## 加载方法

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id, rev = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"
tok = AutoTokenizer.from_pretrained(base_id, revision=rev)
tok.chat_template = open("configs/chat_templates/qwen3_nothink.jinja").read()
model = AutoModelForCausalLM.from_pretrained(base_id, revision=rev, dtype=torch.bfloat16, device_map="auto")

# 专家 / MOPD：加载 LoRA
model = PeftModel.from_pretrained(model, "cokoro/qwen3-8b-moe-merge-experts", subfolder="expert_swe")
# 合版完整权重：直接加载对应子目录，例如 AutoModelForCausalLM.from_pretrained("cokoro/qwen3-8b-moe-merge-merged", subfolder="merged_ties")
```

用 sglang 部署时，一个基座进程挂多个 LoRA：

```bash
python -m sglang.launch_server --model-path models/Qwen3-8B --served-model-name base \
  --enable-lora --max-lora-rank 64 --lora-paths expert_swe=adapters/expert_swe merged_linear=adapters/merged_linear \
  --chat-template configs/chat_templates/qwen3_nothink.jinja
# 请求时把 model 写成 "base:merged_linear"
```

**许可证**：基座 Qwen3-8B 采用 Apache-2.0。各训练数据的许可证见 `data/README.md`，Adapter 和合版权重沿用 Apache-2.0。
