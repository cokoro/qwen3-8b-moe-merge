# 训练数据：来源、许可、清洗、去重、污染检查

生成脚本是 `scripts/prepare_data.sh`，会调用 `src/data/prepare_{swe,hle,asi,terminal}.py` 和 `src/data/common.py`。

- 每个专家目录下生成 `train.jsonl`、`validation.jsonl`（5% 留作开发集，seed 42）和 `stats.json`；
- 每个文件的 sha256 记在 `data/manifests/manifest.json`。

| 专家 | 来源（固定版本） | 许可 | train / val | 平均长度（字符） | 说明 |
|---|---|---|---|---|---|
| swe | `princeton-nlp/SWE-bench` 的 **train** split：issue → gold patch | MIT | 4750 / 250 | 4056 | 只保留 issue+patch 不超过 12k 字符的样本；去掉 hints（它常泄露修复讨论）；和 SWE-bench Pro 的仓库、任务不重叠 |
| hle | `openai/gsm8k` 的 main/train（**代理数据**） | MIT | 2850 / 150 | 512 | HLE 本身没有许可用于训练的数据，HLE 官方仓库也明确禁止拿 benchmark 训练，所以用小学数学推理作代理 |
| asi | `tau/commonsense_qa` 的 train，套「Step 1–4 + Answer」规划模板，再加 10 条手写的工具规划样本（**代理数据**） | MIT | 3800 / 200 | 490 | 格式高度模板化，验证集 loss 降到 0.006，说明模型只是背下了模板，这是已知的弱点 |
| terminal | `westenfelder/NL2SH-ALFA` 的 train（NL→bash）+ 67 个 `camel-ai/SETA-Env` 任务（instruction → solve.sh，来自已人工审核、不含 canary 的候选集）+ 6 条手写 | MIT / Apache-2.0 | 1900 / 100 | 249 | 带 canary 标记的任务全部排除；SETA 上游的逐条来源链还没逐一核实 |

**配比为什么不一样**：
- SWE 样本长（平均约 1k token），每条信号多，所以给了最多的条数；
- terminal 的公开、可用、许可清楚的数据最少；
- 但 4 个专家统一训练 500 步 × 有效 batch 8，也就是每个专家看到的样本数相同，所以样本数差异不会变成训练强度的差异。

**清洗与去重**（`src/data/common.py`）：
- 按 user prompt 去完全重复：先去掉空白、统一小写再比对；
- 过长样本直接丢弃，不做截断；
- 最终每个领域 8-gram Jaccard ≥ 0.5 的近重复对数：swe 15 对，其余为 0。

**难度分层**：没有单独做。各数据源本身难度比较单一（GSM8K、CSQA 偏简单；SWE-bench train 是真实 issue），这是一个局限。

**防污染**（`src/data/audit_contamination.py`，报告在 `results/contamination_report.json`）：
- 把 4 个专家的 train+val 全部，和 4 个官方评测集逐一比较：HLE 全部 2500 题、SWE-bench Pro v2 全部 642 个 instruction、Terminal-Bench 2.0 全部 89 个 instruction、ASI seed31415 的 240 个 prompt。
- 计算 13-gram 包含率，即测试题的 13-gram 有多少比例出现在训练文本里；包含率 ≥ 0.5 就标记为污染。
- 结果：**0 条被标记**，最大包含率 0.035（swe 对 hle）。canary 字符串命中 0 条。

**其他保证**：
- 评测的题目、答案、参考补丁、隐藏测试和测试反馈，都没有用于训练或挑选 checkpoint。专家一律取第 500 步，不做 checkpoint 挑选。
- MOPD 的 prompt 只来自上述训练集。
