# 多专家训练与模型合版：Qwen3-8B × 4 专家 × {Linear, TIES, MOPD}

- 仓库：https://github.com/cokoro/qwen3-8b-moe-merge
- 模型：见 [`models/README.md`](models/README.md)（Hugging Face Public 链接: https://huggingface.co/cokoro/qwen3-8b-moe-merge-experts）
- 一页报告：[`report.pdf`](report.pdf)

## 1. 研究问题

能否把「各自专精一个 benchmark」的 LoRA 专家合成**一个**模型，并且在四项 benchmark（SWE-bench Pro v2-hard、HLE、ASI-Bench、Terminal-Bench 2.0）上，相对 Base 的等权宏平均提升 ≥3 个百分点？

在**同一批专家**上，比较三种得到单一模型的方式：
1. **Linear 权重平均**（task arithmetic，等权）；
2. **TIES**：剪枝 + 符号选举，用来处理参数冲突；
3. **MOPD**：自行设计的「蒸馏式合版」。不合并参数，让学生在自己的 on-policy 采样上，逐 token 拟合对应领域专家的分布。

我们事先的判断是：权重合并靠人定的规则（系数、剪枝率、符号投票）去调和参数，既保不住各专家的能力，还可能破坏基座原有能力；蒸馏让单一模型直接从专家的分布中学，更有希望。

## 2. 实验矩阵与完成情况

| 模型 \ benchmark | HLE（500 题，纯文本） | ASI-Bench（60，seed31415） | Terminal-Bench 2.0（74） | SWE-bench Pro HARD-51 子集 |
|---|---|---|---|---|
| base | ✅ | ✅ | ✅ | ✅ |
| expert_hle / expert_asi | ✅ 本专家对应项 | ✅ 本专家对应项 | — | — |
| expert_terminal | — | — | ✅ 本专家对应项 | — |
| expert_swe | — | — | — | ✅ 本专家对应项 |
| merged_linear | ✅ | ✅ | ✅ | ✅ |
| **merged_ties（最终合版模型）** | ✅ | ✅ | ✅ | ✅ |
| mopd | ✅ | ✅ | ✅ | ✅ |

- 必做项（Base 和合版模型完成全部四项；每个专家完成本专家对应的 benchmark）都已覆盖。三种单模型方法（Linear、TIES、MOPD）都跑完了全部四项。
- **最终合版模型 = merged_ties**。选择规则只用开发集：在 4 个专家的验证集上（每个领域取前 100 条，训练和评测都没用过），计算平均 completion loss，最低者入选（`src/evaluation/dev_loss.py` → `analysis/dev_selection.json`）。结果：merged_ties 0.494，merged_linear 0.546，mopd 1.139，base 2.417。**说明**：这条规则是在看到 HLE/ASI 的部分测试结果之后才定的（原计划没有写明最终模型怎么选，是个疏漏）。规则本身不依赖测试数据，而它选中的恰好不是测试分数最高的那个；所有候选模型的结果也都完整报告。
- merged_ties 无法写成低秩 LoRA，不能和其他模型共用 x86 上的多 LoRA 推理服务。所以它的 TB2/SWE 评测是：推理服务放在 DGX 上，通过 SSH 反向隧道接到 x86 主机，harbor 和 agent 设置完全相同（`scripts/serve_ties_remote.sh`、`scripts/run_x86_ties_remote.sh`）。推理硬件不同（GB10 vs L40S）会影响速度，因此可能影响 TB2 的超时率；另外 05:15–05:35 期间，开发集 loss 的计算也和它共用 DGX 的 GPU。这两点在 §8 列出。

## 3. 硬件、时间与成本

| 资源 | 用途 | 规格 |
|---|---|---|
| NVIDIA DGX Spark（自有） | 数据、专家训练、合版、MOPD、HLE/ASI 评测、本地 judge | GB10，aarch64，128GB 统一内存，1 GPU |
| 租用 x86 GPU 容器 | TB2 / SWE 评测（镜像只有 amd64） | 1× L40S 48GB，224 vCPU，容器内存上限 128GB，非特权 k8s pod |

费用估算：DGX 按 $1.5/GPU·h 折算（自有设备，不另付费）；租用机按实际租期计费，见 `logs/resources.csv`。各训练 / 评测 run 的起止时间由 `scripts/timed_run.sh` 记录在 `logs/experiment_times.csv`。

| 项目 | GPU·h | 估算费用 | 说明 |
|---|---|---|---|
| 专家训练（4 个） | 4.09 | $6.1 | DGX，`train_{swe,hle,asi,terminal}` |
| 合版 + 折叠 | 0.25 | $0.4 | DGX，`merge_*`、`fold_*` |
| MOPD | 2.56 | $3.8 | DGX，`train_mopd`（含 1 次按计划停掉的 batch 8 尝试） |
| HLE / ASI 评测 | 15.6 | $23.3 | DGX，`eval_*`（ASI 的评分阶段占大头） |
| 本地 judge、HLE 重复运行、开发集选择、merged_ties 远程推理 | ≈8.1 | ≈$12.2 | DGX，未经 `timed_run.sh` 计时，按日志时间戳估算 |
| TB2 / SWE 评测 | 20.3（多个 run 共用 1 张卡，墙钟约 14 h） | ≈$14 | 租用 L40S，按 $1.0/h × 墙钟估算；实际以租用账单为准 |
| **合计** | DGX ≈30.6，L40S 墙钟 ≈14 h | **≈$60** | |

- 每个 run 的 GPU·h、峰值显存、退出码和尝试次数见 `logs/resources.csv`；计算方式见 `logs/README.md`。
- 平均每题成本（推理时间）：HLE 约 3–4 s/题（48 并发，GB10），ASI 约 1–4 min/题（含评分），TB2 每个模型 2–5 h / 74 题（12 并发），SWE 每个模型 20–90 min / 6 题。

## 4. 基座与专家设计

- **基座**：Qwen/Qwen3-8B @ `b968826d9c46dd6066d109eabc6255188de91218`，bf16，tokenizer 和基座一致，non-thinking chat template（`configs/chat_templates/qwen3_nothink.jinja`）。
- **为什么不用 Qwen3.6-35B-A3B**：它是 MoE，总参数约 35B。显存要按全部权重算（bf16 约 70GB），不能按 3B 激活参数估。再加上 LoRA 训练和 4 个专家、多轮评测的时间预算，在单卡 GB10 上不可行。题目允许降到 7B 级，所以选同系列的 8B。
- **专家划分**：4 个专家，每个对应一个 benchmark，模型大小都和基座相同（8B + LoRA）。按 benchmark 划分，可以直接回答两个问题：合版后每项保留了多少，各专家之间有没有干扰。
- **训练方式**：用 bf16 LoRA，不用 QLoRA。统一内存放得下 8B bf16，aarch64 上也没有可靠的 bitsandbytes；量化噪声还会混进要合并的增量里。

| 超参 | 值 |
|---|---|
| target modules | q,k,v,o,gate,up,down 共 7 个投影层；r=16，α=32，dropout 0.05 |
| 学习率 / 调度 | 2e-4，cosine，warmup 3% |
| batch | 1 × 梯度累积 8 |
| 步数 | 500（所有专家相同，保证训练强度一致） |
| 上下文 | 4096 |
| loss | 只算 assistant 回答部分（trl `completion_only_loss`） |
| 优化器 | AdamW (fused)，grad clip 0.3 |
| seed | 42 |
| checkpoint 选择 | 固定取最终 step，不按测试集挑 |

**每个专家的训练资源**（DGX GB10，单卡；数据来自 `checkpoints/expert_*/training_metrics.json`）：

| 专家 | train / val 条数 | 训练 token | 时间 | 峰值显存 | 吞吐 | adapter 大小 | final val loss |
|---|---|---|---|---|---|---|---|
| swe | 4750 / 250 | 4.39M | 2.51 h | 20.0 GB | 486 tok/s | 175 MB | 0.725 |
| hle | 2850 / 150 | 0.77M | 0.52 h | 17.5 GB | 412 tok/s | 175 MB | 0.307 |
| asi | 3800 / 200 | 0.51M | 0.45 h | 17.3 GB | 314 tok/s | 175 MB | 0.006 |
| terminal | 1900 / 100 | 0.32M | 0.44 h | 19.3 GB | 201 tok/s | 175 MB | 0.614 |

曲线见 `curves/sft_loss.png`，源数据在 `curves/source/sft_loss.csv`。

**可解释的对照**：Linear 与 TIES 对比（合版方法）；权重合版与 MOPD 对比（参数合并还是蒸馏）；以及各专家和合版模型在本专家对应 benchmark 上的对比（合版保留了多少）。

## 5. 数据

详见 [`data/README.md`](data/README.md)，包括来源、许可、清洗、去重和污染检查。**独立污染审计**（`analysis/contamination_review.md`）做了四类检查：全字段 n-gram / 最长公共子串，覆盖 solution、tests、gold patch、答案；embedding 近邻；专家与 base 的记忆探测，带正对照；选模和流程泄漏。**未发现训练数据污染，也未发现测试信息泄漏**；确认存在的是评测覆盖与有效性问题，见 §8。**HLE 和 ASI 用的是代理数据**（GSM8K、CommonsenseQA 模板），这是本实验最主要的局限，见 §8。

## 6. 合版方法

详见 [`models/README.md`](models/README.md) 中的合版配方。要点：
- 所有合并都在 LoRA 增量 ΔW 空间里逐个目标模块计算，峰值只需一份基座（约 16GB），不用把 4 个完整模型同时放进内存（约 110GB）。和 PEFT `merge_and_unload` 的结果逐位一致（`tests/test_merge_equivalence.py`）。
- **参数冲突的处理**：Linear 不处理，直接平均，冲突的方向会互相抵消。TIES 先剪掉每个专家 80% 的小幅值更新，再按「总量多数派」选出每个元素的符号，只对同号的元素求平均。MOPD 不做参数合并，冲突交给学生在输出分布层面去拟合。
- **MOPD**：两个教师 expert_hle 和 expert_asi，即 HLE 与 ASI 这两项上有训练信号的专家。学生 = 基座 + 一个新 LoRA，共 152 步，每步 32 条 on-policy 采样。损失是逐 token、在完整词表上计算的反向 KL。曲线见 `curves/stage2_metrics.png`。

## 7. 评测协议与结果

固定配置写在 [`configs/eval/benchmarks.yaml`](configs/eval/benchmarks.yaml)，其中每一处偏离官方 / 模型卡设置的地方都标了 `DEVIATION`。同一个 benchmark 内，所有模型的 thinking 模式（关闭）、量化（bf16）、工具、上下文和预算都相同。

<!-- BEGIN RESULTS_TABLES (generated: scripts/render_results.py) -->
**表 1：主指标（0–100）**。macro4 = 四项等权平均；「—」= 未评测。n = 题目数。

| 模型 | HLE (n) | ASI v2 (n) | TB2 (n) | SWE-Pro (n) | **macro4** | HLE 原始 judge | HLE 选择题 | ASI v1（环境缺包，仅参考） |
|---|---|---|---|---|---|---|---|---|
| base | 2.80 (500) | 0.45 (60) | 0.00 (74) | 0.00 (4) | **0.81** | 4.40 | 5.43 | 1.20 |
| expert_hle | 4.20 (500) | — | — | — | **—** | 4.20 | 8.53 | — |
| expert_asi | — | 0.00 (60) | — | — | **—** | — | — | 0.21 |
| expert_terminal | — | — | 0.00 (74) | — | **—** | — | — | — |
| expert_swe | — | — | — | 0.00 (4) | **—** | — | — | — |
| merged_linear | 4.80 (500) | 0.39 (60) | 1.35 (74) | 0.00 (4) | **1.64** | 11.20 | 8.53 | 0.03 |
| merged_ties | 3.40 (500) | 0.00 (60) | 0.00 (72) | 0.00 (4) | **0.85** | 6.80 | 7.75 | 0.00 |
| mopd | 3.40 (500) | 0.14 (60) | 0.00 (74) | 0.00 (4) | **0.88** | 4.00 | 6.98 | 0.14 |

**表 2：相对 base 的变化（百分点），配对 bootstrap 95% CI（10k 次，seed 42）**。

| 模型 | HLE | ASI v2 | TB2 | SWE-Pro | **macro4** | HLE 原始 judge | HLE 选择题 | ASI v1（环境缺包，仅参考） |
|---|---|---|---|---|---|---|---|---|
| expert_hle | +1.40 [-0.6, +3.4] | — | — | — | **—** | -0.20 [-2.4, +2.0] | +3.10 [-2.3, +8.5] | — |
| expert_asi | — | -0.45 [-1.3, +0.0] | — | — | **—** | — | — | -0.99 [-2.7, +0.0] |
| expert_terminal | — | — | +0.00 [+0.0, +0.0] | — | **—** | — | — | — |
| expert_swe | — | — | — | +0.00 [+0.0, +0.0] | **—** | — | — | — |
| merged_linear | +2.00 [-0.4, +4.4] | -0.05 [-0.4, +0.3] | +1.35 [+0.0, +4.1] | +0.00 [+0.0, +0.0] | **+0.82 [+0.0, +1.8]** | +6.80 [+3.4, +10.2] | +3.10 [-3.1, +10.1] | -1.17 [-3.1, +0.1] |
| merged_ties | +0.60 [-1.4, +2.8] | -0.45 [-1.3, +0.0] | +0.00 [+0.0, +0.0] | +0.00 [+0.0, +0.0] | **+0.04 [-0.5, +0.6]** | +2.40 [-0.2, +5.0] | +2.33 [-3.9, +8.5] | -1.20 [-3.1, +0.0] |
| mopd | +0.60 [-1.4, +2.6] | -0.31 [-1.3, +0.4] | +0.00 [+0.0, +0.0] | +0.00 [+0.0, +0.0] | **+0.07 [-0.5, +0.6]** | -0.40 [-2.6, +2.0] | +1.55 [-4.7, +7.8] | -1.06 [-2.9, +0.0] |

**表 3：重复运行波动**（相同题目、相同设置，第二次采样；TB2 第二次运行若未跑完，只在两次都完成的任务上比较）。

| benchmark | 模型 | n | 第 1 次 | 第 2 次 | 差值 | 逐题一致率 |
|---|---|---|---|---|---|---|
| hle | base | 500 | 2.80 | 3.60 | 0.80 | 0.956 |
| hle | merged_linear | 500 | 4.80 | 5.20 | 0.40 | 0.928 |
| terminal | base | 71 | 0.00 | 1.41 | 1.41 | 0.986 |
| terminal | merged_linear | 74 | 1.35 | 0.00 | 1.35 | 0.986 |

**运行失败（errored）**：base: asi=0, terminal=42, swe=1；expert_asi: asi=0；expert_terminal: terminal=49；expert_swe: swe=1；merged_linear: asi=0, terminal=22, swe=1；merged_ties: asi=0, terminal=26, swe=1；mopd: asi=0, terminal=27, swe=1

**SWE-Pro 计分任务**（gold patch 在本机通过）：4 个：`ansible__ansible-d33bedc48fdd933b5abd65a`, `flipt-io__flipt-e2bd19dafa7166c96b082fb2`, `gravitational__teleport-2b15263e49da5625`, `internetarchive__openlibrary-322d7a46cdc`
<!-- END RESULTS_TABLES -->

**结论：最终合版模型（merged_ties）相对 base 的宏平均变化是 +0.04 个百分点（95% CI [−0.5, +0.6]），没有达到 +3 的目标。** 三种单模型方法里最好的是 merged_linear：+0.82 [0.0, +1.8]，同样远低于目标，且其中 TB2 的 +1.35 在第二次运行中没有复现。MOPD 为 +0.07 [−0.5, +0.6]。（ASI 采用 v2 结果：v1 的评测环境缺少任务声明的依赖包，见 §8 第 4 条和 §12。）

逐项看（不隐藏退化）：

1. **HLE**：每个模型都比 base 略高，但置信区间都包含 0。merged_linear +2.0 [−0.4, +4.4]，第二次运行 +1.6，方向一致；expert_hle +1.4；TIES 和 MOPD 各 +0.6。注意 HLE 专家是用 GSM8K 训练的，提升主要体现在「给出一个短的最终答案」，而不是推理变强了。
2. **ASI（v2，修正评测环境后）**：所有模型都在地板上，60 题里最多 2 题拿到部分分。base 0.45、ASI 专家 0.00、Linear 0.39、TIES 0.00、MOPD 0.14，ASI 专家仍未超过 base（差 1 道题，CI [−1.3, 0]）。**更正**：先前版本写的「ASI 专家背下模板、不再写代码」是错的。专家对 60/60 题都写出了代码；失败原因是代码运行时报错。v1 里主要是环境缺少任务**已声明**的包（如 matplotlib）。v2 里，一部分是 import 了任务**未声明**的包（主要是 pandas：ASI 专家 13 次，base 4 次），按官方规则算作模型错误；其余是 SyntaxError、KeyError 等。
3. **TB2**：第一次运行只有 merged_linear 解出 1 题（`modernize-scientific-stack`）。但在第二次运行中，base 解出 1 题，merged_linear 是 0，所以这 1 题的差别是采样噪声。除此之外，base、expert_terminal、TIES、MOPD 都是 0/74。各模型的行为差别很大：base 每题中位输出 12.7k token，常常一直循环到 agent 超时（42/74）；所有合版模型都只输出约 1k token 就结束了（超时 22–27 次）。终端专家本身也没有提升（0/74，49 次超时）。NL→bash 的单轮数据教不会多轮 agent 行为。
4. **SWE-Pro**：在 4 个 gold patch 能在本机通过的任务上，所有模型都是 0。但这一项的评测配置有缺陷（patch 末尾换行被丢掉、agent 异常退出时修改丢失、32K 上下文不够、SWE 专家的训练格式与 agent 接口不匹配），详见 §8 第 6 条。修正 patch 换行后重新判分，base 仍然是 0（代码本身有编译和运行错误），但其他问题导致的 0 分不能说明模型的能力。

**评测本身的问题（重要）**：本地 8B judge 有大量假阳性。原始 judge 给 merged_linear 的 HLE 分数是 11.2%，审核后只有 4.8%：
- 27 个回答直接照抄了 prompt 里的格式模板 `Answer: {your chosen answer}`，judge 看到 prompt 里的标准答案，就把它当作模型的答案判为正确；
- 另外还有选择题字母不一致、数值不相等也被判对的情况。
如果不做审核，就会得出「Linear 合版 HLE +6.8 且显著」的错误结论。

**合版中的参数干扰（TIES）**：merged_ties 对 HLE 的 500 题中有 198 题（40%）直接回答一个 bash 代码块，没有任何答案。这是终端专家的输出格式。机制上的解释：Linear 把 4 个增量各乘 1/4，每个专家的影响都被稀释了；TIES 先按幅值剪枝，再对同号的增量做「不相交平均」，不再除以 4，所以每个专家的主方向几乎以全强度保留下来。哪个专家在某个方向上幅值最大，就会接管所有输入的输出格式。

**开发集选择规则的盲区**：teacher-forced 的验证 loss 选中了 TIES（在 HLE 验证集上它的 loss 也最低：0.356 vs 0.400），但它自由生成时会切换到错误的格式。loss 只衡量「给定正确答案的概率」，衡量不到「模型会选择哪种回答格式」。下次应该用小规模的生成式开发集来做选择。

**MOPD（蒸馏）没有带来额外收益**：HLE +0.6，ASI 也低于 base。学生忠实地复制了两个教师的行为，包括 ASI 教师在训练 prompt 上的模板化短回答：在 MOPD 的 ASI 训练 prompt（CSQA 代理题）上，回答长度从 320 token 降到约 43（`curves/stage2_metrics.png`）；在真实的 ASI 题目上，学生仍然会写代码。教师本身没有能力，蒸馏也传不出能力。这一点和我们事先的假设（蒸馏优于权重合并）都只是部分成立：MOPD 没有出现 TIES 那样的格式接管（HLE 中 0 个代码块回答，0 个模板回声），但在本实验的专家质量下，它也没有超过 Linear。 MOPD 与权重合并的对比（行为完整性证据、机制分析、为什么在分数上还没赢）见 [`docs/mopd_vs_merge.md`](docs/mopd_vs_merge.md)；相关文献见 [`docs/mopd_literature.md`](docs/mopd_literature.md)，包括小米 MOPD / MiMo-V2-Flash、DeepSeek-V4、Kimi K3、Qwen3、GKD、MiniLLM 等，以及反例和适用前提。

**下一步实验**（按预期收益排序）：
1. **换掉代理数据**：TB2 / SWE 用真实的多轮 agent 轨迹（如 SWE-smith、开源终端 agent 轨迹）做 SFT；HLE 用「Explanation / Answer / Confidence」格式的高难推理数据；ASI 用科学计算代码数据。先让每个专家在本领域显著超过 base，再比较合版方法，结论才有意义。
2. **用生成式开发集选模型、扫合版系数**：每个领域留 50 条做生成 + 格式检查，在上面扫 Linear 的系数和 TIES 的密度 / 缩放（例如 TIES 加上 1/k 缩放），代替 teacher-forced loss。
3. **蒸馏改为「多专家生成数据 → 更大的学生」**：用 4 个专家（以及 base）各自在本领域生成并过滤正确的轨迹，混合后对更大的学生（例如 Qwen3-14B）做 SFT，再做 on-policy 蒸馏，并把 4 个教师都纳入 MOPD。这样学生学的是专家的共性能力，不是参数平均。
4. **评测规范化**：用官方 judge（o3-mini）重判 HLE；TB2 按模型卡设置跑 5 次；SWE 跑完整的 HARD-51。

逐题 / 逐任务的结果在 `results/<model>/<benchmark>/detailed_results.jsonl`。汇总表 `results/summary.csv`、置信区间和重复运行波动 `analysis/uncertainty.json` 都由 `scripts/summarize.py` 从原始结果重新生成；表格由 `scripts/render_results.py` 生成。失败分析（1859 个失败逐条归类，分层抽样 63 例）见 [`analysis/failures.md`](analysis/failures.md)。HLE 维护方要求不要在网上公开题目和答案，所以公开仓库里的 HLE 标准答案和题目片段都已删除（`scripts/redact_hle_public.py`）。评分脚本从本地 gated 的 `data/eval/hle/test.parquet` 读取答案，可以完整复现。

## 8. 已知限制

按影响从大到小：

1. **专家训练数据是代理数据**：HLE 专家用 GSM8K，ASI 专家用 CommonsenseQA 套模板，和评测考察的能力差距很大。ASI 专家在代理验证集上的 loss 降到 0.006，说明它只学会了代理数据的模板，而不是 ASI 需要的科学计算编程能力。专家本身几乎没有学到目标能力，是总分上不去的根本原因。合版和蒸馏都只能重新分配专家已有的能力，没法凭空创造。
2. **基座是 8B，不是 35B-A3B**（题目允许）。在 TB2、SWE-hard、ASI 上，8B 的分数接近地板。分数越低，评测噪声相对越大。
3. **HLE 没有用官方 judge**：o3-mini 不可用，改用本地 Qwen3-8B judge，并在其上加了一层确定性审核（`configs/eval/benchmarks.yaml` → `judge_audit`）。审核前后的分数都报告了（`hle` / `hle_judge`）。审核规则是看过 judge 输出之后才加的，但规则对所有模型完全相同，而且只会把「判为正确」改成「错误」，不会反过来。
4. **ASI**：只能用 seed31415 公开集（60 题），而且只评了 b1 这一个 prompt 级别。**v1 的评测环境有缺陷**：用默认的 `--sandbox none` 运行，任务声明的依赖包（60 题里有 36 题声明了 matplotlib）不可用，大部分代码因 ModuleNotFoundError 崩溃，而且各模型受影响的程度不同（base 33/60，merged_ties 1/60）。v2 改用 `--sandbox task`，按任务声明建环境，对所有模型重跑，作为主结果；v1 只作参考列出。13 题里含 GPT-5.5 看图打分项，统一记 0 分；少数题打分超时，也记 0 分（`asi_errored` 列）。分数绝对值被压低，但不同模型之间仍可比较。
5. **Terminal-Bench 2.0**：
   - 只跑 74 题，且只跑 1 次（要求 5 次）。去掉的 15 题是因为共享网络下的端口 / 进程冲突（`scripts/tb2/lanes.sh`），不是因为「依赖外网」，也不是根据模型结果挑的；其中 8 题 oracle 在本机其实能通过。74 题中 oracle 在本机能通过的只有 66 题。merged_ties 只有 72/74 条记录，分母按 72 计；只在 66 道 oracle 可解题上算，merged_linear 是 1/66，其余模型都是 0（`analysis/contamination_review.md`）；
   - temperature 0.7，32K 上下文，单步最多输出 4096 token（模型卡：T=1.0，256K 上下文，80K 输出）；
   - 环境用 singularity 代替 docker（非特权容器）。
   所以这个分数不能和官方榜单对比，只能在本实验内部比较。
6. **SWE-bench Pro 的配置有缺陷，这一项的 0 分不能当作模型能力的结论**（10:30 复查发现，原始结果未改动）：
   - a. **patch 被截断**：harness 抓取 `git diff` 时丢掉了末尾换行，所有非空的 model.patch 都不以换行结尾，`git apply` 报 corrupt patch，退回到部分 fuzzy 应用。补回换行后重新判分（`scripts/swe_regrade_fixed.sh`），patch 都能干净应用，但 base 的 3 个、mopd 的 1 个仍然是 0：代码本身有错（Go 语法错误、Python `NameError`、测试不通过），所以分数没有变。
   - b. **agent 异常退出时修改丢失**：`host_mini_swe` 只在 agent 正常结束时才保存 patch。遇到 `ContextWindowExceeded` 或 `RuntimeError` 时，已经做的修改全部丢失（例如 base 在 ansible 上做了 200 多次编辑）。
   - c. **上下文只有 32K**：多次触发 `ContextWindowExceeded`。
   - d. **expert_swe 的训练格式与评测接口不匹配**：训练数据是单轮的 issue → diff，评测用的是需要多轮工具调用的 agent。所以每道题都以 `RepeatedFormatError` 结束，这个专家实际上从未执行过任何动作。合版模型也大多以 `LimitsExceeded` 或 `ContextWindowExceeded` 结束，没有产生修改。
   - e. 只计分 4 题。
   修正方向：在 finally 里保存 patch 并补回换行；用 YaRN 扩展到 64K+ 上下文；用多轮工具调用轨迹训练 SWE 专家；跑完整的 HARD-51。
   原第 6 条：只跑了 HARD-51 中按种子抽样的少量任务（镜像拉取速度和磁盘限制），而且只统计 gold patch 在本机能通过的任务。agent 运行阶段网络为 public（singularity 无法强制断网）。这项的置信区间非常宽，结论意义有限。
7. **merged_ties 的 TB2/SWE** 由 DGX 上的推理服务经 SSH 隧道提供，其他模型由 L40S 上的多 LoRA 服务提供。harbor 和 agent 配置相同，但推理速度不同，可能影响 TB2 的超时率。
8. **重复运行波动**：HLE 对 base、merged_linear 做了第二次运行；TB2 对 base、merged_linear 做了第二次运行（截止前没跑完的部分，只在两次都完成的任务上比较）。SWE 只有单次运行。mopd 和 merged_ties 的 HLE 重复运行都被停掉了：它们和 merged_ties 的 TB2 推理共用 DGX 的 GPU，会拖慢 TB2，造成额外超时。
9. 专家统一取第 500 步，没有做 checkpoint 选择，也没有做合版系数扫描。这样避免了用测试集调参，但也意味着没有调到最优。
10. **最终合版模型的选择规则是事后才定的**（见 §2）：规则只用开发集 loss，但确定规则时已经看到部分测试结果。

## 9. 复现步骤

在仓库根目录执行。DGX 部分使用 conda 环境 `moe-merge`，依赖见 `locks/requirements-train-eval.lock`；harbor 和 asibench 各自在独立 venv 里（`locks/requirements-harness.lock`、`locks/requirements-asibench.lock`）。

```bash
bash scripts/setup_env.sh                     # 依赖 + models/Qwen3-8B（固定 revision）
bash scripts/prepare_data.sh                  # data/*/{train,validation}.jsonl + 污染检查（约 10 分钟）
bash scripts/train_all.sh                     # 4 个专家，约 4.1 GPU·h（GB10）
bash scripts/after_training.sh                # merged_linear、merged_ties、校验、把专家合成完整权重（约 20 分钟）
python -m tests.test_merge_equivalence --adapter checkpoints/expert_swe   # 合版与 PEFT 逐位比对
bash scripts/run_mopd_plan.sh                 # MOPD 学生（约 2.6 GPU·h）+ 其 HLE/ASI 评测
bash scripts/run_eval_plan.sh                 # HLE + ASI：base、合版模型、专家（两条 lane）
bash scripts/judge_hle_local.sh               # 本地 Qwen3-8B judge 给所有 HLE 预测打分（judge 服务在 :30000）
bash scripts/hle_repeat_run.sh && bash scripts/judge_hle_repeat.sh   # HLE 重复运行 + 打分
python -m src.evaluation.dev_loss --models merged_linear=checkpoints/merged_linear \
    merged_ties=checkpoints/merged_ties mopd=checkpoints/full/expert_mopd base=models/Qwen3-8B   # 选最终合版模型
# x86 主机（TB2 / SWE，非特权容器 + singularity）：
bash scripts/run_x86_lora_plan.sh && bash scripts/run_x86_phase2.sh   # base + 专家 + merged_linear + mopd（多 LoRA 服务）
bash scripts/serve_ties_remote.sh   # （DGX）merged_ties 推理服务 + 反向隧道
bash scripts/run_x86_ties_remote.sh # （x86）merged_ties 的 TB2 / SWE
bash scripts/run_x86_tb2_rep2.sh    # （x86）TB2 重复运行：base、merged_linear
# 汇总（DGX）：拉取 x86 结果 → summary.csv、置信区间、失败分析、曲线、日志索引、结果表
bash scripts/aggregate_all.sh
```

## 10. 日志索引

`logs/` 下的文件列表和格式说明见 [`logs/README.md`](logs/README.md)，包括训练、评测、错误与重试、资源使用。harbor 每个 trial 的完整轨迹太大，打包后放在 GitHub Release，校验值在 `logs/README.md`。

## 11. 模型上传

```bash
huggingface-cli login   # 需要 write token
python scripts/upload_hf.py --user cokoro   # 上传 4 个专家 adapter、merged_linear/ties 完整权重、MOPD adapter，均设为 public
```

## 12. 经验总结与判断

首先是算力限制了评测设置。8B 模型每个 token 的 KV 缓存约 144KB；而 TB2 同时跑 12 道题、每道最长 28K，需要约 34 万 token。请求只能排队，base 有一半以上的题是超时失败，不是做错。
**算力判断**：Qwen3-8B 每个 token 的 KV 缓存约 144KB。L40S 上能放的 KV 约 14.5 万 token，而 TB2 12 并发时最多需要约 34 万 token，于是请求排队、大量超时。按今天的设置（32K 上下文、12 并发）不排队地跑 TB2，需要约 75GB 显存（1 张 80GB 卡）；接近模型卡设置（128K 上下文、32 并发）需要约 290GB（4 张 H100 级别）。

算力决定了分数的地板，融合方式决定了专家的能力能不能被保住。

- 权重融合是在参数上做加法，但网络是非线性的，不同专家的参数改动会互相干扰。我们观察到，TIES 让终端专家「接管」了模型，40% 的数学题直接输出 bash 代码；Linear 有 27 次把题目模板原样抄回来。这些病态行为，任何一个单独的专家都没有。
- MOPD（多教师在线蒸馏）是在输出上对齐：每道题由对应领域的教师逐个 token 指导，学生在自己生成的回答上被纠错。所以它的输出格式和 base 一样健全：0 次输出代码，0 次抄模板。
- 业界也是这样做的：
  - 小米的 MOPD 论文：MOPD 0.937，Task Arithmetic 合并 0.857，简单权重平均只有 0.328；
  - DeepSeek-V4 用 10 个以上教师做在线蒸馏，取代了混合 RL；
  - Qwen3 用在线蒸馏，以 RL 约 1/10 的算力拿到了更高的分数。

这次 MOPD 的分数没拉开，原因很明确：教师不够强。文献同样指出，蒸馏的收益取决于教师比学生强多少。所以下一步不是换融合方法，而是先把教师训强。

在只有一张 48GB L40S 和一台 GB10 的条件下，评测被迫缩到 32K 上下文、单次运行和题目子集，大部分分数因此落在地板上，看不出各方法之间的真实差距。但实验揭示了更本质的问题：权重融合会让某个专家「接管」模型，而多教师在线蒸馏在行为上始终保持健全，它的上限只取决于教师质量。所以下一步应该先把专家训强，再在 4 张 H100 上，按接近模型卡的设置验证蒸馏路线。
