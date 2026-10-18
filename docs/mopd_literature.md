# 为什么多教师在线蒸馏（MOPD）比权重合并更适合整合多个专家：文献依据与本仓库实证对照

*调研日期：2026-10-18。所有来源都打开过原文（arXiv 摘要页或 HTML 全文、官方博客）。「原文要点」是中文转述，带引号的是原文短句。只读到摘要、没读到正文数字的，标注「仅摘要」；读不到或无法确认的，标注「未能核实」。本仓库的数字来自 `README.md` §7 和 [`docs/mopd_vs_merge.md`](mopd_vs_merge.md)。*

## 0. 先说结论

- 文献对「MOPD 优于权重合并」的支持，**主要来自机制分析和工业实践**，直接的受控对比只有一篇：小米的 MOPD 论文（Ma et al., 2026）。在 Qwen3-30B-A3B 上，它的归一化分数是 MOPD 0.937、Task Arithmetic 0.857、权重平均 0.328。
- 2026 年的几份前沿技术报告把「先训各领域专家，再用多教师在线蒸馏合成一个模型」当作主流程：小米 MiMo-V2-Flash、DeepSeek-V4（超过 10 个教师，完全取代了混合 RL 阶段）、Kimi K3（9 个教师）。DeepSeek-V4 的原话是，这样做可以绕开「传统权重合并或混合 RL 常见的性能退化」。
- **但文献同样给出了明确的前提和反例**：
  - 教师要明显强于学生。MOPD 的归一化分数就是按「学生到教师之间的差距」定义的。
  - 教师要和学生同源，异源的大教师反而会让训练不稳定。
  - MOPD 也不是每项都能追平教师：MiMo-V2-Flash 在 BrowseComp 上比教师低 6.3 分。
  - 在指令模型、较大模型、近似正交的 RL 任务向量这些条件下，权重合并可以很强：MergeBench 中部分设置超过 100% 保留率；在 AppWorld 上，合并与联合 RL 统计上不可区分。
- 和本仓库的对应：**我们观察到的「合并病态 vs. MOPD 行为健全」与文献机制一致；「MOPD 分数没赢」也与文献的前提条件一致**——我们的教师相对 base 几乎没有可蒸馏的差距。

---

## 1. 机制性论点：MOPD 为什么比权重合并强

### 论点 1：在策略空间（输出分布）里整合能力，而不是在权重空间里做算术

权重合并默认「各专家的参数增量近似可加、互不干扰」。文献表明，这个假设在 LLM 上经常不成立，而且失败与否取决于人为设定的系数。

- **TIES-Merging**（Yadav et al., NeurIPS 2023）指出了两类干扰：「冗余参数值造成的干扰」和「同一参数在不同模型间的符号不一致」。
  - 「即使只合并 2 个不同任务的模型，也会出现符号冲突」，而且模型越多，冲突越可能发生。在附录的 checkpoint 实验里，保留的参数越多，符号冲突越多，「接近 80%」。
  - 「随着合并任务数增加，所有方法的性能都会下降」。
  - 即使用 TIES，合并模型仍明显低于单任务微调：ViT-B/32 73.6 vs 90.5，T5-Large 76.9 vs 88.8。
  - 用 oracle（多任务模型）的符号替换后，(IA)³ 上的差距从约 7 分缩小到 1.1 分。可见差距的主要来源是符号估计。
- **DARE**（Yu et al., ICML 2024）原文给出了 TIES 在 LLM 上的严重失败案例（不加 DARE）：
  - 合并 LM 与 Code 专家后，HumanEval 和 MBPP 都降到 0.0；
  - 合并 LM 与 Math 专家后，GSM8K 从 64.22 跌到 15.77。
  - 加上 DARE 后有所恢复（HumanEval 18.29、GSM8K 36.16），但仍远低于源模型。
- **MOPD 论文**（Ma et al., 2026，小米 LLM Core 与北京大学）在引言中把方法定位为「直接在策略空间中整合，而不是在权重空间或静态数据集空间中整合」。
  - 主表（Qwen3-30B-A3B，0 = Stage-1 SFT 学生，1 = 各领域 RL 教师）：

    | 方法 | 归一化分数 |
    |---|---|
    | 权重平均（Param-Merge, Avg.） | 0.3280 |
    | Cascade RL | 0.7752 |
    | Off-Policy Finetune | 0.8241 |
    | Task Arithmetic | 0.8574 |
    | Mix-RL | 0.8818 |
    | **MOPD** | **0.9373** |

  - 对合并的结论是：结果「同时取决于合并系数和 benchmark」，所以参数合并是「不可靠的整合工具」。
- **DeepSeek-V4**（DeepSeek-AI, 2026）§5.1.2：把 OPD 作为「把专家能力合并进最终模型的主要技术」，在单一参数空间里做 logit 级对齐，「实际上绕开了传统权重合并或混合 RL 中常见的性能退化」。注意：**这是定性表述，该节没有给出与合并的数值对比。**

### 论点 2：在线（on-policy）采样，消除训练与推理的分布错配（暴露偏差）

合并完全不看模型会生成什么；离线蒸馏或 SFT 只在教师的轨迹上训练。MOPD 在学生自己的轨迹上纠错。

- **GKD**（Agarwal et al., ICLR 2024）认为，自回归模型蒸馏的核心问题是「训练时看到的序列」与「学生推理时自己生成的序列」不一致。
  - 按 T5 学生规模平均，on-policy GKD 相对初始学生的提升，是 supervised KD 等基线的 2.1×（摘要）、1.7×（翻译）、1.9×（GSM8K）。
  - 在 XSum 上，on-policy GKD 只用 5% 的数据、不用任何真值摘要，就超过了用全量数据的 supervised KD。
  - 代价是学生采样带来约 1.8–2.2× 的额外计算。
- **MiniLLM**（Gu et al., ICLR 2024）：用反向 KL 加 on-policy 优化，报告了更低的暴露偏差、更好的校准和更好的长文本生成（仅摘要级结论，未取具体数字）。
- **Thinking Machines「On-Policy Distillation」**（Kevin Lu, 2025-10-27）用象棋做类比：
  - off-policy 蒸馏 = 看大师下棋，但大师所处的局面，新手很少遇到；
  - on-policy RL = 没有教练，只知道输赢；
  - on-policy 蒸馏 = 教练给你自己的每一步打分。
- **MOPD 论文**也把「只在学生自己的 rollout 上训练，从而消除暴露偏差」列为三大好处之一。

### 论点 3：逐 token 的稠密信号，样本和算力效率远高于 RL

合并没有任何反馈信号；RL 每条轨迹只有约 1 bit 的结果信号。在线蒸馏每个 token 都有教师给出的完整监督。

- **Qwen3 技术报告**（Qwen Team, 2025）表 21，Qwen3-8B 从同一个 off-policy 蒸馏 checkpoint 出发：

  | 方法 | AIME'24 | AIME'25 | GPQA-D | GPU 小时 |
  |---|---|---|---|---|
  | + RL | 67.6 | 55.5 | 61.3 | 17,920 |
  | + On-policy 蒸馏 | 74.4 | 65.5 | 63.3 | 1,800 |

  即用约 1/10 的算力取得了更高的分数。On-policy 蒸馏还提升了 pass@64（AIME'24 从 90.0 到 93.3），而 RL「没有带来 pass@64 的任何提升」。教师是 Qwen3-32B 或 Qwen3-235B-A22B，做法是对齐 logits、最小化 KL。
- **Thinking Machines 博客**：
  - Qwen3-8B-Base 学生，从 400k 条 SFT 的 checkpoint（AIME'24 60%）出发，约 150 步 on-policy 蒸馏就达到 70%；
  - 计算量比外推的 SFT-2M 低 9–30×（FLOPs 口径），GPU 小时约低 18×；
  - 把 RL 教师蒸馏回去，比 RL 少约 7–10× 梯度步，算力低 50–100×。
- **LoRA Without Regret**（Schulman et al., Thinking Machines, 2025-09-29）给出了信息量的论证：策略梯度 RL 每条轨迹只吸收约 O(1) bit，而监督学习的信息量与 token 数同阶。这个论证解释了上面的效率差距。它同时意味着，在线蒸馏比 RL 需要更大的适配器容量（见 §4.3）。
- **MiMo-V2-Flash 技术报告**（Xiaomi LLM-Core Team, 2026）：教师提供的是「稠密的、token 级的奖励」，并把 MOPD 写成一个 on-policy RL 过程。

### 论点 4：按输入路由教师，实现「有条件」的整合；合并只能给出一组对所有输入都相同的参数

- **MOPD 论文**：每条 rollout 路由到它所属领域的教师，对该教师最小化逐 token 反向 KL。论文把这种「按 prompt 路由、在策略空间整合」列为相对权重融合的第三个好处。
  - MOPD 各领域的归一化分数都落在 0.91–0.95，是所有方法中最窄的区间。
  - Task Arithmetic 在 IF 上达到 1.00，但在 Math 上只追回 73% 的差距。也就是说，合并的收益在不同领域间严重不均。
- **MiMo-V2-Flash**：学生在 prompt x 上对「x 所属领域的教师 π_domain_x」计算 log-ratio 作为优势。教师覆盖搜索、编码、通用工具使用、数学、通用推理和安全。
- **Kimi K3**（Kimi Team, 2026）§4.1.3：3 个领域 × 3 档推理强度（low/high/max）共 9 个教师，按（领域, 强度）路由，学生也以强度为条件。这种「同一领域内、按推理预算条件化」的整合，权重合并在结构上无法表达。
- **TrustMOPD**（Sun et al., 2026，仅摘要）：去掉领域标签，改为按 token 估计每个教师的可靠性并加权。在 SingleCap 和 MultiCap 两种设置上，分别追回初始学生到 oracle 路由教师之间 91.5% 和 98.0% 的差距；最强的无标签基线只有 54.4% 和 54.5%。说明「谁来监督哪段输出」本身就是能力整合的关键变量。

### 论点 5：反向 KL 是 mode-seeking，学生会稳定地采用教师的主要模式，不在多种格式之间摇摆

- **MiniLLM**：前向 KL 会让学生「高估教师分布中的低概率区域」，生成式任务更适合用反向 KL。
- **GKD**：
  - 前向 KL 是 mode-covering，反向 KL 是 mode-seeking（附录 A.7）；
  - 当学生容量远小于教师时，前向 KL「可能导致幻觉和低质量生成」；
  - 在指令微调上，反向 KL「大幅优于」其他目标；
  - 但**最优散度依赖任务**：WMT 上广义 JSD 最好，GSM8K 上前向 KL 也表现很好。
- **Thinking Machines 博客**：逐 token 反向 KL 是「不可被 hack 的」（低 KL 总是对应教师认为好的行为），而且是 mode-seeking 的。

### 论点 6：对结构、数量和规模的约束更少（但有「同源」前提）

- **权重合并要求同源同结构**：
  - DARE：「我们专注于合并从同一预训练 backbone 微调而来的模型」。并且只有 delta 很小时才有效：继续预训练后 delta「很快达到 0.03 左右，使 DARE 不可行」。在 WizardCoder-Python-13B 上只删 10% 的 delta，HumanEval/MBPP 就从 63.41/55.4 跌到 0.0/0.0。
  - DARE 还指出，有效合并的前提是「每个源模型都充分微调」。
  - LoRA 合并更难：**KnOTS**（Stoica et al., 2024，仅摘要）报告，LoRA 微调模型的权重「对齐程度更低」，现有合并方法直接用于 LoRA 效果不好。
- **蒸馏只需要教师的 logprob，以及与学生相同的 tokenizer**，学生可以比教师大或小。DeepSeek-V4 用了超过 10 个教师，并专门设计了「全词表 OPD 的教师调度」，支持「几乎无上限」的教师数量；TIES 则报告性能随合并任务数增加而下降。
- **但必须如实说明：MOPD 论文发现「同源教师对稳定优化至关重要」。**
  - 他们把数学教师换成更大的 Qwen3-235B-A22B，结果数学反而变差：初始逐 token KL 约 0.19，同源教师约 0.04；策略梯度形式下熵从 0.30 降到 0.21；top-k 形式在第 18 步左右发散。
  - 所以「不要求同结构」不等于「任何教师都能用」。分布距离太远的教师会让训练不稳定。

### 论点 7：能恢复被专精化破坏的能力，缓解 see-saw 和遗忘

- **MiMo-V2-Flash**：动机是「能力失衡」，即提升一项技能会让其他技能退化（「see-saw」效应）。报告明确选择「不合并模型参数，也不从专家生成静态离线数据集」，而是把多教师知识整合写成 on-policy RL 过程。
- **Thinking Machines 博客**的个性化实验（Qwen3-8B 在内部文档上做 mid-training）：

  | 模型 | 内部 QA | IF-eval |
  |---|---|---|
  | Qwen3-8B | 18% | 85% |
  | + mid-training（100% 文档） | 43% | 45% |
  | + mid-training（70% 文档） | 36% | 79% |
  | + 70% 文档 + 以原 Qwen3-8B 为教师的 on-policy 蒸馏 | 41% | 83% |

  蒸馏恢复了大部分指令遵循能力，同时保住了知识。
- **CaMOPD**（Chen et al., 2026，仅摘要）进一步指出，原始 MOPD 中「通用能力恢复」与「领域保持」的梯度会相互抵消（counteraction），需要交替训练。这说明 MOPD 有自己的冲突问题，但冲突发生在梯度层面，可以诊断，也可以调度。

### 论点 8：能和结果奖励结合，有机会超过教师；合并的上限被源模型锁死

- **MiMo-V2-Flash** 式 (9)：MOPD 的优势 = sg[log(π_teacher/π_θ)] + α·A_ORM，默认与 GRPO 类结果奖励结合（α 未给出）。表 7 中学生在若干项上超过了最强教师：
  - AIME 2025：94.1 vs 93.9
  - HMMT Feb 2025：84.4 vs 82.6
  - LiveCodeBench：83.2 vs 82.6
  - Tau2-Bench：80.3 vs 79.6
- **MOPD 论文**的多轮迭代（只做了 Math 和 IF）：MOPD 学生作为下一轮的起点，归一化分数从 0.937 升到 0.986；第 2 轮 RL 教师为 1.030。

---

## 2. 反面证据与前提条件（不支持「MOPD 一定更强」的部分）

1. **MOPD 并不在每一项上都追平教师。**
   - MiMo-V2-Flash 表 7：BrowseComp 45.4 vs 教师 51.7（**−6.3**），SWE-Bench Verified 73.4 vs 74.2（−0.8）；Arena-Hard Creative Writing 从 90.1 降到 86.2（−3.9），GPQA-Diamond 从 84.9 降到 84.3。
   - MOPD 论文在 Qwen3-30B-A3B 上：AIME25 51.46 vs 教师 54.79（−3.33），IFEval 93.84 vs 95.50。
   - 「保持所有领域最强教师峰值」这一说法，被这些数字部分否定。
2. **MOPD 的受控证据面很窄。** MOPD 论文的主对比只有 1 个 base 模型、3 个领域，每种方法只跑 1 次，没有报告方差或种子。合并基线只有权重平均和 Task Arithmetic，系数也未说明。TIES、DARE 没有进入实验，只在相关工作中提及。论文也没有消融反向 KL 与前向 KL、教师数量、学生规模和 ORM 权重。
3. **在有利条件下，权重合并并不差。**
   - **MergeBench**（He et al., NeurIPS 2025 D&B）表 8，按「合并模型得分 / 专家得分」计算的保留率：
     - 预训练 base，2B/3B：最好的方法约 80%；
     - 指令 base：普遍在 85–106% 之间，Gemma-2-9B-IT 上 Consensus TA 达到 106.5%，TIES 101.1%；
     - 但同一张表里 Llama-3.2-3B 预训练 base 的 TIES 只有 63.0%。
     - 论文结论是「合并在更强的 base 上效果更好」，但合并模型的领域内表现仍落后于多任务训练。
   - **What Matters for Model Merging at Scale?**（Yadav et al., 2024，仅摘要）：base 越强、模型越大，合并越容易；8 个大专家合并后，held-out 泛化常常优于多任务训练；规模大时，各合并方法表现相近。
   - **When Model Merging Rivals Joint Multi-Task RL**（McClendon, 2026，仅摘要）：Qwen3-8B 在 AppWorld 上的两个 RL 专家，用 TIES 或 RAM+ 合并后，任务完成率与联合 RL 统计上不可区分。原因是两个任务向量近似正交（余弦 0.06–0.10）。**这是「合并在 agent 任务上不一定退化」的直接反例**，但它只有 1 个 benchmark、2 个专家，而且专家都是 RL 训练的。
   - **RAM**（Yuan et al., 2026，仅摘要）：on-policy RL 产生的任务向量稀疏且异质，SFT 时代的合并方法会把非重叠的更新平均稀释掉。改进后的 RAM 合并据称能超过各领域专家（未取到数字）。
4. **合并在长推理和 agent 上的退化，确实被报告过。**
   - **Long-to-Short 合并研究**（Wu et al., 2025）：1.5B 模型上，AIME24 从基线 20.0 降到 TA 10.0、TIES 10.0、权重平均 0.0，作者称之为「虚假反思」；把 TA 系数调到 0.3 时，AIME24 −30.0。原文还说：「多数合并方法对超参数敏感」，α 变化 0.1 就会造成约 0.5–1.0 分的差距。
   - **AgentPatch**（Shao et al., 2026，仅摘要）：合并 agent 型多模态模型时，交互复杂度不同的能力「保留不均」，合并后出现「弱任务」，关键动作丢失会引发轨迹级失败。
   - **Kimi k1.5**（Kimi Team, 2025）把「直接平均长 CoT 与短 CoT 模型的权重」作为 long2short 方法之一，并报告 long2short RL 的 token 效率高于 DPO 和模型合并。图 7 的具体数值没能从正文中读出（**未能核实具体数字**）。
5. **MOPD 的成本和工程前提**：
   - 需要教师的 logprob，每步要做教师 prefill；
   - 学生和教师要用相同的 tokenizer（本仓库 `src/distill/mopd.py` 中有断言）；
   - GKD 报告学生采样带来 1.8–2.2× 的计算开销；
   - Thinking Machines 指出，学生初始时必须已经对教师的行为有一定概率覆盖，否则需要大得多的 batch。
6. **用逐 token 估计还是全词表**，报告之间有分歧：
   - DeepSeek-V4 认为，把 KL 简化成逐 token 估计、当作 RL 优势来用，会「导致梯度估计方差高、训练不稳定」，所以改用全词表 logit 蒸馏；
   - Kimi K3 用的是采样 token 上截断的 log-ratio，并称更细粒度的 top-k 目标「没有明显优势」；
   - MOPD 论文的策略梯度形式与 top-k（k=64）形式的总分为 0.937 vs 0.909。哪个数字对应哪种形式，原文表述不够清楚，**未能完全核实**。
7. **其他厂商**：
   - **DeepSeek-V3.2** 的「Specialist Distillation」是让各专家**生成离线数据**，再用混合 RL 收尾。原文说蒸馏模型「只略低于」各领域专家，差距由后续 RL 消除。所以它不是 on-policy 蒸馏，也不涉及权重合并。到了 V4，混合 RL 阶段才被「完全替换」为 OPD。
   - **GLM-4.5** 的「Expert Model Iteration」先训练推理、Agent、通用对话三个专家，再「用自蒸馏技术整合多个专家」。从原文看更像是对专家输出做 SFT（off-policy），**是否 on-policy 未能核实**。
   - **Kimi K2 / K2.5** 没有找到使用 MOPD 的证据（**未能核实**）。目前只确认 K3 用了 MOPD。

---

## 3. 与本仓库实证的对应

| 本仓库观察到的现象 | 对应的文献机制 | 文献依据 |
|---|---|---|
| **TIES 有 198/500（40%）的 HLE 回答被终端专家的 bash 格式接管** | 合并没有输入条件（论点 4）。TIES 剪枝 + 选举符号 + 不除以 k，使幅值最大的专家方向以接近全强度保留，所有输入共享这组参数 | TIES：两模型即有符号冲突；DARE 中 TIES 合并 LM+Code 后 HumanEval 为 0.0；AgentPatch：能力「保留不均」；MOPD 论文：Task Arithmetic 在不同领域间收益不均 |
| **Linear 有 27 次把 prompt 模板原样抄回** | 1/4 系数稀释了每个专家的格式行为（论点 1）。合并效果强烈依赖系数，而系数是人定的 | MOPD 论文：权重平均只有 0.328；TIES：简单平均在合并 2 个任务时就损失约 10%；Wu et al.：α 变化 0.1 即有 0.5–1.0 分差距 |
| **MOPD 0 次抄模板、0 次被接管，495/500 有效答案（与 base 持平）** | 在线采样 + 按领域路由 + 反向 KL 的 mode-seeking（论点 2、4、5）：学生在自己开始「跑偏」的位置被对应教师纠正，并锁定教师的主要格式 | GKD、MiniLLM（暴露偏差）；Thinking Machines（mode-seeking、不可 hack）；MOPD 论文（per-prompt routing） |
| **MOPD 学生忠实复制了 ASI 教师的模板化短回答（320 → 约 43 token），ASI 低于 base** | 反向 KL 会锁定教师的主模式，**包括教师的坏模式**。蒸馏的上限是教师 | GKD/MiniLLM 的 mode-seeking 分析；MiMo 和 MOPD 论文中的教师都是大规模 RL 训练、明显强于学生的模型 |
| **MOPD 在分数上没有赢**（宏平均 MOPD +0.07 vs Linear +0.82 vs TIES +0.04，ASI 为 v2，差别都在噪声范围内） | MOPD 的收益 = 学生到教师差距中被追回的比例。差距约为 0 时，追回 90% 也约为 0。我们的 expert_hle 相对 base 只有 +1.40 [−0.6, +3.4]，expert_asi 为 −0.99 | MOPD 论文的归一化定义（0 = 学生，1 = RL 教师）；DARE：合并的前提同样是源模型「充分微调」 |
| **SWE / TB2 两项 MOPD 没有学习信号**（只用了 2 个教师，终端 dev loss 2.31 vs 合并的 0.83–0.99） | 没纳入的领域，MOPD 什么也学不到；合并则「白拿」了这些参数（虽然没变成分数） | MiMo（SWE、工具使用教师）、DeepSeek-V4（agent 教师）、Kimi K3（coding agent 教师）都把 agent 领域纳入了在线蒸馏；Kimi K3 提到复用 partial rollout 等 RL 基础设施来处理长程任务 |
| **教师与学生同源**（全部是 Qwen3-8B + LoRA） | 满足 MOPD 论文强调的同源前提，初始 KL 应该较低，训练稳定 | MOPD 论文：异源 235B 教师的初始 KL 0.19 vs 同源 0.04，top-k 形式发散 |
| **学生 = base + 新 LoRA，没有先做 SFT 预热**；152 步 × 32 条，约 2.6 GPU·h | 文献中的流程都是先 SFT（或 off-policy 蒸馏），再做 on-policy。学生初始覆盖不足时，需要更大的 batch | Qwen3（先 off-policy 再 on-policy）；MOPD 论文 Stage 1 是通用 SFT；Thinking Machines 从 400k SFT checkpoint 起步，并说明了初始化的作用 |
| **teacher-forced 开发集 loss 选中了 TIES（HLE val 0.356 vs 0.400），但看不出它的格式接管** | 合并的超参搜索和验证成本高，而 loss 类指标衡量不到生成行为 | MergeBench：TIES/DARE 各需搜索 30 组配置，验证成本「可能主导总运行时间」 |
| **MOPD 训练成本约为合并的 50 倍**（2.6 vs 0.05 GPU·h） | 文献承认在线蒸馏有采样和教师 prefill 的成本；但与 RL 相比，它便宜 1 个数量级 | GKD（采样 1.8–2.2×）；Qwen3（1,800 vs 17,920 GPU h，对比的是 RL 而非合并） |

**小结**：文献机制完整解释了我们的「行为」结果（合并出现病态，MOPD 保持健全），也解释了「分数」结果（教师相对 base 几乎没有可蒸馏的差距）。所有给出 MOPD 大幅收益的文献，都建立在「教师显著强于学生、同源、覆盖全部领域、学生先做 SFT 预热」这些前提上，而本实验只满足了「同源」一条。

---

## 4. 对下一步的启示

### 4.1 教师质量（最优先）
- 文献中 MOPD 的收益都以「教师减学生」的差距为分母。**先让每个教师在本领域显著超过 base**（CI 不含 0），再做 MOPD，否则复现的只是本实验的结论。
- 教师应该用 RL 或高质量真实数据训练：MiMo 的教师是「独立 RL 优化」得到的，MOPD 论文的教师是逐领域 RL 训练的。不要继续用 GSM8K、CSQA 这类代理数据。
- 教师有坏模式时（例如本实验的 ASI 模板），反向 KL 会把坏模式也复制过来。可以加 ORM 优势项（MiMo 的 α·A_ORM），或者按 TrustMOPD 的思路对不可靠的 token 降权。

### 4.2 教师数量
- 工业做法是 6 个左右（MiMo），9 个（Kimi K3），超过 10 个（DeepSeek-V4）。TIES 报告合并性能随任务数增加而下降，而在线蒸馏原则上可以扩展到很多教师。**下一轮应纳入全部 4 个教师。**
- SWE 和 TB2 是多轮 agent 任务，在线 rollout 很贵。可以参考 Kimi K3 复用 RL 的 partial rollout，或者先用教师轨迹做 off-policy 预热，再做少量 on-policy 回合。

### 4.3 学生规模与适配器容量
- **合并对规模更敏感**：Yadav et al. 2024 和 MergeBench 都报告，模型越大、base 越强，合并越好。Wu et al. 报告合并在 1.5B 上失败。所以 8B 本来就不是合并的有利区间。
- **蒸馏到更大的学生**（例如 Qwen3-14B）需要注意 MOPD 论文的同源发现：异源教师会让初始 KL 偏高、训练不稳定。建议先用多专家过滤后的数据对 14B 做 SFT，缩小分布距离，再做 on-policy 蒸馏。另外要确认 tokenizer 一致（代码中已有断言）。
- 学生如果太小（≤3B），可能学不了强教师的长推理（Li et al. 2025，「Small Model Learnability Gap」，仅摘要）。本实验的 8B 不在这个区间。
- **LoRA 容量**：LoRA Without Regret 表明 LoRA 对 RL 足够（rank 1 即可），但监督类信号的信息量与 token 数同阶，数据量超过容量时 LoRA 效率会下降。在线蒸馏是逐 token 的稠密信号，**如果教师变强、步数增加，应该检查学生 LoRA 的 rank（目前是 16）是否成为瓶颈**。

### 4.4 on-policy 采样的比例与课程
- GKD：只要有 ≥25% 的 on-policy 数据，GSM8K 的表现就随 on-policy 占比上升而提升；WMT 上纯 on-policy 最好。
- Qwen3、Thinking Machines、MOPD 论文都采用「先 off-policy（SFT 或蒸馏），再 on-policy」。
- **建议**：用多专家过滤后的正确轨迹做 SFT 预热 → MOPD（λ = 1）。如果 agent 任务的在线采样太贵，可以用混合比例（例如 λ = 0.5），但不要低于 25%。

### 4.5 反向 KL 还是前向 KL
- 默认继续用反向 KL：MiniLLM、Thinking Machines、MiMo、DeepSeek-V4、Kimi K3 都用它。它的 mode-seeking 正好解释了我们的格式稳定性。
- GKD 明确指出最优散度依赖任务：WMT 上 JSD 更好，GSM8K 上前向 KL 也不错。如果担心学生丢掉 base 的多样性，或者锁定教师的单一坏模式（如 ASI 模板），可以在开发集上对比广义 JSD(β)。
- 本仓库已经用全词表精确反向 KL，与 DeepSeek-V4 的选择一致（DeepSeek-V4 认为它的梯度方差比采样 token 的估计更低）。如果显存紧张，可以退到 Kimi K3 式的采样 token log-ratio，Kimi K3 称 top-k 没有明显优势。

### 4.6 防遗忘和结果奖励
- 在通用 prompt 上加入以 base 为教师的蒸馏项（Thinking Machines 的个性化实验思路），防止 ASI 这类领域性退化。混合时注意 CaMOPD 指出的「恢复与保持」梯度抵消，必要时交替更新。
- 加入 ORM（MiMo 式 (9)）可以让学生在教师不完美时超过教师。这对本仓库「教师偏弱」的情况尤其重要。

### 4.7 让合并基线更公平
- 合并的结论对系数敏感（MergeBench、Wu et al.、MOPD 论文都这么说）。下一轮应该在生成式开发集上扫 Linear 系数和 TIES 的密度、缩放，例如给 TIES 加上 1/k 缩放。
- 如果专家改用 RL 训练，可以加入 RAM 这类针对 RL 任务向量的合并方法作为基线，避免拿弱基线和 MOPD 比。

---

## 参考文献

**MOPD 与工业技术报告**
1. Xiaomi LLM-Core Team. *MiMo-V2-Flash Technical Report*. arXiv:2601.02780, 2026-01. https://arxiv.org/abs/2601.02780 （MOPD 见 §4.1，结果见表 7）
2. Wenhan Ma, …, Zhifang Sui, Fuli Luo（北京大学、小米 LLM Core）. *MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training*. arXiv:2606.30406, 2026-06. https://arxiv.org/abs/2606.30406
3. DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*. arXiv:2606.19348, 2026-04. https://arxiv.org/abs/2606.19348 （§5.1.2、§5.2.2）
4. DeepSeek-AI. *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*. arXiv:2512.02556, 2025-12. https://arxiv.org/abs/2512.02556
5. Kimi Team. *Kimi K3: Open Frontier Intelligence*. arXiv:2607.24653, 2026-07. https://arxiv.org/abs/2607.24653 （§4.1.3）
6. Kimi Team. *Kimi k1.5: Scaling Reinforcement Learning with LLMs*. arXiv:2501.12599, 2025-01. https://arxiv.org/abs/2501.12599 （long2short，§2.4 / §3.4）
7. Qwen Team. *Qwen3 Technical Report*. arXiv:2505.09388, 2025-05. https://arxiv.org/abs/2505.09388 （§4.5，表 21）
8. GLM-4.5 Team（Zhipu AI 等）. *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*. arXiv:2508.06471, 2025-08. https://arxiv.org/abs/2508.06471

**On-policy 蒸馏方法论**
9. Kevin Lu, Thinking Machines Lab. *On-Policy Distillation*. Connectionism, 2025-10-27. https://thinkingmachines.ai/blog/on-policy-distillation/
10. John Schulman et al., Thinking Machines Lab. *LoRA Without Regret*. Connectionism, 2025-09-29. https://thinkingmachines.ai/blog/lora/
11. Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (GKD). ICLR 2024. https://arxiv.org/abs/2306.13649
12. Yuxian Gu, Li Dong, Furu Wei, Minlie Huang. *MiniLLM: On-Policy Distillation of Large Language Models*. ICLR 2024. https://arxiv.org/abs/2306.08543
13. Jie Sun et al. *Distill What You Trust: Reliability-Aware Multi-Teacher On-Policy Distillation* (TrustMOPD). arXiv:2609.23697, 2026-09. https://arxiv.org/abs/2609.23697 （仅摘要）
14. Tianlei Chen et al. *Counteraction-Aware Multi-Teacher On-Policy Distillation for General Capability Recovery with Domain Preservation* (CaMOPD). arXiv:2605.27115, 2026-05. https://arxiv.org/abs/2605.27115 （仅摘要）
15. Yuetai Li et al. *Small Models Struggle to Learn from Strong Reasoners*. arXiv:2502.12143, 2025-02. https://arxiv.org/abs/2502.12143 （仅摘要）

**权重合并及其局限**
16. Gabriel Ilharco et al. *Editing Models with Task Arithmetic*. ICLR 2023. https://arxiv.org/abs/2212.04089
17. Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal. *TIES-Merging: Resolving Interference When Merging Models*. NeurIPS 2023. https://arxiv.org/abs/2306.01708
18. Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, Yongbin Li. *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch* (DARE). ICML 2024. https://arxiv.org/abs/2311.03099
19. George Stoica, Pratik Ramesh, Boglarka Ecsedi, Leshem Choshen, Judy Hoffman. *Model merging with SVD to tie the KnOTS*. arXiv:2410.19735, 2024. https://arxiv.org/abs/2410.19735 （仅摘要）
20. Prateek Yadav, Tu Vu, Jonathan Lai, Alexandra Chronopoulou, Manaal Faruqui, Mohit Bansal, Tsendsuren Munkhdalai. *What Matters for Model Merging at Scale?* arXiv:2410.03617, 2024. https://arxiv.org/abs/2410.03617 （仅摘要）
21. Yifei He, Siqi Zeng, Yuzheng Hu, Rui Yang, Tong Zhang, Han Zhao. *MergeBench: A Benchmark for Merging Domain-Specialized LLMs*. NeurIPS 2025 Datasets & Benchmarks. https://arxiv.org/abs/2505.10833
22. Han Wu et al. *Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging*. arXiv:2503.20641, 2025. https://arxiv.org/abs/2503.20641
23. Xiangchi Yuan et al. *Behavior Knowledge Merge in Reinforced Agentic Models* (RAM). arXiv:2601.13572, 2026-01. https://arxiv.org/abs/2601.13572 （仅摘要）
24. S. Aaron McClendon. *When Model Merging Rivals Joint Multi-Task Reinforcement Learning: A Task-Vector Geometry Analysis*. arXiv:2607.16062, 2026-07. https://arxiv.org/abs/2607.16062 （仅摘要）
25. Zibo Shao et al. *AgentPatch: Coarse-to-Fine Weak-Task Repair for Merging Agentic Multimodal Large Language Models*. arXiv:2608.06699, 2026-08. https://arxiv.org/abs/2608.06699 （仅摘要）
