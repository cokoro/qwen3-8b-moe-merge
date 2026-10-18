# 日志索引

由 `scripts/collect_logs.py` 根据两台机器上的原始日志生成。每个 run_id 对应一次训练或一次评测，记录在 `scripts/timed_run.sh` 写出的 `results/experiment_times.csv` 中。

```
logs/
├── resources.csv          # 每个 run：主机、GPU、耗时（秒）、GPU 小时、峰值显存、估算费用、退出码、重试次数
├── experiment_times.csv   # DGX 原始计时
├── experiment_times_x86.csv
└── <run_id>/
    ├── metadata.json      # 主机、GPU、开始/结束时间、退出码、尝试次数、原始日志路径、配置文件、种子
    ├── train.log | eval.log
    ├── metrics.jsonl      # 训练 run 逐步指标：HF Trainer log_history；MOPD 为每步 KL 和回答长度
    └── errors.log         # 从原始日志中筛出的 error / retry / timeout / killed 行；没有则写 none
```

**run_id 命名规则**：

| 前缀 | 含义 |
|---|---|
| `train_<expert>` | 专家 SFT，例如 `train_swe` |
| `train_mopd` | MOPD |
| `merge_linear` / `merge_ties` | 合版 |
| `fold_<expert>` | 把专家 LoRA 合进基座，导出完整权重，供推理服务加载 |
| `eval_<model>_<hle\|asi>` | 在 DGX 上的评测 |
| `terminal_<model>` / `swe_<model>` | 在 x86 主机上的评测 |

**重试与中断**：同一 run_id 如果出现多次，`resources.csv` 的 `attempts` 会大于 1。

| run | 原因 |
|---|---|
| `train_mopd` | 第一次以 batch 8 启动，按计划主动停止，改用 batch 16 重新运行 |
| `eval_base_hle` / `eval_merged_linear_hle` | 退出码为 1：ModelScope judge 余额用完后，评分进程被主动终止。生成的答案保留了下来，之后改用本地 judge 评分 |
| `eval_*_asi`（退出码 143） | ASI 评分阶段的 VLM judge 卡住，进程被终止，之后改为逐题评分重新计分 |
| `swe_expert_swe`、`terminal_expert_terminal` | harbor 写完 job 级 `result.json` 后没有退出（日志里反复出现 `cannot schedule new futures after shutdown`）。所有 trial 都已完成，进程被手动终止，剩下的重判 / 计分步骤用 `scripts/finish_swe_regrade.sh`、`scripts/finish_tb2_score.sh` 按原参数补跑 |
| `swe_base`、`swe_merged_linear`（attempts=3） | SWE 镜像拉取失败后重试；最终 6 个任务的镜像在 DGX 上构建后推送到 x86 |
| HLE 重复运行 mopd / merged_ties | 主动中止：会和 merged_ties 的 TB2 推理争用 DGX 的 GPU（见 README §8） |

**harbor 原始轨迹**：TB2 / SWE 每个 trial 的 agent 轨迹、终端录屏、patch 和 verifier 输出体积较大，没有放进 git，打包为 `harbor_jobs.tar.gz`（GitHub Release 附件，sha256 见 `logs/harbor_jobs.sha256`）。包内路径：
- `results/<model>/<terminal|swe>/jobs/`、`results_rep2/<model>/terminal/jobs/`；
- base 的第一次 TB2 运行在 x86 主机共用的 tb2 工作区执行（同一个 `serve_model.sh` 推理栈），位于 `x86/base_tb2_jobs/qwen3-8b-base-main` 和 `qwen3-8b-base-r2-main`（续跑）。

查看方法：解包后运行 `harbor view <job_dir>`，或者直接读每个 trial 目录下的 `result.json`、`agent/trajectory.json`、`verifier/`。

**未计入 `timed_run.sh` 的 GPU 用量**（DGX，由日志时间戳估算）：本地 judge 服务约 4.5 h（02:10–06:00，与评测并行），HLE 重复运行约 1.2 h，开发集选择约 0.4 h，merged_ties 远程推理服务约 2 h。

**资源与费用的计算方式**：

- GPU 小时 = 运行时长 × GPU 数（每个 run 都是 1 张卡）。
- DGX 是自有设备，按 $1.5 / GPU·h 折算。
- 租用的 L40S 按实际租用单价计（见 README §3）。
- 两台机器上的多个 run 在时间上有重叠：GPU 小时之和衡量的是用量，不等于墙钟时间。
