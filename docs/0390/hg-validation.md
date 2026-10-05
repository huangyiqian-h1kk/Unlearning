# HG 上验证原始模型及 SFT checkpoints

`validate-series` 在 HF 上把不同 checkpoints 分配给不同 GPU，每个 checkpoint 已经是
单卡推理。HG 使用同一入口、`--nproc 1`，依次评估原始模型、全部 `checkpoint-N/`
和 `final/`。每个模型验证结束后清理 CUDA cache，再加载下一个模型。

[ABCI 官方资源表](https://docs.abci.ai/v3/en/job-execution/#available-resource-types)
规定 HG 分配 1 GPU、16 逻辑 CPU 核和 160 GB 主存；H 节点的 GPU 为
[H200 SXM 141 GB](https://docs.abci.ai/v3/en/system-overview/#compute-node-h)。
已有两个 backbone 的单卡诊断执行证据；本次 batch 16 的完整序列仍需由实际作业验证。

| 模型 | run-id | 资源 / 进程 | inference batch | 申请 walltime |
| --- | --- | --- | --- | --- |
| Llama 3B | `l3sftv3hg` | HG / 1 | 16 | 02:00:00 |
| Qwen 7B | `q7sftv3hg` | HG / 1 | 16 | 03:00:00 |

原推理 batch 为 8；此处增至 16，以增加单卡吞吐。walltime 是首轮运行预算，
不是实测耗时。只改变资源、调度和推理 batch：仍使用 BF16、SDPA、greedy，
完整 PMC 和固定 1,140 题 MMLU。输入长度、生成长度和评分规则由原配置决定。
这一步不进行梯度更新，SFT 的 batch、学习率和已保存权重均不需要调整。

## 生成并提交

加载已有 `local.env` 和 venv 后，在仓库根目录执行。以下入口保留公共提交器的
项目、预约队列、`0390` 作业前缀、两作业上限和自动日志。

```bash
# 仅生成并打印两个 PBS 文件，不访问调度器、不撤销或提交作业。
python scripts/abci/0390_validate_hg.py

# 撤销对应的排队 HF 验证作业，然后提交两个 HG 验证作业。
python scripts/abci/0390_validate_hg.py --submit --replace-queued-hf
```

切换只查找本仓库 `.jobid` 记录中 `l3sftv3`、`q7sftv3` 对应的旧作业。
它检查用户、作业名及 Q 状态，删除前再次查询；检测到 R 或其他活动状态就停止，
不会主动终止已开始的验证。旧作业退出后补收其日志，再通过原提交器申请 HG。
其他项目的作业也计入两作业限制。执行前会先生成两份新 PBS，检查 SFT 完成标记
与 final 的模型配置文件，以免撤销后才发现基本输入缺失。

若 HF 从未提交或已结束，直接提交也可：

```bash
python scripts/abci/0390_validate_hg.py --submit
```

只处理一个模型时加 `--model llama3b` 或 `--model qwen7b`。重复执行时，若所选模型
已有 HG 作业排队或运行，脚本会停止，避免重复提交。

## 输出与后续选择

- PBS：`logs/0390/jobs/0390_validate-series_<model>_<run-id>.pbs`。
- 日志：`logs/0390/runs/<PBS job ID>/console.log`，总览为 `logs/0390/INDEX.md`。
- 结果：`results/validated_v2/0390/<model>/sft-validation-v3-hg/`。
- 结果子目录仍为 `base/`、`checkpoint-N/` 和 `final/`，各自保存指标与原始预测。

HG 使用新的输出目录，避免与旧 batch 配置的缓存混用。后续 `select` 的
`--baseline-metrics` 和 `--metrics` 都应指向 `sft-validation-v3-hg`。
若因 walltime 中断，使用相同参数重新提交时，已完整保存且协议匹配的 checkpoint
验证会被跳过；未完成的 checkpoint 重新评估。

作业结束后在登录节点补收最终 PBS 记录：

```bash
python scripts/abci/0390_logs.py collect
python scripts/abci/0390_logs.py list
```

本地检查：`python tests/0390/test_hg_validation.py -v` 的 6 个测试覆盖实际 PBS 生成、
仅替换目标排队作业、作业开始运行后的中止、账号归属和两作业上限；公共提交器的 7 个
测试和评测协议的 6 个测试亦通过。这些检查不替代 batch 16 的实际 GPU 运行验证。
