# Llama 3.1 8B：一个 PBS 作业完成知识注入、验证与选择

入口 `sft-pipeline --model llama8b` 使用 `meta-llama/Llama-3.1-8B-Instruct`，
直接复用已有 PMC 多格式注入数据与原验证集。对应新版论文的 knowledge injection
阶段；输出是随后所有 unlearning 方法共享的候选起点，尚未进行 unlearning。

同一份 HF 分配依次执行：

1. 8 GPU 普通 full SFT，使用现有 Trainer 和 DeepSpeed ZeRO-2。
2. 等训练进程全部退出后，启动新的 8 个独立验证进程。原始 backbone、全部
   `checkpoint-N` 和 `final` 分配给各 GPU；每个模型都跑完整的现有 PMC 验证
   与 1,140 题 MMLU validation，包含原始回答与 invalid 统计。
3. 所有验证进程成功退出后，生成汇总表，并按现有约束自动选择 SFT checkpoint。

| 设置 | 值 |
| --- | --- |
| 注入数据 | 已准备的 `data/processed/0390/pmc/injection.jsonl` |
| 训练 | Full SFT，5 epochs，LR `1e-5`，BF16，SDPA，ZeRO-2 |
| 有效 batch | 8 GPU × 每卡 1 × 梯度累积 4 = 32 |
| 训练长度 / 保存 | 1,024 tokens；每 50 updates 保存，另保留最后 checkpoint 与 final |
| 评测 | 现有 PMC QA/cloze/background/MCQ；MMLU instructed-generation v3；batch 8 |
| 选择 | 相对自身原模型 MMLU 降幅 ≤ 2 个百分点；QA 均值提升 ≥ 5 个百分点；近似最优容差 0.5 个百分点，优先较早 checkpoint |
| PBS | `gcg51557` / `R9920261000` / `rt_HF` / 单节点 8 GPU / `0390_l8sft42` |
| walltime | 首轮申请上限 3 小时，包含训练与验证；不是实测耗时 |

现有 Python 3.11 / Transformers 4.48.3 环境继续使用。Wiki 在后续 unlearning
中作为 general retain；这一轮知识注入继续沿用 PMC 数据。

## 更新、下载、提交

在 ABCI 登录节点执行。`huggingface-cli download` 需要联网及该模型的访问权限；
计算作业内保持离线。已完成的数据准备无需重跑。

```bash
(
set -eo pipefail

cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"

git pull --ff-only origin codex/0390-conrep-v2
git rev-parse --short HEAD
mkdir -p logs/0390 "$CONREP_MODEL_ROOT"

unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --exclude 'original/*' \
  --local-dir "$CONREP_MODEL_ROOT/Llama-3.1-8B-Instruct" \
  2>&1 | tee -a logs/0390/llama8b_download.log

unset CUDA_VISIBLE_DEVICES
python scripts/abci/0390_submit.py sft-pipeline \
  --model llama8b \
  --run-id l8sft42 \
  --rtype rt_HF \
  --nproc 8 \
  --walltime 03:00:00 \
  --output results/validated_v2/0390/llama8b/injection-seed42

qstat -u "$USER"
)
```

提交命令追加 `--dry-run` 可只查看 PBS；不加则实际提交一个作业。提交器继续统计
当前账户所有项目的排队/运行作业，达到两个时停止提交。下载若报 403，需先在
Hugging Face 的该模型页面开通访问；不要把 token 写入命令或日志。

## 结果与作业关联

工作流结果根目录为 `results/validated_v2/0390/llama8b/injection-seed42/`：

| 文件或目录 | 内容 |
| --- | --- |
| `sft/checkpoint-N/`、`sft/final/` | 原始 SFT Trainer 输出；checkpoint 保留训练恢复状态 |
| `sft/TRAINING_COMPLETE.json` | 仅表示训练及 final 保存完成 |
| `validation/base/`、`validation/checkpoint-N/`、`validation/final/` | 每个模型的 PMC/MMLU 预测与 metrics.json |
| `sft-validation-summary.tsv` | QA、MMLU、相对 base 的变化、invalid 百分比；同时打印进作业日志 |
| `selected-sft.json` | 按既定规则选中的路径、指标、约束与拒选原因 |
| `sft-pipeline-config.json` | 所有子进程共用的已解析配置，JSON 数值类型保留 |
| `pipeline.json` | 当前作业 ID、各阶段状态、命令、时间、最终所选 checkpoint |
| `pipeline-events.jsonl` | 追加记录每次执行的阶段事件，带完整 PBS job ID |

全部 stdout/stderr 继续实时写到 `logs/0390/runs/<PBS job ID>/console.log`；
上述配置、状态、汇总、选择和 metrics 一并进入轻量归档。完整预测和权重保留在结果目录。

作业结束后：

```bash
python scripts/abci/0390_logs.py collect
python scripts/abci/0390_logs.py list

cat results/validated_v2/0390/llama8b/injection-seed42/pipeline.json
cat results/validated_v2/0390/llama8b/injection-seed42/sft-validation-summary.tsv
cat results/validated_v2/0390/llama8b/injection-seed42/selected-sft.json
```

`pipeline.json` 的 `status=completed` 表示训练、全部验证和选择完成。如果无模型通过
既定标准，状态为 `no_eligible_checkpoint`，`selected-sft.json` 的 `selected=null`，
作业以非零状态结束；保留全部训练/验证结果，不擅自放宽门槛或选择 final。

## 中断后的恢复

训练中断、尚未产生验证结果：使用新的 `--run-id`、相同 `--output` 和配置，并加
`--resume /该结果根目录/sft/checkpoint-N`。继续训练后自动运行验证与选择。

训练已完成、验证中断：用相同提交命令、相同 `--output`，换新 `--run-id` 并加
`--skip-training`。它检查训练完成标记、实际配置和注入 split，直接进入验证。
现有验证器按 protocol hash 复用已完成报告，未完成部分继续运行；不重新训练。
同一输出目录上的并发执行会被文件锁拒绝。已验证的权重不会被再次训练覆盖。

## 已验证范围

本地通过 CPU 编排测试：阶段顺序、只有一个协调进程、训练/验证失败阻断后续阶段、
完整报告检查、真实 checkpoint 选择、拒选状态、完成训练复用及 PBS 日志/结果关联。
已有模型加载与 SFT Trainer 继续复用；本地尚未实际运行 Llama 8B 的 GPU 训练，
显存、实际 walltime 和注入质量以这次 ABCI 作业结果为准。
