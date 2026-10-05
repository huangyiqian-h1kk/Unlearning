# 0390 新版实验执行方案

这条入口用于新版实验：Qwen2.5-7B-Instruct 和 Llama-3.2-3B-Instruct，
原 PMC 多格式 knowledge injection，新的 ConRep，以及 NPO、RMU、FALCON、
LUNAR、SAGO、ReLearn。默认先执行 PMC；原 celebrity 数据也可通过 data profile 接入。
旧论文结果与训练代码保留，不与新实验输出混写。

## 1. 文件与环境

| 路径 | 用途 |
| --- | --- |
| `src/conrep/v2/` | Corruption、causal embedding、contrastive losses、ConRep trainer |
| `src/experiments/` | 数据转换、标准 SFT、统一验证、checkpoint 选择、表示分析 |
| `src/experiments/baselines/` | 六种 baseline 的训练和数据准备 |
| `configs/0390/` | 两个 backbone、共同训练参数、method overlays、ablations |
| `environments/0390/` | Python 3.11 依赖和已解析的版本约束 |
| `scripts/abci/0390_*.py`、`0390_*.sh` | PBS 生成、提交、GPU 启动、环境安装 |
| `jobs/0390_*.pbs` | 可直接检查的短验证 PBS 样本 |
| `results/validated_v2/0390/` | 新训练、验证和分析输出；不提交模型或运行结果 |

服务器工作根目录固定为 `/groups/gcg51557/experiments/0390_rlsd/unlearning`，
仓库目录为其下的 `Unlearning/`。环境、模型和下载缓存均放在这个工作根目录下。
以下命令在仓库根目录执行。

```bash
cp environments/0390/local.env.example local.env
source local.env
# 使用已安装的 miniforge，在本项目目录下创建 Python 3.11 bootstrap。
source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh
conda create --prefix "$CONREP_WORK_ROOT/envs/bootstrap-py311" python=3.11 -y
bash scripts/abci/0390_setup_env.sh
source "$CONREP_ENV/bin/activate"
```

安装需要联网，计算节点训练按离线方式运行。基础版本为 PyTorch 2.5.1/CUDA 12.4、
Transformers 4.48.3、PEFT 0.14.0、Accelerate 1.3.0；SFT 使用 DeepSpeed 0.16.2。
采用 SDPA，不要求 FlashAttention 编译。安装脚本运行 `pip check` 并保存实际 freeze。
CPU 验证与 ABCI GPU 验证的范围见 [verification.md](verification.md)。

## 2. 准备原数据和离线资产

输入是原 PMC 的 CSV/JSONL；保留原 forget/retain 划分，不生成新的 injection 内容。
先在可访问资产的环境中物化 LFS 数据；已有服务器副本也可放回 catalog 对应路径。

```bash
git lfs pull --include='data/clinicia/**'
python scripts/reproduce.py data-status --require-materialized

# 在联网环境执行；模型已下载时省略 --models。
python scripts/0390_experiment.py assets --models --model-root "$CONREP_MODEL_ROOT"
python scripts/0390_experiment.py prepare --config configs/0390/llama3b.yaml
```

`assets` 准备最多 50,000 条 WikiText-103 训练段落、固定 MMLU validation 子集、
完整 MMLU 导出、STS-B validation，以及 LUNAR 的官方 refusal calibration 资产。
Llama 权重下载需要该账户已有模型访问权限。可在联网机器准备后整体复制到 ABCI，
训练作业不会自行联网下载。若已有合适的 Wiki 数据，可设置 `data.general_file` 替换来源。

`prepare` 输出 `data/processed/0390/pmc/`：

- `injection.jsonl`：原 CSV 第一列逐行保留；QA 用 assistant-only CE，文档用完整 causal LM CE。
- `injection_retain_only.jsonl`：从同一 injection 数据中筛出只涉及 retain patient ID 的行。
- `forget.jsonl`、`retain.jsonl`：canonical fact 与原有 retain paraphrase 列。
- `general.jsonl`：固定 Wiki 候选池，对所有方法一致。
- `manifest.json`：来源文件 SHA-256、实际行数、无法解析 injection 身份的行数。

Retain-only 对照只适用于当前 PMC 按 patient ID 划分的设定。若身份无法解析，
对应行不会被悄悄纳入 ideal 模型，retain-only SFT 会要求先解决匹配问题。
它仍从原始 backbone 开始，不能从已注入 forget 的 SFT checkpoint 开始。

## 3. ABCI 短验证

提交器生成的 PBS 始终包含：

```bash
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1
#PBS -N 0390_...
#PBS -j oe
#PBS -k oe
```

[ABCI 官方资源表](https://docs.abci.ai/v3/en/job-execution/)将 Reserved 服务列为 `rt_HF`。
所以这里不假定预约队列支持 `rt_HG`。节点申请保持 `rt_HF`，按阶段缩短 walltime。

| 阶段 | 默认 walltime | GPU 进程数 |
| --- | --- | --- |
| preflight / 真实 backbone 两步 ConRep smoke | 20 分钟 | 8 |
| Llama 3B / Qwen 7B SFT | 3 / 6 小时 | 8 |
| Llama 3B / Qwen 7B ConRep | 1 / 2 小时 | 8 |
| baseline | 3 小时 | 8 |
| 一组 checkpoints 验证 | 4 小时 | 8，各 rank 分配不同 checkpoints |
| FALCON MI / ReLearn augmentation | 2 小时 | 1 |
| embedding 分析 | 1 小时 | 1 |

这些是初始申请上限，不是实测运行时间。首轮根据日志调整 `--walltime`。
所有 GPU 任务均为单节点，当前不引入跨节点通信。

```bash
# 第一遍可以加 --dry-run 看完整 PBS；去掉后才提交。
python scripts/abci/0390_submit.py smoke --model llama3b --run-id smoke \
  --set run.output_dir=results/validated_v2/0390/llama3b/gpu-smoke

# 第二个短任务确认 full SFT 的 ZeRO-2 路径。
python scripts/abci/0390_submit.py sft --model llama3b --run-id sftsmoke \
  --walltime 00:20:00 --set sft.max_steps=2 --set sft.save_steps=2 \
  --set run.output_dir=results/validated_v2/0390/llama3b/gpu-sft-smoke
```

smoke 会检查依赖、数据、chat template、GPU all-reduce，然后在原始 3B backbone
上运行两步 ConRep 和 checkpoint 保存。它只验证执行，不作为实验结果。
第二个任务用原 injection 数据执行两步标准 SFT，输出同样仅用于验证。
提交器统计当前用户所有排队/运行中的作业，包括其他项目；已有两个时不会提交第三个。

## 4. 知识注入与停止点

SFT 是标准 full-parameter causal LM 训练，5 epochs、LR `1e-5`，8 GPU 下有效 batch 32，
最大长度 1024，ZeRO-2；保存间隔 250 optimizer steps，并保留最终模型。
Wiki 不混入这一步。参数是首轮起点，实际配置写入每个 run 的 `resolved_config.json`。

```bash
python scripts/abci/0390_submit.py sft --model llama3b --run-id sft \
  --set run.output_dir=results/validated_v2/0390/llama3b/sft

python scripts/abci/0390_submit.py sft --model qwen7b --run-id sft \
  --set run.output_dir=results/validated_v2/0390/qwen7b/sft
```

输出为每个 SFT run 的 `checkpoint-N/` 与 `final/`。SFT checkpoints 包含 Trainer
恢复状态。恢复时使用相同 config、输出目录，并追加 `--resume /绝对路径/checkpoint-N`。

训练结束后，先验证原始 backbone，再并行验证 SFT checkpoints。下面以 Llama 为例：

```bash
python scripts/abci/0390_submit.py validate --model llama3b --run-id baseval \
  --checkpoint "$CONREP_MODEL_ROOT/Llama-3.2-3B-Instruct" \
  --output results/validated_v2/0390/llama3b/base-validation

python scripts/abci/0390_submit.py validate-series --model llama3b --run-id sftval \
  --checkpoint-root results/validated_v2/0390/llama3b/sft \
  --output results/validated_v2/0390/llama3b/sft-validation

# 两个验证作业完成后在登录节点选择，无需 GPU。
python scripts/0390_experiment.py select --config configs/0390/llama3b.yaml \
  --selection-stage sft \
  --baseline-metrics results/validated_v2/0390/llama3b/base-validation/metrics.json \
  --metrics results/validated_v2/0390/llama3b/sft-validation/*/metrics.json \
  --output results/validated_v2/0390/llama3b/selected-sft.json

export CONREP_SFT_CHECKPOINT="$(python -c 'import json; print(json.load(open("results/validated_v2/0390/llama3b/selected-sft.json"))["selected"])')"
export SFT_VALIDATION_METRICS="results/validated_v2/0390/llama3b/sft-validation/$(basename "$CONREP_SFT_CHECKPOINT")/metrics.json"
```

初始选择规则：MMLU validation 相对原 backbone 下降不超过 2 个百分点，
forget/retain QA 的等权均值至少上升 5 个百分点；在合格 checkpoints 中最大化该 QA 均值，
差距不超过 0.5 个百分点时选更早的 checkpoint。约束在配置中显式给出。
没有合格 checkpoint 时输出拒绝原因，不自动选择训练最后一步。

MMLU validation 固定取每个 subject 的 20 个 test 样本，使用该 subject 的 5 个 dev
示例；指标为 subject macro accuracy、native chat prompt 下答案字母 continuation
likelihood。这是用于控制训练成本的固定 validation 协议，不冒充完整 MMLU 测量。
`mmlu_full.jsonl` 留给后续完整 utility 测量；不得混用两套结果选择 checkpoints。

Retain-only 对照使用同一 SFT 入口：

```bash
python scripts/abci/0390_submit.py sft --model llama3b --run-id retainonly \
  --retain-only --set run.output_dir=results/validated_v2/0390/llama3b/retain-only
```

验证后用 `select --selection-stage retain-only`，以 retain QA 为选择目标。

## 5. ConRep 和 baseline

每个 backbone 的所有方法共用该 backbone 选定的 SFT checkpoint。
把 `--checkpoint` 作为提交参数传入 PBS，避免依赖登录 shell 中未传入计算节点的变量。

```bash
python scripts/abci/0390_submit.py unlearn --model llama3b --run-id conrep42 \
  --checkpoint "$CONREP_SFT_CHECKPOINT" \
  --set run.output_dir=results/validated_v2/0390/llama3b/conrep-seed42

python scripts/abci/0390_submit.py baseline --model llama3b --run-id npo42 \
  --method npo --checkpoint "$CONREP_SFT_CHECKPOINT" \
  --set run.output_dir=results/validated_v2/0390/llama3b/npo-seed42
```

RMU/SAGO 使用同样命令，将 method 与输出目录改成 `rmu` / `sago`。
FALCON、LUNAR 和 ReLearn 的准备与完整命令见 [baselines.md](baselines.md)。
输出包括 `train.jsonl`、方法参数和起始模型 lineage、`checkpoint-N/`、优化器及 RNG 状态。
ConRep/NPO/SAGO/ReLearn 保存 LoRA adapter；RMU/FALCON/LUNAR 保存完整修改后模型。
所有结果共用 `validate`，不需要先手动 merge adapter。

完成后按 SFT 相同方式 `validate-series`；unlearning selection 的 baseline 必须是
**选定 SFT checkpoint 的验证结果**，而不是原始 backbone。默认在 MMLU 下降不超过
2 点、specified retain 六任务均值下降不超过 5 点的 checkpoints 中，最小化 forget
六任务均值；所有候选结果保留。它是当前 validation 选择规则，不是最终论文综合指标。

```bash
python scripts/abci/0390_submit.py validate-series --model llama3b --run-id conrepval \
  --checkpoint-root results/validated_v2/0390/llama3b/conrep-seed42 \
  --output results/validated_v2/0390/llama3b/conrep-validation

# 验证作业完成后：
python scripts/0390_experiment.py select --config configs/0390/llama3b.yaml \
  --selection-stage unlearn \
  --baseline-metrics "$SFT_VALIDATION_METRICS" \
  --metrics results/validated_v2/0390/llama3b/conrep-validation/*/metrics.json \
  --output results/validated_v2/0390/llama3b/selected-conrep.json
```

`SFT_VALIDATION_METRICS` 指向前一步选中 checkpoint 对应的 `metrics.json`。
评测缓存绑定输入文件、scoring code 和评测参数；变更协议后必须使用新的输出目录，
不能直接沿用旧 scores。Checkpoint 选择同样要求候选与参照具有相同协议 hash。
先用 seed 42 跑通两个 backbone 的 SFT 和所有方法，再按相同配置扩展 seed 43、44。
`--set run.seed=43` 必须搭配新的输出目录。各方法的 retain 权重和学习率再做同等预算搜索，
不要把首轮默认配置直接当作最终最优 baseline。

## 6. 表示分析、ablation 和原 celebrity 数据

```bash
python scripts/abci/0390_submit.py analyze --model llama3b --run-id before \
  --checkpoint "$CONREP_SFT_CHECKPOINT" \
  --output results/validated_v2/0390/llama3b/geometry-before.json

python scripts/abci/0390_submit.py unlearn --model llama3b --run-id nogcl \
  --ablation no_general_cl --checkpoint "$CONREP_SFT_CHECKPOINT" \
  --set run.output_dir=results/validated_v2/0390/llama3b/ablation-no-general-cl
```

对选定 unlearned checkpoint 再运行同一 `analyze`。当前诊断包括 STS-B Spearman、
各集合独立的 effective rank / cosine、retain 自身 paraphrase 相似度与 hardest-other
相似度、forget 到自身 corruption 的相似度。几何变化要与生成验证一起解释。
已实现的 ablation 名称与确切参数见 `configs/0390/ablations.yaml`。

Regime A 不做 PMC SFT，直接从原始 backbone 开始。以 diagnosis 为例：

```bash
python scripts/0390_experiment.py prepare --config configs/0390/llama3b.yaml --data-profile diagnosis
python scripts/abci/0390_submit.py unlearn --model llama3b --run-id diagnosis \
  --data-profile diagnosis \
  --set run.output_dir=results/validated_v2/0390/llama3b/diagnosis-conrep
```

`--data-profile deaths` 对应另一方向。specified retain 使用另一个 celebrity 集合，
general retain 仍为 Wiki；这保留历史数据，不能把它称为 PMC 式 near-i.i.d. 划分。
Diagnosis 旧发布物只有两类 MCQ，缺失任务不补造。当前主线方法验证和注入对照以 PMC 为主。

## 7. 当前验证边界

本地使用从配置随机初始化的微型 Llama/Qwen 实际训练，不加载目标数据或模型权重。
尚未在 ABCI 安装环境或执行 3B/7B 训练，没有任何新论文结果。
首个服务器任务应为 20 分钟 smoke；通过后再执行 SFT。现有 ClinicIA probes 在本轮
充当 checkpoint validation，后续论文 final evaluation 仍需按计划另外定稿。
