# Qwen3-4B-Instruct-2507 rich SFT 与 reference evaluation

本次仅实施 Qwen 原模型和 rich SFT 的基线评测，不提交 unlearning 方法训练。
模型是 `Qwen/Qwen3-4B-Instruct-2507`，原生 non-thinking，无需附加 `/no_think`
或 `enable_thinking=False`。不能换成混合思考版本 `Qwen3-4B`。

官方说明：<https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507>。
该架构要求 Transformers >=4.51.0；旧 Gemma/Llama 的 4.48.3 环境保持原样。

## 固定配置

| 项目 | 配置 |
| --- | --- |
| 数据 | 现有 `pmc-rich-all-ia`，975 位患者、2,925 条结构化关系 |
| SFT 样本 | 5,850 条 authored relation 表达 + 975 病例 + 975 note + 975 dialogue + 975 profile = 9,750 |
| 通用文本混合 | 0；沿用旧 rich SFT 的注入构成，Wiki 不混入本次 SFT |
| 训练 | 全参数、5 epochs、seed42、LR1e-5、linear scheduler、warmup.05 |
| Batch | 每卡1 × 8卡 × accumulation4 = 32 |
| 精度与内存 | BF16、SDPA、ZeRO-2、gradient checkpointing、max length4096 |
| Loss | 原生 chat template，assistant-only，包含 assistant EOS；不静默截断 |
| 其他 | weight decay.01，clip1；完整优化器/RNG checkpoint 每50步，轮换保留最近2份 |
| 评测权重 | step50、100、300、600、900、1200、1500，以及 final |
| 基线评测 | 原模型 + 上述8个SFT checkpoint，全部沿用父 campaign 冻结的 v5 validation |
| 额外诊断 | 每个checkpoint对全部2,925条关系做独立QA，按属性与原split分层；不冒充held-out成绩 |
| 选择 | 输出曲线与相对base的变化，不自动把final交给unlearning |

按该版本 Trainer 的 dataloader/accumulation 计算约1,525 optimizer steps；以
`sft/TRAINING_COMPLETE.json` 的实际 global_step/epoch 为准，不沿用旧环境的1,520步标签。
MMLU使用父campaign完整的固定validation文件，并记录实际行数，不把它称为全量MMLU test。
5pp MMLU下降和5pp QA均值增益只作报告标记，不能单独证明注入质量足够。

评测保留QA/cloze/background、attribute/identifier MCQ、F-any/R-any、MMLU、
strict MMLU与解析review。抽查年龄单位、不同患者ID、拒答/解释性输出；模型不同不改变判分器。
目前所有结果属于SFT validation/注入诊断，尚未执行hold-out test或六种unlearning baselines。

## 启动：不覆盖当前checkout，不打断已有job

从最新Llama补齐campaign复制其已冻结代码，以本分支的Qwen专用入口覆盖新增文件。
父campaign的数据/代码哈希均验证；Qwen使用独立输出、环境、模型目录。
`download` 首次解析并记录HF revision，重试继续同一revision，不悄悄切换模型版本。

在ABCI login节点完整复制：

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
export QWEN3_ENV="$(dirname "$CONREP_ENV")/conrep-qwen3-py311-cu124"
q3_base="$PWD/results/validated_v2/0390"
q3_campaign="$q3_base/qwen3-4b-rich-sft-20261011"
q3_model="$CONREP_MODEL_ROOT/Qwen3-4B-Instruct-2507"
q3_boot="$(mktemp -d /tmp/0390_qwen3_boot.XXXXXX)"

git fetch origin codex/0390-qwen3-rich-sft-20261011
q3_ref="$(git rev-parse FETCH_HEAD)"
git show "${q3_ref}:scripts/abci/0390_prepare_qwen3_rich.py" > "$q3_boot/prepare.py"
python3 -I "$q3_boot/prepare.py" --root "$PWD" --ref "$q3_ref" \
  --parent "$q3_base/conrep-llama-positive-completion-20261010" \
  --campaign "$q3_campaign"

bash "$q3_campaign/code/scripts/abci/0390_setup_qwen3_env.sh"
q3_entry="$q3_campaign/code/scripts/abci/0390_qwen3_rich.py"
"$QWEN3_ENV/bin/python" -I "$q3_entry" download --model-dir "$q3_model"
"$QWEN3_ENV/bin/python" -I "$q3_entry" prepare --campaign "$q3_campaign" --model-dir "$q3_model"
"$QWEN3_ENV/bin/python" -I "$q3_entry" start --campaign "$q3_campaign"
)
```

准备阶段在CPU核对全部9,750行的模板、loss mask、EOS和长度，确认2925事实覆盖及原始文件哈希。
按checkpoint政策估算所需空间，包含100GB reserve；空间不足时不提交。
模型下载/环境安装可在已有GPU任务运行时进行。

后台supervisor等待用户账户有空位后，仅提交一个`0390_`作业，使用既有
`R9920261000`/`rt_HF`/单节点8GPU/12小时walltime。所有账户活动job与节点数均计入3上限，
使用与现有campaign相同的submit lock。已有三个job时等待，不删除或暂停它们。
人工在其他入口同时qsub不受本脚本控制，应继续遵守三节点上限。

执行顺序：原模型评测 → 8GPU全参SFT → 各里程碑/最终模型分配到8张GPU并行评测。
不会同时启动另一个Qwen训练。初次base评测只使用一张GPU，随后释放模型并进行8卡SFT。
不自动清理旧Gemma/Llama、父campaign或历史SFT权重。

## 状态、续跑、停止

查看状态；`logs/supervisor.log`记录队列等待/恢复，`logs/<PBS_JOBID>/console.log`
记录训练和评测。`state.json`是队列状态，`progress.json`是每50步更新的训练状态。

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
q3_python="$(dirname "$CONREP_ENV")/conrep-qwen3-py311-cu124/bin/python"
q3_campaign="$PWD/results/validated_v2/0390/qwen3-4b-rich-sft-20261011"
"$q3_python" -I "$q3_campaign/code/scripts/abci/0390_qwen3_rich.py" status --campaign "$q3_campaign"
qstat
)
```

管理员中断、明确walltime/node故障或符合旧恢复策略的queued termination，等待20分钟后续投。
普通训练错误、资源/权限错误、人工qdel均阻塞，不无限重试。已通过身份与内容核对的完整评测复用。
checkpoint必须有所有8 rank RNG、ZeRO优化器shards、完整权重和保存完成标记。
不完整保存移到本campaign的`interrupted-saves`，保留现场。
尚未到第一个50步保存点时被中断，只能从原模型重新开始，并记录resume=null。

SSH断开不影响后台watcher；login主机重启/进程丢失时，重复`start`恢复watcher，
保留STOP/人工取消状态。没有额外10小时campaign截止时间，按任务完成或显式STOP终止。

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
q3_python="$(dirname "$CONREP_ENV")/conrep-qwen3-py311-cu124/bin/python"
q3_campaign="$PWD/results/validated_v2/0390/qwen3-4b-rich-sft-20261011"
"$q3_python" -I "$q3_campaign/code/scripts/abci/0390_qwen3_rich.py" start --campaign "$q3_campaign"
)
```

如果此前主动STOP/qdel，或已修复阻塞原因，使用明确恢复命令。它要求前一个PBS allocation
已经结束，才允许清除阻塞。不要通过修改frozen config来修复参数错误，应创建新campaign。

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
q3_python="$(dirname "$CONREP_ENV")/conrep-qwen3-py311-cu124/bin/python"
q3_campaign="$PWD/results/validated_v2/0390/qwen3-4b-rich-sft-20261011"
"$q3_python" -I "$q3_campaign/code/scripts/abci/0390_qwen3_rich.py" resume --campaign "$q3_campaign"
)
```

停止命令是在上述同一环境中将`resume`改为`stop`。它记录STOP并停止后续提交；
训练在下一步检查时退出，单个正在进行的评测会先结束；不代替`qdel`立刻撤销排队allocation。

## 导出

默认只导出已提交完成标记的评测、配置/来源、队列状态与曲线，不含模型或优化器，
不包含大体积prediction payload。需要答案解析复核时加`--include-predictions`。
可在运行中导出，manifest明确complete=false，未完成评测不混入正式结果。
ZIP位于持久project路径，SFTP使用打印的BUNDLE路径，不使用login节点的`/tmp`。

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
q3_python="$(dirname "$CONREP_ENV")/conrep-qwen3-py311-cu124/bin/python"
q3_campaign="$PWD/results/validated_v2/0390/qwen3-4b-rich-sft-20261011"
"$q3_python" -I "$q3_campaign/code/scripts/abci/0390_qwen3_rich.py" export --campaign "$q3_campaign"
)
```

## 本地验证与边界

本地用真实Qwen3-4B-Instruct-2507 tokenizer审计全部9,750行：最长2,018tokens，
最少13个监督tokens，原生non-thinking prefix，零截断。
20项测试通过，覆盖完整checkpoint/缺rank/部分保存恢复、存档轮换、队列上限、
管理员中断与人工取消区分、重复提交阻止、轻量导出；tiny Qwen3的连续3步训练与
2步保存后续跑到3步的全部模型权重逐项完全一致。
另外42项既有队列监督/恢复回归测试通过；离线准备集成检查通过真实数据和tokenizer，
并验证重复prepare的幂等性（该检查的模型权重、MMLU条目和环境检查使用测试替身，不产生模型成绩）。

这个测试实际发现并修复了HF4.51.3在torch2.5.1下恢复NumPy RNG的allowlist缺失：
只在训练调用内允许必要的NumPy RNG类型，保持weights_only restricted loader。
本地CPU测试使用Python3.12/torch2.5.1+cpu及指定Transformers/Accelerate/NumPy版本。
ABCI目标环境是Python3.11/CUDA12.4；真实4B权重的8GPU ZeRO-2训练尚未在此执行，
不能把CPU恢复测试称为ABCI分布式精确续跑验证。启动时会检查实际8卡分配与全部环境版本。
