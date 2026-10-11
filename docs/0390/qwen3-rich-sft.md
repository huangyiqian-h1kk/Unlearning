# Qwen3-4B-Instruct-2507 rich SFT 与 reference evaluation

本次仅实施 Qwen 原模型和 rich SFT 的基线评测，不提交 unlearning 方法训练。
模型是 `Qwen/Qwen3-4B-Instruct-2507`，原生 non-thinking，无需附加 `/no_think`
或 `enable_thinking=False`。不能换成混合思考版本 `Qwen3-4B`。

官方说明：<https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507>。
该架构要求 Transformers >=4.51.0；旧 Gemma/Llama 的 4.48.3 环境保持原样。

## 固定配置

依赖补充：PyArrow 固定为 `19.0.1`，与 NumPy `1.26.4` 配套。
只执行 `pip check` 或核对 distribution metadata 不足以验证可导入性。
新环境安装及 prepare/start/worker 均实际检查 Arrow/NumPy/Pandas/Datasets
互操作，并导入 Trainer、PEFT 和 DeepSpeed，失败时不进入评测或训练。

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

## 修复首次 prepare 的 Parent configuration changed

初版 `88b73509` 错将父campaign的文件字节SHA256与解析后JSON的SHA256比较，
导致未修改的配置也被拒绝。修复后使用父campaign一致的 `file_sha`，保留严格校验。
新增测试使用父campaign实际的缩进JSON写入方式；配置内容、格式字节或输入改动仍被拒绝。

如果首次启动停在此异常，环境安装和模型下载已经完成，尚未生成Qwen plan或提交job。
以下命令只归档这个尚未准备完成的目录，然后重建冻结代码并继续prepare/start。
原环境、模型、父campaign和评测数据均复用；后续状态/续跑/导出路径不变。
若已有plan、state、训练或调度文件，归档步骤拒绝执行；不能用它重置已开始的实验。

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
q3_python="$(dirname "$CONREP_ENV")/conrep-qwen3-py311-cu124/bin/python"
q3_base="$PWD/results/validated_v2/0390"
q3_campaign="$q3_base/qwen3-4b-rich-sft-20261011"
q3_model="$CONREP_MODEL_ROOT/Qwen3-4B-Instruct-2507"
q3_boot="$(mktemp -d /tmp/0390_qwen3_fix.XXXXXX)"

git fetch origin codex/0390-qwen3-rich-sft-20261011
q3_ref="$(git rev-parse FETCH_HEAD)"
git show "${q3_ref}:scripts/abci/0390_prepare_qwen3_rich.py" > "$q3_boot/prepare.py"

"$q3_python" -I - "$q3_campaign" "$q3_ref" <<'PY'
import json, pathlib, sys, uuid
p = pathlib.Path(sys.argv[1])
if p.exists():
    boot = json.loads((p / "PREPARING.json").read_text())
    if boot["ref"] != sys.argv[2]:
        allowed = {"prepare.lock", "PREPARING.json", "code", "code-files.json"}
        extra = {x.name for x in p.iterdir()} - allowed
        if extra:
            raise SystemExit(f"Refusing to archive a campaign with runtime files: {sorted(extra)}")
        backup = p.with_name(p.name + ".before-hash-fix-" + uuid.uuid4().hex[:8])
        p.rename(backup)
        print(f"Preserved failed preparation: {backup}")
PY

"$q3_python" -I "$q3_boot/prepare.py" --root "$PWD" --ref "$q3_ref" \
  --parent "$q3_base/conrep-llama-positive-completion-20261010" \
  --campaign "$q3_campaign"
q3_entry="$q3_campaign/code/scripts/abci/0390_qwen3_rich.py"
"$q3_python" -I "$q3_entry" prepare --campaign "$q3_campaign" --model-dir "$q3_model"
"$q3_python" -I "$q3_entry" start --campaign "$q3_campaign"
"$q3_python" -I "$q3_entry" status --campaign "$q3_campaign"
)
```

## 修复 Trainer 导入时的 PyArrow / NumPy 冲突

如果日志包含 `pyarrow requires NumPy 2.0 or newer, found 1.26.4`，
说明初版依赖遗漏了 PyArrow pin；错误发生于 Trainer 导入，尚未进入 SFT 更新。
保持 NumPy `1.26.4`，将 PyArrow 固定为 `19.0.1`。
新版本 requirements、环境安装和 prepare/start/worker 都补充了真实训练导入检查。

已准备的 `51569a25` campaign 使用下面的独立修复入口；不要重新运行 prepare
或重建 `code/`。修复入口验证原有冻结代码/配置/输入，检查 PBS 最终记录和
supervisor/worker 锁，只在没有活动作业时安装 PyArrow wheel（`--no-deps`）。
随后执行 `pip check`、Arrow/Datasets 数据转换和 tiny Qwen3 的一次 CPU Trainer
更新；其他包的版本必须保持一致，全部通过才调用原有 `resume`。
这不是八卡 GPU/DeepSpeed 运行验证，实际 GPU 训练仍需查看重启后的日志。

修复记录和前后依赖清单写入 `maintenance/pyarrow-*/`。原 `plan.json`、
冻结代码、评测 marker 和逐题结果保留，原 worker 会验证并复用完整的 base 评测。
该环境仅用于本 Qwen campaign；Gemma/Llama 的独立环境不在安装目标中。

在 ABCI login 节点执行，输出直接写到终端，避免当前 shell 的 stdout 管道隐藏输出：

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
q3_campaign="$PWD/results/validated_v2/0390/qwen3-4b-rich-sft-20261011"
q3_python="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$q3_campaign/plan.json")"
q3_boot="$(mktemp -d /tmp/0390_qwen3_arrow.XXXXXX)"
git fetch origin codex/0390-qwen3-rich-sft-20261011
q3_ref="$(git rev-parse FETCH_HEAD)"
git show "${q3_ref}:scripts/abci/0390_repair_qwen3_arrow.py" > "$q3_boot/repair.py"
"$q3_python" -I "$q3_boot/repair.py" --campaign "$q3_campaign" --resume
"$q3_python" -I "$q3_campaign/code/scripts/abci/0390_qwen3_rich.py" status --campaign "$q3_campaign"
qstat
) > /dev/tty 2>&1
```

出现 `QWEN3_TRAINING_PREFLIGHT_PASS`、`REPAIR_VERIFIED=...` 和
`Supervisor PID=...` 表示环境检查与恢复入口成功。调度器仍遵守账户最多三个
活动 job/node 的限制，可能先等待空位，再提交新的 Qwen job ID。
如果提示 PBS 作业或 supervisor 仍活动，等待其结束后重跑同一段；
入口不会主动删除作业、清除 STOP 或在状态不明时改环境。

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
