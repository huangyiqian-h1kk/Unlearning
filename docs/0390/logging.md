# 0390 作业日志与历史补收

以 **完整 PBS job ID** 为主键，例如 `1234567.pbs1`。`run-id` 是便于阅读的标签，
同一标签重新提交得到不同 job ID，因此不会覆盖前一个作业的档案。
继续使用现有 `0390_submit.py` 命令即可，不需要每次手工添加 `tee`。

## 每个作业保存什么

统一入口是仓库内的 `logs/0390/INDEX.md`，同时提供 `index.tsv` 和 `index.json`。
单个作业位于 `logs/0390/runs/<完整 PBS job ID>/`：

| 文件 | 内容 |
| --- | --- |
| `job.json` | 作业名、阶段、模型、run-id、资源、状态及来源、结果目录、各记录的链接 |
| `submission.json`、`submission.pbs` | 提交时的 Git commit、参数、队列和资源，以及实际提交的 PBS 文本 |
| `submission-config.json` | 提交时解析的入口配置，包括 `--set` 等覆盖项 |
| `runtime.json`、`runtime-config.json` | 实际启动时的 Git commit、已跟踪文件修改状态、主机、Python/包版本、入口配置、开始/结束 UTC 时间、进程退出码 |
| `console.log` | 实时合并 stdout/stderr；包含 CUDA 初始化和所有 torchrun rank 的输出 |
| `scheduler.json`、`scheduler-query.json` | `collect` 补收的 PBS 状态、退出码、实际耗时和资源；另记最近查询是否成功 |
| `pbs.stdout.log`、`pbs.stderr.log` | `collect` 从 `$HOME` 等原位置复制的 PBS 原始输出；`-j oe` 合并输出时可以没有 stderr 文件 |
| `captured-logs.json` | 原始日志位置、快照时间、字节数和 SHA-256；缺失日志明确标注 |
| `artifacts/`、`artifacts.json` | 轻量结果快照及其来源、时间、SHA-256 |
| `history.json` | 旧作业在对话中已确认的事实；与真实采集的 PBS 记录分开保存 |

提交版本和实际启动版本分别记录，能够发现作业排队期间 `git pull` 带来的代码变化。
修改过的工作区会在 `tracked_changes` 中显示；提交 SHA 本身不代表这些未提交修改。
入口配置发生阶段内调整时，以结果目录中的 `resolved_config.json` 为训练配置证据。
不复制 `local.env`、完整环境变量或 PBS `Variable_List`，避免把访问凭据写进档案。

监督进程在 CUDA 初始化前启动，训练退出码不会被日志管道掩盖。训练过程和结束记录
自动归档；PBS 最终状态和 `$HOME` 原始输出需在作业结束后执行 `collect` 补收。
若作业被强制杀死而来不及写结束记录，后续 PBS 状态可补足这一信息。
同一个 PBS ID 被调度器重跑时，`console.log` 追加新的开始/结束标记，旧运行元数据
进入 `attempts.jsonl`；`runtime.json` 和自动结果快照反映最新一次运行。

自动结果快照包括完成标记、`resolved_config.json`、`lineage.json`、`trainer_state.json`、
`train.jsonl`、`metrics.json`、`report.json`（单文件最多 20 MiB）。模型和优化器权重、
完整预测文件保留在 `output_dir` 指向的结果目录。结果快照一旦采集，后续普通 `collect`
不会用可能已复用的输出目录覆盖它。快照表示采集时的文件内容，不单独证明每个文件
均由该作业生成，尤其是历史回收或 resume 的情形。

## 首次回收现有日志

在 ABCI 登录节点执行，不提交 GPU 作业：

```bash
cd /path/to/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
git pull --ff-only origin codex/0390-conrep-v2

python scripts/abci/0390_logs.py collect
python scripts/abci/0390_logs.py list
```

命令会自动发现本仓库 `logs/0390/jobs/*.jobid` 和已经建立档案的作业。它只读取这些
ID，不扫描其他项目的全部作业。也可通过位置参数明确传入一个或多个 PBS job ID。
源日志保留在原位置。登录节点上的顶层 `logs/0390/*.log`（如 CPU 测试、数据准备日志）
复制到 `logs/0390/setup/`，不虚构 PBS job ID。

若有从对话整理出的私有台账，可以放在 gitignored 的 `logs/0390/history.json`，再运行：

```bash
python scripts/abci/0390_logs.py collect --history
# 或显式指定另一份本地 JSON 文件。
python scripts/abci/0390_logs.py collect --history /path/to/private-history.json
```

JSON 结构为 `{"jobs": [{"job_id": "1234567.pbs1", "job_name": "0390_example",
"stage": "sft", "model": "llama3b", "run_id": "example", "observed_exit_code": 0}]}`。
还可记录 `observed_walltime`、`submitted_commit`、`output_dir`、`observed_results`、`note`。
历史作业 ID、结果、服务器记录均保存在本地，不随日志工具上传公开仓库。

找不到原始日志会显示 `log=MISSING`；PBS 历史不可查询会显示 `PBS=unavailable`，
不会被解释成成功。已采集到的 PBS 记录仍保留，对话确认的状态标注为 `conversation`。
总补收报告见 `logs/0390/collection-report.json`。后续找到缺失文件后可以再次运行。
历史 `.pbs` 若仍在，会保存为 `recovered.pbs`；`recovered-entry.json` 只是按当前文件
解析该脚本的结果，不能冒充当时配置。以前没有记录的实际启动 commit 保持未知。

## 以后怎么看

```bash
# 作业结束后，补收所有本项目已知作业的 PBS 信息及原始输出。
python scripts/abci/0390_logs.py collect

# 或只补收一个作业。
python scripts/abci/0390_logs.py collect 1234567.pbs1

# 总览 / 单个作业。
python scripts/abci/0390_logs.py list
python scripts/abci/0390_logs.py show 1234567.pbs1

# 新作业运行中可直接 tail，以实际 PBS ID 替换示例值。
# tail -f logs/0390/runs/NEW_JOB_ID.pbs1/console.log
```

日志、历史台账和结果位于已有项目路径下，受 `.gitignore` 保护，不随代码提交；日志管理
代码随 Git 版本化。归档是同一文件系统内的整理，不替代项目数据的独立备份。
