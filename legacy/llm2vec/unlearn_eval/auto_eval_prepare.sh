#!/usr/bin/env bash
set -euo pipefail

############################################
# 可配置区域
############################################
# 包含模型目录的工作路径（如：graddiff_celebrity_death_llama2 等）
WORKDIR=""   # 改成你的目录（例：/home/ur03151/unlearn）

# 评估脚本（固定路径，按你提供）
EVAL_PY="/gs/bs/tga-TDSAI/h1kkk/unlearn/embedders/LLM2Vec/llm2vec/unlearn_eval/batchedEval_loglikelihood.py"

# qsub 资源与环境（会写进 qsub 脚本，但本脚本不提交）
QSUB_GPU_RES="-l gpu_1=1"
QSUB_HRT="-l h_rt=24:00:00"
QSUB_PRIO="-p -3"
QSUB_CWD="-cwd"

# 环境加载（写入 qsub 脚本）
MODULE_LINES=$'module load jupyterlab/4.1.4\nmodule load miniconda/24.1.2\neval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"\nexport HUGGINGFACE_HUB_CACHE=/gs/bs/tga-TDSAI/h1kkk/HF/huggingface_cache/hub\nexport HF_DATASETS_CACHE=/gs/bs/tga-TDSAI/h1kkk/HF/huggingface_cache/datasets\nconda activate /gs/bs/tga-TDSAI/h1kkk/conda/envs/unlearning\nconda info --envs\npip cache dir\npip list'

# 可选：把输出集中到子目录（留空则写到 WORKDIR 根目录）
CONFIG_DIR="configs"
QSUB_DIR="qsubs"

############################################
# 运行模式
############################################
RUN=0   # 默认 dry-run
[[ "${1:-}" == "--run" ]] && RUN=1

############################################
# 帮助函数
############################################
make_qsub_script_text() {
  local jobname="$1"       # {approach}_{task}_{LLM}_eval
  local eval_body="$2"     # {evaluation bash}
  cat <<EOF
#!/bin/sh
#$QSUB_CWD
$QSUB_GPU_RES
$QSUB_HRT
$QSUB_PRIO
#\$ -N $jobname

$MODULE_LINES

$eval_body
EOF
}

write_file() {
  local path="$1"
  local content="$2"
  if [[ $RUN -eq 1 ]]; then
    echo ">> 写入文件: $path"
    mkdir -p "$(dirname "$path")"
    printf "%s\n" "$content" > "$path"
    [[ "$path" == *.sh ]] && chmod +x "$path" || true
  else
    echo "=== Dry-run: 预览 $path ==="
    echo "$content"
    echo "=== 预览结束 ==="
  fi
}

############################################
# 主流程
############################################
cd "$WORKDIR"
[[ -n "$CONFIG_DIR" ]] && mkdir -p "$CONFIG_DIR"
[[ -n "$QSUB_DIR" ]] && mkdir -p "$QSUB_DIR"

echo "扫描目录：$WORKDIR"
echo "模式：$([[ $RUN -eq 1 ]] && echo RUN || echo DRY-RUN)"

for folder in */ ; do
  folder="${folder%/}"
  [[ ! -d "$folder" ]] && continue

  # celebrity_* 与 PMC 两类
  if [[ "$folder" =~ ^([^_]+)_(celebrity_(death|diagnosis))_([^_]+)$ ]]; then
    approach="${BASH_REMATCH[1]}"
    task_split="${BASH_REMATCH[2]}"      # celebrity_death / celebrity_diagnosis
    llm="${BASH_REMATCH[4]}"

    # 统一的 celebrity config（包含两套评测集）
    cfg_name="config_celebrity_${approach}_${llm}.json"
    cfg_path="${CONFIG_DIR:+$CONFIG_DIR/}$cfg_name"
    cfg_content=$(cat <<EOF
{
  "model_path": "./${folder}",
  "output_dir": "./${folder}",
  "batch_size": 30,
  "max_new_tokens": 50,
  "apply_chat_template": null,
  "mcq_eval_mode": "llh",
  "evaluation_sets": {
    "celebrity_deaths": "data/celebrity_deaths_QA.jsonl",
    "celebrity_diagnosis": "data/celebrity_diagnosis_QA_Test.jsonl"
  },
  "mcq_sets": {
    "mcqs_diagnosis_id": "data/mcqs_diagnosis_id.jsonl",
    "mcqs_diagnosis_att": "data/mcqs_diagnosis_att.jsonl",
    "mcqs_deaths_id_eq": "data/mcqs_deaths_id_eq.jsonl",
    "mcqs_deaths_att": "data/mcqs_deaths_att.jsonl",
    "mcqs_deaths_id_sim": "data/mcqs_deaths_id_sim.jsonl"
  }
}
EOF
)
    write_file "$cfg_path" "$cfg_content"

    # qsub 脚本（先 forget，再 retain）
    jobname="${approach}_${task_split}_${llm}_eval"
    qsub_script="${QSUB_DIR:+$QSUB_DIR/}qsub_${jobname}.sh"
    eval_body=$(cat <<EOC
# Celebrity forget performance
python "$EVAL_PY" "$cfg_path"

# Retain performance (lm-eval)
lm-eval --model hf --model_args pretrained="./${folder},parallelize=True" --tasks wmdp,mmlu --batch_size=32
EOC
)
    qsub_text="$(make_qsub_script_text "$jobname" "$eval_body")"
    write_file "$qsub_script" "$qsub_text"

  elif [[ "$folder" =~ ^([^_]+)_(PMC)_([^_]+)$ ]]; then
    approach="${BASH_REMATCH[1]}"
    task_split="${BASH_REMATCH[2]}"      # PMC
    llm="${BASH_REMATCH[3]}"

    # PMC forget config
    cfg_forget="config_PMC_forget_${approach}_${llm}.json"
    cfg_forget_path="${CONFIG_DIR:+$CONFIG_DIR/}$cfg_forget"
    cfg_forget_content=$(cat <<EOF
{
  "model_path": "./${folder}",
  "output_dir": "./${folder}/forget_eval",
  "batch_size": 15,
  "max_new_tokens": 50,
  "apply_chat_template": null,
  "mcq_eval_mode": "llh",
  "evaluation_sets": {
    "PMC": "/gs/bs/tga-TDSAI/h1kkk/unlearn/embedders/LLM2Vec/llm2vec/UnlearnData/PMC_QA_subset1000_forget100.jsonl"
  },
  "mcq_sets": {
    "mcqs_PMC_forget_att": "/gs/bs/tga-TDSAI/h1kkk/unlearn/embedders/LLM2Vec/llm2vec/UnlearnData/mcqs_PMC_forget_att.jsonl",
    "mcqs_PMC_forget_id_equal": "/gs/bs/tga-TDSAI/h1kkk/unlearn/embedders/LLM2Vec/llm2vec/UnlearnData/mcqs_PMC_forget_id_equal.jsonl",
    "mcqs_PMC_forget_id_identical": "/gs/bs/tga-TDSAI/h1kkk/unlearn/embedders/LLM2Vec/llm2vec/UnlearnData/mcqs_PMC_forget_id_identical.jsonl"
  }
}
EOF
)
    write_file "$cfg_forget_path" "$cfg_forget_content"

    # PMC retain config
    cfg_retain="config_PMC_retain_${approach}_${llm}.json"
    cfg_retain_path="${CONFIG_DIR:+$CONFIG_DIR/}$cfg_retain"
    cfg_retain_content=$(cat <<EOF
{
  "model_path": "./${folder}",
  "output_dir": "./${folder}/retain_eval",
  "batch_size": 15,
  "max_new_tokens": 50,
  "apply_chat_template": null,
  "mcq_eval_mode": "llh",
  "evaluation_sets": {
    "PMC": "/gs/bs/tga-TDSAI/h1kkk/unlearn/embedders/LLM2Vec/llm2vec/UnlearnData/PMC_QA_subset1000_forget100.jsonl"
  },
  "mcq_sets": {
    "mcqs_PMC_forget_att": "/gs/bs/tga-TDSAI/h1kkk/unlearn/embedders/LLM2Vec/llm2vec/UnlearnData/mcqs_PMC_forget_att.jsonl",
    "mcqs_PMC_forget_id_equal": "/gs/bs/tga-TDSAI/h1kkk/unlearn/embedders/LLM2Vec/llm2vec/UnlearnData/mcqs_PMC_forget_id_equal.jsonl",
    "mcqs_PMC_forget_id_identical": "/gs/bs/tga-TDSAI/h1kkk/unlearn/embedders/LLM2Vec/llm2vec/UnlearnData/mcqs_PMC_forget_id_identical.jsonl"
  }
}
EOF
)
    write_file "$cfg_retain_path" "$cfg_retain_content"

    # qsub 脚本（两段 python 评测）
    jobname="${approach}_${task_split}_${llm}_eval"
    qsub_script="${QSUB_DIR:+$QSUB_DIR/}qsub_${jobname}.sh"
    eval_body=$(cat <<EOC
# PMC forget performance
python "$EVAL_PY" "$cfg_forget_path"

# PMC retain performance
python "$EVAL_PY" "$cfg_retain_path"
EOC
)
    qsub_text="$(make_qsub_script_text "$jobname" "$eval_body")"
    write_file "$qsub_script" "$qsub_text"

  else
    echo "跳过未识别目录名: $folder"
  fi
done

echo "完成。模式：$([[ $RUN -eq 1 ]] && echo RUN || echo DRY-RUN)"
echo "提示：本脚本不会自动 qsub，请手动执行。"

