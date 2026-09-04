#!/bin/bash

# 指定需要提交的脚本所在目录
SCRIPT_DIR="./grid_search/jobs"

# 遍历符合 job_MistralUniversal_lm0.7_*.sh 模式的文件并提交
for script in "$SCRIPT_DIR"/job_MistralUniversal_lm0.2_*.sh; do
    if [ -f "$script" ]; then
        echo "Submitting $script"
        qsub -g tga-TDSAI "$script"
    fi
done

