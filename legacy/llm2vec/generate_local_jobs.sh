#!/bin/bash

# 原始job脚本存放目录
JOB_DIR="./grid_search/jobs"

# 新生成的本地job脚本存放目录
LOCAL_JOB_DIR="./grid_search/jobs_local"
mkdir -p "$LOCAL_JOB_DIR"

# 循环处理所有符合模式的job脚本
for job_script in "$JOB_DIR"/job_Qwen_lm0.1_fw*_lr*_gm*.sh; do
    if [ -f "$job_script" ]; then
        filename=$(basename "$job_script")
        local_filename="${filename%.sh}_local.sh"
        local_script="$LOCAL_JOB_DIR/$local_filename"

        echo "Processing $filename -> $local_filename"

        # 提取python命令行并写入新文件
        grep "^python " "$job_script" > "$local_script"

        # 添加bash头
        sed -i '1i#!/bin/bash\n' "$local_script"

        # 赋予执行权限
        chmod +x "$local_script"
    fi
done

