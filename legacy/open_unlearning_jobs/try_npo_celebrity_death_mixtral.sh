#!/bin/sh
#$-cwd
#$-l node_q=1
#$-l h_rt=24:00:00
#$-p -3
#$-N unlearn_npo_mixtral_celebrity_death

module load jupyterlab/4.1.4
module load miniconda/24.1.2
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export HUGGINGFACE_HUB_CACHE=/gs/bs/tga-TDSAI/h1kkk/HF/huggingface_cache/hub
export HF_DATASETS_CACHE=/gs/bs/tga-TDSAI/h1kkk/HF/huggingface_cache/datasets
conda activate /gs/bs/tga-TDSAI/h1kkk/conda/envs/unlearning
conda info --envs

CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python src/train.py --config-name=unlearn.yaml experiment=unlearn/celebrity_death_npo/default task_name=celebrity_npo_death_Mixtral
