#!/bin/sh
#$-cwd
#$-l gpu_1=1
#$-l h_rt=24:00:00
#$-p -3
#$-N gs_MistralUniversal_lm0.3_fw0.9_lr7e-06_gm1

module load jupyterlab/4.1.4
module load miniconda/24.1.2
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export HUGGINGFACE_HUB_CACHE=/gs/bs/tga-TDSAI/h1kkk/HF/huggingface_cache/hub
export HF_DATASETS_CACHE=/gs/bs/tga-TDSAI/h1kkk/HF/huggingface_cache/datasets
conda activate /gs/bs/tga-TDSAI/h1kkk/conda/envs/unlearning


python unlearn_eval/batchedEval_loglikelihood.py grid_search/configs/forget_MistralUniversal_lm0.4_fw0.3_lr1e-05_gm1_step100.json
python unlearn_eval/batchedEval_loglikelihood.py grid_search/configs/retain_MistralUniversal_lm0.4_fw0.3_lr1e-05_gm1_step100.json
