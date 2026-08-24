#!/bin/sh
#$-cwd
#$-l gpu_1=1
#$-l h_rt=24:00:00
#$-p -3
#$-N gs_Llama3_lm0.05_fw0.1_lr1e-05_gm1_epochs4

module load jupyterlab/4.1.4
module load miniconda/24.1.2
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export HUGGINGFACE_HUB_CACHE=/gs/bs/tga-TDSAI/h1kkk/HF/huggingface_cache/hub
export HF_DATASETS_CACHE=/gs/bs/tga-TDSAI/h1kkk/HF/huggingface_cache/datasets
conda activate /gs/bs/tga-TDSAI/h1kkk/conda/envs/unlearning
python ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py grid_search/configs/train_Llama3_lm0.05_fw0.1_lr1e-05_gm1_epochs4.json
python unlearn_eval/batchedEval_loglikelihood.py grid_search/configs/retain_Llama3_lm0.05_fw0.1_lr1e-05_gm1_epochs4.json
python unlearn_eval/batchedEval_loglikelihood.py grid_search/configs/forget_Llama3_lm0.05_fw0.1_lr1e-05_gm1_epochs4.json
