#!/bin/sh
#$-cwd
#$-l gpu_1=1
#$-l h_rt=24:00:00
#$-p -3
#$-N gs_Llama3_lm0_fw0.45_lr6e-06_gm1_epochs4

module load jupyterlab/4.1.4
module load miniconda/24.1.2
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
export HUGGINGFACE_HUB_CACHE=/gs/bs/tga-TDSAI/h1kkk/HF/huggingface_cache/hub
export HF_DATASETS_CACHE=/gs/bs/tga-TDSAI/h1kkk/HF/huggingface_cache/datasets
conda activate /gs/bs/tga-TDSAI/h1kkk/conda/envs/unlearning
python ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py grid_search_diagnosis/configs/train_Llama3_lm0_fw0.45_lr6e-06_gm1_epochs4.json
python unlearn_eval/batchedEval_loglikelihood.py grid_search_diagnosis/configs/forget_Llama3_lm0_fw0.45_lr6e-06_gm1_epochs4.json
lm-eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,peft=grid_search_diagnosis/results/Llama3_lm0_fw0.45_lr6e-06_gm1_epochs4/model,tokenizer=meta-llama/Llama-2-7b-chat-hf,parallelize=True --tasks mmlu --batch_size=32