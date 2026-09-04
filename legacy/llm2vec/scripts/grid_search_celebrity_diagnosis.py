import json
import itertools
from pathlib import Path

# Paths to pretrained checkpoints to unlearn
MODEL_CHECKPOINTS = {
    "Llama3": "meta-llama/Llama-2-7b-chat-hf",
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}

# Ranges for grid search
LM_WEIGHTS = [0, 1]
FORGET_WEIGHTS = [0.45]
LEARNING_RATES = [3e-5,2e-5]
GAMMAS = [1]
EPOCHS = [3,4,5]

# Paths to training CSVs (edit accordingly)
RETAIN_CSV = "/gs/bs/tga-TDSAI/h1kkk/unlearn/embedders/LLM2Vec/llm2vec/UnlearnData/wikitext_dup_1_trunc_1.csv"
FORGET_CSV = "/gs/bs/tga-TDSAI/h1kkk/unlearn/embedders/LLM2Vec/llm2vec/UnlearnData/easy_QACelebrityDiagnosis.csv"

# Template for evaluation configs
FORGET_EVAL_TEMPLATE = json.load(open("unlearn_eval/batchedEval_config_loglikelihood_lmZero.json"))

CONFIG_DIR = Path("grid_search_diagnosis/configs")
JOB_DIR = Path("grid_search_diagnosis/jobs")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
JOB_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_TEMPLATE = {
    "model_name_or_path": None,
    "peft_model_name_or_path": None,
    "simcse_dropout": 0.3,
    "bidirectional": False,
    "pooling_mode": "mean",
    "retain_csv_path": RETAIN_CSV,
    "forget_csv_path": FORGET_CSV,
    "remove_unused_columns": False,
    "learning_rate": 3e-5,
    "loss_scale": 20,
    "n_augment": 2,
    "per_device_train_batch_size": 20,
    "gradient_accumulation_steps": 1,
    "do_train": True,
    "disable_tqdm": False,
    "max_seq_length": 128,
    "overwrite_output_dir": True,
    "output_dir": None,
    "logging_steps": 50,
    "save_strategy": "epoch",
    "save_only_model": True,
    "stop_after_n_steps": 4000,
    "lora_r": 16,
    "gradient_checkpointing": True,
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    "seed": 42,
    "hidden_size": 4096,
    "forget_weight": 0.5,
    "lm_weight": 0.1,
    "gamma": 0.5,
    "num_train_epochs":3
}

def main():
    for name, ckpt in MODEL_CHECKPOINTS.items():
        for lm_w, fw, lr, gm, epochs in itertools.product(LM_WEIGHTS, FORGET_WEIGHTS, LEARNING_RATES, GAMMAS, EPOCHS):

            exp_name = f"{name}_lm{lm_w}_fw{fw}_lr{lr}_gm{gm}_epchs{epochs}"
            train_cfg = TRAIN_TEMPLATE.copy()
            if ckpt== "meta-llama/Llama-2-7b-chat-hf":
                peft = "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp"
            else:
                peft = "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
            train_cfg.update({
                "model_name_or_path": ckpt,
                "learning_rate": lr,
                "peft_model_name_or_path": peft,
                "forget_weight": fw,
                "lm_weight": lm_w,
                "gamma": gm,
                "num_train_epochs": epochs,
                "output_dir": f"grid_search_diagnosis/results/{exp_name}/model",
            })
            train_cfg_path = CONFIG_DIR / f"train_{exp_name}.json"
            with open(train_cfg_path, "w") as f:
                json.dump(train_cfg, f, indent=2)

            # evaluation configs

            eval_forget = FORGET_EVAL_TEMPLATE.copy()
            model_out = train_cfg["output_dir"]
            eval_forget.update({"model_path": model_out, "output_dir": f"grid_search_diagnosis/results/{exp_name}/eval_forget"})
            forget_cfg_path = CONFIG_DIR / f"forget_{exp_name}.json"
            json.dump(eval_forget, open(forget_cfg_path, "w"), indent=2)

            # job script
            job_script = JOB_DIR / f"job_{exp_name}.sh"
            with open(job_script, "w") as f:
                f.write(f"#!/bin/sh\n")
                f.write("#$-cwd\n#$-l gpu_1=1\n#$-l h_rt=24:00:00\n#$-p -3\n")
                f.write(f"#$-N gs_{exp_name}\n\n")
                f.write("module load jupyterlab/4.1.4\n")
                f.write("module load miniconda/24.1.2\n")
                f.write("eval \"$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)\"\n")
                f.write("export HUGGINGFACE_HUB_CACHE=/gs/bs/tga-TDSAI/h1kkk/HF/huggingface_cache/hub\n")
                f.write("export HF_DATASETS_CACHE=/gs/bs/tga-TDSAI/h1kkk/HF/huggingface_cache/datasets\n")
                f.write("conda activate /gs/bs/tga-TDSAI/h1kkk/conda/envs/unlearning\n")
                f.write("python ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py {}\n".format(train_cfg_path))
                f.write("python unlearn_eval/batchedEval_loglikelihood.py {}\n".format(forget_cfg_path))
                f.write("lm-eval --model hf --model_args pretrained={},peft={},tokenizer={},parallelize=True --tasks mmlu --batch_size=32".format(ckpt,model_out,ckpt))
                

if __name__ == "__main__":
    main()

