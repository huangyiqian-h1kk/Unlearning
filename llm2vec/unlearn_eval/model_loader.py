import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def load_model(cfg):
    path = cfg["model"]["path"]
    dtype_map = {"auto":None,"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}
    dtype = dtype_map[cfg["model"]["dtype"]]
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["model"]["device"]=="auto"
                          else cfg["model"]["device"])

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 判断 LoRA
    if cfg["model"]["peft"]["enabled"]:
        base_name = PeftConfig.from_pretrained(path).base_model_name_or_path
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype, device_map="auto")
        model = PeftModel.from_pretrained(base, path)
        if cfg["model"]["peft"]["merge"]: model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map="auto")
    model.eval()
    return tokenizer, model

