from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(path, local_only=True):
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=local_only, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    path,
    device,
    *,
    train=False,
    lora=None,
    local_only=True,
    dtype="bfloat16",
    attention="sdpa",
):
    from peft import PeftModel, LoraConfig, get_peft_model

    adapter = Path(path) / "adapter_config.json"
    if adapter.exists():
        import json

        base = json.loads(adapter.read_text())["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(
            base,
            local_files_only=local_only,
            torch_dtype=getattr(torch, dtype),
            attn_implementation=attention,
        )
        model = PeftModel.from_pretrained(model, path, is_trainable=train)
        if lora:
            raise ValueError(
                "Start a new unlearning run from a full SFT checkpoint, not a nested adapter"
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=local_only,
            torch_dtype=getattr(torch, dtype),
            attn_implementation=attention,
        )
        if lora:
            model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **lora))
    model.to(device)
    model.config.use_cache = not train
    model.train(train)
    return model


def text_batch(tokenizer, texts, max_length, device):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    pool = encoded.pop("special_tokens_mask").eq(0) & encoded["attention_mask"].bool()
    for token in tokenizer.all_special_ids:
        pool &= encoded["input_ids"].ne(token)
    if not pool.any(dim=1).all():
        raise ValueError("Embedding input has no non-special content tokens")
    return {key: value.to(device) for key, value in encoded.items()}, pool.to(device)


def embed(model, batch, pool_mask):
    # Use the decoder backbone directly: do not allocate full-vocabulary logits for CL.
    from peft import PeftModel

    causal = model.get_base_model() if isinstance(model, PeftModel) else model
    hidden = causal.model(**batch, return_dict=True, use_cache=False).last_hidden_state
    weights = pool_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * weights).sum(1).float() / weights.sum(1).clamp_min(1).float()
    return F.normalize(pooled, dim=-1)


def lm_loss(model, batch):
    return model(**batch, use_cache=False).loss
