"""NPO/RMU/SAGO ports from MIT-licensed OpenUnlearning and SAGO.

Copyright (c) 2025 CMU Locus Lab. See third_party/0390_baselines/SAGO/LICENSE.
ClinicIA adaptations: padding masks, two retain streams, explicit gradient accumulation.
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


def vendor_module(relative, name):
    path = ROOT / "third_party" / "0390_baselines" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sequence_nll(model, batch):
    logits = model(**batch, use_cache=False).logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    return F.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
    ).sum(1)


def npo(model, reference, batch, beta):
    current = sequence_nll(model, batch)
    with torch.no_grad():
        original = sequence_nll(reference, batch)
    return -2.0 / beta * F.logsigmoid(beta * (current - original)).mean()


def sago(forget_gradient, retain_gradient, forget_weight=1.0, retain_weight=1.0):
    # Official 'sago', not the different 'sago_prefer_retain' variant.
    conflict = forget_gradient.sign() * retain_gradient.sign() < 0
    return torch.where(
        conflict, retain_weight * retain_gradient, forget_weight * forget_gradient
    )


def activation(model, batch, layer):
    cache = []
    module = model.model.layers[layer]
    handle = module.register_forward_hook(
        lambda _m, _i, out: cache.append(out[0] if isinstance(out, tuple) else out)
    )
    try:
        model.model(
            **{k: v for k, v in batch.items() if k != "labels"}, use_cache=False
        )
    finally:
        handle.remove()
    return cache[0]


def masked_mse(a, b, mask):
    error = (a.float() - b.float()).square().mean(-1)
    return ((error * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()


def masked_cosine(a, b, mask):
    error = 1 - F.cosine_similarity(a.float(), b.float(), dim=-1)
    return ((error * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()


def masked_kl(model, reference, batch):
    current = model(**batch, use_cache=False)
    with torch.no_grad():
        original = reference(**batch, use_cache=False)
    mask = batch["attention_mask"].bool()
    logits = F.log_softmax(current.logits[mask].float(), -1)
    target = F.log_softmax(original.logits[mask].float(), -1)
    return current.loss, F.kl_div(
        logits, target, reduction="batchmean", log_target=True
    )
