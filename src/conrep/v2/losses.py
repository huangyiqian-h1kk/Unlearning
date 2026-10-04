"""Per-positive InfoNCE; all temperatures/margins act on cosine similarities."""

import math

import torch
import torch.nn.functional as F


def gather(value):
    if torch.distributed.is_initialized():
        from torch.distributed.nn.functional import all_gather

        return torch.cat(all_gather(value), dim=0)
    return value


def multi_positive(anchor, targets, positive, allowed, temperature, log_weights=None):
    if temperature <= 0 or not positive.any(1).all():
        raise ValueError(
            "Positive temperature and at least one positive per anchor required"
        )
    logits = (
        F.normalize(anchor.float(), dim=-1)
        @ F.normalize(targets.float(), dim=-1).T
        / temperature
    )
    if log_weights is not None:
        logits = logits + log_weights
    # Each positive has its own denominator; other positives are not negatives.
    negatives = allowed & ~positive
    negative_lse = torch.logsumexp(
        logits.masked_fill(~negatives, -torch.inf), dim=1, keepdim=True
    )
    losses = torch.logaddexp(logits, negative_lse) - logits
    return (losses.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()


def forget_loss(
    forget,
    controls,
    retain,
    *,
    temperature=0.08,
    retain_weight=2.0,
    margin=0.1,
    inter_instance_negatives=True,
    shared_target=False,
):
    # controls [K,F,D]; the anchor is never its own negative.
    n, dim = forget.shape
    k = controls.shape[0]
    if shared_target:
        controls = controls.mean((0, 1), keepdim=True).expand(k, n, dim)
    targets = torch.cat([controls.reshape(-1, dim), forget, retain])
    positive = torch.zeros(n, len(targets), dtype=torch.bool, device=forget.device)
    index = torch.arange(n, device=forget.device)
    if shared_target:
        positive[:, : k * n] = True
    else:
        for view in range(k):
            positive[index, view * n + index] = True
    allowed = torch.ones_like(positive)
    allowed[index, k * n + index] = False
    if not inter_instance_negatives:
        allowed[:, : k * n + n] = positive[:, : k * n + n]
    if retain_weight <= 0:
        raise ValueError("retain_weight must be positive")
    weights = torch.zeros_like(positive, dtype=torch.float32)
    weights[:, k * n + n :] = math.log(retain_weight) + margin / temperature
    return multi_positive(forget, targets, positive, allowed, temperature, weights)


def paired_loss(first, second, other=None, temperature=0.1):
    n = len(first)
    targets = torch.cat([first, second] + ([] if other is None else [other]))
    positive = torch.zeros(n, len(targets), dtype=torch.bool, device=first.device)
    index = torch.arange(n, device=first.device)
    positive[index, n + index] = True
    allowed = torch.ones_like(positive)
    allowed[index, index] = False
    return multi_positive(first, targets, positive, allowed, temperature)
