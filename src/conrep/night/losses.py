"""The existing per-positive objective with an explicit negative-view budget."""

import math
import torch
from conrep.v2.losses import multi_positive


def forget_loss(forget, controls, retain, *, temperature=0.08,
                retain_weight=2.0, margin=0.1, inter_instance_negatives=True,
                shared_target=False, negative_views=None,
                stop_gradient_controls=False, stop_gradient_retain=False):
    # R/S change only these gradient paths, not logits, masks or loss values.
    if stop_gradient_controls:
        controls = controls.detach()
    if stop_gradient_retain:
        retain = retain.detach()
    n, dim = forget.shape
    k = controls.shape[0]
    budget = k if negative_views is None else int(negative_views)
    if not 1 <= budget <= k:
        raise ValueError("negative_views must be in [1, views]")
    if shared_target and budget != k:
        raise ValueError("A restricted negative budget requires per-instance targets")
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
    # Own extra views stay positive; other instances contribute only budget views.
    if budget < k:
        allowed[:, budget * n : k * n] = positive[:, budget * n : k * n]
    if not inter_instance_negatives:
        allowed[:, :k * n + n] = positive[:, :k * n + n]
    if retain_weight <= 0:
        raise ValueError("retain_weight must be positive")
    weights = torch.zeros_like(positive, dtype=torch.float32)
    weights[:, k * n + n:] = math.log(retain_weight) + margin / temperature
    return multi_positive(forget, targets, positive, allowed, temperature, weights)
