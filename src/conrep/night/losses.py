"""The existing per-positive objective with an explicit negative-view budget."""

import math
import torch
from conrep.v2.losses import multi_positive


def retain_noise_loss(anchor, positives, clean_negative, forget, *, temperature=0.1):
    """No noisy positive is ever another anchor's negative; average over K.

    Each anchor has exactly 2*(R-1)+F negatives: the other clean anchors,
    one independently encoded clean dropout view each, and forget anchors.
    The own clean dropout view is excluded. All branches receive gradients.
    """
    k, n, dim = positives.shape
    if k < 1 or anchor.shape != (n, dim) or clean_negative.shape != anchor.shape:
        raise ValueError("Expected [K,R,D] positives and one [R,D] clean negative bank")
    targets = torch.cat([anchor, clean_negative, positives.reshape(-1, dim), forget])
    positive = torch.zeros(n, len(targets), device=anchor.device, dtype=torch.bool)
    index = torch.arange(n, device=anchor.device)
    for view in range(k):
        positive[index, (2 + view) * n + index] = True
    allowed = torch.ones_like(positive)
    allowed[index, index] = False
    allowed[index, n + index] = False
    allowed[:, 2*n:(2+k)*n] = positive[:, 2*n:(2+k)*n]
    return multi_positive(anchor, targets, positive, allowed, temperature)


def retain_loss(anchor, positives, forget=None, *, temperature=0.1, negative_views=1):
    """Average per-positive InfoNCE with a fixed other-anchor view budget.

    positives is [K, R, D]. All K views of the anchor are positives. Other
    anchors contribute their clean representation and only negative_views
    extra views. K=1/budget=1 has the same target order/masks as paired_loss.
    """
    k, n, dim = positives.shape
    if anchor.shape != (n, dim) or not 1 <= negative_views <= k:
        raise ValueError("Expected [K,R,D] positives and a negative budget in [1,K]")
    targets = torch.cat([anchor, positives.reshape(-1, dim)]
                        + ([] if forget is None else [forget]))
    positive = torch.zeros(n, len(targets), dtype=torch.bool, device=anchor.device)
    index = torch.arange(n, device=anchor.device)
    for view in range(k):
        positive[index, (view + 1) * n + index] = True
    allowed = torch.ones_like(positive)
    allowed[index, index] = False
    allowed[:, (negative_views + 1) * n:(k + 1) * n] = positive[
        :, (negative_views + 1) * n:(k + 1) * n]
    return multi_positive(anchor, targets, positive, allowed, temperature)


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
