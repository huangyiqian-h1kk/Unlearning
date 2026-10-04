import torch


def safe_token_ids(tokenizer, embedding_size):
    excluded = set(tokenizer.all_special_ids)
    ids = sorted(
        {
            i
            for i in tokenizer.get_vocab().values()
            if 0 <= i < embedding_size and i not in excluded
        }
    )
    if len(ids) < 2:
        raise ValueError("Need at least two legal non-special token IDs")
    return torch.tensor(ids, dtype=torch.long)


def corrupt(
    input_ids, eligible_mask, safe_ids, *, views=4, probability=0.7, generator=None
):
    """Return [K,B,L] IDs; independently sample every pivot/view/position.

    eligible_mask excludes padding, special tokens and external instructions.
    Work on CPU so a seeded CPU generator reproduces masks on every backbone.
    """
    if not 0 <= probability <= 1 or views < 1:
        raise ValueError("Invalid corruption rate/views")
    original = input_ids.cpu().unsqueeze(0).expand(views, -1, -1)
    eligible = eligible_mask.cpu().bool().unsqueeze(0).expand_as(original)
    selected = (
        torch.rand(original.shape, generator=generator) < probability
    ) & eligible
    pool = safe_ids.cpu()
    # Uniform conditional sampling excluding the original token, without rejection loops.
    position = torch.searchsorted(pool, original.contiguous())
    in_pool = (position < len(pool)) & (
        pool[position.clamp_max(len(pool) - 1)] == original
    )
    count = len(pool) - in_pool.long()
    draw = (torch.rand(original.shape, generator=generator) * count).long()
    draw += (in_pool & (draw >= position)).long()
    replacement = pool[draw]
    return torch.where(selected, replacement, original).to(input_ids.device)
