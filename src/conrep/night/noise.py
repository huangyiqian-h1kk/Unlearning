"""Token-level retain positives; protected spans come only from training facts.

The entire attribute, entity and copula/value clause are immutable. Tokens
overlapping a protected character (including a subword boundary) are immutable.
Unrecognised grammar, truncation of a fact, or tokenisation disagreement fails
closed. No validation answers and no forced changes are used.
"""

import re

import torch

from conrep.v2.corruption import corrupt, safe_token_ids
from .io import sha
from .positives import protected_ranges

POLICY = "training-fact-offset-protection-v1"
FACT = re.compile(r"\s*(?:The\s+)?(?P<attribute>.+?)\s+of\s+(?P<entity>.+?)\s+"
                  r"(?P<copula>is|are|was|were)\s+(?P<value>.+?)\s*", re.I | re.S)


def fact_spans(row):
    match = FACT.fullmatch(row["text"])
    if match is None:
        raise ValueError(f"Unsupported retain noise fact grammar: {row.get('id')}")
    spans, _ = protected_ranges(row)
    spans += [match.span("attribute"), match.span("entity"),
              (match.start("copula"), len(row["text"].rstrip()))]
    return sorted(set(spans))


def token_record(tokenizer, row, max_length):
    if not tokenizer.is_fast:
        raise ValueError("Protected retain noise requires a fast tokenizer with offsets")
    spans = fact_spans(row)
    # A bounded training-row cache saves repeated offset tokenisation. Its
    # contents are deterministic and consume no random state.
    cache = getattr(tokenizer, "_conrep_noise_cache", None)
    if cache is None:
        cache = tokenizer._conrep_noise_cache = {}
    key = (row["text"], tuple(spans), max_length)
    if key in cache:
        return cache[key]
    encoded = tokenizer(row["text"], truncation=True, max_length=max_length,
        return_offsets_mapping=True, return_special_tokens_mask=True,
        return_token_type_ids=False)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if max((end for start, end in offsets if end > start), default=0) < max(b for a, b in spans):
        raise ValueError(f"Retain noise would use a truncated fact: {row.get('id')}")
    special = set(tokenizer.all_special_ids)
    eligible = [end > start and token not in special and not is_special
                and bool(row["text"][start:end].strip())
                and not any(start < b and end > a for a, b in spans)
                for token, (start, end), is_special in zip(
                    ids, offsets, encoded["special_tokens_mask"])]
    record = {"ids": ids, "eligible": eligible, "spans": spans,
              "offsets": offsets}
    if len(cache) >= 10000:
        cache.clear()
    cache[key] = record
    return record


def eligible_mask(tokenizer, rows, batch, pool_mask, max_length):
    """Align offsets with the actual frozen text_batch result (left/right pad)."""
    ids, attention = batch["input_ids"].cpu(), batch["attention_mask"].cpu().bool()
    eligible = torch.zeros_like(ids, dtype=torch.bool)
    for index, row in enumerate(rows):
        record = token_record(tokenizer, row, max_length)
        if ids[index, attention[index]].tolist() != record["ids"]:
            raise ValueError("Protected offsets disagree with the frozen text_batch helper")
        eligible[index, attention[index]] = torch.tensor(record["eligible"], dtype=torch.bool)
    return (eligible & pool_mask.cpu().bool()).to(batch["input_ids"].device)


def noise_metrics(original, views, eligible, pool_mask):
    """Global counts first, then ratios; unchanged draws remain unchanged."""
    changed = views.ne(original[None])
    if (changed & ~eligible[None]).any():
        raise AssertionError("Retain noise changed a protected token")
    k, n, _ = views.shape
    unique = sum(len({tuple(view[i].tolist()) for view in views.cpu()}) for i in range(n))
    count = torch.tensor([eligible.sum().item(), pool_mask.sum().item(), n,
        changed.sum().item(), k * eligible.sum().item(), k * pool_mask.sum().item(),
        changed.any(-1).sum().item(), k * n, unique],
        device=original.device, dtype=torch.float32)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(count)
    return {
        "specified_noise_eligible_tokens": count[0],
        "specified_noise_eligible_tokens_per_anchor": count[0] / count[2],
        "specified_noise_eligible_fraction": count[0] / count[1].clamp_min(1),
        "specified_noise_replaced_tokens": count[3],
        "specified_noise_eligible_opportunities": count[4],
        "specified_noise_replacement_fraction_eligible": count[3] / count[4].clamp_min(1),
        "specified_noise_replacement_fraction_content": count[3] / count[5].clamp_min(1),
        "specified_changed_fraction": count[6] / count[7],
        "specified_unchanged_fraction": 1 - count[6] / count[7],
        "specified_unique_token_views_per_anchor": count[8] / count[2],
    }


def audit_rows(tokenizer, rows, max_length, embedding_size):
    """CPU-only preflight over every retain training row, before PBS submission."""
    from conrep.v2.model import text_batch
    pool = safe_token_ids(tokenizer, embedding_size)
    records, errors, examples = [], [], []
    generators = {p: torch.Generator().manual_seed(39027) for p in (.1, .2)}
    totals = {str(p): {"changed_tokens": 0, "eligible_opportunities": 0,
                       "changed_views": 0, "views": 0} for p in generators}
    for row in rows:
        try:
            record = token_record(tokenizer, row, max_length)
            batch, content = text_batch(tokenizer, [row["text"]], max_length, "cpu")
            mask = eligible_mask(tokenizer, [row], batch, content, max_length)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        ids = batch["input_ids"]
        records.append({"id": row.get("id"), "tokens": len(record["ids"]),
            "eligible_tokens": int(mask.sum()),
            "protected_token_ids_sha256": sha([x for x, e in zip(record["ids"], record["eligible"]) if not e])})
        example = {"id": row.get("id"), "original": row["text"],
                   "protected_spans": record["spans"],
                   "eligible_offsets": [o for o, e in zip(record["offsets"], record["eligible"]) if e]}
        for probability, generator in generators.items():
            views = corrupt(ids, mask, pool, views=4, probability=probability, generator=generator)
            changed = views.ne(ids[None])
            assert not (changed & ~mask[None]).any()
            total = totals[str(probability)]
            total["changed_tokens"] += int(changed.sum())
            total["eligible_opportunities"] += 4 * int(mask.sum())
            total["changed_views"] += int(changed.any(-1).sum())
            total["views"] += 4
            if len(examples) < 12:
                example[f"p{probability}"] = tokenizer.batch_decode(views[:, 0], skip_special_tokens=False)
        if len(examples) < 12:
            examples.append(example)
    return {"policy": POLICY, "rows": len(rows), "audited_rows": len(records),
        "unsupported_rows": len(errors), "errors": errors[:30],
        "eligible_rows": sum(r["eligible_tokens"] > 0 for r in records),
        "eligible_tokens": sum(r["eligible_tokens"] for r in records),
        "records": records, "examples": examples, "sampled_diagnostics": totals,
        "uses_validation_or_test": False, "forced_mutation": False,
        "replacement_pool": "uniform legal non-special vocabulary, excluding original ID",
        "replacement_pool_size": len(pool), "replacement_pool_sha256": sha(pool.tolist()),
        "qualification": "Preserves protected token IDs; random inserted tokens are not a guarantee of semantic equivalence."}
