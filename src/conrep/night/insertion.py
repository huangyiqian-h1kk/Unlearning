"""Insert vocabulary IDs at audited gaps, preserving every original token.

No decode/re-tokenize round trip: protected subword spans remain contiguous.
All randomness comes from the checkpointed positive CPU generator.
"""

import torch

from .io import sha
from .noise import token_record
from conrep.v2.corruption import safe_token_ids

POLICY = "training-fact-gap-insertion-v1"
MODES = ("fixed1", "fixed2", "binomial2p20")


def gap_record(tokenizer, row, max_length, maximum=2):
    record = token_record(tokenizer, row, max_length)
    # Check the entire original sequence, including any closing special IDs.
    full = tokenizer(row["text"], truncation=False, return_token_type_ids=False)["input_ids"]
    if full != record["ids"] or len(full) + maximum > max_length:
        raise ValueError(f"Insertion would truncate an original token: {row.get('id')}")
    offsets = record["offsets"]
    content = [i for i, (a, b) in enumerate(offsets) if b > a]
    if not content:
        raise ValueError("Insertion requires content offsets")
    blocked = set()
    for start, end in record["spans"]:
        covered = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if not covered:
            raise ValueError("Protected span has no token coverage")
        blocked.update(range(min(covered) + 1, max(covered) + 1))
    # Prefix/suffix mean after BOS / before EOS, never inside special wrappers.
    gaps = [i for i in range(content[0], content[-1] + 2) if i not in blocked]
    if len(gaps) < maximum:
        raise ValueError("Insufficient distinct legal insertion gaps")
    return dict(record, gaps=gaps)


def insertion_views(tokenizer, rows, batch, pool_mask, safe_ids, *, views, mode,
                    max_length, generator):
    if mode not in MODES or type(views) is not int or views < 1:
        raise ValueError("Invalid insertion mode/views")
    maximum = 1 if mode == "fixed1" else 2
    records = [gap_record(tokenizer, row, max_length, maximum) for row in rows]
    active = batch["attention_mask"].cpu().bool()
    original = batch["input_ids"].cpu()
    original_pool = pool_mask.cpu().bool()
    pools = []
    for i, record in enumerate(records):
        if original[i, active[i]].tolist() != record["ids"]:
            raise ValueError("Insertion offsets disagree with the frozen text_batch helper")
        pools.append(original_pool[i, active[i]].tolist())
    if tokenizer.padding_side not in ("left", "right") or tokenizer.pad_token_id is None:
        raise ValueError("Insertion requires an explicit padding side and token")
    vocabulary = safe_ids.cpu()
    if not len(vocabulary):
        raise ValueError("Empty insertion vocabulary")
    outputs, details = [], []
    for _ in range(views):
        sequences, masks, provenance = [], [], []
        for record, content_pool in zip(records, pools):
            count = (int((torch.rand(2, generator=generator) < .2).sum())
                     if mode == "binomial2p20" else maximum)
            chosen = (torch.randperm(len(record["gaps"]), generator=generator)[:count].tolist()
                      if count else [])
            gaps = sorted(record["gaps"][i] for i in chosen)
            noise = vocabulary[torch.randint(len(vocabulary), (count,), generator=generator)].tolist() if count else []
            at = dict(zip(gaps, noise))
            ids, mask, inserted_positions, original_positions = [], [], [], []
            for index in range(len(record["ids"]) + 1):
                if index in at:
                    inserted_positions.append(len(ids))
                    ids.append(at[index])
                    mask.append(True)
                if index < len(record["ids"]):
                    original_positions.append(len(ids))
                    ids.append(record["ids"][index])
                    mask.append(content_pool[index])
            if [ids[i] for i in original_positions] != record["ids"]:
                raise AssertionError("Insertion modified the original token sequence")
            sequences.append(ids)
            masks.append(mask)
            provenance.append({"gaps": gaps, "noise_ids": noise,
                "inserted_positions": inserted_positions, "original_positions": original_positions,
                "ids": ids})
        length = max(map(len, sequences))
        ids = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=original.dtype)
        attention = torch.zeros_like(ids)
        pool = torch.zeros_like(ids, dtype=torch.bool)
        for i, (sequence, mask) in enumerate(zip(sequences, masks)):
            start = length - len(sequence) if tokenizer.padding_side == "left" else 0
            end = start + len(sequence)
            ids[i, start:end] = torch.tensor(sequence)
            attention[i, start:end] = 1
            pool[i, start:end] = torch.tensor(mask)
        device = batch["input_ids"].device
        outputs.append(({"input_ids": ids.to(device), "attention_mask": attention.to(device)}, pool.to(device)))
        details.append(provenance)
    return outputs, insertion_metrics(records, pools, details, batch["input_ids"].device), details


def insertion_metrics(records, pools, details, device):
    k, n = len(details), len(records)
    counts = [len(view["gaps"]) for group in details for view in group]
    unique = sum(len({tuple(group[i]["ids"]) for group in details}) for i in range(n))
    totals = torch.tensor([sum(len(r["gaps"]) for r in records), n, sum(counts), k*n,
        sum(x > 0 for x in counts), unique, k*sum(sum(p) for p in pools),
        counts.count(0), counts.count(1), counts.count(2)], dtype=torch.float32, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals)
    return {"specified_insertion_eligible_gaps": totals[0],
        "specified_insertion_gaps_per_anchor": totals[0] / totals[1],
        "specified_insertion_tokens": totals[2], "specified_insertion_views": totals[3],
        "specified_insertion_tokens_per_view": totals[2] / totals[3],
        "specified_insertion_fraction_original_content": totals[2] / totals[6].clamp_min(1),
        "specified_insertion_fraction_augmented_content": totals[2] / (totals[6]+totals[2]).clamp_min(1),
        "specified_changed_fraction": totals[4] / totals[3],
        "specified_unchanged_fraction": 1 - totals[4] / totals[3],
        "specified_unique_token_views_per_anchor": totals[5] / totals[1],
        **{f"specified_insertion_count_{i}_views": totals[7+i] for i in range(3)}}


def audit_rows(tokenizer, rows, max_length, embedding_size):
    from conrep.v2.model import text_batch
    pool = safe_token_ids(tokenizer, embedding_size)
    generators = {mode: torch.Generator().manual_seed(39028) for mode in MODES}
    records, examples, errors = [], [], []
    histograms = {mode: {str(i): 0 for i in range(3)} for mode in MODES}
    for row in rows:
        try:
            record = gap_record(tokenizer, row, max_length)
            batch, mask = text_batch(tokenizer, [row["text"]], max_length, "cpu")
            example = {"id": row.get("id"), "original": row["text"],
                "protected_spans": record["spans"], "offsets": record["offsets"], "legal_gaps": record["gaps"]}
            for mode in MODES:
                _, _, details = insertion_views(tokenizer, [row], batch, mask, pool,
                    views=4, mode=mode, max_length=max_length, generator=generators[mode])
                for group in details:
                    histograms[mode][str(len(group[0]["gaps"]))] += 1
                if len(examples) < 12:
                    example[mode] = [{**group[0], "decoded": tokenizer.decode(group[0]["ids"], skip_special_tokens=False)} for group in details]
            records.append({"id": row.get("id"), "original_tokens": len(record["ids"]),
                "eligible_gaps": len(record["gaps"]), "legacy_replacement_eligible_tokens": sum(record["eligible"]),
                "original_ids_sha256": sha(record["ids"])})
            if len(examples) < 12:
                examples.append(example)
        except ValueError as exc:
            errors.append(str(exc))
    return {"policy": POLICY, "rows": len(rows), "audited_rows": len(records),
        "unsupported_rows": len(errors), "errors": errors[:30], "records": records,
        "examples": examples, "count_histograms": histograms, "uses_validation_or_test": False,
        "insertion_pool": "uniform legal non-special vocabulary IDs", "pool_size": len(pool),
        "pool_sha256": sha(pool.tolist()), "binomial_count": {"trials": 2, "probability": .2},
        "distinct_gaps": True, "original_tokens_preserved": True, "truncate_original": False,
        "pooling": "all original content and inserted tokens; excludes special and padding",
        "qualification": "Protected spans stay contiguous; semantic equivalence is not guaranteed."}
