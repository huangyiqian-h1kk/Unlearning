"""Use the released PMC rows without inventing new injection examples."""

from __future__ import annotations

import random
import re
from pathlib import Path

from .config import digest, read_rows, write_json, write_rows


def first_text(row):
    return str(next(iter(row.values()))).strip()


def parse_injection(text):
    # Do not classify arbitrary documents as QA merely because they contain '?'.
    match = re.match(r"^\s*Question:\s*(.*?)\s*Answer:\s*(.+)$", text, re.S | re.I)
    if not match:
        match = re.match(
            r"^((?:What|Who|When|Where|Which|How|Why|Is|Does|Did|Can)\b[^\n]*?\?)\s*(?:Answer:\s*)?(.+)$",
            text,
            re.I,
        )
    if match:
        return {
            "kind": "qa",
            "prompt": match[1].strip(),
            "answer": match[2].strip(),
            "text": text,
        }
    return {"kind": "document", "text": text}


def facts(path, prefix):
    from conrep.corruption import parse_token_swap

    rows = []
    for index, raw in enumerate(read_rows(path)):
        values = [
            str(v).strip() for v in raw.values() if v is not None and str(v).strip()
        ]
        if not values:
            raise ValueError(f"Empty fact at {path}:{index + 1}")
        values = [parse_token_swap(value).model_text for value in values]
        rows.append(
            {
                "id": f"{prefix}:{index}",
                "text": values[0],
                "views": list(dict.fromkeys(values[1:])),
            }
        )
    return rows


def prepare(cfg):
    spec = cfg["data"]
    dest = Path(spec["prepared_dir"])
    if (dest / "manifest.json").exists():
        raise FileExistsError(
            f"Prepared data already exists: {dest}; choose a new prepared_dir"
        )
    forget = facts(spec["forget_csv"], "forget")
    retain = facts(spec["retain_csv"], "retain")
    if {r["text"] for r in forget} & {r["text"] for r in retain}:
        raise ValueError(
            "The same canonical text occurs in forget and specified retain"
        )
    # Split membership comes from the existing benchmark, not a newly drawn split.
    memberships = {}
    for group in ("forget", "retain"):
        for row in read_rows(spec[f"{group}_generation"]):
            identifier = str(row["question value"]).strip()
            previous = memberships.setdefault(identifier, group)
            if previous != group and spec.get("injection_csv"):
                raise ValueError(
                    f"Identifier occurs in both legacy splits: {identifier}"
                )
    patterns = [
        (key, re.compile(r"(?<!\w)" + re.escape(key) + r"(?!\w)"), group)
        for key, group in memberships.items()
    ]
    injection, retain_injection = [], []
    injection_rows = (
        read_rows(spec["injection_csv"]) if spec.get("injection_csv") else []
    )
    for index, row in enumerate(injection_rows):
        text = first_text(row)
        if not text:
            raise ValueError(f"Empty injection row {index}")
        matches = [
            (key, group) for key, pattern, group in patterns if pattern.search(text)
        ]
        sample = dict(
            parse_injection(text),
            id=f"injection:{index}",
            source_row=index,
            identifiers=[m[0] for m in matches],
        )
        injection.append(sample)
        # No uncertain row is silently admitted to the retain-only model.
        if matches and all(group == "retain" for _, group in matches):
            retain_injection.append(sample)
    if injection_rows and not retain_injection:
        raise ValueError(
            "No retain-only injection rows resolved; check identifier formatting"
        )
    wiki = []
    seen = set()
    for row in read_rows(spec["general_file"]):
        text = str(row.get("text", first_text(row))).strip()
        if text and text not in seen:
            seen.add(text)
            wiki.append({"id": f"general:{len(wiki)}", "text": text, "views": []})
    random.Random(cfg["run"]["seed"]).shuffle(wiki)
    wiki = wiki[: spec["general_max_rows"]]
    if len(wiki) < 2:
        raise ValueError(
            "General contrastive training needs at least two distinct texts"
        )
    collections = {
        "forget": forget,
        "retain": retain,
        "general": wiki,
        "injection": injection,
        "injection_retain_only": retain_injection,
    }
    for name, rows in collections.items():
        write_rows(dest / f"{name}.jsonl", rows)
    files = {
        k: {"path": spec[k], "sha256": digest(spec[k])}
        for k in (
            "forget_csv",
            "retain_csv",
            "injection_csv",
            "general_file",
            "forget_generation",
            "retain_generation",
        )
        if spec.get(k)
    }
    write_json(
        dest / "manifest.json",
        {
            "inputs": files,
            "counts": {k: len(v) for k, v in collections.items()},
            "injection_unresolved_rows": sum(not x["identifiers"] for x in injection),
            "retain_only_rule": "All matched legacy patient identifiers are retain; unresolved rows excluded",
            "seed": cfg["run"]["seed"],
        },
    )
    return {k: len(v) for k, v in collections.items()}


def chat_example(tokenizer, row, max_length):
    if row["kind"] == "qa":
        user = [{"role": "user", "content": row["prompt"]}]
        prefix = tokenizer.apply_chat_template(
            user, tokenize=True, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            user + [{"role": "assistant", "content": row["answer"]}],
            tokenize=True,
            add_generation_prompt=False,
        )
        if full[: len(prefix)] != prefix:
            raise ValueError(
                "Native template's assistant prefix does not match; refusing incorrect loss mask"
            )
        labels = [-100] * len(prefix) + full[len(prefix) :]
    else:
        full = tokenizer.encode(row["text"], add_special_tokens=True)
        if tokenizer.eos_token_id is not None and full[-1:] != [tokenizer.eos_token_id]:
            full.append(tokenizer.eos_token_id)
        labels = full.copy()
    full, labels = full[:max_length], labels[:max_length]
    if not any(x != -100 for x in labels[1:]):
        raise ValueError(f"No supervised tokens after truncation: {row.get('id')}")
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


def pad_examples(examples, pad_id, device=None):
    import torch

    size = max(len(x["input_ids"]) for x in examples)
    return {
        key: torch.tensor(
            [x[key] + [fill] * (size - len(x[key])) for x in examples],
            dtype=torch.long,
            device=device,
        )
        for key, fill in (
            ("input_ids", pad_id),
            ("attention_mask", 0),
            ("labels", -100),
        )
    }


class SFTDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.examples = [chat_example(tokenizer, row, max_length) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def sample_batch(rows, size, seed, step, rank, world_size):
    # Every rank reconstructs the same global draw, then takes a disjoint slice.
    total = size * world_size
    if total > len(rows):
        raise ValueError(f"Global batch {total} exceeds dataset size {len(rows)}")
    indices = random.Random(seed + step * 1000003).sample(range(len(rows)), total)
    return [rows[index] for index in indices[rank * size : (rank + 1) * size]]
