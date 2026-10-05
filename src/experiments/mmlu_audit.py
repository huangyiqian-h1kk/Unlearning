"""Diagnose answer-format effects without changing the validation protocol."""

from collections import Counter, defaultdict
from pathlib import Path
import math

from .config import digest, read_rows, write_json
from . import mmlu_protocol
from .mmlu_protocol import INSTRUCTION, extract_letter


MODES = ("legacy_letter", "same_prompt_generate", "instructed_generate")


def sample_rows(rows, per_subject):
    """Use the same deterministic, subject-balanced slice for every mode/model."""
    if per_subject < 1:
        raise ValueError("audit.per_subject must be positive")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if row["answer"] not in "ABCD" or len(row["answer"]) != 1:
            raise ValueError(f"Invalid MMLU answer at row {index}")
        if not row["prompt"].rstrip().endswith("Answer:"):
            raise ValueError(f"Expected prepared five-shot MMLU prompt at row {index}")
        groups[row["subject"]].append((index, row))
    if not groups or any(len(items) < per_subject for items in groups.values()):
        raise ValueError("Not enough rows to sample every subject equally")
    return [item for subject in sorted(groups) for item in groups[subject][:per_subject]]


def summarize(details):
    result = {}
    for mode in MODES:
        subjects = defaultdict(list)
        predictions = []
        for item in details:
            pred = item[mode]["predicted"]
            subjects[item["subject"]].append(int(pred == item["answer"]))
            predictions.append(pred)
        per_subject = {key: sum(values) / len(values) for key, values in subjects.items()}
        result[mode] = {
            "subject_macro_accuracy": sum(per_subject.values()) / len(per_subject),
            "invalid_fraction": sum(x is None for x in predictions) / len(predictions),
            "predicted_counts": dict(Counter(x or "INVALID" for x in predictions)),
            "per_subject": per_subject,
        }
    return result


def run(cfg, checkpoint, output):
    import torch
    from conrep.v2.model import load_model, load_tokenizer
    from .validation import continuation_score, generate
    from . import validation

    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Audit output exists; choose a new --output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    options = cfg.get("audit", {})
    source = cfg["evaluation"]["mmlu_file"]
    selected = sample_rows(read_rows(source), int(options.get("per_subject", 5)))
    max_length = cfg["evaluation"]["mmlu_max_length"]
    max_new = int(options.get("max_new_tokens", 10))
    batch_size = cfg["evaluation"]["batch_size"]
    if max_new < 1 or batch_size < 1:
        raise ValueError("Audit generation length and batch size must be positive")
    mc = cfg["model"]
    tokenizer = load_tokenizer(checkpoint, mc["local_only"])
    model = load_model(
        checkpoint, "cuda" if torch.cuda.is_available() else "cpu",
        local_only=mc["local_only"], dtype=mc["dtype"], attention=mc["attention"],
    )
    model.eval()
    choice_tokens = {k: tokenizer.encode(k, add_special_tokens=False) for k in "ABCD"}
    longest_choice = max(len(ids) for ids in choice_tokens.values())
    # Match the existing likelihood scorer's left cropping, preserving the final
    # question. Record truncation so it cannot masquerade as a format effect.
    tokenizer.truncation_side = "left"
    generation_limit = max_length - longest_choice
    if generation_limit < 1:
        raise ValueError("MMLU context limit leaves no room for the prompt")
    generation = dict(cfg["evaluation"], max_length=generation_limit, max_new_tokens=max_new)
    details = []
    counts = Counter(row["subject"] for _, row in selected)
    print(f"[0390] MMLU audit: {len(selected)} rows, {len(counts)} subjects", flush=True)
    # JSONL is written incrementally; report.json is the completion marker.
    import json

    with (root / "predictions.jsonl").open("w") as stream:
        for start in range(0, len(selected), batch_size):
            chunk = selected[start:start + batch_size]
            texts = [row["prompt"] for _, row in chunk]
            instructed = [INSTRUCTION + text for text in texts]
            lengths = []
            for text in texts + instructed:
                tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True, add_generation_prompt=True,
                )
                lengths.append(len(tokens))
            original_outputs = generate(model, tokenizer, texts, generation)
            instructed_outputs = generate(model, tokenizer, instructed, generation)
            for offset, ((index, row), original, explicit) in enumerate(zip(chunk, original_outputs, instructed_outputs)):
                # Reuse the exact existing scorer; this is not a replacement MMLU metric.
                scores = {key: continuation_score(model, tokenizer, row["prompt"], key, max_length) for key in "ABCD"}
                if not all(math.isfinite(value) for value in scores.values()):
                    raise ValueError(f"Non-finite likelihood at source row {index}")
                item = {
                    "source_row": index, "subject": row["subject"],
                    "answer": row["answer"], "prompt": row["prompt"],
                    "prompt_tokens": {"same_prompt": lengths[offset], "instructed": lengths[offset + len(chunk)]},
                    "legacy_letter": {
                        "predicted": max(scores, key=scores.get), "log_probabilities": scores,
                        "choice_probability_mass": sum(math.exp(x) for x in scores.values()),
                    },
                    "same_prompt_generate": {"predicted": extract_letter(original), "output": original},
                    "instructed_generate": {"predicted": extract_letter(explicit), "output": explicit},
                }
                details.append(item)
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            print(f"[0390] MMLU audit: {len(details)}/{len(selected)}", flush=True)
    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "usage": "format diagnostic only; do not use for checkpoint selection",
        "source": str(source), "source_sha256": digest(source),
        "code_sha256": {"audit": digest(__file__), "validation": digest(validation.__file__), "mmlu_protocol": digest(mmlu_protocol.__file__)},
        "subjects": dict(counts), "rows": len(selected),
        "max_length": max_length, "max_new_tokens": max_new,
        "generation_prompt_limit": generation_limit, "truncation_side": "left",
        "truncated_rows": {name: sum(x["prompt_tokens"][name] > generation_limit for x in details) for name in ("same_prompt", "instructed")},
        "instruction": INSTRUCTION,
        "choice_token_ids": choice_tokens,
        "mean_choice_probability_mass": sum(x["legacy_letter"]["choice_probability_mass"] for x in details) / len(details),
        "metrics": summarize(details),
    }
    write_json(root / "report.json", report)
    return {
        "report": str(root / "report.json"), "rows": len(details),
        "metrics": {key: {k: v for k, v in values.items() if k != "per_subject"} for key, values in report["metrics"].items()},
        "mean_choice_probability_mass": report["mean_choice_probability_mass"],
        "truncated_rows": report["truncated_rows"],
    }
