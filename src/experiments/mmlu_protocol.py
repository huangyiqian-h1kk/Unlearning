"""MMLU generation protocol fixed after the 0390 answer-format audit."""

from collections import Counter, defaultdict
import re

from .config import read_rows


PROTOCOL = "mmlu-instructed-generate-v1"
INSTRUCTION = (
    "Answer only the final multiple-choice question below. "
    "Respond with exactly one letter: A, B, C, or D.\n\n"
)


def extract_letter(text):
    # This is the parser used in the predeclared audit, not the legacy PMC parser.
    # Unparsed outputs remain incorrect; never drop them from the denominator.
    match = re.fullmatch(
        r"\s*(?:(?:the\s+)?(?:(?:correct|best)\s+)?answer\s*(?:is|:)\s*)?"
        r"(?:\(([ABCD])\)|([ABCD]))[.)]?\s*",
        text,
        flags=re.I,
    )
    return (match[1] or match[2]).upper() if match else None


def evaluate(model, tokenizer, path, options, generate):
    if options.get("mmlu_mode", "instructed_generate") != "instructed_generate":
        raise ValueError("Validation requires the fixed instructed_generate MMLU mode")
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"No MMLU rows: {path}")
    for index, row in enumerate(rows):
        if row["answer"] not in tuple("ABCD"):
            raise ValueError(f"Invalid MMLU answer at source row {index}")
        if not row["prompt"].rstrip().endswith("Answer:"):
            raise ValueError(f"Expected prepared MMLU question at source row {index}")
    batch_size = options["batch_size"]
    max_new = options.get("mmlu_max_new_tokens", 10)
    # Match the audited context budget exactly (4095 prompt tokens for these
    # backbones). Existing assets, demonstrations and question order are reused.
    choice_size = max(
        len(tokenizer.encode(key, add_special_tokens=False)) for key in "ABCD"
    )
    prompt_limit = options["mmlu_max_length"] - choice_size
    if min(batch_size, max_new, prompt_limit) < 1:
        raise ValueError("Invalid MMLU batch size or token budget")
    generation = dict(options, max_length=prompt_limit, max_new_tokens=max_new)
    original_sides = tokenizer.padding_side, tokenizer.truncation_side
    hits, details = defaultdict(list), []
    print(f"[0390] MMLU: {PROTOCOL}; {len(rows)} rows", flush=True)
    try:
        tokenizer.truncation_side = "left"
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            texts = [INSTRUCTION + row["prompt"] for row in chunk]
            lengths = [
                len(tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True, add_generation_prompt=True,
                ))
                for text in texts
            ]
            predictions = generate(model, tokenizer, texts, generation)
            for offset, (row, text, prediction, length) in enumerate(
                zip(chunk, texts, predictions, lengths, strict=True)
            ):
                pred = extract_letter(prediction)
                correct = pred == row["answer"]
                hits[row["subject"]].append(int(correct))
                details.append({
                    "source_row": start + offset, "subject": row["subject"],
                    "prompt": text, "answer": row["answer"],
                    "output": prediction, "predicted": pred,
                    "correct": correct, "parsed": pred is not None,
                    "prompt_tokens": length, "truncated": length > prompt_limit,
                })
            if (start // batch_size + 1) % 16 == 0 or len(details) == len(rows):
                print(f"[0390] MMLU: {len(details)}/{len(rows)}", flush=True)
    finally:
        tokenizer.padding_side, tokenizer.truncation_side = original_sides
    per_subject = {subject: sum(values) / len(values) for subject, values in hits.items()}
    invalid = sum(not item["parsed"] for item in details)
    counts = {subject: len(values) for subject, values in hits.items()}
    diagnostics = {
        "protocol": PROTOCOL, "rows": len(rows), "instruction": INSTRUCTION,
        "max_new_tokens": max_new, "prompt_limit": prompt_limit,
        "truncation_side": "left", "invalid_count": invalid,
        "invalid_fraction": invalid / len(rows),
        "truncated_rows": sum(item["truncated"] for item in details),
        "predicted_counts": dict(Counter(item["predicted"] or "INVALID" for item in details)),
        "per_subject_accuracy": per_subject,
    }
    return sum(per_subject.values()) / len(per_subject), counts, diagnostics, details
