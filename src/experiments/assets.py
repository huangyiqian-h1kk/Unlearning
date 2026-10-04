"""Run on a machine with network access, before offline ABCI jobs."""

from pathlib import Path
import random
import subprocess

from .config import write_rows, write_json


def prepare_assets(root, models=False, model_root=None, mmlu_per_subject=20, seed=42):
    from datasets import load_dataset

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    # Preserve original lines/paragraphs; exclude headings and empty rows.
    rows = [
        {"text": row["text"].strip()}
        for row in wiki
        if len(row["text"].split()) >= 30 and not row["text"].lstrip().startswith("=")
    ]
    random.Random(seed).shuffle(rows)
    write_rows(root / "wiki.jsonl", rows[:50000])
    mmlu = load_dataset("cais/mmlu", "all")
    demonstrations = {}
    for row in mmlu["dev"]:
        demonstrations.setdefault(row["subject"], []).append(row)

    def format_question(row, answered):
        text = (
            row["question"]
            + "\n"
            + "\n".join(
                f"{chr(65 + i)}. {choice}" for i, choice in enumerate(row["choices"])
            )
            + "\nAnswer:"
        )
        return text + (" " + chr(65 + row["answer"]) + "\n\n" if answered else "")

    subjects = {}
    for row in mmlu["test"]:
        subject = row["subject"]
        prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        prompt += "".join(
            format_question(example, True) for example in demonstrations[subject][:5]
        )
        prompt += format_question(row, False)
        subjects.setdefault(subject, []).append(
            {"subject": subject, "prompt": prompt, "answer": chr(65 + row["answer"])}
        )
    full, validation = [], []
    for subject, items in sorted(subjects.items()):
        full.extend(items)
        indices = list(range(len(items)))
        random.Random(seed).shuffle(indices)
        validation.extend(items[index] for index in indices[:mmlu_per_subject])
    write_rows(root / "mmlu_full.jsonl", full)
    write_rows(root / "mmlu_validation.jsonl", validation)
    sts = load_dataset("sentence-transformers/stsb", split="validation")
    write_rows(root / "sts_validation.jsonl", [dict(row) for row in sts])
    write_json(
        root / "assets.json",
        {
            "wiki_source": "wikitext/wikitext-103-raw-v1/train",
            "wiki_rows": min(50000, len(rows)),
            "mmlu_source": "cais/mmlu/all",
            "mmlu_fewshot": 5,
            "mmlu_validation_per_subject": mmlu_per_subject,
            "mmlu_validation_rows": len(validation),
            "mmlu_full_rows": len(full),
            "seed": seed,
            "sts_source": "sentence-transformers/stsb/validation",
            "sts_rows": len(sts),
            "utility_protocol": "subject-macro accuracy; native chat template; letter continuation likelihood",
        },
    )
    vendor = Path("vendor_checkouts/LUNAR")
    if not vendor.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/facebookresearch/LUNAR.git",
                str(vendor),
            ],
            check=True,
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(vendor),
            "checkout",
            "dfa56eb0291a93e967284a9ec2d28d5572d235b1",
        ],
        check=True,
    )
    if models:
        from huggingface_hub import snapshot_download

        if not model_root:
            raise ValueError("--model-root is required with --models")
        for repository in (
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ):
            snapshot_download(
                repository, local_dir=str(Path(model_root) / repository.split("/")[-1])
            )
