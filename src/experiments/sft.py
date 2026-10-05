from pathlib import Path

from transformers import Trainer, TrainingArguments, set_seed

from conrep.v2.model import load_model, load_tokenizer
from .config import output_dir, read_rows, write_json
from .data import SFTDataset, pad_examples


def run(cfg, resume=None, retain_only=False):
    import os
    import torch

    seed = cfg["run"]["seed"]
    set_seed(seed)
    root = output_dir(cfg)
    if root.exists() and any(root.iterdir()) and resume is None:
        raise FileExistsError(
            f"Output exists; use --resume or choose another run.output_dir: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    model_cfg, options = cfg["model"], cfg["sft"]
    if retain_only:
        import json

        manifest = json.loads(
            (Path(cfg["data"]["prepared_dir"]) / "manifest.json").read_text()
        )
        if manifest["injection_unresolved_rows"]:
            raise ValueError(
                "Retain-only SFT needs complete identifier membership; unresolved injection rows remain"
            )
    tokenizer = load_tokenizer(model_cfg["name_or_path"], model_cfg["local_only"])
    if not tokenizer.chat_template:
        raise ValueError("SFT requires the backbone's native chat template")
    model = load_model(
        model_cfg["name_or_path"],
        "cpu",
        train=True,
        local_only=model_cfg["local_only"],
        dtype=model_cfg["dtype"],
        attention=model_cfg["attention"],
    )
    name = "injection_retain_only" if retain_only else "injection"
    rows = read_rows(Path(cfg["data"]["prepared_dir"]) / f"{name}.jsonl")
    data = SFTDataset(rows, tokenizer, options["max_length"])
    args = TrainingArguments(
        output_dir=str(root),
        learning_rate=options["learning_rate"],
        num_train_epochs=options["epochs"],
        max_steps=options.get("max_steps", -1),
        per_device_train_batch_size=options["batch_size"],
        gradient_accumulation_steps=options["gradient_accumulation_steps"],
        warmup_ratio=options["warmup_ratio"],
        weight_decay=options["weight_decay"],
        bf16=model_cfg["dtype"] == "bfloat16",
        fp16=False,
        gradient_checkpointing=options["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=options["save_steps"],
        logging_steps=options.get("logging_steps", 5),
        save_total_limit=None,
        report_to=[],
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        deepspeed=options.get("deepspeed"),
        ddp_find_unused_parameters=False,
        use_cpu=not torch.cuda.is_available(),
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data,
        processing_class=tokenizer,
        data_collator=lambda rows: pad_examples(rows, tokenizer.pad_token_id),
    )
    if trainer.is_world_process_zero():
        write_json(root / "resolved_config.json", cfg)
        write_json(
            root / "lineage.json",
            {
                "stage": "retain_only_sft" if retain_only else "injection_sft",
                "base_model": model_cfg["name_or_path"],
                "training_file": name,
                "examples": len(rows),
                "effective_batch": options["batch_size"]
                * options["gradient_accumulation_steps"]
                * int(os.environ.get("WORLD_SIZE", 1)),
            },
        )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(root / "final"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(root / "final")
        write_json(
            root / "TRAINING_COMPLETE.json",
            {
                "global_step": trainer.state.global_step,
                "final": str(root / "final"),
                "checkpoint_selection": "Run validate and select; final is not auto-selected",
            },
        )
