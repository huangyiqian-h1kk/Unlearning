from contextlib import nullcontext
from pathlib import Path
import json
import math
import random

import torch
from transformers import get_linear_schedule_with_warmup

from experiments.config import output_dir, read_rows, write_json
from experiments.data import chat_example, pad_examples, sample_batch
from experiments.runtime import initialize, sync_gradients, barrier, local_batch_sizes
from .corruption import corrupt, safe_token_ids
from .losses import forget_loss, paired_loss, gather
from .model import embed, load_model, load_tokenizer, text_batch


def losses(model, tokenizer, groups, cfg, generator):
    device = next(model.parameters()).device
    options = cfg["conrep"]
    encoded = {}
    for name, rows in groups.items():
        encoded[name] = text_batch(
            tokenizer, [row["text"] for row in rows], options["max_length"], device
        )
    f_batch, f_mask = encoded["forget"]
    safe = safe_token_ids(tokenizer, model.get_input_embeddings().num_embeddings)
    controls = corrupt(
        f_batch["input_ids"],
        f_mask,
        safe,
        views=options["views"],
        probability=options["corruption_rate"],
        generator=generator,
    )
    f = gather(embed(model, f_batch, f_mask))
    r = gather(embed(model, *encoded["retain"]))
    g = gather(embed(model, *encoded["general"]))
    c = torch.stack(
        [gather(embed(model, dict(f_batch, input_ids=ids), f_mask)) for ids in controls]
    )
    loss_f = forget_loss(
        f,
        c,
        torch.cat([r, g]),
        temperature=options["temperature_forget"],
        retain_weight=options["retain_negative_weight"],
        margin=options["retain_margin"],
        inter_instance_negatives=options["inter_instance_negatives"],
        shared_target=options["shared_target"],
    )
    metrics = {"forget_cl": loss_f}
    total = loss_f
    if options["specified_cl_weight"]:
        texts = []
        for row in groups["retain"]:
            if not row["views"]:
                raise ValueError(
                    "Specified retain needs an existing paraphrase column; no synthetic positives are invented"
                )
            texts.append(
                row["views"][
                    int(torch.randint(len(row["views"]), (), generator=generator))
                ]
            )
        r_second = gather(
            embed(model, *text_batch(tokenizer, texts, options["max_length"], device))
        )
        value = paired_loss(r, r_second, f, options["temperature_retain"])
        metrics["specified_cl"] = value
        total = total + options["specified_cl_weight"] * value
    if options["general_cl_weight"]:
        if cfg["lora"]["lora_dropout"] <= 0:
            raise ValueError("Two general views need nonzero LoRA dropout")
        g_second = gather(embed(model, *encoded["general"]))
        value = paired_loss(g, g_second, temperature=options["temperature_general"])
        metrics["general_cl"] = value
        total = total + options["general_cl_weight"] * value
    # No language-model loss is applied to forget texts or corrupted targets.
    for name, key in (
        ("retain", "specified_lm_weight"),
        ("general", "general_lm_weight"),
    ):
        if options[key]:
            examples = [
                chat_example(
                    tokenizer, dict(row, kind="document"), options["max_length"]
                )
                for row in groups[name]
            ]
            batch = pad_examples(examples, tokenizer.pad_token_id, device)
            value = model(**batch, use_cache=False).loss
            metrics[name + "_lm"] = value
            total = total + options[key] * value
    return total, metrics


def save_checkpoint(
    model, tokenizer, optimizer, scheduler, root, step, cfg, generator, rank, world
):
    target = root / f"checkpoint-{step}"
    if rank == 0:
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target)
        tokenizer.save_pretrained(target)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "world_size": world,
                "config": cfg,
            },
            target / "training_state.pt",
        )
    barrier()
    torch.save(
        {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "corruption": generator.get_state(),
        },
        target / f"rng-rank-{rank}.pt",
    )
    barrier()
    if rank == 0:
        write_json(
            target / "COMPLETE.json",
            {
                "step": step,
                "world_size": world,
                "kind": "adapter"
                if (target / "adapter_config.json").exists()
                else "full_model",
            },
        )


def run(cfg, resume=None):
    device, rank, world = initialize(cfg["run"]["seed"])
    root = output_dir(cfg)
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(
            f"Output exists: {root}; choose a new output or pass --resume"
        )
    mc, options = cfg["model"], cfg["unlearn"]
    base = options["checkpoint"]
    tokenizer = load_tokenizer(base, mc["local_only"])
    model = load_model(
        resume or base,
        device,
        train=True,
        lora=None if resume else cfg["lora"],
        local_only=mc["local_only"],
        dtype=mc["dtype"],
        attention=mc["attention"],
    )
    if options["gradient_checkpointing"]:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=options["learning_rate"], weight_decay=options["weight_decay"]
    )
    data = {
        name: read_rows(Path(cfg["data"]["prepared_dir"]) / f"{name}.jsonl")
        for name in ("forget", "retain", "general")
    }
    sizes = local_batch_sizes(options, world)
    accumulation = options["gradient_accumulation_steps"]
    steps = options.get("max_steps") or math.ceil(
        options["forget_passes"]
        * len(data["forget"])
        / (sizes["forget"] * world * accumulation)
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, round(steps * options["warmup_ratio"]), steps
    )
    generator = torch.Generator().manual_seed(cfg["run"]["seed"] + rank)
    first_step = 0
    if resume:
        if not (Path(resume) / "COMPLETE.json").exists():
            raise ValueError("Cannot resume an incomplete checkpoint")
        state = torch.load(
            Path(resume) / "training_state.pt", map_location=device, weights_only=False
        )
        if state["world_size"] != world:
            raise ValueError("Exact resume requires unchanged world size")
        saved_cfg = state["config"]
        if saved_cfg["run"]["seed"] != cfg["run"]["seed"]:
            raise ValueError("Exact resume requires unchanged seed")
        for field in ("data", "model", "unlearn", "conrep", "lora"):
            if saved_cfg[field] != cfg[field]:
                raise ValueError(f"Resume configuration changed: {field}")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        first_step = state["step"]
        rng = torch.load(Path(resume) / f"rng-rank-{rank}.pt", weights_only=False)
        torch.set_rng_state(rng["torch"])
        if rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        random.setstate(rng["python"])
        generator.set_state(rng["corruption"])
    else:
        # Initialization is identical on all ranks; stochastic views differ thereafter.
        torch.manual_seed(cfg["run"]["seed"] + rank)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "resolved_config.json", cfg)
        write_json(
            root / "lineage.json",
            {
                "method": "conrep",
                "checkpoint": base,
                "steps": steps,
                "global_batch_sizes": {k: v * world for k, v in sizes.items()},
                "gradient_accumulation_steps": accumulation,
                "trainable_parameters": sum(p.numel() for p in parameters),
            },
        )
    for step in range(first_step, steps):
        optimizer.zero_grad(set_to_none=True)
        log = {}
        for micro in range(accumulation):
            draw = step * accumulation + micro
            groups = {
                name: sample_batch(
                    rows,
                    sizes[name],
                    cfg["run"]["seed"] + offset * 10007,
                    draw,
                    rank,
                    world,
                )
                for offset, (name, rows) in enumerate(data.items())
            }
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and mc["dtype"] == "bfloat16"
                else nullcontext()
            )
            with context:
                loss, components = losses(model, tokenizer, groups, cfg, generator)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            (loss / accumulation).backward()
            for key, value in dict(components, total=loss).items():
                log[key] = (
                    log.get(key, 0.0) + value.detach().float().item() / accumulation
                )
        sync_gradients(parameters, world)
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, options["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        if rank == 0:
            record = dict(
                step=step + 1,
                **log,
                grad_norm=float(norm),
                learning_rate=scheduler.get_last_lr()[0],
            )
            with (root / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if (step + 1) % options["save_steps"] == 0 or step + 1 == steps:
            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                scheduler,
                root,
                step + 1,
                cfg,
                generator,
                rank,
                world,
            )
    if rank == 0:
        write_json(
            root / "TRAINING_COMPLETE.json",
            {"steps": steps, "checkpoint_selection": "Pending validate/select"},
        )
