"""Shared execution, separately defined baseline objectives and update rules.

FALCON algorithm functions are used verbatim from the pinned MIT source.
ReLearn CE+retain CE+KL is adapted from ZJUNLP (2023), MIT license in third_party.
"""

from contextlib import nullcontext
import json
import math
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from conrep.v2.model import load_model, load_tokenizer
from conrep.v2.trainer import save_checkpoint
from experiments.config import output_dir, read_rows, write_json
from experiments.data import chat_example, pad_examples, sample_batch
from experiments.runtime import initialize, sync_gradients, local_batch_sizes
from .common import (
    activation,
    masked_cosine,
    masked_kl,
    masked_mse,
    npo,
    sago,
    vendor_module,
)


def losses(method, model, reference, batches, options, control, falcon=None):
    f = batches["forget"]
    if method in ("npo", "sago"):
        forget = npo(model, reference, f, options["beta"])
    elif method == "relearn":
        forget = model(**f, use_cache=False).loss
    else:
        current = activation(model, f, options["layer"])
        mask = f["attention_mask"].bool()
        if method == "rmu":
            forget = masked_mse(current, control.to(current).expand_as(current), mask)
        else:
            with torch.no_grad():
                original = activation(reference, f, options["layer"])
                target = falcon.generate_steering_vector(
                    reference, original[mask].unsqueeze(0)
                )
            # Exclude padding, while calling the upstream contrastive kernel unchanged.
            anchor = current[mask].unsqueeze(0)
            forget = falcon.compute_contrastive_loss(
                anchor,
                target.expand_as(anchor),
                original[mask].unsqueeze(0),
                temperature=options["temperature"],
            )
    retained = 0.0
    for name, weight in (
        ("retain", options["specified_weight"]),
        ("general", options["general_weight"]),
    ):
        batch = batches[name]
        if method in ("npo", "sago"):
            value = model(**batch, use_cache=False).loss
        elif method == "relearn":
            ce, kl = masked_kl(model, reference, batch)
            value = ce + options["kl_weight"] * kl
        else:
            current = activation(model, batch, options["layer"])
            with torch.no_grad():
                original = activation(reference, batch, options["layer"])
            mask = batch["attention_mask"].bool()
            value = (
                masked_mse(current, original, mask)
                if method == "rmu"
                else masked_cosine(current, original, mask)
            )
        retained = retained + weight * value
    return forget, retained


def run(cfg, resume=None):
    method = cfg["baseline"]["method"]
    if method == "lunar":
        from .lunar import run as lunar_run

        return lunar_run(cfg, resume)
    options, train, mc = dict(cfg["baseline"]), cfg["unlearn"], cfg["model"]
    device, rank, world = initialize(cfg["run"]["seed"])
    root = output_dir(cfg)
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"Output exists: {root}")
    if method == "falcon":
        selection = json.loads(Path(options["layer_selection"]).read_text())
        if selection["checkpoint"] != str(Path(train["checkpoint"]).resolve()):
            raise ValueError("FALCON layer selection belongs to a different checkpoint")
        options["layer"] = selection["selected_layer"]
        options["train_layers"] = list(
            range(max(0, options["layer"] - 2), options["layer"] + 1)
        )
    use_lora = method in ("npo", "sago", "relearn")
    tokenizer = load_tokenizer(train["checkpoint"], mc["local_only"])
    model = load_model(
        resume or train["checkpoint"],
        device,
        train=True,
        lora=cfg["lora"] if use_lora and not resume else None,
        local_only=mc["local_only"],
        dtype=mc["dtype"],
        attention=mc["attention"],
    )
    reference = load_model(
        train["checkpoint"],
        device,
        local_only=mc["local_only"],
        dtype=mc["dtype"],
        attention=mc["attention"],
    )
    reference.requires_grad_(False)
    if not use_lora:
        model.requires_grad_(False)
        for layer in options["train_layers"]:
            model.model.layers[layer].mlp.down_proj.weight.requires_grad_(True)
    elif train["gradient_checkpointing"]:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    parameters = [p for p in model.parameters() if p.requires_grad]
    if method == "falcon" and options["optimizer"] == "sophia":
        module = vendor_module("zeta/sophia.py", "conrep_sophia")
        optimizer = module.SophiaG(
            parameters, lr=train["learning_rate"], rho=0.9, weight_decay=1e-3
        )
    else:
        optimizer = torch.optim.AdamW(
            parameters, lr=train["learning_rate"], weight_decay=train["weight_decay"]
        )
    data = {
        name: read_rows(Path(cfg["data"]["prepared_dir"]) / f"{name}.jsonl")
        for name in ("forget", "retain", "general")
    }
    if method == "relearn":
        augmented = read_rows(options["augmented_file"])
        coverage = {row["fact_id"] for row in augmented}
        missing = {row["id"] for row in data["forget"]} - coverage
        if missing:
            raise ValueError(
                f"ReLearn augmentation has no approved target for {len(missing)} forget facts"
            )
        data["forget"] = augmented
    original_forget_count = len(
        read_rows(Path(cfg["data"]["prepared_dir"]) / "forget.jsonl")
    )
    sizes, accumulation = (
        local_batch_sizes(train, world),
        train["gradient_accumulation_steps"],
    )
    steps = train.get("max_steps") or math.ceil(
        train["forget_passes"]
        * original_forget_count
        / (sizes["forget"] * world * accumulation)
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, round(steps * train["warmup_ratio"]), steps
    )
    rng = torch.Generator().manual_seed(cfg["run"]["seed"])
    control = torch.rand(1, 1, model.config.hidden_size, generator=rng)
    control = control / control.norm() * options.get("steering_coefficient", 20.0)
    falcon = (
        vendor_module("FALCON/falcon/algorithms.py", "conrep_falcon")
        if method == "falcon"
        else None
    )
    first_step = 0
    if resume:
        if not (Path(resume) / "COMPLETE.json").exists():
            raise ValueError("Incomplete checkpoint")
        state = torch.load(
            Path(resume) / "training_state.pt", weights_only=False, map_location=device
        )
        if state["world_size"] != world:
            raise ValueError("Resume requires the original world size")
        if state["config"]["run"]["seed"] != cfg["run"]["seed"]:
            raise ValueError("Resume requires unchanged seed")
        for field in ("data", "model", "unlearn", "baseline", "lora"):
            if state["config"][field] != cfg[field]:
                raise ValueError(f"Resume configuration changed: {field}")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        first_step = state["step"]
        saved_rng = torch.load(Path(resume) / f"rng-rank-{rank}.pt", weights_only=False)
        torch.set_rng_state(saved_rng["torch"])
        if saved_rng["cuda"]:
            torch.cuda.set_rng_state_all(saved_rng["cuda"])
    else:
        torch.manual_seed(cfg["run"]["seed"] + rank)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "resolved_config.json", cfg)
        write_json(
            root / "lineage.json",
            {
                "method": method,
                "checkpoint": train["checkpoint"],
                "resolved_method": options,
                "parameterization": "lora"
                if use_lora
                else "selected down_proj weights",
                "steps": steps,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "official_equivalence": "See docs/0390/baselines.md",
            },
        )
    surgery = method in ("sago", "falcon")
    for step in range(first_step, steps):
        optimizer.zero_grad(set_to_none=True)
        forget_grads = [torch.zeros_like(p) for p in parameters] if surgery else None
        retain_grads = [torch.zeros_like(p) for p in parameters] if surgery else None
        totals = [0.0, 0.0]
        for micro in range(accumulation):
            draw = step * accumulation + micro
            groups = {
                name: sample_batch(
                    rows, sizes[name], cfg["run"]["seed"] + i * 10007, draw, rank, world
                )
                for i, (name, rows) in enumerate(data.items())
            }
            batches = {}
            for name, rows in groups.items():
                examples = [
                    chat_example(
                        tokenizer,
                        row
                        if method == "relearn" and name == "forget"
                        else dict(row, kind="document"),
                        train["max_length"],
                    )
                    for row in rows
                ]
                batches[name] = pad_examples(examples, tokenizer.pad_token_id, device)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and mc["dtype"] == "bfloat16"
                else nullcontext()
            )
            with context:
                forget, retained = losses(
                    method, model, reference, batches, options, control, falcon
                )
            if not torch.isfinite(forget + retained):
                raise FloatingPointError(f"Non-finite baseline loss at {step}")
            if surgery:
                fg = torch.autograd.grad(
                    forget / accumulation,
                    parameters,
                    retain_graph=False,
                    allow_unused=True,
                )
                rg = torch.autograd.grad(
                    retained / accumulation, parameters, allow_unused=True
                )
                for destination, gradient in zip(forget_grads, fg):
                    if gradient is not None:
                        destination.add_(gradient.detach())
                for destination, gradient in zip(retain_grads, rg):
                    if gradient is not None:
                        destination.add_(gradient.detach())
            else:
                (
                    (options["forget_weight"] * forget + retained) / accumulation
                ).backward()
            totals[0] += float(forget.detach()) / accumulation
            totals[1] += float(retained.detach()) / accumulation
        if surgery:
            # First accumulate and synchronize each task, then apply nonlinear surgery.
            if world > 1:
                for gradient in forget_grads + retain_grads:
                    torch.distributed.all_reduce(gradient)
                    gradient.div_(world)
            if method == "sago":
                combined = [
                    sago(f, r, options["forget_weight"], 1.0)
                    for f, r in zip(forget_grads, retain_grads)
                ]
            else:
                combined, _ = falcon.resolve_gradient_conflict(
                    forget_grads,
                    retain_grads,
                    tuple(options["conflict_weights"]),
                    tuple(options["align_weights"]),
                )
            for parameter, gradient in zip(parameters, combined):
                parameter.grad = gradient
        else:
            sync_gradients(parameters, world)
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, train["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        if rank == 0:
            record = dict(
                step=step + 1,
                forget_loss=totals[0],
                retain_loss=totals[1],
                grad_norm=float(norm),
            )
            with (root / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if (step + 1) % train["save_steps"] == 0 or step + 1 == steps:
            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                scheduler,
                root,
                step + 1,
                cfg,
                rng,
                rank,
                world,
            )
    if rank == 0:
        write_json(root / "TRAINING_COMPLETE.json", {"method": method, "steps": steps})
