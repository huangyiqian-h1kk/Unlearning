"""125-step ConRep experiments using the server's model/data helpers.

Checkpoints publish atomically after every rank has saved its RNG state.
Stopping is coordinated at optimizer boundaries and does not change the LR
schedule. A pause returns 75; only reaching max_steps publishes completion.
"""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from experiments.config import read_rows
from experiments.data import chat_example, pad_examples, sample_batch
from experiments.runtime import initialize, sync_gradients, barrier, local_batch_sizes
from conrep.v2.model import embed, load_model, load_tokenizer, text_batch
from conrep.v2.corruption import corrupt, safe_token_ids
from conrep.v2.losses import gather, paired_loss
from .losses import forget_loss
from .positives import make_positive
from .io import complete_checkpoint, read, write, training_identity


def losses(model, tokenizer, groups, cfg, corruption_rng, positive_rng):
    device = next(model.parameters()).device
    options = cfg["conrep"]
    encoded = {name: text_batch(tokenizer, [row["text"] for row in rows],
                                options["max_length"], device)
               for name, rows in groups.items()}
    f_batch, f_mask = encoded["forget"]
    controls = corrupt(f_batch["input_ids"], f_mask,
                       safe_token_ids(tokenizer, model.get_input_embeddings().num_embeddings),
                       views=options["views"], probability=options["corruption_rate"],
                       generator=corruption_rng)
    f = gather(embed(model, f_batch, f_mask))
    r = gather(embed(model, *encoded["retain"]))
    g = gather(embed(model, *encoded["general"]))
    c = torch.stack([gather(embed(model, dict(f_batch, input_ids=ids), f_mask))
                     for ids in controls])
    loss_f = forget_loss(f, c, torch.cat([r, g]),
                        temperature=options["temperature_forget"],
                        retain_weight=options["retain_negative_weight"],
                        margin=options["retain_margin"],
                        inter_instance_negatives=options["inter_instance_negatives"],
                        shared_target=options["shared_target"],
                        negative_views=options.get("negative_views"))
    total = options.get("forget_cl_weight", 1.0) * loss_f
    metrics = {"forget_cl": loss_f, "forget_positive_cosine": (f[None] * c).sum(-1).mean().detach()}
    if options["specified_cl_weight"]:
        texts, changed = [], 0
        mode = options.get("specified_positive", "paraphrase")
        for row in groups["retain"]:
            if mode == "dropout":
                text = row["text"]
            elif mode in {"paraphrase", "views"}:
                if not row.get("views"):
                    raise ValueError("A paraphrase positive was requested but row.views is empty")
                text = row["views"][int(torch.randint(len(row["views"]), (), generator=corruption_rng))]
            else:
                raise ValueError(f"Unknown specified_positive: {mode}")
            if options.get("protected_positive", False):
                augmented, did_change = make_positive(
                    row, protected=True,
                    probability=options.get("protected_positive_probability", 0.5),
                    generator=positive_rng)
                if did_change:
                    text = augmented
                    changed += 1
            texts.append(text)
        second = gather(embed(model, *text_batch(tokenizer, texts, options["max_length"], device)))
        value = paired_loss(r, second, f, options["temperature_retain"])
        metrics["specified_cl"] = value
        metrics["specified_positive_cosine"] = (r * second).sum(-1).mean().detach()
        count = torch.tensor([changed, len(texts)], device=device, dtype=torch.float32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(count)
        metrics["specified_changed_fraction"] = count[0] / count[1]
        total = total + options["specified_cl_weight"] * value
    if options["general_cl_weight"]:
        if cfg["lora"]["lora_dropout"] <= 0:
            raise ValueError("General positives require nonzero dropout")
        second = gather(embed(model, *encoded["general"]))
        value = paired_loss(g, second, temperature=options["temperature_general"])
        metrics["general_cl"] = value
        total = total + options["general_cl_weight"] * value
    for name, key in (("retain", "specified_lm_weight"), ("general", "general_lm_weight")):
        if options[key]:
            batch = pad_examples([chat_example(tokenizer, dict(row, kind="document"),
                                                options["max_length"])
                                  for row in groups[name]], tokenizer.pad_token_id, device)
            value = model(**batch, use_cache=False).loss
            metrics[name + "_lm"] = value
            total = total + options[key] * value
    return total, metrics


def save_checkpoint(model, tokenizer, optimizer, scheduler, root, step, cfg,
                    generators, rank, world):
    identity = training_identity(cfg)
    target = root / f"checkpoint-{step}"
    # A completed checkpoint is immutable, including during a repeated resume.
    if complete_checkpoint(target, identity=identity, world=world):
        barrier()
        return
    temp = root / f".checkpoint-{step}.incomplete"
    if rank == 0:
        temp.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(temp)
        tokenizer.save_pretrained(temp)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "step": step, "world_size": world, "identity": identity,
                    "config": cfg}, temp / "training_state.pt")
    barrier()
    torch.save({"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python": random.getstate(), "numpy": np.random.get_state(),
                "generators": {key: gen.get_state() for key, gen in generators.items()}},
               temp / f"rng-rank-{rank}.pt")
    barrier()
    if rank == 0:
        files = {p.name: p.stat().st_size for p in temp.iterdir()
                 if p.is_file() and p.name != "COMPLETE.json"}
        write(temp / "COMPLETE.json", {"schema": "0390-conrep-night-v1", "step": step,
              "world_size": world, "identity": identity, "kind": "adapter", "files": files})
        if target.exists():
            target.rename(root / f".invalid-checkpoint-{step}-{time.time_ns()}")
        temp.rename(target)
    barrier()


def run(cfg, resume=None, *, deadline=None, stop_file=None, stop_after_step=None):
    device, rank, world = initialize(cfg["run"]["seed"])
    root = Path(cfg["run"]["output_dir"]).resolve()
    identity = training_identity(cfg)
    if (root / "identity.json").exists() and read(root / "identity.json")["identity"] != identity:
        raise ValueError("Existing training directory belongs to a different configuration")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        (root / "PAUSED.json").unlink(missing_ok=True)
    barrier()
    requested_stop = [False]
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda *_: requested_stop.__setitem__(0, True))
    try:
        mc, options = cfg["model"], cfg["unlearn"]
        steps = int(options["max_steps"])
        tokenizer = load_tokenizer(options["checkpoint"], mc["local_only"])
        model = load_model(resume or options["checkpoint"], device, train=True,
                           lora=None if resume else cfg["lora"], local_only=mc["local_only"],
                           dtype=mc["dtype"], attention=mc["attention"])
        if options["gradient_checkpointing"]:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=options["learning_rate"],
                                      weight_decay=options["weight_decay"])
        scheduler = get_linear_schedule_with_warmup(optimizer, round(steps * options["warmup_ratio"]), steps)
        data = {name: read_rows(Path(cfg["data"]["prepared_dir"]) / f"{name}.jsonl")
                for name in ("forget", "retain", "general")}
        sizes = local_batch_sizes(options, world)
        generators = {"corruption": torch.Generator().manual_seed(cfg["run"]["seed"] + rank),
                      "positive": torch.Generator().manual_seed(cfg["run"]["seed"] + 10000019 + rank)}
        first_step = 0
        if resume:
            if not complete_checkpoint(resume, identity=identity, world=world):
                raise ValueError("Cannot resume incomplete, incompatible, or wrong-world-size checkpoint")
            # CPU mapping is essential: torch RNG state must stay a CPU ByteTensor.
            state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
            if state["identity"] != identity:
                raise ValueError("Training identity changed")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            first_step = int(state["step"])
            rng = torch.load(Path(resume) / f"rng-rank-{rank}.pt", map_location="cpu", weights_only=False)
            torch.set_rng_state(rng["torch"])
            if rng["cuda"]:
                torch.cuda.set_rng_state_all(rng["cuda"])
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            for key, gen in generators.items():
                gen.set_state(rng["generators"][key])
        else:
            torch.manual_seed(cfg["run"]["seed"] + rank)
            np.random.seed(cfg["run"]["seed"] + rank)
        if rank == 0:
            root.mkdir(parents=True, exist_ok=True)
            write(root / "identity.json", {"identity": identity})
            write(root / "resolved_config.json", cfg)
            write(root / "lineage.json", {"checkpoint": options["checkpoint"], "world_size": world,
                  "steps": steps, "trainable_parameters": sum(p.numel() for p in parameters),
                  "identity": identity})
        barrier()
        def should_stop():
            stop = requested_stop[0] or (deadline is not None and time.time() >= deadline)
            stop |= bool(stop_file and Path(stop_file).exists())
            flag = torch.tensor(int(stop), device=device)
            if world > 1:
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            return bool(flag.item())
        if first_step < steps and should_stop():
            if rank == 0:
                write(root / "PAUSED.json", {"identity": identity, "step": first_step,
                                             "created_at": time.time()})
            barrier()
            return 75
        for step in range(first_step, steps):
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            logged = {}
            accumulation = options["gradient_accumulation_steps"]
            for micro in range(accumulation):
                draw = step * accumulation + micro
                groups = {name: sample_batch(rows, sizes[name], cfg["run"]["seed"] + offset * 10007,
                                             draw, rank, world)
                          for offset, (name, rows) in enumerate(data.items())}
                context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and mc["dtype"] == "bfloat16" else nullcontext()
                with context:
                    loss, components = losses(model, tokenizer, groups, cfg,
                                               generators["corruption"], generators["positive"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step + 1}")
                (loss / accumulation).backward()
                for key, value in dict(components, total=loss).items():
                    logged[key] = logged.get(key, 0.0) + value.detach().float().item() / accumulation
            sync_gradients(parameters, world)
            norm = torch.nn.utils.clip_grad_norm_(parameters, options["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if rank == 0:
                record = dict(step=step + 1, **logged, grad_norm=float(norm),
                              learning_rate=scheduler.get_last_lr()[0], elapsed_seconds=time.monotonic() - begin,
                              job_id=os.environ.get("PBS_JOBID"), resume_from=first_step)
                with (root / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record), flush=True)
            stop = should_stop() or (stop_after_step is not None and step + 1 >= stop_after_step)
            if (step + 1) % options["save_steps"] == 0 or step + 1 == steps or stop:
                save_checkpoint(model, tokenizer, optimizer, scheduler, root, step + 1, cfg,
                                generators, rank, world)
            if stop and step + 1 < steps:
                if rank == 0:
                    write(root / "PAUSED.json", {"identity": identity, "step": step + 1,
                                                 "created_at": time.time()})
                barrier()
                return 75
        if rank == 0:
            write(root / "TRAINING_COMPLETE.json", {"steps": steps, "identity": identity,
                  "checkpoint": str(root / f"checkpoint-{steps}"), "selection": "disabled"})
        barrier()
        return 0
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
