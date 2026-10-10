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
from .losses import forget_loss, retain_loss, retain_noise_loss
from .positives import make_positive, fact_positive_candidates
from .io import complete_checkpoint, read, write, training_identity


def noisy_retain(model, tokenizer, rows, encoded, anchor, forget, options, generator):
    from .noise import POLICY, eligible_mask, noise_metrics
    from .insertion import POLICY as INSERTION_POLICY, insertion_views
    k = options.get("specified_views", 1)
    insertion = options.get("specified_noise_kind", "replacement") == "insertion"
    if (type(k) is not int or k < 1 or options.get("specified_negative_views", 1) != 1
            or options.get("specified_positive") != "dropout"
            or options.get("protected_positive", False)
            or options.get("specified_noise_policy") != (INSERTION_POLICY if insertion else POLICY)
            or options.get("specified_negative_source") != "clean_dropout"):
        raise ValueError("Retain noise requires audited dropout positives and a separate clean negative bank")
    batch, pool_mask = encoded
    vocabulary = safe_token_ids(tokenizer, model.get_input_embeddings().num_embeddings)
    if insertion:
        if options.get("specified_noise_probability", 0):
            raise ValueError("Insertion cannot also enable replacement")
        encoded_views, metrics, _ = insertion_views(tokenizer, rows, batch, pool_mask, vocabulary,
            views=k, mode=options["specified_insertion_mode"], max_length=options["max_length"], generator=generator)
    else:
        eligible = eligible_mask(tokenizer, rows, batch, pool_mask, options["max_length"])
        ids = corrupt(batch["input_ids"], eligible, vocabulary,
            views=k, probability=options["specified_noise_probability"], generator=generator)
        encoded_views = [(dict(batch, input_ids=view), pool_mask) for view in ids]
    clean_negative = gather(embed(model, batch, pool_mask))
    positives = torch.stack([gather(embed(model, view, mask)) for view, mask in encoded_views])
    value = retain_noise_loss(anchor, positives, clean_negative, forget,
                              temperature=options["temperature_retain"])
    if not insertion:
        metrics = noise_metrics(batch["input_ids"], ids, eligible, pool_mask)
    metrics.update(specified_cl=value,
        specified_positive_cosine=(anchor[None] * positives).sum(-1).mean().detach(),
        specified_views=value.detach().new_tensor(k),
        specified_negatives_per_anchor=value.detach().new_tensor(2 * (len(anchor) - 1) + len(forget)))
    if k > 1:
        p = positives.detach().float()
        metrics["specified_view_pair_cosine"] = ((p.sum(0).square().sum(-1)
            - p.square().sum((0, 2))) / (k * (k - 1))).mean()
    return value, metrics


def losses(model, tokenizer, groups, cfg, corruption_rng, positive_rng):
    device = next(model.parameters()).device
    options = cfg["conrep"]
    noise_probability = options.get("specified_noise_probability", 0.0)
    noise_kind = options.get("specified_noise_kind", "replacement")
    if noise_kind not in ("replacement", "insertion"):
        raise ValueError("Unknown specified_noise_kind")
    if not 0 <= noise_probability <= 1:
        raise ValueError("specified_noise_probability must be in [0,1]")
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
                        negative_views=options.get("negative_views"),
                        stop_gradient_controls=options.get("stop_gradient_controls", False),
                        stop_gradient_retain=options.get("stop_gradient_retain", False))
    total = options.get("forget_cl_weight", 1.0) * loss_f
    metrics = {"forget_cl": loss_f, "forget_positive_cosine": (f[None] * c).sum(-1).mean().detach()}
    if options["specified_cl_weight"] and (noise_probability > 0 or noise_kind == "insertion"):
        value, noise_metrics = noisy_retain(model, tokenizer, groups["retain"],
            encoded["retain"], r, f, options, positive_rng)
        metrics.update(noise_metrics)
        total = total + options["specified_cl_weight"] * value
    elif options["specified_cl_weight"]:
        count_views = options.get("specified_views", 1)
        negative_views = options.get("specified_negative_views", 1)
        if (type(count_views) is not int or type(negative_views) is not int
                or not 1 <= negative_views <= count_views):
            raise ValueError("specified_views must be positive with negative budget in [1,views]")
        mode = options.get("specified_positive", "paraphrase")
        if count_views > 1 and mode == "dropout" and cfg["lora"]["lora_dropout"] <= 0:
            raise ValueError("Multiple dropout positives require nonzero LoRA dropout")
        positives, all_texts, changed = [], [], 0
        for _ in range(count_views):
            texts = []
            for row in groups["retain"]:
                if mode == "dropout":
                    text = row["text"]
                elif mode == "fact_paraphrase":
                    views = fact_positive_candidates(row)
                    text = views[int(torch.randint(len(views), (), generator=positive_rng))]
                    changed += int(text != row["text"])
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
            all_texts.append(texts)
            positives.append(gather(embed(model, *text_batch(tokenizer, texts, options["max_length"], device))))
        # Preserve the original scalar/gradient path and forward order at K=1.
        second = positives[0]
        value = (paired_loss(r, second, f, options["temperature_retain"]) if count_views == 1
                 else retain_loss(r, torch.stack(positives), f,
                    temperature=options["temperature_retain"], negative_views=negative_views))
        metrics["specified_cl"] = value
        metrics["specified_positive_cosine"] = torch.stack([
            (r * positive).sum(-1).mean().detach() for positive in positives]).mean()
        if count_views > 1:
            p = torch.stack(positives).detach().float()
            metrics["specified_view_pair_cosine"] = (
                (p.sum(0).square().sum(-1) - p.square().sum((0, 2)))
                / (count_views * (count_views - 1))).mean()
        unique_texts = sum(len({texts[i] for texts in all_texts}) for i in range(len(texts)))
        count = torch.tensor([changed, count_views * len(texts), unique_texts, len(texts)],
                             device=device, dtype=torch.float32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(count)
        metrics["specified_changed_fraction"] = count[0] / count[1]
        metrics["specified_unique_texts_per_anchor"] = count[2] / count[3]
        metrics["specified_views"] = count.new_tensor(count_views)
        metrics["specified_negatives_per_anchor"] = count.new_tensor((len(r) - 1) * (1 + negative_views) + len(f))
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
                    generators, rank, world, sampling=None):
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
    if sampling is not None:
        sampling.snapshot(temp, step)
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
        sampling = None
        if cfg.get("diagnostics", {}).get("sampling_coverage", False):
            from .sampling import SamplingAudit
            sampling = SamplingAudit(data, root, identity, rank=rank,
                resume=Path(resume) if resume else None, first_step=first_step)
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
        diagnostic = None
        if cfg.get("diagnostics", {}).get("enabled", False):
            from .diagnostics import Diagnostics
            diagnostic = Diagnostics(model, tokenizer, cfg, root, rank, world,
                                     first_step=first_step)
            if first_step == 0:
                diagnostic.observe(0)
        for step in range(first_step, steps):
            begin = time.monotonic()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            logged = {}
            accumulation = options["gradient_accumulation_steps"]
            for micro in range(accumulation):
                draw = step * accumulation + micro
                groups = {name: sample_batch(rows, sizes[name], cfg["run"]["seed"] + offset * 10007,
                                             draw, rank, world)
                          for offset, (name, rows) in enumerate(data.items())}
                if sampling is not None:
                    sampling.observe(groups, step + 1, micro)
                context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and mc["dtype"] == "bfloat16" else nullcontext()
                with context:
                    loss, components = losses(model, tokenizer, groups, cfg,
                                               generators["corruption"], generators["positive"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step + 1}")
                if diagnostic is not None and diagnostic.due(step + 1, gradients=True):
                    diagnostic.gradients(components, step + 1, micro,
                                         accumulation=accumulation)
                (loss / accumulation).backward()
                for key, value in dict(components, total=loss).items():
                    logged[key] = logged.get(key, 0.0) + value.detach().float().item() / accumulation
            sync_gradients(parameters, world)
            norm = torch.nn.utils.clip_grad_norm_(parameters, options["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if diagnostic is not None and diagnostic.due(step + 1):
                diagnostic.observe(step + 1)
            peak = torch.zeros(2, device=device, dtype=torch.long)
            if device.type == "cuda":
                peak[0] = torch.cuda.max_memory_allocated(device)
                peak[1] = torch.cuda.max_memory_reserved(device)
            if world > 1:
                torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            if rank == 0:
                record = dict(step=step + 1, **logged, grad_norm=float(norm),
                              learning_rate=scheduler.get_last_lr()[0], elapsed_seconds=time.monotonic() - begin,
                              cuda_peak_allocated_bytes_max_rank=int(peak[0]),
                              cuda_peak_reserved_bytes_max_rank=int(peak[1]),
                              job_id=os.environ.get("PBS_JOBID"), resume_from=first_step)
                with (root / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record), flush=True)
            stop = should_stop() or (stop_after_step is not None and step + 1 >= stop_after_step)
            if (step + 1) % options["save_steps"] == 0 or step + 1 == steps or stop:
                save_checkpoint(model, tokenizer, optimizer, scheduler, root, step + 1, cfg,
                                generators, rank, world, sampling=sampling)
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
