"""LUNAR local down-projection fitting with a frozen feature model.

Adapted from facebookresearch/LUNAR (MIT).
Copyright (c) Meta Platforms, Inc. and affiliates.
See third_party/0390_baselines/LUNAR/LICENSE and docs/0390/baselines.md.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from conrep.v2.model import load_model, load_tokenizer
from experiments.config import output_dir, read_rows, write_json
from experiments.data import sample_batch
from experiments.runtime import initialize, sync_gradients, barrier, local_batch_sizes
from .common import activation
from .prepare import requests_for_facts


def prompt_batch(tok, text, max_length, device):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
    )
    ids = torch.tensor([ids[-max_length:]], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def run(cfg, resume=None):
    device, rank, world = initialize(cfg["run"]["seed"])
    options, train, mc = cfg["baseline"], cfg["unlearn"], cfg["model"]
    sizes = local_batch_sizes(train, world)
    if train["gradient_accumulation_steps"] != 1:
        raise ValueError(
            "LUNAR fits local linear weights per step; set gradient_accumulation_steps=1"
        )
    root = output_dir(cfg)
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"Output exists: {root}")
    model = load_model(
        train["checkpoint"],
        device,
        local_only=mc["local_only"],
        dtype=mc["dtype"],
        attention=mc["attention"],
    )
    model.requires_grad_(False)
    tok = load_tokenizer(train["checkpoint"], mc["local_only"])
    layer = options["layer"]
    original = model.model.layers[layer].mlp.down_proj
    fit = torch.nn.Linear(original.in_features, original.out_features, bias=False).to(
        device, dtype=torch.float32
    )
    fit.weight.data.copy_(original.weight.float())
    data = {
        "forget": [
            dict(row, text=row["prompt"]) for row in requests_for_facts(cfg, "forget")
        ],
        "retain": [
            dict(row, text=row["prompt"]) for row in requests_for_facts(cfg, "retain")
        ],
        "general": read_rows(Path(cfg["data"]["prepared_dir"]) / "general.jsonl"),
    }
    refusal = read_rows(options["refusal_file"])
    refusal = [row.get("instruction", row.get("prompt", "")) for row in refusal][
        : options["calibration_samples"]
    ]
    if not refusal or not all(refusal):
        raise ValueError("LUNAR requires non-empty refusal-calibration instructions")
    with torch.no_grad():

        def mean(texts):
            vectors = [
                activation(
                    model, prompt_batch(tok, text, train["max_length"], device), layer
                )[0, -1].float()
                for text in texts
            ]
            return torch.stack(vectors).mean(0)

        direction = mean(refusal) - mean(
            [row["text"] for row in data["forget"][: options["calibration_samples"]]]
        )
    optimizer = torch.optim.AdamW(
        fit.parameters(), lr=options["fit_learning_rate"], weight_decay=0.01
    )
    steps = train.get("max_steps") or options["fit_steps"]
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=options["scheduler_gamma"]
    )
    first_step = 0
    if resume:
        if not (Path(resume) / "COMPLETE.json").exists():
            raise ValueError("Incomplete LUNAR checkpoint")
        state = torch.load(
            Path(resume) / "lunar_state.pt", weights_only=False, map_location=device
        )
        if state["config"]["run"]["seed"] != cfg["run"]["seed"]:
            raise ValueError("Resume requires unchanged seed")
        for field in ("baseline", "unlearn", "data", "model"):
            if state["config"][field] != cfg[field]:
                raise ValueError(f"Resume configuration changed: {field}")
        if state["world_size"] != world:
            raise ValueError("Resume requires unchanged world size")
        fit.load_state_dict(state["fit"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        direction = state["direction"].to(device)
        first_step = state["step"]
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "resolved_config.json", cfg)
        write_json(
            root / "lineage.json",
            {
                "method": "lunar",
                "checkpoint": train["checkpoint"],
                "layer": layer,
                "feature_model": "frozen start checkpoint",
                "direction": "refusal mean minus forget mean at final prompt token",
                "target_shift": "all valid tokens, matching pinned upstream code",
                "steps": steps,
            },
        )

    def features(text):
        cache = []
        handle = original.register_forward_hook(
            lambda _m, args, out: cache.append(
                (args[0].detach().float(), out.detach().float())
            )
        )
        try:
            with torch.no_grad():
                batch = prompt_batch(tok, text, train["max_length"], device)
                model.model(**batch, use_cache=False)
        finally:
            handle.remove()
        return cache[0]

    for step in range(first_step, steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for index, (name, rows) in enumerate(data.items()):
            selected = sample_batch(
                rows, sizes[name], cfg["run"]["seed"] + index * 10007, step, rank, world
            )
            weight = (
                options["forget_weight"]
                if name == "forget"
                else options["specified_weight"]
                if name == "retain"
                else options["general_weight"]
            )
            for row in selected:
                x, target = features(row["text"])
                if name == "forget":
                    target = target + options["direction_coefficient"] * direction
                loss = F.mse_loss(fit(x), target) * weight / len(selected)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite LUNAR fitting loss")
                loss.backward()
                total += float(loss.detach())
        sync_gradients(list(fit.parameters()), world)
        torch.nn.utils.clip_grad_norm_(
            fit.parameters(), train["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        if (step + 1) % options["scheduler_every_steps"] == 0:
            scheduler.step()
        if rank == 0:
            with (root / "train.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step + 1, "fit_loss": total}) + "\n")
            print(json.dumps({"step": step + 1, "fit_loss": total}), flush=True)
        if (step + 1) % train["save_steps"] == 0 or step + 1 == steps:
            if rank == 0:
                path = root / f"checkpoint-{step + 1}"
                backup = original.weight.detach().clone()
                try:
                    original.weight.data.copy_(fit.weight.to(original.weight))
                    model.save_pretrained(path)
                    tok.save_pretrained(path)
                finally:
                    original.weight.data.copy_(backup)
                torch.save(
                    {
                        "fit": fit.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "direction": direction,
                        "step": step + 1,
                        "config": cfg,
                        "world_size": world,
                    },
                    path / "lunar_state.pt",
                )
                write_json(
                    path / "COMPLETE.json",
                    {"method": "lunar", "step": step + 1, "kind": "full_model"},
                )
            barrier()
    if rank == 0:
        write_json(root / "TRAINING_COMPLETE.json", {"method": "lunar", "steps": steps})
