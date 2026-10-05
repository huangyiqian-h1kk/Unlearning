from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, merge, read_rows


def parser():
    p = argparse.ArgumentParser(description="0390 ClinicIA experiment entry point")
    p.add_argument(
        "stage",
        choices=[
            "config",
            "prepare",
            "assets",
            "preflight",
            "smoke",
            "sft",
            "unlearn",
            "baseline",
            "validate",
            "validate-series",
            "audit-mmlu",
            "select",
            "analyze",
            "falcon-layers",
            "relearn-augment",
        ],
    )
    p.add_argument("--config", default="configs/0390/qwen7b.yaml")
    p.add_argument(
        "--data-profile", choices=["pmc", "diagnosis", "deaths"], default="pmc"
    )
    p.add_argument(
        "--method", choices=["npo", "rmu", "falcon", "lunar", "sago", "relearn"]
    )
    p.add_argument("--ablation")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--resume")
    p.add_argument("--retain-only", action="store_true")
    p.add_argument("--checkpoint")
    p.add_argument("--checkpoint-root")
    p.add_argument("--output")
    p.add_argument("--metrics", nargs="+")
    p.add_argument("--baseline-metrics")
    p.add_argument(
        "--selection-stage",
        choices=["sft", "retain-only", "unlearn"],
        default="unlearn",
    )
    p.add_argument("--models", action="store_true")
    p.add_argument("--model-root")
    p.add_argument("--distributed", action="store_true")
    return p


def resolve(args):
    cfg = load_config(args.config)
    if args.data_profile != "pmc":
        cfg = merge(
            cfg,
            load_config(
                Path(__file__).resolve().parents[2]
                / "configs/0390/data"
                / f"{args.data_profile}.yaml"
            ),
        )
        cfg["unlearn"]["checkpoint"] = cfg["model"]["name_or_path"]
    if args.method:
        cfg = merge(
            cfg,
            load_config(
                Path(__file__).resolve().parents[2]
                / "configs/0390/methods"
                / f"{args.method}.yaml"
            ),
        )
    if args.ablation:
        all_ablations = load_config(
            Path(__file__).resolve().parents[2] / "configs/0390/ablations.yaml"
        )["ablations"]
        if args.ablation not in all_ablations:
            raise ValueError(f"Unknown ablation {args.ablation}")
        cfg = merge(cfg, all_ablations[args.ablation])
    # Overrides have highest priority, including over a method overlay.
    if args.set:
        cfg = merge(cfg, load_config_overrides(args.set))
    if args.checkpoint and args.stage in {"unlearn", "baseline", "falcon-layers"}:
        cfg["unlearn"]["checkpoint"] = args.checkpoint
    return cfg


def load_config_overrides(overrides):
    import os
    import yaml

    result = {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Expected KEY=VALUE: {item}")
        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(os.path.expandvars(value))
    return result


def preflight(cfg, distributed=False):
    import importlib.metadata
    import shutil
    import torch
    from .runtime import initialize, barrier

    device, rank, world = (
        initialize(cfg["run"]["seed"])
        if distributed
        else (torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, 1)
    )
    report = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "world_size": world,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate")
        },
        "shm_free_bytes": shutil.disk_usage("/dev/shm").free
        if Path("/dev/shm").exists()
        else None,
    }
    for name in ("forget", "retain", "general", "injection"):
        report[name + "_rows"] = len(
            read_rows(Path(cfg["data"]["prepared_dir"]) / f"{name}.jsonl")
        )
    from conrep.v2.model import load_tokenizer

    tok = load_tokenizer(cfg["model"]["name_or_path"], cfg["model"]["local_only"])
    if not tok.chat_template:
        raise ValueError("Native chat template missing")
    report["tokenizer_vocab"] = len(tok)
    if distributed and world > 1:
        value = torch.tensor(float(rank + 1), device=device)
        torch.distributed.all_reduce(value)
        assert value.item() == world * (world + 1) / 2, (
            "Distributed all-reduce mismatch"
        )
        report["all_reduce"] = "passed"
    barrier()
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = resolve(args)
    stage = args.stage
    result = None
    if stage == "config":
        result = cfg
    elif stage == "assets":
        from .assets import prepare_assets

        prepare_assets(
            args.output or "data/processed/0390", args.models, args.model_root
        )
    elif stage == "prepare":
        from .data import prepare

        result = prepare(cfg)
    elif stage == "preflight":
        result = preflight(cfg, args.distributed)
    elif stage == "smoke":
        report = preflight(cfg, True)
        cfg["unlearn"].update(
            checkpoint=cfg["model"]["name_or_path"], max_steps=2, save_steps=2
        )
        from conrep.v2.trainer import run

        run(cfg)
        result = {
            "preflight": report,
            "conrep_updates": 2,
            "purpose": "execution validation, not scientific evidence",
        }
    elif stage == "sft":
        from .sft import run

        run(cfg, args.resume, args.retain_only)
    elif stage == "unlearn":
        from conrep.v2.trainer import run

        run(cfg, args.resume)
    elif stage == "baseline":
        if not args.method:
            raise ValueError("baseline requires --method")
        from .baselines.trainer import run

        run(cfg, args.resume)
    elif stage == "validate":
        if not args.checkpoint or not args.output:
            raise ValueError("validate requires --checkpoint and --output")
        from .validation import run

        result = run(cfg, args.checkpoint, args.output)
    elif stage == "validate-series":
        import os

        if not args.checkpoint_root or not args.output:
            raise ValueError("validate-series requires --checkpoint-root and --output")
        import torch

        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        root = Path(args.checkpoint_root)
        paths = sorted(
            root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
        )
        if (root / "final").exists():
            paths.append(root / "final")
        if not paths:
            raise ValueError(f"No checkpoints found in {root}")
        from .validation import run
        import gc

        for path in paths[rank::world]:
            dest = Path(args.output) / path.name
            run(cfg, str(path), str(dest))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elif stage == "audit-mmlu":
        if not args.checkpoint or not args.output:
            raise ValueError("audit-mmlu requires --checkpoint and --output")
        from .mmlu_audit import run

        result = run(cfg, args.checkpoint, args.output)
    elif stage == "select":
        if not args.metrics or not args.baseline_metrics or not args.output:
            raise ValueError(
                "select requires --metrics, --baseline-metrics and --output"
            )
        from .selection import select

        result = select(
            cfg, args.metrics, args.baseline_metrics, args.selection_stage, args.output
        )
    elif stage == "analyze":
        if not args.checkpoint or not args.output:
            raise ValueError("analyze requires --checkpoint and --output")
        from .analysis import run

        result = run(cfg, args.checkpoint, args.output)
    elif stage == "falcon-layers":
        from .baselines.prepare import falcon_layers

        result = falcon_layers(cfg, args.output or cfg["baseline"]["layer_selection"])
    elif stage == "relearn-augment":
        from .baselines.prepare import relearn_augment

        result = relearn_augment(cfg, args.output or cfg["baseline"]["augmented_file"])
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
