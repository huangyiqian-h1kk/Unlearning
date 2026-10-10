#!/usr/bin/env python3
"""Read-only Qwen SFT inventory; no model loading, GPU, writes or PBS calls.

File presence is not a model-load or exact-resume verification. In particular,
validation reports can survive after their associated weights are removed.
"""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def checkpoint_inventory(path):
    path = path.resolve()
    result = {"path": str(path), "exists": path.is_dir(), "hf_weight_files_complete": False,
              "model_load_verified": False, "exact_resume_verified": False}
    if not path.is_dir():
        return result
    formats = []
    for index_name, single_name in (("model.safetensors.index.json", "model.safetensors"),
                                    ("pytorch_model.bin.index.json", "pytorch_model.bin")):
        index = path / index_name
        if index.is_file():
            try:
                mapping = read(index)["weight_map"]
                if not isinstance(mapping, dict) or not mapping:
                    raise ValueError("Empty or invalid weight_map")
                shards = sorted(set(mapping.values()))
                if any(not isinstance(name, str) or not (path / name).resolve().is_relative_to(path) for name in shards):
                    raise ValueError("Invalid/outside-checkpoint shard path")
                missing = [name for name in shards if not nonempty(path / name)]
                formats.append({"index": index_name, "shards": shards, "missing_or_empty": missing,
                    "complete": not missing, "present_bytes": sum((path / name).stat().st_size
                        for name in shards if nonempty(path / name))})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                formats.append({"index": index_name, "complete": False, "error": str(exc)})
        elif nonempty(path / single_name):
            formats.append({"file": single_name, "complete": True,
                            "present_bytes": (path / single_name).stat().st_size})
    result["weight_formats"] = formats
    result["config_present"] = nonempty(path / "config.json")
    result["hf_weight_files_complete"] = result["config_present"] and any(item["complete"] for item in formats)
    result["tokenizer_files"] = [name for name in ("tokenizer.json", "tokenizer.model", "vocab.json",
        "merges.txt", "tokenizer_config.json", "special_tokens_map.json") if nonempty(path / name)]
    result["resume_files"] = sorted({str(p.relative_to(path)) for pattern in (
        "trainer_state.json", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state*.pth",
        "latest", "global_step*/*optim_states.pt", "global_step*/*model_states.pt")
        for p in path.glob(pattern) if nonempty(p)})
    if (path / "trainer_state.json").is_file():
        try:
            state = read(path / "trainer_state.json")
            result["trainer_state"] = {k: state.get(k) for k in ("global_step", "epoch", "max_steps")}
        except (OSError, ValueError) as exc:
            result["trainer_state_error"] = str(exc)
    return result


def inspect(root):
    root = Path(root).resolve()
    injection, sft = root / "injection", root / "injection/sft"
    report = {"root": str(root), "root_exists": root.is_dir(), "read_only": True, "records": {},
        "qualification": "Nonempty files/index coverage only. No weights deserialized; exact optimizer/RNG resume is not verified."}
    candidates = {sft / "checkpoint-700", sft / "final"}
    candidates.update(p for p in sft.glob("checkpoint-*") if p.is_dir())
    for path in (injection / "selected-sft.json", injection / "pipeline.json",
                 sft / "TRAINING_COMPLETE.json", sft / "lineage.json", sft / "resolved_config.json"):
        if not path.is_file():
            report["records"][str(path.relative_to(root))] = {"exists": False}
            continue
        try:
            value = read(path)
            if path.name == "resolved_config.json":
                value = {key: value.get(key) for key in ("model", "data", "sft", "run")}
            elif path.name == "pipeline.json":
                value = {key: value.get(key) for key in ("status", "global_step", "started_at", "ended_at", "selected")}
            report["records"][str(path.relative_to(root))] = value
            selected = value.get("selected")
            if isinstance(selected, str) and Path(selected).is_absolute():
                candidates.add(Path(selected))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            report["records"][str(path.relative_to(root))] = {"error": str(exc)}
    report["checkpoints"] = [checkpoint_inventory(path) for path in sorted(candidates)]
    # Report metrics separately; never infer model availability from these files.
    report["validation"] = {}
    for name in ("base", "checkpoint-700", "final"):
        path = injection / "validation" / name / "metrics.json"
        if path.is_file():
            try:
                value = read(path)
                report["validation"][name] = {key: value.get(key) for key in
                    ("checkpoint", "protocol_hash", "metrics", "mmlu_diagnostics")}
            except (OSError, ValueError, AttributeError) as exc:
                report["validation"][name] = {"error": str(exc)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
        default=Path.cwd() / "results/validated_v2/0390/rich-all-ia-seed42/qwen7b")
    args = parser.parse_args()
    print(json.dumps(inspect(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
