"""Follow-up matrices built from the previous frozen campaign."""

import copy
import csv
import json
from pathlib import Path
import sys

from . import campaign as c
from .io import read, write, file_sha, sha, training_identity
from .positives import audit, fact_positive_audit, fact_fields

NAME = "0390-conrep-followup-v1"
PROFILES = ("original", "llama-ms")
CHANGES = {
    "M": ("K", {"lora.r": 256, "lora.lora_alpha": 512}),
    "N": ("K", {"unlearn.learning_rate": 5e-6}),
    "O": ("J", {"unlearn.learning_rate": 2e-5}),
    "P": ("K", {"conrep.forget_cl_weight": 1.0}),
    "Q": ("J", {"conrep.specified_positive": "fact_paraphrase"}),
    "R": ("J", {"conrep.stop_gradient_controls": True}),
    "S": ("J", {"conrep.stop_gradient_retain": True}),
}


def specs(profile="original"):
    if profile not in PROFILES:
        raise ValueError(f"Unknown follow-up profile: {profile}")
    if profile == "llama-ms":
        # Admit every seed-42 setting before admitting the seed-43 repeats.
        # Worker preference is only a tie-breaker; all three share this pool.
        return [{"id": f"llama8b-{variant}-s{seed}", "model": "llama8b",
                 "variant": variant, "seed": seed, "priority": priority,
                 "preferred_worker": index % 3, "late_admission": True,
                 "stage": "completion-and-replication"}
                for priority, seed in enumerate((42, 43))
                for index, variant in enumerate(CHANGES)]
    first = [("llama8b", "J", 42), ("llama8b", "K", 42), ("gemma2_9b", "K", 43)]
    next_rank = [("llama8b", "J", 43), ("llama8b", "K", 43),
                 ("gemma2_9b", "J", 43), ("gemma2_9b", "J", 44), ("gemma2_9b", "K", 44)]
    rest = [("llama8b", variant, seed) for seed in (42, 43) for variant in "EFGHIL"]
    explore = [("gemma2_9b", variant, 42) for variant in CHANGES]
    result = []
    for priority, group in enumerate((first, next_rank, rest, explore)):
        for index, (model, variant, seed) in enumerate(group):
            result.append({"id": f"{model}-{variant}-s{seed}", "model": model,
                           "variant": variant, "seed": seed, "priority": priority,
                           "preferred_worker": index % 3, "late_admission": priority < 3,
                           "stage": "exploration" if priority == 3 else "completion-and-replication"})
    return result


def apply_variant(base, variant, seed, output):
    parent, overrides = CHANGES.get(variant, (variant, {}))
    cfg = c.apply_variant(base, parent, seed, output)
    for key, value in overrides.items():
        section, field = key.split(".")
        cfg[section][field] = value
    return cfg


def diagnostic_rows(rows, group, count):
    """Fixed training-only samples shared across variants/seeds of one model."""
    ordered = sorted(rows, key=lambda row: sha({"seed": 39027, "group": group,
                                               "id": row.get("id"), "text": row["text"]}))
    output, rejected = [], []
    for row in ordered:
        item = {"id": str(row.get("id", sha(row))), "group": group, "text": row["text"]}
        if group != "general":
            try:
                fields = fact_fields(row)
            except ValueError as exc:
                rejected.append(str(exc))
                continue
            item.update(prompt=f"What {fields['copula']} the {fields['attribute']} of {fields['entity']}?",
                        answer=fields["value"], fact=fields)
        output.append(item)
        if len(output) == count:
            break
    if len(output) < count:
        raise ValueError(f"Need {count} complete training facts for {group} diagnostics; "
                         f"got {len(output)}. Examples: {rejected[:3]}")
    return output, {"selected": len(output), "unsupported_before_sample_complete": len(rejected),
                    "source_rows": len(rows), "uses_validation_or_test": False}


def prepare(args):
    from experiments.config import read_rows
    profile = getattr(args, "profile", "original")
    matrix = specs(profile)
    root, origin, campaign = (Path(p).resolve() for p in
                              (args.project_root, args.source_campaign, args.campaign))
    parent = read(origin / "plan.json")
    if campaign == origin or root != Path(parent["project_root"]).resolve():
        raise ValueError("Follow-up must be a new campaign under the same checkout")
    if (campaign / "plan.json").exists():
        existing = read(campaign / "plan.json")
        if (existing.get("followup", {}).get("ref") != args.ref
                or existing.get("followup", {}).get("profile", "original") != profile):
            raise ValueError("Existing follow-up uses another revision/profile; preserve it")
        c.verify_snapshot(campaign, existing)
        print("Follow-up already prepared; no task/state reset and no PBS submitted.")
        return existing
    metadata = read(campaign / "PREPARING.json")
    if (metadata["ref"] != args.ref or metadata["source_campaign"] != str(origin)
            or metadata.get("profile", "original") != profile):
        raise ValueError("Bootstrap provenance mismatch")
    # Bootstrap has checked the original immutable snapshot. Check inputs again
    # before deriving any configs; old trainers/evaluators are not reinstalled.
    source = read(origin / "source.json")
    for name, expected in source["files"].items():
        if file_sha(origin / "code" / name) != expected:
            raise ValueError(f"Parent source changed: {name}")
    inputs = read(origin / "inputs.json")
    for name, expected in inputs.items():
        if file_sha(name) != expected:
            raise ValueError(f"Parent input changed: {name}")
    assets = read(origin / "model-assets.json")
    bases, diagnostics, audits, qa_audits = {}, {}, {}, {}
    for model in sorted({task["model"] for task in matrix}):
        task = next(t for t in parent["tasks"] if t["model"] == model
                    and t["variant"] == "A" and t["seed"] == 42)
        if file_sha(task["config"]) != task["config_hash"]:
            raise ValueError("Parent A/42 configuration changed")
        cfg = c.absolute_paths(read(task["config"]), root)
        if cfg["evaluation"].get("partition") != "validation" or cfg["evaluation"].get("limit"):
            raise ValueError("Full frozen validation protocol is required")
        if cfg["unlearn"]["batch_sizes"] != {"forget": 8, "retain": 16, "general": 32}:
            raise ValueError("Expected unchanged global batches 8/16/32")
        if cfg["unlearn"]["learning_rate"] != 1e-5:
            raise ValueError("The approved LR controls require baseline learning_rate=1e-5")
        collections = {g: read_rows(Path(cfg["data"]["prepared_dir"]) / f"{g}.jsonl")
                       for g in ("forget", "retain", "general")}
        audits[model] = audit(collections["retain"])
        if not audits[model]["eligible_rows"]:
            raise ValueError("Protected positives have zero coverage")
        if any(task["model"] == model and task["variant"] == "Q" for task in matrix):
            qa_audits[model] = fact_positive_audit(collections["retain"])
            write(campaign / "fact-positive-audit.json", qa_audits)
            if qa_audits[model]["unsupported_rows"] or not qa_audits[model]["eligible_rows"]:
                raise ValueError("Q requires complete audited fact positives; inspect fact-positive-audit.json")
        probes, coverage = [], {}
        for group, count in (("forget", 16), ("retain", 16), ("general", 8)):
            sample, coverage[group] = diagnostic_rows(collections[group], group, count)
            probes.extend(sample)
        probe = campaign / "diagnostic-inputs" / f"{model}.json"
        write(probe, {"schema": "training-only-fixed-probes-v1", "rows": probes,
                      "coverage": coverage, "selection_seed": 39027})
        inputs[str(probe)] = file_sha(probe)
        diagnostics[model] = {"enabled": True, "every_steps": 25,
            "probe_file": str(probe), "probe_sha256": file_sha(probe),
            "gradient_tensors": 4, "gradient_max_tensor_numel": 2000000,
            "fixed_corruption_probability": 0.7, "fixed_corruption_views": 4,
            "seed": 39027, "max_length": cfg["conrep"]["max_length"]}
        bases[model] = cfg
    for name, expected in assets.items():
        stat = Path(name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Parent SFT asset changed: {name}")
    frozen = campaign / "code"
    files = {str(p.relative_to(frozen)): file_sha(p) for p in sorted(frozen.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    source_hash = sha(files)
    write(campaign / "source.json", {"files": files, "source_hash": source_hash,
          "parent_source_hash": source["source_hash"], "overlay_ref": args.ref,
          "overlay_files": metadata["overlay_files"], "server_evaluator_preserved": True})
    write(campaign / "inputs.json", inputs)
    write(campaign / "model-assets.json", assets)
    write(campaign / "positive-audit.json", audits)
    tasks = []
    for spec in matrix:
        directory = campaign / "experiments" / spec["id"]
        cfg = apply_variant(bases[spec["model"]], spec["variant"], spec["seed"], directory / "training")
        cfg["diagnostics"] = copy.deepcopy(diagnostics[spec["model"]])
        cfg["night"] = {"source_hash": source_hash, "data_hashes": inputs,
            "model_assets_hash": sha(assets), "campaign": str(campaign),
            "experiment": spec["id"], "parent_campaign": str(origin)}
        path = campaign / "configs" / (spec["id"] + ".json")
        write(path, cfg)
        tasks.append(dict(spec, config=str(path), config_hash=file_sha(path),
                          output=str(directory), identity=training_identity(cfg)))
    plan = {"schema": c.NAME, "project_root": str(root), "campaign": str(campaign),
        "python": sys.executable, "entry": str(frozen / c.ENTRY), "shell": str(frozen / c.SHELL),
        "controller_entry": str(frozen / c.ENTRY), "controller_shell": str(frozen / c.SHELL),
        "workers": 3, "world_size": 8, "queue": "R9920261000", "account": "gcg51557", "rtype": "rt_HF",
        "hours": args.hours, "walltime": "12:00:00", "poll_seconds": 60,
        "max_attempts": 3, "recovery_policy": dict(c.RECOVERY_DEFAULTS),
        "source_hash": source_hash, "tasks": tasks, "initial_task_seconds": 3600,
        "reserve_bytes": int(args.reserve_gb * 10**9),
        "followup": {"version": NAME, "ref": args.ref, "profile": profile,
                     "parent_campaign": str(origin), "parent_source_hash": source["source_hash"],
                     "completion_runs": sum(t["stage"] == "completion-and-replication" for t in tasks),
                     "exploration_runs": sum(t["stage"] == "exploration" for t in tasks)}}
    write(campaign / "state.json", {"status": "prepared", "started_at": None, "deadline": None,
        "tasks": {t["id"]: {"status": "pending", "attempts": 0, "failures": 0} for t in tasks},
        "workers": {str(i): {"job_id": None, "status": "new", "allocations": 0,
                              "recovery_failures": 0} for i in range(3)}})
    write(campaign / "plan.json", plan)  # Publish only after every config/input is complete.
    print(json.dumps({"campaign": str(campaign), "profile": profile, "tasks": len(tasks),
        "llama_runs": sum(t["model"] == "llama8b" for t in tasks),
        "gemma_replications": sum(t["model"] == "gemma2_9b" and t["stage"] == "completion-and-replication" for t in tasks),
        "gemma_explorations": sum(t["model"] == "gemma2_9b" and t["stage"] == "exploration" for t in tasks),
        "workers": 3,
        "pbs_walltime": plan["walltime"], "budget_hours": args.hours,
        "diagnostics_every_steps": 25, "status": "prepared; no PBS submitted"}, indent=2))
    return plan


def _csv(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize_diagnostics(campaign, plan):
    observations, gradients = [], []
    for task in plan["tasks"]:
        root = Path(task["output"]) / "training" / "diagnostics"
        common = {key: task[key] for key in ("id", "model", "variant", "seed")}
        for path in sorted(root.glob("step-*.json")):
            report = read(path)
            observations.append(dict(common, step=report["step"], **report["summary"]))
        for path in sorted(root.glob("gradients-step-*.json")):
            report = read(path)
            row = dict(common, step=report["step"], microbatch=report["microbatch"],
                       sampled_parameter_count=report["sampled_parameter_count"])
            for key in ("raw_norms", "weighted_norms", "cosines"):
                row.update({key + "." + k: v for k, v in report[key].items()})
            gradients.append(row)
    _csv(campaign / "diagnostic-results.csv", observations)
    _csv(campaign / "gradient-diagnostics.csv", gradients)
