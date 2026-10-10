"""Nine authorised Gemma insertion runs; no new dropout/replacement controls."""

import copy
import json
from pathlib import Path
import sys

from . import campaign as c
from .io import read, write, sha, file_sha, training_identity
from .positive_grid import historical_controls

PROFILE = "insertion-grid"
NAME = "0390-insertion-grid-v1"
POLICY = "training-fact-gap-insertion-v1"


def specs():
    tasks = []
    for mode, suffix, priority in (("fixed1", "I1", 0), ("fixed2", "I2", 0), ("binomial2p20", "IB2N20", 1)):
        for p in (2, 3, 4):
            variant = f"G256B32W2P{p}C70V4{suffix}"
            tasks.append({"id": f"gemma2_9b-{variant}-s42", "model": "gemma2_9b",
                "variant": variant, "seed": 42, "priority": priority, "preferred_worker": p-2,
                "late_admission": True, "stage": "insertion-exploration",
                "grid": {"rank": 256, "retain_batch": 32, "forget_weight": 2,
                    "retain_views": p, "forget_corruption": .7, "forget_views": 4,
                    "insertion_mode": mode}})
    return tasks


def apply_spec(base, task, output):
    from .followup import apply_variant
    cfg = apply_variant(base, f"G256B32W2P{task['grid']['retain_views']}", 42, output)
    cfg["conrep"].update(corruption_rate=.7, views=4, negative_views=4,
        specified_noise_probability=0., specified_noise_kind="insertion",
        specified_noise_policy=POLICY, specified_insertion_mode=task["grid"]["insertion_mode"],
        specified_negative_source="clean_dropout", protected_positive=False)
    return cfg


def prepare(args):
    from experiments.config import read_rows
    from conrep.v2.model import load_tokenizer
    from .insertion import audit_rows
    root, origin, campaign = (Path(p).resolve() for p in (args.project_root, args.source_campaign, args.campaign))
    parent = read(origin / "plan.json")
    if campaign == origin or Path(parent["project_root"]).resolve() != root:
        raise ValueError("Insertion grid needs a separate campaign under the same project")
    if (campaign / "plan.json").exists():
        plan = read(campaign / "plan.json")
        if plan.get("followup", {}).get("ref") != args.ref or plan["followup"].get("profile") != PROFILE:
            raise ValueError("Existing insertion campaign uses another revision/profile")
        c.verify_snapshot(campaign, plan)
        print("Insertion grid already prepared; state and existing jobs retained.")
        return plan
    metadata = read(campaign / "PREPARING.json")
    if (metadata["ref"] != args.ref or metadata["source_campaign"] != str(origin)
            or metadata.get("profile") != PROFILE):
        raise ValueError("Insertion bootstrap provenance mismatch")
    source = read(origin / "source.json")
    inputs, assets = read(origin / "inputs.json"), read(origin / "model-assets.json")
    for name, expected in inputs.items():
        if file_sha(name) != expected:
            raise ValueError(f"Parent input changed: {name}")
    for name, expected in assets.items():
        stat = Path(name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Parent asset changed: {name}")
    controls, bases = historical_controls(origin, root)
    base = bases["gemma2_9b"]
    for control in controls:
        inputs[control["config"]] = control["config_hash"]
        inputs.update(control["evidence_hashes"])
    collections = {g: read_rows(Path(base["data"]["prepared_dir"]) / f"{g}.jsonl")
                   for g in ("forget", "retain", "general")}
    counts = {g: len(rows) for g, rows in collections.items()}
    if counts["forget"] != 100 or counts["retain"] != 900:
        raise ValueError("Insertion grid requires the fixed 100/900 training split")
    old = base["diagnostics"]
    if file_sha(old["probe_file"]) != old["probe_sha256"] or inputs.get(old["probe_file"]) != old["probe_sha256"]:
        raise ValueError("Fixed training probe changed")
    probe = campaign / "diagnostic-inputs/gemma2_9b.json"
    write(probe, read(old["probe_file"]))
    inputs[str(probe)] = file_sha(probe)
    diagnostics = dict(old, probe_file=str(probe), probe_sha256=file_sha(probe), sampling_coverage=True)
    tokenizer = load_tokenizer(base["unlearn"]["checkpoint"], True)
    embedding_size = read(Path(base["unlearn"]["checkpoint"]) / "config.json")["vocab_size"]
    audit = audit_rows(tokenizer, collections["retain"], base["conrep"]["max_length"], embedding_size)
    audit_path = campaign / "retain-insertion-audit.json"
    write(audit_path, {"gemma2_9b": audit})
    if audit["unsupported_rows"] or audit["audited_rows"] != 900:
        raise ValueError("Insertion audit failed; no plan/jobs published; inspect retain-insertion-audit.json")
    inputs[str(audit_path)] = file_sha(audit_path)
    frozen = campaign / "code"
    files = {str(p.relative_to(frozen)): file_sha(p) for p in sorted(frozen.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if files != dict(source["files"], **metadata["overlay_files"]):
        raise ValueError("Insertion source differs from the pinned parent plus overlay")
    source_hash = sha(files)
    design = {"schema": NAME, "new_runs": 9, "gemma_runs": 9, "llama_runs": 0,
        "tasks": specs(), "historical_controls": controls, "training_rows": counts,
        "new_dropout_controls": 0, "new_replacement_controls": 0,
        "negative_source": "clean_dropout", "negatives_per_anchor": 70,
        "primary_step": 125, "common_step": 120, "all_steps": list(range(10, 121, 10)) + [125],
        "seed_scope": "unlearning seed42; same SFT/data as historical Gemma controls",
        "automatic_selection": False, "test_evaluation": False, "adaptive_expansion": False,
        "low_dose": "Binomial(2,.2), independent per positive; intensity exploration, not an operator-matched ablation",
        "sampler_sha256": files["src/experiments/data.py"]}
    design_path = campaign / "insertion-grid-design.json"
    write(design_path, design)
    inputs[str(design_path)] = file_sha(design_path)
    write(campaign / "source.json", {"files": files, "source_hash": source_hash,
        "parent_source_hash": source["source_hash"], "overlay_ref": args.ref,
        "overlay_files": metadata["overlay_files"], "server_evaluator_preserved": True})
    write(campaign / "inputs.json", inputs)
    write(campaign / "model-assets.json", assets)
    tasks = []
    for task in specs():
        directory = campaign / "experiments" / task["id"]
        cfg = apply_spec(base, task, directory / "training")
        cfg["diagnostics"] = copy.deepcopy(diagnostics)
        cfg["night"] = {"source_hash": source_hash, "data_hashes": inputs,
            "model_assets_hash": sha(assets), "campaign": str(campaign),
            "experiment": task["id"], "parent_campaign": str(origin)}
        path = campaign / "configs" / (task["id"] + ".json")
        write(path, cfg)
        tasks.append(dict(task, config=str(path), config_hash=file_sha(path),
            output=str(directory), identity=training_identity(cfg)))
    smoke = {str(i): next(t["id"] for t in tasks if t["grid"]["retain_views"] == 4
             and t["grid"]["insertion_mode"] == mode)
             for i, mode in enumerate(("fixed2", "binomial2p20", "fixed1"))}
    plan = {"schema": c.NAME, "project_root": str(root), "campaign": str(campaign),
        "python": sys.executable, "entry": str(frozen / c.ENTRY), "shell": str(frozen / c.SHELL),
        "controller_entry": str(frozen / c.ENTRY), "controller_shell": str(frozen / c.SHELL),
        "workers": 3, "world_size": 8, "queue": "R9920261000", "account": "gcg51557", "rtype": "rt_HF",
        "hours": args.hours, "walltime": "12:00:00", "poll_seconds": 60,
        "max_attempts": 3, "recovery_policy": dict(c.RECOVERY_DEFAULTS),
        "source_hash": source_hash, "tasks": tasks, "initial_task_seconds": 3600,
        "reserve_bytes": int(args.reserve_gb * 10**9), "smoke_tasks": smoke,
        "start_after_campaign": str(origin), "start_after_plan_sha256": file_sha(origin / "plan.json"),
        "followup": {"version": NAME, "profile": PROFILE, "ref": args.ref,
            "parent_campaign": str(origin), "parent_source_hash": source["source_hash"]}}
    write(campaign / "state.json", {"status": "prepared", "started_at": None, "deadline": None,
        "tasks": {t["id"]: {"status": "pending", "attempts": 0, "failures": 0} for t in tasks},
        "workers": {str(i): {"job_id": None, "status": "new", "allocations": 0,
            "recovery_failures": 0} for i in range(3)}})
    write(campaign / "plan.json", plan)
    print(json.dumps({"campaign": str(campaign), "tasks": 9, "gemma": 9, "llama": 0,
        "retain_positives": [2, 3, 4], "workers": 3, "budget_hours": args.hours,
        "start_after_campaign": str(origin), "status": "prepared; no PBS submitted, budget not started"}, indent=2))
    return plan
