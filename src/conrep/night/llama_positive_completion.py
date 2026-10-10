"""Mirror the 18 Gemma positive/noise experiments on the same Llama SFT.

Reuse completed seed42 P1/P4 controls; do not add old seed-only replications.
The frozen parent supplies data and evaluation, including its existing parser.
"""

import copy
import json
from pathlib import Path
import sys

from . import campaign as c, insertion_grid as insertion, positive_grid as positive
from .io import read, write, sha, file_sha, training_identity

PROFILE = "llama-positive-completion"
NAME = "0390-llama-positive-completion-v1"
CONTROLS = positive.CONTROLS + (("llama8b", "G256B32W2P1"), ("llama8b", "G256B32W2P4"))


def specs():
    mirror = [t for t in positive.specs() if t["model"] == "gemma2_9b" and t["seed"] == 42]
    mirror += insertion.specs()
    tasks = []
    first_insertions = {"G256B32W2P4C70V4I1", "G256B32W2P2C70V4I1", "G256B32W2P4C70V4IB2N20"}
    for index, source in enumerate(mirror):
        task = copy.deepcopy(source)
        is_forget = task.get("label", "").startswith("GF")
        task.update(id=f"llama8b-{task['variant']}-s42", model="llama8b",
            mirrored_experiment=source["id"], stage="cross-model-completion",
            priority=0 if is_forget or task["variant"] in first_insertions else 1,
            preferred_worker=index % 3, late_admission=True)
        tasks.append(task)
    return sorted(tasks, key=lambda task: task["priority"])


def apply_spec(base, task, output):
    if "insertion_mode" in task["grid"]:
        return insertion.apply_spec(base, task, output)
    return positive.apply_spec(base, task, output)


def prepare(args):
    from experiments.config import read_rows
    from conrep.v2.model import load_tokenizer
    from .noise import audit_rows as replacement_audit
    from .insertion import audit_rows as insertion_audit
    root, origin, campaign = (Path(p).resolve() for p in (args.project_root, args.source_campaign, args.campaign))
    parent = read(origin / "plan.json")
    if campaign == origin or Path(parent["project_root"]).resolve() != root:
        raise ValueError("Llama completion needs a separate campaign under the same project")
    if (campaign / "plan.json").exists():
        plan = read(campaign / "plan.json")
        if plan.get("followup", {}).get("ref") != args.ref or plan["followup"].get("profile") != PROFILE:
            raise ValueError("Existing Llama completion uses another revision/profile")
        c.verify_snapshot(campaign, plan)
        print("Llama completion already prepared; state and existing jobs retained.")
        return plan
    metadata = read(campaign / "PREPARING.json")
    if (metadata["ref"] != args.ref or metadata["source_campaign"] != str(origin)
            or metadata.get("profile") != PROFILE):
        raise ValueError("Llama completion bootstrap provenance mismatch")
    source = read(origin / "source.json")
    inputs, assets = read(origin / "inputs.json"), read(origin / "model-assets.json")
    for name, expected in inputs.items():
        if file_sha(name) != expected:
            raise ValueError(f"Parent input changed: {name}")
    for name, expected in assets.items():
        stat = Path(name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Parent asset changed: {name}")
    controls, bases = positive.historical_controls(origin, root, controls=CONTROLS)
    base = bases["llama8b"]
    for control in controls:
        inputs[control["config"]] = control["config_hash"]
        inputs.update(control["evidence_hashes"])
    # A second completion campaign must resume the first instead of rerunning it.
    seen, location = set(), origin
    proposed_ids = {task["id"] for task in specs()}
    while location not in seen:
        seen.add(location)
        previous = read(location / "plan.json")
        duplicates = proposed_ids & {task.get("id") for task in previous["tasks"]}
        if duplicates:
            raise ValueError(f"Llama completion tasks already exist in {location}; resume that campaign: {sorted(duplicates)}")
        ancestor = previous.get("followup", {}).get("parent_campaign")
        if not ancestor:
            break
        location = Path(ancestor).resolve()
    collections = {g: read_rows(Path(base["data"]["prepared_dir"]) / f"{g}.jsonl")
                   for g in ("forget", "retain", "general")}
    counts = {g: len(rows) for g, rows in collections.items()}
    if counts["forget"] != 100 or counts["retain"] != 900:
        raise ValueError("Llama completion requires the fixed 100/900 training split")
    old = base["diagnostics"]
    if file_sha(old["probe_file"]) != old["probe_sha256"] or inputs.get(old["probe_file"]) != old["probe_sha256"]:
        raise ValueError("Fixed Llama training probe changed")
    probe = campaign / "diagnostic-inputs/llama8b.json"
    write(probe, read(old["probe_file"]))
    inputs[str(probe)] = file_sha(probe)
    diagnostics = dict(old, probe_file=str(probe), probe_sha256=file_sha(probe), sampling_coverage=True)
    tokenizer = load_tokenizer(base["unlearn"]["checkpoint"], True)
    embedding_size = read(Path(base["unlearn"]["checkpoint"]) / "config.json")["vocab_size"]
    for name, audit_fn in (("retain-noise-audit.json", replacement_audit),
                           ("retain-insertion-audit.json", insertion_audit)):
        audit = audit_fn(tokenizer, collections["retain"], base["conrep"]["max_length"], embedding_size)
        path = campaign / name
        write(path, {"llama8b": audit})
        if (audit["unsupported_rows"] or audit["audited_rows"] != 900
                or (name == "retain-noise-audit.json" and audit["eligible_rows"] != 900)):
            raise ValueError(f"Llama token audit failed; no plan/jobs published; inspect {name}")
        inputs[str(path)] = file_sha(path)
    frozen = campaign / "code"
    files = {str(p.relative_to(frozen)): file_sha(p) for p in sorted(frozen.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if files != dict(source["files"], **metadata["overlay_files"]):
        raise ValueError("Llama completion source differs from the pinned parent plus overlay")
    source_hash = sha(files)
    design = {"schema": NAME, "new_runs": 18, "gemma_runs": 0, "llama_runs": 18,
        "tasks": specs(), "historical_controls": controls, "training_rows": counts,
        "new_dropout_controls": 0, "forget_runs": 5, "replacement_runs": 4, "insertion_runs": 9,
        "forget_negative_views": 4, "retain_negative_views": 1,
        "primary_step": 125, "common_step": 120, "all_steps": list(range(10, 121, 10)) + [125],
        "seed_scope": "unlearning seed42; same Llama SFT/data as historical controls",
        "epoch_scope": "125*8/100=10 forget-equivalent epochs; sampled draws, not guaranteed passes",
        "automatic_selection": False, "test_evaluation": False, "adaptive_expansion": False,
        "low_dose": "Binomial(2,.2), independent per positive; mean .4 inserted tokens, not 20% of tokens",
        "sampler_sha256": files["src/experiments/data.py"]}
    design_path = campaign / "llama-positive-completion-design.json"
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
    smoke = {str(i): f"llama8b-{variant}-s42" for i, variant in enumerate(
        ("G256B32W2P1C90V8N00", "G256B32W2P4C70V4N20", "G256B32W2P4C70V4I2"))}
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
    print(json.dumps({"campaign": str(campaign), "tasks": 18, "gemma": 0, "llama": 18,
        "workers": 3, "budget_hours": args.hours, "smoke_tasks": smoke,
        "start_after_campaign": str(origin), "status": "prepared; no PBS submitted, budget not started"}, indent=2))
    return plan
