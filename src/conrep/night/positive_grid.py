"""Approved 31-run positive/noise sweep, from the completed mixed campaign.

Four historical controls must pass provenance checks. Missing controls stop
preparation; they never silently expand this matrix or become fake completions.
"""

import copy
import itertools
import json
from pathlib import Path
import sys

from . import campaign as c
from .io import read, write, file_sha, sha, training_identity
from .grid import semantic_config

PROFILE = "positive-grid"
NAME = "0390-positive-grid-v1"
CAMPAIGN = "conrep-positive-grid-20261010"
NOISE_POLICY = "training-fact-offset-protection-v1"
CONTROLS = (("gemma2_9b", "G256B32W2P1"), ("gemma2_9b", "G256B32W2P4"),
            ("llama8b", "J"), ("llama8b", "M"))


def spec(model, rank=256, batch=32, weight=2, views=1, *, seed=42, pf=.7, kf=4,
         pr=0., priority=1, worker=0, label=None):
    variant = f"G{rank}B{batch}W{weight}P{views}"
    if label:
        variant += f"C{round(100*pf):02d}V{kf}N{round(100*pr):02d}"
    return {"id": f"{model}-{variant}-s{seed}", "model": model, "variant": variant,
        "seed": seed, "priority": priority, "preferred_worker": worker,
        "late_admission": True, "stage": "replication" if seed != 42 else "exploration",
        "label": label, "grid": {"rank": rank, "retain_batch": batch,
            "forget_weight": weight, "retain_views": views,
            "forget_corruption": pf, "forget_views": kf, "retain_noise": pr}}


def specs():
    tasks = [spec("gemma2_9b", views=p, seed=s, priority=0, worker=1 if p == 1 else 2)
             for s in (43, 44) for p in (1, 4)]
    tasks += [spec("llama8b", views=p, priority=0) for p in (1, 4)]
    forget = ((.5, 4), (.5, 8), (.7, 8), (.9, 4), (.9, 8))
    tasks += [spec("gemma2_9b", pf=p, kf=k, worker=1+i%2, label=f"GF{i+1}")
              for i, (p, k) in enumerate(forget)]
    tasks += [spec("gemma2_9b", views=k, pr=p, worker=1+i%2, label=f"GR{i+1}")
              for i, (k, p) in enumerate(((1, .1), (1, .2), (4, .1), (4, .2)))]
    for r, b, w, p in itertools.product((64, 256), (16, 32), (2, 5), (1, 4)):
        if (b, w, p) == (16, 5, 1) or (r, b, w) == (256, 32, 2):
            continue  # Historical J/M and the two priority targets above.
        tasks.append(spec("llama8b", r, b, w, p))
    tasks += [spec("llama8b", views=p, seed=s, priority=2)
              for s in (43, 44) for p in (1, 4)]
    assert len(tasks) == len({t["id"] for t in tasks}) == 31
    return tasks


def apply_spec(base, task, output):
    from .followup import apply_variant
    g = task["grid"]
    variant = f"G{g['rank']}B{g['retain_batch']}W{g['forget_weight']}P{g['retain_views']}"
    cfg = apply_variant(base, variant, task["seed"], output)
    cfg["conrep"].update(corruption_rate=g["forget_corruption"], views=g["forget_views"], negative_views=4)
    # Absent/zero noise takes the untouched legacy path, including RNG order.
    if g["retain_noise"]:
        cfg["conrep"].update(specified_noise_probability=g["retain_noise"],
            specified_noise_policy=NOISE_POLICY, specified_negative_source="clean_dropout")
    return cfg


def check_base(cfg, model):
    u, o, l = cfg["unlearn"], cfg["conrep"], cfg["lora"]
    required = {"learning_rate": 1e-5, "max_steps": 125, "save_steps": 10,
        "gradient_accumulation_steps": 1, "warmup_ratio": .05, "weight_decay": 0.,
        "max_grad_norm": 1., "max_length": 512, "gradient_checkpointing": True}
    if any(u.get(k) != v for k, v in required.items()):
        raise ValueError(f"Unapproved optimizer/step configuration in {model} control")
    batch, weight = (32, 2.) if model == "gemma2_9b" else (16, 5.)
    required = {"max_length": 512, "views": 4, "negative_views": 4,
        "corruption_rate": .7, "temperature_forget": .08, "temperature_retain": .1,
        "temperature_general": .1, "retain_negative_weight": 2., "retain_margin": .1,
        "specified_cl_weight": 1., "general_cl_weight": 1., "specified_lm_weight": 1.,
        "general_lm_weight": 1., "inter_instance_negatives": True,
        "shared_target": False, "specified_positive": "dropout", "protected_positive": False,
        "forget_cl_weight": weight}
    if (any(o.get(k) != v for k, v in required.items())
            or o.get("specified_noise_probability", 0) != 0
            or o.get("specified_views", 1) != 1 or o.get("specified_negative_views", 1) != 1
            or o.get("stop_gradient_controls", False) or o.get("stop_gradient_retain", False)
            or u["batch_sizes"] != {"forget": 8, "retain": batch, "general": 32}):
        raise ValueError(f"Unapproved contrastive configuration in {model} control")
    if (l.get("r") != 256 or l.get("lora_alpha") != 512 or l.get("lora_dropout") != .05
            or l.get("bias") != "none" or set(l.get("target_modules", [])) !=
            {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}):
        raise ValueError(f"Unapproved LoRA configuration in {model} control")
    if (cfg["run"]["seed"] != 42 or cfg["model"].get("dtype") != "bfloat16"
            or not cfg["model"].get("local_only") or cfg["evaluation"].get("partition") != "validation"
            or cfg["evaluation"].get("limit") or cfg["evaluation"].get("scoring") != "pmc-ia-v5"):
        raise ValueError(f"Expected fixed seed42 SFT/data and full v5 validation: {model}")


def historical_controls(origin, root, *, controls=CONTROLS):
    """Search only the supplied campaign, its ancestry, and the known follow-up."""
    locations, current = [], origin
    while current not in locations:
        locations.append(current)
        parent = read(current / "plan.json").get("followup", {}).get("parent_campaign")
        if not parent:
            break
        current = Path(parent).resolve()
    followup = origin.parent / "conrep-followup-20261008"
    if followup not in locations and (followup / "plan.json").exists():
        locations.append(followup)
    source = read(origin / "source.json")
    core = {n: h for n, h in source["files"].items()
            if n.startswith("src/") and not n.startswith("src/conrep/night/")}
    parent_inputs = read(origin / "inputs.json")
    records, configs = [], {}
    for model, variant in controls:
        found = None
        for location in locations:
            plan = read(location / "plan.json")
            matches = [t for t in plan["tasks"] if (t["model"], t["variant"], t["seed"]) == (model, variant, 42)]
            if matches:
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous historical control {model}/{variant}")
                found = location, plan, matches[0]
                break
        if found is None:
            raise ValueError(f"Missing historical control {model}/{variant}/42; no additional runs authorised")
        location, plan, task = found
        cfg = read(task["config"])
        provenance = read(location / "source.json")
        if (Path(plan["project_root"]).resolve() != root or file_sha(task["config"]) != task["config_hash"]
                or training_identity(cfg) != task["identity"]
                or provenance["source_hash"] != plan["source_hash"]
                or sha(provenance["files"]) != provenance["source_hash"]):
            raise ValueError(f"Historical control identity mismatch: {task['id']}")
        for name, digest in provenance["files"].items():
            if file_sha(location / "code" / name) != digest:
                raise ValueError(f"Historical source changed: {location}/{name}")
        if {n: h for n, h in provenance["files"].items() if n.startswith("src/")
                and not n.startswith("src/conrep/night/")} != core:
            raise ValueError(f"Historical core/evaluator differs: {task['id']}")
        control_inputs = read(location / "inputs.json")
        for path in c.input_files(cfg):
            name = str(path)
            if name not in parent_inputs or control_inputs.get(name) != parent_inputs[name] or file_sha(path) != parent_inputs[name]:
                raise ValueError(f"Historical data/probe mismatch: {name}")
        evidence = {}
        complete_path = Path(task["output"]) / "training/TRAINING_COMPLETE.json"
        complete = read(complete_path)
        if complete.get("identity") != task["identity"] or complete.get("steps") != 125:
            raise ValueError(f"Historical training completion mismatch: {task['id']}")
        evidence[str(complete_path)] = file_sha(complete_path)
        if c.expected_checkpoints(cfg) != list(range(10, 121, 10)) + [125]:
            raise ValueError("Historical control has a different checkpoint schedule")
        for step in c.expected_checkpoints(cfg):
            directory = Path(task["output"]) / "validation" / f"checkpoint-{step}"
            if not c.validation_done(task, step):
                raise ValueError(f"Historical control lacks verified validation: {task['id']}/{step}")
            marker = read(directory / "NIGHT_VALIDATED.json")
            report = read(directory / "metrics.json")
            if (marker.get("protocol_hash") != report.get("protocol_hash")
                    or (cfg["evaluation"].get("mmlu_file") and "mmlu_predictions.jsonl" not in marker["prediction_sizes"])):
                raise ValueError(f"Historical validation protocol/predictions incomplete: {task['id']}/{step}")
            for name in ("metrics.json", "NIGHT_VALIDATED.json"):
                evidence[str(directory / name)] = file_sha(directory / name)
        records.append({"model": model, "variant": variant, "seed": 42, "reuse": True,
            "campaign": str(location), "experiment": task["id"], "config": task["config"],
            "config_hash": task["config_hash"], "identity": task["identity"],
            "source_hash": provenance["source_hash"], "semantic_config_hash": sha(semantic_config(cfg)),
            "validated_steps": c.expected_checkpoints(cfg), "evidence_hashes": evidence})
        configs[model, variant] = cfg
    bases = {"gemma2_9b": configs["gemma2_9b", "G256B32W2P1"], "llama8b": configs["llama8b", "M"]}
    for model, base in bases.items():
        check_base(base, model)
    for model, variant in controls:
        if variant == "J":
            control = spec(model, 64, 16, 5, 1)
        elif variant == "M":
            control = spec(model, 256, 16, 5, 1)
        else:
            control = spec(model, views=4 if variant.endswith("P4") else 1)
        expected = apply_spec(bases[model], control, "/unused")
        actual = configs[model, variant]
        if semantic_config(actual) != semantic_config(expected) or actual["evaluation"] != expected["evaluation"]:
            raise ValueError(f"Incompatible historical training/evaluation config: {model}/{variant}")
    return records, bases


def prepare(args):
    from experiments.config import read_rows
    from conrep.v2.model import load_tokenizer
    from .noise import audit_rows
    root, origin, campaign = (Path(p).resolve() for p in (args.project_root, args.source_campaign, args.campaign))
    if campaign == origin:
        raise ValueError("Positive grid requires a new campaign")
    if (campaign / "plan.json").exists():
        plan = read(campaign / "plan.json")
        if plan.get("followup", {}).get("ref") != args.ref or plan["followup"].get("profile") != PROFILE:
            raise ValueError("Existing campaign uses another revision/profile")
        c.verify_snapshot(campaign, plan)
        print("Positive grid already prepared; no state reset or PBS submitted.")
        return plan
    metadata = read(campaign / "PREPARING.json")
    if (metadata["ref"] != args.ref or metadata["source_campaign"] != str(origin)
            or metadata.get("profile") != PROFILE):
        raise ValueError("Bootstrap provenance mismatch")
    source = read(origin / "source.json")
    inputs, assets = read(origin / "inputs.json"), read(origin / "model-assets.json")
    for name, expected in inputs.items():
        if file_sha(name) != expected:
            raise ValueError(f"Parent input changed: {name}")
    for name, expected in assets.items():
        stat = Path(name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Parent model asset changed: {name}")
    controls, bases = historical_controls(origin, root)
    for control in controls:
        inputs[control["config"]] = control["config_hash"]
        inputs.update(control["evidence_hashes"])
    diagnostics, counts = {}, {}
    for model, cfg in bases.items():
        collections = {g: read_rows(Path(cfg["data"]["prepared_dir"]) / f"{g}.jsonl")
                       for g in ("forget", "retain", "general")}
        counts[model] = {g: len(rows) for g, rows in collections.items()}
        if counts[model]["forget"] != 100 or counts[model]["retain"] != 900:
            raise ValueError("Expected the approved 100-forget/900-retain training split")
        old = cfg["diagnostics"]
        probe = campaign / "diagnostic-inputs" / f"{model}.json"
        if file_sha(old["probe_file"]) != old["probe_sha256"] or inputs.get(old["probe_file"]) != old["probe_sha256"]:
            raise ValueError("Parent fixed training probe changed")
        write(probe, read(old["probe_file"]))
        inputs[str(probe)] = file_sha(probe)
        diagnostics[model] = dict(old, probe_file=str(probe), probe_sha256=file_sha(probe), sampling_coverage=True)
        if model == "gemma2_9b":
            tokenizer = load_tokenizer(cfg["unlearn"]["checkpoint"], True)
            embedding_size = read(Path(cfg["unlearn"]["checkpoint"]) / "config.json")["vocab_size"]
            audit = audit_rows(tokenizer, collections["retain"], cfg["conrep"]["max_length"], embedding_size)
            path = campaign / "retain-noise-audit.json"
            write(path, {model: audit})
            if audit["unsupported_rows"] or not audit["eligible_tokens"]:
                raise ValueError("Retain noise audit failed; inspect retain-noise-audit.json before any submission")
            inputs[str(path)] = file_sha(path)
    frozen = campaign / "code"
    files = {str(p.relative_to(frozen)): file_sha(p) for p in sorted(frozen.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    # Verify the bootstrap copied the server helpers verbatim and applied only
    # the pinned whitelist. This also makes direct invocation fail safely.
    expected_files = dict(source["files"], **metadata["overlay_files"])
    if files != expected_files:
        raise ValueError("Prepared source differs from the verified parent plus pinned overlay")
    source_hash = sha(files)
    design = {"schema": NAME, "new_runs": 31, "gemma_runs": 13, "llama_runs": 18,
        "historical_controls": controls, "training_rows": counts,
        "forget_negative_views": 4, "retain_negative_views": 1,
        "noise_policy": NOISE_POLICY, "noise_negative_source": "clean_dropout",
        "zero_noise_path": "legacy dropout positives also supply the fixed negative view",
        "primary_step": 125, "common_step": 120, "all_steps": list(range(10, 121, 10)) + [125],
        "seed_scope": "unlearning only; original seed42 SFT and prepared data fixed",
        "epoch_scope": "125*8/100=10 forget-equivalent epochs; sampled draws, not guaranteed passes",
        "automatic_selection": False, "test_evaluation": False,
        "adaptive_expansion": False, "sampler_sha256": files["src/experiments/data.py"]}
    write(campaign / "positive-grid-design.json", design)
    inputs[str(campaign / "positive-grid-design.json")] = file_sha(campaign / "positive-grid-design.json")
    write(campaign / "source.json", {"files": files, "source_hash": source_hash,
        "parent_source_hash": source["source_hash"], "overlay_ref": args.ref,
        "overlay_files": metadata["overlay_files"], "server_evaluator_preserved": True})
    write(campaign / "inputs.json", inputs)
    write(campaign / "model-assets.json", assets)
    tasks = []
    for task in specs():
        directory = campaign / "experiments" / task["id"]
        cfg = apply_spec(bases[task["model"]], task, directory / "training")
        cfg["diagnostics"] = copy.deepcopy(diagnostics[task["model"]])
        cfg["night"] = {"source_hash": source_hash, "data_hashes": inputs,
            "model_assets_hash": sha(assets), "campaign": str(campaign),
            "experiment": task["id"], "parent_campaign": str(origin)}
        path = campaign / "configs" / (task["id"] + ".json")
        write(path, cfg)
        tasks.append(dict(task, config=str(path), config_hash=file_sha(path),
                          output=str(directory), identity=training_identity(cfg)))
    smoke = {"0": next(t["id"] for t in tasks if t["label"] == "GR4"),
             "1": next(t["id"] for t in tasks if t["label"] == "GF5"),
             "2": "llama8b-G256B32W5P4-s42"}
    plan = {"schema": c.NAME, "project_root": str(root), "campaign": str(campaign),
        "python": sys.executable, "entry": str(frozen / c.ENTRY), "shell": str(frozen / c.SHELL),
        "controller_entry": str(frozen / c.ENTRY), "controller_shell": str(frozen / c.SHELL),
        "workers": 3, "world_size": 8, "queue": "R9920261000", "account": "gcg51557", "rtype": "rt_HF",
        "hours": args.hours, "walltime": "12:00:00", "poll_seconds": 60,
        "max_attempts": 3, "recovery_policy": dict(c.RECOVERY_DEFAULTS),
        "source_hash": source_hash, "tasks": tasks, "initial_task_seconds": 3600,
        "reserve_bytes": int(args.reserve_gb * 10**9), "smoke_tasks": smoke,
        "followup": {"version": NAME, "profile": PROFILE, "ref": args.ref,
                     "parent_campaign": str(origin), "parent_source_hash": source["source_hash"]}}
    write(campaign / "state.json", {"status": "prepared", "started_at": None, "deadline": None,
        "tasks": {t["id"]: {"status": "pending", "attempts": 0, "failures": 0} for t in tasks},
        "workers": {str(i): {"job_id": None, "status": "new", "allocations": 0,
                              "recovery_failures": 0} for i in range(3)}})
    write(campaign / "plan.json", plan)
    print(json.dumps({"campaign": str(campaign), "tasks": len(tasks), "gemma": 13, "llama": 18,
        "historical_controls": 4, "workers": 3, "smoke_tasks": smoke,
        "budget_hours": args.hours, "status": "prepared; no PBS submitted"}, indent=2))
    return plan
