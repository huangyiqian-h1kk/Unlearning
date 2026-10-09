"""Exact approved scope, old-result verification, isolated prepare and dispatch."""

import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import time

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from conrep.night import campaign as c, followup as f, positive_grid as p
from conrep.night.io import read, write, file_sha, training_identity

spec = importlib.util.spec_from_file_location("positive_fixtures", Path(__file__).with_name("test_conrep_llama_ms.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def test_exact_31_runs_priority_and_factorial_completion():
    tasks = p.specs()
    assert f.specs("positive-grid") == tasks
    assert len(tasks) == len({t["id"] for t in tasks}) == 31
    assert sum(t["model"] == "gemma2_9b" for t in tasks) == 13
    assert sum(t["model"] == "llama8b" for t in tasks) == 18
    gemma = [t for t in tasks if t["model"] == "gemma2_9b" and t["seed"] == 42]
    assert {(g["forget_corruption"], g["forget_views"], g["retain_views"], g["retain_noise"])
            for g in [t["grid"] for t in gemma]} == {
        (.5, 4, 1, 0), (.5, 8, 1, 0), (.7, 8, 1, 0), (.9, 4, 1, 0), (.9, 8, 1, 0),
        (.7, 4, 1, .1), (.7, 4, 1, .2), (.7, 4, 4, .1), (.7, 4, 4, .2)}
    cells = {tuple(t["grid"][k] for k in ("rank", "retain_batch", "forget_weight", "retain_views"))
             for t in tasks if t["model"] == "llama8b" and t["seed"] == 42}
    assert len(cells) == 14
    assert cells | {(64, 16, 5, 1), (256, 16, 5, 1)} == set(itertools.product((64, 256), (16, 32), (2, 5), (1, 4)))
    assert sum(t["priority"] == 0 for t in tasks) == 6
    assert sum(t["priority"] == 1 for t in tasks) == 21
    assert sum(t["priority"] == 2 for t in tasks) == 4
    assert not any(t["variant"] in "MNOPQRS" for t in tasks)


def add_control(source, base, model, variant):
    plan = read(source / "plan.json")
    directory = source / "experiments" / f"{model}-{variant}-s42"
    path = source / "configs" / (directory.name + ".json")
    cfg = f.apply_variant(base, variant, 42, directory / "training")
    write(path, cfg)
    task = dict(id=directory.name, model=model, variant=variant, seed=42,
        config=str(path), config_hash=file_sha(path), output=str(directory), identity=training_identity(cfg))
    plan["tasks"].append(task)
    write(source / "plan.json", plan)
    write(directory / "training/TRAINING_COMPLETE.json", {"identity": task["identity"], "steps": 125})
    for step in c.expected_checkpoints(cfg):
        target = directory / "validation" / f"checkpoint-{step}"
        write(target / "metrics.json", {"metrics": {"forget.qa": .5}, "protocol_hash": "v5"})
        for name in ("predictions.jsonl", "mmlu_predictions.jsonl"):
            (target / name).write_text('{}\n')
        write(target / "NIGHT_VALIDATED.json", {"identity": task["identity"], "protocol_hash": "v5",
            "metrics_sha256": file_sha(target / "metrics.json"),
            "prediction_sizes": {"predictions.jsonl": 3, "mmlu_predictions.jsonl": 3}})
    return task


def completed_parent(tmp_path, bad_noise=False):
    root, source, ref = helpers.frozen_parent(tmp_path)
    base = read(source / "configs/gemma2_9b.json")
    inputs = read(source / "inputs.json")
    for group, count in (("forget", 100), ("retain", 900), ("general", 40)):
        path = tmp_path / "data" / f"{group}.jsonl"
        rows = [{"id": f"{group}:{i}", "text": f"The condition of patient P{i} is asthma ."} for i in range(count)]
        if group == "retain" and bad_noise:
            rows[-1]["text"] = "Unknown fact grammar must stop preparation"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        inputs[str(path)] = file_sha(path)
    words = "[PAD] [UNK] The condition of patient is asthma .".split()
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    tok.save_pretrained(tmp_path / "sft")
    write(tmp_path / "sft/config.json", {"model_type": "llama", "vocab_size": len(tok)})
    write(source / "model-assets.json", {str(path): {"size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns} for path in (tmp_path / "sft").iterdir()})
    base["model"].update(local_only=True, dtype="bfloat16", attention="eager")
    base["unlearn"].update(learning_rate=1e-5, max_steps=125, save_steps=10, gradient_accumulation_steps=1,
        warmup_ratio=.05, weight_decay=0., max_grad_norm=1., max_length=512, gradient_checkpointing=True)
    base["conrep"].update(temperature_forget=.08, temperature_retain=.1, temperature_general=.1,
        retain_negative_weight=2., retain_margin=.1, inter_instance_negatives=True, shared_target=False)
    base["lora"].update(bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    base["evaluation"].update(scoring="pmc-ia-v5", mmlu_file=str(tmp_path / "data/general.jsonl"))
    for group in ("forget", "retain"):
        base["data"][group + "_generation"] = str(tmp_path / "data" / (group + ".jsonl"))
        base["data"][group + "_mcq"] = {}
    probe = source / "diagnostic-inputs/fixed.json"
    write(probe, {"schema": "training-only-fixed-probes-v1", "rows": []})
    inputs[str(probe)] = file_sha(probe)
    base["diagnostics"] = {"enabled": True, "every_steps": 25, "probe_file": str(probe),
        "probe_sha256": file_sha(probe), "sampling_coverage": True, "seed": 39027}
    write(source / "inputs.json", inputs)
    for model, variant in p.CONTROLS[:-1]:
        if variant != "J":
            add_control(source, base, model, variant)
    add_control(source, base, "llama8b", "M")
    previous = source.parent / "conrep-followup-20261008"
    shutil.copytree(source, previous)
    old = read(previous / "plan.json")
    old["tasks"] = []
    write(previous / "plan.json", old)
    add_control(previous, base, "llama8b", "J")
    return root, source, ref


def test_bootstrap_preserves_frozen_evaluator_exact_scope_and_idempotence(tmp_path):
    root, source, ref = completed_parent(tmp_path)
    original = {str(p.relative_to(source)): file_sha(p) for p in source.rglob("*") if p.is_file()}
    target = tmp_path / "positive"
    helpers.bootstrap.prepare(root, source, target, ref, profile="positive-grid")
    plan = read(target / "plan.json")
    assert len(plan["tasks"]) == 31 and plan["workers"] == 3 and plan["world_size"] == 8
    assert plan["queue"] == "R9920261000" and plan["walltime"] == "12:00:00" and plan["hours"] == 10
    assert plan["recovery_policy"]["interruption_retry_seconds"] == 1200
    assert plan["recovery_policy"]["interruption_max_retries"] is None
    assert original == {str(p.relative_to(source)): file_sha(p) for p in source.rglob("*") if p.is_file()}
    assert file_sha(target / "code/src/experiments/validation.py") == file_sha(source / "code/src/experiments/validation.py")
    design = read(target / "positive-grid-design.json")
    assert len(design["historical_controls"]) == 4 and all(x["reuse"] for x in design["historical_controls"])
    audit = read(target / "retain-noise-audit.json")["gemma2_9b"]
    assert audit["audited_rows"] == audit["eligible_rows"] == 900
    for task in plan["tasks"]:
        cfg = read(task["config"])
        assert cfg["unlearn"]["checkpoint"] == str(tmp_path / "sft")
        assert cfg["unlearn"]["learning_rate"] == 1e-5
        assert cfg["conrep"]["negative_views"] == 4 and cfg["conrep"]["specified_negative_views"] == 1
        assert cfg["conrep"].get("specified_noise_probability", 0) == task["grid"]["retain_noise"]
        assert not cfg["conrep"].get("stop_gradient_controls", False)
        assert cfg["diagnostics"]["sampling_coverage"] and len(c.expected_checkpoints(cfg)) == 13
    smoke = [read(c.smoke_task(plan, i)["config"])["conrep"] for i in range(3)]
    assert smoke[0]["specified_views"] == 4 and smoke[0]["specified_noise_probability"] == .2
    assert smoke[1]["views"] == 8 and smoke[2]["specified_views"] == 4
    state = read(target / "state.json")
    state["tasks"][plan["tasks"][0]["id"]]["status"] = "completed"
    write(target / "state.json", state)
    helpers.bootstrap.prepare(root, source, target, ref, profile="positive-grid")
    assert read(target / "state.json") == state
    c.verify_snapshot(target, plan)
    # The reuse audit is frozen too, not just the newly generated configurations.
    write(target / "positive-grid-design.json", {})
    with pytest.raises(ValueError, match="Frozen input changed"):
        c.verify_snapshot(target, plan)


@pytest.mark.parametrize("defect", ["missing", "validation", "core", "config"])
def test_incompatible_control_stops_without_adding_runs(tmp_path, defect):
    root, source, ref = completed_parent(tmp_path)
    previous = source.parent / "conrep-followup-20261008"
    plan = read(previous / "plan.json")
    task = plan["tasks"][0]
    if defect == "missing":
        plan["tasks"] = []
        write(previous / "plan.json", plan)
    elif defect == "validation":
        (Path(task["output"]) / "validation/checkpoint-125/mmlu_predictions.jsonl").unlink()
    elif defect == "core":
        path = previous / "code/src/experiments/data.py"
        path.write_text(path.read_text() + "\n# changed sampler\n")
    else:
        cfg = read(task["config"])
        cfg["conrep"]["temperature_retain"] = .2
        write(task["config"], cfg)
    target = tmp_path / "positive"
    with pytest.raises(subprocess.CalledProcessError):
        helpers.bootstrap.prepare(root, source, target, ref, profile="positive-grid")
    assert not (target / "plan.json").exists()
    assert not (target / "state.json").exists()


def test_noise_grammar_failure_writes_audit_before_any_plan(tmp_path):
    root, source, ref = completed_parent(tmp_path, bad_noise=True)
    target = tmp_path / "positive"
    with pytest.raises(subprocess.CalledProcessError):
        helpers.bootstrap.prepare(root, source, target, ref, profile="positive-grid")
    assert read(target / "retain-noise-audit.json")["gemma2_9b"]["unsupported_rows"] == 1
    assert not (target / "plan.json").exists()


def test_three_workers_share_priorities_and_admit_all_31_once(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    write(config, {})
    tasks = [dict(t, config=str(config)) for t in p.specs()]
    now = time.time()
    plan = {"tasks": tasks, "initial_task_seconds": 3600}
    write(tmp_path / "state.json", {"status": "running", "deadline": now + 36000,
        "tasks": {t["id"]: {"status": "pending", "attempts": 0} for t in tasks},
        "workers": {str(i): {} for i in range(3)}})
    monkeypatch.setattr(c, "training_done", lambda *args: False)
    claimed = [c.claim(tmp_path, plan, i, now + 43200) for i in range(3)]
    seen = list(claimed)
    for i in range(28):
        worker = (2, 0, 1)[i % 3]
        with c.state_transaction(tmp_path) as state:
            state["tasks"][claimed[worker]["id"]]["status"] = "completed"
        claimed[worker] = c.claim(tmp_path, plan, worker, now + 43200)
        seen.append(claimed[worker])
    assert len(seen) == len({t["id"] for t in seen}) == 31
    assert [t["priority"] for t in seen] == [0]*6 + [1]*21 + [2]*4
    assert c.claim(tmp_path, plan, 0, now + 43200) is None
