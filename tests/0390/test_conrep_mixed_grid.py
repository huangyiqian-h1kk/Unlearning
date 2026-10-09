"""Mixed admission, provenance, actual coverage and control reuse (CPU only)."""

import importlib.util
from pathlib import Path
import shutil
import time

import pytest

from conrep.night import campaign as c, followup as f
from conrep.night.grid import historical_controls
from conrep.night.io import read, write, file_sha
from conrep.night.sampling import SamplingAudit

spec = importlib.util.spec_from_file_location("mixed_fixtures", Path(__file__).with_name("test_conrep_llama_ms.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def test_factorial_and_llama_completion_are_exact_and_interleaved():
    tasks = f.specs("mixed-grid")
    assert len(tasks) == len({t["id"] for t in tasks}) == 28
    llama = [t for t in tasks if t["model"] == "llama8b"]
    assert {(t["variant"], t["seed"]) for t in llama} == {(v, s) for v in "MNOPQRS" for s in (42, 43)}
    gemma = [t for t in tasks if t["model"] == "gemma2_9b"]
    cells = {(t["grid"]["rank"], t["grid"]["retain_batch"], t["grid"]["forget_weight"], t["grid"]["retain_views"]) for t in gemma}
    controls = {(64, 16, 5, 1), (256, 16, 5, 1)}
    assert cells | controls == {(r, b, w, p) for r in (64, 256) for b in (16, 32) for w in (2, 5) for p in (1, 4)}
    assert not cells & controls
    for priority in range(7):
        wave = [t for t in tasks if t["priority"] == priority]
        assert [t["model"] for t in wave] == ["llama8b", "gemma2_9b", "gemma2_9b"]
        assert [t["preferred_worker"] for t in wave] == [0, 1, 2]


def test_grid_changes_only_approved_training_factors():
    base = {"run": {}, "model": {}, "data": {}, "lora": {}, "conrep": {},
            "unlearn": {"batch_sizes": {"forget": 8, "retain": 16, "general": 32}, "learning_rate": 1e-5}}
    for task in f.specs("mixed-grid"):
        cfg = f.apply_variant(base, task["variant"], task["seed"], "/run")
        if task["model"] != "gemma2_9b":
            continue
        grid = task["grid"]
        expected = f.apply_variant(base, "J" if grid["rank"] == 64 else "M", 42, "/run")
        expected["unlearn"]["batch_sizes"]["retain"] = grid["retain_batch"]
        expected["conrep"].update(forget_cl_weight=float(grid["forget_weight"]),
            specified_views=grid["retain_views"], specified_negative_views=1, specified_positive="dropout")
        assert cfg == expected
    assert base["unlearn"]["batch_sizes"]["retain"] == 16


def test_free_workers_take_both_models_and_drain_without_duplicates(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    write(config, {})
    tasks = [dict(t, config=str(config)) for t in f.specs("mixed-grid")]
    now = time.time()
    plan = {"tasks": tasks, "initial_task_seconds": 3600}
    write(tmp_path / "state.json", {"status": "running", "deadline": now + 36000,
        "tasks": {t["id"]: {"status": "pending", "attempts": 0} for t in tasks},
        "workers": {str(i): {} for i in range(3)}})
    monkeypatch.setattr(c, "training_done", lambda *args: False)
    active = [c.claim(tmp_path, plan, i, now + 43200) for i in range(3)]
    assert [t["id"] for t in active] == ["llama8b-M-s42", "gemma2_9b-G64B32W2P4-s42", "gemma2_9b-G256B32W2P4-s42"]
    seen = [t["id"] for t in active]
    for i in range(25):
        # Deliberately finish in a different order; preferences cannot reserve nodes.
        worker = (2, 0, 1)[i % 3]
        with c.state_transaction(tmp_path) as state:
            state["tasks"][active[worker]["id"]]["status"] = "completed"
        active[worker] = c.claim(tmp_path, plan, worker, now + 43200)
        seen.append(active[worker]["id"])
    assert len(seen) == len(set(seen)) == 28
    assert all(s.endswith("s42") for s in seen[:21])
    assert all(s.startswith("llama8b-") and s.endswith("s43") for s in seen[21:])
    assert c.claim(tmp_path, plan, 0, now + 43200) is None


def add_control(source, cfg, variant):
    plan = read(source / "plan.json")
    directory = source / "experiments" / f"gemma2_9b-{variant}-s42"
    config = source / "configs" / f"gemma2_9b-{variant}-s42.json"
    write(config, f.apply_variant(cfg, variant, 42, directory / "training"))
    task = {"id": directory.name, "model": "gemma2_9b", "variant": variant, "seed": 42,
        "config": str(config), "config_hash": file_sha(config), "output": str(directory), "identity": "control-" + variant}
    plan["tasks"].append(task)
    write(source / "plan.json", plan)
    for step in c.expected_checkpoints(read(config)):
        output = directory / "validation" / f"checkpoint-{step}"
        write(output / "metrics.json", {"metrics": {"forget.qa": .5}, "protocol_hash": "p"})
        (output / "predictions.jsonl").write_text('{}\n')
        write(output / "NIGHT_VALIDATED.json", {"identity": task["identity"],
            "metrics_sha256": file_sha(output / "metrics.json"), "prediction_sizes": {"predictions.jsonl": 3}})
    return task


def test_mixed_bootstrap_reuses_compatible_controls_and_falls_back_when_missing(tmp_path):
    root, source, ref = helpers.frozen_parent(tmp_path)
    base = read(source / "configs/gemma2_9b.json")
    add_control(source, base, "J")
    previous = source.parent / "conrep-followup-20261008"
    shutil.copytree(source, previous)
    old = read(previous / "plan.json")
    old["tasks"] = []
    write(previous / "plan.json", old)
    control = add_control(previous, base, "M")
    target = tmp_path / "mixed"
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    helpers.bootstrap.prepare(root, source, target, ref, profile="mixed-grid")
    plan = read(target / "plan.json")
    assert len(plan["tasks"]) == 28 and plan["workers"] == 3
    assert all(read(t["config"])["diagnostics"]["sampling_coverage"] for t in plan["tasks"])
    assert all(x["reuse"] for x in read(target / "grid-design.json")["historical_controls"])
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    saved = (target / "state.json").read_bytes()
    helpers.bootstrap.prepare(root, source, target, ref, profile="mixed-grid")
    assert (target / "state.json").read_bytes() == saved
    # An incomplete historical control is scheduled afresh, never labelled done.
    (Path(control["output"]) / "validation/checkpoint-125/NIGHT_VALIDATED.json").unlink()
    second = tmp_path / "mixed-with-control"
    helpers.bootstrap.prepare(root, source, second, ref, profile="mixed-grid")
    plan = read(second / "plan.json")
    assert len(plan["tasks"]) == 29
    assert [t["variant"] for t in plan["tasks"] if t["stage"] == "control"] == ["G256B16W5P1"]
    assert all(s["status"] == "pending" for s in read(second / "state.json")["tasks"].values())


def test_coverage_tracks_actual_rows_and_restores_committed_counts(tmp_path):
    data = {g: [{"id": i, "text": str(i)} for i in range(5)] for g in ("forget", "retain", "general")}
    audit = SamplingAudit(data, tmp_path, "id")
    first = {g: [rows[1], rows[4]] for g, rows in data.items()}
    audit.observe(first, 1, 0, gather_records=lambda x: [x, {g: [1] for g in data}])
    audit.snapshot(tmp_path / "checkpoint-1", 1)
    assert audit.counts["retain"] == [0, 2, 0, 0, 1]
    report = read(tmp_path / "sampling/step-000001-micro-0.json")
    assert report["summary"]["retain"]["coverage"] == .4
    audit.observe(first, 2, 0, gather_records=lambda x: [x])  # Lost uncommitted work.
    restored = SamplingAudit(data, tmp_path, "id", resume=tmp_path / "checkpoint-1", first_step=1)
    assert restored.counts["retain"] == [0, 2, 0, 0, 1]
    copied = {g: [dict(rows[3])] for g, rows in data.items()}
    restored.observe(copied, 2, 0, gather_records=lambda x: [x])
    assert restored.counts["retain"] == [0, 2, 0, 1, 1]
    with pytest.raises(ValueError, match="identity/step"):
        SamplingAudit(data, tmp_path, "wrong", resume=tmp_path / "checkpoint-1", first_step=1)
