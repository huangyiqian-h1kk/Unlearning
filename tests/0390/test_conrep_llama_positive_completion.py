"""Cross-model parity, historical reuse, immutable bootstrap and shared queue."""

import importlib.util
from pathlib import Path
import subprocess
import time

import pytest

from conrep.night import campaign as c, followup as f, positive_grid as p, insertion_grid as i
from conrep.night import llama_positive_completion as g
from conrep.night.io import read, write, file_sha

spec = importlib.util.spec_from_file_location("completion_fixtures", Path(__file__).with_name("test_conrep_positive_grid.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def completed_parent(tmp_path, **kwargs):
    root, source, ref = helpers.completed_parent(tmp_path, **kwargs)
    plan = read(source / "plan.json")
    base = read(next(t["config"] for t in plan["tasks"] if t["variant"] == "M"))
    for variant in ("G256B32W2P1", "G256B32W2P4"):
        helpers.add_control(source, base, "llama8b", variant)
    return root, source, ref


def test_eighteen_exact_mirrors_without_new_controls_or_seed_repeats(tmp_path):
    tasks = g.specs()
    assert f.specs(g.PROFILE) == tasks
    assert len(tasks) == len({t["id"] for t in tasks}) == 18
    assert all(t["model"] == "llama8b" and t["seed"] == 42 for t in tasks)
    assert sum(t["priority"] == 0 for t in tasks) == 8
    assert all(t["priority"] == 0 for t in tasks[:8])
    forget = [t for t in tasks if t.get("label", "").startswith("GF")]
    assert {(t["grid"]["forget_corruption"], t["grid"]["forget_views"]) for t in forget} == {
        (.5, 4), (.5, 8), (.7, 8), (.9, 4), (.9, 8)}
    replacement = [t for t in tasks if t.get("label", "").startswith("GR")]
    assert {(t["grid"]["retain_views"], t["grid"]["retain_noise"]) for t in replacement} == {
        (1, .1), (1, .2), (4, .1), (4, .2)}
    insertions = [t for t in tasks if "insertion_mode" in t["grid"]]
    assert {(t["grid"]["retain_views"], t["grid"]["insertion_mode"]) for t in insertions} == {
        (views, mode) for views in (2, 3, 4) for mode in ("fixed1", "fixed2", "binomial2p20")}
    old_llama = {t["id"] for t in p.specs() if t["model"] == "llama8b"}
    assert not old_llama & {t["id"] for t in tasks}
    base = {"run": {}, "lora": {}, "conrep": {}, "unlearn": {"learning_rate": 1e-5,
        "batch_sizes": {"forget": 8, "retain": 16, "general": 32}}}
    mirrors = {t["id"]: (t, module) for module in (p, i) for t in module.specs()}
    for task in tasks:
        mirror, module = mirrors[task["mirrored_experiment"]]
        assert g.apply_spec(base, task, tmp_path) == module.apply_spec(base, mirror, tmp_path)


def test_bootstrap_reuses_six_controls_and_audits_actual_llama_tokenizer(tmp_path):
    root, source, ref = completed_parent(tmp_path)
    before = {str(p): file_sha(p) for p in source.rglob("*") if p.is_file()}
    target = tmp_path / "completion"
    helpers.helpers.bootstrap.prepare(root, source, target, ref, profile=g.PROFILE)
    assert before == {str(p): file_sha(p) for p in source.rglob("*") if p.is_file()}
    plan, state = read(target / "plan.json"), read(target / "state.json")
    assert len(plan["tasks"]) == 18 and plan["workers"] == 3 and plan["world_size"] == 8
    assert plan["start_after_campaign"] == str(source)
    assert plan["start_after_plan_sha256"] == file_sha(source / "plan.json")
    assert state["started_at"] is None and state["deadline"] is None
    assert plan["queue"] == "R9920261000" and plan["recovery_policy"]["interruption_retry_seconds"] == 1200
    design = read(target / "llama-positive-completion-design.json")
    assert len(design["historical_controls"]) == 6
    assert {(x["model"], x["variant"]) for x in design["historical_controls"]} == set(g.CONTROLS)
    for task in plan["tasks"]:
        cfg = read(task["config"])
        assert cfg["unlearn"]["checkpoint"] == str(tmp_path / "sft")
        assert cfg["unlearn"]["batch_sizes"] == {"forget": 8, "retain": 32, "general": 32}
        assert cfg["conrep"]["negative_views"] == 4 and cfg["conrep"]["specified_negative_views"] == 1
        assert cfg["conrep"]["forget_cl_weight"] == 2
        assert cfg["evaluation"]["scoring"] == "pmc-ia-v5" and len(c.expected_checkpoints(cfg)) == 13
    for name in ("retain-noise-audit.json", "retain-insertion-audit.json"):
        audit = read(target / name)
        assert set(audit) == {"llama8b"}
        assert audit["llama8b"]["audited_rows"] == 900 and audit["llama8b"]["unsupported_rows"] == 0
    assert file_sha(target / "code/src/experiments/validation.py") == file_sha(source / "code/src/experiments/validation.py")
    smoke = [read(c.smoke_task(plan, worker)["config"])["conrep"] for worker in range(3)]
    assert smoke[0]["views"] == 8 and smoke[0]["corruption_rate"] == .9
    assert smoke[1]["specified_views"] == 4 and smoke[1]["specified_noise_probability"] == .2
    assert smoke[2]["specified_views"] == 4 and smoke[2]["specified_insertion_mode"] == "fixed2"
    state["tasks"][plan["tasks"][0]["id"]]["status"] = "completed"
    write(target / "state.json", state)
    helpers.helpers.bootstrap.prepare(root, source, target, ref, profile=g.PROFILE)
    assert state == read(target / "state.json")
    c.verify_snapshot(target, plan)
    write(target / "llama-positive-completion-design.json", {})
    with pytest.raises(ValueError, match="Frozen input changed"):
        c.verify_snapshot(target, plan)


@pytest.mark.parametrize("defect", ["missing_control", "missing_validation", "grammar", "duplicate"])
def test_preflight_blocks_before_plan_or_pbs(tmp_path, defect):
    root, source, ref = completed_parent(tmp_path, bad_noise=defect == "grammar")
    plan = read(source / "plan.json")
    if defect == "missing_control":
        plan["tasks"] = [t for t in plan["tasks"] if t["variant"] != "G256B32W2P4" or t["model"] != "llama8b"]
        write(source / "plan.json", plan)
    if defect == "missing_validation":
        task = next(t for t in plan["tasks"] if t["variant"] == "G256B32W2P4" and t["model"] == "llama8b")
        (Path(task["output"]) / "validation/checkpoint-125/NIGHT_VALIDATED.json").unlink()
    if defect == "duplicate":
        plan["tasks"].append(g.specs()[0])
        write(source / "plan.json", plan)
    target = tmp_path / "completion"
    with pytest.raises(subprocess.CalledProcessError):
        helpers.helpers.bootstrap.prepare(root, source, target, ref, profile=g.PROFILE)
    assert not (target / "plan.json").exists()
    if defect == "grammar":
        assert read(target / "retain-noise-audit.json")["llama8b"]["unsupported_rows"] == 1


def test_three_workers_claim_all_tasks_once_in_priority_order(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    write(config, {})
    tasks = [dict(t, config=str(config)) for t in g.specs()]
    now = time.time()
    plan = {"tasks": tasks, "initial_task_seconds": 3600}
    write(tmp_path / "state.json", {"status": "running", "deadline": now + 36000,
        "tasks": {t["id"]: {"status": "pending", "attempts": 0} for t in tasks},
        "workers": {str(i): {} for i in range(3)}})
    monkeypatch.setattr(c, "training_done", lambda *args: False)
    claimed = [c.claim(tmp_path, plan, worker % 3, now + 43200) for worker in range(18)]
    assert len({t["id"] for t in claimed}) == 18
    assert [t["priority"] for t in claimed] == [0] * 8 + [1] * 10
    assert c.claim(tmp_path, plan, 0, now + 43200) is None
