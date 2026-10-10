"""Scope/provenance checks through the actual frozen-source bootstrap."""

import importlib.util
from pathlib import Path
import subprocess

import pytest

from conrep.night import insertion_grid as g, campaign as c, followup as f
from conrep.night.io import read, write, file_sha

spec = importlib.util.spec_from_file_location("insertion_grid_helpers", Path(__file__).with_name("test_conrep_positive_grid.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def test_nine_insertion_only_runs_and_preserved_frozen_parent(tmp_path):
    tasks = g.specs()
    assert f.specs("insertion-grid") == tasks
    assert len(tasks) == len({t["id"] for t in tasks}) == 9
    assert {(t["grid"]["retain_views"], t["grid"]["insertion_mode"]) for t in tasks} == {
        (p, mode) for p in (2, 3, 4) for mode in ("fixed1", "fixed2", "binomial2p20")}
    assert all(t["model"] == "gemma2_9b" and t["seed"] == 42 for t in tasks)
    root, source, ref = helpers.completed_parent(tmp_path)
    before = {str(p): file_sha(p) for p in source.rglob("*") if p.is_file()}
    target = tmp_path / "insertion"
    helpers.helpers.bootstrap.prepare(root, source, target, ref, profile="insertion-grid")
    assert before == {str(p): file_sha(p) for p in source.rglob("*") if p.is_file()}
    plan, state = read(target / "plan.json"), read(target / "state.json")
    assert len(plan["tasks"]) == 9 and plan["workers"] == 3
    assert plan["start_after_campaign"] == str(source) and plan["start_after_plan_sha256"] == file_sha(source / "plan.json")
    assert state["started_at"] is None and state["deadline"] is None
    assert plan["queue"] == "R9920261000" and plan["recovery_policy"]["interruption_retry_seconds"] == 1200
    for task in plan["tasks"]:
        cfg = read(task["config"])
        assert cfg["conrep"]["specified_noise_kind"] == "insertion"
        assert cfg["conrep"]["specified_noise_probability"] == 0
        assert cfg["conrep"]["specified_views"] in (2, 3, 4)
        assert cfg["conrep"]["specified_negative_views"] == 1
        assert cfg["conrep"]["negative_views"] == cfg["conrep"]["views"] == 4
        assert cfg["conrep"]["specified_negative_source"] == "clean_dropout"
        assert cfg["unlearn"]["max_steps"] == 125 and cfg["unlearn"]["batch_sizes"] == {"forget": 8, "retain": 32, "general": 32}
    audit = read(target / "retain-insertion-audit.json")["gemma2_9b"]
    assert audit["audited_rows"] == 900 and audit["unsupported_rows"] == 0
    assert file_sha(target / "code/src/experiments/validation.py") == file_sha(source / "code/src/experiments/validation.py")
    smokes = [read(c.smoke_task(plan, worker)["config"])["conrep"] for worker in range(3)]
    assert all(s["specified_views"] == 4 for s in smokes)
    assert {s["specified_insertion_mode"] for s in smokes} == {"fixed1", "fixed2", "binomial2p20"}
    helpers.helpers.bootstrap.prepare(root, source, target, ref, profile="insertion-grid")
    assert state == read(target / "state.json")
    c.verify_snapshot(target, plan)
    write(target / "retain-insertion-audit.json", {})
    with pytest.raises(ValueError, match="Frozen input changed"):
        c.verify_snapshot(target, plan)


def test_bad_insertion_grammar_never_publishes_a_plan(tmp_path):
    root, source, ref = helpers.completed_parent(tmp_path, bad_noise=True)
    target = tmp_path / "insertion"
    with pytest.raises(subprocess.CalledProcessError):
        helpers.helpers.bootstrap.prepare(root, source, target, ref, profile="insertion-grid")
    assert read(target / "retain-insertion-audit.json")["gemma2_9b"]["unsupported_rows"] == 1
    assert not (target / "plan.json").exists()
