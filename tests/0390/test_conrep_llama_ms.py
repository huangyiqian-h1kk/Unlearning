"""Llama M-S launch/identity checks; no torch, GPU, network, or PBS required."""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time

import pytest

from conrep.night import campaign as c, followup as f
from conrep.night.io import read, write, file_sha, sha

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "llama_ms_bootstrap", ROOT / "scripts/abci/0390_prepare_conrep_followup.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def frozen_parent(tmp_path, *, bad_retain=False):
    root, source = tmp_path / "checkout", tmp_path / "original"
    root.mkdir()
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in bootstrap.OVERLAY:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], cwd=root, check=True)
    ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    shutil.copytree(root / "src", source / "code/src")
    for name in (c.ENTRY, c.SHELL):
        target = source / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    validator = source / "code/src/experiments/validation.py"
    validator.write_text(validator.read_text() + "\n# frozen server-only evaluator\n")
    old_controller = source / "code/src/conrep/night/campaign.py"
    old_controller.write_text(old_controller.read_text() + "\n# previous controller revision\n")
    (root / "src/experiments/validation.py").write_text("# dirty checkout must remain untouched\n")
    inputs = {}
    for group in ("forget", "retain", "general"):
        path = tmp_path / "data" / f"{group}.jsonl"
        path.parent.mkdir(exist_ok=True)
        rows = [{"id": f"{group}:{i}", "text": f"The condition of patient P{i} is asthma ."}
                for i in range(32)]
        if group == "retain" and bad_retain:
            rows[-1]["text"] = "This unsupported sentence must not be used as a Q positive."
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        inputs[str(path)] = file_sha(path)
    asset = tmp_path / "sft/fixture.bin"
    asset.parent.mkdir()
    asset.write_bytes(b"fixture asset; never loaded as a model")
    cfg = {"run": {"seed": 42}, "model": {"name_or_path": str(asset.parent)},
           "data": {"prepared_dir": str(tmp_path / "data")}, "lora": {},
           "unlearn": {"checkpoint": str(asset.parent), "learning_rate": 1e-5,
                       "batch_sizes": {"forget": 8, "retain": 16, "general": 32}},
           "conrep": {"max_length": 512, "specified_positive": "dropout"},
           "evaluation": {"partition": "validation", "limit": None}}
    tasks = []
    for model in ("gemma2_9b", "llama8b"):
        path = source / "configs" / f"{model}.json"
        write(path, cfg)
        tasks.append({"model": model, "variant": "A", "seed": 42,
                      "config": str(path), "config_hash": file_sha(path)})
    files = {str(p.relative_to(source / "code")): file_sha(p)
             for p in (source / "code").rglob("*") if p.is_file()}
    write(source / "source.json", {"files": files, "source_hash": sha(files)})
    write(source / "inputs.json", inputs)
    write(source / "model-assets.json", {str(asset): {
        "size": asset.stat().st_size, "mtime_ns": asset.stat().st_mtime_ns}})
    write(source / "plan.json", {"project_root": str(root), "source_hash": sha(files), "tasks": tasks})
    return root, source, ref


def test_exact_matrix_keeps_original_and_matches_gemma_factors():
    assert len(f.specs()) == 27
    tasks = f.specs("llama-ms")
    assert len(tasks) == len({t["id"] for t in tasks}) == 14
    assert {(t["model"], t["variant"], t["seed"]) for t in tasks} == {
        ("llama8b", variant, seed) for variant in "MNOPQRS" for seed in (42, 43)}
    assert [t["seed"] for t in tasks] == [42] * 7 + [43] * 7
    base = {"run": {}, "lora": {}, "conrep": {}, "unlearn": {"learning_rate": 1e-5}}
    for variant, (parent, overrides) in f.CHANGES.items():
        expected = c.apply_variant(base, parent, 42, "/run")
        for field, value in overrides.items():
            section, key = field.split(".")
            expected[section][key] = value
        actual = f.apply_variant(base, variant, 42, "/run")
        assert actual == expected
        assert actual["lora"]["lora_alpha"] / actual["lora"]["r"] == 2
    with pytest.raises(ValueError, match="Unknown"):
        f.specs("misspelled")


@pytest.mark.parametrize("profile, count", [("original", 27), ("llama-ms", 14)])
def test_isolated_bootstrap_preserves_parent_and_repeated_start_state(tmp_path, profile, count):
    root, source, ref = frozen_parent(tmp_path)
    target = tmp_path / "followup"
    original = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    bootstrap.prepare(root, source, target, ref, profile=profile)
    plan = read(target / "plan.json")
    assert len(plan["tasks"]) == count and plan["followup"]["profile"] == profile
    assert plan["workers"] == 3 and plan["world_size"] == 8
    assert plan["walltime"] == "12:00:00" and plan["queue"] == "R9920261000"
    assert plan["recovery_policy"]["interruption_retry_seconds"] == 1200
    assert plan["recovery_policy"]["interruption_max_retries"] is None
    assert plan["source_hash"] != read(source / "source.json")["source_hash"]
    assert file_sha(target / "code/src/experiments/validation.py") == file_sha(source / "code/src/experiments/validation.py")
    assert (root / "src/experiments/validation.py").read_text().startswith("# dirty checkout")
    assert original == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    if profile == "llama-ms":
        assert {task["model"] for task in plan["tasks"]} == {"llama8b"}
        assert set(read(target / "fact-positive-audit.json")) == {"llama8b"}
        assert read(target / "fact-positive-audit.json")["llama8b"]["unsupported_rows"] == 0
        assert {p.name for p in (target / "diagnostic-inputs").iterdir()} == {"llama8b.json"}
    for task in plan["tasks"]:
        cfg = read(task["config"])
        assert cfg["diagnostics"]["enabled"]
        assert len(c.expected_checkpoints(cfg)) == 13
        assert cfg["unlearn"]["checkpoint"] == str(tmp_path / "sft")
    state = read(target / "state.json")
    state["tasks"][plan["tasks"][0]["id"]]["status"] = "completed"
    write(target / "state.json", state)
    before = (target / "state.json").read_bytes()
    bootstrap.prepare(root, source, target, ref, profile=profile)
    assert (target / "state.json").read_bytes() == before
    c.verify_snapshot(target, plan)
    with pytest.raises(ValueError, match="different provenance"):
        bootstrap.prepare(root, source, target, ref,
                          profile="original" if profile == "llama-ms" else "llama-ms")
    args = argparse.Namespace(project_root=root, source_campaign=source, campaign=target,
                              ref=ref, profile="original" if profile == "llama-ms" else "llama-ms")
    with pytest.raises(ValueError, match="revision/profile"):
        f.prepare(args)


def test_llama_q_audit_blocks_unsupported_facts_before_publish(tmp_path):
    root, source, ref = frozen_parent(tmp_path, bad_retain=True)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.prepare(root, source, tmp_path / "followup", ref, profile="llama-ms")
    assert read(tmp_path / "followup/fact-positive-audit.json")["llama8b"]["unsupported_rows"] == 1
    assert not (tmp_path / "followup/plan.json").exists()


def test_three_workers_drain_one_pool_without_duplicate_or_gemma_tasks(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    write(config, {})
    tasks = [dict(t, config=str(config)) for t in f.specs("llama-ms")]
    now = time.time()
    plan = {"tasks": tasks, "initial_task_seconds": 3600}
    write(tmp_path / "state.json", {"status": "running", "deadline": now + 36000,
        "tasks": {t["id"]: {"status": "pending", "attempts": 0} for t in tasks},
        "workers": {str(i): {} for i in range(3)}})
    monkeypatch.setattr(c, "training_done", lambda *args: False)
    claimed = [c.claim(tmp_path, plan, i, now + 43200) for i in range(3)]
    assert [t["id"] for t in claimed] == ["llama8b-M-s42", "llama8b-N-s42", "llama8b-O-s42"]
    # Finish a worker, then let it immediately claim its next task in the same allocation.
    seen = [t["id"] for t in claimed]
    for i in range(11):
        worker = i % 3
        with c.state_transaction(tmp_path) as state:
            state["tasks"][claimed[worker]["id"]]["status"] = "completed"
        claimed[worker] = c.claim(tmp_path, plan, worker, now + 43200)
        seen.append(claimed[worker]["id"])
    assert len(seen) == len(set(seen)) == 14
    assert all(task.endswith("s42") for task in seen[:7])
    assert all(task.endswith("s43") for task in seen[7:])
    assert c.claim(tmp_path, plan, 0, now + 43200) is None
