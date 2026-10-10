"""Approved matrix, directional gradients, observational isolation and bootstrap."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
import torch
from safetensors.torch import load_file

from conrep.night import campaign as c, followup as f
from conrep.night.diagnostics import answer_tokens
from conrep.night.io import read, write, file_sha, sha, training_identity
from conrep.night.losses import forget_loss
from conrep.night.positives import fact_positive_candidates, fact_positive_audit
from conrep.night.trainer import run

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("followup_helpers", Path(__file__).with_name("test_conrep_night.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
spec = importlib.util.spec_from_file_location("followup_bootstrap", ROOT / "scripts/abci/0390_prepare_conrep_followup.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
torch.set_num_threads(1)


def test_exact_approved_matrix_and_single_factor_controls(tmp_path):
    tasks = f.specs()
    assert len(tasks) == len({t["id"] for t in tasks}) == 27
    assert [t["id"] for t in tasks[:3]] == ["llama8b-J-s42", "llama8b-K-s42", "gemma2_9b-K-s43"]
    assert {(t["variant"], t["seed"]) for t in tasks if t["model"] == "llama8b"} == {
        (v, s) for v in "EFGHIJKL" for s in (42, 43)}
    assert sum(t["stage"] == "completion-and-replication" for t in tasks) == 20
    assert all(t["late_admission"] for t in tasks[:20])
    assert not any(t["late_admission"] for t in tasks[20:])
    base = helpers.tiny(tmp_path)
    for variant, (control, changes) in f.CHANGES.items():
        old = c.apply_variant(base, control, 42, tmp_path / "out")
        expected = copy.deepcopy(old)
        for key, value in changes.items():
            section, field = key.split(".")
            expected[section][field] = value
        assert f.apply_variant(base, variant, 42, tmp_path / "out") == expected


@pytest.mark.parametrize("controls,retain", [(True, False), (False, True)])
def test_stop_gradient_preserves_values_and_only_removes_named_branch(controls, retain):
    torch.manual_seed(7)
    anchor = torch.randn(3, 8, requires_grad=True)
    positive = torch.randn(4, 3, 8, requires_grad=True)
    negative = torch.randn(6, 8, requires_grad=True)
    old = forget_loss(anchor, positive, negative)
    new = forget_loss(anchor, positive, negative,
                      stop_gradient_controls=controls, stop_gradient_retain=retain)
    assert torch.equal(old, new)
    a = torch.autograd.grad(old, (anchor, positive, negative), retain_graph=True)
    b = torch.autograd.grad(new, (anchor, positive, negative), allow_unused=True)
    assert torch.equal(a[0], b[0])
    assert (b[1] is None) == controls and (b[2] is None) == retain
    if not controls:
        assert torch.equal(a[1], b[1])
    if not retain:
        assert torch.equal(a[2], b[2])


def test_fact_positive_preserves_complete_negated_qualified_value():
    value = "no evidence of disease before 2020, dose 5 mg."
    row = {"id": "x", "text": "The diagnosis of patient P12 is " + value,
           "views": ["The diagnosis of patient P12 is disease.", "unverified paraphrase"]}
    views = fact_positive_candidates(row)
    assert len(views) == 2
    assert all(value in view and "patient P12" in view and "diagnosis" in view for view in views)
    assert fact_positive_audit([row])["eligible_fraction"] == 1
    assert fact_positive_audit([{"text": "Patient P12 might be healthy"}])["unsupported_rows"] == 1


def test_shared_pool_first_wave_and_late_completion_priority(tmp_path, monkeypatch):
    tasks = f.specs()
    config = tmp_path / "config.json"
    write(config, {})
    tasks = [dict(task, config=str(config)) for task in tasks]
    plan = {"tasks": tasks, "initial_task_seconds": 3600}
    now = time.time()
    state = {"status": "running", "deadline": now + 5000,
             "tasks": {t["id"]: {"status": "pending", "attempts": 0} for t in tasks},
             "workers": {str(i): {} for i in range(3)}}
    write(tmp_path / "state.json", state)
    monkeypatch.setattr(c, "training_done", lambda *args: False)
    claimed = [c.claim(tmp_path, plan, worker, now + 43200)["id"] for worker in range(3)]
    assert claimed == ["llama8b-J-s42", "llama8b-K-s42", "gemma2_9b-K-s43"]
    for _ in range(17):
        task = c.claim(tmp_path, plan, 0, now + 43200)
        assert task is not None and task["stage"] == "completion-and-replication"
    assert c.claim(tmp_path, plan, 0, now + 43200) is None


def configured(tmp_path, family="llama", mode="default"):
    cfg = helpers.tiny(tmp_path, family)
    cfg["conrep"]["protected_positive"] = False
    if mode == "Q":
        cfg["conrep"]["specified_positive"] = "fact_paraphrase"
    elif mode == "R":
        cfg["conrep"]["stop_gradient_controls"] = True
    elif mode == "S":
        cfg["conrep"]["stop_gradient_retain"] = True
    elif mode == "insertion":
        from conrep.night.insertion import POLICY
        cfg["conrep"].update(specified_views=3, specified_negative_views=1,
            specified_noise_kind="insertion", specified_noise_probability=0.,
            specified_noise_policy=POLICY, specified_insertion_mode="binomial2p20",
            specified_negative_source="clean_dropout")
    probe_rows = []
    for group, offset in (("forget", 0), ("retain", 4), ("general", 8)):
        rows = [{"id": f"{group}:{i}", "text": f"The condition of patient P{i + offset} is asthma .",
                 "views": []} for i in range(4)]
        path = Path(cfg["data"]["prepared_dir"]) / f"{group}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        selected, _ = f.diagnostic_rows(rows, group, 2)
        probe_rows += selected
    probe = tmp_path / "probes.json"
    write(probe, {"rows": probe_rows})
    cfg["diagnostics"] = {"enabled": True, "every_steps": 1, "probe_file": str(probe),
        "probe_sha256": file_sha(probe), "max_length": 96, "gradient_tensors": 2,
        "fixed_corruption_views": 2, "fixed_corruption_probability": .7, "seed": 39027}
    return cfg


def weights(path, step=3):
    return load_file(Path(path) / f"checkpoint-{step}/adapter_model.safetensors")


def identical(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


@pytest.mark.parametrize("family", ["llama", "gemma"])
@pytest.mark.parametrize("mode", ["default", "Q", "R", "S", "insertion"])
def test_diagnostics_do_not_change_training_and_resume_is_exact(tmp_path, family, mode):
    cfg = configured(tmp_path, family, mode)
    disabled = copy.deepcopy(cfg)
    disabled["diagnostics"]["enabled"] = False
    disabled["run"]["output_dir"] = str(tmp_path / "without")
    assert run(disabled) == 0
    cfg["run"]["output_dir"] = str(tmp_path / "with")
    assert run(cfg) == 0
    identical(weights(tmp_path / "without"), weights(tmp_path / "with"))
    resumed = copy.deepcopy(cfg)
    resumed["run"]["output_dir"] = str(tmp_path / "resume")
    assert run(resumed, stop_after_step=1) == 75
    assert run(resumed, resume=str(tmp_path / "resume/checkpoint-1")) == 0
    identical(weights(tmp_path / "with"), weights(tmp_path / "resume"))
    report = read(tmp_path / "resume/diagnostics/step-000003.json")
    assert report["summary"]["forget.answers_scored"] == 2
    assert report["summary"]["retain.answers_scored"] == 2
    assert report == read(tmp_path / "with/diagnostics/step-000003.json")
    gradients = read(tmp_path / "resume/diagnostics/gradients-step-000003-micro-0.json")
    assert len(gradients["parameters"]) == 2
    assert gradients["sampled_parameter_count"] > 0
    assert "forget_cl__retain_lm" in gradients["cosines"]
    assert gradients == read(tmp_path / "with/diagnostics/gradients-step-000003-micro-0.json")


def test_answer_mask_excludes_role_and_end_tokens_and_refuses_truncation(tmp_path):
    _, tokenizer = helpers.helpers.tiny_setup(tmp_path)
    row = {"prompt": "What is condition of patient P0 ?", "answer": "asthma ."}
    ids, positions = answer_tokens(tokenizer, row, 100)
    assert [ids[i] for i in positions] == tokenizer.encode(row["answer"], add_special_tokens=False)
    assert not (set(ids[i] for i in positions) & set(tokenizer.all_special_ids))
    assert answer_tokens(tokenizer, row, max(positions)) is None


def test_missing_reference_cannot_be_rebuilt_from_unlearned_weights(tmp_path):
    cfg = configured(tmp_path)
    cfg["run"]["output_dir"] = str(tmp_path / "run")
    assert run(cfg, stop_after_step=1) == 75
    (tmp_path / "run/diagnostics/reference-rank-0.json").unlink()
    with pytest.raises(ValueError, match="Missing step-zero"):
        run(cfg, resume=str(tmp_path / "run/checkpoint-1"))


def test_two_rank_diagnostics_preserve_update_and_resume(tmp_path):
    cfg = configured(tmp_path)
    cfg["unlearn"].update(max_steps=2, save_steps=1)
    entry = ROOT / c.ENTRY
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=2", str(entry), "train", "--config", str(tmp_path / "config.json")]
    def execute(options, extra=()):
        write(tmp_path / "config.json", options)
        result = subprocess.run(command + list(extra), capture_output=True, text=True, timeout=120)
        if "gloo/transport/tcp/device.cc" in result.stderr and "Operation not permitted" in result.stderr:
            pytest.skip("Gloo interface unavailable; ABCI eight-GPU smoke remains required")
        assert result.returncode == 0, result.stdout + result.stderr
    disabled = copy.deepcopy(cfg)
    disabled["diagnostics"]["enabled"] = False
    disabled["run"]["output_dir"] = str(tmp_path / "without")
    execute(disabled)
    cfg["run"]["output_dir"] = str(tmp_path / "with")
    execute(cfg, ["--stop-after-step", "1"])
    execute(cfg, ["--resume", str(tmp_path / "with/checkpoint-1")])
    identical(weights(tmp_path / "without", 2), weights(tmp_path / "with", 2))
    report = read(tmp_path / "with/diagnostics/step-000002.json")
    assert len(report["rows"]) == 6 and report["summary"]["forget.rows"] == 2


def test_bootstrap_uses_frozen_server_helpers_and_is_idempotent(tmp_path):
    root, source, target = tmp_path / "checkout", tmp_path / "old", tmp_path / "new"
    root.mkdir()
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in bootstrap.OVERLAY:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], cwd=root, check=True)
    ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source_code = source / "code"
    shutil.copytree(root / "src", source_code / "src")
    for name in (c.ENTRY, c.SHELL):
        (source_code / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, source_code / name)
    frozen_validator = source_code / "src/experiments/validation.py"
    frozen_validator.write_text(frozen_validator.read_text() + "\n# server-only frozen evaluator\n")
    checkout_validator = root / "src/experiments/validation.py"
    checkout_validator.write_text("# later dirty checkout must never be installed\n")
    frozen_hash = file_sha(frozen_validator)
    cfg = helpers.tiny(tmp_path / "fixture")
    cfg["unlearn"].update(learning_rate=1e-5, batch_sizes={"forget": 8, "retain": 16, "general": 32})
    cfg["evaluation"].update(partition="validation", limit=None)
    cfg["conrep"]["specified_positive"] = "dropout"
    inputs = {}
    for group in ("forget", "retain", "general"):
        path = Path(cfg["data"]["prepared_dir"]) / f"{group}.jsonl"
        path.write_text("".join(json.dumps({"id": f"{group}:{i}",
            "text": f"The condition of patient P{i} is asthma .", "views": []}) + "\n" for i in range(32)))
        inputs[str(path)] = file_sha(path)
    tasks = []
    for model in ("gemma2_9b", "llama8b"):
        path = source / "configs" / f"{model}.json"
        write(path, cfg)
        tasks.append({"model": model, "variant": "A", "seed": 42,
                      "config": str(path), "config_hash": file_sha(path)})
    files = {str(p.relative_to(source_code)): file_sha(p) for p in source_code.rglob("*") if p.is_file()}
    write(source / "source.json", {"files": files, "source_hash": sha(files)})
    write(source / "inputs.json", inputs)
    assets = {str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
              for p in Path(cfg["unlearn"]["checkpoint"]).iterdir() if p.is_file()}
    write(source / "model-assets.json", assets)
    write(source / "plan.json", {"source_hash": sha(files), "project_root": str(root), "tasks": tasks})
    bootstrap.prepare(root, source, target, ref)
    plan = read(target / "plan.json")
    assert len(plan["tasks"]) == 27 and plan["walltime"] == "12:00:00"
    assert plan["recovery_policy"]["interruption_max_retries"] is None
    assert file_sha(target / "code/src/experiments/validation.py") == frozen_hash
    assert checkout_validator.read_text().startswith("# later dirty")
    before = (target / "state.json").read_bytes()
    bootstrap.prepare(root, source, target, ref)
    assert (target / "state.json").read_bytes() == before
    c.verify_snapshot(target, plan)
    c.summarize(target)
    assert (target / "diagnostic-results.csv").exists()
    with pytest.raises(ValueError, match="Parent frozen input"):
        Path(next(iter(inputs))).write_text("changed")
        bootstrap.prepare(root, source, target, ref)
