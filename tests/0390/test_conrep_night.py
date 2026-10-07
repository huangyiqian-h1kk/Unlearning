import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file

from conrep.night import campaign as c
from conrep.night.io import read, write, training_identity, complete_checkpoint, latest_checkpoint
from conrep.night.losses import forget_loss
from conrep.night.positives import candidates, audit
from conrep.night.trainer import run
from conrep.v2.losses import forget_loss as original_loss

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("night_tiny_helpers", Path(__file__).with_name("test_core.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
torch.set_num_threads(1)


def tiny(tmp_path, family="llama"):
    cfg, tokenizer = helpers.tiny_setup(tmp_path)
    if family == "gemma":
        from transformers import Gemma2Config, Gemma2ForCausalLM
        torch.manual_seed(19)
        model = Gemma2ForCausalLM(Gemma2Config(vocab_size=len(tokenizer), hidden_size=16,
                    intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                    num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
                    sliding_window=64, pad_token_id=tokenizer.pad_token_id))
        model.save_pretrained(cfg["unlearn"]["checkpoint"])
    cfg["unlearn"].update(max_steps=3, save_steps=1)
    cfg["unlearn"]["gradient_checkpointing"] = family == "gemma"
    cfg["conrep"].update(specified_positive="dropout", protected_positive=True,
                          forget_cl_weight=5.0, views=4, negative_views=2)
    path = Path(cfg["data"]["prepared_dir"]) / "retain.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["text"] = "The condition of the " + row["text"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return cfg


def test_default_loss_and_gradients_are_unchanged():
    torch.manual_seed(4)
    f = torch.randn(3, 7, requires_grad=True)
    controls = torch.randn(4, 3, 7, requires_grad=True)
    retain = torch.randn(5, 7, requires_grad=True)
    old = original_loss(f, controls, retain)
    new = forget_loss(f, controls, retain, negative_views=4)
    assert torch.equal(old, new)
    a = torch.autograd.grad(old, (f, controls, retain), retain_graph=True)
    b = torch.autograd.grad(new, (f, controls, retain))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_eight_positives_use_only_four_negative_views():
    import torch.nn.functional as F
    torch.manual_seed(9)
    f, controls, retain = torch.randn(3, 7), torch.randn(8, 3, 7), torch.randn(2, 7)
    actual = forget_loss(f, controls, retain, negative_views=4, retain_weight=1, margin=0)
    losses = []
    for i in range(3):
        others = [j for j in range(3) if j != i]
        neg = torch.cat([controls[:4, others].reshape(-1, 7), f[others], retain])
        pos = controls[:, i]
        z = F.normalize(f[i], dim=-1)
        n = (F.normalize(neg, dim=-1) @ z) / .08
        p = (F.normalize(pos, dim=-1) @ z) / .08
        losses.append((torch.logaddexp(p, torch.logsumexp(n, 0)) - p).mean())
    torch.testing.assert_close(actual, torch.stack(losses).mean())


def test_protected_fact_keeps_value_negation_time_and_units():
    text = "The diagnosis of the patient P12 is no evidence of the disease before 2020, dose 5 mg."
    row = {"text": text, "identifier": "P12", "attribute": "diagnosis",
           "value": "no evidence of the disease before 2020, dose 5 mg."}
    output = candidates(row)
    assert output
    assert all(row["value"] in x and "P12" in x and "diagnosis" in x for x in output)
    assert all(len(text.split()) - len(x.split()) == 1 for x in output)
    assert candidates({"text": "blood type a"}) == []
    assert candidates({"text": "The Hague", "value": "The Hague"}) == []
    assert audit([row])["expected_changed_fraction"] == .5


@pytest.mark.parametrize("family", ["llama", "gemma"])
def test_real_training_pause_and_resume_matches_uninterrupted(tmp_path, family):
    cfg = tiny(tmp_path, family)
    uninterrupted = tmp_path / "continuous"
    cfg["run"]["output_dir"] = str(uninterrupted)
    assert run(cfg) == 0
    expected = load_file(uninterrupted / "checkpoint-3/adapter_model.safetensors")
    resumed = copy.deepcopy(cfg)
    resumed["run"]["output_dir"] = str(tmp_path / "resumed")
    assert run(resumed, stop_after_step=1) == 75
    assert not (tmp_path / "resumed/TRAINING_COMPLETE.json").exists()
    checkpoint = latest_checkpoint(tmp_path / "resumed", identity=training_identity(resumed), world=1)
    assert checkpoint.name == "checkpoint-1"
    assert run(resumed, resume=str(checkpoint)) == 0
    actual = load_file(tmp_path / "resumed/checkpoint-3/adapter_model.safetensors")
    assert expected.keys() == actual.keys()
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key
    state_a = torch.load(uninterrupted / "checkpoint-3/training_state.pt", weights_only=False)
    state_b = torch.load(tmp_path / "resumed/checkpoint-3/training_state.pt", weights_only=False)
    assert state_a["scheduler"] == state_b["scheduler"]
    (tmp_path / "resumed/checkpoint-3/rng-rank-0.pt").unlink()
    assert not complete_checkpoint(tmp_path / "resumed/checkpoint-3")
    assert latest_checkpoint(tmp_path / "resumed").name == "checkpoint-2"


def test_real_two_rank_resume(tmp_path):
    cfg = tiny(tmp_path)
    cfg["unlearn"].update(max_steps=2, save_steps=1)
    config = tmp_path / "config.json"
    write(config, cfg)
    entry = ROOT / c.ENTRY
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=2", str(entry), "train", "--config", str(config)]
    first = subprocess.run(command + ["--stop-after-step", "1"], text=True, capture_output=True, timeout=90)
    if "gloo/transport/tcp/device.cc" in first.stderr and "Operation not permitted" in first.stderr:
        pytest.skip("This executor denies Gloo interface access; real eight-GPU resume smoke is required on ABCI")
    assert first.returncode == 0, first.stdout + first.stderr
    root = Path(cfg["run"]["output_dir"])
    assert complete_checkpoint(root / "checkpoint-1", world=2)
    second = subprocess.run(command + ["--resume", str(root / "checkpoint-1")],
                            text=True, capture_output=True, timeout=90)
    assert second.returncode == 0, second.stdout + second.stderr
    assert complete_checkpoint(root / "checkpoint-2", world=2)
    assert (root / "TRAINING_COMPLETE.json").exists()


def test_matrix_and_single_change_controls():
    tasks = c.specs()
    assert len(tasks) == 32 and len({x["id"] for x in tasks}) == 32
    assert sum(x["model"] == "llama8b" for x in tasks) == 8
    assert [x["variant"] for x in tasks[:4]] == list("ABCD")
    assert c.VARIANTS["H"] == {**c.VARIANTS["D"], "conrep.specified_lm_weight": 0.0}
    assert c.expected_checkpoints({"unlearn": {"max_steps": 125, "save_steps": 10}}) == list(range(10, 121, 10)) + [125]


def test_scheduler_counts_all_user_jobs_and_nodes():
    jobs = {"a": {"Job_Owner": "me@host", "job_state": "Q", "Resource_List": {"select": "3:ncpus=96"}},
            "b": {"Job_Owner": "someone@host", "job_state": "R"},
            "c": {"Job_Owner": "me@host", "job_state": "H", "Resource_List": {"select": "1"}}}
    active = c.active_jobs({"Jobs": jobs}, "me")
    assert set(active) == {"a", "c"}
    assert sum(c.allocated_nodes(x) for x in active.values()) == 4


@pytest.mark.parametrize("record,outcome", [
    ({"Exit_status": 143}, "unknown_failure"),
    ({"Exit_status": 271, "comment": "deleted by user via qdel"}, "cancelled"),
    ({"Exit_status": 271, "comment": "job killed: walltime limit"}, "recoverable"),
    ({"Exit_status": 1, "comment": "node failure"}, "recoverable"),
    ({"Exit_status": 0}, "completed"),
    ({"Exit_status": 1}, "unknown_failure"),
])
def test_only_confirmed_scheduler_interruptions_retry(record, outcome):
    assert c.classify_end(record) == outcome


def scheduler_fixture(tmp_path):
    cfg = {"unlearn": {"max_steps": 125, "save_steps": 10}, "run": {"output_dir": str(tmp_path / "training")}}
    write(tmp_path / "config.json", cfg)
    task = {"id": "x", "identity": "i", "priority": 0, "config": str(tmp_path / "config.json"),
            "output": str(tmp_path / "output"), "preferred_worker": 0}
    plan = {"project_root": str(tmp_path), "campaign": str(tmp_path), "shell": "/snapshot/worker.sh",
            "walltime": "06:00:00", "initial_task_seconds": 3600, "max_attempts": 3,
            "source_hash": "s", "queue": "R9920261000", "rtype": "rt_HF", "tasks": [task]}
    state = {"status": "running", "deadline": __import__('time').time() + 36000,
             "tasks": {"x": {"status": "pending", "attempts": 0, "failures": 0}},
             "workers": {str(i): {"status": "new", "allocations": 0, "job_id": None,
                                    "recovery_failures": 0} for i in range(4)}}
    write(tmp_path / "state.json", state)
    write(tmp_path / "plan.json", plan)
    return plan


def test_submit_respects_existing_jobs_and_does_not_duplicate(tmp_path, monkeypatch):
    plan = scheduler_fixture(tmp_path)
    monkeypatch.setattr(c.getpass, "getuser", lambda: "me")
    jobs = {str(i): {"Job_Owner": "me@h", "job_state": "R"} for i in range(3)}
    monkeypatch.setattr(c, "qstat", lambda *a: {"Jobs": jobs})
    submitted = []
    def run(command, **kwargs):
        submitted.append(command)
        jobs["99.pbs"] = {"Job_Owner": "me@h", "job_state": "Q"}
        return subprocess.CompletedProcess(command, 0, "99.pbs\n", "")
    monkeypatch.setattr(c.subprocess, "run", run)
    c.submit_available(tmp_path, plan)
    c.submit_available(tmp_path, plan)
    assert len(submitted) == 1
    assert read(tmp_path / "state.json")["workers"]["0"]["job_id"] == "99.pbs"


def test_qstat_failure_never_submits(tmp_path, monkeypatch):
    plan = scheduler_fixture(tmp_path)
    monkeypatch.setattr(c, "qstat", lambda *a: (_ for _ in ()).throw(RuntimeError("unavailable")))
    with pytest.raises(RuntimeError, match="unavailable"):
        c.submit_available(tmp_path, plan)
    assert read(tmp_path / "state.json")["workers"]["0"]["allocations"] == 0


@pytest.mark.parametrize("returncode,stdout", [(1, ""), (0, "lost job identifier")])
def test_ambiguous_submission_blocks_other_submissions(tmp_path, monkeypatch, returncode, stdout):
    plan = scheduler_fixture(tmp_path)
    monkeypatch.setattr(c.getpass, "getuser", lambda: "me")
    jobs = {str(i): {"Job_Owner": "me@h", "job_state": "R"} for i in range(3)}
    monkeypatch.setattr(c, "qstat", lambda *a: {"Jobs": jobs})
    submitted = []
    def run(command, **kwargs):
        submitted.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout, "connection lost")
    monkeypatch.setattr(c.subprocess, "run", run)
    c.submit_available(tmp_path, plan)
    c.submit_available(tmp_path, plan)
    assert len(submitted) == 1
    state = read(tmp_path / "state.json")
    assert state["workers"]["0"]["status"] == "submitting"
    assert all(w["allocations"] == 0 for i, w in state["workers"].items() if i != "0")


def test_qdel_cancels_interrupted_task_without_requeue(tmp_path, monkeypatch):
    plan = scheduler_fixture(tmp_path)
    with c.state_transaction(tmp_path) as state:
        state["workers"]["0"].update(job_id="99.pbs", status="interrupted")
        state["tasks"]["x"].update(job_id="99.pbs", status="interrupted")
    monkeypatch.setattr(c, "qstat", lambda *a: {"Jobs": {"99.pbs": {
        "job_state": "F", "Exit_status": 271, "comment": "deleted by user"}}})
    monkeypatch.setattr(c, "archive_job", lambda *a: None)
    c.reconcile(tmp_path, plan, {})
    state = read(tmp_path / "state.json")
    assert state["tasks"]["x"]["status"] == "cancelled"
    assert state["workers"]["0"]["status"] == "cancelled"


def test_confirmed_walltime_requeues_same_task(tmp_path, monkeypatch):
    plan = scheduler_fixture(tmp_path)
    with c.state_transaction(tmp_path) as state:
        state["workers"]["0"].update(job_id="99.pbs", status="running")
        state["tasks"]["x"].update(job_id="99.pbs", status="running")
    monkeypatch.setattr(c, "qstat", lambda *a: {"Jobs": {"99.pbs": {
        "job_state": "F", "Exit_status": 271, "comment": "walltime exceeded"}}})
    monkeypatch.setattr(c, "archive_job", lambda *a: None)
    c.reconcile(tmp_path, plan, {})
    state = read(tmp_path / "state.json")
    assert state["tasks"]["x"]["status"] == "paused"
    assert state["workers"]["0"]["status"] == "available"


def test_pbs_has_reserved_queue_single_node_and_prefix():
    plan = {"shell": "/a b/worker.sh", "project_root": "/project", "campaign": "/campaign", "walltime": "06:00:00"}
    script = c.render_pbs(plan, 2, "0390nabc2001")
    assert "#PBS -q R9920261000" in script and "#PBS -v RTYPE=rt_HF" in script
    assert "#PBS -l select=1" in script and "#PBS -N 0390" in script
    assert "'/a b/worker.sh'" in script


def test_pause_marker_is_fresh_and_is_not_a_crash(tmp_path):
    marker = tmp_path / "PAUSED.json"
    code = "import json,time,pathlib; pathlib.Path(" + repr(str(marker)) + ").write_text(json.dumps({'created_at':time.time()}))"
    result = c.run_child([sys.executable, "-c", code], tmp_path / "console.log", project=tmp_path,
                         deadline=__import__('time').time()+1000, stop_file=tmp_path / "STOP",
                         interrupted=lambda: False, pause_marker=marker)
    assert result == 75
    result = c.run_child([sys.executable, "-c", "raise SystemExit(1)"], tmp_path / "console.log", project=tmp_path,
                         deadline=__import__('time').time()+1000, stop_file=tmp_path / "STOP",
                         interrupted=lambda: False, pause_marker=marker)
    assert result == 1


def test_claim_is_exclusive_and_deadline_admission_is_enforced(tmp_path):
    plan = scheduler_fixture(tmp_path)
    end = __import__('time').time() + 20000
    assert c.claim(tmp_path, plan, 0, end)["id"] == "x"
    assert c.claim(tmp_path, plan, 1, end) is None
    with c.state_transaction(tmp_path) as state:
        state["tasks"]["x"]["status"] = "pending"
        state["deadline"] = __import__('time').time() + 500
    assert c.claim(tmp_path, plan, 0, end) is None


def test_prepare_freezes_uncommitted_server_source_and_full_matrix(tmp_path):
    import argparse
    project = tmp_path / "server"
    project.mkdir()
    for name in (c.ENTRY, c.SHELL, "src/experiments/validation.py"):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# local server changes\n")
    model = project / "sft-final"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"test fixture")
    data = project / "data"
    data.mkdir()
    for name, count in (("forget", 8), ("retain", 16), ("general", 32)):
        (data / (name+".jsonl")).write_text("".join(json.dumps({"id":str(i),
            "text":f"The condition of the patient P{i} is asthma.","views":[]})+"\n" for i in range(count)))
    (data / "probes.jsonl").write_text('{}\n')
    cfg = {"run": {"seed": 42}, "model": {}, "unlearn": {"checkpoint": str(model),
           "batch_sizes": {"forget":8,"retain":16,"general":32}}, "lora": {},
           "conrep": {"specified_positive":"dropout"},
           "evaluation": {"partition":"validation","limit":None,"mmlu_file":None},
           "data": {"prepared_dir":str(data),"forget_generation":str(data/"probes.jsonl"),
                    "retain_generation":str(data/"probes.jsonl"),"forget_mcq":{},"retain_mcq":{}}}
    write(project / "config.json", cfg)
    campaign = project / "results/night"
    c.prepare(argparse.Namespace(project_root=str(project),campaign=str(campaign),
               gemma_config="config.json",llama_config="config.json",hours=10,reserve_gb=100))
    plan = read(campaign / "plan.json")
    assert len(plan["tasks"]) == 32
    assert (campaign / "code/src/experiments/validation.py").read_text() == "# local server changes\n"
    c.verify_snapshot(campaign, plan)
    (campaign / "code/src/experiments/validation.py").write_text("# modified during run\n")
    with pytest.raises(ValueError, match="Frozen source changed"):
        c.verify_snapshot(campaign, plan)


def test_installer_preserves_dirty_server_files_and_refuses_collisions(tmp_path):
    spec = importlib.util.spec_from_file_location("night_installer", ROOT / "scripts/abci/0390_install_conrep_night.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git","init","-q"],cwd=repo,check=True)
    for name in installer.EXACT | {"src/conrep/night/__init__.py","src/conrep/v2/trainer.py"}:
        path = repo/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("new package\n")
    subprocess.run(["git","add","."],cwd=repo,check=True)
    subprocess.run(["git","-c","user.name=Test","-c","user.email=test@example.invalid",
                    "commit","-qm","Fixture"],cwd=repo,check=True)
    (repo/"src/conrep/v2/trainer.py").write_text("server modifications\n")
    installer.install(repo,"HEAD")
    assert (repo/"src/conrep/v2/trainer.py").read_text() == "server modifications\n"
    (repo/"src/conrep/night/__init__.py").write_text("local night changes\n")
    with pytest.raises(FileExistsError,match="preserved"):
        installer.install(repo,"HEAD")
