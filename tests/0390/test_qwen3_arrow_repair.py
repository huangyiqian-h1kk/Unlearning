import getpass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from experiments import qwen3_rich as q
from conrep.night import campaign as scheduler

spec = importlib.util.spec_from_file_location("qwen3_arrow_repair", ROOT / "scripts/abci/0390_repair_qwen3_arrow.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


@pytest.mark.parametrize("state", ["Q", "R", "H", "E"])
def test_active_job_never_allows_environment_mutation(state):
    jobs = {"1.pbs1": dict(job_state=state, Job_Owner=getpass.getuser() + "@login4")}
    api = SimpleNamespace(qstat=lambda *a: {"Jobs": jobs}, active_jobs=scheduler.active_jobs)
    with pytest.raises(ValueError, match="still active"):
        repair.require_finished(dict(status="submitted", job_id="1.pbs1"), api)


def test_missing_pbs_history_is_not_treated_as_finished():
    api = SimpleNamespace(qstat=lambda *a: {"Jobs": {}}, active_jobs=scheduler.active_jobs)
    with pytest.raises(ValueError, match="confirmed final PBS history"):
        repair.require_finished(dict(status="submitted", job_id="1.pbs1"), api)


def test_ambiguous_submission_is_not_repaired():
    with pytest.raises(ValueError, match="ambiguous"):
        repair.require_finished(dict(status="submitting", job_id=None), None)


@pytest.fixture
def legacy_campaign(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    env = tmp_path / "venv"
    env.mkdir()
    (env / "0390-qwen3-env").touch()
    monkeypatch.setattr(sys, "prefix", str(env))
    plan = dict(schema="0390-qwen3-rich-v1", campaign=str(campaign), python=str(env / "bin/python"),
                environment={"numpy": "1.26.4"})
    q.write(campaign / "plan.json", plan)
    q.write(campaign / "state.json", dict(status="blocked", job_id="1.pbs1", attempts=1))
    q.write(campaign / "config.json", {"sft": {"epochs": 5}})
    q.write(campaign / "code-files.json", {"original.py": "unchanged"})
    q.write(campaign / "validation/base/EVALUATED.json", {"files": {"metrics.json": "original"}})
    (campaign / "validation/base/metrics.json").write_text("original evaluation")
    (campaign / "code/src").mkdir(parents=True)
    (campaign / "code/src/original.py").write_text("original frozen source")
    original = {p.relative_to(campaign): p.read_bytes() for p in campaign.rglob("*") if p.is_file()}
    monkeypatch.setattr(q, "verify", lambda c: (plan, {}))
    monkeypatch.setattr(scheduler, "qstat", lambda *a: {"Jobs": {"1.pbs1": {"job_state": "F", "Exit_status": 1}}})
    values = iter([{"numpy": "1.26.4", "pyarrow": "26.0.0", "torch": "2.5.1"},
                   {"numpy": "1.26.4", "pyarrow": "19.0.1", "torch": "2.5.1"}])
    monkeypatch.setattr(repair, "installed", lambda: next(values))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(repair.subprocess, "run", run)
    resumes = []
    monkeypatch.setattr(q, "start", lambda c, resume=False: resumes.append((c, resume)))
    return campaign, original, calls, resumes


def test_legacy_repair_preserves_identity_results_and_only_installs_arrow(legacy_campaign):
    campaign, original, calls, resumes = legacy_campaign
    repair.repair(campaign, resume=True)
    assert all((campaign / p).read_bytes() == value for p, value in original.items())
    installs = [c for c in calls if "install" in c]
    assert len(installs) == 1 and "--no-deps" in installs[0] and installs[0][-1] == "pyarrow==19.0.1"
    assert any(c[-2:] == ["--check", "--smoke"] for c in calls)
    assert resumes == [(campaign, True)]
    audit = json.loads(next((campaign / "maintenance").glob("*/repair.json")).read_text())
    assert audit["status"] == "verified" and audit["before"]["pyarrow"] == "26.0.0"


def test_preflight_failure_preserves_evidence_and_never_resumes(legacy_campaign, monkeypatch):
    campaign, original, calls, resumes = legacy_campaign
    def run(command, **kwargs):
        if "--check" in command:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(repair.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        repair.repair(campaign, resume=True)
    assert not resumes
    assert all((campaign / p).read_bytes() == value for p, value in original.items())
    assert q.read(next((campaign / "maintenance").glob("*/repair.json")))["status"] == "failed"


def test_other_dependency_change_blocks_resume(legacy_campaign, monkeypatch):
    campaign, _, _, resumes = legacy_campaign
    values = iter([{"numpy": "1.26.4", "pyarrow": "26.0.0"}, {"numpy": "2.0.0", "pyarrow": "19.0.1"}])
    monkeypatch.setattr(repair, "installed", lambda: next(values))
    with pytest.raises(ValueError, match="other than Arrow changed"):
        repair.repair(campaign, resume=True)
    assert not resumes


def test_live_supervisor_blocks_repair_before_pip(legacy_campaign):
    campaign, _, calls, resumes = legacy_campaign
    with q.locked(campaign / "supervisor.lock", blocking=False):
        with pytest.raises(BlockingIOError):
            repair.repair(campaign, resume=True)
    assert not calls and not resumes


def test_explicit_stop_is_preserved(legacy_campaign):
    campaign, _, calls, resumes = legacy_campaign
    (campaign / "STOP").touch()
    with pytest.raises(ValueError, match="Explicit STOP preserved"):
        repair.repair(campaign, resume=True)
    assert (campaign / "STOP").exists() and not calls and not resumes


def test_broken_training_import_blocks_start_before_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "verify", lambda c: ({}, {}))
    def broken(c):
        raise RuntimeError("Trainer import failed")
    monkeypatch.setattr(q, "training_preflight", broken)
    monkeypatch.setattr(q.subprocess, "Popen", lambda *a, **kw: pytest.fail("Must not launch supervisor"))
    with pytest.raises(RuntimeError, match="Trainer import failed"):
        q.start(tmp_path)
