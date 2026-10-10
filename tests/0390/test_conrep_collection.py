"""Result exports must preserve run identity and never publish partial validation."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/abci/0390_collect_conrep_results.py"
spec = importlib.util.spec_from_file_location("collector", SCRIPT)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(c.encode(value))


def campaign(path, *, finished=True, protocol="p1", identity="first"):
    root = path / "experiments/llama8b-J-s42"
    cfg = {"unlearn": {"max_steps": 20, "save_steps": 10}, "evaluation": {"mmlu_file": "mmlu.jsonl"}}
    config = path / "configs/llama8b-J-s42.json"
    write(config, cfg)
    task = {"id": "llama8b-J-s42", "model": "llama8b", "variant": "J", "seed": 42,
            "output": str(root), "config": str(config), "config_hash": c.digest(config.read_bytes()),
            "identity": identity}
    write(path / "plan.json", {"tasks": [task], "source_hash": "source-" + identity})
    write(path / "state.json", {"tasks": {task["id"]: {"status": "running", "stage": "validation", "job_id": "1.pbs"}}})
    write(path / "source.json", {"source_hash": "source-" + identity, "files": {}, "diff": "excluded working-tree diff"})
    write(root / "training/TRAINING_COMPLETE.json", {"identity": identity, "steps": 20})
    (root / "training/train.jsonl").write_bytes(b'{"step":10}\n{"step":20}\n{"step":')
    (root / "training/adapter.safetensors").write_bytes(b"weights-must-not-be-exported")
    for step in (10, 20):
        directory = root / "validation" / f"checkpoint-{step}"
        report = {"metrics": {"forget.qa": .2, "retain.qa": .7}, "protocol_hash": protocol,
                  "checkpoint": "training/checkpoint-" + str(step), "mmlu_diagnostics": {"scored": 4}}
        write(directory / "metrics.json", report)
        for name in ("predictions.jsonl", "mmlu_predictions.jsonl"):
            (directory / name).write_text('{"prediction":"answer"}\n')
        if finished or step == 10:
            write(directory / "NIGHT_VALIDATED.json", {"identity": identity,
                "metrics_sha256": c.digest((directory / "metrics.json").read_bytes()),
                "prediction_sizes": {n: (directory / n).stat().st_size for n in ("predictions.jsonl", "mmlu_predictions.jsonl")},
                "protocol_hash": protocol})
    write(root / "training/diagnostics/step-000000.json", {"identity": identity, "step": 0,
          "summary": {"forget.answer_mean_logp": -3.2}, "rows": []})
    write(root / "training/diagnostics/gradients-step-000010-micro-0.json", {
        "identity": identity, "step": 10, "microbatch": 0, "scope": "sampled LoRA B tensors",
        "sampled_parameter_count": 16, "raw_norms": {"forget_cl": .1},
        "weighted_norms": {"forget_cl": .5}, "cosines": {"forget_cl__retain_lm": -.2}})
    return root


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_positive_grid_exports_noise_fields_audit_and_predictions_by_default(tmp_path):
    source = tmp_path / "positive"
    root = campaign(source)
    plan = json.loads((source / "plan.json").read_text())
    plan["followup"] = {"profile": "positive-grid"}
    task = plan["tasks"][0]
    config = Path(task["config"])
    cfg = json.loads(config.read_text())
    cfg["conrep"] = {"corruption_rate": .7, "views": 4, "negative_views": 4,
        "specified_views": 4, "specified_negative_views": 1, "specified_noise_probability": .2,
        "specified_noise_policy": "training-fact-offset-protection-v1", "specified_negative_source": "clean_dropout"}
    write(config, cfg)
    task["config_hash"] = c.digest(config.read_bytes())
    write(source / "plan.json", plan)
    write(source / "retain-noise-audit.json", {"gemma2_9b": {"eligible_rows": 900}})
    write(source / "positive-grid-design.json", {"historical_controls": ["fixture"]})
    (root / "training/train.jsonl").write_text(json.dumps({"step": 10,
        "specified_positive_cosine": .99, "specified_noise_replaced_tokens": 7,
        "specified_noise_eligible_opportunities": 200}) + "\n")
    output = tmp_path / "export"
    manifest, bundle = c.collect([source], output)
    assert manifest["include_predictions"] and manifest["prediction_campaigns"] == ["positive"]
    config_row = rows(output / "experiment-configurations.csv")[0]
    assert config_row["retain_noise_probability"] == "0.2"
    assert config_row["forget_negative_views"] == "4" and config_row["retain_negative_source"] == "clean_dropout"
    assert rows(output / "augmentation-diagnostics.csv")[0]["specified_noise_replaced_tokens"] == "7"
    with zipfile.ZipFile(bundle) as archive:
        assert len([name for name in archive.namelist() if name.endswith("predictions.jsonl")]) == 4
        assert "raw/positive/retain-noise-audit.json" in archive.namelist()
        assert "raw/positive/positive-grid-design.json" in archive.namelist()


def test_running_snapshot_merges_campaigns_without_changing_sources(tmp_path):
    first, second = tmp_path / "night", tmp_path / "followup"
    campaign(first)
    campaign(second, finished=False, protocol="p2", identity="second")
    before = snapshot(first), snapshot(second)
    output = tmp_path / "export"
    manifest, bundle = c.collect([first, first, second], output)
    assert before == (snapshot(first), snapshot(second))
    assert manifest["completed_experiments"] == 1 and manifest["validated_checkpoints"] == 3
    assert manifest["diagnostic_rows"] == manifest["gradient_rows"] == 2
    assert manifest["warnings"] == []
    result = rows(output / "all-validated-checkpoints.csv")
    assert len(result) == 3 and {x["campaign"] for x in result} == {"night", "followup"}
    assert {x["protocol_hash"] for x in result} == {"p1", "p2"}
    assert len(rows(output / "completed-experiment-results.csv")) == 2
    inventory = rows(output / "experiment-status.csv")
    assert [x["experiment_complete"] for x in inventory] == ["True", "False"]
    assert inventory[1]["missing_steps"] == "20" and inventory[1]["last_logged_step"] == "20"
    assert rows(output / "gradient-diagnostics.csv")[0]["cosines.forget_cl__retain_lm"] == "-0.2"
    with zipfile.ZipFile(bundle) as archive:
        assert not any(n.endswith((".safetensors", "predictions.jsonl")) for n in archive.namelist())
        train = archive.read("raw/night/experiments/llama8b-J-s42/training/train.jsonl")
        assert train == b'{"step":10}\n{"step":20}\n'
        assert "diff" not in json.loads(archive.read("raw/night/source-provenance.json"))
        for item in manifest["files"]:
            assert c.digest(archive.read(item["path"])) == item["sha256"]


def test_insertion_export_preserves_mode_audit_handoff_and_stays_light(tmp_path):
    source = tmp_path / "insertion"
    root = campaign(source)
    plan = json.loads((source / "plan.json").read_text())
    plan["followup"] = {"profile": "insertion-grid"}
    task = plan["tasks"][0]
    config = Path(task["config"])
    cfg = json.loads(config.read_text())
    cfg["conrep"] = {"specified_views": 3, "specified_noise_kind": "insertion",
        "specified_noise_probability": 0., "specified_insertion_mode": "fixed2"}
    write(config, cfg)
    task["config_hash"] = c.digest(config.read_bytes())
    write(source / "plan.json", plan)
    for name in ("retain-insertion-audit.json", "insertion-grid-design.json", "handoff.json", "handoff-state.json"):
        write(source / name, {"fixture": True})
    output = tmp_path / "export"
    manifest, bundle = c.collect([source], output, include_predictions=False)
    assert not manifest["include_predictions"] and not manifest["warnings"]
    row = rows(output / "experiment-configurations.csv")[0]
    assert row["retain_noise_kind"] == "insertion" and row["retain_insertion_mode"] == "fixed2"
    assert row["retain_views"] == "3"
    with zipfile.ZipFile(bundle) as archive:
        assert "raw/insertion/retain-insertion-audit.json" in archive.namelist()
        assert "raw/insertion/handoff-state.json" in archive.namelist()
        assert not any(n.endswith("predictions.jsonl") for n in archive.namelist())


@pytest.mark.parametrize("defect", ["metrics", "identity", "prediction_size", "missing_mmlu", "protocol"])
def test_uncommitted_or_invalid_results_are_not_counted(tmp_path, defect):
    source = tmp_path / "source"
    root = campaign(source)
    directory = root / "validation/checkpoint-20"
    marker_path = directory / "NIGHT_VALIDATED.json"
    marker = json.loads(marker_path.read_text())
    if defect == "metrics":
        (directory / "metrics.json").write_text("{}")
    elif defect == "identity":
        marker["identity"] = "another experiment"
    elif defect == "prediction_size":
        (directory / "predictions.jsonl").write_text("")
    elif defect == "missing_mmlu":
        marker["prediction_sizes"].pop("mmlu_predictions.jsonl")
    else:
        marker["protocol_hash"] = "changed protocol"
    write(marker_path, marker)
    manifest, _ = c.collect([source], tmp_path / "export")
    assert manifest["validated_checkpoints"] == 1 and manifest["completed_experiments"] == 0
    assert len(manifest["warnings"]) == 1


def test_default_cli_includes_llama_ms_only_when_present(tmp_path, monkeypatch):
    base = tmp_path / "results/validated_v2/0390"
    for name in c.DEFAULT_CAMPAIGNS:
        campaign(base / name)
    first = tmp_path / "first-export"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--root", str(tmp_path), "--output-dir", str(first)])
    c.main()
    assert len(json.loads((first / "manifest.json").read_text())["campaigns"]) == 2
    campaign(base / "conrep-llama-ms-20261009")
    second = tmp_path / "second-export"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--root", str(tmp_path), "--output-dir", str(second)])
    c.main()
    manifest = json.loads((second / "manifest.json").read_text())
    assert len(manifest["campaigns"]) == 3
    assert manifest["completed_experiments"] == 3 and manifest["validated_checkpoints"] == 6


def test_mixed_collection_includes_config_and_only_committed_coverage(tmp_path, monkeypatch):
    base = tmp_path / "results/validated_v2/0390"
    for name in c.DEFAULT_CAMPAIGNS:
        campaign(base / name)
    mixed = base / "conrep-mixed-grid-20261009"
    root = campaign(mixed)
    sample = {"identity": "first", "step": 10,
              "counts": {"forget": [2, 1, 0], "retain": [1, 1, 1, 0]}}
    checkpoint = root / "training/checkpoint-10"
    write(checkpoint / "sampling_state.json", sample)
    write(checkpoint / "COMPLETE.json", {"identity": "first", "step": 10,
        "files": {"sampling_state.json": (checkpoint / "sampling_state.json").stat().st_size}})
    # Uncommitted checkpoint and raw observations must not enter the coverage CSV.
    write(root / "training/checkpoint-20/sampling_state.json", dict(sample, step=20))
    write(root / "training/sampling/step-000011-micro-0.json", {"identity": "first", "step": 11})
    output = tmp_path / "export"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--root", str(tmp_path), "--output-dir", str(output)])
    c.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["campaigns"]) == 3
    assert manifest["sampling_coverage_rows"] == 2
    coverage = rows(output / "sampling-coverage.csv")
    assert all(row["step"] == "10" and row["committed"] == "True" for row in coverage)
    assert coverage[1]["coverage"] == "0.75" and coverage[1]["draws"] == "3"
    assert len(rows(output / "experiment-configurations.csv")) == 3


def test_isolated_cli_can_include_verified_predictions(tmp_path):
    source = tmp_path / "source"
    campaign(source)
    output = tmp_path / "export"
    result = subprocess.run([sys.executable, "-I", str(SCRIPT), "--campaign", str(source),
        "--output-dir", str(output), "--include-predictions"], text=True, capture_output=True, check=True)
    assert "BUNDLE=" in result.stdout
    with zipfile.ZipFile(output / "export.zip") as archive:
        assert len([n for n in archive.namelist() if n.endswith("predictions.jsonl")]) == 4
    with pytest.raises(FileExistsError):
        c.collect([source], output)


def test_refuse_changed_config_and_export_into_source(tmp_path):
    source = tmp_path / "source"
    campaign(source)
    with pytest.raises(ValueError, match="outside source"):
        c.collect([source], source / "export")
    write(source / "configs/llama8b-J-s42.json", {})
    with pytest.raises(ValueError, match="Frozen config changed"):
        c.collect([source], tmp_path / "export")
