"""Missing shards and surviving validation reports must not imply usable weights."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/abci/0390_inspect_qwen_sft.py"
spec = importlib.util.spec_from_file_location("qwen_inventory", SCRIPT)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_surviving_metadata_and_partial_shards_are_not_complete_weights(tmp_path):
    sft = tmp_path / "injection/sft"
    checkpoint = sft / "checkpoint-700"
    write(tmp_path / "injection/selected-sft.json", {"selected": str(checkpoint)})
    write(tmp_path / "injection/validation/final/metrics.json", {"metrics": {"forget.qa": .2}})
    write(checkpoint / "config.json", {"model_type": "qwen2"})
    write(checkpoint / "model.safetensors.index.json", {"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}})
    (checkpoint / "one.safetensors").write_bytes(b"inventory-fixture")
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = q.inspect(tmp_path)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    ckpt, final = report["checkpoints"]
    assert ckpt["exists"] and not ckpt["hf_weight_files_complete"]
    assert ckpt["weight_formats"][0]["missing_or_empty"] == ["two.safetensors"]
    assert not final["exists"] and "final" in report["validation"]
    (checkpoint / "two.safetensors").touch()
    assert not q.checkpoint_inventory(checkpoint)["hf_weight_files_complete"]
    (checkpoint / "two.safetensors").write_bytes(b"second-shard-fixture")
    result = q.checkpoint_inventory(checkpoint)
    assert result["hf_weight_files_complete"] and not result["model_load_verified"]
    assert not result["exact_resume_verified"] and result["resume_files"] == []


def test_missing_root_and_outside_index_shards(tmp_path):
    assert not q.inspect(tmp_path / "missing")["root_exists"]
    checkpoint = tmp_path / "checkpoint-700"
    write(checkpoint / "config.json", {})
    write(checkpoint / "model.safetensors.index.json", {"weight_map": {"a": "../outside.safetensors"}})
    (tmp_path / "outside.safetensors").write_bytes(b"outside")
    result = q.checkpoint_inventory(checkpoint)
    assert not result["hf_weight_files_complete"] and result["weight_formats"][0]["error"]
