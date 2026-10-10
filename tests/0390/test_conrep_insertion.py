"""Actual token-boundary preservation, count distribution and training resume."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tokenizers.processors import TemplateProcessing

from conrep.night.insertion import POLICY, MODES, gap_record, insertion_views, audit_rows
from conrep.night.io import read
from conrep.night.trainer import run
from conrep.v2.model import text_batch
from conrep.v2.corruption import safe_token_ids

spec = importlib.util.spec_from_file_location("insertion_helpers", Path(__file__).with_name("test_conrep_noise.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
torch.set_num_threads(1)


def wrapped_tokenizer():
    tok = helpers.tokenizer()
    tok.add_special_tokens({"bos_token": "[BOS]", "eos_token": "[EOS]"})
    tok.backend_tokenizer.post_processor = TemplateProcessing(single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", tok.bos_token_id), ("[EOS]", tok.eos_token_id)])
    return tok


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("mode", MODES)
def test_insertion_preserves_all_ids_protected_contiguity_wrappers_and_masks(side, mode):
    tok = wrapped_tokenizer()
    tok.padding_side = side
    rows = [{"id": "a", "text": "The diagnosis of patient P12 is no disease before 2020 dose 5 mg."},
            {"id": "b", "text": "The diagnosis of patient P12 is female."}]
    encoded, mask = text_batch(tok, [r["text"] for r in rows], 512, "cpu")
    safe = safe_token_ids(tok, len(tok))
    before = encoded["input_ids"].clone()
    views, metrics, details = insertion_views(tok, rows, encoded, mask, safe, views=12,
        mode=mode, max_length=512, generator=torch.Generator().manual_seed(16))
    assert torch.equal(encoded["input_ids"], before)
    for (batch, pool), group in zip(views, details):
        for index, row in enumerate(rows):
            record, provenance = gap_record(tok, row, 512), group[index]
            ids = batch["input_ids"][index, batch["attention_mask"][index].bool()].tolist()
            assert ids == provenance["ids"]
            assert ids[0] == tok.bos_token_id and ids[-1] == tok.eos_token_id
            assert [ids[p] for p in provenance["original_positions"]] == record["ids"]
            assert len(provenance["gaps"]) == len(set(provenance["gaps"]))
            assert set(provenance["noise_ids"]).issubset(set(safe.tolist()))
            assert pool[index].sum() == mask[index].sum() + len(provenance["gaps"])
            assert not (pool[index] & ~batch["attention_mask"][index].bool()).any()
            for a, b in record["spans"]:
                covered = [i for i, (x, y) in enumerate(record["offsets"]) if x < b and y > a]
                mapped = [provenance["original_positions"][i] for i in covered]
                assert mapped == list(range(min(mapped), max(mapped)+1))
    if mode != "binomial2p20":
        assert metrics["specified_insertion_tokens_per_view"] == (1 if mode == "fixed1" else 2)
        assert metrics["specified_unchanged_fraction"] == 0


def test_low_dose_count_distribution_and_fail_closed_boundaries():
    tok = helpers.tokenizer()
    row = {"text": "The diagnosis of patient P12 is no disease before 2020 dose 5 mg."}
    batch, pool = text_batch(tok, [row["text"]], 512, "cpu")
    _, metrics, _ = insertion_views(tok, [row], batch, pool, safe_token_ids(tok, len(tok)),
        views=3000, mode="binomial2p20", max_length=512, generator=torch.Generator().manual_seed(42))
    assert abs(float(metrics["specified_insertion_tokens_per_view"]) - .4) < .035
    for count, probability in enumerate((.64, .32, .04)):
        assert abs(float(metrics[f"specified_insertion_count_{count}_views"]) / 3000 - probability) < .025
    protected = dict(row, protected_spans=[[0, len(row["text"])]])
    record = gap_record(tok, protected, 512)
    assert record["gaps"] == [0, len(record["ids"])]
    with pytest.raises(ValueError, match="truncat"):
        gap_record(tok, row, len(record["ids"])+1)
    with pytest.raises(ValueError, match="grammar"):
        gap_record(tok, {"text": "Unknown assertion"}, 512)
    with pytest.raises(ValueError, match="disagree"):
        insertion_views(tok, [row], dict(batch, input_ids=batch["input_ids"].roll(1, 1)), pool,
            safe_token_ids(tok, len(tok)), views=2, mode="fixed2", max_length=512,
            generator=torch.Generator().manual_seed(7))
    audit = audit_rows(tok, [row], 512, len(tok))
    assert audit["audited_rows"] == 1 and audit["unsupported_rows"] == 0
    assert audit["original_tokens_preserved"] and not audit["uses_validation_or_test"]


@pytest.mark.parametrize("views,mode", [(2, "fixed1"), (3, "fixed2"), (4, "binomial2p20")])
def test_gemma_insertion_training_resume_restores_weights_rng_and_sampling(tmp_path, views, mode):
    cfg = helpers.noisy_tiny(tmp_path, "gemma")
    cfg["conrep"].update(views=4, specified_views=views, specified_noise_probability=0.,
        specified_noise_kind="insertion", specified_noise_policy=POLICY, specified_insertion_mode=mode)
    cfg["run"]["output_dir"] = str(tmp_path / "full")
    assert run(cfg) == 0
    cfg["run"]["output_dir"] = str(tmp_path / "resumed")
    assert run(cfg, stop_after_step=1) == 75
    assert run(cfg, resume=str(tmp_path / "resumed/checkpoint-1")) == 0
    a, b = (tmp_path / name / "checkpoint-3" for name in ("full", "resumed"))
    wa, wb = (load_file(p / "adapter_model.safetensors") for p in (a, b))
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    assert read(a / "sampling_state.json") == read(b / "sampling_state.json")
    ra, rb = (torch.load(p / "rng-rank-0.pt", weights_only=False) for p in (a, b))
    assert torch.equal(ra["torch"], rb["torch"])
    assert all(torch.equal(ra["generators"][k], rb["generators"][k]) for k in ra["generators"])
    logs = [json.loads(line) for line in (tmp_path / "full/train.jsonl").read_text().splitlines()]
    assert all(r["specified_negatives_per_anchor"] == 4 for r in logs)
    assert all(r["specified_views"] == views for r in logs)
    assert all(r["specified_insertion_views"] == views * 2 for r in logs)
