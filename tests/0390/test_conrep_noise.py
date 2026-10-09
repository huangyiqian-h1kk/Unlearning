"""Protected token boundaries, objective gradients, RNG and actual resume."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from conrep.night.io import read
from conrep.night.losses import retain_noise_loss
from conrep.night.noise import POLICY, token_record, eligible_mask, audit_rows, noise_metrics
from conrep.night.trainer import run
from conrep.v2.corruption import corrupt, safe_token_ids
from conrep.v2.model import text_batch

spec = importlib.util.spec_from_file_location("noise_helpers", Path(__file__).with_name("test_conrep_night.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
torch.set_num_threads(1)


def tokenizer():
    words = "[PAD] [UNK] The diagnosis of patient P12 is no evidence disease before 2020 dose 5 mg . female asthma unused".split()
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_protection_and_observed_probability_without_forced_mutation(padding_side):
    tok = tokenizer()
    tok.padding_side = padding_side
    rows = [{"id": "a", "text": "The diagnosis of patient P12 is no evidence of disease before 2020 dose 5 mg."},
            {"id": "b", "text": "The diagnosis of patient P12 is female."}]
    batch, pool = text_batch(tok, [r["text"] for r in rows], 512, "cpu")
    eligible = eligible_mask(tok, rows, batch, pool, 512)
    assert eligible.sum(1).tolist() == [2, 2]  # Only leading The and linking of.
    assert all(set(tok.convert_ids_to_tokens(row[mask])) == {"The", "of"} for row, mask in zip(batch["input_ids"], eligible))
    for probability in (.1, .2):
        views = corrupt(batch["input_ids"], eligible, safe_token_ids(tok, len(tok)), views=3000,
                        probability=probability, generator=torch.Generator().manual_seed(7))
        metrics = noise_metrics(batch["input_ids"], views, eligible, pool)
        assert abs(float(metrics["specified_noise_replacement_fraction_eligible"]) - probability) < .01
        assert abs(float(metrics["specified_unchanged_fraction"]) - (1-probability)**2) < .02
        assert not (views.ne(batch["input_ids"][None]) & ~eligible[None]).any()
    with pytest.raises(ValueError, match="disagree"):
        wrong = dict(batch, input_ids=batch["input_ids"].roll(1, 1))
        eligible_mask(tok, rows, wrong, pool, 512)
    audit = audit_rows(tok, rows, 512, len(tok))
    assert audit["unsupported_rows"] == 0 and audit["eligible_tokens"] == 4
    assert not audit["uses_validation_or_test"] and not audit["forced_mutation"]


def test_unknown_truncated_overlapping_and_explicit_spans_fail_closed():
    tok = tokenizer()
    row = {"id": "a", "text": "The diagnosis of patient P12 is no disease before 2020 dose 5 mg."}
    with pytest.raises(ValueError, match="grammar"):
        token_record(tok, {"text": "This is not an audited fact"}, 512)
    with pytest.raises(ValueError, match="truncated"):
        token_record(tok, row, 5)
    protected = dict(row, protected_spans=[[0, 1]])
    # Overlap with one character protects the entire The token.
    record = token_record(tok, protected, 512)
    assert sum(record["eligible"]) == 1
    assert tok.convert_ids_to_tokens([i for i, e in zip(record["ids"], record["eligible"]) if e]) == ["of"]
    protected["protected_spans"] = [[0, len(row["text"])]]
    assert not any(token_record(tok, protected, 512)["eligible"])
    assert audit_rows(tok, [row, {"text": "unknown"}], 512, len(tok))["unsupported_rows"] == 1


@pytest.mark.parametrize("k", [1, 4])
def test_separate_clean_negatives_match_manual_value_and_all_gradients(k):
    torch.manual_seed(8)
    r, n, f = [torch.randn(count, 7, requires_grad=True) for count in (4, 4, 2)]
    p = torch.randn(k, 4, 7, requires_grad=True)
    actual = retain_noise_loss(r, p, n, f)
    rows = []
    for i in range(4):
        others = [j for j in range(4) if j != i]
        negative = torch.cat([r[others], n[others], f])
        assert len(negative) == 2 * (4-1) + 2
        z = F.normalize(r[i], dim=-1)
        logits_n = F.normalize(negative, dim=-1) @ z / .1
        logits_p = F.normalize(p[:, i], dim=-1) @ z / .1
        rows.append((torch.logaddexp(logits_p, torch.logsumexp(logits_n, 0)) - logits_p).mean())
    expected = torch.stack(rows).mean()
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual, (r, p, n, f), retain_graph=True),
                    torch.autograd.grad(expected, (r, p, n, f))):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        assert a.abs().sum() > 0


def test_repeating_noisy_positive_does_not_scale_loss_or_gradient():
    torch.manual_seed(10)
    r, n, p, f = [torch.randn(count, 7, requires_grad=True) for count in (4, 4, 4, 2)]
    one = retain_noise_loss(r, p[None], n, f)
    four = retain_noise_loss(r, p[None].expand(4, -1, -1), n, f)
    torch.testing.assert_close(one, four)
    for a, b in zip(torch.autograd.grad(one, (r, n, p, f), retain_graph=True),
                    torch.autograd.grad(four, (r, n, p, f))):
        torch.testing.assert_close(a, b)


def noisy_tiny(tmp_path, family):
    cfg = helpers.tiny(tmp_path, family)
    cfg["conrep"].update(views=8, negative_views=4, specified_views=4, specified_negative_views=1,
        specified_noise_probability=.2, specified_noise_policy=POLICY,
        specified_negative_source="clean_dropout", protected_positive=False)
    cfg["diagnostics"] = {"enabled": False, "sampling_coverage": True}
    path = Path(cfg["data"]["prepared_dir"]) / "retain.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for i, row in enumerate(rows):
        row["text"] = f"The condition of patient P{i} is asthma ."
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return cfg


@pytest.mark.parametrize("family", ["llama", "gemma"])
def test_noise_training_resume_restores_updates_and_random_generators(tmp_path, family):
    cfg = noisy_tiny(tmp_path, family)
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
    for key in ("positive", "corruption"):
        assert torch.equal(ra["generators"][key], rb["generators"][key])
    logs = [json.loads(line) for line in (tmp_path / "full/train.jsonl").read_text().splitlines()]
    assert all(r["specified_negatives_per_anchor"] == 4 for r in logs)
    assert any(r["specified_noise_replaced_tokens"] > 0 for r in logs)


@pytest.mark.parametrize("views", [1, 4])
def test_zero_noise_preserves_legacy_training_and_rng(tmp_path, views, monkeypatch):
    cfg = helpers.tiny(tmp_path)
    cfg["conrep"].update(protected_positive=False, specified_views=views)
    def forbidden(*args, **kwargs):
        raise AssertionError("Zero noise invoked the new augmentation branch")
    monkeypatch.setattr("conrep.night.trainer.noisy_retain", forbidden)
    cfg["run"]["output_dir"] = str(tmp_path / "legacy")
    assert run(cfg) == 0
    cfg["conrep"].update(specified_noise_probability=0., specified_noise_policy=POLICY,
                          specified_negative_source="clean_dropout")
    cfg["run"]["output_dir"] = str(tmp_path / "zero")
    assert run(cfg) == 0
    a, b = (tmp_path / name / "checkpoint-3" for name in ("legacy", "zero"))
    wa, wb = (load_file(p / "adapter_model.safetensors") for p in (a, b))
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    ra, rb = (torch.load(p / "rng-rank-0.pt", weights_only=False) for p in (a, b))
    assert torch.equal(ra["torch"], rb["torch"])
    assert all(torch.equal(ra["generators"][k], rb["generators"][k]) for k in ra["generators"])
