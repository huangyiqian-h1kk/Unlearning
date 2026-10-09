"""Numerical loss controls and real multi-view training/checkpoint recovery."""

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from conrep.night.io import read, write
from conrep.night.losses import retain_loss
from conrep.night.trainer import run
from conrep.v2.losses import paired_loss

spec = importlib.util.spec_from_file_location("views_helpers", Path(__file__).with_name("test_conrep_night.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
torch.set_num_threads(1)


def test_single_view_exactly_matches_legacy_value_and_all_gradients():
    torch.manual_seed(42)
    r, p, f = [torch.randn(n, 11, requires_grad=True) for n in (4, 4, 3)]
    old = paired_loss(r, p, f, .1)
    new = retain_loss(r, p[None], f, temperature=.1)
    assert torch.equal(old, new)
    a = torch.autograd.grad(old, (r, p, f), retain_graph=True)
    b = torch.autograd.grad(new, (r, p, f))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("k", [4, 8])
def test_multi_view_denominators_and_gradients_against_explicit_formula(k):
    torch.manual_seed(20)
    r = torch.randn(3, 9, requires_grad=True)
    p = torch.randn(k, 3, 9, requires_grad=True)
    f = torch.randn(2, 9, requires_grad=True)
    actual = retain_loss(r, p, f)
    reference = []
    for i in range(3):
        other = [j for j in range(3) if j != i]
        negatives = torch.cat([r[other], p[0, other], f])
        z = F.normalize(r[i], dim=-1)
        n = F.normalize(negatives, dim=-1) @ z / .1
        # Exactly 2*(R-1)+F negatives, independent of K. Own other positives excluded.
        assert len(n) == 6
        pos = F.normalize(p[:, i], dim=-1) @ z / .1
        reference.append((torch.logaddexp(pos, torch.logsumexp(n, 0)) - pos).mean())
    expected = torch.stack(reference).mean()
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, (r, p, f), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (r, p, f))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_repeating_identical_positives_does_not_multiply_weight_or_gradients():
    torch.manual_seed(8)
    r, p, f = [torch.randn(n, 7, requires_grad=True) for n in (4, 4, 2)]
    one = paired_loss(r, p, f)
    eight = retain_loss(r, p[None].expand(8, -1, -1), f)
    torch.testing.assert_close(one, eight)
    a = torch.autograd.grad(one, (r, p, f), retain_graph=True)
    b = torch.autograd.grad(eight, (r, p, f))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)


def weights(path, step):
    return load_file(path / f"checkpoint-{step}/adapter_model.safetensors")


def assert_weights_equal(a, b):
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("family", ["llama", "gemma"])
def test_four_views_actual_training_coverage_and_resume_match(tmp_path, family):
    cfg = helpers.tiny(tmp_path, family)
    cfg["conrep"].update(specified_views=4, specified_negative_views=1, protected_positive=False)
    cfg["diagnostics"] = {"enabled": False, "sampling_coverage": True}
    cfg["run"]["output_dir"] = str(tmp_path / "full")
    assert run(cfg) == 0
    cfg["run"]["output_dir"] = str(tmp_path / "resumed")
    assert run(cfg, stop_after_step=1) == 75
    checkpoint = tmp_path / "resumed/checkpoint-1"
    assert "sampling_state.json" in read(checkpoint / "COMPLETE.json")["files"]
    assert run(cfg, resume=str(checkpoint)) == 0
    assert_weights_equal(weights(tmp_path / "full", 3), weights(tmp_path / "resumed", 3))
    assert read(tmp_path / "full/checkpoint-3/sampling_state.json") == read(tmp_path / "resumed/checkpoint-3/sampling_state.json")
    for group, counts in read(tmp_path / "full/checkpoint-3/sampling_state.json")["counts"].items():
        assert sum(counts) == 3 * cfg["unlearn"]["batch_sizes"][group]
    # Logging actual samples cannot change the optimization trajectory.
    cfg["diagnostics"]["sampling_coverage"] = False
    cfg["run"]["output_dir"] = str(tmp_path / "no-audit")
    assert run(cfg) == 0
    assert_weights_equal(weights(tmp_path / "full", 3), weights(tmp_path / "no-audit", 3))


def test_default_and_explicit_single_view_training_are_identical(tmp_path):
    cfg = helpers.tiny(tmp_path)
    cfg["run"]["output_dir"] = str(tmp_path / "default")
    assert run(cfg) == 0
    cfg["conrep"].update(specified_views=1, specified_negative_views=1)
    cfg["run"]["output_dir"] = str(tmp_path / "explicit")
    assert run(cfg) == 0
    assert_weights_equal(weights(tmp_path / "default", 3), weights(tmp_path / "explicit", 3))


def test_two_rank_multi_view_resume_and_global_coverage(tmp_path):
    cfg = helpers.tiny(tmp_path)
    cfg["unlearn"].update(max_steps=2, save_steps=1)
    cfg["conrep"].update(specified_views=4, specified_negative_views=1, protected_positive=False)
    cfg["diagnostics"] = {"enabled": False, "sampling_coverage": True}
    config = tmp_path / "config.json"
    entry = helpers.ROOT / helpers.c.ENTRY
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=2", str(entry), "train", "--config", str(config)]
    def execute(extra=()):
        write(config, cfg)
        result = subprocess.run(command + list(extra), capture_output=True, text=True, timeout=120)
        if "gloo/transport/tcp/device.cc" in result.stderr and "Operation not permitted" in result.stderr:
            pytest.skip("Gloo interface unavailable; ABCI eight-GPU smoke is required")
        assert result.returncode == 0, result.stdout + result.stderr
    cfg["run"]["output_dir"] = str(tmp_path / "full")
    execute()
    cfg["run"]["output_dir"] = str(tmp_path / "resumed")
    execute(["--stop-after-step", "1"])
    execute(["--resume", str(tmp_path / "resumed/checkpoint-1")])
    assert_weights_equal(weights(tmp_path / "full", 2), weights(tmp_path / "resumed", 2))
    expected = read(tmp_path / "full/checkpoint-2/sampling_state.json")
    assert expected == read(tmp_path / "resumed/checkpoint-2/sampling_state.json")
    for group, counts in expected["counts"].items():
        assert sum(counts) == 2 * cfg["unlearn"]["batch_sizes"][group]
