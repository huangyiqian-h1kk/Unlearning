import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from conrep.v2.corruption import corrupt
from conrep.v2.losses import forget_loss
from conrep.v2.model import load_model, text_batch, embed
from conrep.v2.trainer import run as train_conrep
from experiments.baselines.common import sago
from experiments.baselines.trainer import run as train_baseline
from experiments.config import load_config, write_json, write_rows
from experiments.data import chat_example, pad_examples
from experiments.selection import select

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)


def tiny_setup(root, family="llama"):
    model_dir = root / f"tiny-{family}"
    words = [
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]",
        "[USER]",
        "[ASSISTANT]",
        "patient",
        "has",
        "condition",
        "What",
        "is",
        "of",
        "?",
        ".",
        "asthma",
        "diabetes",
        "fever",
        "healthy",
        "conditionless",
        "general",
        "knowledge",
        "medical",
        "note",
        "a",
        "b",
        "c",
        "d",
    ]
    words += [f"P{i}" for i in range(12)] + [str(i) for i in range(20)]
    tokenizer = Tokenizer(
        WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        additional_special_tokens=["[USER]", "[ASSISTANT]"],
    )
    tok.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[USER] {% else %}[ASSISTANT] {% endif %}{{ message['content'] }}{{ eos_token }}{% endfor %}{% if add_generation_prompt %}[ASSISTANT] {% endif %}"
    tok.save_pretrained(model_dir)
    config_type, model_type = (
        (LlamaConfig, LlamaForCausalLM)
        if family == "llama"
        else (Qwen2Config, Qwen2ForCausalLM)
    )
    torch.manual_seed(19)
    model = model_type(
        config_type(
            vocab_size=len(tok),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=256,
            pad_token_id=tok.pad_token_id,
            bos_token_id=tok.bos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    )
    model.save_pretrained(model_dir)
    cfg = load_config(ROOT / "configs/0390/base.yaml")
    cfg["model"].update(name_or_path=str(model_dir), dtype="float32", attention="eager")
    cfg["run"]["output_dir"] = str(root / "conrep")
    cfg["data"]["prepared_dir"] = str(root / "data")
    cfg["lora"].update(r=2, lora_alpha=4)
    cfg["unlearn"].update(
        checkpoint=str(model_dir),
        max_steps=2,
        save_steps=1,
        max_length=48,
        batch_sizes={"forget": 2, "retain": 2, "general": 2},
        gradient_checkpointing=False,
    )
    cfg["conrep"].update(max_length=48, views=2)
    cfg["sft"].update(
        deepspeed=None,
        max_steps=2,
        save_steps=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        max_length=48,
    )
    cfg["evaluation"].update(
        max_new_tokens=3, batch_size=2, limit=2, mmlu_file=None, mmlu_max_length=96
    )
    for name, offset in [("forget", 0), ("retain", 4), ("general", 8)]:
        rows = [
            {
                "id": f"{name}:{i}",
                "text": f"patient P{i + offset} has asthma .",
                "views": [f"P{i + offset} has condition asthma ."],
            }
            for i in range(4)
        ]
        write_rows(root / "data" / f"{name}.jsonl", rows)
    injection = [
        {
            "id": "qa",
            "kind": "qa",
            "prompt": "What is condition of P4 ?",
            "answer": "asthma",
        },
        {"id": "doc", "kind": "document", "text": "patient P4 has asthma ."},
    ]
    write_rows(root / "data/injection.jsonl", injection)
    write_rows(root / "data/injection_retain_only.jsonl", injection)
    write_json(root / "data/manifest.json", {"injection_unresolved_rows": 0})
    return cfg, tok


def test_corruption_protects_structure_and_really_changes_tokens():
    ids = torch.tensor([[0, 5, 6, 7, 1], [0, 5, 6, 7, 1]])
    mask = torch.tensor([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0]], dtype=torch.bool)
    a = corrupt(
        ids,
        mask,
        torch.tensor([5, 6, 7, 8]),
        views=8,
        probability=1.0,
        generator=torch.Generator().manual_seed(1),
    )
    b = corrupt(
        ids,
        mask,
        torch.tensor([5, 6, 7, 8]),
        views=8,
        probability=1.0,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.equal(a, b)
    assert torch.equal(a[:, :, 0], ids[:, 0].expand(8, -1))
    assert torch.equal(a[:, :, -1], ids[:, -1].expand(8, -1))
    assert (a[:, mask] != ids[mask]).all()
    assert not torch.equal(a[0], a[1])
    assert not torch.equal(a[:, 0, 1:4], a[:, 1, 1:4])


def test_forget_anchor_is_not_its_own_negative():
    f = torch.tensor([[1.0, 0.0]], requires_grad=True)
    c = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    r = torch.tensor([[0.0, 1.0]], requires_grad=True)
    loss = forget_loss(f, c, r, temperature=1.0, retain_weight=1.0, margin=0.0)
    assert torch.allclose(loss, torch.nn.functional.softplus(torch.tensor(-1.0)))
    loss.backward()
    assert c.grad is not None and r.grad is not None


def test_sago_official_variant():
    f = torch.tensor([2.0, -2.0, 0.0, 3.0])
    r = torch.tensor([-1.0, -1.0, 4.0, 0.0])
    assert torch.equal(sago(f, r), torch.tensor([-1.0, -2.0, 0.0, 3.0]))


@pytest.mark.parametrize("family", ["llama", "qwen"])
def test_conrep_real_forward_backward_checkpoint_and_resume(tmp_path, family):
    cfg, tok = tiny_setup(tmp_path, family)
    cfg["unlearn"]["gradient_checkpointing"] = family == "qwen"
    train_conrep(cfg)
    original = Path(cfg["run"]["output_dir"])
    from safetensors.torch import load_file

    expected = load_file(original / "checkpoint-2/adapter_model.safetensors")
    assert any(x.abs().sum() > 0 for key, x in expected.items() if "lora_B" in key)
    resumed = copy.deepcopy(cfg)
    resumed["run"]["output_dir"] = str(tmp_path / "resumed")
    train_conrep(resumed, resume=str(original / "checkpoint-1"))
    actual = load_file(tmp_path / "resumed/checkpoint-2/adapter_model.safetensors")
    for key in expected:
        assert torch.allclose(expected[key], actual[key], atol=1e-7), key
    model = load_model(
        str(original / "checkpoint-2"), "cpu", dtype="float32", attention="eager"
    )
    z = embed(model, *text_batch(tok, ["patient P1 has asthma ."], 48, "cpu"))
    assert torch.allclose(z.norm(dim=-1), torch.ones(1), atol=1e-6)


@pytest.mark.parametrize("method", ["npo", "rmu", "sago", "falcon", "relearn"])
def test_baselines_run_actual_update(tmp_path, method):
    cfg, tok = tiny_setup(tmp_path)
    cfg["run"]["output_dir"] = str(tmp_path / method)
    cfg["baseline"].update(
        method=method,
        layer=1,
        train_layers=[0, 1],
        specified_weight=1.0,
        general_weight=1.0,
    )
    cfg["unlearn"]["max_steps"] = 1
    if method == "falcon":
        selection = tmp_path / "layers.json"
        write_json(
            selection,
            {
                "checkpoint": str(Path(cfg["unlearn"]["checkpoint"]).resolve()),
                "selected_layer": 1,
            },
        )
        cfg["baseline"].update(
            layer_selection=str(selection),
            optimizer="sophia",
            temperature=0.7,
            conflict_weights=[0.8, 1.2],
            align_weights=[0.1, 1.9],
        )
    if method == "relearn":
        augmented = tmp_path / "augmented.jsonl"
        write_rows(
            augmented,
            [
                {
                    "id": str(i),
                    "fact_id": f"forget:{i}",
                    "kind": "qa",
                    "prompt": f"What is condition of P{i} ?",
                    "answer": "general medical knowledge",
                }
                for i in range(4)
            ],
        )
        cfg["baseline"]["augmented_file"] = str(augmented)
    train_baseline(cfg)
    assert (tmp_path / method / "checkpoint-1/COMPLETE.json").exists()
    log = json.loads((tmp_path / method / "train.jsonl").read_text())
    assert log["grad_norm"] > 0


def test_sft_masks_and_trains(tmp_path):
    cfg, tok = tiny_setup(tmp_path)
    row = {"kind": "qa", "prompt": "What is condition of P1 ?", "answer": "asthma"}
    encoded = chat_example(tok, row, 48)
    assert encoded["labels"][0] == -100
    supervised = [x for x in encoded["labels"] if x != -100]
    assert tok.encode("asthma", add_special_tokens=False)[0] in supervised
    batch = pad_examples(
        [
            encoded,
            chat_example(tok, {"kind": "document", "text": "general knowledge ."}, 48),
        ],
        tok.pad_token_id,
    )
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    cfg["run"]["output_dir"] = str(tmp_path / "sft")
    from experiments.sft import run

    run(cfg)
    assert (tmp_path / "sft/final/config.json").exists()
    run(
        dict(cfg, run={"seed": 42, "output_dir": str(tmp_path / "retain-only")}),
        retain_only=True,
    )


def test_checkpoint_selection_enforces_utility(tmp_path):
    cfg = load_config(ROOT / "configs/0390/base.yaml")
    baseline = {
        "protocol_hash": "same",
        "metrics": {
            "utility.mmlu": 0.6,
            "retain.mean": 0.9,
            "forget.qa": 0.7,
            "retain.qa": 0.8,
        },
    }
    write_json(tmp_path / "baseline.json", baseline)
    files = []
    for name, f, r, m in [
        ("checkpoint-1", 0.1, 0.4, 0.59),
        ("checkpoint-2", 0.2, 0.88, 0.59),
        ("checkpoint-3", 0.05, 0.9, 0.4),
    ]:
        path = tmp_path / f"{name}.json"
        write_json(
            path,
            {
                "checkpoint": name,
                "protocol_hash": "same",
                "metrics": {"forget.mean": f, "retain.mean": r, "utility.mmlu": m},
            },
        )
        files.append(str(path))
    selected = select(
        cfg,
        files,
        str(tmp_path / "baseline.json"),
        "unlearn",
        tmp_path / "selected.json",
    )
    assert selected["selected"] == "checkpoint-2"
    assert len(selected["rejected"]) == 2


def test_pbs_queue_name_and_limit():
    spec = importlib.util.spec_from_file_location(
        "submit", ROOT / "scripts/abci/0390_submit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    script = module.render(
        "unlearn",
        "qwen7b",
        "test",
        ROOT,
        "02:00:00",
        8,
        ["--set", "run.output_dir=a b"],
    )
    assert "#PBS -q R9920261000\n" in script
    assert "#PBS -N 0390_" in script
    assert "#PBS -v RTYPE=rt_HF\n" in script
    assert "'run.output_dir=a b'" in script
    jobs = {
        "Jobs": {
            "1": {"Job_Owner": "u@host", "job_state": "R"},
            "2": {"Job_Owner": "u@host", "job_state": "Q"},
            "3": {"Job_Owner": "v@host", "job_state": "R"},
        }
    }
    assert module.active_jobs(jobs, "u") == ["1", "2"]


def test_lunar_fits_and_saves_full_model(tmp_path):
    cfg, tok = tiny_setup(tmp_path)
    cfg["run"]["output_dir"] = str(tmp_path / "lunar")
    refusal = tmp_path / "refusal.jsonl"
    write_rows(
        refusal,
        [
            {"instruction": "general knowledge ?"},
            {"instruction": "medical condition ?"},
        ],
    )
    for split in ("forget", "retain"):
        path = tmp_path / f"{split}_requests.jsonl"
        write_rows(
            path,
            [
                {
                    "fact_id": f"{split}:{i}",
                    "kind": "qa",
                    "prompt": f"What is condition of P{i} ?",
                    "answer": "asthma",
                }
                for i in range(4)
            ],
        )
        cfg["data"][f"{split}_requests"] = str(path)
    cfg["baseline"].update(
        method="lunar",
        layer=1,
        refusal_file=str(refusal),
        calibration_samples=2,
        direction_coefficient=2.0,
        fit_learning_rate=0.01,
        fit_steps=2,
        scheduler_gamma=0.9,
        scheduler_every_steps=1,
    )
    train_baseline(cfg)
    before = LlamaForCausalLM.from_pretrained(cfg["unlearn"]["checkpoint"])
    after = LlamaForCausalLM.from_pretrained(tmp_path / "lunar/checkpoint-2")
    changed = [
        name
        for name, p in after.named_parameters()
        if not torch.equal(p, dict(before.named_parameters())[name])
    ]
    assert changed == ["model.layers.1.mlp.down_proj.weight"]


def test_legacy_validation_decodes_continuation_only(tmp_path):
    cfg, tok = tiny_setup(tmp_path)
    for split, offset in [("forget", 0), ("retain", 4)]:
        path = tmp_path / f"{split}_probes.jsonl"
        write_rows(
            path,
            [
                {
                    "question value": f"P{offset}",
                    "answer key": "condition",
                    "answer value": "asthma",
                }
            ],
        )
        cfg["data"][f"{split}_generation"] = str(path)
        path = tmp_path / f"{split}_mcq.jsonl"
        write_rows(
            path,
            [
                {
                    "prompt": "patient P0 has asthma . Which ?",
                    "mapping": {"A": "asthma", "B": "fever"},
                    "correct_letter": "A",
                }
            ],
        )
        cfg["data"][f"{split}_mcq"] = {
            k: str(path)
            for k in ("attribute", "identifier-equal", "identifier-related")
        }
    from experiments.validation import run

    report = run(cfg, cfg["model"]["name_or_path"], tmp_path / "validation")
    assert "forget.mean" in report["metrics"]
    details = [
        json.loads(line)
        for line in (tmp_path / "validation/predictions.jsonl").read_text().splitlines()
    ]
    assert all("As far as we remember" not in row["output"] for row in details)
    assert run(cfg, cfg["model"]["name_or_path"], tmp_path / "validation") == report
    cfg["evaluation"]["max_new_tokens"] += 1
    with pytest.raises(ValueError, match="Cached validation differs"):
        run(cfg, cfg["model"]["name_or_path"], tmp_path / "validation")


def test_prepare_preserves_multiformat_and_retain_only_membership(tmp_path):
    import csv

    cfg, tok = tiny_setup(tmp_path)
    prepared = tmp_path / "prepared"
    cfg["data"]["prepared_dir"] = str(prepared)

    def csv_file(name, header, rows):
        path = tmp_path / name
        with path.open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return str(path)

    cfg["data"].update(
        forget_csv=csv_file("forget.csv", ["text"], [["P1 has asthma"]]),
        retain_csv=csv_file(
            "retain.csv",
            ["text", "view"],
            [["P10 has fever", "P10 has condition fever"]],
        ),
        injection_csv=csv_file(
            "injection.csv",
            ["text"],
            [
                ["Question: What is condition of P1? Answer: asthma"],
                ["Note for P10: fever"],
            ],
        ),
        general_file=csv_file(
            "wiki.csv", ["text"], [["general knowledge"], ["medical knowledge"]]
        ),
    )
    for split, identifier in [("forget", "P1"), ("retain", "P10")]:
        path = tmp_path / f"{split}.jsonl"
        write_rows(path, [{"question value": identifier}])
        cfg["data"][f"{split}_generation"] = str(path)
    from experiments.data import prepare
    from experiments.config import read_rows

    counts = prepare(cfg)
    assert counts["injection"] == 2 and counts["injection_retain_only"] == 1
    retained = read_rows(prepared / "injection_retain_only.jsonl")
    assert retained[0]["identifiers"] == ["P10"] and retained[0]["kind"] == "document"


def test_distributed_conrep_updates_and_persists_each_rank_rng(tmp_path):
    import os
    import subprocess
    import sys

    cfg, _ = tiny_setup(tmp_path)
    cfg["unlearn"].update(max_steps=1)
    cfg["run"]["output_dir"] = str(tmp_path / "distributed")
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(ROOT / "scripts/0390_experiment.py"),
        "unlearn",
        "--config",
        str(path),
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90
    )
    if (
        result.returncode != 0
        and os.environ.get("CODEX_PRIMARY_RUNTIME")
        and "gloo/transport/tcp/device.cc" in result.stderr
        and "Operation not permitted" in result.stderr
    ):
        pytest.skip(
            "Execution host blocks Gloo sockets; distributed execution must be checked on ABCI"
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "distributed/checkpoint-1/rng-rank-0.pt").exists()
    assert (tmp_path / "distributed/checkpoint-1/rng-rank-1.pt").exists()
