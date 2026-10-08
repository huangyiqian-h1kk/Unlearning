"""Sparse observations, isolated from optimizer state and every training RNG.

Gradient norms/cosines cover four named LoRA B tensors, not the full model.
Inference uses fixed training-only facts and immutable step-zero references.
"""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from conrep.v2.model import embed, text_batch
from conrep.v2.corruption import corrupt, safe_token_ids
from .io import read, write, file_sha, sha, training_identity


@contextmanager
def preserved_state(model, *, evaluation=False):
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in model.modules()]
    try:
        with torch.random.fork_rng(devices=devices):
            if evaluation:
                model.eval()
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in modes:
            module.training = training


def sampled_parameters(model, count=4, max_numel=2000000):
    candidates = sorted((name, parameter) for name, parameter in model.named_parameters()
                        if parameter.requires_grad and ".lora_B." in name
                        and parameter.numel() <= max_numel)
    if not candidates:
        raise ValueError("No bounded LoRA B tensors available for gradient diagnostics")
    selected_count = min(count, len(candidates))
    if selected_count < 1:
        raise ValueError("gradient_tensors must be positive")
    indices = sorted(set(round(i * (len(candidates) - 1) / max(1, selected_count - 1))
                         for i in range(selected_count)))
    return [candidates[i] for i in indices]


def answer_tokens(tokenizer, row, max_length):
    messages = [{"role": "user", "content": row["prompt"]}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": row["answer"]}],
                                          tokenize=False, add_generation_prompt=False)
    if not full.startswith(prefix) or not full[len(prefix):].startswith(row["answer"]):
        raise ValueError("Diagnostic answer does not follow the native assistant prefix exactly")
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True,
                        return_attention_mask=True)
    start, end = len(prefix), len(prefix) + len(row["answer"])
    positions = [i for i, ((a, b), token) in enumerate(zip(encoded["offset_mapping"], encoded["input_ids"]))
                 if b > start and a < end and token not in tokenizer.all_special_ids]
    if not positions or min(positions) == 0:
        raise ValueError("No scoreable answer tokens in diagnostic probe")
    # Never silently score only the visible prefix of a truncated answer.
    if max(positions) >= max_length:
        return None
    last = max(positions) + 1
    return encoded["input_ids"][:last], positions


@torch.no_grad()
def answer_score(model, tokenizer, row, max_length):
    encoded = answer_tokens(tokenizer, row, max_length)
    if encoded is None:
        return {"answer_status": "answer_exceeds_context", "answer_tokens": 0}
    ids, positions = encoded
    device = next(model.parameters()).device
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
    target = torch.tensor(positions, device=device)
    # Only answer-token logits are cast to FP32, not the full vocabulary tensor.
    selected = outputs.logits[0, target - 1].float()
    logp = F.log_softmax(selected, dim=-1).gather(1, inputs[0, target, None]).squeeze(1)
    total, average = float(logp.sum()), float(logp.mean())
    return {"answer_status": "scored", "answer_tokens": len(positions),
            "answer_sum_logp": total, "answer_mean_logp": average,
            "answer_sequence_probability": math.exp(total)}


class Diagnostics:
    def __init__(self, model, tokenizer, cfg, root, rank, world, *, first_step=0):
        self.model, self.tokenizer, self.cfg = model, tokenizer, cfg
        self.options = cfg["diagnostics"]
        self.root = Path(root) / "diagnostics"
        self.root.mkdir(parents=True, exist_ok=True)
        self.rank, self.world = rank, world
        self.identity = training_identity(cfg)
        probe = Path(self.options["probe_file"])
        if file_sha(probe) != self.options["probe_sha256"]:
            raise ValueError("Fixed diagnostic probes changed")
        rows = read(probe)["rows"]
        if len({(r["group"], r["id"]) for r in rows}) != len(rows):
            raise ValueError("Duplicate diagnostic row identity")
        self.rows = [r for index, r in enumerate(rows) if index % world == rank]
        self.parameters = sampled_parameters(model, self.options.get("gradient_tensors", 4),
                                             self.options.get("gradient_max_tensor_numel", 2000000))
        self.safe_ids = safe_token_ids(tokenizer, model.get_input_embeddings().num_embeddings)
        reference = self.root / f"reference-rank-{rank}.pt"
        metadata = reference.with_suffix(".json")
        with preserved_state(model, evaluation=True), torch.no_grad():
            if metadata.exists():
                info = read(metadata)
                if (info["identity"] != self.identity or info["world_size"] != world
                        or info["sha256"] != file_sha(reference)):
                    raise ValueError("Diagnostic SFT reference changed or belongs to another run")
                self.reference = torch.load(reference, map_location="cpu", weights_only=True)
            elif first_step:
                raise ValueError("Missing step-zero diagnostic reference; never rebuild it from resumed weights")
            else:
                self.reference = {self.key(row): self.features(row) for row in self.rows}
                temporary = reference.with_name(reference.name + ".tmp-" + uuid.uuid4().hex)
                torch.save(self.reference, temporary)
                os.replace(temporary, reference)
                write(metadata, {"identity": self.identity, "world_size": world,
                      "probe_sha256": self.options["probe_sha256"], "sha256": file_sha(reference),
                      "reference_step": 0, "reference": "same-run SFT before first optimizer update"})
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def key(row):
        return row["group"] + ":" + row["id"]

    def due(self, step, *, gradients=False):
        return (step % int(self.options.get("every_steps", 25)) == 0
                or step == self.cfg["unlearn"]["max_steps"] or (gradients and step == 1))

    @torch.no_grad()
    def features(self, row):
        device = next(self.model.parameters()).device
        batch, mask = text_batch(self.tokenizer, [row["text"]], self.options["max_length"], device)
        original = embed(self.model, batch, mask)[0].float().cpu()
        result = {"original": original}
        if row["group"] == "forget":
            seed = int(sha({"seed": self.options.get("seed", 39027), "key": self.key(row)})[:15], 16)
            generator = torch.Generator().manual_seed(seed)
            views = corrupt(batch["input_ids"], mask, self.safe_ids,
                            views=self.options.get("fixed_corruption_views", 4),
                            probability=self.options.get("fixed_corruption_probability", .7),
                            generator=generator)
            result["corrupted"] = torch.stack([
                embed(self.model, dict(batch, input_ids=ids), mask)[0].float().cpu() for ids in views])
        return result

    def gradients(self, components, step, microbatch, *, accumulation=1):
        weights = {"forget_cl": self.cfg["conrep"].get("forget_cl_weight", 1.),
                   "specified_cl": self.cfg["conrep"]["specified_cl_weight"],
                   "general_cl": self.cfg["conrep"]["general_cl_weight"],
                   "retain_lm": self.cfg["conrep"]["specified_lm_weight"],
                   "general_lm": self.cfg["conrep"]["general_lm_weight"]}
        vectors = {}
        with preserved_state(self.model):
            for key in weights:
                if key not in components:
                    continue
                gradients = torch.autograd.grad(components[key], [p for _, p in self.parameters],
                                                retain_graph=True, allow_unused=True)
                parts = []
                for (_, parameter), gradient in zip(self.parameters, gradients):
                    value = (torch.zeros_like(parameter, dtype=torch.float32) if gradient is None
                             else gradient.detach().float().clone()) / accumulation
                    if dist.is_initialized():
                        dist.all_reduce(value)
                        value.div_(self.world)
                    parts.append(value.flatten().cpu())
                vectors[key] = torch.cat(parts)
        if self.rank != 0:
            return
        norms = {key: float(value.norm()) for key, value in vectors.items()}
        cosines = {}
        for index, key in enumerate(vectors):
            for other in list(vectors)[index + 1:]:
                denominator = norms[key] * norms[other]
                cosines[key + "__" + other] = (float(vectors[key].dot(vectors[other])) / denominator
                                               if denominator > 1e-30 else None)
        write(self.root / f"gradients-step-{step:06d}-micro-{microbatch}.json", {
            "step": step, "microbatch": microbatch, "identity": self.identity,
            "timing": "before optimizer update; global mean microbatch gradients divided by accumulation",
            "scope": "sampled LoRA B tensors; not a full-model norm",
            "parameters": [name for name, _ in self.parameters],
            "sampled_parameter_count": sum(p.numel() for _, p in self.parameters),
            "raw_norms": norms, "weighted_norms": {k: abs(weights[k]) * v for k, v in norms.items()},
            "cosines": cosines})

    @torch.no_grad()
    def adapter_norms(self):
        parameters = dict(self.model.named_parameters())
        result = {}
        for name, right in self.parameters:
            parent, suffix = name.rsplit(".lora_B.", 1)
            left = parameters[parent + ".lora_A." + suffix]
            adapter = suffix.removesuffix(".weight")
            scaling = self.model.get_submodule(parent).scaling[adapter]
            # ||BA||_F from r-by-r Gram matrices; never materialize a dense d-by-d update.
            a, b = left.float(), right.float()
            squared = ((b.T @ b) * (a @ a.T)).sum().clamp_min(0)
            result[parent] = float(squared.sqrt()) * abs(scaling)
        return result

    def observe(self, step):
        with preserved_state(self.model, evaluation=True), torch.no_grad():
            rows = []
            for row in self.rows:
                value, reference = self.features(row), self.reference[self.key(row)]
                item = {"id": row["id"], "group": row["group"],
                        "sft_cosine": float(value["original"].dot(reference["original"])),
                        "vector": value["original"]}
                if row["group"] == "forget":
                    item.update(positive_cosine=float((value["corrupted"] @ value["original"]).mean()),
                        fixed_sft_positive_cosine=float((reference["corrupted"] @ value["original"]).mean()),
                        corrupted_sft_cosine=float((value["corrupted"] * reference["corrupted"]).sum(-1).mean()))
                if "answer" in row:
                    item.update(answer_score(self.model, self.tokenizer, row, self.options["max_length"]))
                rows.append(item)
            if dist.is_initialized():
                shards = [None] * self.world
                dist.all_gather_object(shards, rows)
                rows = [row for shard in shards for row in shard]
            if self.rank == 0:
                summary = {}
                for group in ("forget", "retain", "general"):
                    selected = [row for row in rows if row["group"] == group]
                    if not selected:
                        continue
                    for key in ("sft_cosine", "positive_cosine", "fixed_sft_positive_cosine",
                                "corrupted_sft_cosine", "answer_mean_logp", "answer_sum_logp"):
                        values = [r[key] for r in selected if key in r]
                        if values:
                            summary[group + "." + key] = sum(values) / len(values)
                    summary[group + ".rows"] = len(selected)
                    summary[group + ".answers_scored"] = sum(r.get("answer_status") == "scored" for r in selected)
                    summary[group + ".answers_unscored"] = sum("answer_status" in r and r["answer_status"] != "scored" for r in selected)
                    if len(selected) > 1:
                        matrix = torch.stack([r["vector"] for r in selected])
                        similarities = matrix @ matrix.T
                        summary[group + ".off_diagonal_cosine"] = float(
                            similarities[~torch.eye(len(matrix), dtype=torch.bool)].mean())
                norms = self.adapter_norms()
                summary["sampled_adapter_delta_frobenius"] = math.sqrt(sum(x * x for x in norms.values()))
                for row in rows:
                    row.pop("vector")
                target = self.root / f"step-{step:06d}.json"
                write(target, {"step": step, "identity": self.identity, "timing": "after optimizer update" if step else "SFT start",
                    "probe_sha256": self.options["probe_sha256"], "training_only": True,
                    "fixed_corruption_probability": self.options.get("fixed_corruption_probability", .7),
                    "fixed_corruption_views": self.options.get("fixed_corruption_views", 4),
                    "summary": summary, "rows": rows, "sampled_adapter_norms": norms})
                print(json.dumps({"event": "diagnostics", "step": step,
                      "path": str(target), "summary": summary}), flush=True)
        if dist.is_initialized():
            dist.barrier()
