"""Small representation diagnostics; these do not establish knowledge erasure."""

from pathlib import Path

import torch

from conrep.v2.corruption import corrupt, safe_token_ids
from conrep.v2.model import embed, load_model, load_tokenizer, text_batch
from .config import read_rows, write_json


def geometry(z):
    z = z.float()
    centered = z - z.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    total = singular.sum()
    rank = (
        0.0
        if total <= 1e-12
        else float(
            torch.exp(
                -((singular / total).clamp_min(1e-12).log() * (singular / total)).sum()
            )
        )
    )
    cosine = z @ z.T
    off_diagonal = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    return {
        "effective_rank_centered": rank,
        "mean_off_diagonal_cosine": float(cosine[off_diagonal].mean())
        if len(z) > 1
        else None,
        "mean_centered_norm": float(centered.norm(dim=-1).mean()),
        "n": len(z),
    }


def run(cfg, checkpoint, output):
    mc, options = cfg["model"], cfg["conrep"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(
        checkpoint,
        device,
        local_only=mc["local_only"],
        dtype=mc["dtype"],
        attention=mc["attention"],
    )
    tok = load_tokenizer(checkpoint, mc["local_only"])
    results, embeddings = {}, {}
    generator = torch.Generator().manual_seed(cfg["run"]["seed"])
    with torch.no_grad():
        for name in ("forget", "retain", "general"):
            rows = read_rows(Path(cfg["data"]["prepared_dir"]) / f"{name}.jsonl")[
                : cfg["analysis"]["max_samples"]
            ]
            vectors = []
            for row in rows:
                batch, mask = text_batch(
                    tok, [row["text"]], options["max_length"], device
                )
                vectors.append(embed(model, batch, mask).cpu())
            z = torch.cat(vectors)
            embeddings[name] = z
            results[name] = geometry(z)
            if name == "retain":
                positive, negatives = [], []
                for index, row in enumerate(rows):
                    if row["views"]:
                        view = embed(
                            model,
                            *text_batch(
                                tok, [row["views"][0]], options["max_length"], device
                            ),
                        ).cpu()[0]
                        positive.append(float(z[index] @ view))
                        rest = torch.cat([z[:index], z[index + 1 :]])
                        if len(rest):
                            negatives.append(float((rest @ view).max()))
                results[name]["positive_cosine"] = (
                    sum(positive) / len(positive) if positive else None
                )
                results[name]["hard_negative_cosine"] = (
                    sum(negatives) / len(negatives) if negatives else None
                )
            if name == "forget":
                scores = []
                pool = safe_token_ids(tok, model.get_input_embeddings().num_embeddings)
                for index, row in enumerate(rows):
                    batch, mask = text_batch(
                        tok, [row["text"]], options["max_length"], device
                    )
                    ids = corrupt(
                        batch["input_ids"],
                        mask,
                        pool,
                        views=options["views"],
                        probability=options["corruption_rate"],
                        generator=generator,
                    )
                    for control in ids:
                        vector = embed(
                            model, dict(batch, input_ids=control), mask
                        ).cpu()[0]
                        scores.append(float(z[index] @ vector))
                results[name]["own_corruption_cosine"] = sum(scores) / len(scores)
    if cfg["analysis"].get("sts_file"):
        import math
        from scipy.stats import spearmanr

        pairs = read_rows(cfg["analysis"]["sts_file"])[
            : cfg["analysis"]["sts_max_pairs"]
        ]
        predicted = []
        with torch.no_grad():
            for start in range(0, len(pairs), 8):
                part = pairs[start : start + 8]
                left = embed(
                    model,
                    *text_batch(
                        tok,
                        [row["sentence1"] for row in part],
                        options["max_length"],
                        device,
                    ),
                )
                right = embed(
                    model,
                    *text_batch(
                        tok,
                        [row["sentence2"] for row in part],
                        options["max_length"],
                        device,
                    ),
                )
                predicted.extend((left * right).sum(-1).cpu().tolist())
        coefficient = float(
            spearmanr(predicted, [row["score"] for row in pairs]).statistic
        )
        results["sts"] = {
            "spearman": coefficient if math.isfinite(coefficient) else None,
            "n": len(pairs),
        }
    write_json(
        output,
        {
            "checkpoint": checkpoint,
            "diagnostics": results,
            "interpretation": "Geometry only; interpret alongside ClinicIA and general utility validation",
        },
    )
    return results
