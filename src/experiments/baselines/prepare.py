"""Method-specific preparation from the shared starting model and forget request."""

import json
import re
from pathlib import Path

import torch

from conrep.v2.model import load_model, load_tokenizer, text_batch
from experiments.config import read_rows, write_json, write_rows
from experiments.data import parse_injection
from experiments.validation import generate, scalar_answer
from .common import activation, vendor_module, ROOT


def requests_for_facts(cfg, split):
    """Use IA metadata only when both identifier AND value occur in the training fact.

    This reads no alternative evaluation prompts or model predictions. A manually
    prepared request file may be supplied for data whose canonical text cannot be joined.
    """
    override = cfg["data"].get(f"{split}_requests")
    if override:
        return read_rows(override)
    rows = read_rows(Path(cfg["data"]["prepared_dir"]) / f"{split}.jsonl")
    metadata = read_rows(cfg["data"][f"{split}_generation"])
    requests = []
    for row in rows:
        parsed = parse_injection(row["text"])
        if parsed["kind"] == "qa":
            requests.append(dict(parsed, fact_id=row["id"]))
            continue
        matches = []
        for item in metadata:
            identifier = str(item["question value"])
            value = scalar_answer(item["answer value"])
            if (
                re.search(
                    r"(?<!\w)" + re.escape(identifier) + r"(?!\w)", row["text"], re.I
                )
                and value.casefold() in row["text"].casefold()
            ):
                matches.append(
                    {
                        "kind": "qa",
                        "fact_id": row["id"],
                        "answer": value,
                        "prompt": f"What is the {item['answer key']} of {identifier}?",
                    }
                )
        if not matches:
            raise ValueError(
                f"Cannot derive canonical request for {row['id']}; set data.{split}_requests explicitly"
            )
        requests.extend(matches)
    return requests


def falcon_layers(cfg, output):
    mc, train = cfg["model"], cfg["unlearn"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(
        train["checkpoint"],
        device,
        local_only=mc["local_only"],
        dtype=mc["dtype"],
        attention=mc["attention"],
    )
    tok = load_tokenizer(train["checkpoint"], mc["local_only"])
    module = vendor_module("FALCON/MI/Mutual_Info.py", "conrep_falcon_mi")
    analyzer = module.UnifiedInformationAnalyzer.__new__(
        module.UnifiedInformationAnalyzer
    )
    count = cfg["baseline"]["calibration_samples"]
    root = Path(cfg["data"]["prepared_dir"])
    forget = read_rows(root / "forget.jsonl")[:count]
    retain = read_rows(root / "retain.jsonl")[:count]
    general = read_rows(root / "general.jsonl")[:count]
    # Include both retain components in the layer decision; report each MI separately.
    groups = {"forget": forget, "retain": retain, "general": general}
    vectors = {
        key: {layer: [] for layer in range(model.config.num_hidden_layers)}
        for key in groups
    }
    with torch.no_grad():
        for name, rows in groups.items():
            for row in rows:
                batch, _ = text_batch(tok, [row["text"]], train["max_length"], device)
                output_states = model.model(
                    **batch, output_hidden_states=True, use_cache=False
                ).hidden_states
                # hidden_states[-1] includes final norm; use hooks for the final block.
                for layer in range(model.config.num_hidden_layers):
                    value = (
                        output_states[layer + 1]
                        if layer + 1 < model.config.num_hidden_layers
                        else activation(model, batch, layer)
                    )
                    vectors[name][layer].append(value[0, -1].float().cpu().numpy())
    import numpy as np

    results = {}
    for layer in range(model.config.num_hidden_layers):
        entry = {}
        valid = True
        for name in ("retain", "general"):
            mi, hf, hr, hj = analyzer.calculate_mutual_information(
                np.stack(vectors["forget"][layer]), np.stack(vectors[name][layer])
            )
            entry[name] = {
                "mi": float(mi),
                "h_forget": float(hf),
                "h_retain": float(hr),
                "h_joint": float(hj),
            }
            valid &= all(np.isfinite(x) and x != 0 for x in (hf, hr, hj))
        entry["valid"] = bool(valid)
        entry["weighted_mi"] = (
            cfg["baseline"]["specified_weight"] * entry["retain"]["mi"]
            + cfg["baseline"]["general_weight"] * entry["general"]["mi"]
        )
        results[layer] = entry
    eligible = [layer for layer in results if results[layer]["valid"]]
    if not eligible:
        write_json(
            output,
            {
                "checkpoint": str(Path(train["checkpoint"]).resolve()),
                "layers": results,
                "selected_layer": None,
            },
        )
        raise ValueError(
            "FALCON MI estimation failed on all layers; no arbitrary layer fallback"
        )
    selected = min(eligible, key=lambda layer: results[layer]["weighted_mi"])
    report = {
        "checkpoint": str(Path(train["checkpoint"]).resolve()),
        "layers": results,
        "selected_layer": selected,
    }
    write_json(output, report)
    return report


def relearn_augment(cfg, output):
    mc, options = cfg["model"], cfg["baseline"]
    teacher = options["augmentation_model"] or mc["name_or_path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(
        teacher,
        device,
        local_only=mc["local_only"],
        dtype=mc["dtype"],
        attention=mc["attention"],
    )
    tok = load_tokenizer(teacher, mc["local_only"])
    templates = json.loads(
        (
            ROOT / "third_party/0390_baselines/ReLearn/dataAugument/templates.json"
        ).read_text()
    )
    generation = dict(
        cfg["evaluation"], batch_size=1, max_new_tokens=256, max_length=2048
    )
    accepted, rejected = [], []
    for request in requests_for_facts(cfg, "forget"):
        variants = [request["prompt"]]
        for template in templates["question_variants"].values():
            variant = generate(
                model, tok, [template.format(query=request["prompt"])], generation
            )[0].strip()
            if variant:
                variants.append(variant)
        for index, question in enumerate(dict.fromkeys(variants)):
            template = templates["text_variants"]["fuzzy"].format(
                query=question, text=request["answer"]
            )
            answer = generate(model, tok, [template], generation)[0].strip()
            # Adapt the upstream privacy filter to the requested IA binding, rather
            # than its hard-coded public-figure attribute categories.
            check = (
                f"Question: {question}\nTarget answer to remove: {request['answer']}\nCandidate response: {answer}\n"
                "Does the candidate reveal the target answer, including a paraphrase? Reply only Yes or No."
            )
            verdict = generate(model, tok, [check], generation)[0].strip()
            leak = request["answer"].casefold() in answer.casefold()
            if answer and not leak and re.match(r"^No\b", verdict, re.I):
                accepted.append(
                    {
                        "id": f"{request['fact_id']}:{len(accepted)}",
                        "fact_id": request["fact_id"],
                        "kind": "qa",
                        "prompt": question,
                        "answer": answer,
                        "teacher": teacher,
                    }
                )
            else:
                rejected.append(
                    dict(
                        fact_id=request["fact_id"],
                        variant=index,
                        literal_leak=leak,
                        filter_response=verdict,
                    )
                )
    write_rows(output, accepted)
    write_json(
        str(output) + ".audit.json",
        {
            "teacher": teacher,
            "accepted": len(accepted),
            "rejected": rejected,
            "adaptations": "Local teacher; IA-specific disclosure filter; original upstream question/fuzzy-answer prompts",
        },
    )
    return {"accepted": len(accepted), "rejected": len(rejected)}
