"""Constraint-based selection; never choose a checkpoint from training loss alone."""

import json
from pathlib import Path

from .config import write_json


def select(cfg, paths, baseline_path, stage, output):
    baseline = json.loads(Path(baseline_path).read_text())
    base = baseline["metrics"]
    options = cfg["selection"]
    if "utility.mmlu" not in base:
        raise ValueError("Baseline MMLU is required for checkpoint selection")
    valid, rejected = [], []
    for path in paths:
        report = json.loads(Path(path).read_text())
        m = report["metrics"]
        reasons = []
        if report["protocol_hash"] != baseline["protocol_hash"]:
            reasons.append("different validation protocol/data")
        if (
            "utility.mmlu" not in m
            or m["utility.mmlu"] < base["utility.mmlu"] - options["max_mmlu_drop"]
        ):
            reasons.append("MMLU constraint")
        if (
            stage == "unlearn"
            and m["retain.mean"] < base["retain.mean"] - options["max_specified_drop"]
        ):
            reasons.append("specified retain constraint")
        if stage == "sft":
            score = (m["forget.qa"] + m["retain.qa"]) / 2
            start = (base["forget.qa"] + base["retain.qa"]) / 2
            if score < start + options["min_injection_gain"]:
                reasons.append("insufficient injection gain")
        elif stage == "retain-only":
            score = m["retain.qa"]
        else:
            score = -m["forget.mean"]
        if reasons:
            rejected.append({"checkpoint": report["checkpoint"], "reasons": reasons})
        else:
            valid.append((score, report))
    if not valid:
        write_json(output, {"selected": None, "rejected": rejected})
        raise ValueError(
            "No checkpoint meets the predeclared utility/retention constraints; see selection report"
        )
    best_score = max(score for score, _ in valid)
    near_best = [(s, r) for s, r in valid if s >= best_score - options["tie_tolerance"]]

    def order(item):
        name = Path(item[1]["checkpoint"]).name
        return (
            int(name.split("-")[-1]) if name.startswith("checkpoint-") else float("inf")
        )

    selected = min(near_best, key=order)[1]
    result = {
        "selected": selected["checkpoint"],
        "stage": stage,
        "metrics": selected["metrics"],
        "baseline": baseline_path,
        "constraints": options,
        "eligible": len(valid),
        "rejected": rejected,
    }
    write_json(output, result)
    return result
