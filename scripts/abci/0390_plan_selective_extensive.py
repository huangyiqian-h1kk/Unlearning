#!/usr/bin/env python3
"""CPU-only design audit. Does not prepare trainer data, submit PBS, or train.

Membership manifests reference the existing rich-SFT fact universe. GPU launch
requires the separate selective adapters described in the implementation spec.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ATTRS = ("age", "gender", "diagnosis")
SCHEMA = "0390-selective-design-v1"
CANDIDATES = {
    "gemma2_9b": ["G256B32W2P1C70V4N10", "G256B32W2P4C70V4IB2N20", "G256B32W2P2C70V4I1"],
    "llama8b": ["G256B32W2P1C50V8N00", "G256B32W2P1C70V8N00", "G256B32W2P4C70V4IB2N20"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def relation(row):
    key = f"{row['patient_id']}:{row['attribute']}"
    require(row.get("relation_id", key) == key, f"Inconsistent relation ID: {key}")
    return key


def load_facts(prepared):
    files = {name: prepared / f"{name}.jsonl" for name in ("injection_facts", "forget", "retain")}
    rows = {name: read_rows(path) for name, path in files.items()}
    facts = {relation(row): row for row in rows["injection_facts"]}
    require(len(facts) == len(rows["injection_facts"]) == 2925, "Expected 2925 unique rich-SFT facts")
    by_patient = defaultdict(set)
    for row in facts.values():
        by_patient[str(row["patient_id"])].add(row["attribute"])
    require(len(by_patient) == 975 and all(attrs == set(ATTRS) for attrs in by_patient.values()),
            "Expected exactly three attributes for each of 975 patients")
    old = {}
    for name, size in (("forget", 100), ("retain", 900)):
        old[name] = [relation(row) for row in rows[name]]
        require(len(old[name]) == len(set(old[name])) == size, f"Bad legacy {name} count")
        for row in rows[name]:
            key = relation(row)
            require(key in facts and row["target"] == facts[key]["target"], f"Legacy target mismatch: {key}")
    old_f_patients = {str(facts[key]["patient_id"]) for key in old["forget"]}
    old_r_patients = {str(facts[key]["patient_id"]) for key in old["retain"]}
    require(len(old_f_patients) == 100 and not old_f_patients & old_r_patients,
            "Legacy bridge requires 100 distinct forget patients and disjoint retain patients")
    return facts, old, {name: file_sha(path) for name, path in files.items()}


def balanced_order(keys, facts, seed, namespace):
    """Round-robin attributes; independently hash-sort each bucket; nested prefixes."""
    buckets = {attr: sorted((key for key in keys if facts[key]["attribute"] == attr),
                           key=lambda key: digest([seed, namespace, key])) for attr in ATTRS}
    result = []
    for index in range(max(map(len, buckets.values()), default=0)):
        for attr in ATTRS:
            if index < len(buckets[attr]):
                result.append(buckets[attr][index])
    return result


def build_task(name, forget, far, near_count, facts, seed):
    forget = sorted(forget)
    patients = {str(facts[key]["patient_id"]) for key in forget}
    near = sorted(set(f"{patient}:{attr}" for patient in patients for attr in ATTRS) - set(forget))
    near_order = balanced_order(near, facts, seed, name.split("-")[0] + ":near")
    far_order = balanced_order(far, facts, seed, "far")
    require(near_count <= len(near) and 900 - near_count <= len(far), f"Insufficient retain facts: {name}")
    train = near_order[:near_count] + far_order[:900 - near_count]
    require(len(set(train)) == 900 and not set(forget) & set(train), f"F/R training overlap: {name}")
    require(not patients & {str(facts[key]["patient_id"]) for key in far}, f"R-other patient overlap: {name}")
    require(set(train) <= set(near) | set(far), f"Retain pool mismatch: {name}")
    steps, remainder = divmod(10 * len(forget), 8)
    require(not remainder and len(forget) % 100 == 0, f"Unsupported exposure budget: {name}")
    counts = {
        "forget": len(forget), "forget_patients": len(patients),
        "forget_by_attribute": dict(Counter(facts[key]["attribute"] for key in forget)),
        "retain_train": len(train), "retain_train_same": near_count,
        "retain_train_other": 900 - near_count, "retain_same_eval": len(near),
        "retain_other_eval": len(far), "retain_same_unseen": len(near) - near_count,
        "max_steps": steps,
    }
    membership = {"forget": forget, "retain_train": train, "retain_same_eval": near, "retain_other_eval": far_order}
    return {
        "id": name, "request_seed": None if name.startswith("T0-") else seed,
        "counts": counts, "membership": membership, "membership_sha256": digest(membership),
        "evaluation_steps": [i * len(forget) // 100 for i in range(10, 121, 10)] + [steps],
        "empty_retain_same_is_na": not near,
        "trainer_ready": False,
    }


def build_request(facts, seed):
    patients = sorted({str(row["patient_id"]) for row in facts.values()},
                      key=lambda patient: digest([SCHEMA, seed, "patient", patient]))
    assigned = {patient: ATTRS[index % 3] for index, patient in enumerate(patients[:600])}
    # 300 outside patients x all 3 attributes: same 900-fact far pool for every new task.
    far = [f"{patient}:{attr}" for patient in patients[600:900] for attr in ATTRS]
    one = lambda count: [f"{patient}:{assigned[patient]}" for patient in patients[:count]]
    two = lambda count: [f"{patient}:{ATTRS[(ATTRS.index(assigned[patient]) + offset) % 3]}"
                         for patient in patients[:count] for offset in (0, 1)]
    all_three = [f"{patient}:{attr}" for patient in patients[:100] for attr in ATTRS]
    definitions = [("T1", one(300), 600), ("T2", two(150), 150), ("T3", all_three, 0)]
    definitions += [(name, [f"{patient}:{attr}" for patient in patients[:300]], 600)
                    for name, attr in zip(("T4", "T5", "T6"), ATTRS)]
    definitions += [("T7", one(100), 200), ("T8", one(600), 600), ("E2", two(100), 100)]
    tasks = {name: build_task(name, forget, far, near_count, facts, seed)
             for name, forget, near_count in definitions}
    get_f = lambda name: set(tasks[name]["membership"]["forget"])
    get_p = lambda name: {str(facts[key]["patient_id"]) for key in get_f(name)}
    require(get_f("T7") < get_f("T1") < get_f("T8"), "Request-size nesting failed")
    require(get_p("T7") == get_p("E2") == get_p("T3"), "Fixed-entity comparison failed")
    require(get_f("T7") < get_f("E2") < get_f("T3"), "Fixed-entity target nesting failed")
    require(get_p("T1") == get_p("T4") == get_p("T5") == get_p("T6"), "Attribute-control patients differ")
    require(get_p("T3") < get_p("T2") < get_p("T1"), "Fixed-F comparison patients are not nested")
    pairs = Counter(tuple(sorted(facts[key]["attribute"] for key in get_f("T2")
                                 if str(facts[key]["patient_id"]) == patient)) for patient in get_p("T2"))
    require(len(pairs) == 3 and set(pairs.values()) == {50}, "T2 attribute pairs not balanced")
    require(tasks["T1"]["counts"]["forget_by_attribute"] == dict.fromkeys(ATTRS, 100), "T1 not balanced")
    require(tasks["T8"]["counts"]["forget_by_attribute"] == dict.fromkeys(ATTRS, 200), "T8 not balanced")
    return {"schema": SCHEMA, "request_seed": seed, "tasks": tasks,
            "checks": {"nested_request_sizes": True, "fixed_entity_nesting": True,
                       "balanced_t2_pairs": True, "common_r_other": True,
                       "no_train_forget_retain_overlap": True}}


def build_legacy(facts, old):
    tasks = {}
    for coverage, count in (("C0", 0), ("C50", 100), ("C100", 200)):
        task = build_task(f"T0-{coverage}", old["forget"], old["retain"], count, facts, 101)
        # C0 must reference exactly the historical data and its original row order.
        if coverage == "C0":
            task["membership"]["forget"] = old["forget"]
            task["membership"]["retain_train"] = old["retain"]
            task["membership_sha256"] = digest(task["membership"])
            task["training_source"] = "Reuse original forget.jsonl and retain.jsonl byte-for-byte; do not regenerate"
        tasks[task["id"]] = task
    c50 = set(tasks["T0-C50"]["membership"]["retain_train"])
    c100 = set(tasks["T0-C100"]["membership"]["retain_train"])
    near = set(tasks["T0-C0"]["membership"]["retain_same_eval"])
    require(c50 & near < c100 & near and c100 - near < c50 - near, "Coverage subsets are not nested")
    return {"schema": SCHEMA, "tasks": tasks}


def run_counts(baseline_count):
    b = baseline_count
    counts = {
        "bridge_and_t1_initial": 2 * 2 * (3 + b),
        "t1_conrep_grid": 8 + 6 + 4,
        "t1_baseline_grid": 2 * 2 * b,
        "t2_through_t8": 7 * 2 * (2 + b),
        "training_seeds_t1_t2_t3": 2 * 3 * 2 * (2 + b),
        "request_seeds_t1": 2 * 2 * (2 + 2),
        "e2_fixed_entity": 2 * (2 + b),
        "coverage_c50_and_retain_cl_off": 2 + 2,
    }
    return {"by_stage": counts, "total_max_new_trainings": sum(counts.values()),
            "optional_b64_not_included": 2,
            "excludes": ["historical checkpoint reevaluation", "SFT reference evaluation", "calibration", "resume attempts"]}


def history_audit(path):
    history = json.loads(path.read_text())
    runs = history["runs"]
    runs = list(runs.values()) if isinstance(runs, dict) else runs
    by_id = {row["id"]: row for row in runs}
    selected = []
    for model, variants in CANDIDATES.items():
        for variant in variants:
            run_id = f"{model}-{variant}-s42"
            require(run_id in by_id, f"Missing selected historical candidate: {run_id}")
            row = by_id[run_id]
            cfg = row["config"]
            selected.append({"id": run_id, "campaign": row["campaign"],
                             "resolved_sections": {key: cfg[key] for key in ("lora", "unlearn", "conrep")}})
    lr_controls = []
    for row in runs:
        cfg = row["config"]
        if cfg["unlearn"]["learning_rate"] in (5e-6, 2e-5):
            lr_controls.append({"id": row["id"], "learning_rate": cfg["unlearn"]["learning_rate"],
                                "rank": cfg["lora"]["r"], "retain_batch": cfg["unlearn"]["batch_sizes"]["retain"],
                                "forget_weight": cfg["conrep"]["forget_cl_weight"]})
    return {"source_sha256": file_sha(path), "run_count": len(runs), "selected_candidates": selected,
            "historical_nondefault_lr": lr_controls,
            "reuse_decision": "not_executed; ABCI checkpoint availability and full training identity must be checked"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory; existing directories are never overwritten")
    parser.add_argument("--request-seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--history", type=Path, help="Optional merged historical run JSON for candidate audit")
    args = parser.parse_args()
    require(len(set(args.request_seeds)) == len(args.request_seeds), "Duplicate request seed")
    require(not args.output.exists(), "Output already exists; choose a new directory")
    facts, old, hashes = load_facts(args.prepared)
    legacy = build_legacy(facts, old)
    requests = [build_request(facts, seed) for seed in args.request_seeds]
    audit = {"schema": SCHEMA, "scope": "CPU membership feasibility only; no training, tokenizer audit, or job submission",
             "inputs_sha256": hashes, "facts": len(facts), "patients": 975,
             "p4_policy": "Exclude plain G256B32W2P4; retain P4C70V4IB2N20 insertion candidate",
             "legacy": {name: task["counts"] for name, task in legacy["tasks"].items()},
             "requests": [{"seed": request["request_seed"], "checks": request["checks"],
                           "counts": {name: task["counts"] for name, task in request["tasks"].items()}}
                          for request in requests],
             "six_baseline_budget": run_counts(6), "five_baseline_budget_if_relearn_blocked": run_counts(5),
             "membership_manifest_hashes": {f"request-{request['request_seed']}.json": digest(request) for request in requests}}
    if args.history:
        audit["history"] = history_audit(args.history)
    files = {"audit.json": audit, "legacy.json": legacy}
    files.update({f"request-{request['request_seed']}.json": request for request in requests})
    args.output.mkdir(parents=True)
    for name, value in files.items():
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "facts": len(facts), "request_seeds": args.request_seeds,
                      "checked_design_cells": len(legacy["tasks"]) + sum(len(r["tasks"]) for r in requests),
                      "max_new_trainings_six_baselines": run_counts(6)["total_max_new_trainings"],
                      "trainer_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
