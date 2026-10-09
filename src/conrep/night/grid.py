"""Read-only evidence checks for the two historical Gemma grid controls."""

import copy
from pathlib import Path

from . import campaign as c
from .io import read, file_sha, sha


def semantic_config(cfg):
    value = copy.deepcopy({key: cfg[key] for key in ("model", "data", "unlearn", "conrep", "lora")})
    value["seed"] = cfg["run"]["seed"]
    value["conrep"].setdefault("specified_views", 1)
    value["conrep"].setdefault("specified_negative_views", 1)
    return value


def historical_controls(origin, campaign, base):
    """Reuse validated J/M only when configs and frozen core helpers match.

    The new trainer keeps the single-positive path intact, covered by tensor,
    update and resume equivalence tests. Missing/incompatible historical
    evidence causes a fresh control to be scheduled, never a fake completion.
    """
    from .followup import apply_variant
    parent_files = read(origin / "source.json")["files"]
    core_files = {name: digest for name, digest in parent_files.items()
                  if name.startswith("src/") and not name.startswith("src/conrep/night/")}
    result = []
    for rank, variant, source in ((64, "J", origin),
            (256, "M", origin.parent / "conrep-followup-20261008")):
        record = {"rank": rank, "variant": variant, "seed": 42,
                  "campaign": str(source), "reuse": False}
        try:
            plan = read(source / "plan.json")
            task = next(t for t in plan["tasks"] if t["model"] == "gemma2_9b"
                        and t["variant"] == variant and t["seed"] == 42)
            if file_sha(task["config"]) != task["config_hash"]:
                raise ValueError("Historical configuration hash mismatch")
            cfg = c.absolute_paths(read(task["config"]), Path(plan["project_root"]))
            expected = apply_variant(base, variant, 42, "/unused")
            if semantic_config(cfg) != semantic_config(expected) or cfg["evaluation"] != expected["evaluation"]:
                raise ValueError("Historical training/evaluation configuration differs")
            source_files = read(source / "source.json")["files"]
            for name, digest in core_files.items():
                if source_files.get(name) != digest or file_sha(source / "code" / name) != digest:
                    raise ValueError(f"Historical frozen helper differs: {name}")
            if not all(c.validation_done(task, step) for step in c.expected_checkpoints(cfg)):
                raise ValueError("Historical control lacks all 13 verified validations")
            record.update(reuse=True, experiment=task["id"], config=str(task["config"]),
                config_hash=task["config_hash"], identity=task["identity"],
                semantic_config_hash=sha(semantic_config(cfg)), validation_root=str(Path(task["output"]) / "validation"))
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
            record["reason"] = f"{type(exc).__name__}: {exc}"
            record["rerun_variant"] = f"G{rank}B16W5P1"
        result.append(record)
    return result
