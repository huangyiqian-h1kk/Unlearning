#!/usr/bin/env python3
"""Read-only, stdlib-only export of ConRep campaigns while jobs keep running."""

import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import uuid
import zipfile


DEFAULT_CAMPAIGNS = ("conrep-night-20261008", "conrep-followup-20261008")
OPTIONAL_CAMPAIGNS = ("conrep-llama-ms-20261009",)
INDEX = ["campaign", "experiment", "model", "variant", "seed"]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, dict):
            result.update(flatten(item, name + "."))
        else:
            result[name] = json.dumps(item, ensure_ascii=False) if isinstance(item, list) else item
    return result


def csv_bytes(rows, leading):
    columns = leading + sorted({key for row in rows for key in row} - set(leading))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


class Export:
    def __init__(self, archive):
        self.archive = archive
        self.files = []
        self.warnings = []

    def add(self, name, data, source=None):
        self.archive.writestr(name, data)
        self.files.append({"path": name, "bytes": len(data), "sha256": digest(data),
                           "source": str(source) if source else None})

    def warn(self, source, reason):
        self.warnings.append({"source": str(source), "reason": str(reason)})

    def read(self, path, archive_name=None):
        data = path.read_bytes()
        value = json.loads(data)
        if archive_name:
            self.add(archive_name, data, path)
        return value


def validated_checkpoint(export, directory, task, cfg, include_predictions):
    """Match the campaign's publish marker, hashing the exact exported bytes."""
    marker_path, metrics_path = directory / "NIGHT_VALIDATED.json", directory / "metrics.json"
    if not marker_path.exists():
        return None
    try:
        marker_data, metrics_data = marker_path.read_bytes(), metrics_path.read_bytes()
        marker, report = json.loads(marker_data), json.loads(metrics_data)
        if marker["identity"] != task["identity"] or marker["metrics_sha256"] != digest(metrics_data):
            raise ValueError("Validation identity or metrics digest mismatch")
        if not isinstance(report.get("metrics"), dict) or not report["metrics"]:
            raise ValueError("Missing validation metrics")
        sizes = marker["prediction_sizes"]
        required = {"predictions.jsonl"}
        if cfg.get("evaluation", {}).get("mmlu_file"):
            required.add("mmlu_predictions.jsonl")
        if not required.issubset(sizes):
            raise ValueError("Validation marker is missing required predictions")
        payloads = {"metrics.json": metrics_data, "NIGHT_VALIDATED.json": marker_data}
        for name, size in sizes.items():
            if Path(name).name != name or name not in {"predictions.jsonl", "mmlu_predictions.jsonl"}:
                raise ValueError("Unexpected validation prediction filename")
            path = directory / name
            if path.stat().st_size != size:
                raise ValueError(f"Incomplete predictions: {name}")
            if include_predictions:
                data = path.read_bytes()
                if len(data) != size:
                    raise ValueError(f"Predictions changed during export: {name}")
                payloads[name] = data
        if marker.get("protocol_hash") != report.get("protocol_hash"):
            raise ValueError("Validation protocol mismatch")
        if marker_path.read_bytes() != marker_data or metrics_path.read_bytes() != metrics_data:
            raise ValueError("Validation changed during export; retry next snapshot")
        return report, payloads
    except (OSError, ValueError, KeyError, TypeError) as exc:
        export.warn(directory, exc)
        return None


def collect(campaigns, output, *, include_predictions=False):
    campaigns = list(dict.fromkeys(Path(p).resolve() for p in campaigns))
    output = Path(output).resolve()
    if not campaigns or len({p.name for p in campaigns}) != len(campaigns):
        raise ValueError("Specify campaigns with distinct directory names")
    for campaign in campaigns:
        if output == campaign or output.is_relative_to(campaign):
            raise ValueError("Export destination must be outside source campaigns")
        for name in ("plan.json", "state.json"):
            if not (campaign / name).is_file():
                raise FileNotFoundError(campaign / name)
    output.mkdir(parents=True, exist_ok=False)
    started = now()
    bundle = output / (output.name + ".zip")
    temporary = bundle.with_suffix(".zip.incomplete")
    metrics, statuses, observations, gradients, counts = [], [], [], [], []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        export = Export(archive)
        for campaign in campaigns:
            prefix = "raw/" + campaign.name
            plan = export.read(campaign / "plan.json", prefix + "/plan.json")
            state = export.read(campaign / "state.json", prefix + "/state.json")
            for name in ("positive-audit.json", "fact-positive-audit.json", "inputs.json", "model-assets.json"):
                path = campaign / name
                if path.exists():
                    export.read(path, prefix + "/" + name)
            for path in sorted((campaign / "diagnostic-inputs").glob("*.json")):
                export.read(path, prefix + "/diagnostic-inputs/" + path.name)
            provenance = campaign / "source.json"
            if provenance.exists():
                source = export.read(provenance)
                # Keep provenance hashes; the original source file can also contain
                # a large working-tree diff unrelated to result collection.
                filtered = {k: source[k] for k in ("source_hash", "files", "parent_source_hash", "overlay_ref") if k in source}
                export.add(prefix + "/source-provenance.json", encode(filtered), provenance)
            before = len(metrics)
            campaign_statuses = []
            task_ids = [t["id"] for t in plan["tasks"]]
            if len(task_ids) != len(set(task_ids)):
                raise ValueError(f"Duplicate task identity in {campaign}")
            for task in plan["tasks"]:
                common = dict(campaign=campaign.name, experiment=task["id"],
                              **{k: task[k] for k in ("model", "variant", "seed")})
                config_path = Path(task["config"])
                config_data = config_path.read_bytes()
                if digest(config_data) != task["config_hash"]:
                    raise ValueError(f"Frozen config changed: {config_path}")
                cfg = json.loads(config_data)
                task_root = Path(task["output"]).resolve()
                if not task_root.is_relative_to(campaign):
                    raise ValueError(f"Task output is outside its campaign: {task_root}")
                relative = task_root.relative_to(campaign).as_posix()
                task_prefix = prefix + "/" + relative
                export.add(task_prefix + "/config.json", config_data, config_path)
                maximum, every = int(cfg["unlearn"]["max_steps"]), int(cfg["unlearn"]["save_steps"])
                expected = sorted(set(range(every, maximum + 1, every)) | {maximum})
                valid, task_rows = [], []
                record = state.get("tasks", {}).get(task["id"], {})
                for step in expected:
                    directory = task_root / "validation" / f"checkpoint-{step}"
                    found = validated_checkpoint(export, directory, task, cfg, include_predictions)
                    if found is None:
                        continue
                    report, payloads = found
                    for name, data in payloads.items():
                        export.add(task_prefix + f"/validation/checkpoint-{step}/" + name, data, directory / name)
                    task_rows.append({**flatten(report["metrics"]),
                        **flatten(report.get("mmlu_diagnostics", {}), "mmlu_diagnostic."),
                        **common, "step": step, "identity": task["identity"],
                        "source_hash": plan.get("source_hash"), "config_hash": task["config_hash"],
                        "protocol_hash": report.get("protocol_hash"),
                        "checkpoint": report.get("checkpoint"), "state_status": record.get("status")})
                    valid.append(step)
                training = task_root / "training"
                complete = False
                marker = training / "TRAINING_COMPLETE.json"
                if marker.exists():
                    info = export.read(marker, task_prefix + "/training/TRAINING_COMPLETE.json")
                    complete = info.get("identity") == task["identity"] and info.get("steps") == maximum
                last_step = None
                log = training / "train.jsonl"
                if log.exists():
                    # The trainer can be appending: preserve complete records only.
                    lines = []
                    for line in log.read_bytes().splitlines(keepends=True):
                        if not line.endswith(b"\n"):
                            continue
                        try:
                            row = json.loads(line)
                            last_step = row.get("step", last_step)
                            lines.append(line)
                        except ValueError:
                            export.warn(log, "Skipped malformed training log record")
                    export.add(task_prefix + "/training/train.jsonl", b"".join(lines), log)
                finished = complete and valid == expected
                status = dict(common, state_status=record.get("status"), state_stage=record.get("stage"),
                    job_id=record.get("job_id"), training_complete=complete,
                    last_logged_step=last_step, experiment_complete=finished,
                    validated_checkpoints=len(valid), expected_checkpoints=len(expected),
                    validated_steps=";".join(map(str, valid)),
                    missing_steps=";".join(map(str, sorted(set(expected) - set(valid)))))
                statuses.append(status)
                campaign_statuses.append(status)
                metrics.extend(dict(row, experiment_complete=finished) for row in task_rows)
                for pattern, destination in (("step-*.json", observations), ("gradients-step-*.json", gradients)):
                    for path in sorted((training / "diagnostics").glob(pattern)):
                        try:
                            data = path.read_bytes()
                            report = json.loads(data)
                            if report["identity"] != task["identity"]:
                                raise ValueError("Diagnostic identity mismatch")
                            if destination is observations:
                                row = flatten(report["summary"])
                            else:
                                row = {"microbatch": report["microbatch"],
                                       "sampled_parameter_count": report["sampled_parameter_count"],
                                       "scope": report["scope"]}
                                for key in ("raw_norms", "weighted_norms", "cosines"):
                                    row.update(flatten(report[key], key + "."))
                            destination.append(dict(row, **common, step=report["step"], experiment_complete=finished))
                            export.add(task_prefix + "/training/diagnostics/" + path.name, data, path)
                        except (OSError, ValueError, KeyError, TypeError) as exc:
                            export.warn(path, exc)
            counts.append({"campaign": campaign.name, "source": str(campaign),
                "experiments": len(campaign_statuses),
                "completed_experiments": sum(x["experiment_complete"] for x in campaign_statuses),
                "state_counts": dict(Counter(x["state_status"] for x in campaign_statuses)),
                "validated_checkpoints": len(metrics) - before,
                "protocol_hashes": sorted({str(x["protocol_hash"]) for x in metrics[before:]})})
        tables = {"all-validated-checkpoints.csv": (metrics, INDEX + ["step", "experiment_complete"]),
            "completed-experiment-results.csv": ([r for r in metrics if r["experiment_complete"]], INDEX + ["step"]),
            "experiment-status.csv": (statuses, INDEX + ["state_status", "experiment_complete"]),
            "diagnostic-results.csv": (observations, INDEX + ["step"]),
            "gradient-diagnostics.csv": (gradients, INDEX + ["step", "microbatch"])}
        for name, (rows, leading) in tables.items():
            data = csv_bytes(rows, leading)
            (output / name).write_bytes(data)
            export.add(name, data)
        readme = ("ConRep results snapshot\n\n"
            "all-validated-checkpoints.csv: every committed checkpoint, including partial experiments.\n"
            "completed-experiment-results.csv: all checkpoints of fully completed experiments.\n"
            "experiment-status.csv: every planned experiment and its missing checkpoint validations.\n"
            "diagnostic-results.csv / gradient-diagnostics.csv: sparse training observations.\n"
            "raw/: captured plans, states, configs, committed metrics/markers and diagnostic records.\n\n"
            "Collection is read-only and spans started_at to finished_at; jobs may keep progressing.\n"
            "Completion requires TRAINING_COMPLETE plus every expected NIGHT_VALIDATED marker.\n"
            "Source campaign, config identity and validation protocol are preserved per row.\n"
            "No seed averaging, checkpoint selection, test evaluation, or baseline recomputation occurs.\n"
            "Diagnostic gradients cover sampled LoRA matrices, not the full model.\n"
            "Prediction files included: " + str(include_predictions) + ". No model/optimizer weights included.\n")
        (output / "README.txt").write_text(readme)
        export.add("README.txt", readme.encode())
        manifest = {"schema": "0390-conrep-results-snapshot-v1", "started_at": started, "finished_at": now(),
            "campaigns": counts, "validated_checkpoints": len(metrics),
            "completed_experiments": sum(x["experiment_complete"] for x in statuses),
            "diagnostic_rows": len(observations), "gradient_rows": len(gradients),
            "include_predictions": include_predictions, "warnings": export.warnings,
            "collector_sha256": digest(Path(__file__).read_bytes()), "files": list(export.files)}
        data = encode(manifest)
        (output / "manifest.json").write_bytes(data)
        archive.writestr("manifest.json", data)
    temporary.rename(bundle)
    return manifest, bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--campaign", type=Path, action="append", help="Repeat to override the default campaigns, including Llama M-S when present")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-predictions", action="store_true", help="Also copy verified per-example predictions")
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/validated_v2/0390"
    campaigns = args.campaign or ([base / name for name in DEFAULT_CAMPAIGNS]
        + [base / name for name in OPTIONAL_CAMPAIGNS if (base / name).exists()])
    campaigns = [p if p.is_absolute() else root / p for p in campaigns]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or base / "conrep-exports" / ("0390-conrep-results-" + stamp + "-" + uuid.uuid4().hex[:6])
    if not output.is_absolute():
        output = root / output
    manifest, bundle = collect(campaigns, output, include_predictions=args.include_predictions)
    for item in manifest["campaigns"]:
        print(f"{item['campaign']}: {item['completed_experiments']}/{item['experiments']} experiments complete, "
              f"{item['validated_checkpoints']} validated checkpoints; states={item['state_counts']}")
    print(f"Warnings: {len(manifest['warnings'])}; details in manifest.json")
    print(f"OUTPUT_DIR={bundle.parent}")
    print(f"BUNDLE={bundle}")


if __name__ == "__main__":
    main()
