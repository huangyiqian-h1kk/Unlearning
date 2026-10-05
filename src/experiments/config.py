from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml


def merge(base, update):
    result = dict(base)
    for key, value in update.items():
        result[key] = (
            merge(result.get(key, {}), value) if isinstance(value, dict) else value
        )
    return result


def load_config(path, overrides=()):
    path = Path(path).resolve()
    # Frozen run configurations are JSON. YAML 1.1 can interpret JSON numbers
    # such as 1e-05 as strings, changing the learning rate on worker reload.
    raw = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    parents = raw.pop("inherits", [])
    cfg = {}
    for parent in parents:
        cfg = merge(cfg, load_config(path.parent / parent))
    cfg = merge(cfg, raw)
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value: {override}")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)

    def expand(value):
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return os.path.expandvars(value) if isinstance(value, str) else value

    return expand(cfg)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def output_dir(cfg):
    path = Path(cfg["run"]["output_dir"]).resolve()
    if "paper" in path.parts or "historical" in path.parts:
        raise ValueError("New experiments must not write into historical/paper outputs")
    return path


def read_rows(path):
    import csv

    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        prefix = stream.read(100)
        stream.seek(0)
        if prefix.startswith("version https://git-lfs.github.com/spec"):
            raise ValueError(f"LFS pointer is not data: {path}; materialize it first")
        if path.suffix == ".csv":
            return list(csv.DictReader(stream))
        if path.suffix == ".json":
            result = json.load(stream)
            if not isinstance(result, list):
                raise ValueError(f"Expected JSON array: {path}")
            return result
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
