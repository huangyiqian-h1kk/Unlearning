"""Small durable-state helpers; safe to import on login nodes without torch."""

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def locked(path, *, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield stream


def training_identity(cfg):
    return sha({**{key: cfg[key] for key in ("model", "data", "unlearn", "conrep", "lora")},
                "seed": cfg["run"]["seed"],
                "source_hash": cfg.get("night", {}).get("source_hash"),
                "model_assets_hash": cfg.get("night", {}).get("model_assets_hash"),
                "data_hashes": cfg.get("night", {}).get("data_hashes")})


def complete_checkpoint(path, *, identity=None, world=None):
    path = Path(path)
    try:
        marker = read(path / "COMPLETE.json")
        if marker.get("schema") != "0390-conrep-night-v1":
            return False
        if identity and marker.get("identity") != identity:
            return False
        if world and marker.get("world_size") != world:
            return False
        files = marker["files"]
        required = ["training_state.pt", "adapter_config.json"] + [
            f"rng-rank-{rank}.pt" for rank in range(marker["world_size"])]
        if not all(name in files for name in required):
            return False
        if not any(name.endswith(".safetensors") for name in files):
            return False
        return all(size > 0 and (path / name).is_file()
                   and (path / name).stat().st_size == size for name, size in files.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def latest_checkpoint(root, **kwargs):
    paths = [p for p in Path(root).glob("checkpoint-*")
             if p.name.split("-")[-1].isdigit()]
    for path in sorted(paths, key=lambda x: int(x.name.split("-")[-1]), reverse=True):
        if complete_checkpoint(path, **kwargs):
            return path
    return None
