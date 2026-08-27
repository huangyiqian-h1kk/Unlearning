#!/usr/bin/env python3
"""Create compact metric snapshots from immutable, manifest-declared tar members."""
import argparse, hashlib, io, json, pathlib, tarfile

ALLOWED_STATUSES = {"verified", "label_corrected", "non_comparable", "unresolved", "missing", "manuscript_transcription_error"}

def sha256_bytes(data): return hashlib.sha256(data).hexdigest()
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def validate_member(member):
    p=pathlib.PurePosixPath(member.name)
    if p.is_absolute() or ".." in p.parts or not (member.isfile() or member.isdir()):
        raise ValueError(f"unsafe or non-regular archive member: {member.name}")

def validate_archive(archive, expected_hash):
    if sha256_file(archive) != expected_hash: raise ValueError(f"archive hash mismatch: {archive}")
    with tarfile.open(archive,"r:gz") as tf:
        for member in tf: validate_member(member)

def read_member(archive, archive_hash, member_name, member_hash):
    if sha256_file(archive) != archive_hash: raise ValueError(f"archive hash mismatch: {archive}")
    with tarfile.open(archive,"r:gz") as tf:
        for item in tf:
            validate_member(item)
            if item.isdir(): continue
            if item.name == member_name:
                data=tf.extractfile(item).read()
                if sha256_bytes(data) != member_hash: raise ValueError(f"member hash mismatch: {member_name}")
                return data
    raise FileNotFoundError(f"manifest member missing: {member_name}")

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--output-dir",required=True); a=p.parse_args(argv)
    manifest=json.load(open(a.manifest,encoding="utf-8")); out=pathlib.Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    for archive in manifest["archives"]: validate_archive(archive["path"],archive["sha256"])
    expected=set()
    for exp in manifest["experiments"]:
        if not set(exp["status_flags"]) <= ALLOWED_STATUSES: raise ValueError(f"invalid status: {exp['id']}")
        snap={"schema_version":"1.0","experiment_id":exp["id"],"status_flags":exp["status_flags"],"sources":[],"metrics":{},"trainer_state":{},"resolved_config":exp.get("resolved_config",{}),"sample_counts":exp.get("sample_counts",{}),"mmlu":None}
        for src in exp.get("sources",[]):
            data=read_member(src["archive"],src["archive_sha256"],src["member"],src["member_sha256"])
            snap["sources"].append({k:src[k] for k in ("archive","archive_sha256","member","member_sha256","role","observation")})
            if src["role"] in {"metrics","retain_metrics","forget_metrics"}:
                obj=json.loads(data); snap["metrics"][src["role"]]=obj
            elif src["role"]=="trainer_state":
                obj=json.loads(data); snap["trainer_state"]={k:obj.get(k) for k in ("global_step","epoch","best_model_checkpoint","best_metric")}
            elif src["role"]=="mmlu_log":
                snap["mmlu"]={"accuracy":src.get("mmlu_accuracy"),"settings":src.get("mmlu_settings",{}),"source_member":src["member"]}
        target=out/f"{exp['id']}.json"; expected.add(target.name)
        target.write_text(json.dumps(snap,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for old in out.glob("*.json"):
        if old.name not in expected: old.unlink()

if __name__=="__main__": main()
