#!/usr/bin/env python3
"""Extract compact, evidence-derived snapshots from declared immutable sources."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import tarfile

ALLOWED_STATUSES = {"verified", "label_corrected", "non_comparable", "unresolved", "missing", "manuscript_transcription_error"}
LFS_RE = re.compile(r"\Aversion https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?\Z")
MMLU_RE = re.compile(r"\|mmlu\s*\|[^\n]*?\|acc\s*\|[^\n]*?\|([0-9]+\.[0-9]+)\|")

def sha256_bytes(data): return hashlib.sha256(data).hexdigest()
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def validate_member(member):
    path = pathlib.PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
        raise ValueError(f"unsafe archive member: {member.name}")

def validate_archive(path, expected):
    if sha256_file(path) != expected: raise ValueError(f"archive hash mismatch: {path}")
    with tarfile.open(path, "r:gz") as archive:
        for member in archive: validate_member(member)

def archive_member_bytes(source):
    validate_archive(source["archive"], source["archive_sha256"])
    with tarfile.open(source["archive"], "r:gz") as archive:
        try: member = archive.getmember(source["member"])
        except KeyError as exc: raise FileNotFoundError(source["member"]) from exc
        validate_member(member)
        if not member.isfile(): raise ValueError(f"source is not a file: {member.name}")
        data = archive.extractfile(member).read()
    if sha256_bytes(data) != source["member_sha256"]: raise ValueError(f"member hash mismatch: {member.name}")
    return data

def repository_file_bytes(source):
    commit = source["repository_commit_sha"]
    path = source["path"]
    try:
        subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"declared repository commit does not exist: {commit}") from exc
    try:
        blob = subprocess.check_output(["git", "rev-parse", f"{commit}:{path}"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
        data = subprocess.check_output(["git", "show", f"{commit}:{path}"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise FileNotFoundError(f"{commit}:{path}") from exc
    if blob != source["git_blob_sha"]: raise ValueError(f"Git blob mismatch: {commit}:{path}")
    if sha256_bytes(data) != source["file_sha256"]: raise ValueError(f"repository-file hash mismatch: {commit}:{path}")
    return data

def source_bytes(source):
    if source["source_type"] == "archive_member": return archive_member_bytes(source)
    if source["source_type"] == "repository_file": return repository_file_bytes(source)
    raise ValueError(f"unknown source_type: {source.get('source_type')}")

def public_source_reference(source):
    keys = ("source_type", "archive", "archive_sha256", "member", "member_sha256", "path", "repository_commit_sha", "git_blob_sha", "file_sha256", "role", "locator", "observation")
    return {key: source[key] for key in keys if key in source}

def normalize_checkpoint(value):
    if value is None: return None
    value = str(value).rstrip("/")
    marker = "/llm2vec/"
    if marker in value: return "llm2vec/" + value.split(marker, 1)[1]
    return value

def normalize_command(value):
    """Remove private path prefixes while retaining repository-relative arguments."""
    return re.sub(r"/\S*?/llm2vec/", "llm2vec/", value)

def parse_hydra_scalars(text):
    def scalar(pattern):
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip(" '\"") if match else None
    model = scalar(r"^model:\n(?:^[ ]+.*\n)*?^[ ]{4}pretrained_model_name_or_path:\s*(.+)$")
    epochs = scalar(r"^[ ]{4}num_train_epochs:\s*(.+)$")
    learning_rate = scalar(r"^[ ]{4}learning_rate:\s*(.+)$")
    task = scalar(r"^task_name:\s*(.+)$")
    explicit_lines = [line for line in text.splitlines()
                      if re.search(r"forget|target|remove|\.csv\b|\.jsonl\b", line, re.IGNORECASE)]
    explicit = set()
    for line in explicit_lines:
        lowered = line.lower()
        if "diagnosis" in lowered: explicit.add("diagnosis")
        if "death" in lowered: explicit.add("deaths")
        if "pmc" in lowered: explicit.add("pmc")
    if len(explicit) > 1: raise ValueError(f"conflicting explicit forget targets: {sorted(explicit)}")
    target = next(iter(explicit), None)
    if target is None and task:
        lowered = task.lower(); inferred = set()
        if "diagnosis" in lowered: inferred.add("diagnosis")
        if "death" in lowered: inferred.add("deaths")
        if "pmc" in lowered: inferred.add("pmc")
        if len(inferred) > 1: raise ValueError(f"conflicting task-name targets: {sorted(inferred)}")
        target = next(iter(inferred), None)
    return {"model_name_or_path": normalize_checkpoint(model), "num_train_epochs": int(float(epochs)) if epochs else None, "learning_rate": float(learning_rate) if learning_rate else None, "task_name": task, "forget_target": target}

def parse_mmlu(text, selector):
    positions = [match.start() for match in re.finditer(re.escape(selector), text)]
    if len(positions) != 1: raise ValueError(f"MMLU selector must match once, got {len(positions)}: {selector}")
    block = text[positions[0]:]
    next_model = re.search(r"\nhf \(pretrained=", block[1:])
    if next_model: block = block[:next_model.start() + 1]
    match = MMLU_RE.search(block)
    if not match: raise ValueError(f"aggregate MMLU row missing after selector: {selector}")
    settings = re.search(r"limit:\s*([^,]+),\s*num_fewshot:\s*([^,]+),\s*batch_size:\s*([^\s,]+)", block)
    return {"accuracy": float(match.group(1)), "settings": {"limit": settings.group(1).strip() if settings else None, "num_fewshot": settings.group(2).strip() if settings else None, "batch_size": int(settings.group(3)) if settings else None, "evaluation":"zero-shot"}}

def count_jsonl_or_pointer(path, inventory):
    data = pathlib.Path(path).read_bytes(); text = data.decode("utf-8")
    pointer = LFS_RE.fullmatch(text)
    if pointer:
        if inventory["verification_method"] != "researcher_verified_external_lfs_object": raise ValueError(f"unapproved LFS fallback: {path}")
        if sha256_bytes(data) != inventory["pointer_file_sha256"] or pointer.group(1) != inventory["lfs_content_oid"] or pointer.group(1) != inventory["expected_content_sha256"]: raise ValueError(f"LFS pointer identity mismatch: {path}")
        return inventory["record_count"], "researcher_verified_external_lfs_object"
    if sha256_bytes(data) != inventory["file_sha256"]: raise ValueError(f"JSONL content hash mismatch: {path}")
    count = 0
    for number, line in enumerate(text.splitlines(), 1):
        if line.strip(): json.loads(line); count += 1
    if count != inventory["record_count"]: raise ValueError(f"JSONL count mismatch: {path}: {count}")
    return count, "locally_materialized_jsonl"

def derive_success(accuracy, count, tolerance=1e-9):
    product = accuracy * count; success = round(product); error = abs(product - success)
    if error > tolerance: raise ValueError(f"accuracy*n is non-integral: {accuracy}*{count}={product}")
    return {"accuracy": accuracy, "sample_count": count, "success_count": success, "integrality_error": error, "derivation":"k = archived full-precision accuracy × exact dataset record count"}

def binomial_greater_pvalue(k, n, p0=.25):
    import math
    return sum(math.comb(n, i) * p0 ** i * (1-p0) ** (n-i) for i in range(k, n+1))

def baseline_significance(experiment, metrics, datasets):
    if experiment["method"] != "Baseline": return {}
    by = {(item["split"], item["probe"]): item for item in datasets}
    result = {}
    if experiment["regime"] == "A":
        model = "llama2" if "Llama-2" in experiment["intended_paper_label"] else "mistral"
        keymap = {("celebrity_deaths","ATT"):"MCQ-llh-mcqs_deaths_att", ("celebrity_deaths","IDeq"):"MCQ-llh-mcqs_deaths_id_eq", ("celebrity_deaths","ID"):"MCQ-llh-mcqs_deaths_id_sim", ("celebrity_diagnosis","ATT"):"MCQ-llh-mcqs_diagnosis_att", ("celebrity_diagnosis","ID"):"MCQ-llh-mcqs_diagnosis_id"}
        raw = metrics["metrics"]
        for (domain, probe), key in keymap.items():
            item = by[(domain, probe)]; derived = derive_success(raw[key], item["record_count"]); pvalue = binomial_greater_pvalue(derived["success_count"], derived["sample_count"])
            result[f"{domain}:{probe}"] = {**derived, "pvalue":pvalue, "significant":pvalue < .05, "dataset":item["path"], "dataset_sha256":item.get("file_sha256"), "verification_method":item["verification_method"], "model":model}
    else:
        retain = metrics["retain_metrics"]; forget = metrics["forget_metrics"]
        keys = {"ATT":("mcqs_PMC_retain_att","mcqs_PMC_forget_att"), "IDeq":("mcqs_PMC_retain_id_equal","mcqs_PMC_forget_id_equal"), "ID":("mcqs_PMC_retain_id_identical","mcqs_PMC_forget_id_identical")}
        for probe, (rk, fk) in keys.items():
            ri, fi = by[("retain",probe)], by[("forget",probe)]; rd=derive_success(retain["MCQ-llh-"+rk],ri["record_count"]); fd=derive_success(forget["MCQ-llh-"+fk],fi["record_count"])
            k=rd["success_count"]+fd["success_count"]; n=rd["sample_count"]+fd["sample_count"]; pvalue=binomial_greater_pvalue(k,n)
            result[f"pmc:{probe}"]={"retain":rd,"forget":fd,"success_count":k,"sample_count":n,"accuracy":k/n,"pvalue":pvalue,"significant":pvalue<.05,"datasets":[ri["path"],fi["path"]],"verification_methods":[ri["verification_method"],fi["verification_method"]]}
    return result

def validate_cell_status(experiment):
    valid_a={"R-QA","R-Cloze","R-BG","F-QA","F-Cloze","F-BG","R-ATT","R-IDeq","R-ID","F-ATT","F-IDeq","F-ID","MMLU","Average"}
    valid_b={f"{probe}-{suffix}" for probe in ("QA","Cloze","BG","ATT","IDeq","ID") for suffix in ("R","F","Delta")} | {"MMLU","Average"}
    unknown=set(experiment.get("cell_status",{}))-(valid_a if experiment["regime"]=="A" else valid_b)
    if unknown: raise ValueError(f"unknown cell_status keys for {experiment['id']}: {sorted(unknown)}")

def validate_verified_evidence(experiment, snapshot):
    if "verified" not in experiment["status_flags"]: return
    if "training_config" in snapshot["field_evidence"]:
        for key,value in experiment.get("hyperparameters",{}).items():
            if key in snapshot["normalized_config"] and snapshot["normalized_config"][key] != value: raise ValueError(f"verified config mismatch {experiment['id']}:{key}")
    if "hydra_config" in snapshot["field_evidence"]:
        observed_target=snapshot["normalized_config"].get("forget_target")
        if observed_target != experiment.get("resolved_forget_dataset_target"): raise ValueError(f"verified forget-target mismatch {experiment['id']}: {observed_target} != {experiment.get('resolved_forget_dataset_target')}")
    if "trainer_state" in snapshot["field_evidence"]:
        if snapshot["trainer_state"]["global_step"] != experiment.get("final_step") or snapshot["trainer_state"]["epoch"] != experiment.get("final_epoch"): raise ValueError(f"terminal-state assertion mismatch: {experiment['id']}")

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",required=True);parser.add_argument("--output-dir",required=True);args=parser.parse_args(argv)
    manifest=json.load(open(args.manifest,encoding="utf-8")); inventory=json.load(open(manifest["mcq_dataset_inventory"],encoding="utf-8"))["datasets"]
    for archive in manifest["archives"]: validate_archive(archive["path"],archive["sha256"])
    dataset_evidence=[]
    for item in inventory:
        count, method=count_jsonl_or_pointer(item["path"],item); dataset_evidence.append({**item,"record_count":count,"verification_method":method})
    output=pathlib.Path(args.output_dir);output.mkdir(parents=True,exist_ok=True);expected=set()
    for experiment in manifest["experiments"]:
        if not set(experiment["status_flags"]) <= ALLOWED_STATUSES: raise ValueError(f"invalid status: {experiment['id']}")
        validate_cell_status(experiment)
        snapshot={"schema_version":"1.1","experiment_id":experiment["id"],"status_flags":experiment["status_flags"],"cell_status":experiment.get("cell_status",{}),"sources":[],"metrics":{},"trainer_state":{},"normalized_config":{},"field_evidence":{},"mmlu":None,"significance":{}}
        for source in experiment["sources"]:
            data=source_bytes(source);snapshot["sources"].append(public_source_reference(source));role=source["role"]
            if role in {"metrics","retain_metrics","forget_metrics"}: snapshot["metrics"][role]=json.loads(data)
            elif role=="trainer_state":
                state=json.loads(data);snapshot["trainer_state"]={key:state.get(key) for key in ("global_step","epoch","best_model_checkpoint","best_metric")};snapshot["field_evidence"]["trainer_state"]={"source":public_source_reference(source),"observed":snapshot["trainer_state"],"observation":"directly observed"}
            elif role=="training_config":
                config=json.loads(data); fields=("model_name_or_path","peft_model_name_or_path","learning_rate","forget_weight","lm_weight","gamma","num_train_epochs","output_dir")
                observed={key:(normalize_checkpoint(config.get(key)) if key in {"model_name_or_path","peft_model_name_or_path","output_dir"} else config.get(key)) for key in fields}
                snapshot["normalized_config"].update(observed);snapshot["field_evidence"]["training_config"]={"source":public_source_reference(source),"selectors":list(fields),"observed":observed,"observation":"directly observed"}
            elif role=="config":
                resolved=parse_hydra_scalars(data.decode());snapshot["normalized_config"].update({key:value for key,value in resolved.items() if value is not None});snapshot["field_evidence"]["hydra_config"]={"source":public_source_reference(source),"observed":resolved,"observation":"directly observed"}
            elif role=="job_script":
                lines=[normalize_command(line.strip()) for line in data.decode().splitlines() if "ContrastiveUnlearning" in line];snapshot["field_evidence"]["job_script"]={"source":public_source_reference(source),"observed_commands":lines,"active_commands":[line for line in lines if not line.startswith("#")],"observation":"directly observed; commented commands are not execution evidence"}
            elif role=="mmlu_log":
                parsed=parse_mmlu(data.decode(errors="replace"),source["model_block_selector"])
                if abs(parsed["accuracy"]-source["expected_accuracy"])>5e-5: raise ValueError(f"MMLU assertion mismatch: {experiment['id']}")
                snapshot["mmlu"]={**parsed,"source":public_source_reference(source),"model_block_selector":source["model_block_selector"]}
        validate_verified_evidence(experiment,snapshot)
        snapshot["significance"]=baseline_significance(experiment,snapshot["metrics"],dataset_evidence)
        target=output/f"{experiment['id']}.json";expected.add(target.name);target.write_text(json.dumps(snapshot,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    related=[]
    for evidence in manifest.get("related_evidence",[]):
        item={"id":evidence["id"],"status_flags":evidence["status_flags"],"configuration_identity":evidence["configuration_identity"],"selected_experiment_id":evidence["selected_experiment_id"],"known_limitation":evidence["known_limitation"],"sources":[],"mmlu":None}
        for source in evidence["sources"]:
            data=source_bytes(source); item["sources"].append(public_source_reference(source))
            if source["role"]=="mmlu_log":
                parsed=parse_mmlu(data.decode(errors="replace"),source["model_block_selector"])
                if abs(parsed["accuracy"]-source["expected_accuracy"])>5e-5: raise ValueError(f"MMLU assertion mismatch: {evidence['id']}")
                item["mmlu"]={**parsed,"source":public_source_reference(source),"model_block_selector":source["model_block_selector"]}
        related.append(item)
    if related:
        target=output/"related_evidence.json"; expected.add(target.name); target.write_text(json.dumps(related,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for path in output.glob("*.json"):
        if path.name not in expected:path.unlink()

if __name__=="__main__":main()
