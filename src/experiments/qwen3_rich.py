"""Qwen3-only rich injection, immutable evidence, and a one-node PBS workflow.

Uses the parent campaign's exact v5 evaluator and prepared rich-all data. Does
not modify the parent, select an unlearning checkpoint, or run unlearning.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import getpass
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

from conrep.night.io import file_sha, locked, read, sha, write

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ENTRY = "scripts/abci/0390_qwen3_rich.py"
REQUIRED = {"transformers": "4.51.3", "peft": "0.14.0", "accelerate": "1.3.0",
            "tokenizers": "0.21.1", "huggingface-hub": "0.30.2", "deepspeed": "0.16.2",
            "numpy": "1.26.4", "safetensors": "0.5.2", "datasets": "3.2.0", "PyYAML": "6.0.2"}


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def weights(path):
    path = Path(path)
    if not (path / "config.json").is_file(): raise ValueError(f"Missing model config: {path}")
    index = path / "model.safetensors.index.json"
    names = sorted(set(read(index)["weight_map"].values())) if index.exists() else ["model.safetensors"]
    result = [path / name for name in names]
    if not result or any(not p.resolve().is_relative_to(path.resolve()) or not p.is_file() or p.stat().st_size == 0 for p in result):
        raise ValueError(f"Missing/empty/unsafe weight shards: {path}")
    return result


def assets(path):
    weights(path)
    return {str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(Path(path).iterdir()) if p.is_file()}


def environment():
    found = {k: importlib.metadata.version(k) for k in REQUIRED}
    if found != REQUIRED: raise ValueError(f"Use the isolated pinned Qwen3 environment: {found}")
    torch_version = importlib.metadata.version("torch")
    if torch_version.split("+")[0] != "2.5.1": raise ValueError("Expected torch 2.5.1")
    return dict(found, torch=torch_version, python=sys.version)


def download(model_dir):
    from huggingface_hub import HfApi, snapshot_download
    model_dir = Path(model_dir).resolve()
    receipt = model_dir / "0390-model-origin.json"
    if receipt.exists():
        origin = read(receipt)
        if origin["repo_id"] != MODEL_ID: raise ValueError("Model destination belongs to another model")
        revision = origin["revision"]
    else:
        if model_dir.exists() and any(model_dir.iterdir()):
            raise ValueError("Untracked model destination is nonempty; use a new directory")
        revision = HfApi().model_info(MODEL_ID).sha
        write(receipt, dict(repo_id=MODEL_ID, revision=revision))
    snapshot_download(MODEL_ID, revision=revision, local_dir=str(model_dir),
                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
    weights(model_dir)
    print(json.dumps(dict(model=str(model_dir), **read(receipt)), indent=2))


def check_code(campaign):
    for name, expected in read(campaign / "code-files.json").items():
        path = (campaign / "code" / name).resolve()
        if not path.is_relative_to(campaign / "code") or file_sha(path) != expected:
            raise ValueError(f"Frozen code changed: {name}")


def encode(tokenizer, row, limit):
    if row.get("kind") != "qa" or row.get("truncate_policy") != "error":
        raise ValueError(f"Expected explicit rich assistant-only QA: {row.get('id')}")
    user = [{"role": "user", "content": row["prompt"]}]
    prefix = tokenizer.apply_chat_template(user, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(user + [{"role": "assistant", "content": row["answer"]}],
                                         tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError(f"Unsafe assistant prefix/mask: {row['id']}")
    if len(full) > limit: raise ValueError(f"Row {row['id']} has {len(full)} tokens > {limit}; no truncation")
    if tokenizer.eos_token_id not in full[len(prefix):]:
        raise ValueError(f"Missing assistant EOS in {row['id']}")
    return dict(input_ids=full, attention_mask=[1] * len(full), labels=[-100] * len(prefix) + full[len(prefix):])


def audit_data(cfg, tokenizer):
    root = Path(cfg["data"]["prepared_dir"])
    facts, injection = rows(root / "injection_facts.jsonl"), rows(root / "injection.jsonl")
    ids = {(f["patient_id"], f["attribute"]) for f in facts}
    if len(facts) != 2925 or len(ids) != 2925 or len({p for p,a in ids}) != 975:
        raise ValueError("Expected 975 patients and 2925 unique structured facts")
    if Counter(f["attribute"] for f in facts) != {"age":975,"gender":975,"diagnosis":975}:
        raise ValueError("Incomplete rich fact attributes")
    if any({a for p,a in ids if p == patient} != {"age","gender","diagnosis"} for patient in {p for p,a in ids}):
        raise ValueError("Each patient needs all three attributes")
    if len(injection) != 9750 or Counter(r["source_field"] for r in injection) != {
        "authored_relation":5850,"patient":975,"note":975,"medical_dialogue":975,"archive":975}:
        raise ValueError("Expected unchanged 9750-row rich SFT corpus")
    if len({r["id"] for r in injection}) != len(injection): raise ValueError("Duplicate SFT rows")
    relation_counts = Counter(rid for r in injection if r["source_field"] == "authored_relation" for rid in r["relation_ids"])
    if relation_counts != Counter({f["relation_id"]:2 for f in facts}):
        raise ValueError("Rich SFT needs exactly two authored expressions per fact")
    lengths, supervised = [], []
    for row in injection:
        example = encode(tokenizer, row, cfg["sft"]["max_length"])
        lengths.append(len(example["input_ids"])); supervised.append(sum(x != -100 for x in example["labels"]))
    return dict(rows=len(injection), facts=len(facts), patients=975, max_tokens=max(lengths),
                min_supervised_tokens=min(supervised), truncation=False, assistant_only=True,
                special_tokens=dict(eos=tokenizer.eos_token_id,pad=tokenizer.pad_token_id),
                general_text_mixture=0)


def prepare(campaign, model_dir):
    from transformers import AutoConfig, AutoTokenizer
    from .validation_v5 import protocol_metadata, checked_rows, MCQ
    campaign = Path(campaign).resolve(); check_code(campaign)
    boot = read(campaign / "PREPARING.json"); parent = Path(boot["parent"])
    if (campaign / "plan.json").exists():
        plan, cfg = verify(campaign)
        if Path(cfg["model"]["name_or_path"]) != Path(model_dir).resolve(): raise ValueError("Model changed")
        print("Preparation verified; use start/status."); return plan
    profile = read(campaign / "code/configs/0390/qwen3_4b_rich_sft.json")
    parent_plan = read(parent / "plan.json")
    task = next(t for t in parent_plan["tasks"] if t["model"] == "llama8b")
    original = read(task["config"])
    if sha(original) != task["config_hash"]: raise ValueError("Parent configuration changed")
    for name, expected in read(parent / "inputs.json").items():
        if file_sha(name) != expected: raise ValueError(f"Parent input changed: {name}")
    cfg = {k: copy.deepcopy(original[k]) for k in ("model","data","evaluation")}
    cfg["run"] = dict(seed=profile["seed"], output_dir=str(campaign / "sft"))
    cfg["sft"] = profile["sft"]
    cfg["sft"]["deepspeed"] = str(campaign / "code/configs/0390/zero2.json")
    model_dir = Path(model_dir).resolve()
    origin = read(model_dir / "0390-model-origin.json")
    if origin["repo_id"] != MODEL_ID: raise ValueError("Expected native non-thinking 2507 model")
    cfg["model"] = dict(name_or_path=str(model_dir),local_only=True,dtype="bfloat16",attention="sdpa")
    if cfg["data"].get("injection_scope") != "all_structured" or cfg["evaluation"].get("scoring") != "pmc-ia-v5":
        raise ValueError("Expected the frozen rich-all/v5 parent")
    if cfg["evaluation"].get("partition") != "validation" or cfg["evaluation"].get("limit"):
        raise ValueError("Use full validation split for SFT curves")
    versions = environment()
    if AutoConfig.from_pretrained(model_dir,local_files_only=True).model_type != "qwen3": raise ValueError("Wrong model type")
    tokenizer = AutoTokenizer.from_pretrained(model_dir,local_files_only=True,use_fast=True)
    if not tokenizer.chat_template or not tokenizer.is_fast: raise ValueError("Native fast tokenizer required")
    rendered = tokenizer.apply_chat_template([{"role":"user","content":"Answer briefly."}],tokenize=False,add_generation_prompt=True)
    if "<think>" in rendered or "</think>" in rendered: raise ValueError("Expected native non-thinking template")
    audit = audit_data(cfg, tokenizer)
    protocol, inputs = protocol_metadata(cfg)
    facts = {f["id"]:f for f in rows(Path(cfg["data"]["prepared_dir"]) / "facts.jsonl")}
    for split in ("forget","retain"):
        for task_name in ("generation",*MCQ): checked_rows(cfg,split,task_name,facts)
    for name in ("injection.jsonl","injection_facts.jsonl"):
        p = Path(cfg["data"]["prepared_dir"]) / name; inputs[str(p)] = file_sha(p)
    manifest=read(Path(cfg["data"]["prepared_dir"])/"manifest.json")
    for name in ("injection.jsonl","injection_facts.jsonl"):
        p=Path(cfg["data"]["prepared_dir"])/name
        if manifest["files"].get(name)!=file_sha(p): raise ValueError(f"Prepared rich input changed: {name}")
    # Conservative peak: archived milestones/final + three complete Adam saves.
    weight_bytes = sum(p.stat().st_size for p in weights(model_dir))
    storage_need = (len(profile["milestones"]) + 1 + 3 * 8) * weight_bytes + profile["reserve_bytes"]
    if shutil.disk_usage(campaign).free < storage_need:
        raise ValueError(f"Need approximately {storage_need/1e9:.0f} GB free for this bounded checkpoint policy, including reserve")
    plan = dict(schema="0390-qwen3-rich-v1", campaign=str(campaign),project_root=boot["project_root"],
        python=sys.executable, entry=str(campaign / "code" / ENTRY), config_hash=sha(cfg),
        profile=profile, inputs=inputs, model_assets=assets(model_dir), environment=versions,
        protocol_hash=protocol, mmlu_rows=len(rows(cfg["evaluation"]["mmlu_file"])), model_origin=origin,
        code_files_hash=file_sha(campaign / "code-files.json"), storage_need_bytes=storage_need,
        recovery_policy={"retry_admin_termination":True,"retry_terminated_queued":True})
    write(campaign / "config.json", cfg); write(campaign / "tokenizer-audit.json",audit)
    write(campaign / "state.json",dict(status="ready",attempts=0,job_id=None,retry_after=0))
    write(campaign / "plan.json",plan)
    print(json.dumps(dict(campaign=str(campaign),audit=audit,mmlu_rows=plan["mmlu_rows"],
                         evaluation_checkpoints=["base",*profile["milestones"],"final"],jobs_submitted=0),indent=2))
    return plan


def verify(campaign, *, check_environment=True):
    campaign = Path(campaign).resolve(); plan = read(campaign / "plan.json")
    check_code(campaign)
    if file_sha(campaign / "code-files.json") != plan["code_files_hash"]: raise ValueError("Source manifest changed")
    cfg = read(campaign / "config.json")
    if sha(cfg) != plan["config_hash"]: raise ValueError("Configuration changed")
    for path, expected in plan["inputs"].items():
        if file_sha(path) != expected: raise ValueError(f"Frozen input changed: {path}")
    if assets(cfg["model"]["name_or_path"]) != plan["model_assets"]: raise ValueError("Base model files changed")
    if check_environment and environment() != plan["environment"]: raise ValueError("Runtime environment changed")
    return plan, cfg


def resume_step(path, world=8):
    path = Path(path); marker = read(path / "QWEN3_COMPLETE.json")
    step = int(path.name.removeprefix("checkpoint-")); weights(path)
    if marker["step"] != step or marker["world_size"] != world or read(path/"trainer_state.json")["global_step"] != step:
        raise ValueError("Checkpoint step/world mismatch")
    tag = (path / "latest").read_text().strip()
    if Path(tag).name != tag or not tag: raise ValueError("Invalid DeepSpeed tag")
    required = [path/f"rng_state_{r}.pth" for r in range(world)]
    optim = list((path/tag).glob("*_optim_states.pt")); models = list((path/tag).glob("*_model_states.pt"))
    if len(optim) != world or not models: raise ValueError("Missing DeepSpeed state")
    required += optim + models
    if any(not p.is_file() or p.stat().st_size == 0 for p in required): raise ValueError("Missing optimizer/RNG state")
    if marker["files"] != {str(p.relative_to(path)):p.stat().st_size for p in path.rglob("*") if p.is_file() and p.name != "QWEN3_COMPLETE.json"}:
        raise ValueError("Checkpoint file set/size changed")
    return step


def latest_resume(campaign):
    complete, incomplete = [], []
    for path in (campaign/"sft").glob("checkpoint-*"):
        if not path.is_dir() or not path.name.removeprefix("checkpoint-").isdigit(): continue
        try: complete.append((resume_step(path),path))
        except (ValueError,OSError,KeyError): incomplete.append(path)
    for path in incomplete:
        dest = campaign/"interrupted-saves"/(path.name+"-"+uuid.uuid4().hex[:8]); dest.parent.mkdir(exist_ok=True)
        path.rename(dest)
    return str(max(complete)[1]) if complete else None


def archive_weights(source, destination):
    source, destination = Path(source), Path(destination)
    weights(source)
    if (destination/"WEIGHTS_COMPLETE.json").exists(): return
    temp = destination.with_name(destination.name+".incomplete")
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir(parents=True)
    # Hard links preserve weights after Trainer rotates its own checkpoints.
    for p in source.iterdir():
        if p.is_file() and (p.suffix in {".safetensors",".json",".jinja"} or p.name in {"vocab.json","merges.txt"}):
            if p.name in {"trainer_state.json","QWEN3_COMPLETE.json"}: continue
            try: os.link(p,temp/p.name)
            except OSError: shutil.copy2(p,temp/p.name)
    write(temp/"WEIGHTS_COMPLETE.json",dict(source=str(source),assets=assets(temp)))
    if destination.exists(): raise ValueError(f"Incomplete archive needs inspection: {destination}")
    temp.rename(destination)


def rng_safe_globals():
    """HF 4.51 enables its NumPy RNG allowlist only for torch>=2.6.

    Its RNG loader already uses weights_only=True on our pinned torch2.5.1.
    Supply only the NumPy types used by our own Trainer RNG states, scoped to
    this train call. Do not disable the restricted loader or allow arbitrary
    checkpoint classes.
    """
    import numpy as np
    import torch
    return torch.serialization.safe_globals([np.core.multiarray._reconstruct,
        np.ndarray,np.dtype,type(np.dtype(np.uint32))])


def train(campaign):
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, set_seed
    from .data import pad_examples
    campaign = Path(campaign).resolve(); plan,cfg = verify(campaign)
    world = int(os.environ.get("WORLD_SIZE","1"))
    if world != 8 or not torch.cuda.is_available(): raise ValueError("Full SFT requires the planned eight-GPU torchrun")
    options = cfg["sft"]; set_seed(cfg["run"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name_or_path"],local_files_only=True,use_fast=True)
    encoded = [encode(tokenizer,r,options["max_length"]) for r in rows(Path(cfg["data"]["prepared_dir"])/"injection.jsonl")]
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name_or_path"],local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa")
    model.config.use_cache = False
    args = TrainingArguments(output_dir=str(campaign/"sft"),learning_rate=options["learning_rate"],
        num_train_epochs=options["epochs"],per_device_train_batch_size=options["batch_size"],
        gradient_accumulation_steps=options["gradient_accumulation_steps"],warmup_ratio=options["warmup_ratio"],
        weight_decay=options["weight_decay"],max_grad_norm=options["max_grad_norm"],lr_scheduler_type="linear",
        bf16=True,fp16=False,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
        save_strategy="steps",save_steps=options["save_steps"],save_total_limit=2,save_safetensors=True,
        logging_steps=options["logging_steps"],report_to=[],seed=42,data_seed=42,remove_unused_columns=False,
        deepspeed=options["deepspeed"],ddp_find_unused_parameters=False,dataloader_num_workers=0)

    class DurableSave(TrainerCallback):
        def on_save(self,args,state,control,**kwargs):
            dist.barrier()
            if state.is_world_process_zero:
                path = campaign/"sft"/f"checkpoint-{state.global_step}"
                write(path/"QWEN3_COMPLETE.json",dict(step=state.global_step,world_size=world,
                    files={str(p.relative_to(path)):p.stat().st_size for p in path.rglob("*") if p.is_file() and p.name != "QWEN3_COMPLETE.json"}))
                resume_step(path,world)
                if state.global_step in plan["profile"]["milestones"]:
                    archive_weights(path,campaign/"candidates"/path.name)
                write(campaign/"progress.json",dict(stage="sft",step=state.global_step,epoch=state.epoch,max_steps=state.max_steps))
                write(campaign/"training-log.json",dict(step=state.global_step,log_history=state.log_history))
            dist.barrier()
            return control

        def on_step_begin(self,args,state,control,**kwargs):
            # All ranks make the same stop decision before a forward pass.
            stop = torch.tensor([int((campaign/"STOP").exists() or shutil.disk_usage(campaign).free < plan["profile"]["reserve_bytes"])],device=args.device)
            dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if stop.item(): raise RuntimeError("STOP requested or storage reserve reached; resume from last complete save")
            return control

    trainer = Trainer(model=model,args=args,train_dataset=encoded,processing_class=tokenizer,
        data_collator=lambda batch:pad_examples(batch,tokenizer.pad_token_id),callbacks=[DurableSave()])
    resume = read(campaign/"train-launch.json")["resume"]
    if resume: resume_step(resume)
    with rng_safe_globals():
        trainer.train(resume_from_checkpoint=resume)
    final = campaign/"sft/final"
    trainer.save_model(str(final))
    if trainer.is_world_process_zero(): tokenizer.save_pretrained(final)
    dist.barrier()
    if trainer.is_world_process_zero():
        weights(final)
        write(campaign/"sft/TRAINING_COMPLETE.json",dict(global_step=trainer.state.global_step,epoch=trainer.state.epoch,
            final=str(final),config_hash=plan["config_hash"],examples=len(encoded),effective_batch=32,
            selection="Review SFT curves; no automatic unlearning checkpoint selection"))
        write(campaign/"training-log.json",dict(step=trainer.state.global_step,log_history=trainer.state.log_history))
    dist.barrier()


def evaluation_identity(plan,cfg,checkpoint):
    from .validation_v5 import protocol_metadata
    return dict(config_hash=plan["config_hash"],protocol_hash=protocol_metadata(cfg)[0],
                checkpoint=str(Path(checkpoint).resolve()),weights=assets(checkpoint),code_files_hash=plan["code_files_hash"])


def evaluate(campaign,name,checkpoint):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .validation_v5 import evaluate_loaded, generate
    from .pmc_scoring_v5 import score_answer
    campaign=Path(campaign).resolve(); plan,cfg=verify(campaign)
    output=campaign/"validation"/name; marker=output/"EVALUATED.json"
    identity=evaluation_identity(plan,cfg,checkpoint)
    if marker.exists():
        cached=read(marker)
        if cached["identity"] != identity or any(file_sha(output/n)!=h for n,h in cached["files"].items()):
            raise ValueError("Evaluation cache identity/content changed")
        print(f"Reusing {name}",flush=True);return
    if not torch.cuda.is_available(): raise ValueError("GPU evaluation required")
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    evaluate_loaded(cfg,checkpoint,model,tokenizer,output)
    # Supplemental QA diagnostic for all injected relations, not a new v5 score.
    facts=rows(Path(cfg["data"]["prepared_dir"])/"injection_facts.jsonl")
    prompts=[f"As far as we remember, what is the {f['attribute']} of the patient with id {f['patient_id']}?" for f in facts]
    predictions=generate(model,tokenizer,prompts,cfg["evaluation"])
    if len(predictions)!=len(facts): raise ValueError("All-facts prediction count mismatch")
    counts=defaultdict(list); details=[]
    for f,prompt,pred in zip(facts,prompts,predictions):
        score=score_answer(pred,f["target"],f["patient_id"])
        for key in ("all",f["attribute"],f["split"]): counts[key].append(score["correct"])
        details.append(dict(relation_id=f["relation_id"],patient_id=f["patient_id"],attribute=f["attribute"],
            split=f["split"],target=f["target"],prompt=prompt,output=pred,**score))
    (output/"injection_predictions.jsonl").write_text("".join(json.dumps(d,ensure_ascii=False)+"\n" for d in details))
    write(output/"injection-qa.json",dict(protocol="supplemental-all2925-qa-v1",usage="Injection diagnostic; not held-out evaluation",
        metrics={k:sum(v)/len(v) for k,v in counts.items()},counts={k:len(v) for k,v in counts.items()}))
    write(marker,dict(identity=identity,files={p.name:file_sha(p) for p in output.iterdir() if p.is_file() and p.suffix in {".json",".jsonl"} and p.name != marker.name}))


def summary(campaign):
    plan=read(campaign/"plan.json"); items=[]
    for marker in (campaign/"validation").glob("*/EVALUATED.json"):
        report=read(marker.parent/"metrics.json")
        m=report["metrics"]
        items.append(dict(checkpoint=marker.parent.name,**m,all2925_qa=read(marker.parent/"injection-qa.json")["metrics"]["all"]))
    items.sort(key=lambda r: -1 if r["checkpoint"]=="base" else (10**9 if r["checkpoint"]=="final" else int(r["checkpoint"].split("-")[-1])))
    base=next((r for r in items if r["checkpoint"]=="base"),None)
    if base:
        for r in items:
            r["mmlu_change_pp"]=100*(r["utility.mmlu"]-base["utility.mmlu"])
            r["qa_mean_gain_pp"]=50*(r["forget.qa"]+r["retain.qa"]-base["forget.qa"]-base["retain.qa"])
            r["mmlu_within_5pp"]=r["mmlu_change_pp"]>=-100*plan["profile"]["mmlu_review_drop"]
            r["qa_gain_at_least_5pp"]=r["qa_mean_gain_pp"]>=100*plan["profile"]["min_qa_gain"]
    if items:
        temporary=campaign/(".summary-"+uuid.uuid4().hex+".csv")
        with temporary.open("w") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
        temporary.replace(campaign/"summary.csv")
    write(campaign/"review.json",dict(checkpoints=items,selected=None,
        notice="No automatic selection. Review injection coverage, per-attribute scores, QA gain and MMLU loss before unlearning."))
    return items


def child(plan,command,*extra,env=None):
    return subprocess.Popen([plan["python"],"-I",plan["entry"],command,"--campaign",plan["campaign"],*extra],
        cwd=plan["project_root"],env=env)


def worker(campaign):
    import torch
    campaign=Path(campaign).resolve();plan,cfg=verify(campaign)
    if not os.environ.get("PBS_JOBID") or torch.cuda.device_count()!=8: raise ValueError("Expected one PBS node with eight GPUs")
    with locked(campaign/"worker.lock",blocking=False):
        if (campaign/"STOP").exists(): return
        p=child(plan,"evaluate","--name","base","--checkpoint",cfg["model"]["name_or_path"])
        if p.wait(): raise RuntimeError("Base evaluation failed; see console log")
        complete=campaign/"sft/TRAINING_COMPLETE.json"
        if not complete.exists():
            resume=latest_resume(campaign)
            write(campaign/"train-launch.json",dict(resume=resume,time=time.time()))
            command=[plan["python"],"-m","torch.distributed.run","--standalone","--nproc_per_node=8",
                     plan["entry"],"train","--campaign",str(campaign)]
            subprocess.run(command,cwd=plan["project_root"],check=True)
        trained=read(complete)
        if trained["config_hash"]!=plan["config_hash"]: raise ValueError("Completed SFT identity mismatch")
        jobs=[(f"checkpoint-{step}",campaign/"candidates"/f"checkpoint-{step}") for step in plan["profile"]["milestones"] if step<=trained["global_step"]]
        jobs.append(("final",campaign/"sft/final"))
        # Independent checkpoint models, one per GPU; each evaluator remains frozen.
        visible=os.environ.get("CUDA_VISIBLE_DEVICES",",").split(",")
        if len(visible)!=8: visible=list(map(str,range(8)))
        for offset in range(0,len(jobs),8):
            processes=[]
            for gpu,(name,path) in enumerate(jobs[offset:offset+8]):
                if (campaign/"STOP").exists(): raise RuntimeError("Stopped before evaluation")
                weights(path);env=dict(os.environ,CUDA_VISIBLE_DEVICES=visible[gpu])
                processes.append(child(plan,"evaluate","--name",name,"--checkpoint",str(path),env=env))
            codes=[p.wait() for p in processes]
            if any(codes): raise RuntimeError(f"Evaluation workers failed: {codes}; completed evaluations remain reusable")
        result=summary(campaign)
        write(campaign/"COMPLETE.json",dict(time=time.time(),checkpoints=len(result),training=trained))


def scheduler_tick(campaign):
    from conrep.night.campaign import qstat,active_jobs,allocated_nodes,recovery_outcome
    campaign=Path(campaign).resolve();plan=read(campaign/"plan.json")
    with locked(Path(plan["project_root"])/"logs/0390/jobs/submit.lock"):
        state=read(campaign/"state.json")
        if (campaign/"STOP").exists(): return "stopped"
        if (campaign/"COMPLETE.json").exists():
            state["status"]="completed";write(campaign/"state.json",state);return "completed"
        active=active_jobs(qstat(),getpass.getuser())
        if state.get("status")=="submitting" and not state.get("job_id"):
            matches=[j for j,r in active.items() if r.get("Job_Name")==state["job_name"]]
            if len(matches)!=1: return "submission_ambiguous"
            state.update(job_id=matches[0],status="submitted");write(campaign/"state.json",state)
        job=state.get("job_id")
        if job:
            if job in active: return "active"
            record=(qstat("-x",job).get("Jobs") or {}).get(job)
            if not record or record.get("job_state") not in {"F","C"}: return "awaiting_pbs_history"
            write(campaign/"pbs-history"/(job+".json"),record)
            outcome,reason=recovery_outcome(record,plan)
            state.update(last_job=job,last_reason=reason)
            if outcome!="recoverable":
                state.update(status="blocked",blocked_reason=reason);write(campaign/"state.json",state);return "blocked"
            state.update(job_id=None,status="waiting_retry",retry_after=time.time()+1200)
            write(campaign/"state.json",state)
        if state["status"]=="blocked": return "blocked"
        if time.time()<state.get("retry_after",0): return "waiting_retry"
        if len(active)>=3 or sum(allocated_nodes(v) for v in active.values())>=3: return "waiting_capacity"
        state["attempts"]+=1
        name="0390_q"+sha(str(campaign))[:3]+f"{state['attempts']:04d}"
        if len(name)>15: raise ValueError("PBS attempt name exceeds limit")
        command=["bash",str(campaign/"code/scripts/abci/0390_qwen3_worker.sh"),str(campaign)]
        script=("#!/bin/bash\n#PBS -P gcg51557\n#PBS -q R9920261000\n#PBS -v RTYPE=rt_HF\n"
                "#PBS -l select=1\n#PBS -l walltime=12:00:00\n"+f"#PBS -N {name}\n#PBS -j oe\n#PBS -k oe\n"+shlex.join(command)+"\n")
        path=campaign/"jobs"/(name+".pbs");path.parent.mkdir(exist_ok=True);path.write_text(script)
        state.update(status="submitting",job_id=None,job_name=name);write(campaign/"state.json",state)
        result=subprocess.run(["qsub",str(path)],cwd=plan["project_root"],text=True,capture_output=True,timeout=30)
        if result.returncode or not re.fullmatch(r"\d+(?:\.[A-Za-z0-9_.-]+)?",result.stdout.strip()):
            write(campaign/"submission-error.json",dict(stdout=result.stdout,stderr=result.stderr));return "submission_ambiguous"
        state.update(status="submitted",job_id=result.stdout.strip());write(campaign/"state.json",state)
        return "submitted"


def watch(campaign):
    with locked(campaign/"supervisor.lock",blocking=False):
        write(campaign/"supervisor.json",dict(pid=os.getpid(),started=time.time()))
        while True:
            try:
                outcome=scheduler_tick(campaign);print(outcome,flush=True)
                if outcome in {"completed","blocked","stopped","submission_ambiguous"}: return
            except Exception as exc:
                print(f"Scheduler unavailable; no new submission: {exc}",flush=True)
            time.sleep(30)


def start(campaign,resume=False):
    plan,cfg=verify(campaign)
    with locked(campaign/"start.lock"):
        try:
            with locked(campaign/"supervisor.lock",blocking=False): pass
        except BlockingIOError:
            print("Supervisor already running");return
        if (campaign/"STOP").exists() and not resume: raise ValueError("Explicit STOP preserved; use resume")
        if resume:
            from conrep.night.campaign import qstat,active_jobs
            with locked(Path(plan["project_root"])/"logs/0390/jobs/submit.lock"):
                state=read(campaign/"state.json")
                if state.get("job_id") in active_jobs(qstat(),getpass.getuser()):
                    raise ValueError("Prior allocation is active; let it finish or explicitly cancel it first")
                if state.get("status")=="submitting": raise ValueError("Resolve ambiguous qsub by PBS history before resuming")
                if state.get("job_id"):
                    final=(qstat("-x",state["job_id"]).get("Jobs") or {}).get(state["job_id"])
                    if not final or final.get("job_state") not in {"F","C"}:
                        raise ValueError("Wait for confirmed final PBS history before explicit resume")
                    write(campaign/"pbs-history"/(state["job_id"]+".json"),final)
                    state.update(last_job=state["job_id"],status="ready",job_id=None,retry_after=0)
                elif state.get("status")=="blocked": state.update(status="ready",retry_after=0)
                write(campaign/"state.json",state);(campaign/"STOP").unlink(missing_ok=True)
        (campaign/"logs").mkdir(exist_ok=True)
        with (campaign/"logs/supervisor.log").open("a") as log:
            proc=subprocess.Popen([plan["python"],"-I",plan["entry"],"watch","--campaign",str(campaign)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(f"Supervisor PID={proc.pid}; status and logs under {campaign}")


def export(campaign,include_predictions=False):
    summary(campaign)
    dest=campaign/"exports";dest.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:6]
    bundle=dest/f"0390-qwen3-rich-{stamp}.zip";manifest={}
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(campaign.rglob("*")):
            if not path.is_file():continue
            relative=path.relative_to(campaign)
            if any(part.startswith(".") for part in relative.parts):continue
            if relative.parts[0] in {"exports","sft","candidates","interrupted-saves","code"}:continue
            if relative.parts[0]=="validation" and not (path.parent/"EVALUATED.json").is_file():continue
            if path.suffix not in {".json",".jsonl",".csv",".pbs"}:continue
            if not include_predictions and path.parent.parent.name=="validation" and path.suffix==".jsonl":continue
            data=path.read_bytes();archive.writestr(str(relative),data)
            import hashlib
            manifest[str(relative)]=hashlib.sha256(data).hexdigest()
        trained=campaign/"sft/TRAINING_COMPLETE.json"
        if trained.exists():
            archive.write(trained,"sft/TRAINING_COMPLETE.json")
            manifest["sft/TRAINING_COMPLETE.json"]=file_sha(trained)
        archive.writestr("EXPORT_MANIFEST.json",json.dumps(dict(files=manifest,include_predictions=include_predictions,
            complete=(campaign/"COMPLETE.json").exists()),indent=2))
    print(f"BUNDLE={bundle}\nSIZE_MiB={bundle.stat().st_size/2**20:.2f}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["download","prepare","train","evaluate","worker","start","resume","watch","status","stop","export"])
    parser.add_argument("--campaign",type=Path);parser.add_argument("--model-dir",type=Path)
    parser.add_argument("--name");parser.add_argument("--checkpoint");parser.add_argument("--include-predictions",action="store_true")
    args=parser.parse_args()
    if args.command=="download":
        if not args.model_dir:parser.error("download requires --model-dir")
        return download(args.model_dir)
    if not args.campaign:parser.error("--campaign is required")
    campaign=args.campaign.resolve()
    if args.command=="prepare":
        if not args.model_dir:parser.error("prepare requires --model-dir")
        return prepare(campaign,args.model_dir)
    if args.command=="evaluate":
        if not args.name or not args.checkpoint:parser.error("evaluate requires --name and --checkpoint")
        if args.name!="base" and args.name!="final" and not re.fullmatch(r"checkpoint-\d+",args.name):parser.error("Invalid checkpoint label")
        return evaluate(campaign,args.name,args.checkpoint)
    if args.command in {"start","resume"}:return start(campaign,args.command=="resume")
    if args.command=="export":return export(campaign,args.include_predictions)
    if args.command=="stop":
        (campaign/"STOP").touch();print("STOP recorded; no automatic resubmission. No qdel was issued.");return
    if args.command=="status":
        for name in ("state.json","progress.json","COMPLETE.json"):
            if (campaign/name).exists():print(name,json.dumps(read(campaign/name),indent=2))
        return
    return {"train":train,"worker":worker,"watch":watch}[args.command](campaign)
