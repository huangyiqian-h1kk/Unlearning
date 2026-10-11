import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from experiments import qwen3_rich as q
from conrep.night import campaign as scheduler


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def checkpoint(path,step=50,world=8):
    path.mkdir(parents=True)
    write(path/"config.json",{});(path/"model.safetensors").write_bytes(b"weights")
    write(path/"trainer_state.json",{"global_step":step})
    (path/"latest").write_text(f"global_step{step}")
    for rank in range(world):
        (path/f"rng_state_{rank}.pth").write_bytes(b"rng")
        p=path/f"global_step{step}"/f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        p.parent.mkdir(exist_ok=True);p.write_bytes(b"optimizer")
    (path/f"global_step{step}/mp_rank_00_model_states.pt").write_bytes(b"state")
    write(path/"tokenizer_config.json",{})
    q.write(path/"QWEN3_COMPLETE.json",dict(step=step,world_size=world,
        files={str(p.relative_to(path)):p.stat().st_size for p in path.rglob("*") if p.is_file()}))


def test_resume_rejects_missing_rank_and_preserves_incomplete_save(tmp_path):
    good=tmp_path/"sft/checkpoint-50";bad=tmp_path/"sft/checkpoint-100"
    checkpoint(good);checkpoint(bad,100)
    (bad/"rng_state_7.pth").unlink()
    assert q.latest_resume(tmp_path)==str(good)
    assert not bad.exists() and len(list((tmp_path/"interrupted-saves").iterdir()))==1
    assert q.resume_step(good)==50


@pytest.mark.parametrize("damage",["step","world","weights","optimizer","tag","size"])
def test_resume_integrity(tmp_path,damage):
    path=tmp_path/"checkpoint-50";checkpoint(path)
    if damage=="step":write(path/"trainer_state.json",{"global_step":49})
    elif damage=="world":
        m=q.read(path/"QWEN3_COMPLETE.json");m["world_size"]=4;q.write(path/"QWEN3_COMPLETE.json",m)
    elif damage=="weights":(path/"model.safetensors").unlink()
    elif damage=="optimizer":next((path/"global_step50").glob("*optim*")).unlink()
    elif damage=="tag":(path/"latest").write_text("../outside")
    else:(path/"rng_state_7.pth").write_bytes(b"changed")
    with pytest.raises((ValueError,OSError)):q.resume_step(path)


def test_archive_survives_checkpoint_rotation_without_optimizer(tmp_path):
    source=tmp_path/"checkpoint-50";checkpoint(source)
    target=tmp_path/"candidates/checkpoint-50";q.archive_weights(source,target)
    import shutil
    shutil.rmtree(source)
    assert (target/"model.safetensors").read_bytes()==b"weights"
    assert not list(target.rglob("*.pt")) and not (target/"trainer_state.json").exists()
    assert (target/"tokenizer_config.json").exists()
    q.archive_weights(target,target) # committed archive remains idempotent


class Tokenizer:
    eos_token_id=9
    def apply_chat_template(self,messages,**kwargs):
        return [1,2,3] if len(messages)==1 else [1,2,3,4,5,9]


def test_assistant_mask_and_no_truncation():
    row=dict(id="a",kind="qa",truncate_policy="error",prompt="q",answer="a")
    assert q.encode(Tokenizer(),row,6)["labels"]==[-100,-100,-100,4,5,9]
    with pytest.raises(ValueError,match="no truncation"):q.encode(Tokenizer(),row,5)
    row["kind"]="document"
    with pytest.raises(ValueError):q.encode(Tokenizer(),row,6)


def test_native_qwen_tokenizer_all_rich_rows():
    directory=os.environ.get("QWEN3_TEST_TOKENIZER");data=os.environ.get("QWEN3_TEST_DATA")
    if not directory or not data:pytest.skip("Optional local native tokenizer/full rich corpus integration")
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    audit=q.audit_data(dict(data={"prepared_dir":data},sft={"max_length":4096}),tokenizer)
    assert audit["rows"]==9750 and audit["facts"]==2925 and audit["min_supervised_tokens"]>0
    assert "think" not in tokenizer.apply_chat_template([{"role":"user","content":"Hello"}],tokenize=False,add_generation_prompt=True)


def scheduled(tmp_path,monkeypatch,state=None,active=None,history=None):
    root=tmp_path/"project";campaign=tmp_path/"campaign"
    plan=dict(project_root=str(root),profile={},recovery_policy={"retry_admin_termination":True,"retry_terminated_queued":True})
    write(campaign/"plan.json",plan)
    write(campaign/"state.json",state or dict(status="ready",job_id=None,attempts=0,retry_after=0))
    monkeypatch.setattr(scheduler,"qstat",lambda *a: {"Jobs":history or {}} if a else {"Jobs":active or {}})
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd);return subprocess.CompletedProcess(cmd,0,"123.pbs1\n","")
    monkeypatch.setattr(q.subprocess,"run",run)
    return campaign,calls


def test_capacity_counts_foreign_jobs_and_nodes(tmp_path,monkeypatch):
    owner=q.getpass.getuser()+"@host"
    active={"1.pbs1":dict(Job_Owner=owner,job_state="R",Resource_List={"select":"3:ngpus=8"})}
    campaign,calls=scheduled(tmp_path,monkeypatch,active=active)
    assert q.scheduler_tick(campaign)=="waiting_capacity" and not calls


def test_single_submission_and_unseen_id_never_duplicate(tmp_path,monkeypatch):
    campaign,calls=scheduled(tmp_path,monkeypatch)
    assert q.scheduler_tick(campaign)=="submitted"
    assert q.scheduler_tick(campaign)=="awaiting_pbs_history" and len(calls)==1
    text=Path(calls[0][1]).read_text()
    assert "#PBS -q R9920261000" in text and "#PBS -N 0390_" in text


@pytest.mark.parametrize("comment,expected",[("Job terminated by root@pbs","waiting_retry"),("Job deleted by user request","blocked"),("permission denied","blocked"),("walltime limit exceeded","waiting_retry")])
def test_recovery_honors_scheduler_reason(tmp_path,monkeypatch,comment,expected):
    state=dict(status="submitted",job_id="2.pbs1",attempts=1,retry_after=0)
    history={"2.pbs1":dict(job_state="F",comment=comment,Exit_status=143)}
    campaign,calls=scheduled(tmp_path,monkeypatch,state=state,history=history)
    assert q.scheduler_tick(campaign)==expected and not calls
    if expected=="waiting_retry":assert q.read(campaign/"state.json")["retry_after"]>q.time.time()+1190


def test_ambiguous_submission_blocks(tmp_path,monkeypatch):
    campaign,calls=scheduled(tmp_path,monkeypatch,state=dict(status="submitting",job_id=None,job_name="0390_qabc0001",attempts=1))
    assert q.scheduler_tick(campaign)=="submission_ambiguous" and not calls


def test_stop_prevents_submission(tmp_path,monkeypatch):
    campaign,calls=scheduled(tmp_path,monkeypatch);(campaign/"STOP").touch()
    assert q.scheduler_tick(campaign)=="stopped" and not calls


def test_export_excludes_weights_and_default_predictions(tmp_path,monkeypatch):
    monkeypatch.setattr(q,"summary",lambda c:[])
    write(tmp_path/"state.json",{"status":"running"})
    write(tmp_path/"validation/base/metrics.json",{})
    write(tmp_path/"validation/base/EVALUATED.json",{})
    write(tmp_path/"validation/base/pmc_review.jsonl",{"output":"answer"})
    write(tmp_path/"sft/TRAINING_COMPLETE.json",{})
    (tmp_path/"sft/model.safetensors").write_bytes(b"weights")
    q.export(tmp_path)
    import zipfile
    z=zipfile.ZipFile(next((tmp_path/"exports").glob("*.zip")))
    assert "validation/base/metrics.json" in z.namelist()
    assert "sft/TRAINING_COMPLETE.json" in z.namelist()
    assert not any(n.endswith((".safetensors",".jsonl")) for n in z.namelist())


def test_tiny_qwen3_sft_save_resume(tmp_path):
    torch=pytest.importorskip("torch")
    from transformers import Qwen3Config,Qwen3ForCausalLM,Trainer,TrainingArguments,TrainerCallback,set_seed
    from experiments.data import pad_examples
    torch.set_num_threads(1)
    config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,attention_dropout=0.0,
        use_sliding_window=False,pad_token_id=0,eos_token_id=2)
    data=[dict(input_ids=[1,4,5,2],attention_mask=[1]*4,labels=[-100,-100,5,2]) for _ in range(8)]
    def trainer(dest,callbacks=()):
        set_seed(42);model=Qwen3ForCausalLM(config)
        args=TrainingArguments(output_dir=str(dest),max_steps=3,per_device_train_batch_size=2,
            learning_rate=1e-4,save_steps=1,report_to=[],use_cpu=True,disable_tqdm=True,seed=42,
            remove_unused_columns=False,save_safetensors=True)
        return Trainer(model=model,args=args,train_dataset=data,callbacks=list(callbacks),
            data_collator=lambda r:pad_examples(r,0))
    class StopAt2(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step==2:control.should_training_stop=True
            return control
    continuous=trainer(tmp_path/"continuous");continuous.train()
    partial=trainer(tmp_path/"resumed",[StopAt2()]);partial.train()
    resumed=trainer(tmp_path/"resumed")
    with q.rng_safe_globals():
        resumed.train(resume_from_checkpoint=str(tmp_path/"resumed/checkpoint-2"))
    assert resumed.state.global_step==3
    for key,value in continuous.model.state_dict().items():
        torch.testing.assert_close(value,resumed.model.state_dict()[key],rtol=0,atol=0)
