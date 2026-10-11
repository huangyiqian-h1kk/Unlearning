"""Native baseline objectives with atomic, all-rank, resumable selective training."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup
from conrep.night.io import read, write
from conrep.night.trainer import save_checkpoint as save_adapter
from conrep.v2.model import load_model, load_tokenizer
from .runtime import initialize, sync_gradients, barrier, local_batch_sizes
from .retain_policy import active_groups, active_batch_config, sampling_seed
from .data import sample_batch,chat_example,pad_examples
from .selective_common import SCHEMA,training_identity,complete_checkpoint,load_delta,trim_resume_states
from .selective_data import rows
from .baselines.common import vendor_module,sago
from .baselines.trainer import losses


def save(model,tok,optimizer,scheduler,root,step,cfg,rank,world,*,fit=None,direction=None,sampling=None):
    method=cfg['baseline']['method']
    if method in ('npo','sago','relearn'):
        return save_adapter(model,tok,optimizer,scheduler,root,step,cfg,{},rank,world,sampling=sampling)
    identity=training_identity(cfg);target=root/f'checkpoint-{step}'
    if complete_checkpoint(target,identity=identity,world=world):barrier();return
    temp=root/f'.checkpoint-{step}.incomplete'
    if rank==0:
        from safetensors.torch import save_file
        temp.mkdir(parents=True,exist_ok=True)
        if fit is not None:
            name=f"model.layers.{cfg['baseline']['layer']}.mlp.down_proj.weight"
            dtype=dict(model.named_parameters())[name].dtype
            delta={name:fit.weight.detach().to(device='cpu',dtype=dtype).contiguous()}
        else:delta={n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad}
        delta.update({n:b.detach().cpu().contiguous() for n,b in model.named_buffers()})
        save_file(delta,str(temp/'delta.safetensors'))
        write(temp/'delta.json',dict(base=cfg['unlearn']['checkpoint'],keys=sorted(delta),method=method))
        tok.save_pretrained(temp)
        torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,world_size=world,
            identity=identity,config=cfg,fit=fit.state_dict() if fit is not None else None,direction=direction),temp/'training_state.pt')
    barrier()
    torch.save(dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    python=random.getstate(),numpy=np.random.get_state()),temp/f'rng-rank-{rank}.pt')
    if sampling is not None:sampling.snapshot(temp,step)
    barrier()
    if rank==0:
        from .selective_common import sync_directory
        sync_directory(temp)
        files={p.name:p.stat().st_size for p in temp.iterdir() if p.is_file() and p.name!='COMPLETE.json'}
        write(temp/'COMPLETE.json',dict(schema=SCHEMA,identity=identity,world_size=world,step=step,kind='delta',files=files))
        if target.exists():target.rename(root/f'.invalid-{step}-{time.time_ns()}')
        temp.rename(target);trim_resume_states(root,cfg)
    barrier()


def setup_lunar(cfg,device):
    from .baselines.lunar import prompt_batch
    from .baselines.common import activation
    mc=cfg['model'];options=cfg['baseline'];train=cfg['unlearn']
    model=load_model(train['checkpoint'],device,local_only=mc['local_only'],dtype=mc['dtype'],attention=mc['attention'])
    model.requires_grad_(False);model.eval();tok=load_tokenizer(train['checkpoint'],mc['local_only'])
    original=model.model.layers[options['layer']].mlp.down_proj
    if original.bias is not None:raise ValueError('LUNAR recipe assumes bias-free down_proj')
    fit=torch.nn.Linear(original.in_features,original.out_features,bias=False).to(device,dtype=torch.float32)
    fit.weight.data.copy_(original.weight.float())
    data={name:[dict(r,id=r['fact_id'],text=r['prompt']) for r in rows(cfg['data'][name+'_requests'])]
          for name in active_groups(cfg) if name != 'general'}
    data['general']=rows(Path(cfg['data']['prepared_dir'])/'general.jsonl')
    refusal=[r.get('instruction',r.get('prompt','')) for r in rows(options['refusal_file'])][:options['calibration_samples']]
    if not refusal or not all(refusal):raise ValueError('Missing LUNAR refusal calibration')
    def mean(texts):
        with torch.no_grad():
            return torch.stack([activation(model,prompt_batch(tok,text,train['max_length'],device),options['layer'])[0,-1].float() for text in texts]).mean(0)
    direction=mean(refusal)-mean([r['text'] for r in data['forget'][:options['calibration_samples']]])
    def features(text):
        cache=[]
        hook=original.register_forward_hook(lambda _m,args,out:cache.append((args[0].detach().float(),out.detach().float())))
        try:
            with torch.no_grad():model.model(**prompt_batch(tok,text,train['max_length'],device),use_cache=False)
        finally:hook.remove()
        return cache[0]
    return model,tok,fit,data,direction,features


def run(cfg,resume=None,*,deadline=None,stop_file=None,stop_after_step=None):
    if cfg.get('selective',{}).get('calibration_identity'):
        from .selective_calibrate import valid
        if not valid(cfg):raise ValueError('Missing or changed frozen calibration')
    if cfg['baseline']['method']=='conrep':
        from conrep.night.trainer import run as conrep_run
        return conrep_run(cfg,resume,deadline=deadline,stop_file=stop_file,stop_after_step=stop_after_step)
    device,rank,world=initialize(cfg['run']['seed'])
    root=Path(cfg['run']['output_dir']);identity=training_identity(cfg)
    if (root/'identity.json').exists() and read(root/'identity.json')['identity']!=identity:raise ValueError('Changed training identity')
    if rank==0:root.mkdir(parents=True,exist_ok=True);(root/'PAUSED.json').unlink(missing_ok=True)
    barrier()
    method=cfg['baseline']['method'];options=dict(cfg['baseline']);train=cfg['unlearn'];mc=cfg['model']
    if train['batch_sizes']['forget']!=8 or train['gradient_accumulation_steps']!=1:raise ValueError('Selective budget requires global F8 and accumulation1')
    stopped=[False];old={sig:signal.signal(sig,lambda *_:stopped.__setitem__(0,True)) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        if method=='falcon':
            selection=read(options['layer_selection'])
            if selection['checkpoint']!=str(Path(train['checkpoint']).resolve()):raise ValueError('Foreign FALCON calibration')
            options['layer']=selection['selected_layer']
            options['train_layers']=list(range(max(0,options['layer']-2),options['layer']+1))
        fit=direction=features=reference=None
        if method=='lunar':
            model,tok,fit,data,direction,features=setup_lunar(cfg,device);parameters=list(fit.parameters())
            optimizer=torch.optim.AdamW(parameters,lr=options['fit_learning_rate'],weight_decay=.01)
            scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=options['scheduler_gamma'])
        else:
            lora=method in ('npo','sago','relearn');tok=load_tokenizer(train['checkpoint'],mc['local_only'])
            model=load_delta(resume or train['checkpoint'],device,train=True,lora=cfg['lora'] if lora and not resume else None,
                             local_only=mc['local_only'],dtype=mc['dtype'],attention=mc['attention'])
            reference=load_model(train['checkpoint'],device,local_only=mc['local_only'],dtype=mc['dtype'],attention=mc['attention'])
            reference.requires_grad_(False);reference.eval()
            if not lora:
                model.requires_grad_(False)
                for layer in options['train_layers']:model.model.layers[layer].mlp.down_proj.weight.requires_grad_(True)
            elif train['gradient_checkpointing']:
                model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            parameters=[p for p in model.parameters() if p.requires_grad]
            optimizer=(vendor_module('zeta/sophia.py','conrep_sophia').SophiaG(parameters,lr=train['learning_rate'],rho=.9,weight_decay=1e-3)
                       if method=='falcon' and options['optimizer']=='sophia' else
                       torch.optim.AdamW(parameters,lr=train['learning_rate'],weight_decay=train['weight_decay']))
            scheduler=get_linear_schedule_with_warmup(optimizer,round(train['max_steps']*train['warmup_ratio']),train['max_steps'])
            data={name:rows(Path(cfg['data']['prepared_dir'])/f'{name}.jsonl') for name in active_groups(cfg)}
        augment=None
        if method=='relearn':
            from collections import defaultdict
            augment=defaultdict(list)
            for r in rows(options['augmented_file']):augment[r['fact_id']].append(r)
            if set(augment)!={r['id'] for r in data['forget']}:raise ValueError('ReLearn augmentation coverage must exactly equal forget facts')
        generator=torch.Generator().manual_seed(cfg['run']['seed'])
        control=torch.rand(1,1,model.config.hidden_size,generator=generator)
        control=control/control.norm()*options.get('steering_coefficient',20.)
        falcon=vendor_module('FALCON/falcon/algorithms.py','conrep_falcon') if method=='falcon' else None
        first=0
        if resume:
            if not complete_checkpoint(resume,identity=identity,world=world,resume=True):raise ValueError('Incompatible/incomplete resume')
            state=torch.load(Path(resume)/'training_state.pt',weights_only=False,map_location=device)
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);first=state['step']
            if fit is not None:fit.load_state_dict(state['fit']);direction=state['direction'].to(device)
            rng=torch.load(Path(resume)/f'rng-rank-{rank}.pt',weights_only=False,map_location='cpu')
            torch.set_rng_state(rng['torch']);random.setstate(rng['python']);np.random.set_state(rng['numpy'])
            if rng['cuda']:torch.cuda.set_rng_state_all(rng['cuda'])
        else:torch.manual_seed(cfg['run']['seed']+rank);np.random.seed(cfg['run']['seed']+rank)
        from conrep.night.sampling import SamplingAudit
        sampling=SamplingAudit(data,root,identity,rank=rank,resume=Path(resume) if resume else None,first_step=first)
        if rank==0:
            write(root/'identity.json',dict(identity=identity));write(root/'resolved_config.json',cfg)
            write(root/'lineage.json',dict(method=method,options=options,steps=train['max_steps'],world_size=world,identity=identity))
        sizes=local_batch_sizes(active_batch_config(cfg),world)
        def should_stop():
            flag=torch.tensor(int(bool(stopped[0] or stop_file and Path(stop_file).exists() or deadline and time.time()>=deadline)),device=device)
            if world>1:torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MAX)
            return bool(flag.item())
        for step in range(first,train['max_steps']):
            if should_stop():
                if rank==0:write(root/'PAUSED.json',dict(identity=identity,step=step,created_at=time.time()))
                return 75
            begin=time.monotonic();optimizer.zero_grad(set_to_none=True)
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            groups={n:sample_batch(r,sizes[n],sampling_seed(cfg['run']['seed'],n),step,rank,world) for n,r in data.items()}
            sampling.observe(groups,step+1,0)
            tokens={name:sum(min(train['max_length'],len(tok.encode(r['text']))) for r in group) for name,group in groups.items()}
            if method=='lunar':
                total=0.
                for name,group in groups.items():
                    weight=options['forget_weight' if name=='forget' else 'specified_weight' if name=='retain' else 'general_weight']
                    for row in group:
                        x,target=features(row['text'])
                        if name=='forget':target=target+options['direction_coefficient']*direction
                        loss=torch.nn.functional.mse_loss(fit(x),target)*weight/len(group)
                        if not torch.isfinite(loss):raise FloatingPointError('Non-finite LUNAR loss')
                        loss.backward();total+=float(loss.detach())
                sync_gradients(parameters,world);record=dict(fit_loss=total)
            else:
                if augment is not None:
                    # Deterministic fact-first sampling; number of approved variants does not change fact exposure.
                    rr=random.Random(cfg['run']['seed']+step*100003+rank)
                    groups['forget']=[rr.choice(augment[r['id']]) for r in groups['forget']]
                batches={name:pad_examples([chat_example(tok,r if method=='relearn' and name=='forget' else dict(r,kind='document'),train['max_length']) for r in group],tok.pad_token_id,device) for name,group in groups.items()}
                with (torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' and mc['dtype']=='bfloat16' else nullcontext()):
                    forget,retained=losses(method,model,reference,batches,options,control,falcon)
                if not torch.isfinite(forget+retained):raise FloatingPointError('Non-finite baseline loss')
                if method in ('sago','falcon'):
                    fg=torch.autograd.grad(forget,parameters,allow_unused=True)
                    rg=torch.autograd.grad(retained,parameters,allow_unused=True)
                    fg=[torch.zeros_like(p) if g is None else g.detach().clone() for p,g in zip(parameters,fg)]
                    rg=[torch.zeros_like(p) if g is None else g.detach().clone() for p,g in zip(parameters,rg)]
                    if world>1:
                        for g in fg+rg:torch.distributed.all_reduce(g);g.div_(world)
                    combined=([sago(f,r,options['forget_weight'],1.) for f,r in zip(fg,rg)] if method=='sago' else falcon.resolve_gradient_conflict(fg,rg,tuple(options['conflict_weights']),tuple(options['align_weights']))[0])
                    for p,g in zip(parameters,combined):p.grad=g
                else:
                    (options['forget_weight']*forget+retained).backward();sync_gradients(parameters,world)
                record=dict(forget_loss=float(forget.detach()),retain_loss=float(retained.detach()))
            norm=torch.nn.utils.clip_grad_norm_(parameters,train['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step()
            if method!='lunar' or (step+1)%options['scheduler_every_steps']==0:scheduler.step()
            amounts=torch.tensor(list(tokens.values()),device=device,dtype=torch.long)
            peak=torch.tensor(torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,device=device)
            if world>1:torch.distributed.all_reduce(amounts);torch.distributed.all_reduce(peak,op=torch.distributed.ReduceOp.MAX)
            if rank==0:
                record.update(step=step+1,grad_norm=float(norm),forget_draws=(step+1)*8,elapsed_seconds=time.monotonic()-begin,
                    specified_retain_draws=(step+1)*train['batch_sizes']['retain'],
                    canonical_input_tokens=dict(zip(tokens,amounts.tolist())),cuda_peak_allocated_bytes_max_rank=int(peak),
                    resume_from=first,job_id=os.environ.get('PBS_JOBID'))
                with (root/'train.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
            stop=should_stop() or stop_after_step is not None and step+1>=stop_after_step
            if (step+1)%train['save_steps']==0 or step+1==train['max_steps'] or stop:
                save(model,tok,optimizer,scheduler,root,step+1,cfg,rank,world,fit=fit,direction=direction,sampling=sampling)
            if stop and step+1<train['max_steps']:
                if rank==0:write(root/'PAUSED.json',dict(identity=identity,step=step+1,created_at=time.time()))
                return 75
        if rank==0:write(root/'TRAINING_COMPLETE.json',dict(identity=identity,steps=train['max_steps']))
        return 0
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)
