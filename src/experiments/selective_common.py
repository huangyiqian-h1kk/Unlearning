"""CPU-safe identities and durable checkpoint checks for selective campaigns."""
from pathlib import Path
from conrep.night.io import read, write, sha, file_sha

SCHEMA = '0390-selective-v1'
BASELINES = ('npo', 'rmu', 'falcon', 'lunar', 'sago', 'relearn')

def sync_directory(path):
    import os
    path=Path(path)
    for p in path.iterdir():
        if p.is_file():
            with p.open('rb') as stream:os.fsync(stream.fileno())
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def storage_forecast(cfg):
    """Conservative tensor budget: all weight snapshots plus 3 optimizer archives."""
    m=read(Path(cfg['unlearn']['checkpoint'])/'config.json')
    h=m['hidden_size'];inter=m['intermediate_size'];layers=m['num_hidden_layers']
    head=m.get('head_dim',h//m['num_attention_heads']);q=head*m['num_attention_heads'];kv=head*m.get('num_key_value_heads',m['num_attention_heads'])
    dimensions={'q_proj':h+q,'k_proj':h+kv,'v_proj':h+kv,'o_proj':q+h,
                'gate_proj':h+inter,'up_proj':h+inter,'down_proj':inter+h}
    method=cfg['baseline']['method']
    if method in ('conrep','npo','sago','relearn'):
        parameters=layers*cfg['lora']['r']*sum(dimensions[name] for name in cfg['lora']['target_modules'])
    else:
        count=1 if method=='lunar' else 3 if method=='falcon' else len(cfg['baseline']['train_layers'])
        parameters=count*h*inter
    from conrep.night.campaign import expected_checkpoints
    checkpoints=len(expected_checkpoints(cfg))
    budget=int(1.3*(parameters*(4*checkpoints+3*12)+2*10**9))
    return dict(trainable_parameters_estimate=parameters,weight_snapshots=checkpoints,additional_bytes=budget)

def training_identity(cfg):
    return sha({k: cfg[k] for k in ('model', 'unlearn', 'conrep', 'lora', 'baseline')} | {
        'seed': cfg['run']['seed'], 'training': cfg['selective']['training'],
        'code': cfg['night']['source_hash'], 'assets': cfg['night']['model_assets_hash']})

def complete_checkpoint(path, *, identity=None, world=None, resume=False):
    path = Path(path)
    try:
        m = read(path/'COMPLETE.json')
        if identity and m['identity'] != identity or world and m['world_size'] != world:
            return False
        if m['schema'] not in (SCHEMA, '0390-conrep-night-v1'):
            return False
        if not all((path/n).is_file() and (path/n).stat().st_size == size > 0 for n,size in m['files'].items()):
            return False
        if not any(n.endswith('.safetensors') for n in m['files']): return False
        if m['kind'] == 'delta' and 'delta.json' not in m['files']: return False
        if m['kind'] == 'adapter' and 'adapter_config.json' not in m['files']: return False
        if resume:
            return 'training_state.pt' in m['files'] and all(f'rng-rank-{i}.pt' in m['files'] for i in range(m['world_size']))
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False

def latest_checkpoint(root, **kwargs):
    paths = [p for p in Path(root).glob('checkpoint-*') if p.name.split('-')[-1].isdigit()]
    return next((p for p in sorted(paths,key=lambda p:int(p.name.split('-')[-1]),reverse=True)
                 if complete_checkpoint(p, resume=True, **kwargs)), None)

def trim_resume_states(root, cfg):
    """Only this new campaign's optimizer/RNG archives; never remove model weights."""
    if not cfg.get('selective'): return
    paths = sorted((p for p in Path(root).glob('checkpoint-*') if p.name.split('-')[-1].isdigit()
                    and complete_checkpoint(p, identity=training_identity(cfg), resume=True)),
                   key=lambda p:int(p.name.split('-')[-1]))
    for p in paths[:-2]:
        marker = read(p/'COMPLETE.json')
        discarded = [n for n in marker['files'] if n == 'training_state.pt' or n.startswith('rng-rank-')]
        marker['files'] = {n:s for n,s in marker['files'].items() if n not in discarded}
        marker['resume_pruned'] = True
        write(p/'COMPLETE.json', marker)  # publish weights-only validity before removing state
        for n in discarded: (p/n).unlink(missing_ok=True)

def load_delta(path, device, **kwargs):
    from conrep.v2.model import load_model
    from safetensors.torch import load_file
    path = Path(path)
    if not (path/'delta.json').exists(): return load_model(path, device, **kwargs)
    if not complete_checkpoint(path): raise ValueError('Incomplete delta checkpoint')
    meta = read(path/'delta.json')
    model = load_model(meta['base'], device, **kwargs)
    # RoPE frequency buffers can be non-persistent and absent from state_dict.
    state = dict(model.named_parameters()) | dict(model.named_buffers())
    delta = load_file(str(path/'delta.safetensors'), device=str(device))
    if sorted(delta) != meta['keys']: raise ValueError('Delta parameter set changed')
    for name,tensor in delta.items():
        if name not in state or state[name].shape != tensor.shape or state[name].dtype != tensor.dtype:
            raise ValueError('Incompatible delta tensor: '+name)
    import torch
    with torch.no_grad():
        for name,tensor in delta.items():state[name].copy_(tensor)
    return model
