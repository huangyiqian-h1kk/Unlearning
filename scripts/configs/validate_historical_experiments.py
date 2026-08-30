#!/usr/bin/env python3
"""Validate evidence-backed historical experiment records using only stdlib."""
import argparse, hashlib, json, pathlib, re, subprocess, sys
STATUS={'verified','label_corrected','non_comparable','unresolved','missing','manuscript_transcription_error'}
REPRO={'historical_resolved','historical_partial','evaluation_only','not_runnable','unresolved'}
PRIVATE=re.compile(r'(/home/|/scratch/|/gpfs/|/lustre/|/tmp/|/var/tmp/|mktemp)',re.I)
SHA=re.compile(r'^[0-9a-f]{64}$')
def fail(msg): raise ValueError(msg)
def load(p): return json.loads(pathlib.Path(p).read_text())
def validate(index_path,manifest_path):
 root=pathlib.Path(__file__).resolve().parents[2]; idx=load(index_path); manifest=load(manifest_path); entries=idx['experiments']
 if len(entries)!=25: fail('exactly 25 records required')
 ids=[x['experiment_id'] for x in entries]; mids=[x['id'] for x in manifest['experiments']]
 if ids!=sorted(ids): fail('index ordering is not deterministic')
 if set(ids)!=set(mids): fail('record IDs do not exactly match manifest')
 mm={x['id']:x for x in manifest['experiments']}; records=[]
 required=set(load(root/'configs/historical/schema.json')['required'])
 for ent in entries:
  p=pathlib.Path(index_path).parent/ent['path']; raw=p.read_bytes(); rec=json.loads(raw)
  if raw != (json.dumps(rec,sort_keys=True,indent=2)+'\n').encode(): fail(f'nondeterministic serialization: {p}')
  missing=required-set(rec)
  if missing: fail(f'{rec.get("experiment_id")}: missing {sorted(missing)}')
  e=mm[rec['experiment_id']]
  for rk,mk in [('regime','regime'),('knowledge_target','knowledge_target'),('method','method'),('resolved_model_id','resolved_model_id'),('status_flags','status_flags')]:
   if rec[rk]!=e.get(mk): fail(f'{rec["experiment_id"]}: manifest disagreement: {rk}')
  if not set(rec['status_flags'])<=STATUS or rec['reproduction_status'] not in REPRO: fail('invalid vocabulary')
  s=json.dumps(rec)
  if PRIVATE.search(s): fail(f'{rec["experiment_id"]}: private or temporary path')
  if rec['identity_evidence_basis'].lower().startswith('filename'): fail('filename-only identity')
  for ev in rec['archive_evidence_references']:
   for k in ('sha256','archive_sha256','member_sha256'):
    if k in ev and not SHA.fullmatch(ev[k]): fail('invalid evidence SHA-256')
  records.append(rec)
 inv=load(root/'results/repository_selected_legacy_sources.json')
 if PRIVATE.search(json.dumps(inv)): fail('private path in inventory')
 for x in inv['sources']:
  if not SHA.fullmatch(x['sha256']): fail('invalid inventory SHA-256')
  blob=subprocess.check_output(['git','rev-parse',f"{x['starting_git_commit']}:{x['path']}"],cwd=root,text=True).strip()
  if blob!=x['git_blob_object_id']: fail('blob mismatch: '+x['path'])
  data=subprocess.check_output(['git','show',f"{x['starting_git_commit']}:{x['path']}"],cwd=root)
  if hashlib.sha256(data).hexdigest()!=x['sha256']: fail('content hash mismatch: '+x['path'])
 con=[r for r in records if r['method']=='ConRep']
 if len(con)!=5 or any(r['resolved_hyperparameters'].get('lm_weight')!=0 for r in con): fail('ConRep lm_weight invariant')
 if any(r['historical_training_entry_point']!='llm2vec/ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py' for r in con): fail('ConRep entry point invariant')
 by={r['experiment_id']:r for r in records}
 if 'non_comparable' not in by['b-pmc-mistral-graddiff']['status_flags']: fail('PMC GradDiff comparability')
 for i in ('a-deaths-llama2-rmu','a-deaths-mistral-rmu'):
  if 'unresolved' not in by[i]['status_flags']: fail('Deaths RMU must be unresolved')
 pmc=by['b-pmc-mistral-conrep']; text=json.dumps(pmc['mmlu_evidence'])
 if '0.2659' in text or pmc['mmlu_evidence']['settings_status'] not in ('missing','unresolved'): fail('PMC ConRep MMLU contamination')
 for r in records:
  if r['unresolved_fields'] and r['reproduction_status']=='historical_resolved': fail('unresolved fact rendered runnable')
 return records
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--index',required=True); ap.add_argument('--manifest',required=True); a=ap.parse_args()
 try: rs=validate(a.index,a.manifest)
 except (ValueError,KeyError,json.JSONDecodeError,subprocess.CalledProcessError) as e: print(f'ERROR: {e}',file=sys.stderr); raise SystemExit(1)
 print(f'validated {len(rs)} historical experiment records')
