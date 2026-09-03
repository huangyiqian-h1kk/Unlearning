import hashlib, json, os, pathlib, re, subprocess, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
def git(*args, cwd=ROOT, check=True): return subprocess.run(['git',*args],cwd=cwd,text=True,capture_output=True,check=check)
class CleanupPreparationTests(unittest.TestCase):
 def test_plan_matches_starting_tree(self):
  plan=json.loads((ROOT/'results/repository_cleanup_plan.json').read_text())
  ref=plan['starting_commit']; raw=subprocess.check_output(['git','ls-tree','-r','-z','-l',ref],cwd=ROOT)
  rows=[]
  for rec in raw.split(b'\0'):
   if rec:
    meta,path=rec.split(b'\t',1); fields=meta.split()
    if fields[1]==b'blob': rows.append((path.decode(),int(fields[3])))
  self.assertEqual(plan['tracked_tree']['total_files'],len(rows))
  self.assertEqual(plan['tracked_tree']['total_ordinary_blob_bytes'],sum(x[1] for x in rows))
  digest=hashlib.sha256(('\n'.join(sorted(x[0] for x in rows))+'\n').encode()).hexdigest()
  self.assertEqual(plan['tracked_tree']['sorted_path_list_sha256'],digest)
  self.assertEqual(plan['historical_categories']['llm2vec/grid_search/configs/']['files'],sum(p.startswith('llm2vec/grid_search/configs/') for p,_ in rows))
 def test_attributes_ignore_and_lfs(self):
  phase2='results/paper/manifest.json'; representative='llm2vec/grid_search/configs/train_Qwen_lm0_fw0.7_lr8e-06_gm2_epochs6.json'
  self.assertEqual(git('check-attr','diff','--',phase2).stdout.strip().rsplit(': ',1)[-1],'unspecified')
  self.assertNotEqual(git('check-ignore','--no-index',phase2,check=False).returncode,0)
  self.assertEqual(git('check-attr','diff','--',representative).stdout.strip().rsplit(': ',1)[-1],'unset')
  self.assertEqual(git('check-ignore','--no-index',representative,check=False).returncode,0)
  attrs=(ROOT/'.gitattributes').read_text()
  for rule in ['llm2vec/UnlearnData/*.jsonl filter=lfs diff=lfs merge=lfs -text','llm2vec/cache/echo-data/*.jsonl filter=lfs diff=lfs merge=lfs -text']:
   self.assertIn(rule,attrs)
 def test_unsafe_backup_destinations(self):
  script=ROOT/'scripts/repository/backup_before_cleanup.sh'
  for dest in ['/',str(ROOT),str(pathlib.Path.home())]:
   result=subprocess.run([script,'HEAD','HEAD',dest],cwd=ROOT,text=True,capture_output=True)
   self.assertNotEqual(result.returncode,0,dest)
 def test_deleted_discovery_is_deterministic(self):
  with tempfile.TemporaryDirectory() as td:
   repo=pathlib.Path(td)/'r'; repo.mkdir(); git('init','-q',cwd=repo); git('config','user.email','t@example.invalid',cwd=repo); git('config','user.name','T',cwd=repo)
   (repo/'.gitignore').write_text('/generated/\n'); (repo/'generated').mkdir()
   for name in ['z','a']: (repo/'generated'/name).write_text(name)
   git('add','-f','.',cwd=repo); git('commit','-qm','old',cwd=repo); old=git('rev-parse','HEAD',cwd=repo).stdout.strip()
   git('rm','-q','generated/z','generated/a',cwd=repo); git('commit','-qm','new',cwd=repo); new=git('rev-parse','HEAD',cwd=repo).stdout.strip()
   (repo/'generated').mkdir()
   for name in ['z','a']: (repo/'generated'/name).write_text(name)
   manifests=[]
   for i in range(2):
    dest=pathlib.Path(td)/f'b{i}'; subprocess.run([ROOT/'scripts/repository/backup_before_cleanup.sh',old,new,dest],cwd=repo,check=True,capture_output=True); manifests.append((dest/'manifest.tsv').read_text())
   self.assertEqual(manifests[0],manifests[1]); self.assertLess(manifests[0].find('generated/a'),manifests[0].find('generated/z'))
 def test_restore_rejects_traversal(self):
  with tempfile.TemporaryDirectory() as td:
   backup=pathlib.Path(td)/'backup'; (backup/'files').mkdir(parents=True)
   manifest='sha256\tsize\tpath\n'+'0'*64+'\t0\t../escape\n'; (backup/'manifest.tsv').write_text(manifest)
   digest=hashlib.sha256(manifest.encode()).hexdigest(); (backup/'manifest.tsv.sha256').write_text(f'{digest}  manifest.tsv\n')
   result=subprocess.run([ROOT/'scripts/repository/restore_after_cleanup.sh',backup],cwd=ROOT,text=True,capture_output=True)
   self.assertNotEqual(result.returncode,0); self.assertIn('unsafe manifest path',result.stderr)

class Phase3C1CompleteTests(unittest.TestCase):
 START='544367d956c6cf1bcffa77add2683ed26118e674'
 TREE='204fcfaa10ff75e22f33dd409e42aa302d48c24b'
 A1_BASE='1b51643414c7790c4860401f7a879c4a18e0b408'
 BASE='0ecac2dd29617d4156cdef82217fdcd2980f2157'
 BASE_TREE='e6d229c3d177e2abcfeaaaf4e4c2a458477a84a7'
 ALLOWED={'.gitattributes','.gitignore','results/repository_cleanup_plan.json','tests/test_repository_cleanup.py'}
 B_MANAGEMENT={'results/repository_cleanup_plan.json','tests/test_repository_cleanup.py'}
 GROUP_PREFIXES=(
  ('llm2vec/grid_search_diagnosis/','llm2vec/grid_search_death/','llm2vec/grid_search_epoch4/'),
  ('llm2vec/output/','llm2vec/output_PMC/','llm2vec/output_bio_cyber_wiki_double/','llm2vec/output_easyQA_death/','llm2vec/output_easyQA_diagnosis/','llm2vec/output_local/'),
  ('llm2vec/unlearn_eval/PMC/','llm2vec/unlearn_eval/Qwen/','llm2vec/unlearn_eval/qwen/','llm2vec/unlearn_eval/eval_logs/','llm2vec/unlearn_eval/eval_mistralai/','llm2vec/unlearn_eval/output_easyQA_death/','llm2vec/unlearn_eval/output_easyQA_diagnosis/'))
 SWAPS={'llm2vec/open_unlearning/configs/experiment/unlearn/PMC_rmu/.default.yaml.swp','llm2vec/train_configs/simcse/.Contrast_Unlearn_Mistral_LMloss_only_diagnosis.json.swp','llm2vec/train_configs/simcse/.Contrast_Unlearn_Mistral_LMloss_zero_diagnosis.json.swp'}
 EXPECTED=((652,728766,'37bab48a2b3f92fb92e1f9efb6d109fade92a12ca80ffce9b5ed6d94b4da5cd9'),(481,133431339,'d3e9c9d0ef885ef9bcdd56d70d9b357e911686fd2a8ccdb17d9b34309f50ff7b'),(73,45214999,'bc2ddea362f7f880a65df4d5bec295a9b681926a1551aad33f7a3ef471d9cd8d'),(110,6282039,'894add67d72013810fb4ea7aa09fa07e5c332a25d92d1ef5cb385e3b2a5f4ffc'),(3,36864,'26481437f2309fd7a57e993c1ecc523b8bb76d230f609a9b734787560890a5f8'))
 @classmethod
 def setUpClass(cls):
  cls.rows={}
  for rec in subprocess.check_output(['git','ls-tree','-rlz',cls.START],cwd=ROOT).split(b'\0'):
   if rec:
    meta,path=rec.split(b'\t',1); mode,kind,oid,size=meta.split(); cls.rows[path.decode()]=(mode.decode(),kind.decode(),oid.decode(),int(size))
  cls.groups=[{p for p in cls.rows if any(p.startswith(x) for x in prefixes)} for prefixes in cls.GROUP_PREFIXES]
  parents=('llm2vec/','llm2vec/open_unlearning/','llm2vec/unlearn_eval/')
  cls.groups.append({p for p in cls.rows if any(p.startswith(x) and '/' not in p[len(x):] and re.search(r'\.(?:e|o)[0-9]+$',p[len(x):]) for x in parents)})
  cls.groups.append(cls.SWAPS)
  cls.targets=set().union(*cls.groups)
  ap=('llm2vec/grid_search_diagnosis/','llm2vec/output/','llm2vec/output_PMC/','llm2vec/output_local/','llm2vec/unlearn_eval/PMC/','llm2vec/unlearn_eval/Qwen/','llm2vec/unlearn_eval/eval_logs/')
  cls.batch_a={p for p in cls.rows if any(p.startswith(x) for x in ap)}; cls.batch_b=cls.targets-cls.batch_a
 @classmethod
 def summary(cls,paths):
  return len(paths),sum(cls.rows[p][3] for p in paths if cls.rows[p][1]=='blob'),hashlib.sha256(''.join(p+'\n' for p in sorted(paths)).encode()).hexdigest()
 @classmethod
 def index_rows(cls):
  rows={}
  for rec in subprocess.check_output(['git','ls-files','-s','-z'],cwd=ROOT).split(b'\0'):
   if rec:
    meta,path=rec.split(b'\t',1); mode,oid,stage=meta.split(); rows[path.decode()]=(mode.decode(),oid.decode(),stage.decode())
  return rows
 def test_starting_tree_and_target_census(self):
  self.assertEqual(git('rev-parse',self.START+'^{tree}').stdout.strip(),self.TREE)
  self.assertEqual(git('rev-parse',self.BASE+'^{tree}').stdout.strip(),self.BASE_TREE)
  self.assertEqual(len(self.rows),1951)
  for group,expected in zip(self.groups,self.EXPECTED): self.assertEqual(self.summary(group),expected)
  self.assertEqual(sum(map(len,self.groups)),1319); self.assertEqual(len(self.targets),1319)
  self.assertEqual(self.summary(self.targets),(1319,185694007,'0d1306fb8c9e2399ed3f3a062b3d120fd1123836e065abefc767b656e076f378'))
 def test_batches_are_exact_disjoint_partition(self):
  self.assertEqual(self.summary(self.batch_a),(679,94849293,'78432f869b09aafa43807e6b07539dbd2b66ef37a0c3fa2459b4e995c585acd8'))
  self.assertEqual(self.summary(self.batch_b),(640,90844714,'91f70a94bdc294508c8d87d16170d4881e133e713e258847fb842539591d2bdf'))
  self.assertFalse(self.batch_a&self.batch_b); self.assertEqual(self.batch_a|self.batch_b,self.targets)
 def test_phase3c1_exact_final_tracked_set(self):
  tracked=set(self.index_rows())
  self.assertEqual(tracked,set(self.rows)-self.targets); self.assertEqual(len(tracked),632)
  self.assertFalse(tracked&self.targets); self.assertFalse(tracked&self.batch_a); self.assertFalse(tracked&self.batch_b)
 def test_exact_base_diff(self):
  deleted=set(git('diff','--name-only','--diff-filter=D',self.BASE,'--').stdout.splitlines())
  modified=set(git('diff','--name-only','--diff-filter=M',self.BASE,'--').stdout.splitlines())
  self.assertEqual(deleted,self.batch_b); self.assertEqual(modified,self.B_MANAGEMENT)
 def test_non_target_blob_identities_are_preserved(self):
  current=self.index_rows()
  for path in set(self.rows)-self.targets-self.ALLOWED:
   self.assertEqual(current[path][:2],(self.rows[path][0],self.rows[path][2]),path)
 def test_targets_are_ignored_and_have_diff_unset(self):
  data='\0'.join(sorted(self.targets))+'\0'
  ignored=subprocess.run(['git','check-ignore','--no-index','-z','--stdin'],cwd=ROOT,input=data.encode(),capture_output=True,check=True).stdout
  self.assertEqual(set(ignored.rstrip(b'\0').decode().split('\0')),self.targets)
  attrs=subprocess.run(['git','check-attr','-z','--stdin','diff'],cwd=ROOT,input=data.encode(),capture_output=True,check=True).stdout.split(b'\0')
  self.assertEqual(len(attrs),3*len(self.targets)+1)
  self.assertTrue(all(attrs[i+2]==b'unset' for i in range(0,len(attrs)-1,3)))
 def test_batch_b_swaps_are_untracked_and_ignored(self):
  tracked=set(self.index_rows())
  self.assertTrue(self.SWAPS<=self.batch_b); self.assertFalse(self.SWAPS&tracked)
  for path in self.SWAPS: self.assertEqual(git('check-ignore','--no-index','--',path,check=False).returncode,0,path)
 def test_real_swap_peers_remain_protected(self):
  peers={'llm2vec/open_unlearning/configs/experiment/unlearn/PMC_rmu/default.yaml','llm2vec/train_configs/simcse/Contrast_Unlearn_Mistral_LMloss_only_diagnosis.json','llm2vec/train_configs/simcse/Contrast_Unlearn_Mistral_LMloss_zero_diagnosis.json'}
  current=self.index_rows(); self.assertFalse(peers&self.targets)
  for path in peers:
   self.assertEqual(current[path][:2],(self.rows[path][0],self.rows[path][2]),path)
   self.assertNotEqual(git('check-ignore','--no-index','--',path,check=False).returncode,0,path)
   self.assertEqual(git('check-attr','diff','--',path).stdout.strip().rsplit(': ',1)[-1],'unspecified',path)
 def test_root_archives_remain_tracked_and_unchanged(self):
  expected={'clinicia_provenance_bundle.tar.gz':'6e406e4e96b20413361fa67b2f0af2a67034d0211ba32a1207e8583df8d55fe7','clinicia_configs_mmlu_bundle.tar.gz':'a4b396370aabb6382a028a336202203508991cf910d5e0961d89d8bba75f0bf8'}
  current=self.index_rows()
  for path,digest in expected.items():
   self.assertEqual(current[path][:2],(self.rows[path][0],self.rows[path][2]),path)
   self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest,path)
 def test_protected_non_targets_are_unchanged_and_uncovered(self):
  protected={p for p in self.rows if p not in self.targets and (p.startswith(('configs/historical/','results/paper/','docs/')) or p in {'clinicia_provenance_bundle.tar.gz','clinicia_configs_mmlu_bundle.tar.gz','llm2vec/open_unlearning/configs/experiment/unlearn/PMC_rmu/default.yaml','llm2vec/train_configs/simcse/Contrast_Unlearn_Mistral_LMloss_only_diagnosis.json','llm2vec/train_configs/simcse/Contrast_Unlearn_Mistral_LMloss_zero_diagnosis.json'})}
  current=self.index_rows(); self.assertTrue(protected); self.assertTrue(protected<=set(current))
  self.assertFalse(set(git('diff','--name-only',self.START,'--',*sorted(protected)).stdout.splitlines()))
  for path in sorted(protected):
   self.assertEqual(current[path][:2],(self.rows[path][0],self.rows[path][2]),path)
   self.assertNotEqual(git('check-ignore','--no-index','--',path,check=False).returncode,0,path)
   self.assertEqual(git('check-attr','diff','--',path).stdout.strip().rsplit(': ',1)[-1],'unspecified',path)
 def test_a0_control_files_are_unchanged_from_base(self):
  self.assertEqual(git('diff','--quiet',self.BASE,'--','.gitattributes','.gitignore',check=False).returncode,0)
 def test_cleanup_plan_records_both_batches_and_full_phase_complete(self):
  plan=json.loads((ROOT/'results/repository_cleanup_plan.json').read_text())
  a=[x for x in plan['cleanup_batches'] if x.get('batch_id')=='phase3c1-batch-a']; b=[x for x in plan['cleanup_batches'] if x.get('batch_id')=='phase3c1-batch-b']; self.assertEqual((len(a),len(b)),(1,1))
  a_expected={'status':'complete','task_base_commit':self.A1_BASE,'immutable_census_source_commit':self.START,'index_only_untracking':True,'removed_from_index_count':679,'target_ordinary_blob_bytes':94849293,'sorted_target_path_list_sha256':'78432f869b09aafa43807e6b07539dbd2b66ef37a0c3fa2459b4e995c585acd8','tracked_path_count_before':1951,'tracked_path_count_after':1272,'batch_b_remaining_tracked_count':640,'working_tree_copies_remained_present_and_ignored':True,'scientific_or_provenance_files_changed':False,'validation_status':'passed','batch_b_status':'pending','full_phase3c1_complete':False}
  b_expected={'status':'complete','task_base_commit':self.BASE,'immutable_census_source_commit':self.START,'index_only_untracking':True,'removed_from_index_count':640,'target_ordinary_blob_bytes':90844714,'sorted_target_path_list_sha256':'91f70a94bdc294508c8d87d16170d4881e133e713e258847fb842539591d2bdf','tracked_path_count_before':1272,'tracked_path_count_after':632,'batch_a_previously_removed_count':679,'batch_a_status':'complete','remaining_phase3c1_tracked_count':0,'full_phase3c1_complete':True,'full_phase3c1_removed_from_index_count':1319,'full_phase3c1_ordinary_blob_bytes':185694007,'full_phase3c1_sorted_path_list_sha256':'0d1306fb8c9e2399ed3f3a062b3d120fd1123836e065abefc767b656e076f378','working_tree_copies_remained_present_and_ignored':True,'scientific_or_provenance_files_changed':False,'validation_status':'passed'}
  for record,expected in ((a[0],a_expected),(b[0],b_expected)):
   for key,value in expected.items(): self.assertEqual(record.get(key),value,key)
if __name__=='__main__': unittest.main()
