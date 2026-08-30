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
if __name__=='__main__': unittest.main()
