import importlib.util, json, pathlib, subprocess, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('validator',ROOT/'scripts/configs/validate_historical_experiments.py'); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
class HistoricalExperimentsTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.records,cls.content_checks=v.validate(ROOT/'configs/historical/paper/index.json',ROOT/'results/paper/manifest.json'); cls.by={r['experiment_id']:r for r in cls.records}
 def test_coverage_schema_manifest_and_order(self):
  self.assertEqual(25,len(self.records)); self.assertEqual(sorted(self.by),list(self.by))
 def test_deterministic_serialization(self):
  for r in self.records:
   p=ROOT/'configs/historical/paper'/(r['experiment_id']+'.json'); self.assertEqual(p.read_text(),json.dumps(r,sort_keys=True,indent=2)+'\n')
 def test_backend_mapping(self):
  for r in self.records:
   expected='conrep' if r['method']=='ConRep' else ('open_unlearning' if r['method'] in ('GradDiff','NPO','RMU') else r['backend']); self.assertEqual(expected,r['backend'])
   if r['method']=='Baseline' and r['regime']=='A': self.assertEqual(('none','evaluation_only'),(r['backend'],r['run_kind']))
 def test_status_separation(self):
  self.assertTrue(all(r['reproduction_status'] in v.REPRO for r in self.records))
  self.assertTrue(all(isinstance(r['status_flags'],list) for r in self.records))
 def test_conrep_invariants(self):
  con=[r for r in self.records if r['method']=='ConRep']; self.assertEqual(5,len(con)); self.assertTrue(all(r['resolved_hyperparameters']['lm_weight']==0 for r in con))
 def test_pmc_lineage(self):
  self.assertIn('non_comparable',self.by['b-pmc-mistral-graddiff']['status_flags']); self.assertIn('csv',self.by['b-pmc-mistral-graddiff']['starting_checkpoint_id'].lower())
  for method in ('npo','rmu','conrep'): self.assertIn('universal',self.by[f'b-pmc-mistral-{method}']['starting_checkpoint_id'].lower())
 def test_unresolved_deaths_rmu_and_pmc_mmlu(self):
  for i in ('a-deaths-llama2-rmu','a-deaths-mistral-rmu'): self.assertIn('unresolved',self.by[i]['status_flags'])
  self.assertEqual('missing',self.by['b-pmc-mistral-conrep']['mmlu_evidence']['settings_status'])
 def test_rejects_private_and_temporary_paths(self):
  for text in ('/home/private/data','/tmp/extract-1','/scratch/cluster'): self.assertRegex(text,v.PRIVATE)
 def test_sparse_file_uses_tree_identity_without_blob_read(self):
  calls=[]
  def runner(command,**kwargs):
   calls.append((command,kwargs)); return subprocess.CompletedProcess(command,0,'100644 blob abc123\tlegacy/file.json\n','')
  source={'path':'legacy/file.json','starting_git_commit':'deadbeef','git_blob_object_id':'abc123','sha256':'0'*64}
  with tempfile.TemporaryDirectory() as root:
   self.assertEqual('not_materialized',v.validate_legacy_source(root,source,runner))
  self.assertEqual(['git','ls-tree','--full-tree','deadbeef','--','legacy/file.json'],calls[0][0])
  self.assertEqual('1',calls[0][1]['env']['GIT_NO_LAZY_FETCH'])
 def test_mismatched_blob_id_fails(self):
  def runner(command,**kwargs): return subprocess.CompletedProcess(command,0,'100644 blob other\tlegacy/file.json\n','')
  source={'path':'legacy/file.json','starting_git_commit':'deadbeef','git_blob_object_id':'expected','sha256':'0'*64}
  with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(ValueError,'blob mismatch'):
   v.validate_legacy_source(root,source,runner)
 def test_invalid_inventory_sha_fails_before_git(self):
  source={'path':'legacy/file.json','starting_git_commit':'deadbeef','git_blob_object_id':'abc','sha256':'invalid'}
  with self.assertRaisesRegex(ValueError,'invalid inventory SHA-256'):
   v.validate_legacy_source(ROOT,source,lambda *args,**kwargs: self.fail('Git must not run'))
 def test_validator_has_no_blob_content_git_commands(self):
  source=(ROOT/'scripts/configs/validate_historical_experiments.py').read_text()
  forbidden=('git' + ' show','git' + ' cat-file')
  self.assertTrue(all(command not in source for command in forbidden))
if __name__=='__main__': unittest.main()
