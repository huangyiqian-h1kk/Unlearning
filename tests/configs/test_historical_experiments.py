import importlib.util, json, pathlib, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('validator',ROOT/'scripts/configs/validate_historical_experiments.py'); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
class HistoricalExperimentsTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.records=v.validate(ROOT/'configs/historical/paper/index.json',ROOT/'results/paper/manifest.json'); cls.by={r['experiment_id']:r for r in cls.records}
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
if __name__=='__main__': unittest.main()
