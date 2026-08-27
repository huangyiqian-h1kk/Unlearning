import importlib.util, pathlib, tempfile, unittest, subprocess

P=pathlib.Path(__file__).parents[2]/"scripts/results/build_tables.py"
S=importlib.util.spec_from_file_location("build_tables",P); m=importlib.util.module_from_spec(S); S.loader.exec_module(m)

class FormulaTests(unittest.TestCase):
 def test_regime_a_generation(self):
  self.assertEqual(m.regime_a_generation_retain(.4,.5),80); self.assertAlmostEqual(m.regime_a_generation_forget(.1,.5),80)
 def test_regime_a_mcq_and_clipping(self):
  self.assertAlmostEqual(m.regime_a_mcq_retain(.5,.75),50); self.assertAlmostEqual(m.regime_a_mcq_forget(.5,.75),50)
  self.assertEqual(m.regime_a_generation_retain(2,1),100); self.assertEqual(m.regime_a_generation_forget(2,1),0)
 def test_regime_b_generation(self): self.assertEqual(m.regime_b_generation(.6,.4,.5),(120,80,-40))
 def test_regime_b_mcq_not_clipped(self): self.assertEqual(m.regime_b_mcq(.75,.0,.5),(200,-100,-300))
 def test_exact_binomial_masking(self):
  self.assertTrue(m.baseline_significant(50,100)); self.assertFalse(m.baseline_significant(25,100))
  self.assertEqual(m.mask_if_not_significant(42,25,100),'MASKED'); self.assertEqual(m.mask_if_not_significant(42,50,100),42)
 def test_average_masks_and_mmlu(self): self.assertEqual(m.row_average({'a':20,'b':'MASKED','MMLU':.6}),20)
 def test_probe_order(self):
  m.assert_probe_order(['ATT','IDeq','ID'])
  with self.assertRaises(ValueError): m.assert_probe_order(['IDeq','ATT','ID'])
 def test_status_rendering(self):
  self.assertFalse(m.renderable({'status_flags':['missing']})); self.assertFalse(m.renderable({'status_flags':['unresolved']})); self.assertTrue(m.renderable({'status_flags':['non_comparable']}))
 def test_deterministic_output(self):
  root=P.parents[2]
  with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
   cmd=['python',str(P),'--manifest',str(root/'results/paper/manifest.json')]
   subprocess.run(cmd+['--output-dir',a],check=True); subprocess.run(cmd+['--output-dir',b],check=True)
   one={x.relative_to(a):x.read_bytes() for x in pathlib.Path(a).rglob('*') if x.is_file()}
   two={x.relative_to(b):x.read_bytes() for x in pathlib.Path(b).rglob('*') if x.is_file()}
   self.assertEqual(one,two)

if __name__=='__main__': unittest.main()
