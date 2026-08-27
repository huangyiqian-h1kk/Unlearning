import csv
import importlib.util
import pathlib
import subprocess
import tempfile
import unittest

ROOT=pathlib.Path(__file__).parents[2]
SPEC=importlib.util.spec_from_file_location("build",ROOT/"scripts/results/build_tables.py")
build=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(build)


class FormulaAndBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(); cls.out=pathlib.Path(cls.temp.name)
        subprocess.run(["python",str(ROOT/"scripts/results/build_tables.py"),"--manifest",str(ROOT/"results/paper/manifest.json"),"--output-dir",str(cls.out)],cwd=ROOT,check=True)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def rows(self,name):
        with open(self.out/name/"table.csv",encoding="utf-8") as f:return list(csv.DictReader(f))

    def test_regime_a_formulas_and_clipping(self):
        self.assertEqual(build.regime_a_generation_retain(.4,.5),80)
        self.assertAlmostEqual(build.regime_a_generation_forget(.1,.5),80)
        self.assertEqual(build.regime_a_mcq_retain(.5,.75),50)
        self.assertEqual(build.regime_a_mcq_forget(.5,.75),50)
        self.assertEqual(build.regime_a_generation_retain(2,1),100)
        self.assertEqual(build.regime_a_generation_forget(2,1),0)

    def test_regime_b_formulas_are_not_clipped(self):
        self.assertEqual(build.regime_b_generation(.6,.4,.5),(120,80,-40))
        self.assertEqual(build.regime_b_mcq(.75,0,.5),(200,-100,-300))

    def test_model_specific_regime_a_masking(self):
        rows=self.rows("regime_a_deaths")
        llama=next(r for r in rows if r["Experiment"]=="a-deaths-llama2-conrep")
        mistral=next(r for r in rows if r["Experiment"]=="a-deaths-mistral-conrep")
        self.assertNotEqual(llama["F-ID"],"–"); self.assertEqual(mistral["F-ID"],"–")
        self.assertEqual(llama["F-IDeq"],"–")

    def test_pmc_att_ideq_calculated_and_id_masked(self):
        for row in self.rows("regime_b_pmc"):
            self.assertNotIn(row["ATT-R"],{"N/A","–"}); self.assertNotIn(row["IDeq-R"],{"N/A","–"})
            self.assertIn(row["ID-R"],{"–","–†"}); self.assertIn(row["ID-Delta"],{"–","–†"})

    def test_symbols_remain_distinct(self):
        pmc=self.rows("regime_b_pmc")
        baseline=next(r for r in pmc if r["Method"]=="Baseline")
        grad=next(r for r in pmc if r["Method"]=="GradDiff‡")
        self.assertEqual(baseline["ID-R"],"–†"); self.assertEqual(grad["ID-R"],"–")
        self.assertEqual(grad["MMLU"],"0.23"); self.assertEqual(grad["Method"],"GradDiff‡")
        conrep=next(r for r in pmc if r["Method"]=="ConRep"); self.assertEqual(conrep["MMLU"],"N/A")

    def test_masked_missing_mmlu_and_delta_excluded_from_average(self):
        values={"a":20,"b":build.MASKED,"MMLU":.9,"ATT-Delta":500,"ATT-R":40}
        self.assertEqual(build.row_average(values,"B"),30)

    def test_corrected_pmc_averages(self):
        expected={"Baseline":"96.73","GradDiff‡":"81.36","NPO":"83.13","RMU":"89.69","ConRep":"101.42"}
        self.assertEqual({r["Method"]:r["Average"] for r in self.rows("regime_b_pmc")},expected)

    def test_status_enforced_by_real_builder(self):
        deaths=self.rows("regime_a_deaths")
        for key in ("a-deaths-llama2-rmu","a-deaths-mistral-rmu"):
            row=next(r for r in deaths if r["Experiment"]==key)
            self.assertTrue(all(value=="N/A" for col,value in row.items() if col not in {"Experiment","Model","Method","Status"}))

    def test_probe_order_used_by_real_build(self):
        headers=list(self.rows("regime_b_pmc")[0])
        self.assertLess(headers.index("ATT-R"),headers.index("IDeq-R")); self.assertLess(headers.index("IDeq-R"),headers.index("ID-R"))
        with self.assertRaises(ValueError): build.assert_probe_order(["IDeq","ATT","ID"])

    def test_swapped_raw_probe_mapping_fails(self):
        original=build.PROBE_ORDER
        try:
            build.PROBE_ORDER=("IDeq","ATT","ID")
            with self.assertRaises(ValueError): build.raw_columns("diagnosis")
        finally: build.PROBE_ORDER=original

    def test_cell_status_blocks_numeric_value(self):
        self.assertFalse(build.cell_allowed({"cell_status":{"ATT-R":["missing"]}},"ATT-R"))
        self.assertTrue(build.cell_allowed({"cell_status":{}},"ATT-R"))

    def test_latex_uses_valid_footnotes(self):
        text=(self.out/"regime_b_pmc/table.tex").read_text()
        self.assertIn(r"\textemdash{}\textsuperscript{\dagger}",text)
        self.assertNotIn("†",text)

    def test_build_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as other:
            subprocess.run(["python",str(ROOT/"scripts/results/build_tables.py"),"--manifest",str(ROOT/"results/paper/manifest.json"),"--output-dir",other],cwd=ROOT,check=True)
            one={p.relative_to(self.out):p.read_bytes() for p in self.out.rglob("*") if p.is_file()}
            two={p.relative_to(other):p.read_bytes() for p in pathlib.Path(other).rglob("*") if p.is_file()}
            self.assertEqual(one,two)


if __name__ == "__main__": unittest.main()
