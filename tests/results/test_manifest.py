import copy
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location("extract", ROOT / "scripts/results/extract_archived_metrics.py")
extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract)


class EvidenceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "results/paper/manifest.json").read_text())
        cls.inventory = json.loads((ROOT / "results/paper/mcq_dataset_inventory.json").read_text())["datasets"]
        cls.snapshots = {p.stem: json.loads(p.read_text()) for p in (ROOT / "results/paper/raw_metrics").glob("*.json") if p.name != "related_evidence.json"}

    def test_manifest_shape_and_archive_hashes(self):
        self.assertEqual(self.manifest["probe_order"], ["ATT", "IDeq", "ID"])
        self.assertEqual(len(self.manifest["experiments"]), 25)
        for archive in self.manifest["archives"]:
            self.assertEqual(extract.sha256_file(ROOT / archive["path"]), archive["sha256"])

    def test_regime_a_dataset_counts_and_hashes(self):
        expected = {"mcqs_deaths_att.jsonl": 228, "mcqs_deaths_id_eq.jsonl": 114,
                    "mcqs_deaths_id_sim.jsonl": 114, "mcqs_diagnosis_att.jsonl": 51,
                    "mcqs_diagnosis_id.jsonl": 52}
        for item in self.inventory:
            if item["path"].startswith("llm2vec/unlearn_eval"):
                count, method = extract.count_jsonl_or_pointer(ROOT / item["path"], item)
                self.assertEqual(count, expected[pathlib.Path(item["path"]).name])
                self.assertEqual(method, "locally_materialized_jsonl")

    def test_pmc_pointer_counts_and_oids(self):
        for item in self.inventory:
            if item["split"] in {"retain", "forget"}:
                count, method = extract.count_jsonl_or_pointer(ROOT / item["path"], item)
                self.assertEqual(count, item["record_count"])
                self.assertEqual(item["lfs_content_oid"], item["expected_content_sha256"])
                self.assertEqual(method, "researcher_verified_external_lfs_object")

    def test_lfs_pointer_without_verified_inventory_is_rejected(self):
        item = next(x for x in self.inventory if x["split"] == "retain")
        bad = copy.deepcopy(item); bad["verification_method"] = "unverified"
        with self.assertRaises(ValueError): extract.count_jsonl_or_pointer(ROOT / item["path"], bad)

    def test_success_integrality(self):
        self.assertEqual(extract.derive_success(90 / 228, 228)["success_count"], 90)
        with self.assertRaises(ValueError): extract.derive_success(.333, 100)

    def test_exact_significance_values(self):
        cases = [(90,228,1.0659724714050165e-6,True),(33,114,.1921917142569755,False),
                 (39,114,.01751824612381921,True),(26,51,5.914449162540314e-5,True),
                 (15,52,.3083789711622175,False),(28,51,4.9228078819310914e-6,True),
                 (29,114,.4927715547157413,False),(18,52,.07806484144859673,False),
                 (54,150,.0017832022913445975,True),(48,150,.032081960505484605,True),
                 (41,146,.22032852904022862,False)]
        for k,n,p,sig in cases:
            actual = extract.binomial_greater_pvalue(k,n)
            self.assertAlmostEqual(actual,p,places=14); self.assertEqual(actual < .05,sig)

    def test_repository_hash_mismatch_and_missing_fail(self):
        source = next(s for e in self.manifest["experiments"] for s in e["sources"] if s["source_type"] == "repository_file")
        bad = copy.deepcopy(source); bad["file_sha256"] = "0" * 64
        with self.assertRaises(ValueError): extract.repository_file_bytes(bad)
        bad = copy.deepcopy(source); bad["path"] = "does/not/exist"
        with self.assertRaises(FileNotFoundError): extract.repository_file_bytes(bad)

    def test_archive_member_hash_mismatch_fails(self):
        source = next(s for e in self.manifest["experiments"] for s in e["sources"] if s["source_type"] == "archive_member")
        bad = copy.deepcopy(source); bad["member_sha256"] = "0" * 64
        with self.assertRaises(ValueError): extract.archive_member_bytes(bad)

    def test_config_and_hydra_are_parsed(self):
        conrep = self.snapshots["a-diagnosis-llama2-conrep"]["normalized_config"]
        self.assertEqual(conrep["model_name_or_path"], "meta-llama/Llama-2-7b-chat-hf")
        self.assertEqual((conrep["lm_weight"],conrep["forget_weight"],conrep["num_train_epochs"]),(0, .45, 4))
        rmu = self.snapshots["a-deaths-llama2-rmu"]["normalized_config"]
        self.assertEqual((rmu["model_name_or_path"],rmu["forget_target"],rmu["num_train_epochs"]),
                         ("mistralai/Mistral-7B-Instruct-v0.2","diagnosis",10))

    def test_mmlu_blocks_and_sources(self):
        expected = {"a-diagnosis-llama2-baseline":.4638,"a-diagnosis-mistral-baseline":.5902,
                    "b-pmc-mistral-baseline":.2690,"b-pmc-mistral-graddiff":.2297,
                    "b-pmc-mistral-npo":.2307,"b-pmc-mistral-rmu":.2722}
        experiments = {x["id"]:x for x in self.manifest["experiments"]}
        for key,value in expected.items():
            self.assertEqual(self.snapshots[key]["mmlu"]["accuracy"],value)
            declared = next(s for s in experiments[key]["sources"] if s["role"] == "mmlu_log")
            self.assertEqual(experiments[key]["mmlu_source"]["member"],declared["member"])
        related = json.loads((ROOT/"results/paper/raw_metrics/related_evidence.json").read_text())[0]
        self.assertEqual(related["mmlu"]["accuracy"], .2659)

    def test_rmu_deaths_intended_and_resolved_not_conflated(self):
        experiments = {x["id"]:x for x in self.manifest["experiments"]}
        for key in ("a-deaths-llama2-rmu","a-deaths-mistral-rmu"):
            item=experiments[key]; self.assertIsNone(item["resolved_model_id"])
            self.assertEqual(item["canonical_cell_assignment"],"unresolved_not_assigned")
            self.assertNotEqual(item["intended_model_id"],item["source_resolved_identity"]["model_id"])

    def test_terminal_state_is_present_and_deterministic(self):
        for key,snapshot in self.snapshots.items():
            if "baseline" not in key or key == "b-pmc-mistral-baseline":
                if snapshot["trainer_state"]:
                    self.assertIn("global_step",snapshot["trainer_state"])
                    roles=[s["role"] for s in snapshot["sources"]]
                    self.assertEqual(roles.count("trainer_state"),1)

    def test_extraction_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            cmd=["python",str(ROOT/"scripts/results/extract_archived_metrics.py"),"--manifest",str(ROOT/"results/paper/manifest.json")]
            subprocess.run(cmd+["--output-dir",a],cwd=ROOT,check=True)
            subprocess.run(cmd+["--output-dir",b],cwd=ROOT,check=True)
            one={p.relative_to(a):p.read_bytes() for p in pathlib.Path(a).glob("*.json")}
            two={p.relative_to(b):p.read_bytes() for p in pathlib.Path(b).glob("*.json")}
            self.assertEqual(one,two)

    def test_builder_has_no_archive_dependency(self):
        text=(ROOT/"scripts/results/build_tables.py").read_text()
        self.assertNotIn("tarfile",text); self.assertNotIn("clinicia_provenance_bundle",text)


if __name__ == "__main__": unittest.main()
