import hashlib
import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "historical_validator",
    ROOT / "scripts/configs/validate_historical_experiments.py",
)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class HistoricalExperimentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = ROOT / "configs/paper/historical/index.json"
        cls.records, cls.content_checks = validator.validate(
            cls.index,
            ROOT / "results/paper/manifest.json",
        )
        cls.by_id = {row["experiment_id"]: row for row in cls.records}
        cls.sources = json.loads(
            (ROOT / "results/repository_selected_legacy_sources.json").read_text()
        )["sources"]

    def test_index_has_exact_paper_experiment_set(self):
        self.assertEqual(25, len(self.records))
        self.assertEqual(sorted(self.by_id), list(self.by_id))

    def test_records_are_deterministically_serialized(self):
        for record in self.records:
            path = self.index.parent / f"{record['experiment_id']}.json"
            self.assertEqual(
                path.read_text(),
                json.dumps(record, sort_keys=True, indent=2) + "\n",
            )

    def test_backend_mapping(self):
        for record in self.records:
            if record["method"] == "ConRep":
                self.assertEqual("conrep", record["backend"])
            elif record["method"] in {"GradDiff", "NPO", "RMU"}:
                self.assertEqual("open_unlearning", record["backend"])

    def test_status_vocabularies(self):
        for record in self.records:
            self.assertIn(record["reproduction_status"], validator.REPRODUCTION_STATUS)
            self.assertLessEqual(set(record["status_flags"]), validator.STATUS)

    def test_selected_conrep_invariants(self):
        selected = [row for row in self.records if row["method"] == "ConRep"]
        self.assertEqual(5, len(selected))
        self.assertTrue(all(row["resolved_hyperparameters"]["lm_weight"] == 0 for row in selected))
        self.assertTrue(all(row["resolved_hyperparameters"]["num_train_epochs"] in {3, 4, 5} for row in selected))

    def test_pmc_lineage_and_missing_mmlu_are_explicit(self):
        self.assertIn("non_comparable", self.by_id["b-pmc-mistral-graddiff"]["status_flags"])
        self.assertEqual("missing", self.by_id["b-pmc-mistral-conrep"]["mmlu_evidence"]["settings_status"])

    def test_unresolved_deaths_rmu_is_explicit(self):
        for experiment_id in ("a-deaths-llama2-rmu", "a-deaths-mistral-rmu"):
            self.assertIn("unresolved", self.by_id[experiment_id]["status_flags"])

    def test_all_selected_historical_files_are_materialized(self):
        self.assertEqual(["verified"] * 16, self.content_checks)
        self.assertEqual(16, len(self.sources))

    def test_materialized_files_match_recorded_sha256(self):
        for source in self.sources:
            content = (ROOT / source["materialized_path"]).read_bytes()
            self.assertEqual(source["sha256"], hashlib.sha256(content).hexdigest())

    def test_materialized_paths_are_grouped_by_selected_run(self):
        experiment_ids = {row["experiment_id"] for row in self.sources}
        self.assertEqual(5, len(experiment_ids))
        for source in self.sources:
            self.assertTrue(source["materialized_path"].startswith(f"experiments/paper_runs/{source['experiment_id']}/historical/"))

    def test_rejects_private_and_temporary_paths_in_normalized_records(self):
        for text in ("/home/private/data", "/tmp/extract-1", "/scratch/cluster"):
            self.assertRegex(text, validator.PRIVATE_PATH)

    def test_sparse_file_uses_tree_identity_without_blob_read(self):
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, "100644 blob abc123\tlegacy/file.json\n", "")

        source = {
            "path": "legacy/file.json",
            "materialized_path": "missing/file.json",
            "starting_git_commit": "deadbeef",
            "git_blob_object_id": "abc123",
            "sha256": "0" * 64,
        }
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual("not_materialized", validator.validate_legacy_source(pathlib.Path(root), source, runner))
        self.assertEqual(["git", "ls-tree", "--full-tree", "deadbeef", "--", "legacy/file.json"], calls[0][0])
        self.assertEqual("1", calls[0][1]["env"]["GIT_NO_LAZY_FETCH"])

    def test_mismatched_blob_id_fails(self):
        def runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "100644 blob other\tlegacy/file.json\n", "")

        source = {
            "path": "legacy/file.json",
            "starting_git_commit": "deadbeef",
            "git_blob_object_id": "expected",
            "sha256": "0" * 64,
        }
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(ValueError, "blob mismatch"):
            validator.validate_legacy_source(pathlib.Path(root), source, runner)

    def test_validator_does_not_read_historical_blob_contents(self):
        source = (ROOT / "scripts/configs/validate_historical_experiments.py").read_text()
        self.assertNotIn("git" + " show", source)
        self.assertNotIn("git" + " cat-file", source)


if __name__ == "__main__":
    unittest.main()
