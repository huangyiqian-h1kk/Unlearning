import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_COMMIT = "623e305655a10a87685e49d83404fe7cd5f2ed81"
SPEC = importlib.util.spec_from_file_location(
    "source_inventory_validator",
    ROOT / "scripts/repository/validate_source_inventory.py",
)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def git(*args):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def index_entries():
    result = {}
    for row in git("ls-files", "-s", "-z").split(b"\0"):
        if not row:
            continue
        metadata, path = row.split(b"\t", 1)
        mode, blob, stage = metadata.split()
        result[path.decode()] = (mode.decode(), blob.decode(), stage.decode())
    return result


class PaperFacingSourceInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = index_entries()
        cls.base = validator.tree_entries(BASE_COMMIT)
        cls.catalog = json.loads((ROOT / "data/clinicia/catalog.json").read_text())
        cls.selected = json.loads((ROOT / "results/repository_selected_legacy_sources.json").read_text())
        cls.runs = json.loads((ROOT / "experiments/paper_runs/index.json").read_text())

    def test_validator_function_accepts_complete_layout(self):
        self.assertEqual(
            (718, 28, 16, 5),
            validator.validate(
                ROOT / "data/clinicia/catalog.json",
                ROOT / "results/repository_selected_legacy_sources.json",
            ),
        )

    def test_validator_cli_is_offline_and_deterministic(self):
        result = subprocess.run(
            [sys.executable, "scripts/repository/validate_source_inventory.py"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        self.assertEqual(
            "validated paper-facing layout: 718 tracked paths, 28 ClinicIA datasets, "
            "16 selected historical files across 5 ConRep runs\n",
            result.stdout,
        )
        self.assertEqual("", result.stderr)

    def test_current_roots_make_ownership_obvious(self):
        required = validator.REQUIRED_PREFIXES
        for prefix in required:
            self.assertTrue(any(path.startswith(prefix) for path in self.current), prefix)
        self.assertFalse(any(path.startswith("llm2vec/") for path in self.current))

    def test_clinicia_catalog_is_an_exact_move_map(self):
        rows = self.catalog["datasets"]
        self.assertEqual(28, len(rows))
        for row in rows:
            self.assertEqual(row["git_blob_oid"], self.base[row["historical_path"]])
            self.assertEqual(row["git_blob_oid"], self.current[row["path"]][1])

    def test_clinicia_catalog_covers_both_paper_regimes(self):
        rows = self.catalog["datasets"]
        self.assertEqual({"A", "B"}, {row["regime"] for row in rows})
        self.assertEqual({"diagnosis", "deaths", "pmc", "shared"}, {row["target"] for row in rows})
        self.assertEqual({"probe", "training"}, {row["role"] for row in rows})

    def test_selected_inventory_has_exact_public_materializations(self):
        rows = self.selected["sources"]
        self.assertEqual(16, self.selected["materialized_file_count"])
        self.assertEqual(5, self.selected["selected_experiment_count"])
        for row in rows:
            path = ROOT / row["materialized_path"]
            self.assertTrue(path.is_file())
            self.assertIn(row["materialized_path"], self.current)
            self.assertEqual(row["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_selected_inventory_remains_git_history_anchored(self):
        trees = {}
        for row in self.selected["sources"]:
            trees.setdefault(row["starting_git_commit"], validator.tree_entries(row["starting_git_commit"]))
            self.assertEqual(
                row["git_blob_object_id"],
                trees[row["starting_git_commit"]][row["path"]],
            )

    def test_paper_run_index_covers_all_25_table_rows(self):
        experiments = self.runs["experiments"]
        self.assertEqual(25, len(experiments))
        self.assertEqual(set(range(1, 7)), {table for row in experiments for table in row["paper_tables"]})
        for row in experiments:
            self.assertTrue((ROOT / row["record_path"]).is_file())

    def test_five_selected_capsules_are_human_navigable(self):
        selected = self.runs["selected_conrep_runs"]
        self.assertEqual(5, len(selected))
        self.assertTrue(all(row["objective"] == "symmetric" for row in selected))
        for row in selected:
            self.assertTrue(row["historical_files"])
            self.assertTrue(all((ROOT / path).is_file() for path in row["historical_files"].values()))

    def test_historical_records_moved_without_content_change(self):
        old_prefix = "configs/historical/paper/"
        new_prefix = "configs/paper/historical/"
        old = {path: blob for path, blob in self.base.items() if path.startswith(old_prefix)}
        self.assertEqual(26, len(old))
        for path, blob in old.items():
            current = path.replace(old_prefix, new_prefix, 1)
            self.assertEqual(blob, self.current[current][1], current)

    def test_results_inventory_remains_a_historical_audit_record(self):
        inventory = json.loads((ROOT / "docs/repository_source_inventory.json").read_text())
        self.assertEqual("phase3d0_source_migration_baseline", inventory["record_kind"])
        self.assertFalse(inventory["phase_scope"]["source_movement_performed"])
        self.assertTrue((ROOT / "docs/source_ownership_audit.md").is_file())


if __name__ == "__main__":
    unittest.main()
