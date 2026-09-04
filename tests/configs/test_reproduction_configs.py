import json
import math
import pathlib
import re
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class ReproductionConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = load("configs/reproduction/index.json")
        cls.registries = {
            kind: load(path)
            for kind, path in cls.index["component_registries"].items()
        }
        cls.catalog = load("data/clinicia/catalog.json")

    def test_offline_validator_passes(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/configs/validate_reproduction_configs.py",
                "--index",
                "configs/reproduction/index.json",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            "validated 4 component registries, 1 reproduction candidates, and 1 compact sweeps; runnable=0",
        )
        self.assertEqual("", result.stderr)

    def test_component_registries_have_owned_interfaces(self):
        self.assertEqual(set(self.registries), {"datasets", "methods", "models", "protocols"})
        for kind, record in self.registries.items():
            self.assertEqual("1.0", record["schema_version"])
            self.assertEqual(kind, record["component_kind"])
            self.assertTrue(record[kind])

    def test_dataset_registry_uses_semantic_current_paths(self):
        historical_to_current = {
            row["historical_path"]: row["path"] for row in self.catalog["datasets"]
        }
        paper = load("results/paper/mcq_dataset_inventory.json")
        expected = {historical_to_current[row["path"]] for row in paper["datasets"]}
        dataset_record = self.registries["datasets"]
        self.assertEqual(expected, set(dataset_record["datasets"].values()))
        self.assertEqual(11, len(dataset_record["datasets"]))
        self.assertTrue(all(path.startswith("data/clinicia/") for path in expected))
        self.assertTrue(all((ROOT / path).is_file() for path in expected))

    def test_catalog_preserves_historical_identity(self):
        rows = self.catalog["datasets"]
        self.assertEqual(28, len(rows))
        self.assertEqual(28, len({row["path"] for row in rows}))
        self.assertEqual(28, len({row["historical_path"] for row in rows}))
        self.assertTrue(all(len(row["git_blob_oid"]) == 40 for row in rows))

    def test_method_ownership_is_explicit(self):
        methods = self.registries["methods"]["methods"]
        self.assertEqual({"baseline", "conrep", "graddiff", "npo", "rmu"}, set(methods))
        self.assertEqual("project", methods["conrep"]["owner"])
        for name in ("graddiff", "npo", "rmu"):
            self.assertEqual("open_unlearning_derived", methods[name]["owner"])
            self.assertEqual("third_party/open-unlearning/src/train.py", methods[name]["entrypoint"])

    def test_candidate_is_portable_but_explicitly_blocked(self):
        self.assertEqual(1, len(self.index["candidates"]))
        candidate = load(self.index["candidates"][0])
        self.assertFalse(candidate["runnable"])
        self.assertFalse(candidate["historical_equivalence_claimed"])
        self.assertFalse(candidate["may_write_archived_paper_results"])
        self.assertTrue(candidate["gates"])
        self.assertTrue(all(value is False for value in candidate["gates"].values()))
        env_pattern = re.compile(r"^\$\{[A-Z][A-Z0-9_]*\}(?:/[A-Za-z0-9_.-]+)*$")
        self.assertTrue(all(env_pattern.fullmatch(value) for value in candidate["environment"].values()))

    def test_protocol_keeps_archived_results_read_only(self):
        protocols = self.registries["protocols"]["protocols"]
        self.assertEqual("results/paper", protocols["historical_v1"]["result_namespace"])
        self.assertEqual("results/validated_v2", protocols["validated_v2"]["result_namespace"])
        self.assertTrue(all(not value["may_modify_archived_paper_results"] for value in protocols.values()))

    def test_sweep_is_compact_and_review_only(self):
        self.assertEqual(1, len(self.index["sweeps"]))
        sweep = load(self.index["sweeps"][0])
        self.assertEqual("review_only_not_runnable", sweep["status"])
        self.assertFalse(sweep["expanded_products_tracked"])
        self.assertFalse(sweep["historical_equivalence_claimed"])
        self.assertEqual(
            math.prod(len(values) for values in sweep["axes"].values()),
            sweep["expected_combinations"],
        )
        self.assertEqual(24, sweep["expected_combinations"])


if __name__ == "__main__":
    unittest.main()
