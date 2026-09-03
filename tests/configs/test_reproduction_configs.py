import json
import math
import pathlib
import re
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "configs" / "reproduction" / "index.json"


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
        self.assertEqual(result.stderr, "")

    def test_component_registries_match_owned_interfaces(self):
        self.assertEqual(set(self.registries), {"datasets", "methods", "models", "protocols"})
        for kind, record in self.registries.items():
            self.assertEqual(record["schema_version"], "1.0")
            self.assertEqual(record["component_kind"], kind)
            self.assertTrue(record[kind])

        paper = load("results/paper/mcq_dataset_inventory.json")
        paper_paths = {row["path"] for row in paper["datasets"]}
        datasets = self.registries["datasets"]
        self.assertEqual(set(datasets["datasets"].values()), paper_paths)
        self.assertEqual(len(datasets["datasets"]), 11)
        self.assertEqual(datasets["redistribution_status"], "unresolved_do_not_redistribute")

        methods = self.registries["methods"]["methods"]
        self.assertEqual(set(methods), {"baseline", "conrep", "graddiff", "npo", "rmu"})
        self.assertEqual(methods["conrep"]["owner"], "project")
        for name in ("graddiff", "npo", "rmu"):
            self.assertEqual(methods[name]["owner"], "open_unlearning_derived")

    def test_candidate_is_portable_but_explicitly_blocked(self):
        self.assertEqual(len(self.index["candidates"]), 1)
        candidate = load(self.index["candidates"][0])
        self.assertFalse(candidate["runnable"])
        self.assertFalse(candidate["historical_equivalence_claimed"])
        self.assertFalse(candidate["may_write_archived_paper_results"])
        self.assertTrue(candidate["gates"])
        self.assertTrue(all(value is False for value in candidate["gates"].values()))
        env_pattern = re.compile(r"^\$\{[A-Z][A-Z0-9_]*\}(?:/[A-Za-z0-9_.-]+)*$")
        self.assertTrue(all(env_pattern.fullmatch(value) for value in candidate["environment"].values()))
        self.assertEqual(candidate["protocol"], "validated_v2")
        protocols = self.registries["protocols"]["protocols"]
        self.assertEqual(protocols["historical_v1"]["result_namespace"], "results/paper")
        self.assertEqual(protocols["validated_v2"]["result_namespace"], "results/validated_v2")

    def test_sweep_is_compact_and_review_only(self):
        self.assertEqual(len(self.index["sweeps"]), 1)
        sweep = load(self.index["sweeps"][0])
        self.assertEqual(sweep["status"], "review_only_not_runnable")
        self.assertFalse(sweep["expanded_products_tracked"])
        self.assertFalse(sweep["historical_equivalence_claimed"])
        self.assertEqual(
            sweep["expected_combinations"],
            math.prod(len(values) for values in sweep["axes"].values()),
        )
        self.assertEqual(sweep["expected_combinations"], 24)


if __name__ == "__main__":
    unittest.main()
