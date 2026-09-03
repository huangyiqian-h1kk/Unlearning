import hashlib
import importlib.util
import json
import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "docs" / "repository_source_inventory.json"
VALIDATOR_PATH = ROOT / "scripts" / "repository" / "validate_source_inventory.py"

SPEC = importlib.util.spec_from_file_location("validate_source_inventory", VALIDATOR_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def git(*args):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def index_entries():
    entries = {}
    for record in git("ls-files", "-s", "-z").split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, blob_sha, stage = metadata.split()
        entries[raw_path.decode("utf-8")] = {
            "mode": mode.decode("ascii"),
            "blob_sha": blob_sha.decode("ascii"),
            "stage": stage.decode("ascii"),
        }
    return entries


class Phase3D0SourceInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        cls.baseline = cls.inventory["baseline"]
        cls.allowed = set(cls.inventory["phase_scope"]["allowed_additive_paths"])
        cls.allowed_modified = set(cls.inventory["phase_scope"]["allowed_modified_paths"])
        cls.base_entries = VALIDATOR.tree_entries(cls.baseline["commit"])

    def test_offline_inventory_validator(self):
        self.assertEqual(
            VALIDATOR.validate_inventory(INVENTORY_PATH),
            {
                "baseline_paths": 632,
                "llm2vec_identical": 99,
                "llm2vec_modified": 5,
                "open_unlearning_identical": 199,
                "open_unlearning_modified": 5,
                "portability_files": 107,
                "credential_matches": 0,
            },
        )

    def test_phase3d0_has_six_additions_and_one_test_update(self):
        expected = {
            "docs/repository_source_inventory.json",
            "docs/source_ownership_audit.md",
            "docs/upstream_snapshots/llm2vec-312adcf.json",
            "docs/upstream_snapshots/open-unlearning-d33c476.json",
            "scripts/repository/validate_source_inventory.py",
            "tests/test_source_inventory.py",
        }
        self.assertEqual(self.allowed, expected)
        self.assertEqual(self.allowed_modified, {"tests/test_repository_cleanup.py"})
        changed = {}
        for line in git("diff", "--name-status", self.baseline["commit"], "--").decode().splitlines():
            status, path = line.split("\t", 1)
            changed[path] = status
        expected_changes = {path: "A" for path in expected}
        expected_changes["tests/test_repository_cleanup.py"] = "M"
        self.assertEqual(changed, expected_changes)

    def test_baseline_paths_keep_exact_index_identities(self):
        current = index_entries()
        self.assertEqual(set(current), set(self.base_entries) | self.allowed)
        self.assertEqual(len(current), 638)
        for path, expected in self.base_entries.items():
            self.assertEqual(current[path]["stage"], "0", path)
            if path in self.allowed_modified:
                continue
            self.assertEqual(
                (current[path]["mode"], current[path]["blob_sha"]),
                (expected["mode"], expected["blob_sha"]),
                path,
            )

    def test_upstream_comparison_contract(self):
        comparisons = {item["component"]: item for item in self.inventory["upstream_comparisons"]}
        llm2vec = comparisons["LLM2Vec"]
        open_unlearning = comparisons["OpenUnlearning"]
        self.assertEqual(
            [
                llm2vec[name]["path_count"]
                for name in ("local", "upstream", "common", "identical", "modified", "local_only", "upstream_only")
            ],
            [251, 105, 104, 99, 5, 147, 1],
        )
        self.assertEqual(
            [
                open_unlearning[name]["path_count"]
                for name in ("local", "upstream", "common", "identical", "modified", "local_only", "upstream_only")
            ],
            [289, 204, 204, 199, 5, 85, 0],
        )
        self.assertEqual(
            [item["path"] for item in llm2vec["modified_paths"]],
            [
                "experiments/run_simcse.py",
                "llm2vec/loss/HardNegativeNLLLoss.py",
                "llm2vec/loss/__init__.py",
                "llm2vec/loss/utils.py",
                "train_configs/simcse/Mistral.json",
            ],
        )
        self.assertEqual(
            [item["path"] for item in open_unlearning["modified_paths"]],
            [
                "configs/eval/tofu.yaml",
                "configs/experiment/finetune/tofu/default.yaml",
                "configs/trainer/finetune.yaml",
                "src/data/__init__.py",
                "src/trainer/unlearn/npo.py",
            ],
        )

    def test_historical_conrep_entrypoint_is_blob_anchored(self):
        entrypoints = {item["id"]: item for item in self.inventory["critical_entrypoints"]}
        historical = entrypoints["conrep_historical_training"]
        self.assertEqual(
            historical["path"],
            "llm2vec/ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py",
        )
        self.assertEqual(
            self.base_entries[historical["path"]]["blob_sha"],
            "adbbcab5e9316d6b1b9dea1521a4c54c0a63b1ef",
        )
        self.assertFalse(historical["lightweight_execution_allowed_in_repository_audit"])

    def test_llm22vec_removal_remains_blocked(self):
        duplicate = self.inventory["duplicate_packages"]
        self.assertEqual(duplicate["canonical"]["path_count"], 20)
        self.assertEqual(duplicate["derivative"]["path_count"], 21)
        self.assertEqual(duplicate["same_named_common"]["path_count"], 19)
        self.assertEqual(duplicate["same_named_identical"]["path_count"], 18)
        self.assertEqual(duplicate["same_named_modified_paths"], ["__init__.py"])
        self.assertEqual(duplicate["canonical_only_paths"], ["llm2vec.py"])
        self.assertEqual(duplicate["derivative_only_paths"], ["llm22vec.py", "openunlearn_wrapper.py"])
        self.assertEqual(duplicate["equivalence_status"], "not_validated")
        self.assertFalse(duplicate["removal_allowed"])

    def test_dependency_and_license_gates_are_not_overclaimed(self):
        dependencies = self.inventory["dependency_surfaces"]
        licenses = self.inventory["license_status"]
        self.assertEqual(dependencies["root_environment_status"], "absent")
        self.assertFalse(dependencies["validated_reproduction_environment_available"])
        self.assertEqual(
            dependencies["known_conflicts"],
            [{"package": "transformers", "status": "unresolved_version_mismatch_between_nested_projects"}],
        )
        self.assertEqual(licenses["root_project_license"], "absent_and_requires_researcher_choice")
        self.assertEqual(licenses["project_owned_code_license_status"], "unresolved")
        self.assertEqual(
            {item["component"]: item["license"] for item in licenses["third_party_licenses"]},
            {"LLM2Vec": "MIT", "OpenUnlearning": "MIT"},
        )

    def test_portability_findings_and_credential_scan_are_frozen(self):
        scan = self.inventory["portability_and_privacy_scan"]
        self.assertTrue(scan["baseline_only"])
        self.assertEqual(scan["legacy_absolute_path_matches"]["path_count"], 106)
        self.assertEqual(scan["cluster_or_account_marker_matches"]["path_count"], 106)
        self.assertEqual(scan["legacy_absolute_or_identity_matches"]["path_count"], 107)
        self.assertEqual(scan["credential_pattern_matches"]["path_count"], 0)
        self.assertEqual(
            scan["credential_pattern_matches"]["sorted_path_list_sha256"],
            hashlib.sha256(b"").hexdigest(),
        )
        self.assertTrue(scan["remediation_status"].startswith("not_started"))

    def test_scientific_and_provenance_content_is_unchanged(self):
        phase = self.inventory["phase_scope"]
        self.assertFalse(phase["source_movement_performed"])
        self.assertFalse(phase["scientific_or_provenance_content_changed"])
        expected_archives = {
            "clinicia_provenance_bundle.tar.gz": "6e406e4e96b20413361fa67b2f0af2a67034d0211ba32a1207e8583df8d55fe7",
            "clinicia_configs_mmlu_bundle.tar.gz": "a4b396370aabb6382a028a336202203508991cf910d5e0961d89d8bba75f0bf8",
        }
        for path, expected in expected_archives.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected, path)
        protected = [
            path
            for path in self.base_entries
            if path.startswith(("configs/historical/", "results/paper/"))
            or path in expected_archives
        ]
        self.assertTrue(protected)
        self.assertEqual(
            git("diff", "--name-only", self.baseline["commit"], "--", *protected),
            b"",
        )


if __name__ == "__main__":
    unittest.main()
