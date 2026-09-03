import hashlib
import importlib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_COMMIT = "39b4824640236c33a74ec5d297d834976e2bd388"
BASE_TREE = "97fd3c030cb36374c1d5bf8059bad213a263ac1e"
CONSOLIDATION_PATH = ROOT / "docs" / "llm22vec_consolidation.json"


def git(*args):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def tree_entries(ref):
    entries = {}
    for record in git("ls-tree", "-rlz", ref).split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, blob, size = metadata.split()
        if kind == b"blob":
            entries[raw_path.decode("utf-8")] = {
                "mode": mode.decode("ascii"),
                "blob": blob.decode("ascii"),
                "size": int(size),
            }
    return entries


def index_entries():
    entries = {}
    for record in git("ls-files", "-s", "-z").split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, blob, stage = metadata.split()
        entries[raw_path.decode("utf-8")] = {
            "mode": mode.decode("ascii"),
            "blob": blob.decode("ascii"),
            "stage": stage.decode("ascii"),
        }
    return entries


def path_hash(paths):
    payload = "".join(path + "\n" for path in sorted(paths)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Phase3D3ReleaseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tree_entries(BASE_COMMIT)
        cls.current = index_entries()
        cls.consolidation = json.loads(CONSOLIDATION_PATH.read_text(encoding="utf-8"))

    def test_exact_phase_scope_and_immutable_base(self):
        self.assertEqual(git("rev-parse", BASE_COMMIT + "^{tree}").decode().strip(), BASE_TREE)
        added = {
            "README.md",
            "THIRD_PARTY.md",
            "configs/components/README.md",
            "configs/components/datasets.json",
            "configs/components/methods.json",
            "configs/components/models.json",
            "configs/components/protocols.json",
            "configs/reproduction/README.md",
            "configs/reproduction/index.json",
            "configs/reproduction/validated-v2-conrep-diagnosis-mistral.json",
            "configs/sweeps/README.md",
            "configs/sweeps/validated-v2-review-matrix.json",
            "data/README.md",
            "data/lfs_manifest.json",
            "docs/dependencies.md",
            "docs/dependency_matrix.json",
            "docs/llm22vec_consolidation.json",
            "docs/manuscript/README.md",
            "docs/release_integration.md",
            "environments/README.md",
            "scripts/README.md",
            "scripts/configs/validate_reproduction_configs.py",
            "scripts/repository/validate_release_inventory.py",
            "tests/configs/test_reproduction_configs.py",
            "tests/test_release_integration.py",
        }
        modified = {
            "configs/README.md",
            "docs/repository_architecture.md",
            "llm2vec/llm22vec/__init__.py",
            "tests/test_source_migration.py",
        }
        removed_duplicates = {
            row["derivative_path"] for row in self.consolidation["shared_modules"]
        }
        deleted = removed_duplicates | {"docs/README_DRAFT.md"}
        self.assertEqual(set(self.current), (set(self.base) - deleted) | added)
        self.assertEqual(len(self.current), len(self.base) - len(deleted) + len(added))
        for path, entry in self.base.items():
            if path in deleted:
                self.assertNotIn(path, self.current)
            elif path in modified:
                self.assertNotEqual(self.current[path]["blob"], entry["blob"], path)
            else:
                self.assertEqual(
                    (self.current[path]["mode"], self.current[path]["blob"]),
                    (entry["mode"], entry["blob"]),
                    path,
                )
        self.assertTrue(added.isdisjoint(self.base))

    def test_only_blob_identical_support_modules_were_consolidated(self):
        record = self.consolidation
        self.assertEqual(record["phase_base"], {"commit": BASE_COMMIT, "tree": BASE_TREE})
        rows = record["shared_modules"]
        derivative_paths = {row["derivative_path"] for row in rows}
        self.assertEqual(len(rows), 18)
        self.assertEqual(len(derivative_paths), 18)
        self.assertEqual(path_hash(derivative_paths), record["removed_sorted_path_list_sha256"])
        for row in rows:
            derivative = row["derivative_path"]
            canonical = row["canonical_path"]
            expected = row["base_blob"]
            self.assertEqual(self.base[derivative]["blob"], expected, derivative)
            self.assertEqual(self.base[canonical]["blob"], expected, canonical)
            self.assertEqual(self.current[canonical]["blob"], expected, canonical)
            self.assertNotIn(derivative, self.current)
        for path in (
            "llm2vec/llm22vec/llm22vec.py",
            "llm2vec/llm22vec/openunlearn_wrapper.py",
        ):
            self.assertEqual(self.current[path]["blob"], self.base[path]["blob"], path)
        validation = record["validation"]
        self.assertTrue(validation["base_blob_identity_proven"])
        self.assertTrue(validation["causal_contract_tests_retained"])
        self.assertFalse(validation["causal_adapter_removed"])
        for key in ("real_dependency_import", "real_model_initialization", "real_model_numerical_equivalence"):
            self.assertEqual(validation[key], "not_run")

    def test_shared_support_modules_resolve_from_canonical_package(self):
        package_root = ROOT / "llm2vec"
        managed = {
            "llm22vec",
            "llm22vec.llm22vec",
            "llm22vec.experiment_utils",
            "llm22vec.version",
        }
        missing = object()
        previous = {name: sys.modules.get(name, missing) for name in managed}
        previous_dont_write = sys.dont_write_bytecode
        sys.path.insert(0, str(package_root))
        sys.dont_write_bytecode = True
        stub = types.ModuleType("llm22vec.llm22vec")
        stub.LLM2Vec = object()
        try:
            for name in managed:
                sys.modules.pop(name, None)
            sys.modules["llm22vec.llm22vec"] = stub
            package = importlib.import_module("llm22vec")
            self.assertEqual(
                list(package.__path__),
                [str(package_root / "llm22vec"), str(package_root / "llm2vec")],
            )
            for name in ("experiment_utils", "version"):
                module = importlib.import_module("llm22vec." + name)
                self.assertEqual(pathlib.Path(module.__file__).parent, package_root / "llm2vec")
            for name in ("dataset", "loss", "models"):
                spec = importlib.util.find_spec("llm22vec." + name)
                self.assertIsNotNone(spec)
                self.assertEqual(pathlib.Path(spec.origin).parent, package_root / "llm2vec" / name)
        finally:
            sys.dont_write_bytecode = previous_dont_write
            if sys.path and sys.path[0] == str(package_root):
                sys.path.pop(0)
            else:
                sys.path.remove(str(package_root))
            for name, value in previous.items():
                if value is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
            importlib.invalidate_caches()

    def test_release_inventory_validator_and_dependency_split(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/repository/validate_release_inventory.py",
                "--manifest",
                "data/lfs_manifest.json",
                "--dependencies",
                "docs/dependency_matrix.json",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            "validated 25 base-tree-anchored LFS pointers, 2 dependency components, "
            "2 upstream pins, and 2 license identities",
        )
        self.assertEqual(result.stderr, "")
        matrix = json.loads((ROOT / "docs/dependency_matrix.json").read_text(encoding="utf-8"))
        self.assertEqual(matrix["conflicts"][0]["intersection"], "empty")
        self.assertEqual(matrix["conflicts"][0]["resolution"], "keep_model_environments_separate")
        self.assertTrue(all(value is False for value in matrix["release_policy"].values()))

    def test_lfs_inventory_is_base_anchored_and_rights_conservative(self):
        manifest = json.loads((ROOT / "data/lfs_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["pointer_count"], 25)
        self.assertEqual(manifest["phase_base"], {"commit": BASE_COMMIT, "tree": BASE_TREE})
        self.assertEqual(
            manifest["tracked_pointer_scopes"],
            ["llm2vec/UnlearnData/", "llm2vec/cache/"],
        )
        self.assertFalse(manifest["duplicate_per_object_metadata_published"])
        self.assertEqual(
            manifest["identity_policy"],
            "validate_current_paths_and_pointer_blobs_against_the_phase_base_tree",
        )
        self.assertEqual(manifest["default_redistribution_status"], "unresolved_do_not_redistribute")
        self.assertEqual(manifest["object_availability_status"], "not_verified_by_phase3d3")

    def test_root_release_docs_expose_unresolved_gates(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        third_party = (ROOT / "THIRD_PARTY.md").read_text(encoding="utf-8")
        self.assertIn("does **not** yet claim complete end-to-end reproduction", readme)
        self.assertIn("incompatible Transformers versions", readme)
        self.assertIn("no root project license", readme)
        self.assertIn("No author", readme)
        self.assertIn("Pinned inferred revision", third_party)
        self.assertFalse((ROOT / "LICENSE").exists())
        self.assertFalse((ROOT / "docs" / "README_DRAFT.md").exists())
        self.assertTrue((ROOT / "docs" / "manuscript" / "README.md").is_file())

    def test_new_documentation_has_no_broken_local_links(self):
        paths = [
            ROOT / "README.md",
            ROOT / "THIRD_PARTY.md",
            ROOT / "configs" / "README.md",
            ROOT / "data" / "README.md",
            ROOT / "docs" / "dependencies.md",
            ROOT / "environments" / "README.md",
        ]
        link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
        checked = 0
        for path in paths:
            for target in link_pattern.findall(path.read_text(encoding="utf-8")):
                if target.startswith(("http://", "https://", "#")):
                    continue
                resolved = (path.parent / target.split("#", 1)[0]).resolve()
                self.assertTrue(resolved.exists(), f"broken link in {path.relative_to(ROOT)}: {target}")
                checked += 1
        self.assertGreaterEqual(checked, 15)

    def test_archives_and_scientific_evidence_are_unchanged(self):
        expected_archives = {
            "clinicia_provenance_bundle.tar.gz": "6e406e4e96b20413361fa67b2f0af2a67034d0211ba32a1207e8583df8d55fe7",
            "clinicia_configs_mmlu_bundle.tar.gz": "a4b396370aabb6382a028a336202203508991cf910d5e0961d89d8bba75f0bf8",
        }
        for path, digest in expected_archives.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
            self.assertEqual(self.current[path]["blob"], self.base[path]["blob"])
        for prefix in ("configs/historical/", "results/paper/"):
            for path, entry in self.base.items():
                if path.startswith(prefix):
                    self.assertEqual(self.current[path]["blob"], entry["blob"], path)


if __name__ == "__main__":
    unittest.main()
