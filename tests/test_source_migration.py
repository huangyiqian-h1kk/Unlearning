import ast
import hashlib
import importlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_COMMIT = "2ca1a47ff87fa0376725618a39d092a868b0bfa5"
BASE_TREE = "2ba2f20310c8ae5028f704e58620cf8af95baac9"
MANIFEST_PATH = ROOT / "docs" / "source_migration_manifest.json"


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
        mode, kind, blob_sha, size = metadata.split()
        if kind == b"blob":
            entries[raw_path.decode("utf-8")] = {
                "mode": mode.decode("ascii"),
                "blob_sha": blob_sha.decode("ascii"),
                "size": int(size),
            }
    return entries


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


class Phase3D2SourceMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.phase = cls.manifest["phase_scope"]
        cls.base = tree_entries(BASE_COMMIT)
        cls.current = index_entries()
        sys.path.insert(0, str(ROOT / "src"))
        cls.conrep = importlib.import_module("conrep")
        cls.conrep_entrypoints = importlib.import_module("conrep.entrypoints")
        cls.clinicia = importlib.import_module("clinicia")
        cls.clinicia_entrypoints = importlib.import_module("clinicia.entrypoints")

    @classmethod
    def tearDownClass(cls):
        source_root = str(ROOT / "src")
        if source_root in sys.path:
            sys.path.remove(source_root)
        for name in list(sys.modules):
            if name == "conrep" or name.startswith("conrep.") or name == "clinicia" or name.startswith("clinicia."):
                sys.modules.pop(name, None)

    def migration_rows(self, component):
        record = self.manifest["migrations"][component]
        return {
            item["filename"]: {
                "legacy": record["legacy_prefix"] + item["filename"],
                "canonical": record["canonical_prefix"] + item["filename"],
                "blob_sha": item["git_blob_sha"],
            }
            for item in record["files"]
        }

    def test_phase_scope_and_base_are_exact(self):
        self.assertEqual(self.manifest["schema_version"], "1.0")
        self.assertEqual(self.manifest["record_kind"], "phase3d2_canonical_source_migration")
        self.assertEqual(self.phase["base_commit"], BASE_COMMIT)
        self.assertEqual(self.phase["base_tree"], BASE_TREE)
        self.assertEqual(git("rev-parse", BASE_COMMIT + "^{tree}").decode().strip(), BASE_TREE)
        self.assertTrue(self.phase["source_movement_performed"])
        self.assertTrue(self.phase["legacy_compatibility_paths_retained"])
        for key in (
            "source_contents_modified",
            "dependency_consolidation_performed",
            "duplicate_package_removed",
            "historical_configuration_changed",
            "scientific_or_provenance_content_changed",
            "model_execution_performed",
            "dependency_download_performed",
        ):
            self.assertFalse(self.phase[key], key)

    def test_exact_moved_sources_preserve_base_blobs(self):
        expected_counts = {"conrep": 12, "clinicia": 11}
        for component, expected_count in expected_counts.items():
            rows = self.migration_rows(component)
            self.assertEqual(len(rows), expected_count)
            self.assertEqual(self.manifest["migrations"][component]["path_count"], expected_count)
            for row in rows.values():
                self.assertEqual(self.base[row["legacy"]]["blob_sha"], row["blob_sha"], row["legacy"])
                self.assertEqual(self.current[row["canonical"]]["blob_sha"], row["blob_sha"], row["canonical"])
                self.assertEqual(self.current[row["canonical"]]["mode"], self.base[row["legacy"]]["mode"])
                self.assertNotEqual(self.current[row["legacy"]]["blob_sha"], row["blob_sha"], row["legacy"])

    def test_tracked_set_is_base_plus_exact_additions(self):
        canonical_sources = {
            row["canonical"]
            for component in ("conrep", "clinicia")
            for row in self.migration_rows(component).values()
        }
        additions = canonical_sources | {
            "docs/source_migration.md",
            "docs/source_migration_manifest.json",
            "scripts/train_conrep.py",
            "scripts/evaluate_clinicia.py",
            "src/conrep/__init__.py",
            "src/conrep/entrypoints.py",
            "src/conrep/legacy/__init__.py",
            "src/clinicia/__init__.py",
            "src/clinicia/adapters.py",
            "src/clinicia/entrypoints.py",
            "src/clinicia/protocols.py",
            "src/clinicia/registry.py",
            "src/clinicia/legacy/__init__.py",
            "tests/test_source_migration.py",
        }
        modified = {
            row["legacy"]
            for component in ("conrep", "clinicia")
            for row in self.migration_rows(component).values()
        } | {
            "docs/repository_architecture.md",
            "tests/test_llm2vec_migration_contract.py",
            "tests/test_repository_cleanup.py",
        }
        self.assertEqual(set(self.current), set(self.base) | additions)
        self.assertEqual(len(self.current), 678)
        for path, entry in self.base.items():
            if path in modified:
                self.assertNotEqual(self.current[path]["blob_sha"], entry["blob_sha"], path)
            else:
                self.assertEqual(
                    (self.current[path]["mode"], self.current[path]["blob_sha"]),
                    (entry["mode"], entry["blob_sha"]),
                    path,
                )

    def test_legacy_redirects_are_small_and_target_canonical_filenames(self):
        for component in ("conrep", "clinicia"):
            for filename, row in self.migration_rows(component).items():
                source = (ROOT / row["legacy"]).read_text(encoding="utf-8")
                ast.parse(source, filename=row["legacy"])
                self.assertLess(len(source.encode("utf-8")), 800)
                self.assertIn("_Path(__file__).name", source)
                self.assertIn("exec(compile(_SOURCE.read_bytes()", source)
                self.assertIn(f'"{component}"', source)
                self.assertEqual((ROOT / row["canonical"]).name, filename)

    def test_known_parse_finding_moved_without_repair(self):
        failures = []
        for component in ("conrep", "clinicia"):
            for row in self.migration_rows(component).values():
                path = ROOT / row["canonical"]
                try:
                    ast.parse(path.read_text(encoding="utf-8"), filename=row["canonical"])
                except SyntaxError as exc:
                    failures.append((row["canonical"], exc.lineno))
        finding = self.manifest["compatibility"]["known_parse_finding"]
        self.assertEqual(failures, [(finding["canonical_path"], finding["line"])])
        self.assertEqual(finding["status"], "preserved")

    def test_historical_conrep_path_remains_traceable(self):
        compatibility = self.manifest["compatibility"]
        legacy = compatibility["historical_conrep_entrypoint"]
        canonical = compatibility["canonical_conrep_source"]
        self.assertIn(legacy, self.current)
        self.assertEqual(
            self.current[canonical]["blob_sha"],
            self.base[legacy]["blob_sha"],
        )
        records = json.loads((ROOT / "configs/historical/paper/index.json").read_text(encoding="utf-8"))
        record_paths = [item["path"] for item in records["experiments"]]
        self.assertTrue(record_paths)
        for relative in record_paths:
            record = json.loads((ROOT / "configs/historical/paper" / relative).read_text(encoding="utf-8"))
            if record.get("method") == "ConRep":
                self.assertEqual(record["historical_training_entry_point"], legacy)

    def test_stable_dispatch_is_lazy_and_restores_process_state(self):
        self.assertEqual(len(self.conrep.VARIANTS), 12)
        self.assertEqual(self.conrep.DEFAULT_VARIANT, "adaptive-random-token-lmloss-margin")
        self.assertEqual(len(self.clinicia_entrypoints.ENTRYPOINTS), 4)
        for module, function_name, choice in (
            (self.conrep_entrypoints, "run_variant", "base"),
            (self.clinicia_entrypoints, "run_entrypoint", "main"),
        ):
            before_argv = sys.argv
            before_path = list(sys.path)
            with mock.patch.object(module.runpy, "run_path") as run_path:
                getattr(module, function_name)(choice, ["fixture-config.json"])
            run_path.assert_called_once()
            self.assertEqual(run_path.call_args.kwargs, {"run_name": "__main__"})
            self.assertEqual(sys.argv, before_argv)
            self.assertEqual(sys.path, before_path)

    def test_stable_launchers_list_without_model_dependencies(self):
        for script, expected_lines in (
            ("scripts/train_conrep.py", 12),
            ("scripts/evaluate_clinicia.py", 4),
        ):
            result = subprocess.run(
                [sys.executable, script, "--list"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(len(result.stdout.splitlines()), expected_lines, script)
            self.assertEqual(result.stderr, "", script)

    def test_clinicia_registry_and_integrity_adapter(self):
        registry = self.clinicia.load_registry()
        self.assertEqual(len(registry), 11)
        self.assertEqual(tuple(registry), self.clinicia.dataset_ids())
        self.assertTrue(all(spec.dataset_id.endswith("@historical_v1") for spec in registry.values()))
        records = self.clinicia.load_records("clinicia/a/diagnosis/id@historical_v1")
        self.assertEqual(len(records), 52)
        with self.assertRaises(self.clinicia.DatasetNotMaterializedError):
            self.clinicia.load_records("clinicia/b/pmc/forget/att@historical_v1")

        spec = self.clinicia.get_dataset("clinicia/a/diagnosis/id@historical_v1")
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / spec.repository_path
            target.parent.mkdir(parents=True)
            target.write_text('{"tampered": true}\n', encoding="utf-8")
            with self.assertRaises(self.clinicia.DatasetIntegrityError):
                self.clinicia.load_records(spec.dataset_id, root=root)

    def test_protocol_namespaces_are_disjoint_and_not_overclaimed(self):
        historical = self.clinicia.get_protocol("historical_v1")
        validated = self.clinicia.get_protocol("validated_v2")
        self.assertEqual(historical.result_namespace, "results/paper")
        self.assertEqual(validated.result_namespace, "results/validated_v2")
        self.assertNotEqual(historical.result_namespace, validated.result_namespace)
        self.assertFalse(historical.runnable)
        self.assertFalse(validated.runnable)
        self.assertFalse(historical.may_modify_archived_paper_results)
        self.assertFalse(validated.may_modify_archived_paper_results)

    def test_protected_scopes_and_third_party_packages_are_unchanged(self):
        protected = [
            path
            for path in self.base
            if path.startswith((
                "configs/historical/",
                "results/paper/",
                "llm2vec/llm2vec/",
                "llm2vec/llm22vec/",
                "llm2vec/open_unlearning/",
            ))
            or path in {
                "clinicia_provenance_bundle.tar.gz",
                "clinicia_configs_mmlu_bundle.tar.gz",
            }
        ]
        self.assertTrue(protected)
        for path in protected:
            self.assertEqual(
                (self.current[path]["mode"], self.current[path]["blob_sha"]),
                (self.base[path]["mode"], self.base[path]["blob_sha"]),
                path,
            )
        expected_archives = {
            "clinicia_provenance_bundle.tar.gz": "6e406e4e96b20413361fa67b2f0af2a67034d0211ba32a1207e8583df8d55fe7",
            "clinicia_configs_mmlu_bundle.tar.gz": "a4b396370aabb6382a028a336202203508991cf910d5e0961d89d8bba75f0bf8",
        }
        for path, digest in expected_archives.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
