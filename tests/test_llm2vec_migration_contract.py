import ast
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_COMMIT = "623e305655a10a87685e49d83404fe7cd5f2ed81"


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
    for row in git("ls-tree", "-rz", ref).split(b"\0"):
        if not row:
            continue
        metadata, raw_path = row.split(b"\t", 1)
        mode, kind, blob = metadata.split()
        if kind == b"blob":
            entries[raw_path.decode()] = (mode.decode(), blob.decode())
    return entries


def index_entries():
    entries = {}
    for row in git("ls-files", "-s", "-z").split(b"\0"):
        if not row:
            continue
        metadata, raw_path = row.split(b"\t", 1)
        mode, blob, stage = metadata.split()
        entries[raw_path.decode()] = (mode.decode(), blob.decode(), stage.decode())
    return entries


class LLM2VecOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tree_entries(BASE_COMMIT)
        cls.current = index_entries()
        cls.contract = json.loads((ROOT / "docs/llm2vec_migration_contract.json").read_text())
        cls.consolidation = json.loads((ROOT / "docs/llm22vec_consolidation.json").read_text())

    def test_obsolete_top_level_is_gone(self):
        self.assertFalse([path for path in self.current if path.startswith("llm2vec/")])

    def test_ownership_roots_are_visibly_separate(self):
        expected = {
            "third_party/llm2vec/": 104,
            "third_party/open-unlearning/": 274,
        }
        for prefix, count in expected.items():
            self.assertEqual(count, sum(path.startswith(prefix) for path in self.current), prefix)
        self.assertTrue(any(path.startswith("src/conrep/backends/llm22vec/") for path in self.current))
        self.assertTrue(any(path.startswith("legacy/llm2vec/") for path in self.current))

    def test_upstream_llm2vec_snapshot_is_byte_identical(self):
        old = {path: entry for path, entry in self.base.items() if path.startswith("llm2vec/llm2vec/")}
        self.assertEqual(20, len(old))
        for historical, entry in old.items():
            current = historical.replace("llm2vec/llm2vec/", "third_party/llm2vec/llm2vec/", 1)
            self.assertEqual(entry, self.current[current][:2], current)

    def test_project_causal_adapter_core_is_byte_identical(self):
        mapping = {
            "llm2vec/llm22vec/llm22vec.py": "src/conrep/backends/llm22vec/llm22vec.py",
            "llm2vec/llm22vec/openunlearn_wrapper.py": "src/conrep/backends/llm22vec/openunlearn_wrapper.py",
        }
        for historical, current in mapping.items():
            self.assertEqual(self.base[historical], self.current[current][:2], current)

    def test_adapter_init_connects_to_upstream_support(self):
        path = ROOT / "src/conrep/backends/llm22vec/__init__.py"
        source = path.read_text()
        ast.parse(source, filename=str(path))
        self.assertIn('"third_party"', source)
        self.assertIn('"llm2vec"', source)
        self.assertIn("__path__.append", source)

    def test_consolidation_record_explains_current_layout(self):
        layout = self.consolidation["current_layout"]
        self.assertEqual("third_party/llm2vec/llm2vec/", layout["canonical_support_prefix"])
        self.assertEqual("src/conrep/backends/llm22vec/", layout["project_adapter_prefix"])
        self.assertEqual(
            {
                "src/conrep/backends/llm22vec/__init__.py",
                "src/conrep/backends/llm22vec/llm22vec.py",
                "src/conrep/backends/llm22vec/openunlearn_wrapper.py",
            },
            set(self.consolidation["retained_derivative_modules"]),
        )

    def test_consolidated_support_modules_still_match_recorded_blobs(self):
        rows = self.consolidation["shared_modules"]
        self.assertEqual(18, len(rows))
        for row in rows:
            current = row["canonical_path"].replace(
                "llm2vec/llm2vec/", "third_party/llm2vec/llm2vec/", 1
            )
            self.assertEqual(row["base_blob"], self.current[current][1], current)

    def test_historical_contract_anchors_remain_traceable(self):
        anchors = {row["path"]: row for row in self.contract["source_anchors"]}
        self.assertEqual(
            anchors["llm2vec/llm2vec/llm2vec.py"]["git_blob_sha"],
            self.current["third_party/llm2vec/llm2vec/llm2vec.py"][1],
        )
        self.assertEqual(
            anchors["llm2vec/llm22vec/llm22vec.py"]["git_blob_sha"],
            self.current["src/conrep/backends/llm22vec/llm22vec.py"][1],
        )
        historical = "llm2vec/ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py"
        self.assertEqual(
            anchors[historical]["git_blob_sha"],
            self.current["src/conrep/legacy/ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py"][1],
        )

    def test_selected_entrypoint_uses_project_adapter(self):
        path = ROOT / "src/conrep/legacy/ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py"
        source = path.read_text()
        self.assertIn("from llm22vec import LLM2Vec", source)

    def test_dispatch_inserts_adapter_before_support_snapshot(self):
        spec = importlib.util.spec_from_file_location("conrep_entrypoints", ROOT / "src/conrep/entrypoints.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        old_argv = sys.argv
        old_path = list(sys.path)
        observed = []

        def capture(*args, **kwargs):
            observed.extend(sys.path[:2])

        with mock.patch.object(module.runpy, "run_path", side_effect=capture) as run_path:
            module.run_variant(module.DEFAULT_VARIANT, ["fixture.json"])
            active_path = run_path.call_args.args[0]
            self.assertTrue(active_path.endswith("ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py"))
        self.assertEqual(
            [str(module.PROJECT_BACKEND_ROOT), str(module.LEGACY_LLM2VEC_ROOT)],
            observed,
        )
        self.assertIs(sys.argv, old_argv)
        self.assertEqual(old_path, sys.path)

    def test_third_party_licenses_remain_pinned(self):
        matrix = json.loads((ROOT / "docs/dependency_matrix.json").read_text())
        for record in matrix["components"].values():
            content = (ROOT / record["license_path"]).read_bytes()
            self.assertEqual(record["license_sha256"], hashlib.sha256(content).hexdigest())

    def test_project_extensions_are_not_presented_as_upstream(self):
        upstream = {path.removeprefix("third_party/llm2vec/") for path in self.current if path.startswith("third_party/llm2vec/")}
        self.assertNotIn("experiments/mteb_eval_unlearn_cyber_wiki.py", upstream)
        self.assertIn("legacy/llm2vec_extensions/experiments/mteb_eval_unlearn_cyber_wiki.py", self.current)


if __name__ == "__main__":
    unittest.main()
