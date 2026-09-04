import hashlib
import json
import pathlib
import re
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_COMMIT = "623e305655a10a87685e49d83404fe7cd5f2ed81"
BASE_TREE = "ddd353d31d5df680918ad6a255826dd3a8aa9553"


def git(*args, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    ).stdout


def entries(ref=None):
    command = ("ls-tree", "-rz", ref) if ref else ("ls-files", "-s", "-z")
    result = {}
    for row in git(*command).split(b"\0"):
        if not row:
            continue
        metadata, raw_path = row.split(b"\t", 1)
        fields = metadata.split()
        if ref:
            mode, kind, blob = fields
            if kind != b"blob":
                continue
        else:
            mode, blob, stage = fields
            if stage != b"0":
                raise AssertionError(f"non-zero index stage: {raw_path!r}")
        result[raw_path.decode()] = (mode.decode(), blob.decode())
    return result


class Phase3DRReleaseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = entries(BASE_COMMIT)
        cls.current = entries()
        cls.catalog = json.loads((ROOT / "data/clinicia/catalog.json").read_text())

    def test_base_and_final_scope_are_exact(self):
        self.assertEqual(BASE_TREE, git("rev-parse", BASE_COMMIT + "^{tree}").decode().strip())
        diff = git("diff", "--name-status", "--find-renames=50%", BASE_COMMIT, "--").decode().splitlines()
        counts = {"A": 0, "M": 0, "R": 0, "D": 0}
        for line in diff:
            status = line.split("\t", 1)[0]
            counts[status[0]] = counts.get(status[0], 0) + 1
        self.assertEqual({"A": 34, "M": 39, "R": 548, "D": 0}, counts)
        self.assertEqual(621, len(diff))
        self.assertEqual(718, len(self.current))

    def test_source_inventory_validator_passes(self):
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
            "16 selected historical files across 5 ConRep runs",
            result.stdout.strip(),
        )
        self.assertEqual("", result.stderr)

    def test_release_inventory_validator_passes(self):
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
            "validated 25 path-migrated, base-tree-anchored LFS pointers, "
            "2 dependency components, 2 upstream pins, and 2 license identities",
            result.stdout.strip(),
        )
        self.assertEqual("", result.stderr)

    def test_lfs_migrations_preserve_base_pointer_blobs(self):
        manifest = json.loads((ROOT / "data/lfs_manifest.json").read_text())
        self.assertEqual("2.0", manifest["schema_version"])
        self.assertEqual(25, manifest["pointer_count"])
        self.assertEqual(25, len(manifest["path_migrations"]))
        for row in manifest["path_migrations"]:
            self.assertEqual(row["pointer_blob_oid"], self.base[row["historical_path"]][1])
            self.assertEqual(row["pointer_blob_oid"], self.current[row["path"]][1])
        self.assertEqual("unresolved_do_not_redistribute", manifest["default_redistribution_status"])

    def test_dependency_components_are_owned_and_separate(self):
        matrix = json.loads((ROOT / "docs/dependency_matrix.json").read_text())
        self.assertEqual({"llm2vec", "open_unlearning"}, set(matrix["components"]))
        self.assertEqual("third_party/llm2vec/setup.py", matrix["components"]["llm2vec"]["packaging_path"])
        self.assertEqual("third_party/open-unlearning/setup.py", matrix["components"]["open_unlearning"]["packaging_path"])
        self.assertEqual("empty", matrix["conflicts"][0]["intersection"])
        self.assertEqual("keep_model_environments_separate", matrix["conflicts"][0]["resolution"])
        self.assertTrue(all(value is False for value in matrix["release_policy"].values()))

    def test_paper_readme_exposes_the_reproduction_route(self):
        readme = (ROOT / "README.md").read_text()
        for phrase in (
            "Towards Unlearning Beyond Textual Expressions for LLMs",
            "ConRep",
            "ClinicIA",
            "python scripts/reproduce.py table 1",
            "experiments/paper_runs/",
            "data/clinicia/",
        ):
            self.assertIn(phrase, readme)

    def test_repository_documentation_has_no_broken_local_links(self):
        paths = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
        pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
        checked = 0
        for path in paths:
            for target in pattern.findall(path.read_text(encoding="utf-8")):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                local = target.split("#", 1)[0]
                if not local:
                    continue
                self.assertTrue((path.parent / local).resolve().exists(), f"broken link in {path.relative_to(ROOT)}: {target}")
                checked += 1
        self.assertGreaterEqual(checked, 20)

    def test_archives_and_paper_results_are_unchanged(self):
        expected = {
            "clinicia_provenance_bundle.tar.gz": "6e406e4e96b20413361fa67b2f0af2a67034d0211ba32a1207e8583df8d55fe7",
            "clinicia_configs_mmlu_bundle.tar.gz": "a4b396370aabb6382a028a336202203508991cf910d5e0961d89d8bba75f0bf8",
        }
        for path, digest in expected.items():
            self.assertEqual(digest, hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
            self.assertEqual(self.base[path], self.current[path])
        for path, entry in self.base.items():
            if path.startswith("results/paper/"):
                self.assertEqual(entry, self.current[path], path)

    def test_no_root_project_license_is_invented(self):
        self.assertNotIn("LICENSE", self.current)
        self.assertTrue((ROOT / "third_party/llm2vec/LICENSE").is_file())
        self.assertTrue((ROOT / "third_party/open-unlearning/LICENSE").is_file())


if __name__ == "__main__":
    unittest.main()
