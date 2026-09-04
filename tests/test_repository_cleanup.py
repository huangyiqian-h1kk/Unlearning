import hashlib
import json
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
START_COMMIT = "544367d956c6cf1bcffa77add2683ed26118e674"
PHASE3C_COMPLETE = "9e843af06e9f5dcf6e69c14e6500ca0c812c84fc"
PHASE3DR_BASE = "623e305655a10a87685e49d83404fe7cd5f2ed81"


def git(*args, cwd=ROOT, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


class RepositoryCleanupHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((ROOT / "results/repository_cleanup_plan.json").read_text())
        cls.batches = {row.get("batch_id"): row for row in cls.plan["cleanup_batches"] if row.get("batch_id")}
        cls.current = set(git("ls-files").stdout.splitlines())

    def test_original_cleanup_census_is_immutable(self):
        rows = []
        raw = subprocess.check_output(
            ["git", "ls-tree", "-rlz", self.plan["starting_commit"]], cwd=ROOT
        )
        for record in raw.split(b"\0"):
            if not record:
                continue
            metadata, path = record.split(b"\t", 1)
            mode, kind, _blob, size = metadata.split()
            if kind == b"blob":
                rows.append((path.decode(), int(size)))
        self.assertEqual(self.plan["tracked_tree"]["total_files"], len(rows))
        self.assertEqual(self.plan["tracked_tree"]["total_ordinary_blob_bytes"], sum(size for _, size in rows))
        digest = hashlib.sha256("".join(path + "\n" for path, _ in sorted(rows)).encode()).hexdigest()
        self.assertEqual(self.plan["tracked_tree"]["sorted_path_list_sha256"], digest)

    def test_phase3c_complete_tree_remains_available(self):
        self.assertEqual(
            "db501bce7d205a3de9f3ef75a1fe6855aadf0d08",
            git("rev-parse", PHASE3C_COMPLETE + "^{tree}").stdout.strip(),
        )
        self.assertEqual(632, len(git("ls-tree", "-r", "--name-only", PHASE3C_COMPLETE).stdout.splitlines()))

    def test_batch_a_record_is_exact(self):
        batch = self.batches["phase3c1-batch-a"]
        self.assertEqual("complete", batch["status"])
        self.assertEqual(679, batch["removed_from_index_count"])
        self.assertEqual(94849293, batch["target_ordinary_blob_bytes"])
        self.assertEqual("78432f869b09aafa43807e6b07539dbd2b66ef37a0c3fa2459b4e995c585acd8", batch["sorted_target_path_list_sha256"])

    def test_batch_b_record_is_exact_and_phase_complete(self):
        batch = self.batches["phase3c1-batch-b"]
        self.assertEqual("complete", batch["status"])
        self.assertTrue(batch["full_phase3c1_complete"])
        self.assertEqual(640, batch["removed_from_index_count"])
        self.assertEqual(1319, batch["full_phase3c1_removed_from_index_count"])
        self.assertEqual("91f70a94bdc294508c8d87d16170d4881e133e713e258847fb842539591d2bdf", batch["sorted_target_path_list_sha256"])

    def test_generated_outputs_remain_untracked(self):
        forbidden = (
            "llm2vec/grid_search_diagnosis/",
            "llm2vec/grid_search_death/",
            "llm2vec/output/",
            "llm2vec/output_PMC/",
            "llm2vec/unlearn_eval/eval_logs/",
        )
        self.assertFalse([path for path in self.current if path.startswith(forbidden)])

    def test_old_generated_paths_remain_ignored_for_existing_worktrees(self):
        samples = (
            "llm2vec/grid_search_diagnosis/configs/example.json",
            "llm2vec/output/checkpoint-1/config.json",
            "llm2vec/unlearn_eval/eval_logs/example.log",
        )
        for path in samples:
            self.assertEqual(0, git("check-ignore", "--no-index", "--", path, check=False).returncode, path)

    def test_semantic_data_paths_are_not_ignored(self):
        catalog = json.loads((ROOT / "data/clinicia/catalog.json").read_text())
        for row in catalog["datasets"]:
            self.assertIn(row["path"], self.current)
            self.assertNotEqual(0, git("check-ignore", "--no-index", "--", row["path"], check=False).returncode)

    def test_archives_remain_byte_identical(self):
        expected = {
            "clinicia_provenance_bundle.tar.gz": "6e406e4e96b20413361fa67b2f0af2a67034d0211ba32a1207e8583df8d55fe7",
            "clinicia_configs_mmlu_bundle.tar.gz": "a4b396370aabb6382a028a336202203508991cf910d5e0961d89d8bba75f0bf8",
        }
        for path, digest in expected.items():
            self.assertIn(path, self.current)
            self.assertEqual(digest, hashlib.sha256((ROOT / path).read_bytes()).hexdigest())

    def test_paper_results_are_unchanged_from_phase3dr_base(self):
        self.assertEqual(b"", subprocess.run(
            ["git", "diff", "--name-only", PHASE3DR_BASE, "--", "results/paper"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout)

    def test_backup_rejects_unsafe_destination(self):
        script = ROOT / "scripts/repository/backup_before_cleanup.sh"
        for destination in ("/", str(ROOT), str(pathlib.Path.home())):
            result = subprocess.run([script, "HEAD", "HEAD", destination], cwd=ROOT, capture_output=True)
            self.assertNotEqual(0, result.returncode, destination)

    def test_restore_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            backup = pathlib.Path(temporary) / "backup"
            (backup / "files").mkdir(parents=True)
            manifest = "sha256\tsize\tpath\n" + "0" * 64 + "\t0\t../escape\n"
            (backup / "manifest.tsv").write_text(manifest)
            digest = hashlib.sha256(manifest.encode()).hexdigest()
            (backup / "manifest.tsv.sha256").write_text(f"{digest}  manifest.tsv\n")
            result = subprocess.run(
                [ROOT / "scripts/repository/restore_after_cleanup.sh", backup],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(0, result.returncode)
            self.assertIn("unsafe manifest path", result.stderr)


if __name__ == "__main__":
    unittest.main()
