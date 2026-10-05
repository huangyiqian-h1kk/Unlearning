"""Check resource-aware PBS generation without contacting the scheduler."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load_submitter():
    spec = importlib.util.spec_from_file_location(
        "submit0390", ROOT / "scripts/abci/0390_submit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ABCISubmitTests(unittest.TestCase):
    def render_dry_run(self, stage, *arguments):
        submit = load_submitter()
        with tempfile.TemporaryDirectory() as directory:
            # main() resolves the checkout relative to its own script location.
            submit.__file__ = str(Path(directory) / "scripts/abci/0390_submit.py")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                submit.main(
                    [stage, "--model", "llama3b", "--run-id", "resource-test",
                     "--dry-run", *arguments]
                )
            scripts = list((Path(directory) / "logs/0390/jobs").glob("*.pbs"))
            self.assertEqual(len(scripts), 1)
            script = scripts[0].read_text()
            subprocess.run(["bash", "-n"], input=script, text=True, check=True)
            self.assertIn("#PBS -P gcg51557\n", script)
            self.assertIn("#PBS -q R9920261000\n", script)
            self.assertIn("#PBS -N 0390_", script)
            return script

    def test_single_gpu_stages_default_to_hg(self):
        for stage in ("validate", "audit-mmlu", "analyze", "falcon-layers", "relearn-augment"):
            with self.subTest(stage=stage):
                script = self.render_dry_run(stage)
                self.assertIn("#PBS -v RTYPE=rt_HG\n", script)
                self.assertIn("--nproc 1 --", script)

    def test_training_and_parallel_validation_keep_hf(self):
        for stage in ("sft", "unlearn", "baseline", "smoke", "validate-series"):
            with self.subTest(stage=stage):
                script = self.render_dry_run(stage)
                self.assertIn("#PBS -v RTYPE=rt_HF\n", script)
                self.assertIn("--nproc 8 --", script)

    def test_hg_override_keeps_training_arguments(self):
        script = self.render_dry_run(
            "unlearn", "--rtype", "rt_HG", "--checkpoint", "/model path/sft",
            "--set", "unlearn.batch_sizes.general=32",
        )
        self.assertIn("#PBS -v RTYPE=rt_HG\n", script)
        self.assertIn("--nproc 1 --", script)
        self.assertIn("--checkpoint '/model path/sft'", script)
        self.assertIn("--set unlearn.batch_sizes.general=32", script)

    def test_hg_rejects_eight_gpu_processes(self):
        with self.assertRaisesRegex(ValueError, "rt_HG allocates one GPU"):
            self.render_dry_run("sft", "--rtype", "rt_HG", "--nproc", "8")

    def test_unsharded_evaluation_rejects_multiple_writers(self):
        with self.assertRaisesRegex(ValueError, "no multi-rank output sharding"):
            self.render_dry_run("validate", "--rtype", "rt_HF", "--nproc", "8")

    def test_explicit_hf_single_gpu_is_supported(self):
        script = self.render_dry_run("validate", "--rtype", "rt_HF")
        self.assertIn("#PBS -v RTYPE=rt_HF\n", script)
        self.assertIn("--nproc 1 --", script)


if __name__ == "__main__":
    unittest.main()
