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

    def test_paper_readme_exposes_executable_reproduction_routes(self):
        readme = (ROOT / "README.md").read_text()
        for phrase in (
            "Towards Unlearning Beyond Textual Expressions for LLMs",
            "ConRep",
            "ClinicIA",
            "complete commands",
            "Required inputs",
            "Paper experiment",
            "Verification",
            "python scripts/reproduce.py sft-pmc",
            "python scripts/train_conrep.py run",
            "python scripts/reproduce.py baseline-unlearn",
            "python scripts/evaluate_clinicia.py run-model",
            "lm-eval --model hf",
            "python scripts/reproduce.py rebuild-tables",
            "experiments/paper_runs/",
            "data/clinicia/",
        ):
            self.assertIn(phrase, readme)
        self.assertNotIn("baseline-unlearn ...", readme)
        self.assertNotIn("run-model ...", readme)
        sections = [
            "## 0. Install the two model environments",
            "## 1. Materialize and check the data",
            "## 2. Prepare the Regime B PMC starting model with SFT",
            "## 3. Perform unlearning",
            "## 4. Evaluate all six ClinicIA views",
            "## 5. Evaluate general utility with MMLU",
            "## 6. Rebuild the archived paper tables",
        ]
        for index, heading in enumerate(sections):
            start = readme.index(heading)
            end = readme.index(sections[index + 1]) if index + 1 < len(sections) else readme.index("## Paper matrix")
            section = readme[start:end]
            self.assertIn("```bash", section, heading)
            self.assertIn("- **Inputs:**", section, heading)
            self.assertIn("- **Output:", section, heading)
            self.assertIn("- **Paper experiment", section, heading)
            self.assertIn("- **Verification:**", section, heading)

    def test_model_commands_have_offline_inspection_paths(self):
        commands = [
            [
                sys.executable,
                "scripts/reproduce.py",
                "sft-pmc",
                "--output-dir",
                "results/validated_v2/b-pmc-mistral-baseline/model",
                "--dry-run",
            ],
            [
                sys.executable,
                "scripts/reproduce.py",
                "baseline-unlearn",
                "a-diagnosis-mistral-npo",
                "--model-path",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "--forget-data",
                "data/clinicia/regime_a/diagnosis/training/easy_qa.csv",
                "--output-dir",
                "results/validated_v2/a-diagnosis-mistral-npo/model",
                "--dry-run",
            ],
        ]
        outputs = []
        for command in commands:
            result = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.assertEqual("", result.stderr)
            self.assertIn("working_directory=third_party/open-unlearning", result.stdout)
            self.assertIn("verification=", result.stdout)
            outputs.append(result.stdout)
        self.assertIn("experiment=finetune/pmc/default", outputs[0])
        self.assertIn("--config-name=train.yaml", outputs[0])
        self.assertIn("experiment=unlearn/celebrity_diagnosis_npo/default", outputs[1])
        self.assertIn("--config-name=unlearn.yaml", outputs[1])

        historical = json.loads(
            (ROOT / "configs/paper/historical/index.json").read_text(encoding="utf-8")
        )
        rendered = 0
        for row in historical["experiments"]:
            record = json.loads(
                (ROOT / "configs/paper/historical" / row["path"]).read_text(encoding="utf-8")
            )
            if record["method"] not in {"GradDiff", "NPO", "RMU"}:
                continue
            experiment_id = record["experiment_id"]
            command = [
                sys.executable,
                "scripts/reproduce.py",
                "baseline-unlearn",
                experiment_id,
                "--model-path",
                record["starting_checkpoint_id"],
                "--forget-data",
                (
                    "data/clinicia/regime_b/pmc/training/easy_QA_PMC_forget100_state.csv"
                    if record["regime"] == "B"
                    else f"data/clinicia/regime_a/{record['knowledge_target']}/training/easy_qa.csv"
                ),
                "--output-dir",
                f"results/validated_v2/{experiment_id}/model",
                "--dry-run",
            ]
            if record["regime"] == "B":
                command.extend(
                    [
                        "--retain-data",
                        "data/clinicia/regime_b/pmc/training/easy_QA_PMC_retain900_full.csv",
                    ]
                )
            result = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.assertIn(f"paper_experiment={experiment_id}", result.stdout)
            for name, value in record["resolved_hyperparameters"].items():
                self.assertIn(f"trainer.args.{name}={value}", result.stdout)
            rendered += 1
        self.assertEqual(15, rendered)

    def test_conrep_and_clinicia_configs_are_inspectable_without_models(self):
        conrep = subprocess.run(
            [
                sys.executable,
                "scripts/train_conrep.py",
                "config",
                "b-pmc-mistral-conrep",
                "--model-path",
                "results/validated_v2/b-pmc-mistral-baseline/model",
                "--peft-model",
                "none",
                "--output-root",
                "results/validated_v2",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        conrep_config = json.loads(conrep.stdout)
        self.assertIsNone(conrep_config["peft_model_name_or_path"])
        self.assertEqual(
            "results/validated_v2/b-pmc-mistral-baseline/model",
            conrep_config["model_name_or_path"],
        )
        self.assertTrue(conrep_config["forget_csv_path"].endswith("easy_QA_PMC_forget100_state.csv"))

        historical = json.loads(
            (ROOT / "configs/paper/historical/index.json").read_text(encoding="utf-8")
        )
        for row in historical["experiments"]:
            experiment_id = row["experiment_id"]
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_clinicia.py",
                    "paper-config",
                    experiment_id,
                    "--model-path",
                    f"results/validated_v2/{experiment_id}/model",
                    "--output-root",
                    "results/validated_v2",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            config = json.loads(result.stdout)
            self.assertTrue(config["evaluation_sets"])
            self.assertTrue(config["mcq_sets"])
            self.assertIn(experiment_id, config["output_dir"])

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
