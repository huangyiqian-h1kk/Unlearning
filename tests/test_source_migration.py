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
SOURCE_ROOT = ROOT / "src"


class PaperFacingPythonInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SOURCE_ROOT))
        cls.conrep = importlib.import_module("conrep")
        cls.clinicia = importlib.import_module("clinicia")
        cls.conrep_entrypoints = importlib.import_module("conrep.entrypoints")
        cls.clinicia_entrypoints = importlib.import_module("clinicia.entrypoints")

    @classmethod
    def tearDownClass(cls):
        if str(SOURCE_ROOT) in sys.path:
            sys.path.remove(str(SOURCE_ROOT))
        for name in list(sys.modules):
            if name == "conrep" or name.startswith("conrep.") or name == "clinicia" or name.startswith("clinicia."):
                sys.modules.pop(name, None)

    def run_script(self, script, *args):
        return subprocess.run(
            [sys.executable, script, *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def test_metadata_imports_do_not_load_model_dependencies(self):
        code = (
            "import sys; import conrep, clinicia; "
            "blocked={'torch','transformers','peft'}; "
            "assert not (blocked & set(sys.modules))"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
            check=True,
        )

    def test_objectives_map_the_paper_equations(self):
        self.assertEqual((5, 6), self.conrep.get_objective("general").paper_equations)
        self.assertEqual((7,), self.conrep.get_objective("token-swap").paper_equations)
        self.assertEqual((8, 9, 10), self.conrep.get_objective("combined").paper_equations)
        self.assertEqual((11, 12, 13), self.conrep.get_objective("symmetric").paper_equations)
        self.assertEqual("symmetric", self.conrep.SELECTED_PAPER_OBJECTIVE)

    def test_token_swap_convention_round_trips(self):
        annotated = self.conrep.annotate_token_swap("The diagnosis is ", "private", ".")
        parsed = self.conrep.parse_token_swap(annotated)
        self.assertTrue(parsed.annotated)
        self.assertEqual("private", parsed.replacement_span)
        self.assertEqual("The diagnosis is private.", parsed.model_text)
        self.assertFalse(self.conrep.parse_token_swap("plain").annotated)

    def test_probe_registry_matches_the_paper_labels(self):
        labels = [self.clinicia.get_probe(name).paper_label for name in self.clinicia.PAPER_ORDER]
        self.assertEqual(["QA", "Cloze", "BG", "ATT", "IDeq", "ID"], labels)

    def test_metric_helpers_match_normalization_examples(self):
        self.assertEqual(50.0, self.clinicia.regime_a_generation_retain(0.4, 0.8))
        self.assertEqual(50.0, self.clinicia.regime_a_generation_forget(0.4, 0.8))
        self.assertEqual(50.0, self.clinicia.regime_a_mcq_retain(0.5, 0.75))
        self.assertEqual(50.0, self.clinicia.regime_a_mcq_forget(0.5, 0.75))
        self.assertEqual((80.0, 20.0, -60.0), self.clinicia.regime_b_generation(0.8, 0.2, 1.0))

    def test_paper_table_registry_is_complete(self):
        self.assertEqual(set(range(1, 7)), set(self.clinicia.TABLES))
        self.assertEqual(("A", "diagnosis", "normalized"), (
            self.clinicia.get_table(1).regime,
            self.clinicia.get_table(1).target,
            self.clinicia.get_table(1).view,
        ))
        self.assertEqual(("B", "pmc"), (self.clinicia.get_table(2).regime, self.clinicia.get_table(2).target))

    def test_selected_run_registry_is_complete(self):
        runs = self.conrep.load_runs()
        self.assertEqual(5, len(runs))
        self.assertEqual({"diagnosis", "deaths", "pmc"}, {run.target for run in runs.values()})
        self.assertTrue(all(run.path_for("train").is_file() for run in runs.values()))

    def test_normalized_training_config_replaces_machine_paths(self):
        output = pathlib.Path("/portable/artifacts")
        config = self.conrep.normalized_config(
            "a-diagnosis-mistral-conrep",
            "train",
            output_root=output,
        )
        self.assertTrue(config["retain_csv_path"].endswith("data/clinicia/regime_a/shared/retain/wikitext_dup_1_trunc_1.csv"))
        self.assertTrue(config["forget_csv_path"].endswith("data/clinicia/regime_a/diagnosis/training/easy_qa.csv"))
        self.assertEqual(str(output / "a-diagnosis-mistral-conrep/model"), config["output_dir"])

    def test_normalized_evaluation_config_replaces_nested_paths(self):
        output = pathlib.Path("/portable/artifacts")
        config = self.conrep.normalized_config(
            "b-pmc-mistral-conrep",
            "evaluate_retain",
            output_root=output,
        )
        self.assertEqual(str(output / "b-pmc-mistral-conrep/model"), config["model_path"])
        self.assertTrue(all("data/clinicia/regime_b/pmc/" in path for path in config["mcq_sets"].values()))
        self.assertTrue(all(not path.startswith("/gs/") for path in config["evaluation_sets"].values()))

    def test_train_cli_lists_and_describes_selected_runs(self):
        listed = self.run_script("scripts/train_conrep.py", "list")
        self.assertEqual(5, len(listed.stdout.splitlines()))
        shown = self.run_script("scripts/train_conrep.py", "show", "a-diagnosis-llama2-conrep")
        record = json.loads(shown.stdout)
        self.assertEqual("diagnosis", record["target"])
        self.assertIn("data/clinicia/catalog.json", record["portable_execution"])

    def test_evaluation_cli_lists_selected_roles(self):
        result = self.run_script("scripts/evaluate_clinicia.py", "list")
        rows = result.stdout.splitlines()
        self.assertEqual(5, len(rows))
        self.assertTrue(any(row.endswith("evaluate_retain,evaluate_forget") for row in rows))

    def test_reproduce_cli_lists_all_tables(self):
        result = self.run_script("scripts/reproduce.py", "tables")
        self.assertEqual(6, len(result.stdout.splitlines()))
        self.assertIn("Regime A\tdiagnosis\tnormalized", result.stdout)

    def test_dispatch_is_lazy_and_restores_process_state(self):
        cases = (
            (self.conrep_entrypoints, "run_variant", "base"),
            (self.clinicia_entrypoints, "run_entrypoint", "main"),
        )
        for module, function, choice in cases:
            before_argv = sys.argv
            before_path = list(sys.path)
            with mock.patch.object(module.runpy, "run_path") as run_path:
                getattr(module, function)(choice, ["fixture.json"])
            run_path.assert_called_once()
            self.assertEqual({"run_name": "__main__"}, run_path.call_args.kwargs)
            self.assertIs(before_argv, sys.argv)
            self.assertEqual(before_path, sys.path)

    def test_clinicia_registry_retains_current_and_historical_paths(self):
        registry = self.clinicia.load_registry()
        self.assertEqual(11, len(registry))
        for spec in registry.values():
            self.assertTrue(spec.repository_path.startswith("data/clinicia/"))
            self.assertTrue(spec.historical_path.startswith("llm2vec/"))

    def test_clinicia_adapter_checks_materialization_and_integrity(self):
        records = self.clinicia.load_records("clinicia/a/diagnosis/id@historical_v1")
        self.assertEqual(52, len(records))
        with self.assertRaises(self.clinicia.DatasetNotMaterializedError):
            self.clinicia.load_records("clinicia/b/pmc/forget/att@historical_v1")
        spec = self.clinicia.get_dataset("clinicia/a/diagnosis/id@historical_v1")
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / spec.repository_path
            path.parent.mkdir(parents=True)
            path.write_text('{"tampered": true}\n')
            with self.assertRaises(self.clinicia.DatasetIntegrityError):
                self.clinicia.load_records(spec.dataset_id, root=pathlib.Path(temporary))

    def test_protocol_namespaces_are_read_only_and_disjoint(self):
        historical = self.clinicia.get_protocol("historical_v1")
        future = self.clinicia.get_protocol("validated_v2")
        self.assertEqual("results/paper", historical.result_namespace)
        self.assertEqual("results/validated_v2", future.result_namespace)
        self.assertFalse(historical.runnable)
        self.assertFalse(future.runnable)
        self.assertFalse(historical.may_modify_archived_paper_results)
        self.assertFalse(future.may_modify_archived_paper_results)

    def test_unknown_public_identifiers_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "unknown ConRep objective"):
            self.conrep.get_objective("unknown")
        with self.assertRaisesRegex(ValueError, "unknown selected ConRep run"):
            self.conrep.get_run("unknown")
        with self.assertRaisesRegex(ValueError, "unknown ClinicIA probe"):
            self.clinicia.get_probe("unknown")


if __name__ == "__main__":
    unittest.main()
