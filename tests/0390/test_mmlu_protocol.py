"""Check fixed MMLU semantics, validation artifacts and checkpoint job partitioning.

Generation is stubbed: these CPU tests verify scoring/orchestration, not GPU math.
The actual generator and audited prompt have already run on both ABCI backbones.
"""

from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from experiments import validation
from experiments.cli import parser, validation_jobs
from experiments.config import write_rows, write_json
from experiments.mmlu_protocol import INSTRUCTION, evaluate


class Tokenizer:
    padding_side = "right"
    truncation_side = "right"

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(messages[0]["content"])


class MMLUProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "mmlu.jsonl"
        self.rows = [
            {"subject": "a", "prompt": "Q1\nAnswer:", "answer": "A"},
            {"subject": "b", "prompt": "Q2\nAnswer:", "answer": "C"},
            {"subject": "b", "prompt": "Q3\nAnswer:", "answer": "D"},
        ]
        write_rows(self.path, self.rows)
        self.options = {
            "batch_size": 2, "max_new_tokens": 128, "max_length": 512,
            "mmlu_max_length": 4096, "mmlu_max_new_tokens": 10,
            "mmlu_mode": "instructed_generate",
        }

    def test_audited_prompt_budget_macro_and_invalid_denominator(self):
        tok, observed = Tokenizer(), []
        answers = iter(["A", "The correct answer is C.", "I do not know"])

        def generate(model, tokenizer, texts, options):
            self.assertEqual(tokenizer.truncation_side, "left")
            self.assertEqual(options["max_new_tokens"], 10)
            self.assertEqual(options["max_length"], 4095)
            tokenizer.padding_side = "left"
            observed.extend(texts)
            return [next(answers) for _ in texts]

        score, counts, diagnostics, details = evaluate(None, tok, self.path, self.options, generate)
        self.assertEqual(observed, [INSTRUCTION + row["prompt"] for row in self.rows])
        self.assertEqual(counts, {"a": 1, "b": 2})
        self.assertEqual(score, 0.75)  # subject macro, not the 2/3 micro average
        self.assertAlmostEqual(diagnostics["invalid_fraction"], 1 / 3)
        self.assertEqual([row["source_row"] for row in details], [0, 1, 2])
        self.assertIsNone(details[-1]["predicted"])
        self.assertFalse(details[-1]["correct"])
        self.assertEqual((tok.padding_side, tok.truncation_side), ("right", "right"))

    def test_records_truncation_and_restores_tokenizer_after_failure(self):
        options = dict(self.options, mmlu_max_length=20)
        tok = Tokenizer()
        _, _, diagnostics, _ = evaluate(
            None, tok, self.path, options, lambda m, t, texts, o: ["A"] * len(texts)
        )
        self.assertEqual(diagnostics["truncated_rows"], 3)
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            evaluate(None, tok, self.path, options,
                     lambda *args: (_ for _ in ()).throw(RuntimeError("generation failed")))
        self.assertEqual(tok.truncation_side, "right")

    def test_rejects_old_scoring_mode(self):
        with self.assertRaisesRegex(ValueError, "instructed_generate"):
            evaluate(None, Tokenizer(), self.path, dict(self.options, mmlu_mode="legacy_letter"), None)

    def validation_config(self):
        data = {}
        for split in ("forget", "retain"):
            path = self.root / f"{split}.jsonl"
            write_rows(path, [{"question value": "P0", "answer key": "condition", "answer value": "asthma"}])
            data[f"{split}_generation"] = str(path)
            path = self.root / f"{split}-mcq.jsonl"
            write_rows(path, [{"prompt": "Which condition?", "mapping": {"A": "asthma", "B": "fever"}, "correct_letter": "A"}])
            data[f"{split}_mcq"] = {name: str(path) for name in ("attribute", "identifier-equal", "identifier-related")}
        options = dict(self.options, regime="pmc", limit=None, mcq_mode="generate", mmlu_file=str(self.path))
        return {"data": data, "evaluation": options}

    def test_validation_writes_raw_mmlu_and_preserves_pmc_scores_and_cache(self):
        cfg = self.validation_config()
        output = self.root / "validation"

        def generate(model, tokenizer, texts, options):
            return ["A" if text.startswith(INSTRUCTION) else "A) asthma" for text in texts]

        with patch.object(validation, "generate", generate):
            report = validation.evaluate_loaded(cfg, "/model", SimpleNamespace(eval=lambda: None), Tokenizer(), output)
        self.assertEqual(report["protocol"], "clinicia-legacy-validation-v3")
        for split in ("forget", "retain"):
            for task in ("qa", "cloze", "background", "attribute", "identifier-equal", "identifier-related", "mean"):
                self.assertEqual(report["metrics"][f"{split}.{task}"], 1.0)
        self.assertEqual(report["metrics"]["utility.mmlu"], 0.5)
        saved = [json.loads(line) for line in (output / "mmlu_predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len(saved), 3)
        self.assertEqual(saved[0]["prompt"], INSTRUCTION + self.rows[0]["prompt"])
        self.assertEqual(validation.run(cfg, "/model", output), report)
        report["protocol_hash"] = "old-protocol"
        write_json(output / "metrics.json", report)
        with self.assertRaisesRegex(ValueError, "Cached validation differs"):
            validation.run(cfg, "/model", output)

    def test_protocol_hash_includes_shared_mmlu_code(self):
        cfg = self.validation_config()
        before, _ = validation.protocol_metadata(cfg)
        original = validation.digest

        def digest(path):
            return "changed" if Path(path).name == "mmlu_protocol.py" else original(path)

        with patch.object(validation, "digest", digest):
            after, _ = validation.protocol_metadata(cfg)
        self.assertNotEqual(before, after)

    def test_backbone_and_checkpoints_have_unique_eight_rank_assignments(self):
        root = self.root / "sft"
        for name in ("checkpoint-100", "checkpoint-50", "final"):
            (root / name).mkdir(parents=True)
        cfg = {"model": {"name_or_path": "/original-model"}}
        jobs = validation_jobs(cfg, root, self.root / "scores", include_backbone=True)
        self.assertEqual([dest.name for _, dest in jobs], ["base", "checkpoint-50", "checkpoint-100", "final"])
        distributed = [job for rank in range(8) for job in jobs[rank::8]]
        self.assertCountEqual(distributed, jobs)
        self.assertEqual(len({dest for _, dest in distributed}), len(jobs))
        args = parser().parse_args(["validate-series", "--include-backbone"])
        self.assertTrue(args.include_backbone)
        self.assertEqual(len(validation_jobs(cfg, root, self.root / "scores")), 3)


if __name__ == "__main__":
    unittest.main()
