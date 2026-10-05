"""CPU-only checks for diagnostic sampling and answer extraction."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from experiments.cli import parser
from experiments.mmlu_audit import MODES, extract_letter, sample_rows, summarize


class MMLUAuditTests(unittest.TestCase):
    def test_subject_balanced_slice_preserves_source_indices(self):
        rows = [{"subject": s, "answer": "A", "prompt": "Q\nAnswer:"} for s in ["z", "a", "z", "a", "z"]]
        self.assertEqual([index for index, _ in sample_rows(rows, 2)], [1, 3, 0, 2])
        with self.assertRaises(ValueError):
            sample_rows(rows, 3)

    def test_rejects_bad_answer_and_prompt(self):
        for row in ({"subject": "a", "answer": "", "prompt": "Answer:"}, {"subject": "a", "answer": "A", "prompt": "broken"}):
            with self.assertRaises(ValueError):
                sample_rows([row], 1)

    def test_strict_letter_extraction(self):
        for text, expected in [("B", "B"), (" (C). ", "C"), ("The correct answer is D.", "D"), ("Answer: A", "A"), ("Answer", None), ("A hypothesis is", None), ("C or D", None), ("I cannot answer", None)]:
            with self.subTest(text=text):
                self.assertEqual(extract_letter(text), expected)

    def test_invalid_outputs_count_as_wrong_and_macro_is_subject_weighted(self):
        details = []
        for subject, pred in [("a", "B"), ("b", "A"), ("b", None)]:
            details.append({"subject": subject, "answer": "B", **{mode: {"predicted": pred} for mode in MODES}})
        for values in summarize(details).values():
            self.assertEqual(values["subject_macro_accuracy"], 0.5)
            self.assertAlmostEqual(values["invalid_fraction"], 1 / 3)
            self.assertEqual(values["predicted_counts"]["INVALID"], 1)

    def test_cli_accepts_diagnostic(self):
        args = parser().parse_args(["audit-mmlu", "--checkpoint", "/model", "--output", "/out"])
        self.assertEqual(args.stage, "audit-mmlu")


if __name__ == "__main__":
    unittest.main()
