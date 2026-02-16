from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROGRAM_SYNTHESIS_DIR = Path(__file__).resolve().parents[1]
if str(PROGRAM_SYNTHESIS_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_SYNTHESIS_DIR))

from thesis_evaluator import ThesisEvaluator, ThesisSampleResult  # noqa: E402


class ThesisEvaluatorTests(unittest.TestCase):
    def test_evaluate_expected_values(self) -> None:
        def code0(x):
            return 1 if float(x.get("x1", 0)) > 5 else 0

        def code1(x):
            return float(x.get("x1", 0)) > 3

        evaluator = ThesisEvaluator(
            code0_fn=code0,
            train_lines=[
                "x1:8 -> 1",
                "x1:2 -> 0",
                "x1:6 -> 1",
                "x1:4 -> 0",
            ],
        )
        result = evaluator.evaluate_thesis(
            sample_x={"x1": 7},
            pred_label=1,
            check_conditions_fn=code1,
        )

        self.assertEqual(result.s_size, 4)
        self.assertEqual(result.a_s_size, 3)
        self.assertTrue(result.x_in_a)
        self.assertAlmostEqual(result.coverage_ratio, 0.75, places=8)
        self.assertAlmostEqual(result.coverage_eq, 0.75, places=8)
        self.assertEqual(result.agreement_count, 2)
        self.assertAlmostEqual(float(result.faithfulness), 2.0 / 3.0, places=8)

    def test_coverage_eq_zero_when_x_not_in_a(self) -> None:
        def code0(_x):
            return 0

        def code1(x):
            return float(x.get("x1", 0)) > 3

        evaluator = ThesisEvaluator(
            code0_fn=code0,
            train_lines=[
                "x1:8 -> 1",
                "x1:2 -> 0",
                "x1:6 -> 1",
                "x1:4 -> 0",
            ],
        )
        result = evaluator.evaluate_thesis(
            sample_x={"x1": 2},
            pred_label=0,
            check_conditions_fn=code1,
        )

        self.assertFalse(result.x_in_a)
        self.assertAlmostEqual(result.coverage_ratio, 0.75, places=8)
        self.assertAlmostEqual(result.coverage_eq, 0.0, places=8)

    def test_faithfulness_none_when_a_s_empty(self) -> None:
        def code0(_x):
            return 1

        def code1(_x):
            return False

        evaluator = ThesisEvaluator(
            code0_fn=code0,
            train_lines=[
                "x1:8 -> 1",
                "x1:2 -> 0",
            ],
        )
        result = evaluator.evaluate_thesis(
            sample_x={"x1": 9},
            pred_label=1,
            check_conditions_fn=code1,
        )

        self.assertEqual(result.a_s_size, 0)
        self.assertIsNone(result.faithfulness)

    def test_error_accounting(self) -> None:
        def parse_line(line: str):
            return {"id": line}, 0

        def predict(_code0_fn, sample):
            if sample.get("id") == "b":
                raise RuntimeError("code0_eval_failed")
            return 1, "dict"

        def check_conditions(sample):
            if sample.get("id") == "a":
                raise RuntimeError("code1_eval_failed")
            return True

        evaluator = ThesisEvaluator(
            code0_fn=lambda _x: 1,
            train_lines=["a", "b", "c"],
            parse_line_fn=parse_line,
            predict_code0_label_fn=predict,
        )
        result = evaluator.evaluate_thesis(
            sample_x={"id": "sample"},
            pred_label=1,
            check_conditions_fn=check_conditions,
        )

        self.assertEqual(result.code1_eval_errors, 1)
        self.assertEqual(result.code0_eval_errors, 1)
        self.assertEqual(result.a_s_size, 2)
        self.assertEqual(result.agreement_count, 1)

    def test_summarize_aggregation(self) -> None:
        rows = [
            {
                "x_in_A": True,
                "S_size": 4,
                "A_S_size": 2,
                "coverage_ratio": 0.5,
                "coverage_eq": 0.5,
                "agreement_count": 2,
                "faithfulness": 1.0,
                "code0_eval_errors": 1,
                "code1_eval_errors": 0,
            },
            {
                "x_in_A": False,
                "S_size": 4,
                "A_S_size": 0,
                "coverage_ratio": 0.0,
                "coverage_eq": 0.0,
                "agreement_count": 0,
                "faithfulness": None,
                "code0_eval_errors": 0,
                "code1_eval_errors": 2,
            },
        ]
        metric_results = [ThesisEvaluator.result_from_mapping(r) for r in rows]
        report = ThesisEvaluator.summarize(metric_results)

        self.assertEqual(report.n_cases, 2)
        self.assertAlmostEqual(report.x_in_a_rate, 0.5, places=8)
        self.assertAlmostEqual(report.mean_coverage_eq_all, 0.25, places=8)
        self.assertAlmostEqual(report.mean_coverage_ratio_all, 0.25, places=8)
        self.assertAlmostEqual(float(report.mean_faithfulness_defined_only), 1.0, places=8)
        self.assertAlmostEqual(report.mean_faithfulness_all_missing_as_zero, 0.5, places=8)
        self.assertAlmostEqual(report.mean_a_s_size, 1.0, places=8)
        self.assertAlmostEqual(report.median_a_s_size, 1.0, places=8)
        self.assertEqual(report.code0_eval_error_total, 1)
        self.assertEqual(report.code1_eval_error_total, 2)

    def test_legacy_dict_schema(self) -> None:
        result = ThesisSampleResult(
            x_in_a=True,
            s_size=10,
            a_s_size=4,
            coverage_ratio=0.4,
            coverage_eq=0.4,
            agreement_count=3,
            faithfulness=0.75,
            code0_eval_errors=0,
            code1_eval_errors=1,
        )
        legacy = result.to_legacy_dict()
        self.assertTrue({"S_size", "A_S_size", "x_in_A", "coverage_ratio", "coverage_eq"}.issubset(legacy))


if __name__ == "__main__":
    unittest.main()
