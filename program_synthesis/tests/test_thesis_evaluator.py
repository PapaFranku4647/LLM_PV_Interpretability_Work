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
        self.assertEqual(result.agreement_count_code0, 2)
        self.assertAlmostEqual(float(result.faithfulness_code0), 2.0 / 3.0, places=8)
        # GT: A_S = {x1:8(gt=1), x1:6(gt=1), x1:4(gt=0)}, pred=1 → agree on 8,6 → 2/3
        self.assertEqual(result.agreement_count_gt, 2)
        self.assertAlmostEqual(float(result.faithfulness_gt), 2.0 / 3.0, places=8)

    def test_faithfulness_gt_diverges_from_code0(self) -> None:
        """When Code0 is imperfect, faithfulness_gt and faithfulness_code0 should differ."""
        # Code0 is wrong: always returns 0 regardless of actual data
        def code0_bad(_x):
            return 0

        def code1(x):
            return float(x.get("x1", 0)) > 3

        evaluator = ThesisEvaluator(
            code0_fn=code0_bad,
            train_lines=[
                "x1:8 -> 1",   # gt=1
                "x1:2 -> 0",   # gt=0
                "x1:6 -> 1",   # gt=1
                "x1:4 -> 0",   # gt=0
            ],
        )
        result = evaluator.evaluate_thesis(
            sample_x={"x1": 7},
            pred_label=1,
            check_conditions_fn=code1,
        )
        # A_S = {x1:8, x1:6, x1:4} (3 samples), pred_label=1
        # Code0 always returns 0, so code0 agrees with pred_label=1 on NONE → 0/3
        self.assertEqual(result.agreement_count_code0, 0)
        self.assertAlmostEqual(float(result.faithfulness_code0), 0.0, places=8)
        # GT: 8→1✓, 6→1✓, 4→0✗ → 2/3
        self.assertEqual(result.agreement_count_gt, 2)
        self.assertAlmostEqual(float(result.faithfulness_gt), 2.0 / 3.0, places=8)
        # Verify they actually differ
        self.assertNotAlmostEqual(
            float(result.faithfulness_code0), float(result.faithfulness_gt), places=8
        )

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
        self.assertIsNone(result.faithfulness_code0)
        self.assertIsNone(result.faithfulness_gt)

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
        self.assertEqual(result.agreement_count_code0, 1)

    def test_summarize_aggregation(self) -> None:
        rows = [
            {
                "x_in_A": True,
                "S_size": 4,
                "A_S_size": 2,
                "coverage_ratio": 0.5,
                "coverage_eq": 0.5,
                "agreement_count_code0": 2,
                "faithfulness_code0": 1.0,
                "agreement_count_gt": 1,
                "faithfulness_gt": 0.5,
                "code0_eval_errors": 1,
                "code1_eval_errors": 0,
            },
            {
                "x_in_A": False,
                "S_size": 4,
                "A_S_size": 0,
                "coverage_ratio": 0.0,
                "coverage_eq": 0.0,
                "agreement_count_code0": 0,
                "faithfulness_code0": None,
                "agreement_count_gt": 0,
                "faithfulness_gt": None,
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
        # Code0 faithfulness: defined=[1.0], mean=1.0; all_zero=(1.0+0)/2=0.5
        self.assertAlmostEqual(float(report.mean_faithfulness_code0_defined), 1.0, places=8)
        self.assertAlmostEqual(report.mean_faithfulness_code0_all_zero, 0.5, places=8)
        # GT faithfulness: defined=[0.5], mean=0.5; all_zero=(0.5+0)/2=0.25
        self.assertAlmostEqual(float(report.mean_faithfulness_gt_defined), 0.5, places=8)
        self.assertAlmostEqual(report.mean_faithfulness_gt_all_zero, 0.25, places=8)
        self.assertAlmostEqual(report.mean_a_s_size, 1.0, places=8)
        self.assertAlmostEqual(report.median_a_s_size, 1.0, places=8)
        self.assertEqual(report.code0_eval_error_total, 1)
        self.assertEqual(report.code1_eval_error_total, 2)

    def test_summarize_backward_compat_keys(self) -> None:
        """Verify that to_legacy_dict emits backward-compat aliases."""
        rows = [
            {
                "x_in_A": True,
                "S_size": 4,
                "A_S_size": 2,
                "coverage_ratio": 0.5,
                "coverage_eq": 0.5,
                "agreement_count_code0": 2,
                "faithfulness_code0": 1.0,
                "agreement_count_gt": 1,
                "faithfulness_gt": 0.5,
                "code0_eval_errors": 0,
                "code1_eval_errors": 0,
            },
        ]
        metric_results = [ThesisEvaluator.result_from_mapping(r) for r in rows]
        report = ThesisEvaluator.summarize(metric_results)
        d = report.to_legacy_dict()
        # New names present
        self.assertIn("mean_faithfulness_code0_defined", d)
        self.assertIn("mean_faithfulness_code0_all_zero", d)
        self.assertIn("mean_faithfulness_gt_defined", d)
        self.assertIn("mean_faithfulness_gt_all_zero", d)
        # Backward compat aliases present
        self.assertIn("mean_faithfulness_defined_only", d)
        self.assertIn("mean_faithfulness_all_missing_as_zero", d)
        # Aliases match new keys
        self.assertEqual(d["mean_faithfulness_defined_only"], d["mean_faithfulness_code0_defined"])
        self.assertEqual(d["mean_faithfulness_all_missing_as_zero"], d["mean_faithfulness_code0_all_zero"])

    def test_legacy_dict_schema(self) -> None:
        result = ThesisSampleResult(
            x_in_a=True,
            s_size=10,
            a_s_size=4,
            coverage_ratio=0.4,
            coverage_eq=0.4,
            agreement_count_code0=3,
            faithfulness_code0=0.75,
            agreement_count_gt=2,
            faithfulness_gt=0.5,
            code0_eval_errors=0,
            code1_eval_errors=1,
        )
        legacy = result.to_legacy_dict()
        self.assertTrue({"S_size", "A_S_size", "x_in_A", "coverage_ratio", "coverage_eq"}.issubset(legacy))
        # New fields present
        self.assertIn("agreement_count_code0", legacy)
        self.assertIn("faithfulness_code0", legacy)
        self.assertIn("agreement_count_gt", legacy)
        self.assertIn("faithfulness_gt", legacy)
        # Backward compat aliases
        self.assertIn("agreement_count", legacy)
        self.assertIn("faithfulness", legacy)
        self.assertEqual(legacy["agreement_count"], legacy["agreement_count_code0"])
        self.assertEqual(legacy["faithfulness"], legacy["faithfulness_code0"])

    def test_result_from_mapping_old_format(self) -> None:
        """Verify loading old JSONL that uses 'faithfulness' and 'agreement_count'."""
        row = {
            "x_in_A": True,
            "S_size": 10,
            "A_S_size": 4,
            "coverage_ratio": 0.4,
            "coverage_eq": 0.4,
            "agreement_count": 3,
            "faithfulness": 0.75,
            "code0_eval_errors": 0,
            "code1_eval_errors": 0,
        }
        result = ThesisEvaluator.result_from_mapping(row)
        self.assertEqual(result.agreement_count_code0, 3)
        self.assertAlmostEqual(float(result.faithfulness_code0), 0.75, places=8)
        # GT fields default when absent
        self.assertEqual(result.agreement_count_gt, 0)
        self.assertIsNone(result.faithfulness_gt)


if __name__ == "__main__":
    unittest.main()
