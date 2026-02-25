from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROGRAM_SYNTHESIS_DIR = Path(__file__).resolve().parents[1]
if str(PROGRAM_SYNTHESIS_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_SYNTHESIS_DIR))

from thesis_analysis import (  # noqa: E402
    compute_trivial_baselines,
    coverage_faith_pairs,
    diagnose_failures,
    load_cases,
    per_function_summary,
    print_markdown_table,
    thesis_complexity_stats,
    compare_prompt_versions,
)


def _make_case(
    fn: str = "fn_m",
    seed: int = 2201,
    sample_index: int = 1,
    coverage_eq: float = 0.5,
    faithfulness_code0: float | None = 1.0,
    faithfulness_gt: float | None = 0.8,
    x_in_A: bool = True,
    code1_accepted: bool = True,
    code1_eval_errors: int = 0,
    thesis_conditions: str = "x0 >= 40.0",
    code1_compile_ok: bool = True,
    code0_pred_error: str | None = None,
) -> dict:
    return {
        "fn": fn,
        "seed": seed,
        "sample_index": sample_index,
        "coverage_eq": coverage_eq,
        "coverage_ratio": coverage_eq,
        "faithfulness_code0": faithfulness_code0,
        "faithfulness_gt": faithfulness_gt,
        # backward compat alias
        "faithfulness": faithfulness_code0,
        "x_in_A": x_in_A,
        "code1_accepted": code1_accepted,
        "code1_eval_errors": code1_eval_errors,
        "thesis_conditions": thesis_conditions,
        "code1_compile_ok": code1_compile_ok,
        "code0_pred_error": code0_pred_error,
    }


class TestPerFunctionSummary(unittest.TestCase):
    def test_single_function(self) -> None:
        cases = [
            _make_case(fn="fn_m", coverage_eq=0.5, faithfulness_code0=1.0, faithfulness_gt=0.8),
            _make_case(fn="fn_m", coverage_eq=0.3, faithfulness_code0=0.8, faithfulness_gt=0.6),
        ]
        result = per_function_summary(cases)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["fn"], "fn_m")
        self.assertEqual(result[0]["n_cases"], 2)
        self.assertAlmostEqual(result[0]["mean_coverage_eq"], 0.4, places=6)
        self.assertAlmostEqual(result[0]["mean_faithfulness"], 0.9, places=6)
        self.assertAlmostEqual(result[0]["mean_faithfulness_gt"], 0.7, places=6)

    def test_multiple_functions(self) -> None:
        cases = [
            _make_case(fn="fn_m", coverage_eq=0.5),
            _make_case(fn="fn_n", coverage_eq=0.3),
            _make_case(fn="fn_n", coverage_eq=0.7),
        ]
        result = per_function_summary(cases)
        self.assertEqual(len(result), 2)
        fns = [r["fn"] for r in result]
        self.assertIn("fn_m", fns)
        self.assertIn("fn_n", fns)

    def test_faithfulness_none_excluded(self) -> None:
        cases = [
            _make_case(fn="fn_m", faithfulness_code0=1.0, faithfulness_gt=0.5),
            _make_case(fn="fn_m", faithfulness_code0=None, faithfulness_gt=None),
        ]
        result = per_function_summary(cases)
        self.assertAlmostEqual(result[0]["mean_faithfulness"], 1.0, places=6)
        self.assertAlmostEqual(result[0]["mean_faithfulness_gt"], 0.5, places=6)


class TestDiagnoseFailures(unittest.TestCase):
    def test_only_x_in_A_false_cases(self) -> None:
        cases = [
            _make_case(x_in_A=True),
            _make_case(x_in_A=False, code1_eval_errors=5, sample_index=2),
        ]
        failures = diagnose_failures(cases)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["reason"], "categorical_mismatch")

    def test_boundary_exclusive(self) -> None:
        cases = [
            _make_case(
                x_in_A=False,
                thesis_conditions="x0 > 40.0",
                code1_eval_errors=0,
            ),
        ]
        failures = diagnose_failures(cases)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["reason"], "boundary_exclusive")

    def test_wrong_code_path(self) -> None:
        cases = [
            _make_case(
                x_in_A=False,
                thesis_conditions="x0 >= 40.0 AND x1 <= 5",
                code1_eval_errors=0,
            ),
        ]
        failures = diagnose_failures(cases)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["reason"], "wrong_code_path")

    def test_skips_code0_pred_errors(self) -> None:
        cases = [
            _make_case(x_in_A=False, code0_pred_error="some error"),
        ]
        failures = diagnose_failures(cases)
        self.assertEqual(len(failures), 0)


class TestThesisComplexityStats(unittest.TestCase):
    def test_counts_conditions(self) -> None:
        cases = [
            _make_case(thesis_conditions="x0 >= 40.0 AND x1 <= 5 OR x2 == 'c1'"),
        ]
        stats = thesis_complexity_stats(cases)
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["n_conditions"], 3)

    def test_single_condition(self) -> None:
        cases = [_make_case(thesis_conditions="x0 >= 40.0")]
        stats = thesis_complexity_stats(cases)
        self.assertEqual(stats[0]["n_conditions"], 1)

    def test_empty_conditions(self) -> None:
        cases = [_make_case(thesis_conditions="")]
        stats = thesis_complexity_stats(cases)
        self.assertEqual(stats[0]["n_conditions"], 0)


class TestCoverageFaithPairs(unittest.TestCase):
    def test_filters_none_faithfulness(self) -> None:
        cases = [
            _make_case(faithfulness_code0=1.0, faithfulness_gt=0.8),
            _make_case(faithfulness_code0=None, faithfulness_gt=None, sample_index=2),
        ]
        pairs = coverage_faith_pairs(cases)
        self.assertEqual(len(pairs), 1)

    def test_returns_expected_keys(self) -> None:
        cases = [_make_case()]
        pairs = coverage_faith_pairs(cases)
        self.assertIn("coverage_eq", pairs[0])
        self.assertIn("faithfulness", pairs[0])
        self.assertIn("faithfulness_gt", pairs[0])
        self.assertIn("fn", pairs[0])


class TestComparePromptVersions(unittest.TestCase):
    def test_comparison(self) -> None:
        v1 = [_make_case(coverage_eq=0.3, faithfulness_code0=0.9, faithfulness_gt=0.7)]
        v2 = [_make_case(coverage_eq=0.5, faithfulness_code0=0.85, faithfulness_gt=0.8)]
        result = compare_prompt_versions(v1, v2)
        self.assertAlmostEqual(result["delta"]["delta_mean_coverage_eq"], 0.2, places=6)
        self.assertAlmostEqual(result["delta"]["delta_mean_faithfulness"], -0.05, places=6)
        self.assertAlmostEqual(result["delta"]["delta_mean_faithfulness_gt"], 0.1, places=6)


class TestComputeTrivialBaselines(unittest.TestCase):
    def test_baselines_have_both_faithfulness_types(self) -> None:
        def code0(x):
            return 1 if float(x.get("x1", 0)) > 5 else 0

        train_lines = [
            "x1:8 -> 1",
            "x1:2 -> 0",
            "x1:6 -> 1",
            "x1:4 -> 0",
        ]
        baselines = compute_trivial_baselines(train_lines, code0)
        for strategy in ["always_0", "always_1"]:
            self.assertIn(strategy, baselines)
            self.assertIn("faithfulness_code0", baselines[strategy])
            self.assertIn("faithfulness_gt", baselines[strategy])
            self.assertIn("coverage_eq", baselines[strategy])

        # always_1: code0 returns 1 for x1=8,6 → 2 agree out of 4 → 0.5
        self.assertAlmostEqual(baselines["always_1"]["faithfulness_code0"], 0.5, places=6)
        # always_1: gt labels 1 for x1=8,6 → 2 agree out of 4 → 0.5
        self.assertAlmostEqual(baselines["always_1"]["faithfulness_gt"], 0.5, places=6)
        # always_0: code0 returns 0 for x1=2,4 → 2 agree out of 4 → 0.5
        self.assertAlmostEqual(baselines["always_0"]["faithfulness_code0"], 0.5, places=6)
        # always_0: gt labels 0 for x1=2,4 → 2 agree out of 4 → 0.5
        self.assertAlmostEqual(baselines["always_0"]["faithfulness_gt"], 0.5, places=6)


class TestLoadCases(unittest.TestCase):
    def test_loads_jsonl(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"fn": "fn_m", "coverage_eq": 0.5}) + "\n")
            f.write(json.dumps({"fn": "fn_n", "coverage_eq": 0.3}) + "\n")
            tmp_path = Path(f.name)
        try:
            cases = load_cases(tmp_path)
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0]["fn"], "fn_m")
        finally:
            tmp_path.unlink()


class TestPrintMarkdownTable(unittest.TestCase):
    def test_formats_table(self) -> None:
        rows = [{"fn": "fn_m", "coverage": 0.5}]
        table = print_markdown_table(rows)
        self.assertIn("fn_m", table)
        self.assertIn("0.5000", table)

    def test_empty_rows(self) -> None:
        table = print_markdown_table([])
        self.assertEqual(table, "(no data)")


if __name__ == "__main__":
    unittest.main()
