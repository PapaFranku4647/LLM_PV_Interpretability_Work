from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


PROGRAM_SYNTHESIS_DIR = Path(__file__).resolve().parents[1]
if str(PROGRAM_SYNTHESIS_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_SYNTHESIS_DIR))


def _build_stub_modules() -> dict[str, types.ModuleType]:
    openai_mod = types.ModuleType("openai")

    class DummyOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    openai_mod.OpenAI = DummyOpenAI
    return {"openai": openai_mod}


class Step23MatrixMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_modules = {}
        for module_name, module_obj in _build_stub_modules().items():
            cls._saved_modules[module_name] = sys.modules.get(module_name)
            sys.modules[module_name] = module_obj

        sys.modules.pop("thesis_runner", None)
        cls.mod = importlib.import_module("thesis_runner")
        sys.modules.pop("thesis_evaluator", None)
        cls.eval_mod = importlib.import_module("thesis_evaluator")

    @classmethod
    def tearDownClass(cls) -> None:
        sys.modules.pop("thesis_runner", None)
        sys.modules.pop("thesis_evaluator", None)
        for module_name, original in cls._saved_modules.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original

    def test_compute_equation_metrics_expected_values(self) -> None:
        def code0(x):
            return 1 if float(x.get("x1", 0)) > 5 else 0

        def code1(x):
            return float(x.get("x1", 0)) > 3

        sample = {"x1": 7}
        train = [
            "x1:8 -> 1",
            "x1:2 -> 0",
            "x1:6 -> 1",
            "x1:4 -> 0",
        ]
        out = self.mod.compute_equation_metrics(
            code0_fn=code0,
            code1_fn=code1,
            sample=sample,
            pred_label=1,
            train_lines=train,
        )
        evaluator = self.eval_mod.ThesisEvaluator(code0_fn=code0, train_lines=train)
        expected = evaluator.evaluate_thesis(
            sample_x=sample,
            pred_label=1,
            check_conditions_fn=code1,
        ).to_legacy_dict()
        self.assertEqual(out, expected)
        self.assertEqual(out["S_size"], 4)
        self.assertEqual(out["A_S_size"], 3)
        self.assertTrue(out["x_in_A"])
        self.assertAlmostEqual(float(out["coverage_ratio"]), 0.75, places=8)
        self.assertAlmostEqual(float(out["coverage_eq"]), 0.75, places=8)
        self.assertEqual(out["agreement_count"], 2)
        self.assertAlmostEqual(float(out["faithfulness"]), 2.0 / 3.0, places=8)

    def test_coverage_eq_zero_when_x_not_in_a(self) -> None:
        def code0(x):
            return 0

        def code1(x):
            return float(x.get("x1", 0)) > 3

        sample = {"x1": 2}
        train = [
            "x1:8 -> 1",
            "x1:2 -> 0",
            "x1:6 -> 1",
            "x1:4 -> 0",
        ]
        out = self.mod.compute_equation_metrics(
            code0_fn=code0,
            code1_fn=code1,
            sample=sample,
            pred_label=0,
            train_lines=train,
        )
        self.assertFalse(out["x_in_A"])
        self.assertAlmostEqual(float(out["coverage_ratio"]), 0.75, places=8)
        self.assertAlmostEqual(float(out["coverage_eq"]), 0.0, places=8)

    def test_summarize_group(self) -> None:
        rows = [
            {
                "code1_accepted": True,
                "code1_compile_ok": True,
                "response_label_matches_prediction": True,
                "x_in_A": True,
                "coverage_eq": 0.2,
                "coverage_ratio": 0.2,
                "faithfulness": 1.0,
            },
            {
                "code1_accepted": False,
                "code1_compile_ok": False,
                "response_label_matches_prediction": True,
                "x_in_A": False,
                "coverage_eq": 0.0,
                "coverage_ratio": 0.1,
                "faithfulness": None,
            },
        ]
        s = self.mod.summarize_group(rows)
        self.assertEqual(s["n_cases"], 2)
        self.assertAlmostEqual(float(s["accepted_rate"]), 0.5, places=8)
        self.assertAlmostEqual(float(s["compile_ok_rate"]), 0.5, places=8)
        self.assertAlmostEqual(float(s["label_match_rate"]), 1.0, places=8)
        self.assertAlmostEqual(float(s["x_in_A_rate"]), 0.5, places=8)
        self.assertAlmostEqual(float(s["mean_coverage_eq_all"]), 0.1, places=8)
        self.assertAlmostEqual(float(s["mean_coverage_ratio_all"]), 0.15, places=8)
        self.assertAlmostEqual(float(s["mean_faithfulness_defined_only"]), 1.0, places=8)
        self.assertAlmostEqual(float(s["mean_faithfulness_all_missing_as_zero"]), 0.5, places=8)

    def test_compute_auto_split_basic(self) -> None:
        from live_eval_common import compute_auto_split
        train, val, test = compute_auto_split(1527, 1669, train_cap=200, total_cap=5000)
        total = train + val + test
        self.assertEqual(train % 2, 0)
        self.assertEqual(val % 2, 0)
        self.assertEqual(test % 2, 0)
        self.assertLessEqual(total, 2 * 1527)
        self.assertLessEqual(train, 200)
        self.assertGreater(test, val)

    def test_compute_auto_split_total_cap(self) -> None:
        from live_eval_common import compute_auto_split
        train, val, test = compute_auto_split(90000, 87000, train_cap=200, total_cap=5000)
        total = train + val + test
        self.assertLessEqual(total, 5000)
        self.assertEqual(train % 2, 0)
        self.assertEqual(val % 2, 0)
        self.assertEqual(test % 2, 0)
        self.assertLessEqual(train, 200)

    def test_compute_auto_split_sizes_are_even(self) -> None:
        from live_eval_common import compute_auto_split
        for pos, neg in [(1527, 1669), (1639, 1800), (3200, 3500), (7500, 8000), (87000, 90000)]:
            train, val, test = compute_auto_split(pos, neg)
            self.assertEqual(train % 2, 0, f"train odd for pos={pos},neg={neg}")
            self.assertEqual(val % 2, 0, f"val odd for pos={pos},neg={neg}")
            self.assertEqual(test % 2, 0, f"test odd for pos={pos},neg={neg}")
            self.assertLessEqual(train, 200)
            self.assertGreater(train, 0)
            self.assertGreater(val, 0)
            self.assertGreater(test, 0)

    def test_compute_auto_split_small_dataset(self) -> None:
        from live_eval_common import compute_auto_split
        train, val, test = compute_auto_split(50, 50, train_cap=200, total_cap=5000)
        total = train + val + test
        self.assertLessEqual(total, 100)
        self.assertEqual(train % 2, 0)
        self.assertEqual(val % 2, 0)
        self.assertEqual(test % 2, 0)

    def test_predict_code0_label_falls_back_to_list_input(self) -> None:
        def fn_requires_sequence(x):
            _, _, _, x3, _, _, _, _ = x
            return 1 if float(x3) > 8 else 0

        sample = {
            "x0": -7.77,
            "x1": -11.70,
            "x2": 2.34,
            "x3": 21.39,
            "x4": -20.98,
            "x5": -33.84,
            "x6": 0.06,
            "x7": -1.77,
        }
        pred, mode = self.mod.predict_code0_label(fn_requires_sequence, sample)
        self.assertEqual(pred, 1)
        self.assertIn(mode, {"list", "tuple"})


if __name__ == "__main__":
    unittest.main()
