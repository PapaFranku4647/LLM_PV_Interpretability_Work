from __future__ import annotations

import os
import sys
import unittest


PROGRAM_SYNTHESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROGRAM_SYNTHESIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PROGRAM_SYNTHESIS_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_SYNTHESIS_DIR)

from program_synthesis import baseline_runner  # noqa: E402


class TabularDataParserTests(unittest.TestCase):
    def test_parser_one_hot_encodes_semantic_rows(self) -> None:
        lines = [
            "HighBP:yes,BMI:high,Age:medium -> 1",
            "HighBP:no,BMI:low,Age:low -> 0",
            "HighBP:yes,BMI:very high,Age:high -> 1",
        ]

        parser = baseline_runner.TabularDataParser("cdc_diabetes")
        X, y = parser.fit_transform(lines)
        X2, y2 = parser.transform(["HighBP:no,BMI:medium,Age:low -> 0"])

        self.assertEqual(y.tolist(), [1, 0, 1])
        self.assertEqual(y2.tolist(), [0])
        self.assertEqual(parser.numeric_features, [])
        self.assertIn("HighBP=yes", parser.output_feature_names)
        self.assertIn("BMI=high", parser.output_feature_names)
        self.assertEqual(X.shape[1], X2.shape[1])

    def test_parser_keeps_numeric_columns_numeric(self) -> None:
        lines = [
            "x0:1.5,x1:c0 -> 1",
            "x0:-2.0,x1:c1 -> 0",
        ]

        parser = baseline_runner.TabularDataParser("cdc_diabetes")
        X, y = parser.fit_transform(lines)

        self.assertEqual(y.tolist(), [1, 0])
        self.assertEqual(parser.numeric_features, ["x0"])
        self.assertEqual(parser.categorical_features, ["x1"])
        self.assertAlmostEqual(X[0, 0], 1.5)
        self.assertAlmostEqual(X[1, 0], -2.0)


class BenchmarkRunnerTests(unittest.TestCase):
    def test_benchmark_runner_uses_validation_selected_model(self) -> None:
        class StubDatasetStore:
            @staticmethod
            def get(fn: str, length: int):
                del fn, length
                return (
                    [
                        "x0:0,x1:c0 -> 0",
                        "x0:1,x1:c0 -> 0",
                        "x0:2,x1:c1 -> 1",
                        "x0:3,x1:c1 -> 1",
                    ],
                    [
                        "x0:0.5,x1:c0 -> 0",
                        "x0:2.5,x1:c1 -> 1",
                    ],
                    [
                        "x0:0.25,x1:c0 -> 0",
                        "x0:2.75,x1:c1 -> 1",
                    ],
                )

        tmpdir = os.path.join(PROGRAM_SYNTHESIS_DIR, "tests", "_tmp_baseline_runner")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            cfg = baseline_runner.Config()
            cfg.functions = ["fn_o"]
            cfg.models = ["decision_tree"]
            cfg.num_trials = 1
            cfg.train_size = 4
            cfg.val_size = 2
            cfg.test_size = 2
            cfg.out_jsonl = os.path.join(tmpdir, "baseline.jsonl")
            cfg.out_csv = os.path.join(tmpdir, "baseline.csv")

            runner = baseline_runner.BenchmarkRunner(cfg)
            runner.ds = StubDatasetStore()
            rows = runner.run()
            baseline_runner.write_csv(cfg.out_csv, rows)
        finally:
            for name in ("baseline.jsonl", "baseline.csv"):
                path = os.path.join(tmpdir, name)
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(tmpdir):
                os.rmdir(tmpdir)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["model"], "decision_tree")
        self.assertEqual(rows[0]["selection_split"], "val")
        self.assertAlmostEqual(rows[0]["test_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
