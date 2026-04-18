from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import shutil
import sys
import types
import unittest
from pathlib import Path


PROGRAM_SYNTHESIS_DIR = Path(__file__).resolve().parents[1]
if str(PROGRAM_SYNTHESIS_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_SYNTHESIS_DIR))


def _build_stub_modules() -> dict[str, types.ModuleType]:
    openai_mod = types.ModuleType("openai")

    class DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        async def close(self) -> None:
            return None

    class DummyAsyncAzureOpenAI(DummyAsyncOpenAI):
        pass

    openai_mod.AsyncOpenAI = DummyAsyncOpenAI
    openai_mod.AsyncAzureOpenAI = DummyAsyncAzureOpenAI

    src_mod = types.ModuleType("src")
    src_mod.__path__ = []

    data_handler_mod = types.ModuleType("src.data_handler")
    data_handler_mod.get_data_generator = lambda *args, **kwargs: None
    data_handler_mod.create_stratified_splits = lambda *args, **kwargs: ([], [], [])

    target_functions_mod = types.ModuleType("src.target_functions")
    target_functions_mod.EXPERIMENT_FUNCTION_MAPPING = {"fn_o": "cdc_diabetes"}
    target_functions_mod.EXPERIMENT_FUNCTION_METADATA = {"fn_o": {"lengths": [21]}}

    return {
        "openai": openai_mod,
        "src": src_mod,
        "src.data_handler": data_handler_mod,
        "src.target_functions": target_functions_mod,
    }


class RunnerValSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_sys_path = list(sys.path)
        if str(PROGRAM_SYNTHESIS_DIR) not in sys.path:
            sys.path.insert(0, str(PROGRAM_SYNTHESIS_DIR))

        cls._saved_modules = {}
        for module_name, module_obj in _build_stub_modules().items():
            cls._saved_modules[module_name] = sys.modules.get(module_name)
            sys.modules[module_name] = module_obj

        sys.modules.pop("runner", None)
        cls.runner_mod = importlib.import_module("runner")

    @classmethod
    def tearDownClass(cls) -> None:
        sys.modules.pop("runner", None)
        for module_name, original in cls._saved_modules.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original
        sys.path[:] = cls._saved_sys_path

    def setUp(self) -> None:
        self._saved_external_acc = self.runner_mod.external_get_accuracy

    def tearDown(self) -> None:
        self.runner_mod.external_get_accuracy = self._saved_external_acc

    def test_should_update_best_by_val(self) -> None:
        self.assertTrue(self.runner_mod.should_update_best_by_val(0.70, None))
        self.assertTrue(self.runner_mod.should_update_best_by_val(0.80, 0.70))
        self.assertFalse(self.runner_mod.should_update_best_by_val(0.70, 0.80))
        self.assertFalse(self.runner_mod.should_update_best_by_val(0.80, 0.80))
        self.assertFalse(self.runner_mod.should_update_best_by_val(None, 0.80))
        self.assertFalse(self.runner_mod.should_update_best_by_val(None, None))

    def test_run_selects_best_attempt_using_validation_not_test(self) -> None:
        class StubDatasetStore:
            @staticmethod
            def get(fn, L):
                return (
                    ["TRAIN_MARKER -> 0"],
                    ["VAL_MARKER -> 0"],
                    ["TEST_MARKER -> 0"],
                    False,
                    True,
                )

        async def fake_call_once(self, fn, L, attempt_idx, data_examples, decimal=False, tabular=False):
            code = f"def f(x):\n    return {attempt_idx}\n"
            return {
                "fn": fn,
                "length": L,
                "attempt": attempt_idx,
                "prompt": "fake_prompt",
                "text": json.dumps({"code": code}),
                "usage": {},
                "cached_tokens": 0,
                "duration_ms": 1,
                "tool_uses": 0,
                "tool_results_chars": 0,
            }

        def fake_external_get_accuracy(fn_callable, data_lines, *_args, **_kwargs):
            marker = data_lines[0].split("->", 1)[0].strip()
            attempt_id = int(fn_callable("probe"))
            if marker == "VAL_MARKER":
                return {1: 0.60, 2: 0.80, 3: 0.70}[attempt_id]
            if marker == "TEST_MARKER":
                return {1: 0.90, 2: 0.50, 3: 0.95}[attempt_id]
            if marker == "TRAIN_MARKER":
                return {1: 0.99, 2: 0.98, 3: 0.97}[attempt_id]
            return 0.0

        tmpdir = PROGRAM_SYNTHESIS_DIR / "tests" / "_tmp_runner_val_selection"
        os.makedirs(tmpdir, exist_ok=True)
        try:
            cfg = self.runner_mod.Config(api_key="dummy-key")
            cfg.functions = ["fn_o"]
            cfg.lengths = [21]
            cfg.lengths_explicit = True
            cfg.attempts = 3
            cfg.num_trials = 1
            cfg.train_size = 1
            cfg.val_size = 1
            cfg.test_size = 1
            cfg.out_jsonl = os.path.join(tmpdir, "results.jsonl")
            cfg.out_csv = os.path.join(tmpdir, "results.csv")
            cfg.concurrency = 1
            cfg.allow_tools = False
            cfg.tool_choice = "none"

            logger = logging.getLogger("runner_val_selection_test")
            logger.handlers[:] = [logging.NullHandler()]
            logger.setLevel(logging.INFO)
            logger.propagate = False

            runner = self.runner_mod.Runner(cfg, logger)
            runner.ds = StubDatasetStore()
            runner._call_once = types.MethodType(fake_call_once, runner)
            self.runner_mod.external_get_accuracy = fake_external_get_accuracy

            rows = asyncio.run(runner.run())
            asyncio.run(runner.aclose())

            self.assertEqual(len(rows), 4)

            with open(cfg.out_jsonl, "r", encoding="utf-8") as handle:
                final_rows = [json.loads(line) for line in handle if line.strip()]

            best_rows = [r for r in final_rows if (r.get("attempt") is not None) and (not r.get("is_summary"))]
            self.assertEqual(len(best_rows), 1)
            best_row = best_rows[0]
            self.assertEqual(best_row["attempt"], 2)
            self.assertAlmostEqual(float(best_row["val_acc"]), 0.80)
            self.assertAlmostEqual(float(best_row["test_acc"]), 0.50)

            summary_rows = [r for r in final_rows if r.get("is_summary")]
            self.assertEqual(len(summary_rows), 1)
            summary = summary_rows[0]
            self.assertAlmostEqual(float(summary["val_acc"]), 0.80)
            self.assertAlmostEqual(float(summary["test_acc"]), 0.50)
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
