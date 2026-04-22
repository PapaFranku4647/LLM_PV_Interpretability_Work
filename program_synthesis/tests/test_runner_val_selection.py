from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import sys
import tempfile
import types
from pathlib import Path
import unittest


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
    target_functions_mod.EXPERIMENT_FUNCTION_MAPPING = {"fn_a": "parity_all"}
    target_functions_mod.EXPERIMENT_FUNCTION_METADATA = {"fn_a": {"lengths": [20]}}

    prompt_variants_mod = types.ModuleType("prompt_variants")
    prompt_variants_mod.get_prompt_variant_suffix = lambda *_args, **_kwargs: ""

    numpy_mod = types.ModuleType("numpy")

    def _mean(values):
        vals = list(values)
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def _std(values):
        vals = list(values)
        if not vals:
            return 0.0
        m = _mean(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    numpy_mod.mean = _mean
    numpy_mod.std = _std

    return {
        "openai": openai_mod,
        "src": src_mod,
        "src.data_handler": data_handler_mod,
        "src.target_functions": target_functions_mod,
        "prompt_variants": prompt_variants_mod,
        "numpy": numpy_mod,
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

        sys.modules.pop("runner_val_selection", None)
        cls.runner_mod = importlib.import_module("runner_val_selection")

    @classmethod
    def tearDownClass(cls) -> None:
        sys.modules.pop("runner_val_selection", None)
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
        self.assertTrue(self.runner_mod.should_update_best_by_val(0.7, None))
        self.assertTrue(self.runner_mod.should_update_best_by_val(0.8, 0.7))
        self.assertFalse(self.runner_mod.should_update_best_by_val(0.7, 0.8))
        self.assertFalse(self.runner_mod.should_update_best_by_val(0.8, 0.8))
        self.assertFalse(self.runner_mod.should_update_best_by_val(None, 0.8))
        self.assertFalse(self.runner_mod.should_update_best_by_val(None, None))

    def test_chat_completion_reasoning_effort_only_for_openai_models(self) -> None:
        self.assertTrue(self.runner_mod.chat_completion_supports_reasoning_effort("gpt-5.2"))
        self.assertTrue(self.runner_mod.chat_completion_supports_reasoning_effort("protected.gpt-5"))
        self.assertTrue(self.runner_mod.chat_completion_supports_reasoning_effort("o4-mini"))
        self.assertFalse(self.runner_mod.chat_completion_supports_reasoning_effort("protected.gpt-4.1"))
        self.assertFalse(self.runner_mod.chat_completion_supports_reasoning_effort("protected.gpt-4o"))
        self.assertFalse(self.runner_mod.chat_completion_supports_reasoning_effort("protected.gemini-2.0-flash-lite"))
        self.assertFalse(self.runner_mod.chat_completion_supports_reasoning_effort("protected.claude-haiku-4.5"))

    def test_parse_chat_completion_sse_handles_list_content_parts(self) -> None:
        raw = "\n".join([
            'data: {"choices":[{"delta":{"content":[{"type":"text","text":"hello "}]} }]}',
            'data: {"choices":[{"delta":{"content":[{"type":"text","text":"world"}]}}],"usage":{"prompt_tokens":11}}',
            "data: [DONE]",
        ])

        parsed = self.runner_mod.parse_chat_completion_sse(raw)

        self.assertEqual(parsed["text"], "hello world")
        self.assertEqual(parsed["usage"]["prompt_tokens"], 11)

    def test_extract_text_from_chat_completion_handles_list_message_content(self) -> None:
        response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=[
                            {"type": "text", "text": '{"code":"'},
                            {"type": "text", "text": "def f(x): return 0"},
                            {"type": "text", "text": '"}'},
                        ]
                    )
                )
            ]
        )

        text = self.runner_mod.extract_text_from_chat_completion(response)

        self.assertEqual(text, '{"code":"def f(x): return 0"}')

    def test_parse_chat_completion_http_payload_handles_json_message_content(self) -> None:
        payload = json.dumps({
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": '{"code":"def f(x): return 1"}'}
                        ]
                    }
                }
            ],
            "usage": {"completion_tokens": 7},
        })

        parsed = self.runner_mod.parse_chat_completion_http_payload(payload, "application/json")

        self.assertEqual(parsed["text"], '{"code":"def f(x): return 1"}')
        self.assertEqual(parsed["usage"]["completion_tokens"], 7)

    def test_extract_text_from_chat_completion_handles_choice_text(self) -> None:
        payload = {
            "choices": [
                {
                    "text": '{"code":"def f(x): return 2"}'
                }
            ]
        }

        text = self.runner_mod.extract_text_from_chat_completion(payload)

        self.assertEqual(text, '{"code":"def f(x): return 2"}')

    def test_run_selects_best_attempt_using_validation_not_test(self) -> None:
        class StubDatasetStore:
            @staticmethod
            def get(fn, L):
                return (
                    ["TRAIN_MARKER -> 0"],
                    ["VAL_MARKER -> 0"],
                    ["TEST_MARKER -> 0"],
                    False,
                    False,
                )

            @staticmethod
            def derived_seed(fn, L):
                return 12345

        async def fake_call_once(self, fn, L, attempt_idx, data_examples, decimal, tabular=False):
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
                "request_body_bytes": 1,
                "prompt_chars": 1,
                "tool_uses": 0,
                "tool_results_chars": 0,
            }

        def fake_external_get_accuracy(fn_callable, data_lines):
            marker = data_lines[0].split("->", 1)[0].strip()
            attempt_id = int(fn_callable("probe"))
            if marker == "VAL_MARKER":
                return {1: 0.60, 2: 0.80, 3: 0.70}[attempt_id]
            if marker == "TEST_MARKER":
                return {1: 0.90, 2: 0.50, 3: 0.95}[attempt_id]
            if marker == "TRAIN_MARKER":
                return {1: 0.99, 2: 0.98, 3: 0.97}[attempt_id]
            return 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self.runner_mod.Config(api_key="dummy-key")
            cfg.functions = ["fn_a"]
            cfg.lengths = [20]
            cfg.attempts = 3
            cfg.num_trials = 1
            cfg.train_size = 1
            cfg.val_size = 1
            cfg.test_size = 1
            cfg.prompt_variant = "standard"
            cfg.sanitize_generated_code = False
            cfg.dataset_dir = os.path.join(tmpdir, "datasets")
            cfg.out_jsonl = os.path.join(tmpdir, "results.jsonl")
            cfg.out_csv = os.path.join(tmpdir, "results.csv")
            cfg.out_manifest = os.path.join(tmpdir, "results_manifest.json")
            cfg.retry_delay_s = 0.0

            logger = logging.getLogger("runner_val_selection_test")
            logger.handlers[:] = [logging.NullHandler()]
            logger.setLevel(logging.INFO)
            logger.propagate = False

            runner = self.runner_mod.Runner(cfg, logger)
            runner.ds = StubDatasetStore()
            runner._call_once = types.MethodType(fake_call_once, runner)
            self.runner_mod.external_get_accuracy = fake_external_get_accuracy

            rows = asyncio.run(runner.run())
            asyncio.run(runner.client.close())

            self.assertEqual(len(rows), 4)  # 3 attempts + 1 summary

            with open(cfg.out_jsonl, "r", encoding="utf-8") as f:
                final_rows = [json.loads(line) for line in f if line.strip()]

            trial_best_rows = [
                r for r in final_rows
                if (r.get("attempt") is not None) and (not r.get("is_summary"))
            ]
            self.assertEqual(len(trial_best_rows), 1)
            best_row = trial_best_rows[0]
            self.assertEqual(best_row["attempt"], 2)
            self.assertAlmostEqual(float(best_row["val_acc"]), 0.8)
            self.assertAlmostEqual(float(best_row["test_acc"]), 0.5)
            self.assertIn("train_acc", best_row)
            self.assertAlmostEqual(float(best_row["train_acc"]), 0.98)

            summary_rows = [r for r in final_rows if r.get("is_summary")]
            self.assertEqual(len(summary_rows), 1)
            summary = summary_rows[0]
            self.assertAlmostEqual(float(summary["val_acc"]), 0.8)
            self.assertAlmostEqual(float(summary["test_acc"]), 0.5)
            self.assertIn("train_acc", summary)
            self.assertAlmostEqual(float(summary["train_acc"]), 0.98)

    def test_batched_run_selects_by_seen_train_accuracy(self) -> None:
        class StubDatasetStore:
            @staticmethod
            def get(fn, L):
                return (
                    ["B1_A -> 0", "B1_B -> 0", "B2_A -> 0", "B2_B -> 0"],
                    ["VAL_MARKER -> 0"],
                    ["TEST_MARKER -> 0"],
                    False,
                    False,
                )

            @staticmethod
            def derived_seed(fn, L):
                return 12345

        async def fake_call_once(self, fn, L, attempt_idx, data_examples, decimal, tabular=False):
            code = f"def f(x):\n    return {attempt_idx}\n"
            return {
                "fn": fn,
                "length": L,
                "attempt": attempt_idx,
                "prompt": "batch1_prompt",
                "text": json.dumps({"code": code}),
                "usage": {},
                "cached_tokens": 0,
                "duration_ms": 1,
                "request_body_bytes": 1,
                "prompt_chars": 1,
                "tool_uses": 0,
                "tool_results_chars": 0,
            }

        async def fake_call_with_prompt(self, fn, L, attempt_idx, prompt_text):
            code = f"def f(x):\n    return {attempt_idx}\n"
            return {
                "fn": fn,
                "length": L,
                "attempt": attempt_idx,
                "prompt": prompt_text,
                "text": json.dumps({"code": code}),
                "usage": {},
                "cached_tokens": 0,
                "duration_ms": 1,
                "request_body_bytes": 1,
                "prompt_chars": len(prompt_text),
                "tool_uses": 0,
                "tool_results_chars": 0,
            }

        def fake_external_get_accuracy(fn_callable, data_lines):
            markers = tuple(line.split("->", 1)[0].strip() for line in data_lines)
            attempt_id = int(fn_callable("probe"))
            if markers == ("B1_A", "B1_B"):
                return {1: 0.80, 2: 0.70}.get(attempt_id, 0.0)
            if markers == ("B2_A", "B2_B"):
                return {1: 0.40, 3: 0.90, 4: 1.00}.get(attempt_id, 0.0)
            if markers == ("B1_A", "B1_B", "B2_A", "B2_B"):
                return {1: 0.60, 3: 0.70, 4: 0.65}.get(attempt_id, 0.0)
            if markers == ("VAL_MARKER",):
                return {1: 0.55, 2: 0.95, 3: 0.20, 4: 0.90}.get(attempt_id, 0.0)
            if markers == ("TEST_MARKER",):
                return {1: 0.50, 2: 0.30, 3: 0.40, 4: 0.80}.get(attempt_id, 0.0)
            return 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self.runner_mod.Config(api_key="dummy-key")
            cfg.functions = ["fn_a"]
            cfg.lengths = [20]
            cfg.attempts = 2
            cfg.num_trials = 1
            cfg.train_size = 4
            cfg.val_size = 1
            cfg.test_size = 1
            cfg.prompt_variant = "standard"
            cfg.sanitize_generated_code = False
            cfg.dataset_dir = os.path.join(tmpdir, "datasets")
            cfg.out_jsonl = os.path.join(tmpdir, "results.jsonl")
            cfg.out_csv = os.path.join(tmpdir, "results.csv")
            cfg.out_manifest = os.path.join(tmpdir, "results_manifest.json")
            cfg.retry_delay_s = 0.0
            cfg.code0_train_mode = "batched"
            cfg.code0_batch_size = 2

            logger = logging.getLogger("runner_val_selection_batch_test")
            logger.handlers[:] = [logging.NullHandler()]
            logger.setLevel(logging.INFO)
            logger.propagate = False

            runner = self.runner_mod.Runner(cfg, logger)
            runner.ds = StubDatasetStore()
            runner._call_once = types.MethodType(fake_call_once, runner)
            runner._call_with_prompt = types.MethodType(fake_call_with_prompt, runner)
            self.runner_mod.external_get_accuracy = fake_external_get_accuracy

            rows = asyncio.run(runner.run())
            asyncio.run(runner.client.close())

            self.assertEqual(len(rows), 6)  # 5 candidate rows + 1 summary
            self.assertEqual(len(runner.batch_winner_rows), 2)
            self.assertEqual(runner.batch_winner_rows[0]["attempt"], 1)
            self.assertTrue(runner.batch_winner_rows[0]["selected_for_next_batch"])
            self.assertEqual(runner.batch_winner_rows[1]["attempt"], 3)
            self.assertTrue(runner.batch_winner_rows[1]["is_final_selected_model"])
            self.assertAlmostEqual(float(runner.batch_winner_rows[1]["train_acc"]), 0.70)
            self.assertAlmostEqual(float(runner.batch_winner_rows[1]["current_batch_train_acc"]), 0.90)

            with open(cfg.out_jsonl, "r", encoding="utf-8") as f:
                final_rows = [json.loads(line) for line in f if line.strip()]

            best_rows = [r for r in final_rows if not r.get("is_summary")]
            self.assertEqual(len(best_rows), 1)
            best_row = best_rows[0]
            self.assertEqual(best_row["attempt"], 3)
            self.assertEqual(best_row["batch_index"], 2)
            self.assertTrue(best_row["is_final_selected_model"])

            summary = [r for r in final_rows if r.get("is_summary")][0]
            self.assertAlmostEqual(float(summary["train_acc"]), 0.70)
            self.assertAlmostEqual(float(summary["val_acc"]), 0.20)
            self.assertAlmostEqual(float(summary["test_acc"]), 0.40)

    def test_batched_run_can_keep_incumbent_as_final_model(self) -> None:
        class StubDatasetStore:
            @staticmethod
            def get(fn, L):
                return (
                    ["B1_A -> 0", "B1_B -> 0", "B2_A -> 0", "B2_B -> 0"],
                    ["VAL_MARKER -> 0"],
                    ["TEST_MARKER -> 0"],
                    False,
                    False,
                )

            @staticmethod
            def derived_seed(fn, L):
                return 12345

        async def fake_call_once(self, fn, L, attempt_idx, data_examples, decimal, tabular=False):
            code = f"def f(x):\n    return {attempt_idx}\n"
            return {
                "fn": fn,
                "length": L,
                "attempt": attempt_idx,
                "prompt": "batch1_prompt",
                "text": json.dumps({"code": code}),
                "usage": {},
                "cached_tokens": 0,
                "duration_ms": 1,
                "request_body_bytes": 1,
                "prompt_chars": 1,
                "tool_uses": 0,
                "tool_results_chars": 0,
            }

        async def fake_call_with_prompt(self, fn, L, attempt_idx, prompt_text):
            code = f"def f(x):\n    return {attempt_idx}\n"
            return {
                "fn": fn,
                "length": L,
                "attempt": attempt_idx,
                "prompt": prompt_text,
                "text": json.dumps({"code": code}),
                "usage": {},
                "cached_tokens": 0,
                "duration_ms": 1,
                "request_body_bytes": 1,
                "prompt_chars": len(prompt_text),
                "tool_uses": 0,
                "tool_results_chars": 0,
            }

        def fake_external_get_accuracy(fn_callable, data_lines):
            markers = tuple(line.split("->", 1)[0].strip() for line in data_lines)
            attempt_id = int(fn_callable("probe"))
            if markers == ("B1_A", "B1_B"):
                return {1: 0.90, 2: 0.80}.get(attempt_id, 0.0)
            if markers == ("B2_A", "B2_B"):
                return {1: 0.80, 3: 1.00, 4: 0.70}.get(attempt_id, 0.0)
            if markers == ("B1_A", "B1_B", "B2_A", "B2_B"):
                return {1: 0.85, 3: 0.80, 4: 0.84}.get(attempt_id, 0.0)
            if markers == ("VAL_MARKER",):
                return {1: 0.50, 2: 0.40, 3: 0.90, 4: 0.95}.get(attempt_id, 0.0)
            if markers == ("TEST_MARKER",):
                return {1: 0.51, 2: 0.41, 3: 0.91, 4: 0.96}.get(attempt_id, 0.0)
            return 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self.runner_mod.Config(api_key="dummy-key")
            cfg.functions = ["fn_a"]
            cfg.lengths = [20]
            cfg.attempts = 2
            cfg.num_trials = 1
            cfg.train_size = 4
            cfg.val_size = 1
            cfg.test_size = 1
            cfg.prompt_variant = "standard"
            cfg.sanitize_generated_code = False
            cfg.dataset_dir = os.path.join(tmpdir, "datasets")
            cfg.out_jsonl = os.path.join(tmpdir, "results.jsonl")
            cfg.out_csv = os.path.join(tmpdir, "results.csv")
            cfg.out_manifest = os.path.join(tmpdir, "results_manifest.json")
            cfg.retry_delay_s = 0.0
            cfg.code0_train_mode = "batched"
            cfg.code0_batch_size = 2

            logger = logging.getLogger("runner_val_selection_batch_incumbent_test")
            logger.handlers[:] = [logging.NullHandler()]
            logger.setLevel(logging.INFO)
            logger.propagate = False

            runner = self.runner_mod.Runner(cfg, logger)
            runner.ds = StubDatasetStore()
            runner._call_once = types.MethodType(fake_call_once, runner)
            runner._call_with_prompt = types.MethodType(fake_call_with_prompt, runner)
            self.runner_mod.external_get_accuracy = fake_external_get_accuracy

            rows = asyncio.run(runner.run())
            asyncio.run(runner.client.close())

            self.assertEqual(len(rows), 6)
            self.assertEqual(len(runner.batch_winner_rows), 2)
            final_winner = runner.batch_winner_rows[-1]
            self.assertEqual(final_winner["candidate_source"], "incumbent")
            self.assertIsNone(final_winner["attempt"])
            self.assertTrue(final_winner["is_final_selected_model"])
            self.assertAlmostEqual(float(final_winner["train_acc"]), 0.85)

            with open(cfg.out_jsonl, "r", encoding="utf-8") as f:
                final_rows = [json.loads(line) for line in f if line.strip()]
            best_row = [r for r in final_rows if not r.get("is_summary")][0]
            self.assertEqual(best_row["candidate_source"], "incumbent")
            self.assertIsNone(best_row["attempt"])
            self.assertTrue(best_row["is_final_selected_model"])


if __name__ == "__main__":
    unittest.main()
