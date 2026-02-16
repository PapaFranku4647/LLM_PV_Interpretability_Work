from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
import unittest


PROGRAM_SYNTHESIS_DIR = Path(__file__).resolve().parents[1]
if str(PROGRAM_SYNTHESIS_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_SYNTHESIS_DIR))

from code_normalizer import sanitize_generated_code  # noqa: E402


def _build_stub_modules() -> dict[str, types.ModuleType]:
    openai_mod = types.ModuleType("openai")

    class DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    openai_mod.AsyncOpenAI = DummyAsyncOpenAI

    src_mod = types.ModuleType("src")
    src_mod.__path__ = []

    data_handler_mod = types.ModuleType("src.data_handler")
    data_handler_mod.get_data_generator = lambda *args, **kwargs: None
    data_handler_mod.create_stratified_splits = lambda *args, **kwargs: ([], [], [])

    target_functions_mod = types.ModuleType("src.target_functions")
    target_functions_mod.EXPERIMENT_FUNCTION_MAPPING = {}
    target_functions_mod.EXPERIMENT_FUNCTION_METADATA = {}

    prompt_variants_mod = types.ModuleType("prompt_variants")
    prompt_variants_mod.get_prompt_variant_suffix = lambda *_args, **_kwargs: ""

    return {
        "openai": openai_mod,
        "src": src_mod,
        "src.data_handler": data_handler_mod,
        "src.target_functions": target_functions_mod,
        "prompt_variants": prompt_variants_mod,
    }


class RunnerCodeSanitizationTests(unittest.TestCase):
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
        cls.runner = importlib.import_module("runner")

    @classmethod
    def tearDownClass(cls) -> None:
        sys.modules.pop("runner", None)
        for module_name, original in cls._saved_modules.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original
        sys.path[:] = cls._saved_sys_path

    def test_compile_callable_from_code_accepts_commented_code(self) -> None:
        code = '''
        # top
        def f(x):
            """DOC"""
            # local
            return 1 if x.count("1") % 2 else 0  # trailing
        '''
        fn = self.runner.compile_callable_from_code(code)
        self.assertTrue(callable(fn))
        self.assertEqual(fn("1"), 1)
        self.assertEqual(fn("11"), 0)

    def test_predictions_match_between_commented_and_sanitized_equivalent(self) -> None:
        commented_code = '''
        """MODULE_DOC"""
        def f(x):
            """FUNC_DOC"""
            # branch comment
            return 1 if x.count("1") >= 2 else 0
        '''
        sanitized_code = sanitize_generated_code(commented_code)

        fn_comment = self.runner.compile_callable_from_code(commented_code, sanitize=True)
        fn_sanitized = self.runner.compile_callable_from_code(sanitized_code, sanitize=False)
        sample_inputs = ["", "0", "1", "10", "11", "1010", "1110"]
        for x_val in sample_inputs:
            pred_comment = self.runner._normalize_pred_to01(fn_comment(x_val))
            pred_sanitized = self.runner._normalize_pred_to01(fn_sanitized(x_val))
            self.assertEqual(pred_comment, pred_sanitized)

    def test_analyze_code_structure_returns_comment_free_code(self) -> None:
        code = '''
        # TOP_COMMENT
        """MODULE_DOC_UNIQUE"""
        def f(x):
            """FUNC_DOC_UNIQUE"""
            # INLINE_COMMENT
            return 1 if x else 0
        '''
        info = self.runner.analyze_code_structure(code)
        self.assertIsNone(info["code_analysis_error"])
        self.assertEqual(info["code"], sanitize_generated_code(code))
        self.assertNotIn("TOP_COMMENT", info["code"])
        self.assertNotIn("INLINE_COMMENT", info["code"])
        self.assertNotIn("MODULE_DOC_UNIQUE", info["code"])
        self.assertNotIn("FUNC_DOC_UNIQUE", info["code"])

    def test_uncommented_code_path_still_compiles_and_runs(self) -> None:
        code = """
        def f(x):
            return 1 if x and x[0] == "1" else 0
        """
        fn = self.runner.compile_callable_from_code(code)
        info = self.runner.analyze_code_structure(code)
        self.assertEqual(fn("1"), 1)
        self.assertEqual(fn("0"), 0)
        self.assertEqual(fn(""), 0)
        self.assertIsNone(info["code_analysis_error"])
        self.assertIsInstance(info["num_branches"], int)


if __name__ == "__main__":
    unittest.main()
