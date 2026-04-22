from __future__ import annotations

import sys
from pathlib import Path
import types
import unittest


PROGRAM_SYNTHESIS_DIR = Path(__file__).resolve().parents[1]
if str(PROGRAM_SYNTHESIS_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_SYNTHESIS_DIR))

from code1_verifier import (  # noqa: E402
    TestCase,
    _build_code1_verifier_prompt,
    _build_code1_writer_prompt,
    _detect_categorical_features,
    compile_code1,
    generate_code1_from_thesis,
    verify_code1_with_testcases,
)


class _FakeResponses:
    def __init__(self, payloads: list[str]) -> None:
        self._payloads = list(payloads)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._payloads:
            raise AssertionError("Unexpected responses.create call")
        text = self._payloads.pop(0)
        return types.SimpleNamespace(output_text=text)


class _FakeClient:
    def __init__(self, payloads: list[str]) -> None:
        self.responses = _FakeResponses(payloads)


class Code1VerifierUnitTests(unittest.TestCase):
    def test_compile_code1_accepts_valid_function(self) -> None:
        code = """
        def check_conditions(x):
            value = x.get("x1", 0) if isinstance(x, dict) else x[1]
            return value > 3
        """
        fn, err = compile_code1(code)
        self.assertIsNone(err)
        self.assertTrue(callable(fn))
        self.assertTrue(fn({"x1": 9}))
        self.assertFalse(fn({"x1": 1}))

    def test_compile_code1_rejects_disallowed_call(self) -> None:
        code = """
        def check_conditions(x):
            return bool(open("bad.txt"))
        """
        fn, err = compile_code1(code)
        self.assertIsNone(fn)
        self.assertIsNotNone(err)
        self.assertIn("disallowed_call", err)

    def test_compile_code1_allows_nested_function_helper(self) -> None:
        code = """
        def check_conditions(x):
            def helper(v):
                return v > 3
            return helper(x.get("x1", 0))
        """
        fn, err = compile_code1(code)
        self.assertIsNone(err)
        self.assertTrue(callable(fn))
        self.assertTrue(fn({"x1": 9}))
        self.assertFalse(fn({"x1": 1}))

    def test_verify_code1_with_testcases_accepts_boolish_returns(self) -> None:
        code = """
        def check_conditions(x):
            value = x.get("x1", 0) if isinstance(x, dict) else x[1]
            return 1 if value >= 5 else 0
        """
        fn, err = compile_code1(code)
        self.assertIsNone(err)
        self.assertTrue(callable(fn))

        testcases = [
            TestCase(sample={"x1": 7}, expected=True, note="positive"),
            TestCase(sample={"x1": 2}, expected=False, note="negative"),
        ]
        result = verify_code1_with_testcases(fn, testcases, execution_timeout_s=0.5)
        self.assertEqual(result.total, 2)
        self.assertEqual(result.passed, 2)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.mismatches, [])

    def test_compile_code1_allows_try_except_fallback(self) -> None:
        code = """
        def check_conditions(x):
            try:
                v = x[1]
            except Exception:
                v = x.get("x1")
            return bool(v is not None and v > 3)
        """
        fn, err = compile_code1(code)
        self.assertIsNone(err)
        self.assertTrue(callable(fn))
        self.assertTrue(fn([0, 9]))
        self.assertFalse(fn({"x1": 2}))

    def test_compile_code1_categorical_string_comparison(self) -> None:
        code = """
        def check_conditions(x):
            return x.get("x5") == 'c4' and x.get("x0", 0) >= 40.0
        """
        fn, err = compile_code1(code)
        self.assertIsNone(err)
        self.assertTrue(callable(fn))
        self.assertTrue(fn({"x0": 50.0, "x5": "c4"}))
        self.assertFalse(fn({"x0": 50.0, "x5": "c1"}))
        self.assertFalse(fn({"x0": 30.0, "x5": "c4"}))

    def test_detect_categorical_features(self) -> None:
        sample = "x0=50.0, x1=c0, x5=c4, x10=3.2"
        cats = _detect_categorical_features(sample)
        self.assertIn("x1", cats)
        self.assertIn("x5", cats)
        self.assertNotIn("x0", cats)
        self.assertNotIn("x10", cats)

    def test_detect_categorical_features_none(self) -> None:
        sample = "x0=50.0, x1=3.2"
        cats = _detect_categorical_features(sample)
        self.assertEqual(cats, [])

    def test_writer_prompt_preserves_semantic_feature_names(self) -> None:
        prompt = _build_code1_writer_prompt(
            thesis_conditions="HighBP == 'yes' AND BMI == 'very high'",
            thesis_label=1,
            sample_repr="HighBP=yes, BMI=very high, Age=low",
        )
        self.assertIn('x.get("HighBP")', prompt)
        self.assertIn('x.get("BMI")', prompt)
        self.assertIn("Do NOT rename semantic features to x0/x1", prompt)

    def test_verifier_prompt_preserves_semantic_feature_names(self) -> None:
        prompt = _build_code1_verifier_prompt(
            thesis_conditions="HighBP == 'yes' AND BMI == 'very high'",
            thesis_label=1,
            code1="def check_conditions(x):\n    return x.get('HighBP') == 'yes'",
            sample_repr="HighBP=yes, BMI=very high, Age=low",
        )
        self.assertIn("Reference sample format", prompt)
        self.assertIn("HighBP=yes", prompt)
        self.assertIn("Use the EXACT feature keys", prompt)
        self.assertIn("Do NOT rename semantic features to x0/x1", prompt)

    def test_generate_code1_handles_malformed_writer_json(self) -> None:
        client = _FakeClient(["not-json-at-all"])
        generation = generate_code1_from_thesis(
            client=client,
            model="gpt-5-mini",
            thesis_conditions="x1 > 5",
            thesis_label=1,
            sample_repr="x1=7",
        )
        self.assertFalse(generation.compile_ok)
        self.assertIsNone(generation.code1)
        self.assertIsNotNone(generation.compile_error)
        self.assertIn("writer_json_parse_error", generation.compile_error)


if __name__ == "__main__":
    unittest.main()
