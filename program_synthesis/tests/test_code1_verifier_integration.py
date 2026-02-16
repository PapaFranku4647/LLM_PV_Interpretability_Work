from __future__ import annotations

import json
import sys
from pathlib import Path
import types
import unittest


PROGRAM_SYNTHESIS_DIR = Path(__file__).resolve().parents[1]
if str(PROGRAM_SYNTHESIS_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_SYNTHESIS_DIR))

from code1_verifier import build_code1_with_verification  # noqa: E402


def _writer_payload(threshold: int) -> str:
    return json.dumps(
        {
            "code1": (
                "def check_conditions(x):\n"
                "    value = x.get('x1', 0) if isinstance(x, dict) else x[1]\n"
                f"    return value > {threshold}"
            )
        }
    )


def _verifier_payload(judgement: str, reason: str, cases: list[dict]) -> str:
    return json.dumps(
        {
            "judgement": judgement,
            "reason": reason,
            "testcases": cases,
        }
    )


def _gold_cases() -> list[dict]:
    return [
        {"sample": {"x1": 7}, "expected": True, "note": "positive"},
        {"sample": {"x1": 6}, "expected": True, "note": "positive"},
        {"sample": [0, 9], "expected": True, "note": "positive"},
        {"sample": {"x1": 1}, "expected": False, "note": "negative"},
        {"sample": {"x1": 5}, "expected": False, "note": "negative"},
        {"sample": [0, 2], "expected": False, "note": "negative"},
    ]


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


class Code1VerifierIntegrationTests(unittest.TestCase):
    def test_accepts_on_first_attempt(self) -> None:
        payloads = [
            _writer_payload(5),
            _verifier_payload("pass", "matches thesis", _gold_cases()),
        ]
        client = _FakeClient(payloads)
        bundle = build_code1_with_verification(
            client=client,
            writer_model="gpt-5-mini",
            verifier_model="gpt-5-mini",
            thesis_conditions="x1 > 5",
            thesis_label=1,
            sample_repr="x1=7",
            retry_once=True,
            execution_timeout_s=0.5,
        )
        self.assertTrue(bundle.accepted)
        self.assertEqual(bundle.attempts, 1)
        self.assertIsNotNone(bundle.final_code1)
        self.assertIsNotNone(bundle.semantic_result)
        self.assertEqual(bundle.semantic_result.judgement, "pass")
        self.assertIsNotNone(bundle.testcase_result)
        self.assertEqual(bundle.testcase_result.failed, 0)

    def test_retries_once_then_accepts(self) -> None:
        payloads = [
            _writer_payload(2),
            _verifier_payload("fail", "too broad", _gold_cases()),
            _writer_payload(5),
            _verifier_payload("pass", "fixed", _gold_cases()),
        ]
        client = _FakeClient(payloads)
        bundle = build_code1_with_verification(
            client=client,
            writer_model="gpt-5-mini",
            verifier_model="gpt-5-mini",
            thesis_conditions="x1 > 5",
            thesis_label=1,
            sample_repr="x1=7",
            retry_once=True,
            execution_timeout_s=0.5,
        )
        self.assertTrue(bundle.accepted)
        self.assertEqual(bundle.attempts, 2)
        self.assertIsNotNone(bundle.testcase_result)
        self.assertEqual(bundle.testcase_result.failed, 0)

    def test_marks_invalid_after_retry_failure(self) -> None:
        payloads = [
            _writer_payload(2),
            _verifier_payload("fail", "wrong", _gold_cases()),
            _writer_payload(2),
            _verifier_payload("fail", "still wrong", _gold_cases()),
        ]
        client = _FakeClient(payloads)
        bundle = build_code1_with_verification(
            client=client,
            writer_model="gpt-5-mini",
            verifier_model="gpt-5-mini",
            thesis_conditions="x1 > 5",
            thesis_label=1,
            sample_repr="x1=7",
            retry_once=True,
            execution_timeout_s=0.5,
        )
        self.assertFalse(bundle.accepted)
        self.assertEqual(bundle.attempts, 2)
        self.assertIsNotNone(bundle.error)
        self.assertIsNotNone(bundle.testcase_result)
        self.assertGreater(bundle.testcase_result.failed, 0)

    def test_handles_malformed_verifier_json_non_fatal(self) -> None:
        payloads = [
            _writer_payload(5),
            "this is not json",
            _writer_payload(5),
            "still not json",
        ]
        client = _FakeClient(payloads)
        bundle = build_code1_with_verification(
            client=client,
            writer_model="gpt-5-mini",
            verifier_model="gpt-5-mini",
            thesis_conditions="x1 > 5",
            thesis_label=1,
            sample_repr="x1=7",
            retry_once=True,
            execution_timeout_s=0.5,
        )
        self.assertFalse(bundle.accepted)
        self.assertEqual(bundle.attempts, 2)
        self.assertIsNotNone(bundle.semantic_result)
        self.assertEqual(bundle.semantic_result.judgement, "uncertain")
        self.assertIsNotNone(bundle.error)


if __name__ == "__main__":
    unittest.main()
