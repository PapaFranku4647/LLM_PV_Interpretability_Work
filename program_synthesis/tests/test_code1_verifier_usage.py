from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path


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


def _verifier_payload() -> str:
    return json.dumps(
        {
            "judgement": "pass",
            "reason": "matches thesis",
            "testcases": [
                {"sample": {"x1": 7}, "expected": True, "note": "positive"},
                {"sample": {"x1": 1}, "expected": False, "note": "negative"},
            ],
        }
    )


class _FakeResponses:
    def __init__(self, payloads: list[tuple[str, dict[str, int]]]) -> None:
        self._payloads = list(payloads)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._payloads:
            raise AssertionError("Unexpected responses.create call")
        text, usage = self._payloads.pop(0)
        return types.SimpleNamespace(
            output_text=text,
            usage=usage,
            id=f"resp_{len(self.calls)}",
            status="completed",
        )


class _FakeClient:
    def __init__(self, payloads: list[tuple[str, dict[str, int]]]) -> None:
        self.responses = _FakeResponses(payloads)


class Code1VerifierUsageTests(unittest.TestCase):
    def test_build_code1_with_verification_tracks_usage_and_cost(self) -> None:
        client = _FakeClient(
            [
                (_writer_payload(5), {"prompt_tokens": 100, "completion_tokens": 20}),
                (_verifier_payload(), {"prompt_tokens": 80, "completion_tokens": 30}),
            ]
        )
        bundle = build_code1_with_verification(
            client=client,
            writer_model="protected.gemini-2.0-flash-lite",
            verifier_model="protected.gemini-2.0-flash-lite",
            thesis_conditions="x1 > 5",
            thesis_label=1,
            sample_repr="x1=7",
            retry_once=False,
            execution_timeout_s=0.5,
        )
        self.assertTrue(bundle.accepted)
        self.assertEqual(bundle.writer_usage["prompt_tokens"], 100)
        self.assertEqual(bundle.writer_usage["completion_tokens"], 20)
        self.assertEqual(bundle.verifier_usage["prompt_tokens"], 80)
        self.assertEqual(bundle.verifier_usage["completion_tokens"], 30)
        self.assertEqual(bundle.usage_total["prompt_tokens"], 180)
        self.assertEqual(bundle.usage_total["completion_tokens"], 50)
        self.assertIsNotNone(bundle.writer_cost["estimated_total_cost_usd"])
        self.assertIsNotNone(bundle.verifier_cost["estimated_total_cost_usd"])
        expected_total = (
            float(bundle.writer_cost["estimated_total_cost_usd"])
            + float(bundle.verifier_cost["estimated_total_cost_usd"])
        )
        self.assertAlmostEqual(float(bundle.cost_total["estimated_total_cost_usd"]), expected_total, places=12)


if __name__ == "__main__":
    unittest.main()
