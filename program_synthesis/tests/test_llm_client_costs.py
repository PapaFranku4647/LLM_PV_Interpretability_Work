from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROGRAM_SYNTHESIS_DIR = Path(__file__).resolve().parents[1]
if str(PROGRAM_SYNTHESIS_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_SYNTHESIS_DIR))

from llm_client import (  # noqa: E402
    build_chat_completions_body,
    estimate_usage_cost,
    merge_cost_estimates,
    merge_usage,
    resolve_model_pricing,
    should_use_raw_chat_completions,
)


class LLMClientCostTests(unittest.TestCase):
    def test_resolve_model_pricing_handles_protected_tamu_model(self) -> None:
        pricing = resolve_model_pricing("protected.gemini-2.0-flash-lite")
        self.assertIsNotNone(pricing)
        assert pricing is not None
        self.assertEqual(pricing.canonical_name, "gemini-2.0-flash-lite")
        self.assertAlmostEqual(pricing.input_per_million, 0.075, places=8)
        self.assertAlmostEqual(pricing.output_per_million, 0.30, places=8)

    def test_estimate_usage_cost_uses_catalog_rates(self) -> None:
        cost = estimate_usage_cost(
            {"prompt_tokens": 1000, "completion_tokens": 2000},
            "protected.gemini-2.0-flash-lite",
        )
        self.assertAlmostEqual(float(cost["estimated_input_cost_usd"]), 0.000075, places=12)
        self.assertAlmostEqual(float(cost["estimated_output_cost_usd"]), 0.000600, places=12)
        self.assertAlmostEqual(float(cost["estimated_total_cost_usd"]), 0.000675, places=12)

    def test_merge_usage_sums_normalized_keys_and_details(self) -> None:
        merged = merge_usage(
            [
                {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 3}},
                {"input_tokens": 5, "output_tokens": 7, "completion_tokens_details": {"reasoning_tokens": 2}},
            ]
        )
        self.assertEqual(merged["prompt_tokens"], 15)
        self.assertEqual(merged["completion_tokens"], 7)
        self.assertEqual(merged["cached_tokens"], 3)
        self.assertEqual(merged["reasoning_tokens"], 2)

    def test_merge_cost_estimates_sums_mixed_models(self) -> None:
        first = estimate_usage_cost({"prompt_tokens": 1000, "completion_tokens": 1000}, "protected.gemini-2.0-flash-lite")
        second = estimate_usage_cost({"prompt_tokens": 2000, "completion_tokens": 1000}, "protected.gpt-5")
        merged = merge_cost_estimates([first, second])
        expected = float(first["estimated_total_cost_usd"]) + float(second["estimated_total_cost_usd"])
        self.assertAlmostEqual(float(merged["estimated_total_cost_usd"]), expected, places=12)
        self.assertIn("gemini-2.0-flash-lite", str(merged["pricing_model"]))
        self.assertIn("gpt-5", str(merged["pricing_model"]))

    def test_build_chat_completions_body_uses_azure_shape(self) -> None:
        body = build_chat_completions_body(
            model="gpt-5.2-deep-learning-fundamentals",
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=32,
            reasoning_effort="minimal",
            azure_endpoint="https://example.openai.azure.com/",
        )
        self.assertEqual(body["max_completion_tokens"], 32)
        self.assertNotIn("max_tokens", body)

    def test_should_use_raw_chat_completions_is_disabled_for_azure(self) -> None:
        self.assertFalse(
            should_use_raw_chat_completions(
                api_base_url="https://example.openai.azure.com/",
                azure_endpoint="",
            )
        )
        self.assertFalse(
            should_use_raw_chat_completions(
                api_base_url="https://chat-api.tamu.ai/api",
                azure_endpoint="https://example.openai.azure.com/",
            )
        )
        self.assertTrue(
            should_use_raw_chat_completions(
                api_base_url="https://chat-api.tamu.ai/api",
                azure_endpoint="",
            )
        )


if __name__ == "__main__":
    unittest.main()
