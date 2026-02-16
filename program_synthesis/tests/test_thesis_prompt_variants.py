from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROGRAM_SYNTHESIS_DIR = Path(__file__).resolve().parents[1]
if str(PROGRAM_SYNTHESIS_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_SYNTHESIS_DIR))

from prompt_variants import (  # noqa: E402
    THESIS_GENERATION_TEMPLATE,
    build_thesis_generation_prompt,
    format_sample_for_thesis_prompt,
)


class ThesisPromptVariantsTests(unittest.TestCase):
    def test_build_prompt_includes_code_once(self) -> None:
        code = "def f(x):\n    return 1  # CODE_UNIQUE_TOKEN"
        sample = "x0=1, x1=2"
        prompt = build_thesis_generation_prompt(code, sample, 1)
        self.assertEqual(prompt.count("CODE_UNIQUE_TOKEN"), 1)

    def test_build_prompt_includes_sample_and_label(self) -> None:
        prompt = build_thesis_generation_prompt("def f(x):\n    return 0", "x0=1, x1=2", 1)
        self.assertIn("Sample: [x0=1, x1=2]", prompt)
        self.assertIn("Your code classified it as: 1", prompt)
        self.assertNotIn("[CODE0]", prompt)
        self.assertNotIn("[SAMPLE]", prompt)
        self.assertNotIn("[LABEL]", prompt)

    def test_build_prompt_includes_required_json_contract(self) -> None:
        prompt = build_thesis_generation_prompt("def f(x):\n    return 1", "x0=1", 1)
        self.assertIn('"conditions": "x1 > 5 AND x3 < 2"', prompt)
        self.assertIn('"label": 1', prompt)

    def test_build_prompt_includes_step22_requirement_lines(self) -> None:
        prompt = build_thesis_generation_prompt("def f(x):\n    return 1", "x0=1", 1)
        required_snippets = [
            "- The conditions must be SHORT and human-interpretable",
            "- The conditions must hold for the given sample",
            "- The conditions must reflect what your CODE actually does",
            "- The conditions should define a MEANINGFUL subset of the data",
            "- Every sample that satisfies these conditions should be classified",
            "- The subset should be as LARGE as possible while maintaining accuracy",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, prompt)

    def test_format_sample_mapping_orders_feature_keys_numerically(self) -> None:
        sample = {"x10": 10, "x2": 2, "foo": "bar", "x1": 1}
        formatted = format_sample_for_thesis_prompt(sample)
        self.assertEqual(formatted, "x1=1, x2=2, x10=10, foo=bar")

    def test_format_sample_string_passthrough(self) -> None:
        formatted = format_sample_for_thesis_prompt("  x0=1, x1=2  ")
        self.assertEqual(formatted, "x0=1, x1=2")

    def test_format_sample_is_deterministic(self) -> None:
        sample_a = {"x2": 2, "x1": 1, "foo": "bar"}
        sample_b = {"foo": "bar", "x1": 1, "x2": 2}
        self.assertEqual(
            format_sample_for_thesis_prompt(sample_a),
            format_sample_for_thesis_prompt(sample_b),
        )

    def test_template_contains_expected_placeholders(self) -> None:
        self.assertIn("[CODE0]", THESIS_GENERATION_TEMPLATE)
        self.assertIn("[SAMPLE]", THESIS_GENERATION_TEMPLATE)
        self.assertIn("[LABEL]", THESIS_GENERATION_TEMPLATE)


if __name__ == "__main__":
    unittest.main()
