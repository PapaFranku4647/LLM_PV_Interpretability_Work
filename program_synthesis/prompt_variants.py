"""Prompt suffix variants for interpretability-aware code generation."""

from typing import Dict


STANDARD_SUFFIX = ""

EXPLAIN_SUFFIX = """
IMPORTANT: Generate code that is clear and interpretable. Each condition
and threshold should reflect a meaningful pattern in the training data.
After you generate this code, I will ask you to explain and justify each
decision in the code based on statistical properties of the data.
""".strip()

INTERVIEW_SUFFIX = """
IMPORTANT: After generating this code, I will conduct an interview where
I ask you to:
1. Explain why specific samples are classified the way they are
2. Justify each threshold and condition with data statistics
3. Describe which features matter most and why
4. Predict what would happen if parts of the code were removed
Write code that enables clear, data-grounded answers.
""".strip()

PREVIEW_SUFFIX = """
IMPORTANT: After generating the function, you will be asked:
- Given sample X mapped to y=1, why? Explain from the data distribution.
- Why threshold T for feature xi? What % of each class crosses it?
- What fraction of training examples satisfy condition C?
Generate code that enables clear, data-grounded answers.
""".strip()


PROMPT_VARIANT_SUFFIXES: Dict[str, str] = {
    "standard": STANDARD_SUFFIX,
    "explain": EXPLAIN_SUFFIX,
    "interview": INTERVIEW_SUFFIX,
    "preview": PREVIEW_SUFFIX,
}


def get_prompt_variant_suffix(variant: str) -> str:
    key = (variant or "standard").strip().lower()
    if key not in PROMPT_VARIANT_SUFFIXES:
        supported = ", ".join(sorted(PROMPT_VARIANT_SUFFIXES.keys()))
        raise ValueError(f"Unknown prompt variant '{variant}'. Supported: {supported}.")
    return PROMPT_VARIANT_SUFFIXES[key]
