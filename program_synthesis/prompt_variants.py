"""Prompt helpers for code-generation variants and thesis-generation prompts."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping


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


THESIS_GENERATION_TEMPLATE = """
You previously generated the following classification code from
training data:

[CODE0]

I ran the following sample through your code:
Sample: [SAMPLE]
Your code classified it as: [LABEL]

Explain WHY this classification was made. Your explanation must be
a structured thesis of the following form:

THESIS:
Conditions: [list of conditions on the features, e.g., "x1 > 5 AND x3 < 2"]
Label: [the predicted label, e.g., 1]

Requirements:
- The conditions must be SHORT and human-interpretable
- The conditions must hold for the given sample
- The conditions must reflect what your CODE actually does (not just
  random true facts about the data)
- The conditions should define a MEANINGFUL subset of the data - not
  just this one sample, but a general pattern
- Every sample that satisfies these conditions should be classified
  as [LABEL] by your code (or as close to 100% as possible)
- The subset should be as LARGE as possible while maintaining accuracy

Note on data format:
- Numeric features use float comparisons: x0 >= 40.0
- Categorical features use string values like 'c0', 'c1', 'c4'.
  Write them as: x5 == 'c2', NOT x5 == 2.

Output your thesis in this exact JSON format:
{
  "conditions": "x5 == 'c2' AND x0 >= 40.0",
  "label": 1
}
""".strip()


THESIS_GENERATION_TEMPLATE_V2 = """
You previously generated the following classification code from
training data:

[CODE0]

I ran the following sample through your code:
Sample: [SAMPLE]
Your code classified it as: [LABEL]

Step 1 — Code-path trace:
First, trace through your code for this sample step by step. Identify
which branches and conditions led to the classification [LABEL].

Step 2 — Thesis:
Based on your trace, produce a thesis: a set of conditions that explain
WHY this sample received label [LABEL].

THESIS:
Conditions: [list of conditions on the features]
Label: [the predicted label]

Requirements:
- The conditions must be SHORT and human-interpretable
- The conditions must hold for the given sample
- The conditions must reflect what your CODE actually does — trace the
  actual code path, do not invent conditions
- The conditions should define a BROAD subset of the data — not just
  this one sample, but a general pattern
- Your conditions should capture at least 30% of training samples.
  Think about the BROADEST conditions that faithfully describe what
  your code does.
- Every sample that satisfies these conditions should be classified
  as [LABEL] by your code (or as close to 100% as possible)
- Use >= or <= (not strict > or <) when the sample's value equals
  a threshold in the code
- Focus on the 2-4 most important conditions rather than listing every
  minor check

Note on data format:
- Numeric features use float comparisons: x0 >= 40.0
- Categorical features use string values like 'c0', 'c1', 'c4'.
  Write them as: x5 == 'c2', NOT x5 == 2.

Output your thesis in this exact JSON format:
{
  "conditions": "x5 == 'c2' AND x0 >= 40.0",
  "label": 1
}
""".strip()


_FEATURE_KEY_RE = re.compile(r"^x(\d+)$")


def _sample_sort_key(key: Any) -> tuple[int, int, str]:
    key_str = str(key)
    m = _FEATURE_KEY_RE.fullmatch(key_str)
    if m:
        return (0, int(m.group(1)), key_str)
    return (1, 0, key_str)


def format_sample_for_thesis_prompt(sample: Mapping[str, Any] | str) -> str:
    if isinstance(sample, str):
        return sample.strip()
    parts = []
    for key in sorted(sample.keys(), key=_sample_sort_key):
        value = sample[key]
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def build_thesis_generation_prompt(code0: str, sample_repr: str, predicted_label: int) -> str:
    code_text = (code0 or "").strip()
    sample_text = (sample_repr or "").strip()
    label_text = str(int(predicted_label))

    prompt = THESIS_GENERATION_TEMPLATE
    prompt = prompt.replace("[CODE0]", code_text)
    prompt = prompt.replace("[SAMPLE]", f"[{sample_text}]")
    prompt = prompt.replace("[LABEL]", label_text)
    return prompt


def build_thesis_generation_prompt_v2(code0: str, sample_repr: str, predicted_label: int) -> str:
    code_text = (code0 or "").strip()
    sample_text = (sample_repr or "").strip()
    label_text = str(int(predicted_label))

    prompt = THESIS_GENERATION_TEMPLATE_V2
    prompt = prompt.replace("[CODE0]", code_text)
    prompt = prompt.replace("[SAMPLE]", f"[{sample_text}]")
    prompt = prompt.replace("[LABEL]", label_text)
    return prompt


def get_prompt_variant_suffix(variant: str) -> str:
    key = (variant or "standard").strip().lower()
    if key not in PROMPT_VARIANT_SUFFIXES:
        supported = ", ".join(sorted(PROMPT_VARIANT_SUFFIXES.keys()))
        raise ValueError(f"Unknown prompt variant '{variant}'. Supported: {supported}.")
    return PROMPT_VARIANT_SUFFIXES[key]
