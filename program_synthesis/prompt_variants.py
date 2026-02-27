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


MULTIPATH_SUFFIX = """
IMPORTANT: Structure your code as a decision tree or rule set where EACH
output label (0 and 1) can be reached through MULTIPLE DISTINCT code paths.

For example, there may be several different reasons why a sample should be
classified as 1 — different subgroups of the data may share the same label
but for different feature-based reasons. Similarly for label 0.

Requirements:
- Your function should have at least 2-3 different conditional branches
  that each return the same label, but based on DIFFERENT feature conditions.
  For instance, one branch might return 1 based on x3 and x15, while another
  returns 1 based on x14 and x20.
- Each branch should capture a meaningful subgroup — at least 5% of the
  training data should follow each path.
- Use clear if/elif/else structure so each path is independently traceable.
- Thresholds should reflect actual patterns in the training data.
- Prioritize accuracy: each branch should be correct for the samples it covers.

After you generate this code, I will ask you to explain individual predictions
by tracing which specific branch was triggered for each sample. Do NOT add
comments to your code.
""".strip()

SUBGROUPS_SUFFIX = """
IMPORTANT: The training data likely contains distinct SUBGROUPS — clusters
of samples that share the same label but for different underlying reasons.

Your task is not just to classify accurately, but to DISCOVER and SEPARATE
these subgroups in your code structure. Think of it as building a diagnostic
decision system:

1. First, analyze the training data to identify which features best separate
   the classes. Look for features where the class distributions differ.
2. Then, identify SUBGROUPS within each class. For example, there may be
   2-3 different "profiles" of label-1 samples, each characterized by
   different feature patterns.
3. Structure your code so each subgroup has its own explicit branch.
   Each branch should check for a specific profile and return the
   appropriate label.

The ideal code looks like:
  - Branch 1: If [profile A conditions] → return 1
  - Branch 2: If [profile B conditions] → return 1
  - Branch 3: If [profile C conditions] → return 0
  - Branch 4: If [profile D conditions] → return 0
  - Default: return [majority class]

Every branch should capture a real subgroup, not an arbitrary split.
Make thresholds data-driven: pick values that genuinely separate the classes
in the training data you were given. Do NOT add comments to your code.
""".strip()

THESIS_AWARE_SUFFIX = """
IMPORTANT: After generating this function, I will do the following for
multiple test samples:
1. Run each test sample through your code to get its predicted label.
2. Ask you to produce a THESIS — a small set of conditions explaining
   why THAT SPECIFIC sample got its label, based on which code path
   it followed.
3. Evaluate whether the conditions also hold for training samples with
   the same ground truth label (not just the same predicted label).

For this to work well, your code MUST have the following properties:

(a) MULTIPLE PATHS PER LABEL: Different samples classified as the same
    label should reach that label through DIFFERENT branches. If all
    label-0 samples go through a single fallthrough/else, all theses
    will be identical and uninformative.

(b) INTERPRETABLE BRANCHES: Each branch should check a small set of
    meaningful feature conditions (2-4 conditions per branch). Avoid
    scoring models with many small contributions — they produce vague
    theses. Prefer clear if/elif/else trees.

(c) GROUND-TRUTH ALIGNMENT: Choose thresholds and conditions that
    reflect real patterns in the data. A branch that captures samples
    which the CODE labels correctly (matching ground truth) will produce
    a more useful thesis than a branch that captures misclassified samples.

(d) BALANCED COVERAGE: Each branch should handle a meaningful fraction
    of the data (at least 5-10% of training samples). Avoid tiny
    branches that only catch 1-2 edge cases.

Structure hint: An ideal function has 3-6 if/elif branches, each with
2-3 feature conditions, where multiple branches can return the same label
but through different feature-based logic. Do NOT add comments to your code.
""".strip()

REGIONAL_SUFFIX = """
IMPORTANT: Think of the feature space as a MAP with distinct REGIONS.
Each region is defined by a combination of feature ranges and should
be associated with a single label.

Your goal: partition the feature space into 4-8 distinct regions, each
with clear boundaries and a label assignment. Structure your code as
a region lookup:

  Region 1: [feature conditions] → label 1
  Region 2: [feature conditions] → label 0
  Region 3: [feature conditions] → label 1
  ...etc
  Default region: → [majority label]

Guidelines:
- Identify the 3-5 most predictive features by examining which features
  best separate label 0 from label 1 in the training data.
- Define regions using those features with data-driven thresholds.
- Each region should contain at least 10-20 training samples.
- Multiple regions can map to the same label — that's expected.
  Different parts of the feature space may predict label 1 for
  different reasons.
- Order regions from most confident (clearest separation) to least.
- Make each region's conditions self-contained (no fallthrough logic).
  Every sample should explicitly match a region, not just "fail to
  match anything else."

Generate code that is clear and interpretable. Each region should
reflect a meaningful pattern in the training data. Do NOT add comments
to your code.
""".strip()

ENSEMBLE_SUFFIX = """
IMPORTANT: Generate your function as a SET OF INDEPENDENT RULES that
vote on the final classification.

Structure:
1. Define 4-6 independent rules, each examining a different subset
   of features. Each rule returns a "vote" for label 0 or label 1.
2. Combine the votes to make the final prediction.

Example structure:
  def f(x):
      votes_for_1 = 0
      # Rule 1: Age-based pattern
      if [condition on age-related features]:
          votes_for_1 += 1
      # Rule 2: Health-metric pattern
      if [condition on health features]:
          votes_for_1 += 1
      # Rule 3: Behavioral pattern
      if [condition on behavioral features]:
          votes_for_1 += 1
      ...
      return 1 if votes_for_1 >= threshold else 0

Guidelines:
- Each rule should be INDEPENDENTLY meaningful — it should capture a
  real pattern that predicts the label on its own.
- Rules should examine DIFFERENT feature subsets where possible.
- Each rule should fire for at least 20% of training samples.
- The voting threshold should be chosen to maximize accuracy.
- Do NOT add comments to your code. The code must be self-explanatory
  through its structure alone.

This structure allows me to later ask: "Which rules fired for this
specific sample?" — enabling per-sample explanations.
""".strip()


PROMPT_VARIANT_SUFFIXES: Dict[str, str] = {
    "standard": STANDARD_SUFFIX,
    "explain": EXPLAIN_SUFFIX,
    "interview": INTERVIEW_SUFFIX,
    "preview": PREVIEW_SUFFIX,
    "multipath": MULTIPATH_SUFFIX,
    "subgroups": SUBGROUPS_SUFFIX,
    "thesis_aware": THESIS_AWARE_SUFFIX,
    "regional": REGIONAL_SUFFIX,
    "ensemble": ENSEMBLE_SUFFIX,
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


THESIS_GENERATION_TEMPLATE_V3 = """
You previously generated the following classification code from
training data:

[CODE0]

I ran the following sample through your code:
Sample: [SAMPLE]
Your code classified it as: [LABEL]

Your task: produce a THESIS — a short set of feature conditions that
explain why this sample (and others like it) receive label [LABEL].

Step 1 — Identify the key decision factors:
Look at your code and identify the 1-3 MOST IMPORTANT conditions that
determine this sample's classification. Ignore minor or redundant
checks — focus on the conditions that actually DRIVE the prediction.

Step 2 — Produce your thesis:
Write conditions that define a subset of data points where:
(a) The given sample satisfies all conditions (mandatory).
(b) Most data points satisfying these conditions truly have GROUND
    TRUTH label [LABEL] — i.e., the conditions identify a real
    pattern in the data, not just a quirk of the code.
(c) The subset is not trivially small — aim for at least 5% of
    training samples.

PRIORITY ORDER:
1. FAITHFULNESS: The subset must have high label consistency. It is
   far better to define a smaller subset where 90%+ of samples truly
   belong to class [LABEL] than a large subset where only 70% do.
2. COVERAGE: Given high faithfulness, prefer broader conditions.
   But never sacrifice faithfulness for coverage.
3. SIMPLICITY: Use 1-3 conditions. Fewer is better.

Requirements:
- The conditions must hold for the given sample
- The conditions must be grounded in what the code does — do not
  invent conditions unrelated to the code logic
- Use >= or <= (not strict > or <) when the sample's value equals
  a threshold in the code

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


def build_thesis_generation_prompt_v3(code0: str, sample_repr: str, predicted_label: int) -> str:
    code_text = (code0 or "").strip()
    sample_text = (sample_repr or "").strip()
    label_text = str(int(predicted_label))

    prompt = THESIS_GENERATION_TEMPLATE_V3
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
