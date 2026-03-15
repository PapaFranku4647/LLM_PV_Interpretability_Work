# Code0 Prompt Experiments — Design Document

## Context & Motivation

After 5 runs on fn_o (CDC Diabetes) with varying thesis prompts and reasoning
levels, the key finding is: **Code0 structure determines thesis quality more
than anything else.**

The high-reasoning Code0 produced a clean decision tree:
```python
if x15 <= -6.0: return 1
if x14 >= 10.0 and x3 >= 56.0: return 1
if x3 >= 70.0 or (x19 >= -6.5 and x3 >= 49.0): return 1
return 0
```
This has ONE path to label 0 (the else/fallthrough), so 24/25 theses were
identical. Perfect faithfulness (100% Code0, 98% GT) but zero diversity.

The medium-reasoning Code0 was a scoring model with multiple contributing
factors, which produced slightly more diverse theses but much lower faithfulness.

**Goal**: Design Code0 prompts that produce classifiers with:
1. High accuracy (60%+ on CDC diabetes)
2. Multiple distinct paths to EACH label (thesis diversity)
3. Clear, interpretable decision logic
4. Thresholds that reflect real data patterns (not overfitting)

## Thesis Prompt Decision: V2 Wins, Drop V3

Evidence across 4 controlled runs:
- V2 minimal beats V3 minimal: +8.9pp Code0 faith, +1.3pp GT faith
- V2 medium beats V3 medium: +0.9pp Code0 faith, +2.1pp GT faith

V3's "faithfulness-first + identify key factors" framing consistently produces
BROADER, LESS faithful theses. The model reads "1-3 key factors" as license to
simplify, which hurts precision. V2's "trace the code path" is mechanical and
effective. Recommendation: **stick with V2 for all future experiments.**

A future V4 thesis prompt should address the DIVERSITY problem (e.g., require
sample-specific conditions), not re-attempt the faithfulness framing. But this
is lower priority than fixing Code0 structure.

---

## Code0 Prompt Variants to Test

### Current: "explain" (baseline)

```
IMPORTANT: Generate code that is clear and interpretable. Each condition
and threshold should reflect a meaningful pattern in the training data.
After you generate this code, I will ask you to explain and justify each
decision in the code based on statistical properties of the data.
```

**Problem**: Too vague. Doesn't guide structure. High-reasoning models produce
elegant but structurally simple code (single path per label).


### Variant A: "multipath" — Multiple Decision Paths Per Label

```
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
by tracing which specific branch was triggered for each sample.
```

**Rationale**: Directly solves the diversity problem by requiring multiple paths
per label. Each test sample would trigger a specific branch, producing a unique
thesis. Risk: model might force artificial splits that reduce accuracy.


### Variant B: "subgroups" — Subgroup Discovery Framing

```
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
in the training data you were given.
```

**Rationale**: Frames the task as subgroup discovery rather than pure
classification. The model should look for natural clusters in the data
and encode each as a separate branch. This could produce the most
scientifically meaningful Code0 structures. Risk: the model might
hallucinate subgroups that don't exist.


### Variant C: "thesis_aware" — Thesis-Aware Code Generation

```
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
but through different feature-based logic.
```

**Rationale**: Directly tells the model about the downstream thesis task.
This is the most "meta" approach — the model knows its code will be
analyzed per-sample and can optimize structure accordingly. Risk: the model
might over-optimize for thesis diversity at the expense of accuracy.


### Variant D: "regional" — Feature-Region Decision Map

```
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
reflect a meaningful pattern in the training data.
```

**Rationale**: Forces the model to think in terms of feature-space
regions rather than sequential decisions. The key innovation is "no
fallthrough" — every label-0 prediction should explicitly match a
region, not just fail all label-1 conditions. This directly prevents
the "single else path" problem. Risk: harder for the model to satisfy;
might produce lower accuracy if it can't find clean regions.


### Variant E: "ensemble" — Independent Rule Ensemble

```
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
- Add a comment before each rule briefly describing the pattern.

This structure allows me to later ask: "Which rules fired for this
specific sample?" — enabling per-sample explanations.
```

**Rationale**: Scoring/voting models with NAMED rules. Unlike the
current scoring models (which emerged naturally and are opaque), this
explicitly asks for independently meaningful rules. Each thesis would
list which rules fired. Risk: voting models can have lower accuracy
than decision trees; the model might create correlated rather than
independent rules.

---

## Recommended Testing Order

1. **Variant C (thesis_aware)** — Most directly addresses the problem.
   Tells the model exactly what we need and why. Best chance of getting
   both accuracy and diversity.

2. **Variant A (multipath)** — Simplest structural constraint. Easy to
   verify whether the model complied. Good baseline for diversity.

3. **Variant D (regional)** — Most scientifically interesting. If it
   works, it produces the most interpretable code. The "no fallthrough"
   requirement is key.

4. **Variant B (subgroups)** — Good framing but less prescriptive about
   code structure. The model might not translate the subgroup concept
   into diverse code paths.

5. **Variant E (ensemble)** — Most complex. Interesting but may
   sacrifice accuracy. Try last.

---

## Testing Protocol

For each variant:
1. Run with high reasoning, V2 thesis prompt, same seed/dataset
2. Use --reuse-code0-from to compare thesis diversity on same test samples
   (if the Code0 from a different variant is useful as a baseline)
3. Actually, each variant produces its OWN Code0 — so you can't reuse.
   Run the full pipeline each time.
4. Compare: accuracy, faith-Code0, faith-GT, coverage, AND a new metric:
   **thesis diversity** = number of unique thesis condition sets / 25

Command template:
```bash
python -m program_synthesis.thesis_runner \
  --functions fn_o --seeds 2201 --samples-per-seed 25 \
  --attempts 10 --num-trials 1 \
  --thesis-prompt-version v2 --reasoning-effort high \
  --max-output-tokens 16000 --auto-split --total-cap 10000 \
  --prompt-variant [VARIANT_NAME]
```

---

## Implementation Notes

All variants are defined as suffix strings in `prompt_variants.py` under
`PROMPT_VARIANT_SUFFIXES`. To add a new variant:

1. Define the suffix string (e.g., `MULTIPATH_SUFFIX = """..."""`)
2. Add to `PROMPT_VARIANT_SUFFIXES`: `"multipath": MULTIPATH_SUFFIX`
3. Run with `--prompt-variant multipath`

No other code changes needed — the runner already handles arbitrary
prompt variant keys.
