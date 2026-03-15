# Prompt-Only Optimization Plan (Code0 + Thesis + Code1)

## 1) Overview

This plan improves the Step 2.3 thesis pipeline using **prompt changes only**.

No pipeline logic changes are assumed:
- no metric changes
- no sampling changes
- no evaluator changes
- no new CLI flags

The goal is to improve:
- Code0 accuracy (`val_acc`, `test_acc`)
- thesis quality (`coverage_eq`, `faithfulness_code0`, `faithfulness_gt`)
- Code1 reliability (compile/accept rates and lower eval errors)

---

## 2) Why This Is Needed

### What we are trying to fix

Observed behavior in recent runs:
- low reasoning often produced better end-to-end metrics than medium
- medium/high often drifted into score-like or hybrid code styles
- those structures are harder for thesis prompts to summarize faithfully
- Code1 then becomes brittle because thesis conditions are less explicit

### Project-specific implication

In this codebase, thesis and Code1 are downstream of Code0 structure:
- Code0 prompt suffixes come from `prompt_variants.py`
- thesis prompt versions come from `v1/v2/v3` templates in `prompt_variants.py`
- Code1 writer/verifier prompts come from templates in `code1_verifier.py`

So the strongest non-pipeline lever is to force Code0 into explicit branch logic, then force thesis output into strict condition format, then force Code1 to transcribe exactly.

---

## 3) Scope and Constraints

This plan stays inside current interfaces:
- Code0 output format remains: `{"code":"..."}`
- thesis output format remains:
  - JSON with `conditions` (string) and `label`
- Code1 output format remains:
  - JSON with `code1`

Testing is done with existing `thesis_runner.py` options only.

---

## 4) Prompt Variant Strategy

## 4.1 Code0 Variants

Use existing `--prompt-variant` slots and replace their suffix text:

- `explain` = baseline (current)
- `interview` = C0-A (strict decision-list)
- `preview` = C0-B (multi-path compact rule set)
- `regional` = C0-C (region partition)

### C0-A (strict decision-list) for `interview`

```text
IMPORTANT: Generate an ordered decision-list classifier using explicit if/elif/else branches.

HARD CONSTRAINTS:
- Use explicit branch logic only. Do NOT use any score accumulator, weighted sum, voting tally, probability, or calibration.
- Forbidden patterns: "score +=", "votes +=", "sum(", "logit", "sigmoid".
- At least 3 explicit return branches for label 1 and at least 3 explicit return branches for label 0.
- Each branch should use 1-3 conditions.
- Prefer <= and >= threshold checks with concrete values.
- Keep code compact and interpretable.

QUALITY GOALS:
- Maximize predictive accuracy on the provided examples.
- Use thresholds that appear data-grounded from examples.
- Avoid tiny edge-case branches unless necessary.

Output ONLY:
{"code": "<python function>"}
```

Why:
- Prevents score-blob drift.
- Improves traceability for thesis extraction.

### C0-B (compact multi-path rules) for `preview`

```text
IMPORTANT: Build a compact multi-path rule set.

Requirements:
- Multiple distinct paths must lead to label 1.
- Multiple distinct paths must lead to label 0.
- No additive scoring model, no rule voting model.
- Prefer mutually meaningful branches over one giant fallback else.
- Use simple if/elif/else with direct returns.
- Use 2-4 core features repeatedly if they are strongest.

Do NOT output commentary. Output only:
{"code": "<python function>"}
```

Why:
- Explicitly enforces thesis diversity on both labels.

### C0-C (explicit regions) for `regional`

```text
IMPORTANT: Partition the feature space into explicit regions and assign labels by region.

Rules:
- Implement 4-8 explicit regions via if/elif branches.
- Each region must return a label directly.
- Include explicit negative regions (not only default fallback).
- Do NOT use score accumulation or weighted contributions.
- Keep conditions simple and traceable.

Goal:
- Strong accuracy with branch-level interpretability suitable for downstream thesis extraction.

Output only JSON:
{"code": "<python function>"}
```

Why:
- Encourages broad, explicit subspaces and reduces thesis collapse.

---

## 4.2 Thesis Variants

Use existing `--thesis-prompt-version` slots:
- `v1` = baseline (current)
- `v2` = T1 (strict faithful conjunction)
- `v3` = T2 (path-first concise thesis)

### T1 for `v2`

```text
You previously generated the following classification code:

[CODE0]

Sample:
[SAMPLE]
Predicted label:
[LABEL]

Task:
Produce a faithful thesis derived from the actual executed code path for this sample.

Rules:
- Output conditions as a single conjunction using AND only.
- Use only atomic predicates of the form:
  xN >= c, xN <= c, xN > c, xN < c, xN == 'cK', xN != 'cK'
- No OR, no natural-language conditions, no vague words.
- Conditions must hold for the sample.
- Conditions must reflect actual code logic, not guessed correlations.
- Prefer 2-4 conditions. If unsure, prefer faithfulness over breadth.
- Do not claim percentages or coverage estimates.

Output strict JSON:
{
  "conditions": "x5 == 'c2' AND x0 >= 40.0",
  "label": [LABEL]
}
```

Why:
- Keeps parser contract unchanged.
- Reduces "broad but unfaithful" condition generation.

### T2 for `v3`

```text
You previously generated this code:

[CODE0]

Sample:
[SAMPLE]
Predicted label:
[LABEL]

Step 1:
Identify the exact branch/path conditions that lead to this label.

Step 2:
Return a concise thesis that is still code-faithful:
- Keep only conditions necessary to describe the path.
- Use AND-only conjunction.
- Use exact feature names and threshold directions.
- Use string equality for categorical values like 'c0'/'c1'.

Priority:
1) Faithfulness to code path
2) Simplicity
3) Breadth (only if still faithful)

Output strict JSON:
{
  "conditions": "x3 >= 58.0 AND x15 <= -10.0 AND x11 == 'c1'",
  "label": [LABEL]
}
```

Why:
- Forces path grounding and minimality.

---

## 4.3 Code1 Variants

There is no CLI switch for Code1 prompt variants today. Prompt-only testing is still possible by swapping template text in `code1_verifier.py` between run blocks.

### W1 (Code1 writer template)

```text
You are given thesis conditions and must implement them exactly.

Thesis conditions:
[CONDITIONS]

Label:
[LABEL]

Reference sample format:
[SAMPLE]

Task:
Write exactly one function:
def check_conditions(x):
that returns True iff x satisfies the thesis conditions.

Strict rules:
- Implement condition logic only. No classification logic.
- Preserve operators and thresholds exactly from the thesis.
- Do not add extra predicates.
- Handle x as dict or list/tuple (xN -> x[N]).
- Treat missing/invalid values as non-satisfying for that atomic condition.
- Categorical features are strings ('c0','c1',...). Compare as strings.

Output strict JSON only:
{
  "code1": "def check_conditions(x):\\n    ..."
}
```

### V1 (Code1 verifier template)

```text
Verify whether check_conditions(x) exactly matches the thesis logic.

Thesis:
[CONDITIONS]

Label:
[LABEL]

Candidate code:
[CODE1]

Output strict JSON:
{
  "judgement": "pass|fail|uncertain",
  "reason": "short explanation",
  "testcases": [
    {"sample": {...}, "expected": true, "note": "..."},
    {"sample": {...}, "expected": false, "note": "..."}
  ]
}

Testcase requirements:
- Include positive and negative cases.
- For every numeric threshold in thesis, include boundary tests:
  just below, exactly at, just above.
- For categorical equality, include one matching and one non-matching case.
- Use exact categorical string format (e.g., {"x5":"c4"}).
```

Why:
- Improves deterministic transcription and boundary coverage.

---

## 5) Test Plan and Commands

All commands below are PowerShell from repo root:
- `C:\Users\lucas\Desktop\Coding\TomerResearch\LLM_PV_Working_Copy`

Use fixed settings for fair comparison:
- same dataset setup
- same sample indices file
- same reasoning for Code0/Code1 unless intentionally varied

### Shared run settings

```powershell
$common = @(
  "-m","program_synthesis.thesis_runner",
  "--functions","fn_o",
  "--seeds","2201",
  "--samples-per-seed","50",
  "--attempts","20",
  "--num-trials","1",
  "--train-size","200",
  "--val-size","2300",
  "--test-size","7500",
  "--model","gpt-5.2",
  "--reasoning-effort","low",
  "--text-verbosity","low",
  "--max-output-tokens","16000",
  "--code1-reasoning-effort","low",
  "--code1-text-verbosity","low",
  "--code1-max-output-tokens","1200",
  "--auto-split",
  "--train-cap","200",
  "--total-cap","10000",
  "--compute-baselines",
  "--stratified-sampling",
  "--sample-seed","42",
  "--sample-indices-file","program_synthesis/gpt52_comparison_indices.json"
)
```

### Phase A: Code0 prompt-variant bakeoff

Run 3 repeats per variant:

```powershell
$code0Variants = @("explain","interview","preview","regional")
foreach ($v in $code0Variants) {
  1..3 | ForEach-Object {
    python @common --prompt-variant $v --thesis-prompt-version v2
  }
}
```

Compare:
- best `test_acc`
- mean `coverage_eq`
- mean `faithfulness_code0`
- mean `faithfulness_gt`

### Phase B: Thesis prompt bakeoff (keep best Code0 variant fixed)

```powershell
$bestCode0 = "interview"   # replace after Phase A decision
$thesisVersions = @("v1","v2","v3")
foreach ($tv in $thesisVersions) {
  1..3 | ForEach-Object {
    python @common --prompt-variant $bestCode0 --thesis-prompt-version $tv
  }
}
```

### Phase C: Code1 prompt bakeoff

1) Run baseline templates in `code1_verifier.py`:

```powershell
1..3 | ForEach-Object {
  python @common --prompt-variant interview --thesis-prompt-version v2
}
```

2) Replace writer/verifier template text with `W1` and `V1` (prompt-only change), then rerun same command:

```powershell
1..3 | ForEach-Object {
  python @common --prompt-variant interview --thesis-prompt-version v2
}
```

### Quick result inspection per run

```powershell
$latest = Get-ChildItem program_synthesis/runs_step23_live_matrix -Directory | Sort-Object Name -Descending | Select-Object -First 1
Get-Content "$($latest.FullName)\overall_summary.json"
```

---

## 6) Decision Rules

Primary order of importance:
1. `mean_faithfulness_code0_defined`
2. `mean_coverage_eq_all`
3. `mean_faithfulness_gt_defined`
4. `test_acc` (guardrail: no major regression)

Suggested acceptance:
- `faithfulness_code0` >= baseline
- `coverage_eq` increases
- `faithfulness_gt` stable or better
- `test_acc` drop <= 0.5pp (or improves)

---

## 7) Success/Failure Implications

## If Code0 variants succeed

Implication:
- main bottleneck was Code0 representational drift
- keep strict branch-only constraints permanently
- thesis/code1 improvements should become easier and more stable

## If Code0 variants fail

Implication:
- current data/examples are insufficient for prompt-only structure control at chosen reasoning level
- likely need more candidate attempts or explicit prompt simplification
- if still failing, future work likely needs non-prompt interventions

## If thesis variants succeed (with same Code0)

Implication:
- thesis prompt wording is a major source of faithfulness/coverage behavior
- lock in stricter conjunction format and path grounding

## If thesis variants fail

Implication:
- thesis quality ceiling is mostly set by Code0 structure, not thesis wording
- prioritize Code0 prompt quality first

## If Code1 variants succeed

Implication:
- existing thesis is parseable but transcribed inconsistently
- keep strict exact-match writer + boundary-focused verifier prompts

## If Code1 variants fail

Implication:
- thesis condition strings are still too ambiguous/inconsistent
- revisit thesis output grammar constraints before tuning Code1 again

---

## 8) Practical Execution Order

1. Implement prompt text replacements (no logic changes).
2. Run Phase A and pick best Code0 variant.
3. Run Phase B and pick best thesis version.
4. Run Phase C for Code1 templates.
5. Confirm best combined stack with 5 additional repeats.

This sequence isolates where gains come from and avoids mixing multiple prompt changes at once.
