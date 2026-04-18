# CodeBoost Model Comparison

Run completed on 2026-04-18.

This file compares the strongest saved CDC semantic CodeBoost setting across
model families and against the matched classical baselines.

Configuration for the CodeBoost rows:

- dataset/function: CDC diabetes, `fn_o`, length 21
- representation: semantic named CDC features with qualitative bins
- train/validation/test: 256/256/2000
- prompt batch size: 256
- boost rounds: 1
- trials: 5
- retries: up to 8
- sampling: without replacement
- acceptance: `max_weak_error=0.3025`, best-valid fallback up to 0.499
- tools: disabled

Important API detail: the existing `protected.gpt-5.2` row used TAMU/Azure
chat completions with `reasoning_effort=medium` passed through `extra_body`.
The new `gpt-5.4` row uses Azure Responses API with
`reasoning={"effort": "medium"}` because the TAMU guide says reasoning for
5.4 is exposed through Responses, not chat completions.

Raw comparison CSV:

- `program_synthesis/codeboost_model_comparison_cdc_semantic.csv`

Raw generated 5.4 artifacts:

- `program_synthesis/boosted/runs/model_compare_cdc_semantic_t1_b256_s5/gpt_5_4_medium_final/`

## Headline Table

| Method | Model | API | Reasoning | Mean test | Std | Min | Max | Gap to best CDC baseline | Attempts | Cost |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | logistic regression | local sklearn | n/a | 0.7225 | 0.0000 | n/a | n/a | 0.0000 | n/a | n/a |
| CodeBoost semantic | `protected.gpt-5.2` | chat completions | medium | 0.7173 | 0.0102 | 0.7020 | 0.7300 | -0.0052 | 6 | $0.4465 |
| CodeBoost semantic | `gpt-5.4` | Responses | medium | 0.6995 | 0.0108 | 0.6810 | 0.7150 | -0.0230 | 5 | unknown |

## CDC Baselines

| Model | Test accuracy | Std |
| --- | ---: | ---: |
| logistic regression | 0.7225 | 0.0000 |
| random forest | 0.7203 | 0.0028 |
| AdaBoost | 0.7180 | 0.0000 |
| extra trees | 0.7153 | 0.0034 |
| XGBoost | 0.7037 | 0.0061 |
| gradient boosting | 0.7027 | 0.0034 |
| MLP | 0.6964 | 0.0090 |
| hist gradient boosting | 0.6835 | 0.0000 |
| decision tree | 0.6672 | 0.0002 |

## 5.4 Per-Trial Results

| Trial | Train acc | Val acc | Test acc | Batch acc | Weighted error | Prompt tokens | Completion tokens | Reasoning tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7109 | 0.6758 | 0.6995 | 0.7109 | 0.2891 | 28167 | 8332 | 7840 |
| 2 | 0.7109 | 0.7070 | 0.6810 | 0.7109 | 0.2891 | 28167 | 11020 | 10667 |
| 3 | 0.7227 | 0.6875 | 0.7010 | 0.7227 | 0.2773 | 28167 | 4678 | 4369 |
| 4 | 0.7070 | 0.7227 | 0.7150 | 0.7070 | 0.2930 | 28167 | 9010 | 8462 |
| 5 | 0.7188 | 0.6758 | 0.7010 | 0.7188 | 0.2812 | 28167 | 4526 | 3912 |

Aggregate 5.4 token usage:

- prompt tokens: 140835
- completion tokens: 37566
- reasoning tokens: 35250
- total API attempts: 5
- pricing: not encoded in the runner yet, so cost is intentionally recorded as
  unknown rather than a guessed GPT-5 rate.

## Interpretation

This 5.4 full run is not a win at the current setting. It underperforms the
existing 5.2 matched CDC semantic row by 1.78 percentage points and the best
CDC baseline by 2.30 percentage points.

The 5.4 row does still produce valid weak learners consistently: all five
trials accepted the first candidate with weighted error below 0.3025. The issue
is not basic failure to synthesize code; it is that the learned rules generalize
worse than the previous 5.2 semantic learners on the same split.

The current defensible story remains:

- semantic/named tabular rows are the main accuracy lever;
- executable CodeBoost is competitive on CDC but not broadly baseline-beating;
- model upgrades alone are not enough to make boosting work on large tabular
  train sets;
- the next accuracy work should focus on candidate selection, local post-hoc
  ensembles, and smarter prompt batches, not just a larger model.

Recommended next checks:

- Run a small `gpt-5.4-mini` or `gpt-5.4-nano` comparison if we want cheaper
  proposers for candidate-library sweeps.
- If strict model-only comparison matters, rerun `protected.gpt-5.2` through the
  same Responses API path with `reasoning_effort=medium`; the current 5.2 row is
  the historically matched row, but it used chat completions.
- Do not spend a large multi-round 5.4 full budget until a one-shot or
  candidate-library pilot beats the current 5.2 CDC semantic row.
