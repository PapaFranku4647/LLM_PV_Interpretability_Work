# CodeBoost Model Comparison

Run completed on 2026-04-20.

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
The new 5.4-family rows use Azure Responses API with
`reasoning={"effort": "medium"}` because the TAMU guide says reasoning for
5.4 is exposed through Responses, not chat completions.

Raw comparison CSV:

- `program_synthesis/codeboost_model_comparison_cdc_semantic.csv`

Raw generated 5.4-family artifacts:

- `program_synthesis/boosted/runs/model_compare_cdc_semantic_t1_b256_s5/gpt_5_4_medium_final/`
- `program_synthesis/boosted/runs/model_compare_cdc_semantic_t1_b256_s5/gpt_5_4_mini_medium_final/`
- `program_synthesis/boosted/runs/model_compare_cdc_semantic_t1_b256_s5/gpt_5_4_nano_medium_final/`

## Headline Table

| Method | Model | API | Reasoning | Mean test | Std | Min | Max | Gap to best CDC baseline | Attempts | Cost |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | logistic regression | local sklearn | n/a | 0.7225 | 0.0000 | n/a | n/a | 0.0000 | n/a | n/a |
| CodeBoost semantic | `protected.gpt-5.2` | chat completions | medium | 0.7173 | 0.0102 | 0.7020 | 0.7300 | -0.0052 | 6 | $0.4465 |
| CodeBoost semantic | `gpt-5.4-nano` | Responses | medium | 0.7067 | 0.0041 | 0.7000 | 0.7125 | -0.0158 | 5 | unknown |
| CodeBoost semantic | `gpt-5.4` | Responses | medium | 0.6995 | 0.0108 | 0.6810 | 0.7150 | -0.0230 | 5 | unknown |
| CodeBoost semantic | `gpt-5.4-mini` | Responses | medium | 0.6865 | 0.0213 | 0.6530 | 0.7080 | -0.0360 | 10 | unknown |

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

## 5.4 Full Per-Trial Results

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

## 5.4 Mini Per-Trial Results

| Trial | Train acc | Val acc | Test acc | Attempts | Prompt tokens | Completion tokens | Reasoning tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6992 | 0.6562 | 0.6530 | 2 | 56334 | 17365 | 16956 |
| 2 | 0.7188 | 0.6602 | 0.6700 | 4 | 84501 | 12139 | 11286 |
| 3 | 0.6992 | 0.6953 | 0.7080 | 2 | 56334 | 21834 | 21233 |
| 4 | 0.7031 | 0.6875 | 0.7000 | 1 | 28167 | 3008 | 2815 |
| 5 | 0.7266 | 0.6914 | 0.7015 | 1 | 28167 | 4933 | 4660 |

Aggregate 5.4-mini token usage:

- prompt tokens: 253503
- completion tokens: 59279
- reasoning tokens: 56950
- total API attempts: 10
- pricing: not encoded in the runner yet.

## 5.4 Nano Per-Trial Results

| Trial | Train acc | Val acc | Test acc | Attempts | Prompt tokens | Completion tokens | Reasoning tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7070 | 0.6953 | 0.7125 | 1 | 28167 | 1642 | 1090 |
| 2 | 0.7070 | 0.6875 | 0.7080 | 1 | 28167 | 1321 | 833 |
| 3 | 0.7188 | 0.6758 | 0.7075 | 1 | 28167 | 1681 | 922 |
| 4 | 0.7148 | 0.6953 | 0.7055 | 1 | 28167 | 1506 | 922 |
| 5 | 0.7148 | 0.6797 | 0.7000 | 1 | 28167 | 2054 | 1477 |

Aggregate 5.4-nano token usage:

- prompt tokens: 140835
- completion tokens: 8204
- reasoning tokens: 5244
- total API attempts: 5
- pricing: not encoded in the runner yet.

## Interpretation

The 5.4-family runs are not a direct model-quality win at the current setting.
The best 5.4-family mean is `gpt-5.4-nano` at 0.7067, which is still 1.06
percentage points below the existing 5.2 matched CDC semantic row and 1.58
points below the best CDC baseline.

The full 5.4 row produces valid weak learners consistently: all five trials
accepted the first candidate with weighted error below 0.3025. Nano was also
stable, accepted every first candidate, and used far fewer completion/reasoning
tokens. Mini was the weakest direct row and needed 10 total attempts because of
rejections or malformed responses. The issue is not basic failure to synthesize
code; it is that the learned rules are still highly correlated and do not
generalize better than the previous 5.2 semantic learners on this split.

The current defensible story remains:

- semantic/named tabular rows are the main accuracy lever;
- executable CodeBoost is competitive on CDC but not broadly baseline-beating;
- model upgrades alone are not enough to make boosting work on large tabular
  train sets;
- `gpt-5.4-nano` is a plausible cheap proposer for candidate-library sweeps,
  while `gpt-5.4-mini` is not promising as a direct CDC semantic learner;
- the next accuracy work should focus on candidate selection, local post-hoc
  ensembles, and smarter prompt batches, not just a larger model.

Recommended next checks:

- Use `gpt-5.4-nano` for a cheap candidate-library pilot if we want many
  candidates; avoid spending on direct `gpt-5.4-mini` CDC repeats without a new
  prompt or selector.
- If strict model-only comparison matters, rerun `protected.gpt-5.2` through the
  same Responses API path with `reasoning_effort=medium`; the current 5.2 row is
  the historically matched row, but it used chat completions.
- Do not spend a large multi-round 5.4 full budget until a one-shot or
  candidate-library pilot beats the current 5.2 CDC semantic row.
- For a second positive dataset, HTRU2 is the right target. It already has a
  0.9300 best CodeBoost trial against a 0.9340 best baseline, so a semantic or
  hybrid 5-trial HTRU2 run is much more plausible than mushroom or chess.
