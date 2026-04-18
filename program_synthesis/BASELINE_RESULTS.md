# Baseline Results Snapshot

Run completed from `program_synthesis/baseline_runner.py` on 2026-04-17 around
21:50 local time.

Raw outputs:

- `program_synthesis/baseline_results_core.csv`
- `program_synthesis/baseline_results_core.jsonl`

Run configuration:

- functions: `fn_o`, `fn_n`, `fn_p`, `fn_q`
- datasets: CDC diabetes, mushroom, HTRU2, chess
- train size: 256
- validation size: 256
- test size: 2000
- trials per model: 5
- selection split: validation
- XGBoost status: not run because `xgboost` was not installed in this environment

## Best Baselines By Dataset

| Function | Dataset | Best model | Test accuracy | Notes |
| --- | --- | --- | ---: | --- |
| `fn_o` | CDC diabetes | logistic regression | 0.7225 | Random forest was close at 0.7203; AdaBoost 0.7180. |
| `fn_n` | mushroom | extra trees | 0.8528 | Random forest 0.8277; gradient boosting 0.8106. |
| `fn_p` | HTRU2 | hist gradient boosting | 0.9340 | Gradient boosting 0.9310; AdaBoost 0.9295. |
| `fn_q` | chess | hist gradient boosting | 0.9615 | AdaBoost 0.9560; decision tree 0.9541. |

## Full Core Baseline Summary

| Function | Dataset | Model | Test accuracy | Test std | Val accuracy | Val std |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `fn_o` | CDC diabetes | decision tree | 0.6672 | 0.0002 | 0.6508 | 0.0019 |
| `fn_o` | CDC diabetes | random forest | 0.7203 | 0.0028 | 0.7352 | 0.0029 |
| `fn_o` | CDC diabetes | extra trees | 0.7153 | 0.0034 | 0.7562 | 0.0040 |
| `fn_o` | CDC diabetes | AdaBoost | 0.7180 | 0.0000 | 0.7188 | 0.0000 |
| `fn_o` | CDC diabetes | gradient boosting | 0.7027 | 0.0034 | 0.6992 | 0.0000 |
| `fn_o` | CDC diabetes | hist gradient boosting | 0.6835 | 0.0000 | 0.6680 | 0.0000 |
| `fn_o` | CDC diabetes | logistic regression | 0.7225 | 0.0000 | 0.7305 | 0.0000 |
| `fn_o` | CDC diabetes | MLP | 0.6964 | 0.0090 | 0.7156 | 0.0134 |
| `fn_n` | mushroom | decision tree | 0.7411 | 0.0085 | 0.7633 | 0.0080 |
| `fn_n` | mushroom | random forest | 0.8277 | 0.0033 | 0.8102 | 0.0134 |
| `fn_n` | mushroom | extra trees | 0.8528 | 0.0052 | 0.8586 | 0.0067 |
| `fn_n` | mushroom | AdaBoost | 0.7310 | 0.0000 | 0.7031 | 0.0000 |
| `fn_n` | mushroom | gradient boosting | 0.8106 | 0.0022 | 0.7898 | 0.0029 |
| `fn_n` | mushroom | hist gradient boosting | 0.7885 | 0.0000 | 0.7656 | 0.0000 |
| `fn_n` | mushroom | logistic regression | 0.7835 | 0.0000 | 0.7539 | 0.0000 |
| `fn_n` | mushroom | MLP | 0.7111 | 0.0353 | 0.6781 | 0.0255 |
| `fn_p` | HTRU2 | decision tree | 0.9200 | 0.0000 | 0.9336 | 0.0000 |
| `fn_p` | HTRU2 | random forest | 0.9243 | 0.0009 | 0.9492 | 0.0025 |
| `fn_p` | HTRU2 | extra trees | 0.9215 | 0.0021 | 0.9492 | 0.0043 |
| `fn_p` | HTRU2 | AdaBoost | 0.9295 | 0.0000 | 0.9414 | 0.0000 |
| `fn_p` | HTRU2 | gradient boosting | 0.9310 | 0.0000 | 0.9648 | 0.0000 |
| `fn_p` | HTRU2 | hist gradient boosting | 0.9340 | 0.0000 | 0.9414 | 0.0000 |
| `fn_p` | HTRU2 | logistic regression | 0.9070 | 0.0000 | 0.9453 | 0.0000 |
| `fn_p` | HTRU2 | MLP | 0.8804 | 0.0060 | 0.9148 | 0.0109 |
| `fn_q` | chess | decision tree | 0.9541 | 0.0088 | 0.9609 | 0.0000 |
| `fn_q` | chess | random forest | 0.9412 | 0.0024 | 0.9539 | 0.0029 |
| `fn_q` | chess | extra trees | 0.9418 | 0.0016 | 0.9516 | 0.0019 |
| `fn_q` | chess | AdaBoost | 0.9560 | 0.0000 | 0.9648 | 0.0000 |
| `fn_q` | chess | gradient boosting | 0.9525 | 0.0000 | 0.9688 | 0.0000 |
| `fn_q` | chess | hist gradient boosting | 0.9615 | 0.0000 | 0.9648 | 0.0000 |
| `fn_q` | chess | logistic regression | 0.9275 | 0.0000 | 0.9492 | 0.0000 |
| `fn_q` | chess | MLP | 0.8142 | 0.0170 | 0.8320 | 0.0296 |

## Immediate Interpretation

The CDC result gives CodeBoost a plausible opening. The current best saved
semantic CodeBoost run is 0.7032 mean test accuracy, which is below the strongest
CDC baseline here by about 1.9 percentage points, but it beats the CDC decision
tree, MLP, hist-gradient baseline, and is essentially tied with gradient
boosting. That is close enough to justify more work, especially because our
artifact is executable, compact, and interpretable.

The other datasets set a tougher bar. Mushroom, HTRU2, and chess have strong
classical baselines in the 0.85, 0.93, and 0.96 ranges respectively. The next
question is whether semantic CodeBoost gets near those numbers on matched
train/validation/test sizes. If it does, the paper story is viable. If it does
not, the story should emphasize CDC-style semantic tabular interpretability and
the residual/diverse sampler as the next method contribution.

Before treating this as final, rerun after installing XGBoost and optionally
TabPFN, and expand to more seeds/splits for paper-grade confidence intervals.
