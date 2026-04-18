# Batch-256 Sampler Comparison

Run completed on 2026-04-18 local time.

This run tests the generic sampler methods added after the stratified-diverse
CDC pilot. The goal was to keep the method dataset-agnostic: internal sampler
diversity uses fixed-size hashed row vectors, while prompts still use the
semantic tabular representation.

Configuration:

- dataset: CDC diabetes (`fn_o`, length 21)
- representation: semantic
- train size: 10000
- validation size: 2000
- test size: 10000
- batch size: 256
- boost rounds requested: 3
- trials: 1
- retries per round: 3
- retry behavior: resample each retry
- samplers: `feature_diverse`, `label_balanced_diverse`, `stratified_diverse`
- candidate selection: `best_ensemble_val`
- ensemble validation drop tolerance: 0.005
- weak learner gate: full weighted-train error at most 0.49
- validation: early stop patience 2, restore best-validation ensemble prefix
- model: `protected.gpt-5.2` via TAMU/Azure chat completions

Raw generated artifacts are preserved under:

- `program_synthesis/boosted/runs/semantic_cdc_sampler_compare_b256_s1/`

Aggregate CSV:

- `program_synthesis/codeboost_sampler_compare_b256_s1.csv`

## Results

| Sampler | Final train | Final val | Final test | Best val | Saved rounds | Selected before restore | API attempts | Cost | Stopped reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `feature_diverse` | 0.6984 | 0.6880 | 0.6965 | 0.6880 | 1 | 3 | 9 | $0.6975 | validation early stop after restoring best val |
| `label_balanced_diverse` | 0.7028 | 0.6905 | 0.7036 | 0.6905 | 1 | 3 | 9 | $0.6489 | validation early stop after restoring best val |
| `stratified_diverse` | 0.7015 | 0.6900 | 0.7046 | 0.6900 | 1 | 3 | 9 | $0.6415 | validation early stop after restoring best val |

Best test accuracy in this comparison was `stratified_diverse` at 0.7046.
That does not beat the prior 5-trial CDC semantic `T=1` result:

- mean test: 0.7173
- standard deviation: 0.0102
- best test: 0.7300

It also does not beat the earlier stratified-diverse batch-64 pilot test result
of 0.7076, though that pilot used a stricter 0.45 weak-error gate and no
candidate-library selection.

## Batch Accuracy

First selected learner batch accuracies:

| Sampler | Selected round 1 batch acc | Round 1 full train | Round 1 test |
| --- | ---: | ---: | ---: |
| `feature_diverse` | 0.7227 | 0.6984 | 0.6965 |
| `label_balanced_diverse` | 0.6602 | 0.7028 | 0.7036 |
| `stratified_diverse` | 0.6523 | 0.7015 | 0.7046 |

Later selected residual learners had low prompt-batch accuracy, but still passed
the weighted-train gate:

| Sampler | Round 2 selected batch acc | Round 3 selected batch acc |
| --- | ---: | ---: |
| `feature_diverse` | 0.2461 | 0.1094 |
| `label_balanced_diverse` | 0.3555 | 0.1562 |
| `stratified_diverse` | 0.4844 | 0.3477 |

This is another warning that prompt-batch accuracy is not the right objective
after round 1. The sampler is intentionally pulling hard/high-weight examples,
so raw batch accuracy can collapse even when weighted error is below 0.5.

## Interpretation

This is a useful implementation win and an accuracy negative result.

What worked:

- The generic sampler stack runs on CDC with fixed-size hashed vectors.
- `feature_diverse`, `label_balanced_diverse`, and `stratified_diverse` all
  produce valid candidates without depending on CDC-specific feature geometry.
- Candidate-library selection evaluates all retries in a round and selects
  locally by full ensemble validation behavior.
- Best-validation restoration prevents later weak learners from damaging the
  saved ensemble.

What did not work yet:

- None of the batch-256 generic samplers improved over the first semantic CDC
  scorecard.
- In every method, validation restoration trimmed the saved ensemble back to one
  learner.
- The later learners have small alphas and mostly fail to change validation
  predictions in a helpful way.

Practical takeaway:

- We should keep these samplers as infrastructure and run them on multiple
  datasets, but they are not yet the accuracy lever.
- The next serious boosting attempt should generate a larger offline candidate
  library, save every candidate program, and perform local greedy/post-hoc
  ensemble selection by validation. Spending more calls on blind sequential
  rounds is unlikely to be efficient.

## Next Steps

1. Add candidate-program logging for every valid/rejected attempt, not just the
   restored final ensemble, before larger sweeps.
2. Run the same sampler comparison on mushroom and HTRU2 to see whether the
   generic methods help outside CDC.
3. Increase candidate library size per round only after candidate source logging
   exists, for example 8-12 retries with `best_ensemble_val`.
4. Consider a one-call multi-candidate prompt so each API call returns several
   short candidate functions for local scoring.
5. Add a post-hoc local ensemble builder over a saved candidate library.
