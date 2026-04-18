# Stratified Diverse CodeBoost Pilot

Run completed on 2026-04-18 local time.

This pilot tests whether a residual/diverse sampler can make CDC semantic
CodeBoost improve beyond the current strong `T=1` semantic result.

Configuration:

- dataset: CDC diabetes (`fn_o`, length 21)
- representation: semantic
- train size: 10000
- validation size: 2000
- test size: 10000
- batch sizes: 64 and 128
- boost rounds requested: 4
- trials: 1
- retries per round: 3
- retry behavior: resample each retry
- sampler: `stratified_diverse`
- acceptance: full weighted-train error must be at most 0.45
- validation: early stop patience 2, restore best-validation ensemble prefix
- model: `protected.gpt-5.2` via TAMU/Azure chat completions

Raw generated artifacts are preserved under:

- `program_synthesis/boosted/runs/semantic_cdc_stratified_diverse_t4_b64_b128_s1/`

Aggregate CSV:

- `program_synthesis/codeboost_semantic_cdc_stratified_diverse_t4_b64_b128_s1.csv`

## Results

| Batch | Final train | Final val | Final test | Kept rounds | API attempts | Cost | Stopped reason |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 0.7043 | 0.6985 | 0.7076 | 1 | 6 | $0.3033 | no acceptable learner after restoring best val |
| 128 | 0.6885 | 0.6800 | 0.6902 | 1 | 5 | $0.2848 | validation early stop after restoring best val |

The sampler did generate later candidates with full weighted-train error below
0.45, but adding them hurt validation accuracy. With best-validation restoration,
both saved ensembles trimmed back to one learner.

## Batch Accuracy Check

Across all 11 attempted candidates:

- batch/test correlation: about 0.13
- batch/train correlation: about 0.06

Batch accuracy was therefore not a reliable guide to overall accuracy in this
pilot. The strongest example is batch 64 round 1: batch accuracy was only
0.53125, but full-train/test were 0.7043/0.7076. Conversely, higher prompt-batch
accuracy did not guarantee a better full-train or test result.

## Interpretation

This is a useful negative result. The implementation is doing the intended
full-train gate and validation restoration, but the current `stratified_diverse`
recipe did not produce a better CDC ensemble at batch 64 or 128.

Likely reasons:

- After the first CDC scorecard, the weighted residual distribution is harder
  and the LLM tends to produce weak variants that do not improve validation.
- Batch sizes 64 and 128 are small enough that prompt-batch accuracy is noisy.
- The strict 0.45 weighted-error gate filters out bad candidates, but accepted
  later learners can still overfit the weighted residual and reduce validation.

Next technical adjustments:

- Try batch 256 with `stratified_diverse`; 64/128 are probably too noisy.
- Generate a candidate library per round, then pick the best full-train/val
  candidate locally instead of accepting the first candidate below the gate.
- Add an ensemble-validation acceptance gate: accept a learner only if the
  ensemble validation accuracy does not drop by more than a small tolerance.
- Consider a softer gate such as 0.49 for candidate collection, but select final
  ensemble by validation.
