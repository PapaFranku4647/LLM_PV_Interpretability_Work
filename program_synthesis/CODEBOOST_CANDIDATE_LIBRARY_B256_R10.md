# Candidate-Library CDC Sweep

Run completed on 2026-04-18 local time.

This run was done after adding full candidate source logging. Every generated
candidate function is now preserved in `attempts.jsonl` through:

- `candidate_code`
- `candidate_code_sha256`
- `candidate_code_chars`
- `candidate_source_count`
- `candidate_source_stages`
- `candidate_source_history`

The CSV artifacts intentionally omit the large source blobs, but retain the
hashes, sizes, and source counts.

Configuration:

- dataset: CDC diabetes (`fn_o`, length 21)
- representation: semantic
- train size: 10000
- validation size: 2000
- test size: 10000
- batch size: 256
- boost rounds requested: 4
- retries per round: 10
- candidate attempts: 40
- sampler: `stratified_diverse`
- candidate selection: `best_ensemble_val`
- ensemble validation drop tolerance: 0.005
- weak learner gate: full weighted-train error at most 0.49
- validation: early stop patience 3, restore best-validation ensemble prefix
- model: `protected.gpt-5.2` via TAMU/Azure chat completions

Raw generated artifacts are preserved under:

- `program_synthesis/boosted/runs/semantic_cdc_candidate_library_b256_r10_s1/`

Aggregate CSV:

- `program_synthesis/codeboost_candidate_library_b256_r10_s1.csv`

## Results

| Metric | Value |
| --- | ---: |
| Final train accuracy | 0.7031 |
| Final validation accuracy | 0.6950 |
| Final test accuracy | 0.7027 |
| Best validation accuracy | 0.6950 |
| Saved rounds after restoration | 1 |
| Selected candidates before restoration | 4 |
| API attempts | 40 |
| Candidates with logged source | 40 |
| Estimated cost | $2.9969 |

This did not improve over the prior CDC semantic `T=1` result:

- mean test: 0.7173
- standard deviation: 0.0102
- best test: 0.7300

It also did not improve over the smaller batch-256 sampler comparison, where
`stratified_diverse` reached 0.7046 test.

## Selected Candidates

| Round | Retry | Batch acc | Weighted error | Alpha | Individual val | Individual test | Ensemble val | Ensemble test |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9 | 0.6563 | 0.2969 | 0.4311 | 0.6950 | 0.7027 | 0.6950 | 0.7027 |
| 2 | 10 | 0.3867 | 0.4328 | 0.1353 | 0.6620 | 0.6671 | 0.6950 | 0.7027 |
| 3 | 10 | 0.3633 | 0.4589 | 0.0823 | 0.6890 | 0.6978 | 0.6950 | 0.7027 |
| 4 | 2 | 0.7266 | 0.4726 | 0.0548 | 0.3210 | 0.3189 | 0.6950 | 0.7027 |

The later selected residual learners had small alphas and did not change the
ensemble validation/test predictions enough to help. Best-validation restoration
therefore trimmed the saved ensemble back to round 1.

## Best Individual Candidates

The best individual candidate by test accuracy was not selected into the final
ensemble:

| Round | Retry | Batch acc | Weighted error | Train | Val | Test |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3 | 0.2734 | 0.4543 | 0.7124 | 0.6940 | 0.7097 |

The best individual candidate by validation accuracy was the saved first-round
learner:

| Round | Retry | Batch acc | Weighted error | Train | Val | Test |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9 | 0.6563 | 0.2969 | 0.7031 | 0.6950 | 0.7027 |

This suggests that the candidate library contains some useful alternatives, but
the current online AdaBoost selection rule is not extracting a better ensemble.

## Interpretation

This run confirms two things:

- The artifact problem is fixed. We now preserve source for rejected,
  non-selected, selected, and later-restored candidates.
- Increasing the retry budget from 3 to 10 did not solve the CDC boosting
  problem. The bottleneck is not simply too few candidates per round.

The likely failure mode is that the first semantic scorecard already dominates
the ensemble. Later residual learners either have tiny alpha, are anti-correlated
with ordinary accuracy while barely passing weighted error, or do not move
validation predictions. The online round-by-round AdaBoost loop is therefore too
rigid for this LLM-generated code setting.

## Next Steps

1. Try a relaxed library-generation mode where candidates are collected without
   committing weights after every selected round.
2. Add one-call multi-candidate prompting so a single API call can return several
   short candidate functions for local scoring.
3. Re-run this on HTRU2 and mushroom after post-hoc selection exists.

## Post-Hoc Selector Follow-Up

The post-hoc local selector is now implemented in
`program_synthesis/boosted/posthoc_selector.py` and summarized in
`program_synthesis/CODEBOOST_POSTHOC_SELECTOR_B256.md`.

On this same 40-candidate library:

- `weighted_greedy` selected 1 learner and matched the online result:
  train/val/test 0.7031 / 0.6950 / 0.7027.
- `uniform_greedy` selected 6 learners and improved to:
  train/val/test 0.7068 / 0.7045 / 0.7062.

Conclusion: local post-hoc selection can extract complementary learners from the
logged library, but this library is still not strong enough to beat the best CDC
semantic `T=1` run.
