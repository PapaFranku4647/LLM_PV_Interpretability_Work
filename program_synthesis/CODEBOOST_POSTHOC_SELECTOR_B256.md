# Post-Hoc Candidate-Library Selector

Run completed on 2026-04-18 local time.

This implements and tests a no-API local selector over the saved CodeBoost
candidate library. The selector reads logged `candidate_code` entries from
`attempts.jsonl`, reloads the same dataset split, recompiles every candidate,
precomputes train/validation/test predictions, and then builds a validation
selected ensemble.

Implementation:

- `program_synthesis/boosted/posthoc_selector.py`

Input candidate library:

- `program_synthesis/boosted/runs/semantic_cdc_candidate_library_b256_r10_s1/stratified_diverse/attempts.jsonl`

Output artifacts:

- `program_synthesis/boosted/runs/semantic_cdc_candidate_library_b256_r10_s1/posthoc_weighted_greedy_val/`
- `program_synthesis/boosted/runs/semantic_cdc_candidate_library_b256_r10_s1/posthoc_uniform_greedy_val/`

Aggregate CSV:

- `program_synthesis/codeboost_posthoc_selector_b256_s1.csv`

## Methods

Two selector modes were run:

| Mode | Description |
| --- | --- |
| `weighted_greedy` | Recomputes AdaBoost-style train weights/alphas locally and greedily picks the candidate that maximizes validation accuracy. |
| `uniform_greedy` | Ignores AdaBoost alphas and greedily adds one-vote candidate rules, selecting by validation accuracy. |

Both runs allowed inverted candidates. Inversion means the selector can use
`-h(x)` when a generated candidate is more useful as its complement.

## Results

| Selector | Selected | Train | Val | Test | Stopped reason |
| --- | ---: | ---: | ---: | ---: | --- |
| Online `best_ensemble_val` from library run | 1 | 0.7031 | 0.6950 | 0.7027 | validation restore |
| `weighted_greedy` post-hoc | 1 | 0.7031 | 0.6950 | 0.7027 | no validation improvement |
| `uniform_greedy` post-hoc | 6 | 0.7068 | 0.7045 | 0.7062 | no validation improvement |

The uniform post-hoc selector improves over the online ensemble from the same
candidate library:

- validation: +0.0095
- test: +0.0035

It still does not beat the prior CDC semantic `T=1` result:

- `T=1` semantic mean test: 0.7173
- `T=1` semantic best test: 0.7300

## Selected Uniform Ensemble

| Step | Attempt | Round | Retry | Direction | Train | Val | Test |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9 | 1 | 9 | +1 | 0.7031 | 0.6950 | 0.7027 |
| 2 | 7 | 1 | 7 | +1 | 0.7075 | 0.6960 | 0.7072 |
| 3 | 1 | 1 | 1 | -1 | 0.7022 | 0.6990 | 0.7047 |
| 4 | 32 | 4 | 2 | -1 | 0.7020 | 0.6995 | 0.7031 |
| 5 | 30 | 3 | 10 | +1 | 0.7036 | 0.7000 | 0.7039 |
| 6 | 37 | 4 | 7 | -1 | 0.7068 | 0.7045 | 0.7062 |

Attempt 39 would keep validation tied at 0.7045 and move test to 0.7054, so the
selector stopped at six candidates.

## Interpretation

This confirms that local post-hoc selection is useful. The saved library did
contain complementary rules that the online AdaBoost loop did not keep. The
strict weighted selector still collapsed to the first learner, but uniform
validation-greedy selection extracted a better ensemble from the same 40
candidate functions.

The gain is not large enough yet. The next candidate-library work should make
library generation less tied to online weight updates, then select locally:

- collect diverse first-round candidates across seeds/samplers,
- add one-call multi-candidate prompting,
- keep inverted-candidate support,
- run uniform and weighted selectors across CDC, HTRU2, and mushroom,
- compare against the `T=1` semantic scorecard and tabular baselines.
