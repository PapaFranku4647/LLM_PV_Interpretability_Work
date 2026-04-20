# HTRU2 Second-Dataset Push

Run date: 2026-04-20.

Goal: find a second non-CDC dataset where CodeBoost can beat or at least cross
the strongest matched classical baseline. HTRU2 is the only plausible candidate:
the earlier obfuscated 5-trial run reached a best test accuracy of 0.9300
against the best saved baseline at 0.9340.

## Representation Added

Added `--tabular-representation named_numeric` for HTRU2. It keeps the semantic
feature names but preserves raw numeric values instead of qualitative bins:

- `profile_mean`
- `profile_stdev`
- `profile_skewness`
- `profile_kurtosis`
- `dm_snr_mean`
- `dm_snr_stdev`
- `dm_snr_skewness`
- `dm_snr_kurtosis`

The initial context says to use numeric thresholds. A later calibrated variant
also gives HTRU2-specific threshold ranges from local split analysis. This is an
aggressive HTRU2-specific prompt, not the clean generic semantic row.

## Results

All rows below use HTRU2 unless otherwise noted.

| Run | Representation | Model / method | Train / val / test | Test acc | Notes |
| --- | --- | --- | --- | ---: | --- |
| Prior best CodeBoost trial | obfuscated | `protected.gpt-5.2` | 256 / 256 / 2000 | 0.9300 | Best trial from existing 5-trial run. |
| Prior CodeBoost mean | obfuscated | `protected.gpt-5.2` | 256 / 256 / 2000 | 0.9032 | Mean from existing 5-trial run. |
| Prior semantic mean | semantic bins | `protected.gpt-5.2` | 256 / 256 / 2000 | 0.9001 | Five-trial semantic follow-up. |
| Named numeric candidate library | named numeric | `protected.gpt-5.2`, 10 retries | 256 / 256 / 2000 | 0.9005 | Best-val selected online candidate. |
| Named numeric posthoc | named numeric | uniform greedy over 10 saved candidates | 256 / 256 / 2000 | 0.9155 | Improved over online, still below baseline. |
| Obfuscated partial library | obfuscated | `protected.gpt-5.2`, partial 4x10 sweep | 256 / 256 / 2000 | 0.9230 | Endpoint failed after first useful candidate. |
| 5.4 nano named numeric | named numeric | `gpt-5.4-nano`, 10 retries attempted | 256 / 256 / 2000 | 0.8945 | First 3 calls worked; later calls failed. |
| 5.4 full named numeric | named numeric | `gpt-5.4`, 3 retries | 256 / 256 / 2000 | 0.9175 | Clean run, did not improve over 5.2 obfuscated. |
| 512-example named numeric | named numeric | `protected.gpt-5.2` | 512 / 0 / 2000 | 0.9055 | More prompt examples did not help. |
| 512-example obfuscated | obfuscated | `protected.gpt-5.2` | 512 / 0 / 2000 | 0.9005 | More prompt examples did not help. |
| Fresh obfuscated extra seed | obfuscated | `protected.gpt-5.2`, 3 trials | 256 / 256 / 2000 | 0.9170 max | Different split seed, not comparable to the main table. |
| Calibrated named numeric | named numeric + threshold hints | `protected.gpt-5.2`, 3 retries | 256 / 256 / 2000 | 0.9025 | Threshold hints did not help. |

Best saved HTRU2 baseline remains 0.9340. None of the CodeBoost HTRU2 pushes
beat it.

## Local Upper Bound Check

A small local grid on the same 256/256/2000 HTRU2 split found that the dataset is
beatable from 256 training examples:

- best validation-selected gradient boosting row in the reduced grid:
  validation 0.9648, test 0.9310;
- best test-oracle gradient boosting row in the reduced grid:
  validation 0.9570, test 0.9360.

This means the problem is not impossible at this data scale. The issue is that
the LLM-generated rule families are not finding the right threshold ensemble.
Also, if we include stronger gradient-boosting hyperparameter grids in the
baseline table, the HTRU2 baseline likely moves from 0.9340 to about 0.9360.

## Conclusion

HTRU2 is still the best non-CDC secondary dataset, but it is not currently a
CodeBoost win. The best honest result is still the prior obfuscated 0.9300 best
trial, close to but below the 0.9340 baseline.

Do not spend more API budget on model-only HTRU2 reruns. The next viable method
change is to generate a larger, diverse candidate library with cheaper proposers
and select ensembles locally, or to add a threshold-search/distillation module.
The latter would be a new method component and should be labeled separately from
plain CodeBoost.

Follow-up: the threshold-search/distillation module was implemented after this
note. It reached 0.9375 test accuracy on the matched HTRU2 split with
stability-regularized validation selection. See
`program_synthesis/CODEBOOST_THRESHOLD_DISTILLER_HTRU2.md`. This is a positive
HTRU2 result for `CodeBoost + threshold distillation`, not for plain CodeBoost.
