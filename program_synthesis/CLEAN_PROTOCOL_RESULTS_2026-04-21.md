# Clean Protocol Results (2026-04-21)

This note records the fair-protocol cleanup requested on 2026-04-21:

1. remove dataset names from prompts
2. remove dataset-specific descriptions from the clean headline setting
3. add anonymization for exposed numeric tabular values via positive affine transforms
4. rerun the headline rows under that clean setting

## Code changes

- `program_synthesis/boosted/boosted_runner.py`
  - added `dataset_context_mode` with `schema`, `none`, and legacy hint modes
  - default clean mode is now `schema`, not dataset-specific hints
  - added `tabular_numeric_transform`
- `program_synthesis/runner.py`
  - added split-level positive-affine numeric transforms for explicit numeric tabular features
- `program_synthesis/baseline_runner.py`
  - added matching numeric-transform support so baselines use the same clean splits
- `program_synthesis/tests/test_boosted_runner.py`
  - updated prompt tests to assert no dataset identity in schema mode
  - added numeric-transform coverage

## Regression status

- `python -m unittest discover program_synthesis\tests`
- status: passed (`53` tests)

## Clean baselines

All baseline rows below use the cleaned protocol and the exact cached split sizes listed in each CSV.

### CDC semantic + positive affine (`fn_o`)

Files:
- `program_synthesis/baseline_results_clean_cdc_semantic_affine_s5.csv`
- `program_synthesis/baseline_results_clean_cdc_semantic_affine_s5.jsonl`

Top baselines:

| Model | Test acc | Std |
| --- | ---: | ---: |
| AdaBoost | 0.7150 | 0.0000 |
| Logistic regression | 0.7095 | 0.0000 |
| Random forest | 0.7072 | 0.0074 |

### Pima named numeric + positive affine (`fn_r`)

Files:
- `program_synthesis/baseline_results_clean_pima_named_numeric_affine_s5.csv`
- `program_synthesis/baseline_results_clean_pima_named_numeric_affine_s5.jsonl`

Top baselines:

| Model | Test acc | Std |
| --- | ---: | ---: |
| AdaBoost | 0.7303 | 0.0000 |
| Logistic regression | 0.7237 | 0.0000 |
| Extra trees | 0.7237 | 0.0216 |

### HTRU2 named numeric + positive affine (`fn_p`)

Files:
- `program_synthesis/baseline_results_clean_htru2_named_numeric_affine_s5.csv`
- `program_synthesis/baseline_results_clean_htru2_named_numeric_affine_s5.jsonl`

Top baselines:

| Model | Test acc | Std |
| --- | ---: | ---: |
| Gradient boosting | 0.9320 | 0.0000 |
| XGBoost | 0.9304 | 0.0011 |
| AdaBoost | 0.9280 | 0.0000 |

## Clean CodeBoost headline rows

### CDC semantic, schema-only prompt, positive affine, 5 trials

Files:
- `program_synthesis/clean_results_artifacts/cdc_semantic_schema_affine_s5_summary.csv`

Aggregate:

| Dataset | Representation | Context | Numeric transform | Trials | Mean test acc | Test std | Best clean baseline | Gap |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| CDC diabetes | semantic | schema | positive affine | 5 | 0.7070 | 0.0125 | 0.7150 | -0.0080 |

Per-trial test accuracies:

| Trial | Test acc |
| --- | ---: |
| 1 | 0.7130 |
| 2 | 0.7075 |
| 3 | 0.6900 |
| 4 | 0.6980 |
| 5 | 0.7265 |

Recorded API cost: `$0.9311`

Interpretation:

- The prior stronger CDC semantic result does not survive the fair protocol.
- Removing dataset-specific hint text materially lowers the clean CDC mean.
- This is the correct number to use for any clean writeup.

### Pima named numeric, no descriptions, positive affine, 40-candidate first-round library

Primary clean rerun:

| Dataset | Representation | Context | Numeric transform | Retries | Train acc | Val acc | Test acc | Best clean baseline | Gap |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pima diabetes | named numeric | none | positive affine | 40 | 0.7266 | 0.8047 | 0.7303 | 0.7303 | 0.0000 |

Files:

- `program_synthesis/clean_results_artifacts/pima_named_numeric_none_affine_r40_s1_summary.csv`
- `program_synthesis/clean_results_artifacts/pima_named_numeric_none_affine_r40_s1_manifest.json`

Recorded API cost: `$4.1169`

Interpretation:

- The clean no-description Pima tie survives the positive-affine fairness ablation.
- The generic `schema` block was not a good choice for Pima; the stronger clean result uses `dataset_context_mode=none`.
- This remains a single-trial headline row. A 5-trial replication would still be needed for a stronger paper claim.

Aborted expensive variant retained for reference:

- clean `schema + positive_affine + r40` 5-trial sweep was stopped after the first completed trial because it was too expensive and underperforming
- saved partial artifact:
  - `program_synthesis/clean_results_artifacts/pima_named_numeric_schema_affine_r40_s5_trial1_manifest.json`
- partial trial 1 result:
  - train `0.7344`
  - val `0.7891`
  - test `0.6842`
  - cost `$4.5188`

### HTRU2 obfuscated row already satisfies the clean protocol

The strongest plain-CodeBoost HTRU2 row was already clean before this pass:

- representation: `obfuscated`
- prompt context: no dataset-specific semantic description
- numeric values: already anonymized by the obfuscated tabular generator
- feature names: anonymous (`x_i`)

Existing clean HTRU2 result:

| Dataset | Representation | Trials | Mean test acc | Test std | Best baseline | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HTRU2 | obfuscated | 5 | 0.9032 | 0.0204 | 0.9340 | -0.0308 |

Reference file:

- `program_synthesis/CODEBOOST_MATCHED_RESULTS.md`

So HTRU2 does not require a new clean rerun to answer the fairness question. The clean-protocol work is really about CDC semantic and any named-feature settings such as Pima.
