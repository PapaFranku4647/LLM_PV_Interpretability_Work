# Paper Accuracy Tables

This file is the canonical paper-facing summary for the clean CodeBoost vs baseline runs in `LLM_PV/program_synthesis`.

## Clean Accuracy Matrix

Clean protocol:

- no dataset names in prompts
- no dataset-specific descriptions
- coordinate-wise positive affine anonymization on numeric features
- same transformed split for CodeBoost and all baselines within a run
- 5 trials on the same transformed split unless noted otherwise

| Dataset | CodeBoost | Decision Tree | Random Forest | Extra Trees | AdaBoost | Gradient Boosting | Hist Gradient Boosting | Logistic Regression | MLP | XGBoost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CDC | 0.7070 | 0.6645 | 0.7072 | 0.7022 | 0.7150 | 0.6900 | 0.6825 | 0.7095 | 0.7040 | 0.6856 |
| Pima | 0.7105 | 0.6974 | 0.7158 | 0.7237 | 0.7303 | 0.7039 | 0.6711 | 0.7237 | 0.6632 | 0.7132 |
| Telco | 0.7569 | 0.7045 | 0.7626 | 0.7495 | 0.7580 | 0.7430 | 0.7320 | 0.7595 | 0.7451 | 0.7243 |
| credit_g | 0.7083 | 0.6759 | 0.7444 | 0.7130 | 0.7361 | 0.6991 | 0.6574 | 0.7315 | 0.6472 | 0.6963 |

## CodeBoost Protocol

| Dataset | CodeBoost Representation | Context Mode | Transform | Batch | Train | Val | Test |
|---|---|---|---|---:|---:|---:|---:|
| CDC | semantic | schema | positive_affine | 256 | 256 | 256 | 2000 |
| Pima | named_numeric | none | positive_affine | 256 | 256 | 128 | 152 |
| Telco | named_numeric | none | positive_affine | 256 | 256 | 256 | 2000 |
| credit_g | named_numeric | none | positive_affine | 256 | 256 | 128 | 216 |

## Notes

- These rows are the current paper table for accuracy.
- The clean protocol reduces direct benchmark contamination concerns but does not remove all prior-knowledge effects because column names remain.
- The current 5-trial summaries measure model variability on one transformed split. They are not yet a transform-seed sweep.
- `export_interpretability_artifacts.py` exports the fitted CodeBoost and baseline artifacts used by the interpretability repo.
