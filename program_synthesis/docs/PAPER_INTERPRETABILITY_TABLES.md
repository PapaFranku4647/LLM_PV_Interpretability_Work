# Paper Interpretability Tables

This file is the canonical paper-facing summary for the raw-artifact interpretability runs in `LLM_ERM_Testing/program_synthesis`.

## CDC Raw-Artifact Interpretability Matrix

Protocol:

- dataset: CDC diabetes
- 5 seeds: 42, 43, 44, 45, 46
- 10 shared stratified-random unseen test samples per seed
- raw textual model artifacts only
- no summarization
- thesis prompt sees the model artifact plus the single unseen sample and predicted label
- coverage and faithfulness are computed locally against the train split afterward

| Method | Test Accuracy | Mean Coverage | Mean Faithfulness |
|---|---:|---:|---:|
| CodeBoost | 0.7070 | 0.1553 | 0.7294 |
| Decision Tree | 0.6645 | 0.1177 | 0.8070 |
| Random Forest | 0.7072 | 0.0429 | 0.3028 |
| Extra Trees | 0.7022 | 0.0117 | 0.1259 |
| AdaBoost | 0.7150 | 0.0348 | 0.3348 |
| Gradient Boosting | 0.6900 | 0.0771 | 0.6957 |
| Hist Gradient Boosting | 0.6825 | 0.0000 | 0.0000 |
| Logistic Regression | 0.7095 | 0.0143 | 0.4434 |
| MLP | 0.7040 | 0.1092 | 0.5680 |
| XGBoost | 0.6856 | 0.1959 | 0.2638 |

## Extended Context

| Method | Accepted Rate | Mean Artifact Chars | LLM Cost |
|---|---:|---:|---:|
| CodeBoost | 0.9200 | 985 | 1.1301 |
| Decision Tree | 1.0000 | 2490 | 0.9492 |
| Random Forest | 0.4000 | 872957 | 5.7998 |
| Extra Trees | 0.2000 | 1258174 | 4.4493 |
| AdaBoost | 0.4200 | 20757 | 1.9346 |
| Gradient Boosting | 0.9600 | 35708 | 2.4602 |
| Hist Gradient Boosting | 0.0000 | 1447015 | 0.0000 |
| Logistic Regression | 0.6600 | 7638 | 1.4927 |
| MLP | 0.8600 | 165363 | 6.5448 |
| XGBoost | 0.2000 | 249294 | 7.3587 |

## Notes

- Means penalize failed cases by counting missing faithfulness as zero.
- That penalty is intentional for the raw-artifact comparison.
- Large ensemble artifacts hit real context-limit failures under this regime.
- A bounded train-preview ablation is feasible, but it is not implemented in the current mainline protocol.
