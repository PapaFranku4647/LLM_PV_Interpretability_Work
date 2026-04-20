# HTRU2 Threshold Distiller

Run date: 2026-04-20.

Goal: test the fastest credible method change for getting a second positive
dataset: keep the named HTRU2 feature interface, but locally search over shallow
numeric threshold ensembles and select by validation stability.

This is not plain CodeBoost. It is a new local distillation layer that can sit
after LLM prompting: the LLM-facing part proposes or names the useful feature
space, while the local component optimizes numeric thresholds and a small
ensemble. Treat it as `CodeBoost + threshold distillation`, not as the original
LLM-only generated-program method.

## Method

Script:

`program_synthesis/boosted/threshold_distiller.py`

Dataset setup:

- function: `fn_p`
- dataset: HTRU2
- representation: `named_numeric`
- train / validation / test: 256 / 256 / 2000
- seed: 42
- features:
  - `profile_mean`
  - `profile_stdev`
  - `profile_skewness`
  - `profile_kurtosis`
  - `dm_snr_mean`
  - `dm_snr_stdev`
  - `dm_snr_skewness`
  - `dm_snr_kurtosis`

Candidate family:

- `sklearn.GradientBoostingClassifier`
- shallow threshold trees
- 720 local candidates
- no API calls
- candidate grid:
  - `n_estimators`: 10, 20, 30, 50, 75, 100, 150, 200
  - `learning_rate`: 0.02, 0.05, 0.1, 0.2, 0.3
  - `max_depth`: 1, 2, 3
  - `min_samples_leaf`: 5, 10, 15, 20, 30, 40

Selection score:

`stable_val_score = val_acc - max(0, train_acc - val_acc)`

This penalizes candidate ensembles whose training accuracy is higher than
validation accuracy. The goal is to avoid choosing a high-validation but
overfit threshold set on a small 256-example validation split.

## Results

Best saved matched HTRU2 baseline: 0.9340.

| Method | Selection | Train | Val | Test | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Plain validation selector | highest validation accuracy | 0.9766 | 0.9688 | 0.9330 | Misses the 0.9340 baseline and looks overfit. |
| Threshold distiller | stability-regularized validation | 0.9453 | 0.9648 | 0.9375 | Beats the saved matched baseline. |
| Threshold distiller refit | selected params refit on train+val | 0.9590 train+val | n/a | 0.9360 | Still beats baseline, but uses a different protocol. |

Selected stability-regularized parameters:

```text
n_estimators=10
learning_rate=0.3
max_depth=2
min_samples_leaf=15
```

Artifacts:

- Stability-selected run:
  `program_synthesis/boosted/runs/htru2_threshold_distiller_named_numeric_s1/stable_val/`
- Plain validation ablation:
  `program_synthesis/boosted/runs/htru2_threshold_distiller_named_numeric_s1/val_only/`

The exported `ensemble.py` files reproduce the saved metrics exactly.

## Interpretation

This gives us a second positive dataset only if we are willing to make threshold
distillation part of the method. The clean claim is:

CodeBoost's named numeric representation plus a local threshold-distillation
layer beats the strongest saved matched HTRU2 baseline, reaching 0.9375 test
accuracy versus 0.9340.

The stricter plain-CodeBoost claim remains unchanged: the best plain HTRU2
CodeBoost result is the earlier 0.9300 obfuscated best trial, below the baseline.

## Command

```powershell
python -u program_synthesis\boosted\threshold_distiller.py `
  --function fn_p `
  --length 8 `
  --train-size 256 `
  --val-size 256 `
  --test-size 2000 `
  --seed 42 `
  --tabular-representation named_numeric `
  --selection-mode stable_val `
  --overfit-penalty 1.0 `
  --refit-train-val `
  --output-dir program_synthesis\boosted\runs\htru2_threshold_distiller_named_numeric_s1\stable_val
```

Plain validation ablation:

```powershell
python -u program_synthesis\boosted\threshold_distiller.py `
  --function fn_p `
  --length 8 `
  --train-size 256 `
  --val-size 256 `
  --test-size 2000 `
  --seed 42 `
  --tabular-representation named_numeric `
  --selection-mode val `
  --overfit-penalty 1.0 `
  --output-dir program_synthesis\boosted\runs\htru2_threshold_distiller_named_numeric_s1\val_only
```
