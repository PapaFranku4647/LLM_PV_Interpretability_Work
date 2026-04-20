# CodeBoost Experiment Ledger

Last updated: 2026-04-20.

This is the consolidated experiment map. It does not replace the detailed notes
and raw artifacts; it records where they live, what each method tried to test,
and what currently looks strongest.

## Current Read

- Best overall story: executable, interpretable CodeBoost programs on semantic
  tabular rows, especially CDC diabetes.
- Best saved matched CodeBoost result: CDC semantic `T=1`, batch 256, 5 trials,
  mean test accuracy 0.7173, best trial 0.7300, versus the best matched CDC
  baseline at 0.7225.
- New model comparison: `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.4-nano` via
  Azure Responses with medium reasoning did not beat the existing 5.2 CDC
  semantic row. Nano was the best 5.4-family direct row at 0.7067 mean test
  accuracy; full 5.4 reached 0.6995; mini reached 0.6865.
- Best robust CDC aggregate before matched baselines: semantic CDC `T=1`,
  quality gate `max_weak_error=0.3025`, best-valid fallback, 30 seeds, mean test
  accuracy 0.70321 with std 0.00523.
- HTRU2 is the only plausible non-CDC secondary dataset so far. It reached
  0.9032 mean test accuracy with obfuscated numeric rows and 0.9001 in the
  5-trial semantic follow-up, versus the best matched baseline at 0.9340.
  Semantic bins did not improve HTRU2, likely because threshold detail matters.
- The 2026-04-20 HTRU2 second-dataset push tried named raw numeric features,
  calibrated threshold hints, 5.4 full/nano, 512-example prompts, and local
  post-hoc selection. None beat the 0.9340 baseline; best new comparable row was
  the partial obfuscated 5.2 library at 0.9230. The prior 0.9300 HTRU2 best
  trial remains the strongest CodeBoost HTRU2 result.
- New HTRU2 method component: threshold distillation over named numeric HTRU2
  features reached 0.9375 test accuracy with stability-regularized validation
  selection, beating the saved 0.9340 matched baseline. This is not plain
  CodeBoost; label it as `CodeBoost + threshold distillation`.
- Mushroom and chess improved with semantic rows, but they remain far below
  classical baselines. Do not spend full 5-trial API budgets on them unless a
  pilot closes the gap.
- Hybrid is implemented and saved as an ablation, but the first pilot did not
  improve accuracy. The next serious accuracy lever is sampler diversity or
  stronger task-specific feature descriptions, not another blind hybrid sweep.

## Source Notes And Raw Files

- Baseline snapshot: `program_synthesis/BASELINE_RESULTS.md`
- Matched one-trial CodeBoost pilot: `program_synthesis/CODEBOOST_MATCHED_PILOT.md`
- Matched 5-trial CodeBoost follow-up: `program_synthesis/CODEBOOST_MATCHED_RESULTS.md`
- CDC model comparison: `program_synthesis/CODEBOOST_MODEL_COMPARISON.md`
- Non-CDC semantic pilot: `program_synthesis/CODEBOOST_SEMANTIC_PILOT.md`
- HTRU2 second-dataset push: `program_synthesis/CODEBOOST_HTRU2_SECOND_DATASET_PUSH.md`
- HTRU2 threshold distiller: `program_synthesis/CODEBOOST_THRESHOLD_DISTILLER_HTRU2.md`
- Hybrid non-CDC pilot: `program_synthesis/CODEBOOST_HYBRID_PILOT.md`
- CDC stratified diverse sampler pilot:
  `program_synthesis/CODEBOOST_STRATIFIED_DIVERSE_PILOT.md`
- Running status and cleanup warnings: `program_synthesis/boosted/EXPERIMENT_STATUS.md`
- Baseline raw outputs:
  - `program_synthesis/baseline_results_core.csv`
  - `program_synthesis/baseline_results_core.jsonl`
  - `program_synthesis/baseline_results_core_xgboost.csv`
  - `program_synthesis/baseline_results_core_xgboost.jsonl`
- CodeBoost aggregate CSVs:
  - `program_synthesis/codeboost_matched_pilot_t1_b256_s1.csv`
  - `program_synthesis/codeboost_matched_t1_b256_s5.csv`
  - `program_synthesis/codeboost_model_comparison_cdc_semantic.csv`
  - `program_synthesis/codeboost_semantic_pilot_t1_b256_s1.csv`
  - `program_synthesis/codeboost_hybrid_pilot_t1_b256_s1.csv`
  - `program_synthesis/codeboost_semantic_cdc_stratified_diverse_t4_b64_b128_s1.csv`

Large generated run directories are intentionally ignored by git. Do not delete
these without first saving the useful summaries:

- `program_synthesis/boosted/runs/`
- `program_synthesis/boosted/outputs*/`
- `program_synthesis/meeting_20260403_codeboost_final/`
- `program_synthesis/outputs_llmpv_baseline_20260401_5seed/`
- `data_cache/`

## Baseline Matrix

Baseline run configuration:

- functions: `fn_o`, `fn_n`, `fn_p`, `fn_q`
- datasets: CDC diabetes, mushroom, HTRU2, chess
- train size: 256
- validation size: 256
- test size: 2000
- trials per model: 5
- selection split: validation
- models: decision tree, random forest, extra trees, AdaBoost, gradient
  boosting, hist gradient boosting, logistic regression, MLP, XGBoost

Best matched baselines:

| Function | Dataset | Best model | Test accuracy | Notes |
| --- | --- | --- | ---: | --- |
| `fn_o` | CDC diabetes | logistic regression | 0.7225 | XGBoost 0.7037; random forest 0.7203. |
| `fn_n` | mushroom | extra trees | 0.8528 | XGBoost 0.7921; random forest 0.8277. |
| `fn_p` | HTRU2 | hist gradient boosting | 0.9340 | XGBoost 0.9294; gradient boosting 0.9310. |
| `fn_q` | chess | hist gradient boosting | 0.9615 | XGBoost 0.9499; AdaBoost 0.9560. |

## Representations Tried

| Representation | Rows Look Like | Status |
| --- | --- | --- |
| Obfuscated | `x0:1.23,x1:c2,...` with random numeric affine transforms and anonymous categorical codes. | Poor for mushroom/chess; okay for HTRU2 numeric; useful as raw ablation. |
| CDC semantic | CDC feature names with yes/no fields and five qualitative bins for numeric/ordinal fields. | Strongest current story; best matched trial beats the best CDC baseline. |
| Non-CDC semantic | Mushroom/HTRU2/chess feature names, readable categories, and bins for numeric features. | Helps mushroom/chess relative to obfuscated, but not enough. HTRU2 may lose threshold detail. |
| Hybrid | Numeric fields expose both `_bin` and `_z`; categorical fields expose readable labels plus `code_*` and missingness. | Implemented for mushroom, HTRU2, and chess. First pilot did not improve accuracy. |
| Named numeric | HTRU2 semantic feature names with compact raw numeric values. | Added for the second-dataset push. It preserved thresholds better than bins, but did not beat the HTRU2 baseline. |

## CodeBoost Results

Matched one-trial pilot, train 256, val 256, test 2000, batch 256, `T=1`,
8 retries, without replacement, best-valid fallback:

| Function | Dataset | Representation | Test accuracy | Best baseline | Gap | Attempts | Cost |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fn_o` | CDC diabetes | semantic | 0.7060 | 0.7225 | -0.0165 | 1 | $0.0645 |
| `fn_n` | mushroom | obfuscated | 0.5755 | 0.8528 | -0.2773 | 1 | $0.0970 |
| `fn_p` | HTRU2 | obfuscated | 0.8830 | 0.9340 | -0.0510 | 1 | $0.1935 |
| `fn_q` | chess | obfuscated | 0.5775 | 0.9615 | -0.3840 | 8 | $0.9341 |

Matched 5-trial follow-up:

| Function | Dataset | Representation | Mean test | Std | Min | Max | Best baseline | Gap | Attempts | Total cost |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fn_o` | CDC diabetes | semantic | 0.7173 | 0.0102 | 0.7020 | 0.7300 | 0.7225 | -0.0052 | 6 | $0.4465 |
| `fn_p` | HTRU2 | obfuscated | 0.9032 | 0.0204 | 0.8795 | 0.9300 | 0.9340 | -0.0308 | 5 | $0.7939 |
| `fn_p` | HTRU2 | semantic | 0.9001 | 0.0089 | 0.8875 | 0.9120 | 0.9340 | -0.0339 | 5 | $0.3541 |

HTRU2 second-dataset push:

| Method | Representation | Model / selection | Test accuracy | Takeaway |
| --- | --- | --- | ---: | --- |
| Prior best trial | obfuscated | `protected.gpt-5.2` | 0.9300 | Still the best HTRU2 CodeBoost result. |
| Named numeric posthoc | named numeric | 10 saved 5.2 candidates, uniform greedy | 0.9155 | Posthoc helped, but not enough. |
| Partial obfuscated library | obfuscated | 5.2 candidate library, endpoint failure after first good candidate | 0.9230 | Best new comparable result. |
| 5.4 full named numeric | named numeric | 3 retries, best validation candidate | 0.9175 | Full 5.4 did not improve HTRU2. |
| 5.4 nano named numeric | named numeric | 10 retries attempted | 0.8945 | Nano was not useful here. |
| Calibrated named numeric | named numeric + threshold hints | 5.2, 3 retries | 0.9025 | Threshold hints did not fix it. |

HTRU2 threshold distiller:

| Method | Selection | Train | Val | Test | Takeaway |
| --- | --- | ---: | ---: | ---: | --- |
| Plain validation selector | highest validation accuracy | 0.9766 | 0.9688 | 0.9330 | Overfit; does not beat the 0.9340 baseline. |
| Threshold distiller | stability-regularized validation | 0.9453 | 0.9648 | 0.9375 | First saved HTRU2 positive result, but it is a new local distillation component. |
| Threshold distiller refit | selected params refit on train+val | 0.9590 train+val | n/a | 0.9360 | Different protocol; keep separate from matched train-only selection. |

CDC model comparison at the same semantic `T=1`, batch 256 setting:

| Method | Model | API | Reasoning | Mean test | Std | Min | Max | Best baseline | Gap | Attempts | Cost |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CodeBoost | `protected.gpt-5.2` | chat completions | medium | 0.7173 | 0.0102 | 0.7020 | 0.7300 | 0.7225 | -0.0052 | 6 | $0.4465 |
| CodeBoost | `gpt-5.4-nano` | Responses | medium | 0.7067 | 0.0041 | 0.7000 | 0.7125 | 0.7225 | -0.0158 | 5 | unknown |
| CodeBoost | `gpt-5.4` | Responses | medium | 0.6995 | 0.0108 | 0.6810 | 0.7150 | 0.7225 | -0.0230 | 5 | unknown |
| CodeBoost | `gpt-5.4-mini` | Responses | medium | 0.6865 | 0.0213 | 0.6530 | 0.7080 | 0.7225 | -0.0360 | 10 | unknown |

The 5.4-family results are saved in
`program_synthesis/CODEBOOST_MODEL_COMPARISON.md`. They did not improve the
direct CDC paper story; the strongest current CDC claim is still the 5.2
semantic row. Nano is still useful because it is stable and cheap enough to try
as a large candidate-library proposer.

Non-CDC semantic one-trial pilot:

| Function | Dataset | Semantic test | Prior obfuscated pilot | Best baseline | Gap | Attempts | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fn_n` | mushroom | 0.6380 | 0.5755 | 0.8528 | -0.2148 | 2 | $0.1665 |
| `fn_p` | HTRU2 | 0.8950 | 0.8830 | 0.9340 | -0.0390 | 1 | $0.0522 |
| `fn_q` | chess | 0.6310 | 0.5775 | 0.9615 | -0.3305 | 8 | $0.7695 |

HTRU2 semantic was later expanded to 5 trials and reached 0.9001 mean test
accuracy, still below the obfuscated HTRU2 5-trial mean of 0.9032.

Hybrid one-trial pilot:

| Function | Dataset | Hybrid test | Semantic pilot | Prior obfuscated pilot | Best baseline | Gap | Attempts | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fn_n` | mushroom | 0.6115 | 0.6380 | 0.5755 | 0.8528 | -0.2413 | 2 | $0.2828 |
| `fn_p` | HTRU2 | 0.8770 | 0.8950 | 0.8830 | 0.9340 | -0.0570 | 1 | $0.1016 |

Chess hybrid was not run because the previous chess semantic pilot was already
far from the baseline and hybrid code tokens do not add the missing chess-domain
feature descriptions.

CDC stratified diverse sampler pilot:

| Batch | Train size | Val size | Test size | Kept rounds | Final val | Final test | Attempts | Cost | Takeaway |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 10000 | 2000 | 10000 | 1 | 0.6985 | 0.7076 | 6 | $0.3033 | Later residual learners failed the strict gate or hurt validation. |
| 128 | 10000 | 2000 | 10000 | 1 | 0.6800 | 0.6902 | 5 | $0.2848 | Validation restoration trimmed back to the first learner. |

Across the 11 attempted candidates, batch accuracy had low correlation with
full-train accuracy, about 0.06, and test accuracy, about 0.13. In this pilot,
raising prompt-batch accuracy was not a reliable route to higher overall
accuracy.

## Other Saved Run Families

These are generated from the `summary.csv` files under
`program_synthesis/boosted/runs/`.

| Run family | N | Mean test | Min | Max | Attempts | Mean cost | Takeaway |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `t8_b16_s5/T8` | 5 | 0.6679 | 0.6168 | 0.7011 | 43 | $0.4963 | Small batches made weak learners unstable. |
| `t8_b32_s5/T8` | 5 | 0.6783 | 0.6675 | 0.6896 | 49 | $1.2178 | More calls, no accuracy win. |
| `t8_b64_s5/T8` | 5 | 0.6591 | 0.6053 | 0.6917 | 39 | $1.1798 | Poor batch-size regime. |
| `t8_b128_s5/T8` | 5 | 0.6935 | 0.6631 | 0.7146 | 41 | $0.9988 | Best of early `T=8` batch sweep, still below `T=1` CDC semantic. |
| `t8_b256_s5/T8` | 5 | 0.6961 | 0.6911 | 0.7023 | 45 | $1.0276 | Stable but not better than strong one-shot semantic. |
| `repair_vs_control_medium_b256_s5/t4_control` | 5 | 0.6870 | 0.6447 | 0.7094 | 28 | $0.5828 | Repair not needed for `T=4`. |
| `repair_vs_control_medium_b256_s5/t4_repair` | 5 | 0.6823 | 0.6665 | 0.7181 | 40 | $0.7583 | Extra repair cost did not improve mean. |
| `repair_vs_control_medium_b256_s5/t8_control` | 5 | 0.6870 | 0.6774 | 0.6965 | 45 | $1.0209 | More rounds stayed correlated. |
| `repair_vs_control_medium_b256_s5/t8_repair` | 5 | 0.7001 | 0.6856 | 0.7091 | 86 | $1.7751 | Best repair family mean, but expensive and still below matched CDC semantic. |
| `semantic_cdc_vs_known_medium_b256_s5/semantic_t1_control` | 25 | 0.6991 | 0.6734 | 0.7106 | 25 | $0.0718 | Early semantic CDC control across many seeds. |
| `semantic_cdc_vs_known_medium_b256_s5/semantic_t8_control` | 5 | 0.6926 | 0.6792 | 0.7055 | 18 | $0.2683 | More rounds did not help. |
| `semantic_cdc_vs_known_medium_b256_s5/semantic_t8_gated_repair` | 5 | 0.6970 | 0.6839 | 0.7125 | 47 | $0.9992 | Gated repair not worth the cost. |
| `semantic_cdc_t1_quality_gate_e030_s20/T1` | 20 | 0.6694 | 0.0000 | 0.7135 | 53 | $0.1922 | Too strict; failures scored as zero. |
| `semantic_cdc_t1_quality_gate_e03025_s30/T1` | 30 | 0.6800 | 0.0000 | 0.7130 | 52 | $0.1236 | Still loses runs to rejection. |
| `semantic_cdc_t1_quality_gate_e0305_s30/T1` | 30 | 0.7003 | 0.6891 | 0.7139 | 42 | $0.0983 | Strong stable gate, but lower than fallback variant. |
| `semantic_cdc_t1_quality_gate_e03025_fallback_s30/T1` | 30 | 0.7032 | 0.6927 | 0.7133 | 58 | $0.1445 | Best robust 30-seed CDC aggregate. |
| `semantic_cdc_whole_train_70_20_10_smoke_s5/T1` | 2 | 0.6941 | 0.6930 | 0.6953 | 8 | $0.7525 | Whole-train repair mix was costly and not promising. |

## Prompting And Boosting Lessons

- `T=1` with a strong semantic prompt is currently better than naive multi-round
  boosting. Multi-round batches from the large train set are too similar, so the
  generated weak learners are correlated and do not add much diversity.
- Quality gates help only when paired with best-valid fallback. Strict rejection
  without fallback can produce zero-result runs.
- Repair prompts sometimes improve the best individual run, but they cost a lot
  and have not improved the mean enough to justify broad use.
- Semantic context matters more than round count on the current pipeline.
- For HTRU2, pure bins throw away useful thresholds. The 5-trial semantic
  follow-up was slightly worse than obfuscated, and the first hybrid `_z` pilot
  was worse than semantic and obfuscated one-trial pilots.
- For mushroom, preserving original code tokens and missingness did not improve
  over readable semantic categories.
- For chess, UCI abbreviations are still too opaque. A proper chess semantic
  context would need Shapiro KRKPA7 feature descriptions; until then chess is a
  weak paper target.

## Deferred Diverse Sampler

Do not lose this idea; it is the likely route to making real boosting work on
large train sets.

Problem: naive random or weighted-random batches from a large tabular train set
look too similar. The LLM keeps learning the same global scorecard, so later
boosting rounds are not distinct enough.

Proposed sampler:

1. Vectorize tabular rows internally for sampling only; keep prompts semantic.
2. After each accepted learner, evaluate every train example and update weights.
3. Partition examples by label and current outcome: mistakes, correct positives,
   correct negatives, plus optionally high-confidence and low-confidence regions.
4. Within high-weight partitions, choose diverse points with farthest-point
   traversal or k-medoids over the vectorized rows.
5. Build prompt batches from roughly 60-70% high-weight diverse mistakes,
   20-30% correctly classified anchors from nearby and far regions, and 10%
   global random representatives.
6. Still evaluate every proposed learner on full weighted train before accepting
   it.

Implementation status:

- `--sampling-strategy stratified_diverse` is now implemented. It starts with
  label-balanced diverse sampling before any learner exists, then mixes
  high-weight mistakes, low-margin boundary examples, correct anchors, and
  diverse fill after accepted rounds.
- Generic sampler variants `feature_diverse` and `label_balanced_diverse` are
  also wired. The diversity vectorizer now uses fixed-size hashed features, so
  sampler methods can run on arbitrary tabular feature counts and categorical
  cardinalities without a dense one-hot blowup.
- Candidate-library selection is wired through `--candidate-selection`; use
  `best_ensemble_val` to evaluate all retries in a round and select locally by
  full-train/validation ensemble behavior.
- Candidate programs are still evaluated on full weighted train before
  acceptance.
- Validation early stopping is wired through `--early-stop-val-patience` and
  `--restore-best-val-ensemble`.
- The first batch-64/128 CDC sampler pilot did not beat the current strong `T=1`
  semantic result. Next sampler work should use a candidate-library selection
  loop or batch 256, not more small-batch blind sequential boosting.

## Next Runs

Do not run a full hybrid budget from the current pilot. The first CDC
stratified-diverse sampler pilot is complete and did not improve on `T=1`.

Completed CDC sampler pilot:

```bash
python program_synthesis/boosted/boosted_runner.py \
  --provider openai \
  --api-mode chat_completions \
  --functions fn_o \
  --lengths 21 \
  --train-size 10000 \
  --val-size 2000 \
  --test-size 10000 \
  --seed 42 \
  --batch-sizes 64 128 \
  --boost-rounds 4 \
  --round-retries 3 \
  --resample-each-retry \
  --sampling-strategy stratified_diverse \
  --tabular-representation semantic \
  --max-weak-error 0.45 \
  --early-stop-val-patience 2 \
  --restore-best-val-ensemble \
  --reasoning-effort medium \
  --max-output-tokens 20000 \
  --no-tools \
  --output-dir program_synthesis/boosted/runs/semantic_cdc_stratified_diverse_t4_b64_b128_s1
```

The already-run hybrid HTRU2 command was:

```bash
python program_synthesis/boosted/boosted_runner.py \
  --provider openai \
  --api-mode chat_completions \
  --functions fn_p \
  --lengths 8 \
  --train-size 256 \
  --val-size 256 \
  --test-size 2000 \
  --seed 42 \
  --batch-sizes 256 \
  --boost-rounds 1 \
  --num-trials 1 \
  --round-retries 8 \
  --sample-without-replacement \
  --tabular-representation hybrid \
  --max-weak-error 0.3025 \
  --accept-best-on-failure \
  --best-fallback-max-weak-error 0.499 \
  --reasoning-effort medium \
  --max-output-tokens 20000 \
  --no-tools \
  --output-dir program_synthesis/boosted/runs/hybrid_codeboost_pilot_t1_b256_s1/fn_p_htru2
```

The mushroom command was the same with `fn_n --lengths 20` and output directory
`program_synthesis/boosted/runs/hybrid_codeboost_pilot_t1_b256_s1/fn_n_mushroom`.
Skip chess unless a new chess feature-description prompt is added or the budget
is explicitly worth spending.

## Batch-256 Sampler Comparison

The batch-256 CDC sampler comparison is now saved in
`program_synthesis/CODEBOOST_SAMPLER_COMPARE_B256.md`, with aggregate CSV at
`program_synthesis/codeboost_sampler_compare_b256_s1.csv` and raw artifacts under
`program_synthesis/boosted/runs/semantic_cdc_sampler_compare_b256_s1/`.

Configuration: CDC semantic, 10000/2000/10000 train/val/test, batch 256, one
seed, 3 boosting rounds, 3 retries per round, resample each retry,
`best_ensemble_val` candidate selection, validation drop tolerance 0.005,
max weak error 0.49, early stopping patience 2, restore best-validation
ensemble.

Test accuracies:

- `feature_diverse`: final test 0.6965, final val 0.6880, saved 1 round after
  best-validation restoration.
- `label_balanced_diverse`: final test 0.7036, final val 0.6905, saved 1 round
  after best-validation restoration.
- `stratified_diverse`: final test 0.7046, final val 0.6900, saved 1 round
  after best-validation restoration.

Conclusion: the generic sampler and candidate-library machinery works, but this
batch-256 comparison still did not improve over the current CDC semantic `T=1`
scorecard. All three methods selected later weak learners during the run, but
the best-validation restored ensemble kept only the first learner. Before larger
sweeps, add candidate-program logging for every attempt so we preserve rejected
and later-restored learner source, then try larger candidate libraries and
post-hoc local ensemble selection.

## Candidate Source Logging and Larger Candidate Library

Candidate source logging is now implemented in `boosted_runner.py`. Attempt
JSONL artifacts include full candidate source through `candidate_code` and
`candidate_source_history`. CSV artifacts omit those large blobs but keep source
hashes, sizes, counts, and stages.

The larger CDC candidate-library sweep is saved in
`program_synthesis/CODEBOOST_CANDIDATE_LIBRARY_B256_R10.md`, with aggregate CSV
at `program_synthesis/codeboost_candidate_library_b256_r10_s1.csv` and raw
artifacts under
`program_synthesis/boosted/runs/semantic_cdc_candidate_library_b256_r10_s1/`.

Configuration: CDC semantic, 10000/2000/10000 train/val/test, batch 256, one
seed, 4 rounds, 10 retries per round, `stratified_diverse`,
`best_ensemble_val`, validation drop tolerance 0.005, max weak error 0.49,
early-stop patience 3, restore best-validation ensemble.

Results:

- Final train/val/test: 0.7031 / 0.6950 / 0.7027.
- API attempts: 40.
- Candidates with logged source: 40.
- Estimated cost: $2.9969.
- Saved rounds after restoration: 1.
- Selected candidates before restoration: 4.
- Best individual test candidate: round 2 retry 3, train/val/test
  0.7124 / 0.6940 / 0.7097.

Conclusion: the candidate library has useful alternatives, but online
AdaBoost-style selection still does not extract a better CDC ensemble. The next
step should be post-hoc local ensemble selection over logged candidates, plus a
library-generation mode that collects candidates without committing weights after
each selected round.

## Post-Hoc Local Selection

Post-hoc local selection is implemented in
`program_synthesis/boosted/posthoc_selector.py` and summarized in
`program_synthesis/CODEBOOST_POSTHOC_SELECTOR_B256.md`, with aggregate CSV at
`program_synthesis/codeboost_posthoc_selector_b256_s1.csv`.

It reloads the same dataset split, recompiles every logged candidate from
`candidate_code`, precomputes predictions, and greedily builds a validation
selected ensemble without API calls.

CDC 40-candidate library results:

- Online `best_ensemble_val`: selected 1 learner, train/val/test
  0.7031 / 0.6950 / 0.7027.
- Post-hoc `weighted_greedy`: selected 1 learner, train/val/test
  0.7031 / 0.6950 / 0.7027.
- Post-hoc `uniform_greedy`: selected 6 learners, train/val/test
  0.7068 / 0.7045 / 0.7062.

Conclusion: uniform post-hoc voting is a real improvement over the online
ensemble from the same saved candidates, but the candidate library is still not
strong enough. Next work should generate broader libraries, especially
multi-candidate prompts and diverse first-round candidates across seeds/samplers.
