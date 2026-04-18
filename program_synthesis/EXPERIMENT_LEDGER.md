# CodeBoost Experiment Ledger

Last updated: 2026-04-18.

This is the consolidated experiment map. It does not replace the detailed notes
and raw artifacts; it records where they live, what each method tried to test,
and what currently looks strongest.

## Current Read

- Best overall story: executable, interpretable CodeBoost programs on semantic
  tabular rows, especially CDC diabetes.
- Best saved matched CodeBoost result: CDC semantic `T=1`, batch 256, 5 trials,
  mean test accuracy 0.7173, best trial 0.7300, versus the best matched CDC
  baseline at 0.7225.
- Best robust CDC aggregate before matched baselines: semantic CDC `T=1`,
  quality gate `max_weak_error=0.3025`, best-valid fallback, 30 seeds, mean test
  accuracy 0.70321 with std 0.00523.
- HTRU2 is the only plausible non-CDC secondary dataset so far. It reached
  0.9032 mean test accuracy with obfuscated numeric rows and 0.8950 in the
  one-trial semantic pilot, versus the best matched baseline at 0.9340.
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
- Non-CDC semantic pilot: `program_synthesis/CODEBOOST_SEMANTIC_PILOT.md`
- Hybrid non-CDC pilot: `program_synthesis/CODEBOOST_HYBRID_PILOT.md`
- Running status and cleanup warnings: `program_synthesis/boosted/EXPERIMENT_STATUS.md`
- Baseline raw outputs:
  - `program_synthesis/baseline_results_core.csv`
  - `program_synthesis/baseline_results_core.jsonl`
  - `program_synthesis/baseline_results_core_xgboost.csv`
  - `program_synthesis/baseline_results_core_xgboost.jsonl`
- CodeBoost aggregate CSVs:
  - `program_synthesis/codeboost_matched_pilot_t1_b256_s1.csv`
  - `program_synthesis/codeboost_matched_t1_b256_s5.csv`
  - `program_synthesis/codeboost_semantic_pilot_t1_b256_s1.csv`
  - `program_synthesis/codeboost_hybrid_pilot_t1_b256_s1.csv`

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

Non-CDC semantic one-trial pilot:

| Function | Dataset | Semantic test | Prior obfuscated pilot | Best baseline | Gap | Attempts | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fn_n` | mushroom | 0.6380 | 0.5755 | 0.8528 | -0.2148 | 2 | $0.1665 |
| `fn_p` | HTRU2 | 0.8950 | 0.8830 | 0.9340 | -0.0390 | 1 | $0.0522 |
| `fn_q` | chess | 0.6310 | 0.5775 | 0.9615 | -0.3305 | 8 | $0.7695 |

Hybrid one-trial pilot:

| Function | Dataset | Hybrid test | Semantic pilot | Prior obfuscated pilot | Best baseline | Gap | Attempts | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fn_n` | mushroom | 0.6115 | 0.6380 | 0.5755 | 0.8528 | -0.2413 | 2 | $0.2828 |
| `fn_p` | HTRU2 | 0.8770 | 0.8950 | 0.8830 | 0.9340 | -0.0570 | 1 | $0.1016 |

Chess hybrid was not run because the previous chess semantic pilot was already
far from the baseline and hybrid code tokens do not add the missing chess-domain
feature descriptions.

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
- For HTRU2, pure bins may throw away useful thresholds, but the first hybrid
  `_z` pilot was worse than semantic and obfuscated one-trial pilots.
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
- Candidate programs are still evaluated on full weighted train before
  acceptance.
- Validation early stopping is wired through `--early-stop-val-patience` and
  `--restore-best-val-ensemble`.
- The next test is whether this sampler makes `T>1` CDC semantic boosting beat
  the current strong `T=1` semantic result.

## Next Runs

Do not run a full hybrid budget from the current pilot. The next run should test
the stratified diverse/residual sampler on CDC first, then HTRU2 if CDC improves.

Planned CDC sampler pilot:

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
