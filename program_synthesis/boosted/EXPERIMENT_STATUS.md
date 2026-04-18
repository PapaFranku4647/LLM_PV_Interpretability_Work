# CodeBoost Experiment Status

Last updated: 2026-04-18.

This note preserves the current experimental state before repository cleanup. Large run
artifacts are intentionally kept on disk and ignored by git; this file records the
important locations and conclusions so we do not lose the experiment context.

## Current Focus

- Primary target so far: CDC diabetes (`fn_o`, length 21).
- Strongest current method: semantic CDC rows with named features and qualitative
  bins, one synthesized program (`T=1`), batch size 256, full-train weak-error
  gate, and best-valid fallback.
- Best robust saved aggregate:
  `program_synthesis/boosted/runs/semantic_cdc_t1_quality_gate_e03025_fallback_s30`
  - 30 seeds
  - `T=1`, `batch=256`, `round_retries=8`
  - `cdc_representation=semantic`
  - `sample_without_replacement=True`
  - `max_weak_error=0.3025`
  - `accept_best_on_failure=True`
  - mean test accuracy: 0.70321
  - test standard deviation: 0.00523
  - min/max test accuracy: 0.69268 / 0.71325
  - mean cost: about $0.1445 per run

## Important Ignored Artifacts

- `program_synthesis/boosted/runs/` contains recent semantic, repair, and scale
  sweeps.
- `program_synthesis/boosted/outputs*/` contains earlier matrix and scale sweeps.
- `program_synthesis/meeting_20260403_codeboost_final/` contains the meeting
  bundle comparing CodeBoost and LLM-PV.
- `program_synthesis/outputs_llmpv_baseline_20260401_5seed/` contains the saved
  LLM-PV baseline attempt outputs.
- `data_cache/` contains cached CDC raw and transformed data.

These directories should not be deleted without first copying or summarizing the
results needed for the paper table.

## Meeting Takeaways

- The semantic/binning idea from the weak-learner paper transferred well to our
  executable-code setting.
- Our method is different from per-example LLM classification: we pay the LLM
  synthesis cost once, then run generated code locally.
- The next major requirement is a credible multi-dataset comparison table, not
  another CDC-only sweep.
- We need classical tabular baselines, neural-ish baselines, raw-vs-semantic
  ablations, and multiple datasets (`adult_income`, `mushroom`, `cdc_diabetes`,
  `htru2`, `chess` are already represented in the codebase).
- The likely paper story is interpretability plus competitive/reasonable accuracy,
  not necessarily state-of-the-art accuracy against every tabular method.

## Baseline Snapshot

The first core baseline matrix has been saved in
`program_synthesis/BASELINE_RESULTS.md`, with raw outputs in
`program_synthesis/baseline_results_core.csv` and
`program_synthesis/baseline_results_core.jsonl`. A corrected rerun with XGBoost
installed is saved in `program_synthesis/baseline_results_core_xgboost.csv` and
`program_synthesis/baseline_results_core_xgboost.jsonl`.

Best test accuracies from that run:

- CDC diabetes: logistic regression, 0.7225
- Mushroom: extra trees, 0.8528
- HTRU2: hist gradient boosting, 0.9340
- Chess: hist gradient boosting, 0.9615

The current best semantic CodeBoost CDC result, 0.70321 mean test accuracy, is
about 1.9 percentage points below the strongest CDC baseline but competitive
with several core baselines. XGBoost reached 0.7037 on CDC, essentially tied
with the current semantic CodeBoost result under this small-train setting.

## Matched CodeBoost Pilot

A one-trial matched CodeBoost pilot is saved in
`program_synthesis/CODEBOOST_MATCHED_PILOT.md`, with generated artifacts under
`program_synthesis/boosted/runs/matched_codeboost_pilot_t1_b256_s1/`.

Matched pilot test accuracies:

- CDC diabetes semantic: 0.7060 vs best baseline 0.7225
- Mushroom obfuscated: 0.5755 vs best baseline 0.8528
- HTRU2 obfuscated: 0.8830 vs best baseline 0.9340
- Chess obfuscated: 0.5775 vs best baseline 0.9615

Conclusion: full-budget experiments should continue on CDC semantic and maybe
HTRU2 numeric. Do not spend a 5-trial budget on mushroom or chess until they
have semantic/named feature representations.

The 5-trial matched follow-up is saved in
`program_synthesis/CODEBOOST_MATCHED_RESULTS.md`, with aggregate CSV at
`program_synthesis/codeboost_matched_t1_b256_s5.csv`.

5-trial matched results:

- CDC diabetes semantic: mean test 0.7173, std 0.0102, best trial 0.7300,
  best baseline 0.7225.
- HTRU2 obfuscated: mean test 0.9032, std 0.0204, best trial 0.9300, best
  baseline 0.9340.

Status update: semantic/named tabular representations are now wired for mushroom,
HTRU2, and chess through `--tabular-representation semantic`. This creates:

- Mushroom rows with fields like `cap_shape`, `gill_color`, `stem_width`,
  readable category names, and binned numeric size fields.
- HTRU2 rows with fields like `profile_skewness` and `dm_snr_kurtosis`, all
  binned into qualitative levels.
- Chess rows with UCI KRKPA7 feature names such as `bkblk`, `bkxwp`, `wkpos`,
  and readable values like true/false/none/white.

Next experiment: rerun one-trial semantic pilots for `fn_n`, `fn_p`, and `fn_q`
using `--tabular-representation semantic`; only run 5-trial aggregates where the
pilot closes a meaningful part of the current baseline gap.

The non-CDC semantic pilot is now saved in
`program_synthesis/CODEBOOST_SEMANTIC_PILOT.md`, with aggregate CSV at
`program_synthesis/codeboost_semantic_pilot_t1_b256_s1.csv`.

Semantic pilot test accuracies:

- Mushroom semantic: 0.6380 vs obfuscated pilot 0.5755 and best baseline 0.8528.
- HTRU2 semantic: 0.8950 vs obfuscated pilot 0.8830 and best baseline 0.9340.
- Chess semantic: 0.6310 vs obfuscated pilot 0.5775 and best baseline 0.9615.

Conclusion: semantic rows help, but mushroom/chess are still not ready for
5-trial runs. HTRU2 remains the only plausible non-CDC secondary dataset, but a
hybrid named-feature plus numeric-value representation may be better than bins
only.

## Deferred Sampler Design

Do not implement this during the initial cleanup pass, but preserve it for the
next boosting iteration.

Naive random weighted batches from a large training set are too similar. They
mostly show the same global pattern, so the LLM proposes correlated scorecard-like
programs instead of complementary weak learners. To make boosting behave more like
boosting, the prompt batch should be selected as a diverse residual slice.

Proposed sampler:

1. Vectorize each tabular row only for internal sampling; keep prompts semantic.
2. After each accepted learner, evaluate every train example and update weights.
3. Partition examples by label and current outcome: mistakes, correct positives,
   correct negatives, and optionally high-confidence/low-confidence regions.
4. Within high-weight partitions, choose diverse points using farthest-point
   traversal or k-medoids over the vectorized rows.
5. Build each prompt batch from:
   - 60-70% high-weight diverse mistakes,
   - 20-30% correctly classified anchors from nearby and far regions,
   - 10% global random representatives.
6. Evaluate the proposed learner on full weighted train before acceptance.

Expected implementation shape:

- Add a `--sampling-strategy` flag to `boosted_runner.py`.
- Start with `weighted_random`, `weighted_without_replacement`,
  `mistake_clustered`, and `stratified_diverse`.
- Reuse the existing full-train evaluation and whole-train repair plumbing.
