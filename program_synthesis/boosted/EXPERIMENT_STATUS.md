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
