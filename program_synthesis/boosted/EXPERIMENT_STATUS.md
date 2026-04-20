# CodeBoost Experiment Status

Last updated: 2026-04-20.

This note preserves the current experimental state before repository cleanup. Large run
artifacts are intentionally kept on disk and ignored by git; this file records the
important locations and conclusions so we do not lose the experiment context.
The consolidated index is now `program_synthesis/EXPERIMENT_LEDGER.md`; keep
that file updated when adding new result docs or run families.

## Current Focus

- Primary target so far: CDC diabetes (`fn_o`, length 21).
- Strongest current method: semantic CDC rows with named features and qualitative
  bins, one synthesized program (`T=1`), batch size 256, full-train weak-error
  gate, and best-valid fallback.
- Current non-CDC method under test: hybrid rows with named fields, qualitative
  bins, z-score numeric features, category code tokens, and missingness markers.
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
- New model comparison:
  `program_synthesis/boosted/runs/model_compare_cdc_semantic_t1_b256_s5/`
  - `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.4-nano` via Azure Responses API
  - `reasoning_effort=medium`
  - same CDC semantic `T=1`, batch 256, 5-trial setup as the matched 5.2 row
  - full 5.4 mean test accuracy: 0.6995
  - 5.4-mini mean test accuracy: 0.6865
  - 5.4-nano mean test accuracy: 0.7067
  - conclusion: no 5.4-family direct row improved over the existing 5.2 matched
    row at 0.7173. Nano was the strongest 5.4-family direct row and is the best
    candidate for cheap proposer sweeps.

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
- HTRU2 semantic follow-up: mean test 0.9001, std 0.0089, best trial 0.9120,
  best baseline 0.9340.

Conclusion: HTRU2 remains the best non-CDC secondary dataset, but the semantic
bin-only representation did not improve it. The likely issue is that HTRU2's
numeric threshold detail matters more than qualitative bin labels.

The CDC model comparison is saved in
`program_synthesis/CODEBOOST_MODEL_COMPARISON.md`, with aggregate CSV at
`program_synthesis/codeboost_model_comparison_cdc_semantic.csv`.

- `protected.gpt-5.2` chat-completions row: mean test 0.7173.
- `gpt-5.4` Azure Responses row: mean test 0.6995.
- `gpt-5.4-mini` Azure Responses row: mean test 0.6865.
- `gpt-5.4-nano` Azure Responses row: mean test 0.7067.
- Best CDC baseline: logistic regression, 0.7225.

Conclusion: this does not change the current paper story. The best CDC
CodeBoost row is still the semantic 5.2 matched run, and larger model quality
alone is not enough to solve the large-train-set boosting issue. Nano is the
only 5.4-family result worth using immediately, and only as a cheaper candidate
generator rather than as a direct accuracy upgrade.

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

Hybrid tabular representation is now wired for mushroom, HTRU2, and chess
through `--tabular-representation hybrid`. This creates:

- Mushroom numeric size fields as `cap_diameter_bin` plus `cap_diameter_z`
  style pairs, while categorical values preserve readable labels, missingness,
  and original code tokens such as `convex|missing_no|code_x`.
- HTRU2 numeric fields as named `_bin` and `_z` pairs, e.g.
  `profile_skewness_bin` and `profile_skewness_z`.
- Chess values as readable labels plus UCI code tokens, e.g. `true|code_t`.

Validation completed before API pilots:

- `python -m py_compile src\data_handler.py program_synthesis\boosted\boosted_runner.py program_synthesis\baseline_runner.py`
- `python -m unittest discover program_synthesis\tests` passed 36 tests.
- Hybrid baseline smoke passed for mushroom, HTRU2, and chess using 20/10/20
  splits with a decision tree.

The hybrid pilot is now saved in
`program_synthesis/CODEBOOST_HYBRID_PILOT.md`, with aggregate CSV at
`program_synthesis/codeboost_hybrid_pilot_t1_b256_s1.csv`.

Hybrid pilot test accuracies:

- Mushroom hybrid: 0.6115 vs semantic pilot 0.6380, obfuscated pilot 0.5755,
  and best baseline 0.8528.
- HTRU2 hybrid: 0.8770 vs semantic pilot 0.8950, obfuscated pilot 0.8830, and
  best baseline 0.9340.

Conclusion: hybrid is worth preserving as an ablation, but it is not the next
accuracy lever. Do not run a 5-trial hybrid aggregate from these numbers. The
next serious method step should be the diverse/residual sampler, or stronger
task-specific feature descriptions for chess.

The diverse/residual sampler is now wired as `--sampling-strategy
stratified_diverse`. It uses semantic prompts but selects batches from
high-weight mistakes, low-margin boundary examples, correct anchors, and
feature-diverse fill after each accepted learner. It also supports validation
early stopping through `--early-stop-val-patience` and
`--restore-best-val-ensemble`.

The CDC stratified-diverse pilot is now saved in
`program_synthesis/CODEBOOST_STRATIFIED_DIVERSE_PILOT.md`, with aggregate CSV at
`program_synthesis/codeboost_semantic_cdc_stratified_diverse_t4_b64_b128_s1.csv`.

CDC stratified-diverse pilot test accuracies:

- Batch 64: final test 0.7076, final val 0.6985, kept 1 round after
  best-validation restoration.
- Batch 128: final test 0.6902, final val 0.6800, kept 1 round after
  best-validation restoration.

Batch accuracy was not predictive in this pilot: across 11 candidates, the
batch/train correlation was about 0.06 and batch/test correlation was about
0.13. Do not optimize prompt-batch accuracy alone.

Conclusion: the sampler machinery is useful and preserved, but small-batch
sequential residual boosting did not improve CDC. Next sampler work should use
batch 256 or a candidate-library selection loop that generates several weak
programs and chooses by full-train/validation behavior.

Generic sampler update: sampler diversity vectors now use fixed-size hashed
features, so they are not tied to CDC's feature count. Added
`feature_diverse`, `label_balanced_diverse`, and candidate-library selection via
`--candidate-selection best_ensemble_val`. Next run should compare those methods
at batch 256 before spending on larger sweeps.

Batch-256 sampler comparison is now complete and saved in
`program_synthesis/CODEBOOST_SAMPLER_COMPARE_B256.md`.

Results on CDC semantic, 10000/2000/10000 train/val/test, one seed, batch 256,
3 rounds, 3 retries, `best_ensemble_val`, weak-error gate 0.49:

- `feature_diverse`: test 0.6965, val 0.6880.
- `label_balanced_diverse`: test 0.7036, val 0.6905.
- `stratified_diverse`: test 0.7046, val 0.6900.

All three restored to a one-learner ensemble. This confirms the generic sampler
machinery, but it is still an accuracy negative result relative to the CDC
semantic `T=1` 5-trial mean of 0.7173 and best of 0.7300. Next sampler work
should add candidate source logging, then build a larger offline candidate
library for local validation/greedy ensemble selection.

Candidate source logging is now implemented. Attempt JSONL artifacts include
full candidate source through `candidate_code` and `candidate_source_history`;
CSV artifacts keep hashes/counts/sizes but omit full source blobs.

The larger CDC candidate-library sweep is now saved in
`program_synthesis/CODEBOOST_CANDIDATE_LIBRARY_B256_R10.md`, with raw artifacts
under
`program_synthesis/boosted/runs/semantic_cdc_candidate_library_b256_r10_s1/`.

Configuration: CDC semantic, 10000/2000/10000 train/val/test, batch 256, one
seed, 4 rounds, 10 retries per round, `stratified_diverse`,
`best_ensemble_val`, weak-error gate 0.49, validation drop tolerance 0.005,
early-stop patience 3, restore best-validation ensemble.

Results:

- Final train/val/test: 0.7031 / 0.6950 / 0.7027.
- API attempts: 40.
- Candidates with logged source: 40.
- Estimated cost: $2.9969.
- Saved rounds after restoration: 1.
- Selected candidates before restoration: 4.
- Best individual test candidate: round 2 retry 3, test 0.7097, val 0.6940.

Conclusion: increasing retries from 3 to 10 did not solve CDC boosting. The
source-logging artifact problem is fixed, but the next method step should be
post-hoc local ensemble selection over the saved candidate library rather than
more online AdaBoost rounds.

Post-hoc local selection is now implemented in
`program_synthesis/boosted/posthoc_selector.py` and summarized in
`program_synthesis/CODEBOOST_POSTHOC_SELECTOR_B256.md`.

Results on the 40-candidate CDC library:

- `weighted_greedy` with inverted candidates selected 1 learner:
  train/val/test 0.7031 / 0.6950 / 0.7027.
- `uniform_greedy` with inverted candidates selected 6 learners:
  train/val/test 0.7068 / 0.7045 / 0.7062.

Conclusion: post-hoc selection does help. Uniform validation-greedy selection
found complementary rules that the online AdaBoost-style loop missed. The gain is
still too small to beat the CDC semantic `T=1` best result, so the next lever is
better candidate-library generation rather than more online boosting retries.

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
