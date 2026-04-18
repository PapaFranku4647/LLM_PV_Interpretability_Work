# Boosted Program Synthesis

This folder contains an AdaBoost-style experiment driver for symbolic program synthesis.

Current focus:
- Target dataset: `cdc_diabetes` (`fn_o`)
- Split style: train/test by default (`--val-size 0`)
- Weak learner: one LLM-synthesized Python function per round
- Ensemble output: both a manifest of component learners and an `ensemble.py` wrapper

## What it does

For each boosting round:
1. Sample a weighted training batch from the train split.
2. Prompt the model to synthesize one Python classifier from that batch.
3. Optionally run `--repair-rounds K` repair prompts that pass the current code plus its misclassified batch examples back to the model.
4. Evaluate the best batch candidate on the full train split with the current example weights.
5. Accept the learner if its weighted train error is below the threshold.
6. Update the example distribution and continue.

The runner writes:
- `summary.csv` - final train/test accuracy per trial and batch size
- `attempts.csv` / `attempts.jsonl` - one row per proposal attempt
- `outputs/<target>/L<length>/batch<batch>/trial<trial>/manifest.json`
- `outputs/<target>/L<length>/batch<batch>/trial<trial>/ensemble.py`

TAMU / Azure-compatible access:
- The boosted runner accepts `--api-key`, `--api-base-url`, `--azure-endpoint`, and `--api-version`.
- Environment aliases also work: `TAMU_API_KEY`, `TAMU_AZURE_ENDPOINT`, and `TAMU_API_VERSION`.
- A lightweight smoke checker lives at `program_synthesis/boosted/tamu_api_smoke.py`.

## Example

```bash
python program_synthesis/boosted/boosted_runner.py \
  --functions fn_o \
  --lengths 21 \
  --train-size 200 \
  --test-size 1000 \
  --boost-rounds 8 \
  --batch-sizes 32 64 128 \
  --round-retries 3 \
  --repair-rounds 3 \
  --num-trials 3 \
  --enable-code-interpreter
```

```bash
python program_synthesis/boosted/tamu_api_smoke.py \
  --model gpt-5 \
  --api-mode responses
```

## Notes

- `--val-size` defaults to `0`, so the dataset split is train/test only.
- If you later set `--val-size > 0`, the runner will log validation metrics as well.
- `--tabular-representation semantic` enables named features and readable bins/categories for mushroom, HTRU2, chess, and CDC.
- `--tabular-representation hybrid` keeps named fields while adding numeric z-scores and category code tokens for non-CDC tabular datasets. CDC defaults to semantic when this flag is used.
- `--cdc-representation semantic` remains available as a CDC-specific override.
- `--sampling-strategy stratified_diverse` builds prompt batches from high-weight mistakes, low-margin boundary examples, correct anchors, and feature-diverse fill. Candidate programs are still accepted only after full weighted-train evaluation.
- `--early-stop-val-patience N --restore-best-val-ensemble` stops on validation stagnation and saves the best-validation ensemble prefix.
- Batch-size sweeps plus the per-round attempt logs are intended to support plotting train/test accuracy against boosting round `T`.
- This reuses the provider and dataset machinery from `program_synthesis/runner.py`, but keeps outputs isolated under `program_synthesis/boosted/`.

## Semantic Pilot Commands

```bash
python program_synthesis/boosted/boosted_runner.py \
  --provider openai \
  --api-mode chat_completions \
  --functions fn_n \
  --lengths 20 \
  --train-size 256 \
  --val-size 256 \
  --test-size 2000 \
  --batch-sizes 256 \
  --boost-rounds 1 \
  --num-trials 1 \
  --round-retries 8 \
  --sample-without-replacement \
  --tabular-representation semantic \
  --max-weak-error 0.3025 \
  --accept-best-on-failure \
  --best-fallback-max-weak-error 0.499 \
  --reasoning-effort medium \
  --max-output-tokens 20000 \
  --no-tools \
  --output-dir program_synthesis/boosted/runs/semantic_mushroom_pilot_t1_b256_s1
```

Swap `fn_n --lengths 20` for `fn_p --lengths 8` or `fn_q --lengths 35` to run HTRU2 or chess with the same semantic representation.

## Hybrid Pilot Command

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

This pilot has been run for HTRU2 and mushroom. Results are summarized in
`program_synthesis/CODEBOOST_HYBRID_PILOT.md`; hybrid did not beat the semantic
pilot on either dataset. Chess hybrid is wired, but should stay deprioritized
until the chess feature abbreviations get stronger domain descriptions.

## Stratified Diverse CDC Pilot

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
