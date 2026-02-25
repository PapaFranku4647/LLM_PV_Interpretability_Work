# LLM-PV (OpenAI Responses API)

Goal: prompt a reasoning-capable model to generate executable Python that implements a hidden mapping. Generated code is compiled and evaluated on validation/test splits with deterministic data generation.

## Key points
- Deterministic dataset cache per `(fn, length, sizes, seed)` under `program_synthesis/datasets/<target>/L<length>/seed<derived>/`.
- Strict output contract: model should return `{"code": "<python function>"}`.
- Early stop: stop attempts for a trial when validation reaches `1.0`.
- Reproducibility metadata is written into every row (`run_id`, model settings, seeds, split sizes, dataset path).

## Setup
```bash
conda env create -f environment.yaml
conda activate llm_pv
export OPENAI_API_KEY=sk-...
```

## TAMU API setup (GPT-5.2)
If you are using the TAMU gateway (OpenAI-compatible chat completions endpoint), set:

```bash
export TAMU_API_KEY=...
export API_MODE=chat_completions
export API_BASE_URL="https://<tamu-host>/openai"
```

Then run with:

```bash
python program_synthesis/runner.py \
  --model gpt-5.2 \
  --reasoning-effort medium \
  --tool-choice none
```

Notes:
- `API_BASE_URL` should be the TAMU gateway base that serves `/chat/completions`.
- If both `TAMU_API_KEY` and `OPENAI_API_KEY` are set, `TAMU_API_KEY` is used.
- For OpenAI Responses API, keep `API_MODE=responses` (default) and use your OpenAI key.

## Minimal run
```bash
python program_synthesis/runner.py \
  --functions fn_a \
  --lengths 50 \
  --attempts 5 \
  --num-trials 3 \
  --concurrency 1 \
  --timeout 1200
```

## Important runner choice
For research runs and any result you plan to report, use:

```bash
python program_synthesis/runner_val_selection.py ...
```

Why:
- `runner_val_selection.py` selects the best attempt per trial by `val_acc`.
- This avoids test-set leakage from selecting candidates by `test_acc`, which can inflate test results.

For tabular tasks (`fn_m`, `fn_n`, `fn_o`, `fn_p`, `fn_q`) lengths are fixed by task metadata, so `--lengths` is optional.

## Phase 2.3 Code1 verification
Step 2.3 uses a Code1-only verifier pipeline (no condition parser):
- `code1_verifier.py` writes `check_conditions(x)` from raw thesis conditions.
- A verifier LLM judges Code1 and supplies thesis-grounded positive/negative testcases.
- Code1 is accepted only if semantic judgement is `pass` and testcase execution passes.
- On failure, one retry is attempted; if still failing, the sample is marked invalid and processing continues.

`run_step22_live_once.py` now includes this flow and writes Code1 verification diagnostics into `summary.json`.

### Step 2.3 live matrix (multi-task, multi-seed, equation metrics)
Use `thesis_runner.py` (the main pipeline orchestrator) for end-to-end matrix runs with heavy logging and equation-level metrics.

Example (all tabular tasks, 3 seeds, 3 samples/seed, `gpt-5-mini` for all calls):

```bash
python program_synthesis/thesis_runner.py \
  --functions fn_m fn_n fn_o fn_p fn_q \
  --seeds 2201 2202 2203 \
  --samples-per-seed 3 \
  --attempts 3 \
  --num-trials 1 \
  --train-size 100 \
  --val-size 100 \
  --test-size 3000 \
  --prompt-variant explain \
  --model gpt-5-mini \
  --reasoning-effort minimal \
  --text-verbosity low \
  --max-output-tokens 1400 \
  --code1-model gpt-5-mini \
  --code1-verifier-model gpt-5-mini \
  --code1-reasoning-effort minimal \
  --code1-text-verbosity low \
  --code1-max-output-tokens 1200
```

Useful options:
- `--thesis-prompt-version v1|v2` to switch between original and trace-guided thesis prompts.
- `--compute-baselines` to add trivial baseline metrics (`always_0`, `always_1`) into `overall_summary.json`.
- `--auto-split --train-cap 200 --total-cap 5000` to auto-size train/val/test per function from available class pools.

Artifacts are written under:
- `runs_step23_live_matrix/<timestamp>/matrix.log`
- `runs_step23_live_matrix/<timestamp>/cases.jsonl` (one row per sample, includes coverage/faithfulness)
- `runs_step23_live_matrix/<timestamp>/combo_summaries.jsonl`
- `runs_step23_live_matrix/<timestamp>/overall_summary.json`
- `runs_step23_live_matrix/<timestamp>/per_function_summary.csv`
- `runs_step23_live_matrix/<timestamp>/cases/<fn_seed>/sample_####/code0_sanitized.py`
- `runs_step23_live_matrix/<timestamp>/cases/<fn_seed>/sample_####/code1.py`
- `runs_step23_live_matrix/<timestamp>/cases/<fn_seed>/sample_####/summary.json`

Guardrail:
- The matrix runner hard-fails if sanitized Code0 still has comment tokens or docstrings before thesis/Code1 prompting.

### Metrics

**Coverage** (per sample i, averaged over n test samples):

```
coverage_eq_i = 1[c_i(x_i) = true] * |A_{x_i}| / |S|

coverage_mean = (1/n) * sum_{i=1}^{n} coverage_eq_i
```

Where `c_i` is Code1's condition check, `A_{x_i}` is the set of training samples satisfying the thesis conditions, and `S` is the full training set. When the test sample does not satisfy the thesis conditions, its contribution is zero — this penalizes theses that fail to cover their own sample.

**Faithfulness (Ground Truth)** — PRIMARY metric:

```
faithfulness_gt_i = |{x_j in A_{x_i} : y_j^true = L_i}| / |A_{x_i}|
```

Measures what fraction of training samples in the acceptance set share the same ground truth label as the thesis predicts. This tells us whether the thesis captures real data structure.

**Faithfulness (Code0)** — secondary metric:

```
faithfulness_code0_i = |{x_j in A_{x_i} : Code0(x_j) = L_i}| / |A_{x_i}|
```

Measures what fraction of training samples in the acceptance set receive the same label from Code0 as the thesis predicts. This measures self-consistency with the model.

When Code0 is a perfect classifier, both metrics are equal. When Code0 is imperfect, `faithfulness_code0` is typically inflated relative to `faithfulness_gt`.

## Step 2.4 Thesis evaluator module
Step 2.4 extracts equation-metric logic into a shared evaluator:
- `thesis_evaluator.py` provides `ThesisEvaluator`.
- `evaluate_thesis(...)` computes per-sample `coverage_ratio`, `coverage_eq`, `faithfulness_code0`, and `faithfulness_gt`.
- `summarize(...)` computes aggregate metric summaries across many samples.
- Both `thesis_runner.py` (formerly `run_step23_live_matrix.py`) and `run_step22_live_once.py` now use this module so metric semantics stay consistent.

Additional Step 2.4 details:
- `load_split_lines(...)` is the shared split reader for `train.txt`/`test.txt`.
- `run_step22_live_once.py` includes equation metrics in `summary.json` under `equation_metrics`.

## Phase 2.5: Dual Faithfulness & Equation Corrections
Phase 2.5 (2/24 meeting) corrects the faithfulness metric per Tomer's guidance:

**Problem**: The original faithfulness only measured Code0-vs-Code0 agreement (how well the thesis captures Code0's behavior). The ground truth label from `parse_tabular_line()` was discarded (`x_i, _ = ...`).

**Fix**: Now computes both metrics. The ground truth label is captured and compared against the thesis's predicted label. Both metrics are emitted in all outputs, with backward-compatible field names for loading old JSONL data.

**Key fields** (new names with backward-compat aliases):
- `faithfulness_gt` / `agreement_count_gt` — primary, against ground truth
- `faithfulness_code0` / `agreement_count_code0` — secondary, against Code0
- `faithfulness` / `agreement_count` — backward-compat aliases (equal to code0 values)
- `mean_faithfulness_gt_defined` / `mean_faithfulness_gt_all_zero` — aggregate GT metrics
- `mean_faithfulness_code0_defined` / `mean_faithfulness_code0_all_zero` — aggregate Code0 metrics

**Class balance**: All splits are guaranteed to have exactly 50/50 class 0/1 distribution via `compute_auto_split()` (even sizes) and `create_stratified_splits()` (exact class targets).

See `EXAMPLES.md` for recommended PowerShell commands.

## Common flags
- Grid: `--functions`, `--lengths`, `--attempts`, `--num-trials`
- OpenAI: `--model`, `--max-output-tokens`, `--reasoning-effort`, `--verbosity`, `--tool-choice`, `--enable-code-interpreter`
- API routing: `--api-mode` (`responses` or `chat_completions`), `--api-base-url`
- Data: `--train-size`, `--val-size`, `--test-size`, `--seed`, `--dataset-dir`
- Outputs: `--out-jsonl`, `--out-csv`, `--out-manifest`, `--run-id`
- Infra: `--concurrency`, `--timeout`, `--retry-delay`
- Dry run: `--dry-run`

## Output artifacts
- `results_attempts.jsonl`: best row per trial plus summary rows.
- `results_attempts.csv`: flattened table with token usage, accuracy, errors, and run metadata.
- `results_attempts_manifest.json` (or `--out-manifest`): config, environment, argv, and row counts for traceability.
- Trial-level logs: `<out-jsonl>_<fn>_L<length>_trial<trial>.jsonl`.

## Analysis scripts
- `program_synthesis/analyze_baseline_run.py`: leakage checks, compile stats, summary tables, plots.
- `program_synthesis/analyze_run_advanced.py`: run-level summary plots and text readout.
- `program_synthesis/thesis_analysis.py`: thesis metric summaries, failure diagnosis, complexity stats, and prompt-version comparison.

Example:
```bash
python program_synthesis/analyze_baseline_run.py --run-dir program_synthesis/runs/example_run
python program_synthesis/analyze_run_advanced.py --run-dir program_synthesis/runs/example_run
python program_synthesis/thesis_analysis.py --results-dir program_synthesis/runs_step23_live_matrix/<timestamp>
```

## Usage and spend tracking
Per-attempt tokens are written to CSV columns (`prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `cached_tokens`).

Use the usage report helper:

```bash
python program_synthesis/usage_report.py program_synthesis/runs*/llm_*.csv
```

Override model pricing if needed:

```bash
python program_synthesis/usage_report.py program_synthesis/runs*/llm_*.csv \
  --input-rate 1.75 \
  --output-rate 14.0
```

## Safety note
`exec()` is not a sandbox.
