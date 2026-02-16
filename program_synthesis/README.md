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
Use `run_step23_live_matrix.py` for end-to-end matrix runs with heavy logging and equation-level metrics.

Example (all tabular tasks, 3 seeds, 3 samples/seed, `gpt-5-mini` for all calls):

```bash
python program_synthesis/run_step23_live_matrix.py \
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

Coverage metric implemented:
- `coverage_eq = I(x in A^x) * (|A^x_S| / |S|)`

Faithfulness metric implemented:
- `faithfulness = Pr[Code0(x_i) = Code0(x) | x_i in A_S]`

## Step 2.4 Thesis evaluator module
Step 2.4 extracts equation-metric logic into a shared evaluator:
- `thesis_evaluator.py` provides `ThesisEvaluator`.
- `evaluate_thesis(...)` computes per-sample `coverage_ratio`, `coverage_eq`, and `faithfulness`.
- `summarize(...)` computes aggregate metric summaries across many samples.
- Both `run_step23_live_matrix.py` and `run_step22_live_once.py` now use this module so metric semantics stay consistent.

Additional Step 2.4 details:
- `load_split_lines(...)` is the shared split reader for `train.txt`/`test.txt`.
- `run_step22_live_once.py` includes equation metrics in `summary.json` under `equation_metrics`.

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

Example:
```bash
python program_synthesis/analyze_baseline_run.py --run-dir program_synthesis/runs/example_run
python program_synthesis/analyze_run_advanced.py --run-dir program_synthesis/runs/example_run
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
