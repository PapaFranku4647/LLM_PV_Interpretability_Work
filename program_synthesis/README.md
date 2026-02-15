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

For tabular tasks (`fn_m`, `fn_n`, `fn_o`, `fn_p`, `fn_q`) lengths are fixed by task metadata, so `--lengths` is optional.

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
