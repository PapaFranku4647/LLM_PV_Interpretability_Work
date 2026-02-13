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
- Data: `--train-size`, `--val-size`, `--test-size`, `--seed`, `--dataset-dir`
- Outputs: `--out-jsonl`, `--out-csv`, `--out-manifest`, `--run-id`
- Infra: `--concurrency`, `--timeout`
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

## Safety note
`exec()` is not a sandbox.
