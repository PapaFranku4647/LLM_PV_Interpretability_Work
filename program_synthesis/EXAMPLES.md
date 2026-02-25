# Thesis Runner Example Commands

All commands use `gpt-5-mini` by default (from `OPENAI_MODEL` env var or CLI default).
All splits are class-balanced (50/50 class 0 and class 1) by design.

## 1. Unit tests (always run first)

```powershell
python -m pytest program_synthesis/tests/ -v
```

## 2. Minimal smoke test (1 function, 1 seed, 1 sample)

Cheapest possible live run to verify the pipeline works end-to-end.

```powershell
python -m program_synthesis.thesis_runner `
  --functions fn_m `
  --seeds 2201 `
  --samples-per-seed 1 `
  --train-size 50 `
  --val-size 50 `
  --test-size 100 `
  --attempts 1 `
  --num-trials 1 `
  --compute-baselines `
  --thesis-prompt-version v2
```

## 3. Two functions, more samples (good for checking GT vs Code0 divergence)

```powershell
python -m program_synthesis.thesis_runner `
  --functions fn_m fn_n `
  --seeds 2201 `
  --samples-per-seed 2 `
  --train-size 100 `
  --val-size 100 `
  --test-size 200 `
  --attempts 2 `
  --compute-baselines `
  --thesis-prompt-version v2
```

## 4. Auto-split mode (balanced sizes per dataset)

```powershell
python -m program_synthesis.thesis_runner `
  --functions fn_m fn_o `
  --seeds 2201 `
  --samples-per-seed 2 `
  --auto-split `
  --train-cap 100 `
  --total-cap 500 `
  --attempts 2 `
  --compute-baselines `
  --thesis-prompt-version v2
```

## 5. Cost-optimized research run (per Tomer 2/24 meeting)

1 seed, 10 attempts (no trials), 25 test samples per seed.
Spend budget on test samples, not on Code0 seeds.

```powershell
python -m program_synthesis.thesis_runner `
  --functions fn_o `
  --seeds 2201 `
  --samples-per-seed 25 `
  --train-size 200 `
  --val-size 200 `
  --test-size 3000 `
  --attempts 10 `
  --num-trials 1 `
  --compute-baselines `
  --thesis-prompt-version v2
```

## 6. Inspect results after a run

```powershell
# Find the latest run directory
$latest = Get-ChildItem program_synthesis/runs_step23_live_matrix | Sort-Object Name -Descending | Select-Object -First 1

# Show overall summary with dual faithfulness metrics
Get-Content "$($latest.FullName)/overall_summary.json" | python -m json.tool

# Quick check for new fields in cases.jsonl
Select-String -Pattern "faithfulness_gt|faithfulness_code0" "$($latest.FullName)/cases.jsonl" | Select-Object -First 5
```

## 7. Reuse existing Code0 runs (skip expensive LLM Code0 generation)

```powershell
python -m program_synthesis.thesis_runner `
  --functions fn_m `
  --seeds 2201 `
  --samples-per-seed 5 `
  --skip-runner `
  --compute-baselines `
  --thesis-prompt-version v2
```

## 8. Multi-model comparison (future)

Run the same setup with different models and compare:

```powershell
# GPT-5-mini
python -m program_synthesis.thesis_runner `
  --functions fn_o `
  --seeds 2201 `
  --samples-per-seed 25 `
  --attempts 10 `
  --model gpt-5-mini `
  --auto-split `
  --compute-baselines `
  --thesis-prompt-version v2

# GPT-5.2 with medium reasoning
python -m program_synthesis.thesis_runner `
  --functions fn_o `
  --seeds 2201 `
  --samples-per-seed 25 `
  --attempts 10 `
  --model gpt-5.2 `
  --reasoning-effort medium `
  --auto-split `
  --compute-baselines `
  --thesis-prompt-version v2
```

Then compare with the analysis tool:

```powershell
python program_synthesis/thesis_analysis.py `
  --results-dir program_synthesis/runs_step23_live_matrix/<timestamp_run1> `
  --results-dir program_synthesis/runs_step23_live_matrix/<timestamp_run2>
```
