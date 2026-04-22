# Boosted CodeBoost

This folder contains the tabular CodeBoost runner and related helpers.

## Main files

- `boosted_runner.py`: primary CodeBoost tabular runner
- `threshold_distiller.py`: optional threshold-search / distillation layer
- `tamu_api_smoke.py`: Azure / TAMU API smoke test for boosted runs

## Mainline use

Use this runner for CodeBoost tabular experiments:

```bash
python program_synthesis/boosted/boosted_runner.py \
  --functions fn_o \
  --lengths 21 \
  --train-size 256 \
  --val-size 256 \
  --test-size 2000 \
  --seed 42 \
  --batch-sizes 256 \
  --boost-rounds 1 \
  --num-trials 5
```

Important knobs:

- `--tabular-representation semantic|named_numeric|obfuscated|anonymous_numeric`
- `--dataset-context-mode none|schema`
- `--tabular-numeric-transform positive_affine`
- `--sample-without-replacement`
- `--round-retries`
- `--accept-best-on-failure`

## Notes

- Generated `runs/`, `outputs*/`, and dataset caches are intentionally gitignored.
- Canonical paper-facing accuracy tables live in `program_synthesis/docs/PAPER_ACCURACY_TABLES.md`.
- The interpretability pipeline consumes exported raw artifacts from this repo in the sibling `LLM_ERM_Testing` repo.
