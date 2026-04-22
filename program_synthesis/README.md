# CodeBoost

CodeBoost synthesizes standalone Python tabular classifiers with an LLM, evaluates them locally, and keeps the accepted program as the deployable artifact.

## Main entry points

- `boosted/boosted_runner.py`: CodeBoost runner for tabular experiments
- `baseline_runner.py`: matched classical baseline runner
- `export_interpretability_artifacts.py`: raw artifact exporter for the interpretability repo
- `runner.py`: dataset split generation, tabular transforms, and core runner utilities

## Canonical paper-facing results

Use `program_synthesis/docs/PAPER_ACCURACY_TABLES.md` as the source of truth for the clean accuracy tables and protocol.

Current clean headline datasets:

- CDC
- Pima
- Telco
- credit_g

## Clean protocol summary

- no dataset names in prompts
- no dataset-specific descriptions
- numeric anonymization via coordinate-wise positive affine transforms
- matched transformed split for CodeBoost and all baselines
- 5 trials on the same transformed split

## Notes

- Generated reports, CSVs, JSONLs, and run directories are intentionally kept out of version control.
- The interpretability pipeline lives in the sibling `LLM_ERM_Testing` repo and consumes exported raw artifacts from this repo.
