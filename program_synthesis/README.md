# Program Synthesis

This repo now carries the unified CodeBoost, baseline, and interpretability workflows in one branch.

## Main entry points

- `boosted/boosted_runner.py`: CodeBoost runner for tabular synthesis experiments
- `baseline_runner.py`: matched classical tabular baseline runner
- `export_interpretability_artifacts.py`: raw textual artifact exporter for downstream interpretability runs
- `runner.py`: core dataset split generation, tabular transforms, and general runner utilities
- `thesis_runner.py`: interpretability matrix runner for thesis generation, Code1 verification, coverage, and faithfulness
- `thesis_evaluator.py`: shared coverage and faithfulness evaluator
- `code1_verifier.py`: Code1 verifier generation and validation

## Canonical paper-facing tables

- Accuracy: `program_synthesis/docs/PAPER_ACCURACY_TABLES.md`
- Interpretability: `program_synthesis/docs/PAPER_INTERPRETABILITY_TABLES.md`

## Current clean protocol

- no dataset names in prompts
- no dataset-specific descriptions in the clean setting
- numeric anonymization via coordinate-wise positive affine transforms
- matched transformed splits for CodeBoost and all baselines
- raw textual artifact evaluation for interpretability

## Notes

- Generated runs, CSVs, JSONLs, local planning notes, and temporary matrices are intentionally gitignored.
- The repo keeps the code paths for both tabular synthesis and interpretability evaluation; generated outputs are not treated as source.
