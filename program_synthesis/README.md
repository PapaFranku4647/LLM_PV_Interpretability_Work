# LLM_ERM_Testing Program Synthesis

This repo evaluates the interpretability of executable classifiers and raw exported model artifacts.

## Main entry points

- `thesis_runner.py`: matrix runner for thesis generation, Code1 verification, and equation metrics
- `thesis_evaluator.py`: shared coverage and faithfulness evaluator
- `prompt_variants.py`: thesis prompt builders
- `code1_verifier.py`: Code1 verifier generation and validation

## Canonical paper-facing results

Use `program_synthesis/docs/PAPER_INTERPRETABILITY_TABLES.md` as the source of truth for the current raw-artifact interpretability table.

## Current mainline protocol

- raw textual artifact only
- no summarization
- thesis prompt sees the artifact plus one unseen test sample and predicted label
- coverage and faithfulness are computed locally against the training split afterward

## Notes

- External artifact mode consumes `artifact.json` exports produced by the sibling `LLM_PV` repo.
- Generated runs, matrix outputs, and local planning notes are intentionally kept out of version control.
