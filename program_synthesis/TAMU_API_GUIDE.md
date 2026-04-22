# TAMU API Guide

This repo keeps the TAMU Azure wiring in code, but not local wrapper scripts or tutorial notebooks.

## Verified endpoint

```text
https://tamu-it-ae-ai-prod-prod-eastus2.openai.azure.com/
```

## Environment

PowerShell:

```powershell
$env:TAMUS_AI_CHAT_API_KEY = "<your TAMU key>"
$env:TAMU_API_KEY = $env:TAMUS_AI_CHAT_API_KEY
$env:TAMU_AZURE_ENDPOINT = "https://tamu-it-ae-ai-prod-prod-eastus2.openai.azure.com/"
$env:TAMU_API_VERSION = "2024-12-01-preview"
$env:OPENAI_MODEL = "gpt-5.2-deep-learning-fundamentals"
$env:API_MODE = "chat_completions"
```

## Minimal smoke tests

SDK smoke test:

```powershell
python .\program_synthesis\test_tamu_api_sdk.py `
  --model gpt-5.2-deep-learning-fundamentals `
  --reasoning-effort minimal `
  --max-tokens 64
```

Boosted smoke test:

```powershell
python .\program_synthesis\boosted\tamu_api_smoke.py `
  --model gpt-5.2-deep-learning-fundamentals `
  --api-mode chat_completions `
  --max-output-tokens 64
```

## Running experiments

Use the Python entry points directly:

- `program_synthesis/runner.py`
- `program_synthesis/baseline_runner.py`
- `program_synthesis/boosted/boosted_runner.py`
- `program_synthesis/thesis_runner.py`

## Notes

- The repo intentionally no longer keeps local PowerShell wrappers, notebooks, or slide/tutorial artifacts.
- Azure/TAMU support is preserved in `program_synthesis/runner.py`, `program_synthesis/llm_client.py`, and `program_synthesis/boosted/tamu_api_smoke.py`.
