# TAMU API Guide

This guide captures the verified TAMU Azure endpoint and the scripts to move from a single successful completion to repeatable Code0 experiments.

## Verified endpoint

The working endpoint is:

```text
https://tamu-it-ae-ai-prod-prod-eastus2.openai.azure.com/
```

## Environment setup

PowerShell:

```powershell
$env:TAMUS_AI_CHAT_API_KEY = "<your TAMU key>"
$env:TAMU_API_KEY = $env:TAMUS_AI_CHAT_API_KEY
$env:TAMU_AZURE_ENDPOINT = "https://tamu-it-ae-ai-prod-prod-eastus2.openai.azure.com/"
$env:TAMU_API_VERSION = "2024-12-01-preview"
$env:OPENAI_MODEL = "gpt-5.2-deep-learning-fundamentals"
$env:API_MODE = "chat_completions"
```

Optional UTF-8 console fix if response text shows mojibake like `â`:

```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
```

## 1. Exact raw REST smoke test

This verifies your key, Azure endpoint, and deployment.

```powershell
$body = @{
    messages = @(
        @{
            role = "user"
            content = "Reply with exactly: TAMU API OK"
        }
    )
    max_completion_tokens = 32
} | ConvertTo-Json -Depth 10

$response = Invoke-RestMethod -Uri "$($env:TAMU_AZURE_ENDPOINT.TrimEnd('/'))/openai/deployments/$env:OPENAI_MODEL/chat/completions?api-version=$env:TAMU_API_VERSION" `
    -Headers @{
        "api-key" = $env:TAMUS_AI_CHAT_API_KEY
        "Content-Type" = "application/json"
    } `
    -Method Post `
    -Body $body

$response | ConvertTo-Json -Depth 100
```

Expected success signals:
- `choices[0].message.content` is populated
- `usage` is populated
- no `401 Unauthorized`

## 2. Scripted smoke test

This uses the repo helper and hits the Azure deployment directly.

```powershell
powershell -ExecutionPolicy Bypass -File .\program_synthesis\test_tamu_api.ps1 `
  -Model gpt-5.2-deep-learning-fundamentals `
  -MaxTokens 32 `
  -Prompt "Reply with exactly: TAMU API OK"
```

Useful variants:

```powershell
powershell -ExecutionPolicy Bypass -File .\program_synthesis\test_tamu_api.ps1 `
  -Model gpt-5.2-deep-learning-fundamentals `
  -Prompt "Why is the sky blue?"
```

```powershell
powershell -ExecutionPolicy Bypass -File .\program_synthesis\test_tamu_api.ps1 `
  -Model gpt-5.2-deep-learning-fundamentals `
  -ReasoningEffort medium `
  -MaxTokens 256
```

## 3. Model and reasoning sweep

Use this when one message works and you want a broader stability check.

```powershell
powershell -ExecutionPolicy Bypass -File .\program_synthesis\run_tamu_api_smoke_matrix.ps1 `
  -Deployments gpt-5.2-deep-learning-fundamentals `
  -ReasoningEfforts minimal `
  -PromptSet quick `
  -MaxTokens 64 `
  -PauseBetweenCallsMs 500
```

Artifacts:
- `program_synthesis/runs_tamu_api_smoke/<timestamp>/results.jsonl`
- `program_synthesis/runs_tamu_api_smoke/<timestamp>/results.csv`
- `program_synthesis/runs_tamu_api_smoke/<timestamp>/manifest.json`

Fields to inspect in `results.csv`:
- `status`
- `returned_model`
- `finish_reason`
- `total_tokens`
- `reasoning_tokens`
- `estimated_total_cost_usd`
- `elapsed_ms`
- `reply_preview`
- `error`

## 4. Tiny Code0 TAMU run

After the smoke test is stable, move to the smallest end-to-end Code0 comparison.

```powershell
powershell -ExecutionPolicy Bypass -File .\program_synthesis\run_tamu_code0_batch_compare.ps1 `
  -AzureEndpoint "https://tamu-it-ae-ai-prod-prod-eastus2.openai.azure.com/" `
  -Models gpt-5.2-deep-learning-fundamentals `
  -FunctionId fn_o `
  -TrainSize 20 `
  -ValSize 50 `
  -TestSize 200 `
  -BatchSize 10 `
  -BatchAttempts 1 `
  -ReasoningEffort minimal `
  -MaxOutputTokens 400
```

This is intentionally tiny. The goal is to verify the pipeline, not to measure final research quality.

After the run, summarize token/cost totals with:

```powershell
.\.venv-3-11\Scripts\python.exe .\program_synthesis\usage_report.py `
  .\program_synthesis\runs_tamu_batch_compare\*\*\results.csv
```

## 5. Larger TAMU Code0 compare

Only do this after the tiny run works and token cost looks reasonable.

```powershell
powershell -ExecutionPolicy Bypass -File .\program_synthesis\run_tamu_code0_batch_compare.ps1 `
  -AzureEndpoint "https://tamu-it-ae-ai-prod-prod-eastus2.openai.azure.com/" `
  -Models gpt-5.2-deep-learning-fundamentals `
  -FunctionId fn_o `
  -TrainSize 100 `
  -ValSize 100 `
  -TestSize 3000 `
  -BatchSize 20 `
  -BatchAttempts 2 `
  -ReasoningEffort minimal `
  -MaxOutputTokens 1200
```

## 6. Failure triage

If `Invoke-RestMethod` fails before returning JSON:
- check `TAMUS_AI_CHAT_API_KEY`
- check `TAMU_AZURE_ENDPOINT`
- check `TAMU_API_VERSION`
- confirm you are using `api-key`, not `Authorization: Bearer`

If the response works in raw REST but fails in the runner:
- confirm `API_MODE=chat_completions`
- confirm `TAMU_AZURE_ENDPOINT=https://tamu-it-ae-ai-prod-prod-eastus2.openai.azure.com/`
- clear any stale `API_BASE_URL`
- expect the first CDC/UCI-backed tabular run to be slower because it builds the local dataset cache
- rerun `test_tamu_api_sdk.py` before the full compare script

Python smoke test:

```powershell
.\.venv-3-11\Scripts\python.exe .\program_synthesis\test_tamu_api_sdk.py `
  --model gpt-5.2-deep-learning-fundamentals `
  --reasoning-effort minimal `
  --max-tokens 64
```
