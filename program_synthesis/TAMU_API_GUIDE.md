# TAMU API Guide

This guide captures the verified TAMU path and the scripts to move from a single successful completion to repeatable Code0 experiments.

## Verified endpoint

The working chat completions path is:

```text
https://chat-api.tamu.ai/api/chat/completions
```

For runner code that appends `"/chat/completions"` to a base URL, use:

```text
https://chat-api.tamu.ai/api
```

## Environment setup

PowerShell:

```powershell
$env:TAMUS_AI_CHAT_API_KEY = "<your TAMU key>"
$env:TAMUS_AI_CHAT_API_ENDPOINT = "https://chat-api.tamu.ai"
$env:TAMU_API_KEY = $env:TAMUS_AI_CHAT_API_KEY
$env:API_MODE = "chat_completions"
$env:API_BASE_URL = "https://chat-api.tamu.ai/api"
```

Optional UTF-8 console fix if response text shows mojibake like `â`:

```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
```

## 1. Exact raw REST smoke test

This stays closest to the TAMU docs and verifies your key, endpoint, and model.

```powershell
$body = @{
    model = "protected.gpt-5"
    stream = $false
    messages = @(
        @{
            role = "user"
            content = "Why is the sky blue?"
        }
    )
} | ConvertTo-Json -Depth 10

$response = Invoke-RestMethod -Uri "$env:TAMUS_AI_CHAT_API_ENDPOINT/api/chat/completions" `
    -Headers @{
        Authorization = "Bearer $env:TAMUS_AI_CHAT_API_KEY"
        "Content-Type" = "application/json"
    } `
    -Method Post `
    -Body $body

$response | ConvertTo-Json -Depth 100
```

Expected success signals:
- `object = "chat.completion"`
- `choices[0].message.content` is populated
- `usage` is populated

## 2. Scripted smoke test

This uses the repo helper and will try `/api` first.

```powershell
powershell -ExecutionPolicy Bypass -File .\program_synthesis\test_tamu_api.ps1 `
  -Mode api `
  -Model protected.gpt-5 `
  -ReasoningEffort minimal `
  -Prompt "Reply with exactly: TAMU API OK"
```

Useful variants:

```powershell
powershell -ExecutionPolicy Bypass -File .\program_synthesis\test_tamu_api.ps1 `
  -Mode api `
  -Model protected.gemini-2.0-flash-lite `
  -Prompt "Why is the sky blue?"
```

```powershell
powershell -ExecutionPolicy Bypass -File .\program_synthesis\test_tamu_api.ps1 `
  -Mode auto `
  -Model protected.gpt-5 `
  -ReasoningEffort medium `
  -MaxTokens 256
```

## 3. Model and reasoning sweep

Use this when one message works and you want a broader stability check.

```powershell
powershell -ExecutionPolicy Bypass -File .\program_synthesis\run_tamu_api_smoke_matrix.ps1 `
  -Models protected.gpt-5 protected.gemini-2.0-flash-lite protected.gemini-2.5-flash-lite `
  -ReasoningEfforts minimal medium `
  -PromptSet standard `
  -MaxTokens 512 `
  -PauseBetweenCallsMs 500
```

Artifacts:
- `program_synthesis/runs_tamu_api_smoke/<timestamp>/models.json`
- `program_synthesis/runs_tamu_api_smoke/<timestamp>/results.jsonl`
- `program_synthesis/runs_tamu_api_smoke/<timestamp>/results.csv`
- `program_synthesis/runs_tamu_api_smoke/<timestamp>/manifest.json`

Fields to inspect in `results.csv`:
- `status`
- `returned_model`
- `finish_reason`
- `total_tokens`
- `reasoning_tokens`
- `elapsed_ms`
- `reply_preview`
- `error`

## 4. Tiny Code0 TAMU run

After the smoke test is stable, move to the smallest end-to-end Code0 comparison.

```powershell
powershell -ExecutionPolicy Bypass -File .\program_synthesis\run_tamu_code0_batch_compare.ps1 `
  -ApiBaseUrl "https://chat-api.tamu.ai/api" `
  -Models protected.gpt-5 `
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

## 5. Larger TAMU Code0 compare

Only do this after the tiny run works and token cost looks reasonable.

```powershell
powershell -ExecutionPolicy Bypass -File .\program_synthesis\run_tamu_code0_batch_compare.ps1 `
  -ApiBaseUrl "https://chat-api.tamu.ai/api" `
  -Models protected.gpt-5 `
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
- check `TAMUS_AI_CHAT_API_ENDPOINT`
- confirm the path is `/api/chat/completions`

If the response works in raw REST but fails in the runner:
- confirm `API_MODE=chat_completions`
- confirm `API_BASE_URL=https://chat-api.tamu.ai/api`
- rerun `test_tamu_api_sdk.py` before the full compare script

Python smoke test:

```powershell
.\.venv-3-11\Scripts\python.exe .\program_synthesis\test_tamu_api_sdk.py `
  --model protected.gpt-5 `
  --reasoning-effort minimal `
  --max-tokens 64
```
