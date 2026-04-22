# Overnight experiment: Normal vs Batched Code0 with qwen3-coder-30b-a3b
# Estimated runtime: ~5 hours
# Make sure LM Studio is running with qwen3-coder-30b-a3b-instruct loaded at 32K+ context

Set-Location "C:\Users\pipin\Desktop\Coding\Personal\TomerResearchFolder\LLM_ERM_Testing"

# Activate venv
& ".\.venv-3-11\Scripts\Activate.ps1"

Remove-Item Env:TAMU_API_KEY -ErrorAction SilentlyContinue
if (-not $env:OPENAI_API_KEY) {
    # Local OpenAI-compatible servers only need a non-empty placeholder.
    $env:OPENAI_API_KEY = "lm-studio-local"
}
$env:API_MODE = "chat_completions"
$env:API_BASE_URL = "http://192.168.56.1:1234/v1"
$env:OPENAI_MODEL = "qwen3-coder-30b-a3b-instruct"
$env:REASONING_EFFORT = "medium"
$env:TOOL_CHOICE = "none"
$env:RETRY_DELAY_S = "5"
$env:PYTHONPATH = "program_synthesis"

New-Item -ItemType Directory -Force -Path "program_synthesis\runs_coder30b" | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Write-Host "[$timestamp] Starting overnight experiments..." -ForegroundColor Cyan

# --- RUN 1: Normal mode ---
Write-Host "[$( Get-Date -Format 'HH:mm:ss' )] RUN 1/2: Normal mode (3 fn x 5 seeds x 5 attempts)" -ForegroundColor Yellow
python program_synthesis/runner.py `
    --api-mode chat_completions `
    --api-base-url "http://192.168.56.1:1234/v1" `
    --model qwen3-coder-30b-a3b-instruct `
    --functions fn_o fn_n fn_q `
    --attempts 5 `
    --num-trials 5 `
    --max-output-tokens 1500 `
    --tool-choice none `
    --seed 42 `
    --concurrency 1 `
    --out-jsonl "program_synthesis/runs_coder30b/normal_results.jsonl" `
    --out-csv "program_synthesis/runs_coder30b/normal_results.csv" `
    --run-id "overnight_normal_$timestamp"

Write-Host "[$( Get-Date -Format 'HH:mm:ss' )] RUN 1 FINISHED" -ForegroundColor Green

# --- RUN 2: Batched mode ---
Write-Host "[$( Get-Date -Format 'HH:mm:ss' )] RUN 2/2: Batched mode (3 fn x 5 seeds x 5 attempts x 5 batches)" -ForegroundColor Yellow
python program_synthesis/runner_val_selection.py `
    --api-mode chat_completions `
    --api-base-url "http://192.168.56.1:1234/v1" `
    --model qwen3-coder-30b-a3b-instruct `
    --functions fn_o fn_n fn_q `
    --attempts 5 `
    --num-trials 5 `
    --max-output-tokens 1500 `
    --tool-choice none `
    --seed 42 `
    --concurrency 1 `
    --code0-train-mode batched `
    --code0-batch-size 20 `
    --out-jsonl "program_synthesis/runs_coder30b/batched_results.jsonl" `
    --out-csv "program_synthesis/runs_coder30b/batched_results.csv" `
    --run-id "overnight_batched_$timestamp"

Write-Host "[$( Get-Date -Format 'HH:mm:ss' )] RUN 2 FINISHED" -ForegroundColor Green
Write-Host "ALL DONE. Results in program_synthesis/runs_coder30b/" -ForegroundColor Cyan
