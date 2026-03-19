# Qwen 3.5 9B: Normal vs Feedback-Batched with 5 seeds
# Estimated runtime: ~1.5 hours (faster model, ~16 tok/s)
# Prerequisites: LM Studio running qwen3.5-9b (thinking OFF!) at 32K+ context

Set-Location "C:\Users\pipin\Desktop\Coding\Personal\TomerResearchFolder\LLM_ERM_Testing"
& ".\.venv-3-11\Scripts\Activate.ps1"

Remove-Item Env:TAMU_API_KEY -ErrorAction SilentlyContinue
if (-not $env:OPENAI_API_KEY) {
    # Local OpenAI-compatible servers only need a non-empty placeholder.
    $env:OPENAI_API_KEY = "lm-studio-local"
}
$env:API_MODE = "chat_completions"
$env:API_BASE_URL = "http://192.168.56.1:1234/v1"
$env:OPENAI_MODEL = "qwen3.5-9b@q6_k"
$env:REASONING_EFFORT = "medium"
$env:TOOL_CHOICE = "none"
$env:RETRY_DELAY_S = "5"
$env:PYTHONPATH = "program_synthesis"

$outDir = "program_synthesis\runs_qwen9b_v2"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$seeds = @(42, 123, 456, 789, 2201)
$startTime = Get-Date
Write-Host "[$( Get-Date -Format 'HH:mm:ss' )] Starting Qwen 3.5 9B experiments (5 seeds x 2 functions x 2 modes)" -ForegroundColor Cyan
Write-Host "Estimated completion: $( (Get-Date).AddHours(1.5).ToString('HH:mm:ss') )" -ForegroundColor Cyan

# --- NORMAL MODE: all 5 seeds ---
foreach ($seed in $seeds) {
    Write-Host "[$( Get-Date -Format 'HH:mm:ss' )] NORMAL seed=$seed (fn_o, fn_n)" -ForegroundColor Yellow
    python program_synthesis/runner.py `
        --api-mode chat_completions `
        --api-base-url "http://192.168.56.1:1234/v1" `
        --model "qwen3.5-9b@q6_k" `
        --functions fn_o fn_n `
        --attempts 5 `
        --num-trials 1 `
        --max-output-tokens 1500 `
        --tool-choice none `
        --seed $seed `
        --concurrency 1 `
        --out-jsonl "$outDir/normal_seed${seed}.jsonl" `
        --out-csv "$outDir/normal_seed${seed}.csv" `
        --run-id "qwen9b_normal_seed${seed}"
}
Write-Host "[$( Get-Date -Format 'HH:mm:ss' )] ALL NORMAL RUNS DONE" -ForegroundColor Green

# --- BATCHED MODE: all 5 seeds ---
foreach ($seed in $seeds) {
    Write-Host "[$( Get-Date -Format 'HH:mm:ss' )] BATCHED seed=$seed (fn_o, fn_n)" -ForegroundColor Yellow
    python program_synthesis/runner_val_selection.py `
        --api-mode chat_completions `
        --api-base-url "http://192.168.56.1:1234/v1" `
        --model "qwen3.5-9b@q6_k" `
        --functions fn_o fn_n `
        --attempts 5 `
        --num-trials 1 `
        --max-output-tokens 1500 `
        --tool-choice none `
        --seed $seed `
        --concurrency 1 `
        --code0-train-mode batched `
        --code0-batch-size 20 `
        --out-jsonl "$outDir/batched_seed${seed}.jsonl" `
        --out-csv "$outDir/batched_seed${seed}.csv" `
        --run-id "qwen9b_batched_seed${seed}"
}

$elapsed = (Get-Date) - $startTime
Write-Host "[$( Get-Date -Format 'HH:mm:ss' )] ALL DONE in $( [math]::Round($elapsed.TotalMinutes) ) minutes" -ForegroundColor Green
Write-Host "Results in $outDir/" -ForegroundColor Cyan
