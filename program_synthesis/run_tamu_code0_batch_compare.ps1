param(
    [string]$ApiKey = $env:TAMUS_AI_CHAT_API_KEY,
    [string]$ApiBaseUrl = "https://chat-api.tamu.ai/api",
    [string]$PythonExe = ".\\.venv-3-11\\Scripts\\python.exe",
    [string[]]$Models = @(
        "protected.gemini-2.0-flash-lite",
        "protected.gemini-2.5-flash-lite"
    ),
    [string]$FunctionId = "fn_o",
    [int]$Seed = 2201,
    [int]$TrainSize = 40,
    [int]$ValSize = 100,
    [int]$TestSize = 500,
    [int]$BatchSize = 10,
    [int]$BatchAttempts = 2,
    [int]$NumTrials = 1,
    [string]$PromptVariant = "explain",
    [string]$ReasoningEffort = "minimal",
    [int]$Concurrency = 1,
    [int]$MaxOutputTokens = 1200,
    [string]$OutRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $ApiKey) {
    throw "TAMUS_AI_CHAT_API_KEY is missing."
}
if (-not $OutRoot) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutRoot = "program_synthesis/runs_tamu_batch_compare/$stamp"
}

$env:TAMU_API_KEY = $ApiKey
$env:API_MODE = "chat_completions"
$env:API_BASE_URL = $ApiBaseUrl

$sharedDatasetDir = Join-Path $OutRoot "shared_datasets"
$numBatches = [int][Math]::Ceiling($TrainSize / [double]$BatchSize)
$normalAttempts = $BatchAttempts * $numBatches

Write-Host "Output root: $OutRoot"
Write-Host "Shared dataset dir: $sharedDatasetDir"
Write-Host "Fairness rule: batch attempts=$BatchAttempts, batches=$numBatches, normal attempts=$normalAttempts"
Write-Host "Runner settings: concurrency=$Concurrency, max_output_tokens=$MaxOutputTokens"
Write-Host ""

foreach ($model in $Models) {
    $safeModel = ($model -replace "[^A-Za-z0-9._-]", "_")
    $modelRoot = Join-Path $OutRoot $safeModel
    $batchedRoot = Join-Path $modelRoot "batched"
    $normalRoot = Join-Path $modelRoot "normal"
    New-Item -ItemType Directory -Force -Path $batchedRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $normalRoot | Out-Null

    Write-Host "=== $model ==="
    Write-Host "Running batched mode..."
    & $PythonExe program_synthesis/runner_val_selection.py `
        --functions $FunctionId `
        --attempts $BatchAttempts `
        --num-trials $NumTrials `
        --train-size $TrainSize `
        --val-size $ValSize `
        --test-size $TestSize `
        --seed $Seed `
        --model $model `
        --reasoning-effort $ReasoningEffort `
        --concurrency $Concurrency `
        --max-output-tokens $MaxOutputTokens `
        --tool-choice none `
        --prompt-variant $PromptVariant `
        --code0-train-mode batched `
        --code0-batch-size $BatchSize `
        --dataset-dir $sharedDatasetDir `
        --out-jsonl (Join-Path $batchedRoot "results.jsonl") `
        --out-csv (Join-Path $batchedRoot "results.csv")
    if ($LASTEXITCODE -ne 0) {
        throw "Batched run failed for model $model with exit code $LASTEXITCODE"
    }

    Write-Host "Running matched normal control..."
    & $PythonExe program_synthesis/runner_val_selection.py `
        --functions $FunctionId `
        --attempts $normalAttempts `
        --num-trials $NumTrials `
        --train-size $TrainSize `
        --val-size $ValSize `
        --test-size $TestSize `
        --seed $Seed `
        --model $model `
        --reasoning-effort $ReasoningEffort `
        --concurrency $Concurrency `
        --max-output-tokens $MaxOutputTokens `
        --tool-choice none `
        --prompt-variant $PromptVariant `
        --dataset-dir $sharedDatasetDir `
        --out-jsonl (Join-Path $normalRoot "results.jsonl") `
        --out-csv (Join-Path $normalRoot "results.csv")
    if ($LASTEXITCODE -ne 0) {
        throw "Normal control run failed for model $model with exit code $LASTEXITCODE"
    }

    $rows = @()
    foreach ($mode in @("batched", "normal")) {
        $csvPath = Join-Path (Join-Path $modelRoot $mode) "results.csv"
        if (Test-Path $csvPath) {
            $summary = Import-Csv $csvPath | Where-Object { $_.is_summary -eq "True" } | Select-Object -First 1
            if ($summary) {
                $rows += [PSCustomObject]@{
                    model = $model
                    mode = $mode
                    train_acc = $summary.train_acc
                    val_acc = $summary.val_acc
                    test_acc = $summary.test_acc
                    csv = $csvPath
                }
            }
        }
    }
    if ($rows.Count -gt 0) {
        $rows | Format-Table -AutoSize
    }
    Write-Host ""
}

Write-Host "Done."
Write-Host "Budget note: the default settings are intentionally tiny to stay well under the $5 range on low-cost TAMU models."
