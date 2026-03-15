param(
    [string]$ApiKey = $env:OPENAI_API_KEY,
    [string]$PythonExe = ".\\.venv-3-11\\Scripts\\python.exe",
    [string]$Model = "gpt-5-mini",
    [string]$FunctionId = "fn_o",
    [int]$Seed = 2201,
    [int]$TrainSize = 100,
    [int]$ValSize = 2300,
    [int]$TestSize = 7500,
    [int]$BatchSize = 20,
    [int]$BatchAttempts = 2,
    [int]$NumTrials = 1,
    [string]$PromptVariant = "explain",
    [string]$ReasoningEffort = "medium",
    [int]$Concurrency = 1,
    [int]$MaxOutputTokens = 1200,
    [string]$DatasetSourceSeedDir = "",
    [string]$OutRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $ApiKey) {
    throw "OPENAI_API_KEY is missing."
}
if (-not $OutRoot) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutRoot = "program_synthesis/runs_openai_batch_compare/$stamp"
}

$safeModel = ($Model -replace "[^A-Za-z0-9._-]", "_")
$modelRoot = Join-Path $OutRoot $safeModel
$batchedRoot = Join-Path $modelRoot "batched"
$normalRoot = Join-Path $modelRoot "normal"
$sharedDatasetDir = Join-Path $OutRoot "shared_datasets"
$sessionLog = Join-Path $modelRoot "session.log"
$comparisonCsv = Join-Path $modelRoot "comparison_summary.csv"
$stagedDatasetInfoPath = Join-Path $modelRoot "staged_dataset_info.json"

New-Item -ItemType Directory -Force -Path $batchedRoot | Out-Null
New-Item -ItemType Directory -Force -Path $normalRoot | Out-Null
New-Item -ItemType Directory -Force -Path $sharedDatasetDir | Out-Null

$numBatches = [int][Math]::Ceiling($TrainSize / [double]$BatchSize)
$normalAttempts = $BatchAttempts * $numBatches

$env:OPENAI_API_KEY = $ApiKey
Remove-Item Env:TAMU_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:API_BASE_URL -ErrorAction SilentlyContinue
Remove-Item Env:API_MODE -ErrorAction SilentlyContinue

$repoRoot = Split-Path -Parent $PSScriptRoot
$env:CODEX_WRAPPER_REPO_ROOT = $repoRoot
$env:CODEX_WRAPPER_FN = $FunctionId
$env:CODEX_WRAPPER_TRAIN = "$TrainSize"
$env:CODEX_WRAPPER_VAL = "$ValSize"
$env:CODEX_WRAPPER_TEST = "$TestSize"
$env:CODEX_WRAPPER_SEED = "$Seed"

$datasetInfoScript = @'
import hashlib
import json
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["CODEX_WRAPPER_REPO_ROOT"])
sys.path.insert(0, str(repo_root))

from src.target_functions import EXPERIMENT_FUNCTION_MAPPING, EXPERIMENT_FUNCTION_METADATA

fn = os.environ["CODEX_WRAPPER_FN"]
train_size = int(os.environ["CODEX_WRAPPER_TRAIN"])
val_size = int(os.environ["CODEX_WRAPPER_VAL"])
test_size = int(os.environ["CODEX_WRAPPER_TEST"])
base_seed = int(os.environ["CODEX_WRAPPER_SEED"])

target_name = EXPERIMENT_FUNCTION_MAPPING[fn]
length = int(EXPERIMENT_FUNCTION_METADATA[fn]["lengths"][0])
key = f"{fn}|L={length}|train={train_size+val_size}|test={test_size}|base_seed={base_seed}"
derived_seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & 0x7FFFFFFF

print(json.dumps({
    "target_name": target_name,
    "length": length,
    "derived_seed": derived_seed,
}, indent=2))
'@

$datasetInfo = ($datasetInfoScript | & $PythonExe - | ConvertFrom-Json)
if (-not $datasetInfo -or -not $datasetInfo.target_name -or -not $datasetInfo.length -or -not $datasetInfo.derived_seed) {
    throw "Failed to resolve dataset metadata for $FunctionId"
}
$destSeedDir = Join-Path (Join-Path (Join-Path $sharedDatasetDir $datasetInfo.target_name) ("L{0}" -f $datasetInfo.length)) ("seed{0}" -f $datasetInfo.derived_seed)

if ($DatasetSourceSeedDir) {
    $sourceTrain = Join-Path $DatasetSourceSeedDir "train.txt"
    $sourceVal = Join-Path $DatasetSourceSeedDir "val.txt"
    $sourceTest = Join-Path $DatasetSourceSeedDir "test.txt"
    $sourceMeta = Join-Path $DatasetSourceSeedDir "meta.json"

    foreach ($path in @($sourceTrain, $sourceVal, $sourceTest)) {
        if (-not (Test-Path $path)) {
            throw "DatasetSourceSeedDir is missing required file: $path"
        }
    }

    $trainLines = @(Get-Content $sourceTrain | Select-Object -First $TrainSize)
    $valLines = @(Get-Content $sourceVal | Select-Object -First $ValSize)
    $testLines = @(Get-Content $sourceTest | Select-Object -First $TestSize)

    if ($trainLines.Count -ne $TrainSize) {
        throw "Source train split has $($trainLines.Count) lines, expected at least $TrainSize"
    }
    if ($valLines.Count -ne $ValSize) {
        throw "Source val split has $($valLines.Count) lines, expected at least $ValSize"
    }
    if ($testLines.Count -ne $TestSize) {
        throw "Source test split has $($testLines.Count) lines, expected at least $TestSize"
    }

    New-Item -ItemType Directory -Force -Path $destSeedDir | Out-Null
    Set-Content -Path (Join-Path $destSeedDir "train.txt") -Value $trainLines -Encoding UTF8
    Set-Content -Path (Join-Path $destSeedDir "val.txt") -Value $valLines -Encoding UTF8
    Set-Content -Path (Join-Path $destSeedDir "test.txt") -Value $testLines -Encoding UTF8

    $sourceMetaContent = $null
    if (Test-Path $sourceMeta) {
        $sourceMetaContent = Get-Content $sourceMeta -Raw | ConvertFrom-Json
    }

    $stagedMeta = [ordered]@{
        fn = $FunctionId
        target_name = $datasetInfo.target_name
        length = [int]$datasetInfo.length
        decimal = if ($sourceMetaContent) { [bool]$sourceMetaContent.decimal } else { $false }
        tabular = if ($sourceMetaContent) { [bool]$sourceMetaContent.tabular } else { $true }
        derived_seed = [int]$datasetInfo.derived_seed
        sizes = [ordered]@{
            train = $TrainSize
            val = $ValSize
            test = $TestSize
        }
        staged_from = $DatasetSourceSeedDir
        source_sizes = if ($sourceMetaContent) { $sourceMetaContent.sizes } else { $null }
        created_ts = [int][DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    }
    $stagedMeta | ConvertTo-Json -Depth 10 | Set-Content -Path (Join-Path $destSeedDir "meta.json") -Encoding UTF8

    $stagedInfo = [ordered]@{
        function = $FunctionId
        source_seed_dir = $DatasetSourceSeedDir
        staged_seed_dir = $destSeedDir
        source_meta = $sourceMeta
        requested_sizes = [ordered]@{
            train = $TrainSize
            val = $ValSize
            test = $TestSize
        }
        staged_counts = [ordered]@{
            train = $trainLines.Count
            val = $valLines.Count
            test = $testLines.Count
        }
    }
    $stagedInfo | ConvertTo-Json -Depth 10 | Set-Content -Path $stagedDatasetInfoPath -Encoding UTF8
}

Write-Host "Output root: $OutRoot"
Write-Host "Model root: $modelRoot"
Write-Host "Shared dataset dir: $sharedDatasetDir"
Write-Host "Expected dataset seed dir: $destSeedDir"
if ($DatasetSourceSeedDir) {
    Write-Host "Staged dataset from: $DatasetSourceSeedDir"
    Write-Host "Staged dataset info: $stagedDatasetInfoPath"
}
Write-Host "Fairness rule: batch attempts=$BatchAttempts, batches=$numBatches, normal attempts=$normalAttempts"
Write-Host "Selection rule: batched selects by cumulative train_acc; normal selects by val_acc"
Write-Host "Runner settings: model=$Model, reasoning_effort=$ReasoningEffort, concurrency=$Concurrency, max_output_tokens=$MaxOutputTokens"
Write-Host "Session log: $sessionLog"
Write-Host ""

"=== Batched vs Normal Code0 Compare ===" | Tee-Object -FilePath $sessionLog
"Model: $Model" | Tee-Object -FilePath $sessionLog -Append
"Function: $FunctionId" | Tee-Object -FilePath $sessionLog -Append
"Train/Val/Test: $TrainSize / $ValSize / $TestSize" | Tee-Object -FilePath $sessionLog -Append
"Batch size / per-batch attempts / matched normal attempts: $BatchSize / $BatchAttempts / $normalAttempts" | Tee-Object -FilePath $sessionLog -Append
"" | Tee-Object -FilePath $sessionLog -Append

$savedErrorActionPreference = $ErrorActionPreference

"=== Running batched mode ===" | Tee-Object -FilePath $sessionLog -Append
$ErrorActionPreference = "Continue"
& $PythonExe program_synthesis/runner_val_selection.py `
    --functions $FunctionId `
    --attempts $BatchAttempts `
    --num-trials $NumTrials `
    --train-size $TrainSize `
    --val-size $ValSize `
    --test-size $TestSize `
    --seed $Seed `
    --model $Model `
    --reasoning-effort $ReasoningEffort `
    --concurrency $Concurrency `
    --max-output-tokens $MaxOutputTokens `
    --tool-choice none `
    --prompt-variant $PromptVariant `
    --code0-train-mode batched `
    --code0-batch-size $BatchSize `
    --dataset-dir $sharedDatasetDir `
    --out-jsonl (Join-Path $batchedRoot "results.jsonl") `
    --out-csv (Join-Path $batchedRoot "results.csv") `
    2>&1 | Tee-Object -FilePath $sessionLog -Append
$ErrorActionPreference = $savedErrorActionPreference
if ($LASTEXITCODE -ne 0) {
    throw "Batched run failed with exit code $LASTEXITCODE"
}

"" | Tee-Object -FilePath $sessionLog -Append
"=== Running matched normal control ===" | Tee-Object -FilePath $sessionLog -Append
$ErrorActionPreference = "Continue"
& $PythonExe program_synthesis/runner_val_selection.py `
    --functions $FunctionId `
    --attempts $normalAttempts `
    --num-trials $NumTrials `
    --train-size $TrainSize `
    --val-size $ValSize `
    --test-size $TestSize `
    --seed $Seed `
    --model $Model `
    --reasoning-effort $ReasoningEffort `
    --concurrency $Concurrency `
    --max-output-tokens $MaxOutputTokens `
    --tool-choice none `
    --prompt-variant $PromptVariant `
    --dataset-dir $sharedDatasetDir `
    --out-jsonl (Join-Path $normalRoot "results.jsonl") `
    --out-csv (Join-Path $normalRoot "results.csv") `
    2>&1 | Tee-Object -FilePath $sessionLog -Append
$ErrorActionPreference = $savedErrorActionPreference
if ($LASTEXITCODE -ne 0) {
    throw "Normal control run failed with exit code $LASTEXITCODE"
}

$rows = @()
foreach ($mode in @("batched", "normal")) {
    $csvPath = Join-Path (Join-Path $modelRoot $mode) "results.csv"
    if (Test-Path $csvPath) {
        $summary = Import-Csv $csvPath | Where-Object { $_.is_summary -eq "True" } | Select-Object -First 1
        if ($summary) {
            $rows += [PSCustomObject]@{
                model = $Model
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
    $rows | Export-Csv -NoTypeInformation -Path $comparisonCsv
    "" | Tee-Object -FilePath $sessionLog -Append
    "=== Final Comparison ===" | Tee-Object -FilePath $sessionLog -Append
    $rows | Format-Table -AutoSize | Out-String | Tee-Object -FilePath $sessionLog -Append
    $rows | Format-Table -AutoSize
}

Write-Host ""
Write-Host "Done."
Write-Host "Artifacts:"
Write-Host "  session_log = $sessionLog"
Write-Host "  comparison_csv = $comparisonCsv"
if ($DatasetSourceSeedDir) {
    Write-Host "  staged_dataset_info = $stagedDatasetInfoPath"
}
Write-Host "  batched_csv = $(Join-Path $batchedRoot 'results.csv')"
Write-Host "  batched_batches_csv = $(Join-Path $batchedRoot 'results_batches.csv')"
Write-Host "  normal_csv = $(Join-Path $normalRoot 'results.csv')"
