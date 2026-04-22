param(
    [string]$ApiKey = $env:OPENAI_API_KEY,
    [string]$PythonExe = ".\\.venv-3-11\\Scripts\\python.exe",
    [string]$Model = "gpt-5-mini",
    [string]$FunctionId = "fn_o",
    [ValidateSet("cached_splits", "fresh_seeds")]
    [string]$Mode = "cached_splits",
    [int[]]$Seeds = @(2201, 2202, 2203),
    [string[]]$DatasetSourceSeedDirs = @(),
    [int]$TrainSize = 100,
    [int]$ValSize = 100,
    [int]$TestSize = 3000,
    [int]$BatchSize = 20,
    [int]$BatchAttempts = 2,
    [int]$NumTrials = 1,
    [string]$PromptVariant = "explain",
    [string]$ReasoningEffort = "minimal",
    [int]$Concurrency = 1,
    [int]$MaxOutputTokens = 2000,
    [string]$OutRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $ApiKey) {
    throw "OPENAI_API_KEY is missing."
}

if (-not $OutRoot) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutRoot = "program_synthesis/runs_openai_batch_matrix/$stamp"
}

$compareScript = Join-Path $PSScriptRoot "run_openai_code0_batch_compare.ps1"
if (-not (Test-Path $compareScript)) {
    throw "Missing compare wrapper: $compareScript"
}

if ($Mode -eq "cached_splits" -and $DatasetSourceSeedDirs.Count -eq 0) {
    if ($FunctionId -ne "fn_o") {
        throw "Default cached split list is only defined for fn_o. Pass -DatasetSourceSeedDirs explicitly."
    }
    $DatasetSourceSeedDirs = @(
        ".\program_synthesis\runs_step23_live_matrix\20260218_192740\datasets\cdc_diabetes\L21\seed2062597811",
        ".\program_synthesis\runs_step23_live_matrix\20260218_192740\datasets\cdc_diabetes\L21\seed436481847",
        ".\program_synthesis\runs_step23_live_matrix\20260218_192740\datasets\cdc_diabetes\L21\seed747204404"
    )
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$safeModel = ($Model -replace "[^A-Za-z0-9._-]", "_")
$matrixLog = Join-Path $OutRoot "matrix.log"
$summaryCsv = Join-Path $OutRoot "summary.csv"
$manifestJson = Join-Path $OutRoot "manifest.json"

function Get-SummaryRow {
    param(
        [string]$CsvPath
    )

    if (-not (Test-Path $CsvPath)) {
        return $null
    }

    return Import-Csv $CsvPath | Where-Object { $_.is_summary -eq "True" } | Select-Object -First 1
}

function New-RunDescriptor {
    param(
        [string]$RunMode,
        [int]$RunSeed,
        [string]$SourceDir
    )

    if ($RunMode -eq "fresh_seeds") {
        $label = "seed$RunSeed"
        return [PSCustomObject]@{
            label = $label
            seed = $RunSeed
            dataset_source_seed_dir = ""
            run_out_root = (Join-Path $OutRoot $label)
        }
    }

    $leaf = Split-Path $SourceDir -Leaf
    $match = [regex]::Match($leaf, "^seed(?<seed>\d+)$")
    if (-not $match.Success) {
        throw "Cached split path must end in seed<digits>: $SourceDir"
    }
    $seedValue = [int]$match.Groups["seed"].Value
    $label = "cached_$leaf"
    return [PSCustomObject]@{
        label = $label
        seed = $seedValue
        dataset_source_seed_dir = $SourceDir
        run_out_root = (Join-Path $OutRoot $label)
    }
}

$descriptors = @()
if ($Mode -eq "fresh_seeds") {
    foreach ($seed in $Seeds) {
        $descriptors += New-RunDescriptor -RunMode $Mode -RunSeed $seed -SourceDir ""
    }
} else {
    foreach ($sourceDir in $DatasetSourceSeedDirs) {
        $descriptors += New-RunDescriptor -RunMode $Mode -RunSeed 0 -SourceDir $sourceDir
    }
}

$manifest = [ordered]@{
    mode = $Mode
    model = $Model
    function = $FunctionId
    train_size = $TrainSize
    val_size = $ValSize
    test_size = $TestSize
    batch_size = $BatchSize
    batch_attempts = $BatchAttempts
    normal_attempts = $BatchAttempts * [int][Math]::Ceiling($TrainSize / [double]$BatchSize)
    num_trials = $NumTrials
    prompt_variant = $PromptVariant
    reasoning_effort = $ReasoningEffort
    concurrency = $Concurrency
    max_output_tokens = $MaxOutputTokens
    run_count = $descriptors.Count
    created_at = (Get-Date).ToString("s")
    descriptors = $descriptors
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestJson -Encoding UTF8

"=== Code0 Batch Matrix ===" | Tee-Object -FilePath $matrixLog
"Mode: $Mode" | Tee-Object -FilePath $matrixLog -Append
"Model: $Model" | Tee-Object -FilePath $matrixLog -Append
"Function: $FunctionId" | Tee-Object -FilePath $matrixLog -Append
"Train/Val/Test: $TrainSize / $ValSize / $TestSize" | Tee-Object -FilePath $matrixLog -Append
"Batch size / attempts: $BatchSize / $BatchAttempts" | Tee-Object -FilePath $matrixLog -Append
"Summary CSV: $summaryCsv" | Tee-Object -FilePath $matrixLog -Append
"Manifest: $manifestJson" | Tee-Object -FilePath $matrixLog -Append
"" | Tee-Object -FilePath $matrixLog -Append

$results = @()

foreach ($descriptor in $descriptors) {
    "" | Tee-Object -FilePath $matrixLog -Append
    "=== Running $($descriptor.label) ===" | Tee-Object -FilePath $matrixLog -Append
    "Seed: $($descriptor.seed)" | Tee-Object -FilePath $matrixLog -Append
    if ($descriptor.dataset_source_seed_dir) {
        "Dataset source: $($descriptor.dataset_source_seed_dir)" | Tee-Object -FilePath $matrixLog -Append
    }
    "Run root: $($descriptor.run_out_root)" | Tee-Object -FilePath $matrixLog -Append

    $runStatus = "ok"
    $errorText = ""
    try {
        $scriptParams = @{
            ApiKey = $ApiKey
            PythonExe = $PythonExe
            Model = $Model
            FunctionId = $FunctionId
            Seed = $descriptor.seed
            TrainSize = $TrainSize
            ValSize = $ValSize
            TestSize = $TestSize
            BatchSize = $BatchSize
            BatchAttempts = $BatchAttempts
            NumTrials = $NumTrials
            PromptVariant = $PromptVariant
            ReasoningEffort = $ReasoningEffort
            Concurrency = $Concurrency
            MaxOutputTokens = $MaxOutputTokens
            OutRoot = $descriptor.run_out_root
        }
        if ($descriptor.dataset_source_seed_dir) {
            $scriptParams.DatasetSourceSeedDir = $descriptor.dataset_source_seed_dir
        }

        & $compareScript @scriptParams 2>&1 | Tee-Object -FilePath $matrixLog -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Compare wrapper exited with code $LASTEXITCODE"
        }
    } catch {
        $runStatus = "error"
        $errorText = $_.Exception.Message
        "Run failed: $errorText" | Tee-Object -FilePath $matrixLog -Append
    }

    $modelRoot = Join-Path $descriptor.run_out_root $safeModel
    $batchedCsvPath = Join-Path $modelRoot "batched\results.csv"
    $normalCsvPath = Join-Path $modelRoot "normal\results.csv"
    $batchedSummary = Get-SummaryRow -CsvPath $batchedCsvPath
    $normalSummary = Get-SummaryRow -CsvPath $normalCsvPath

    if ($runStatus -eq "ok" -and (-not $batchedSummary -or -not $normalSummary)) {
        $runStatus = "missing_summary"
        if (-not $errorText) {
            $errorText = "Missing final summary row in batched or normal results.csv"
        }
    }

    $batchedTrain = $null
    $batchedVal = $null
    $batchedTest = $null
    $normalTrain = $null
    $normalVal = $null
    $normalTest = $null
    $deltaTrain = $null
    $deltaVal = $null
    $deltaTest = $null

    if ($batchedSummary -and $normalSummary) {
        $batchedTrain = [double]$batchedSummary.train_acc
        $batchedVal = [double]$batchedSummary.val_acc
        $batchedTest = [double]$batchedSummary.test_acc
        $normalTrain = [double]$normalSummary.train_acc
        $normalVal = [double]$normalSummary.val_acc
        $normalTest = [double]$normalSummary.test_acc
        $deltaTrain = $batchedTrain - $normalTrain
        $deltaVal = $batchedVal - $normalVal
        $deltaTest = $batchedTest - $normalTest
    }

    $results += [PSCustomObject]@{
        label = $descriptor.label
        seed = $descriptor.seed
        status = $runStatus
        mode = $Mode
        model = $Model
        function = $FunctionId
        dataset_source_seed_dir = $descriptor.dataset_source_seed_dir
        run_root = $descriptor.run_out_root
        batched_csv = $batchedCsvPath
        normal_csv = $normalCsvPath
        batched_train = $batchedTrain
        batched_val = $batchedVal
        batched_test = $batchedTest
        normal_train = $normalTrain
        normal_val = $normalVal
        normal_test = $normalTest
        delta_train = $deltaTrain
        delta_val = $deltaVal
        delta_test = $deltaTest
        error = $errorText
    }
}

$results | Export-Csv -NoTypeInformation -Path $summaryCsv

"" | Tee-Object -FilePath $matrixLog -Append
"=== Final Summary ===" | Tee-Object -FilePath $matrixLog -Append
$results | Format-Table -AutoSize | Out-String | Tee-Object -FilePath $matrixLog -Append
$results | Format-Table -AutoSize

$okRows = @($results | Where-Object { $_.status -eq "ok" -and $_.delta_train -ne $null })
if ($okRows.Count -gt 0) {
    $aggregate = [PSCustomObject]@{
        runs = $okRows.Count
        mean_delta_train = ($okRows | Measure-Object delta_train -Average).Average
        mean_delta_val = ($okRows | Measure-Object delta_val -Average).Average
        mean_delta_test = ($okRows | Measure-Object delta_test -Average).Average
    }
    "" | Tee-Object -FilePath $matrixLog -Append
    "=== Aggregate Deltas ===" | Tee-Object -FilePath $matrixLog -Append
    $aggregate | Format-List | Out-String | Tee-Object -FilePath $matrixLog -Append
    $aggregate | Format-List
}

Write-Host ""
Write-Host "Done."
Write-Host "Artifacts:"
Write-Host "  matrix_log = $matrixLog"
Write-Host "  summary_csv = $summaryCsv"
Write-Host "  manifest_json = $manifestJson"
