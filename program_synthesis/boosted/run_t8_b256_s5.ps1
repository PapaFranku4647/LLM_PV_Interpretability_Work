param(
    [string]$PythonExe = "",
    [string]$OutRoot = "program_synthesis\boosted\runs\t8_b256_s5",
    [int[]]$Seeds = @(2201, 2202, 2203, 2204, 2205),
    [int]$TrainSize = 10000,
    [int]$TestSize = 60000,
    [int]$BatchSize = 256,
    [int]$BoostRounds = 8,
    [int]$NumTrials = 1,
    [int]$RoundRetries = 3
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

if (-not $PythonExe) {
    $candidate = Join-Path $repoRoot ".llm-pv-venv\Scripts\python.exe"
    if (Test-Path $candidate) {
        $PythonExe = $candidate
    } else {
        $PythonExe = "python"
    }
}

$fullOutRoot = if ([System.IO.Path]::IsPathRooted($OutRoot)) {
    $OutRoot
} else {
    Join-Path $repoRoot $OutRoot
}

New-Item -ItemType Directory -Force -Path $fullOutRoot | Out-Null
$logPath = Join-Path $fullOutRoot "run.log"

function Write-RunLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "s"), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

Write-RunLog "start T=$BoostRounds batch=$BatchSize train=$TrainSize test=$TestSize seeds=$($Seeds -join ',')"

foreach ($seed in $Seeds) {
    $seedOut = Join-Path $fullOutRoot ("T{0}\seed{1}" -f $BoostRounds, $seed)
    $summaryCsv = Join-Path $seedOut "summary.csv"
    if (Test-Path $summaryCsv) {
        Write-RunLog "skip existing seed=$seed"
        continue
    }

    New-Item -ItemType Directory -Force -Path $seedOut | Out-Null
    Write-RunLog "start seed=$seed"

    & $PythonExe `
        "program_synthesis\boosted\boosted_runner.py" `
        --provider openai `
        --api-mode chat_completions `
        --functions fn_o `
        --lengths 21 `
        --train-size $TrainSize `
        --val-size 0 `
        --test-size $TestSize `
        --seed $seed `
        --batch-sizes $BatchSize `
        --boost-rounds $BoostRounds `
        --num-trials $NumTrials `
        --round-retries $RoundRetries `
        --no-tools `
        --output-dir $seedOut

    if ($LASTEXITCODE -ne 0) {
        Write-RunLog "failed seed=$seed exit=$LASTEXITCODE"
        throw "Boosted runner failed for seed $seed"
    }

    Write-RunLog "done seed=$seed"
}

Write-RunLog "analyze aggregate"
& $PythonExe "program_synthesis\boosted\analyze_scale_matrix.py" --run-dir $fullOutRoot
if ($LASTEXITCODE -ne 0) {
    Write-RunLog "analysis_failed exit=$LASTEXITCODE"
    throw "analyze_scale_matrix.py failed"
}

Write-RunLog "complete"
