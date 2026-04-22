param(
    [string]$PythonExe = "",
    [string]$OutRoot = "program_synthesis\outputs_llmpv_baseline_20260401_5seed\trainbudget10000_test60000_attempts4",
    [int[]]$BatchSizes = @(16, 32, 64, 128),
    [int[]]$Seeds = @(2201, 2202, 2203, 2204, 2205),
    [int]$Attempts = 4,
    [int]$NumTrials = 1,
    [int]$TestSize = 60000,
    [int]$TotalLabelBudget = 10000,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
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
$logPath = Join-Path $fullOutRoot "resume_run.log"

function Write-ResumeLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "s"), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

function Test-RunnerAlreadyActive {
    param([string]$OutJsonlPath)
    $needle = $OutJsonlPath.Replace('\', '\\')
    $active = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -and
            $_.CommandLine -like "*runner.py*" -and
            $_.CommandLine -like "*$needle*"
        }
    return ($active | Measure-Object).Count -gt 0
}

foreach ($batch in $BatchSizes) {
    $valSize = $TotalLabelBudget - $batch
    if ($valSize -lt 0) {
        throw "Invalid batch size $batch for total label budget $TotalLabelBudget"
    }

    foreach ($seed in $Seeds) {
        $batchDir = Join-Path $fullOutRoot ("batch{0}" -f $batch)
        $seedDir = Join-Path $batchDir ("seed{0}" -f $seed)
        $jsonl = Join-Path $seedDir "results_attempts.jsonl"
        $csv = Join-Path $seedDir "results_attempts.csv"

        if (Test-Path $csv) {
            Write-ResumeLog "skip existing batch=$batch seed=$seed"
            continue
        }

        while (Test-RunnerAlreadyActive -OutJsonlPath $jsonl) {
            Write-ResumeLog "waiting for active run batch=$batch seed=$seed"
            Start-Sleep -Seconds $PollSeconds
            if (Test-Path $csv) {
                break
            }
        }
        if (Test-Path $csv) {
            Write-ResumeLog "completed while waiting batch=$batch seed=$seed"
            continue
        }

        New-Item -ItemType Directory -Force -Path $seedDir | Out-Null
        $cmd = @(
            "program_synthesis\runner.py",
            "--provider", "openai",
            "--api-mode", "chat_completions",
            "--functions", "fn_o",
            "--lengths", "21",
            "--attempts", "$Attempts",
            "--num-trials", "$NumTrials",
            "--train-size", "$batch",
            "--val-size", "$valSize",
            "--test-size", "$TestSize",
            "--seed", "$seed",
            "--no-tools",
            "--out-jsonl", $jsonl,
            "--out-csv", $csv,
            "--concurrency", "1",
            "--timeout", "1200"
        )

        Write-ResumeLog "starting batch=$batch seed=$seed"
        & $PythonExe @cmd
        if ($LASTEXITCODE -ne 0) {
            Write-ResumeLog "runner failed batch=$batch seed=$seed exit=$LASTEXITCODE"
        } elseif (Test-Path $csv) {
            Write-ResumeLog "finished batch=$batch seed=$seed"
        } else {
            Write-ResumeLog "runner exited without csv batch=$batch seed=$seed"
        }
    }
}

Write-ResumeLog "resume matrix complete"
