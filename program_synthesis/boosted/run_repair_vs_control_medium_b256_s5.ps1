param(
    [string]$PythonExe = "",
    [string]$OutRoot = "LLM_PV\program_synthesis\boosted\runs\repair_vs_control_medium_b256_s5",
    [int[]]$Seeds = @(2201, 2202, 2203, 2204, 2205),
    [int]$TrainSize = 10000,
    [int]$TestSize = 60000,
    [int]$BatchSize = 256,
    [int]$NumTrials = 1,
    [int]$RoundRetries = 3,
    [int]$RepairMistakeLimit = 8,
    [int]$RepairAnchorCount = 4,
    [string]$ReasoningEffort = "medium",
    [int]$MaxOutputTokens = 20000,
    [ValidateSet("t4_control", "t4_repair", "t8_control", "t8_repair")]
    [string[]]$Jobs = @("t4_control", "t4_repair", "t8_control", "t8_repair"),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$workspaceRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$repoRoot = Join-Path $workspaceRoot "LLM_PV"
Set-Location $workspaceRoot

if (-not $PythonExe) {
    $candidates = @(
        (Join-Path $workspaceRoot ".llm-pv-venv\Scripts\python.exe"),
        (Join-Path $workspaceRoot ".venv-3-11\Scripts\python.exe"),
        (Join-Path $repoRoot ".llm-pv-venv\Scripts\python.exe"),
        (Join-Path $repoRoot ".venv-3-11\Scripts\python.exe")
    )
    $PythonExe = ($candidates | Where-Object { Test-Path $_ } | Select-Object -First 1)
    if (-not $PythonExe) {
        $PythonExe = "python"
    }
}

$fullOutRoot = if ([System.IO.Path]::IsPathRooted($OutRoot)) {
    $OutRoot
} else {
    Join-Path $workspaceRoot $OutRoot
}

New-Item -ItemType Directory -Force -Path $fullOutRoot | Out-Null
$logPath = Join-Path $fullOutRoot "run.log"

function Write-RunLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "s"), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

function Get-JobSpec {
    param([string]$JobName)

    switch ($JobName) {
        "t4_control" { return @{ Name = $JobName; BoostRounds = 4; RepairRounds = 0 } }
        "t4_repair"  { return @{ Name = $JobName; BoostRounds = 4; RepairRounds = 1 } }
        "t8_control" { return @{ Name = $JobName; BoostRounds = 8; RepairRounds = 0 } }
        "t8_repair"  { return @{ Name = $JobName; BoostRounds = 8; RepairRounds = 1 } }
        default { throw "Unknown job: $JobName" }
    }
}

function Invoke-Runner {
    param(
        [string]$SeedOut,
        [int]$Seed,
        [hashtable]$Spec
    )

    $cmdArgs = @(
        "LLM_PV\program_synthesis\boosted\boosted_runner.py",
        "--provider", "openai",
        "--api-mode", "chat_completions",
        "--functions", "fn_o",
        "--lengths", "21",
        "--train-size", $TrainSize,
        "--val-size", "0",
        "--test-size", $TestSize,
        "--seed", $Seed,
        "--batch-sizes", $BatchSize,
        "--boost-rounds", $Spec.BoostRounds,
        "--num-trials", $NumTrials,
        "--round-retries", $RoundRetries,
        "--repair-rounds", $Spec.RepairRounds,
        "--repair-mistake-limit", $RepairMistakeLimit,
        "--repair-anchor-count", $RepairAnchorCount,
        "--reasoning-effort", $ReasoningEffort,
        "--max-output-tokens", $MaxOutputTokens,
        "--no-tools",
        "--output-dir", $SeedOut
    )

    if ($DryRun) {
        Write-RunLog ("dryrun python={0} args={1}" -f $PythonExe, ($cmdArgs -join " "))
        return
    }

    & $PythonExe @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Boosted runner failed for job=$($Spec.Name) seed=$Seed exit=$LASTEXITCODE"
    }
}

function Invoke-Analysis {
    param([string]$RunRoot)

    $analysisArgs = @(
        "LLM_PV\program_synthesis\boosted\analyze_scale_matrix.py",
        "--run-dir", $RunRoot
    )

    if ($DryRun) {
        Write-RunLog ("dryrun python={0} args={1}" -f $PythonExe, ($analysisArgs -join " "))
        return
    }

    & $PythonExe @analysisArgs
    if ($LASTEXITCODE -ne 0) {
        throw "analyze_scale_matrix.py failed for run-dir $RunRoot"
    }
}

function Get-StdDev {
    param([double[]]$Values)

    if (-not $Values -or $Values.Count -le 1) {
        return 0.0
    }

    $mean = ($Values | Measure-Object -Average).Average
    $sumSq = 0.0
    foreach ($value in $Values) {
        $sumSq += [math]::Pow($value - $mean, 2)
    }
    return [math]::Sqrt($sumSq / ($Values.Count - 1))
}

function Get-JobAggregateRows {
    param(
        [string]$JobName,
        [hashtable]$Spec
    )

    $jobRoot = Join-Path $fullOutRoot $JobName
    $aggPath = Join-Path $jobRoot "combined_aggregate_summary.csv"
    if (Test-Path $aggPath) {
        return Import-Csv $aggPath | ForEach-Object {
            [PSCustomObject]@{
                source             = "current"
                run_name           = $JobName
                T                  = $_.T
                batch_size         = $_.batch_size
                seeds              = $_.seeds
                mean_train_acc     = $_.mean_train_acc
                std_train_acc      = $_.std_train_acc
                mean_test_acc      = $_.mean_test_acc
                std_test_acc       = $_.std_test_acc
                mean_cost_usd      = $_.mean_cost_usd
                total_cost_usd     = $_.total_cost_usd
                all_rounds_accepted = $_.all_rounds_accepted
            }
        }
    }

    $summaryFiles = Get-ChildItem $jobRoot -Recurse -Filter summary.csv -ErrorAction SilentlyContinue
    if (-not $summaryFiles) {
        throw "Missing aggregate summary and per-seed summaries for job $JobName"
    }

    $summaryRows = foreach ($summaryFile in $summaryFiles) {
        Import-Csv $summaryFile.FullName
    }

    $trainValues = @($summaryRows | ForEach-Object { [double]$_.final_train_acc })
    $testValues = @($summaryRows | ForEach-Object { [double]$_.final_test_acc })
    $costValues = @($summaryRows | ForEach-Object { [double]$_.total_estimated_cost_usd })
    $acceptedAll = (($summaryRows | Where-Object { [int]$_.accepted_rounds -lt [int]$_.requested_rounds }).Count -eq 0)

    return @(
        [PSCustomObject]@{
            source              = "current"
            run_name            = $JobName
            T                   = $Spec.BoostRounds
            batch_size          = $BatchSize
            seeds               = $summaryRows.Count
            mean_train_acc      = ($trainValues | Measure-Object -Average).Average
            std_train_acc       = Get-StdDev -Values $trainValues
            mean_test_acc       = ($testValues | Measure-Object -Average).Average
            std_test_acc        = Get-StdDev -Values $testValues
            mean_cost_usd       = ($costValues | Measure-Object -Average).Average
            total_cost_usd      = ($costValues | Measure-Object -Sum).Sum
            all_rounds_accepted = $acceptedAll
        }
    )
}

Write-RunLog "start jobs=$($Jobs -join ',') batch=$BatchSize train=$TrainSize test=$TestSize retries=$RoundRetries reasoning=$ReasoningEffort max_output_tokens=$MaxOutputTokens seeds=$($Seeds -join ',')"
Write-RunLog "python=$PythonExe dry_run=$DryRun"

foreach ($jobName in $Jobs) {
    $spec = Get-JobSpec -JobName $jobName
    $runRoot = Join-Path $fullOutRoot $spec.Name

    New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
    Write-RunLog "job_start name=$($spec.Name) T=$($spec.BoostRounds) repair_rounds=$($spec.RepairRounds)"

    foreach ($seed in $Seeds) {
        $seedOut = Join-Path $runRoot ("T{0}\seed{1}" -f $spec.BoostRounds, $seed)
        $summaryCsv = Join-Path $seedOut "summary.csv"
        if ((-not $DryRun) -and (Test-Path $summaryCsv)) {
            Write-RunLog "skip_existing job=$($spec.Name) seed=$seed"
            continue
        }

        New-Item -ItemType Directory -Force -Path $seedOut | Out-Null
        Write-RunLog "seed_start job=$($spec.Name) seed=$seed"
        Invoke-Runner -SeedOut $seedOut -Seed $seed -Spec $spec
        Write-RunLog "seed_done job=$($spec.Name) seed=$seed"
    }

    Write-RunLog "analyze_start job=$($spec.Name)"
    Invoke-Analysis -RunRoot $runRoot
    Write-RunLog "analyze_done job=$($spec.Name)"
}

if (-not $DryRun) {
    $priorPath = Join-Path $repoRoot "program_synthesis\boosted\runs\s5_all_t1_t8\combined_aggregate_summary_ranked.csv"
    $comparisonCsv = Join-Path $fullOutRoot "comparison.csv"
    $comparisonTxt = Join-Path $fullOutRoot "comparison.txt"

    $rows = @()
    foreach ($jobName in $Jobs) {
        $spec = Get-JobSpec -JobName $jobName
        $rows += Get-JobAggregateRows -JobName $jobName -Spec $spec
    }

    if (Test-Path $priorPath) {
        $rows += Import-Csv $priorPath |
            Where-Object { $_.batch_size -eq "$BatchSize" -and $_.T -in @("4", "8") } |
            ForEach-Object {
                [PSCustomObject]@{
                    source                = "prior_bundle"
                    run_name               = "prior_T$($_.T)_b$($_.batch_size)"
                    T                      = $_.T
                    batch_size             = $_.batch_size
                    seeds                  = $_.seeds
                    mean_train_acc         = $_.mean_train_acc
                    std_train_acc          = $_.std_train_acc
                    mean_test_acc          = $_.mean_test_acc
                    std_test_acc           = $_.std_test_acc
                    mean_cost_usd          = $_.mean_cost_usd
                    total_cost_usd         = $_.total_cost_usd
                    all_rounds_accepted    = $_.all_rounds_accepted
                }
            }
    }

    $rows | Export-Csv -NoTypeInformation -Path $comparisonCsv

    $lines = @(
        "Repair vs control comparison",
        "batch_size=$BatchSize reasoning_effort=$ReasoningEffort max_output_tokens=$MaxOutputTokens",
        ""
    )
    $lines += ($rows |
        Sort-Object source, {[int]$_.T}, run_name |
        Format-Table source,run_name,T,batch_size,seeds,mean_train_acc,mean_test_acc,std_test_acc,mean_cost_usd,all_rounds_accepted -AutoSize |
        Out-String).TrimEnd()
    Set-Content -Path $comparisonTxt -Value $lines

    Write-RunLog "comparison_written csv=$comparisonCsv txt=$comparisonTxt"
}

Write-RunLog "complete"
