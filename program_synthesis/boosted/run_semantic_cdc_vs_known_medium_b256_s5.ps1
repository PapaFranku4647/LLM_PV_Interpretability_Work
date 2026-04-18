param(
    [int[]]$Seeds = @(2206, 2207, 2208, 2209, 2210),
    [string[]]$Jobs = @("semantic_t1_control", "semantic_t8_control", "semantic_t8_gated_repair"),
    [string]$OutRoot = "LLM_PV\program_synthesis\boosted\runs\semantic_cdc_vs_known_medium_b256_s5",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (Test-Path ".\LLM_PV\.llm-pv-venv\Scripts\python.exe") {
    $python = ".\LLM_PV\.llm-pv-venv\Scripts\python.exe"
} elseif (Test-Path ".\.llm-pv-venv\Scripts\python.exe") {
    $python = ".\.llm-pv-venv\Scripts\python.exe"
} else {
    $python = "python"
}

$jobDefs = @{
    "semantic_t1_control" = @{
        T = 1
        RepairRounds = 0
        RepairArgs = @()
    }
    "semantic_t8_control" = @{
        T = 8
        RepairRounds = 0
        RepairArgs = @()
    }
    "semantic_t8_gated_repair" = @{
        T = 8
        RepairRounds = 1
        RepairArgs = @(
            "--repair-mistake-limit", "8",
            "--repair-anchor-count", "4",
            "--repair-trigger-batch-acc-below", "0.60",
            "--repair-trigger-weighted-error-above", "0.40",
            "--repair-trigger-min-mistakes", "8"
        )
    }
}

function Mean($values) {
    $nums = @($values | Where-Object { $_ -ne $null } | ForEach-Object { [double]$_ })
    if ($nums.Count -eq 0) { return "" }
    return ($nums | Measure-Object -Average).Average
}

function StdSample($values) {
    $nums = @($values | Where-Object { $_ -ne $null } | ForEach-Object { [double]$_ })
    if ($nums.Count -le 1) { return 0.0 }
    $avg = (Mean $nums)
    $sumSq = 0.0
    foreach ($num in $nums) {
        $sumSq += ($num - $avg) * ($num - $avg)
    }
    return [math]::Sqrt($sumSq / ($nums.Count - 1))
}

function Summarize-Job($jobName, $jobDef) {
    $rows = @()
    foreach ($seed in $Seeds) {
        $summaryPath = Join-Path $OutRoot (Join-Path $jobName ("T{0}\seed{1}\summary.csv" -f $jobDef.T, $seed))
        if (Test-Path $summaryPath) {
            $rows += Import-Csv $summaryPath
        }
    }
    if ($rows.Count -eq 0) { return $null }
    $acceptedAll = @($rows | Where-Object { [int]$_.accepted_rounds -eq [int]$_.requested_rounds }).Count -eq $rows.Count
    return [pscustomobject]@{
        source = "semantic_current"
        run_name = $jobName
        T = $jobDef.T
        batch_size = 256
        seeds = $rows.Count
        mean_train_acc = Mean ($rows | ForEach-Object { $_.final_train_acc })
        std_train_acc = StdSample ($rows | ForEach-Object { $_.final_train_acc })
        mean_test_acc = Mean ($rows | ForEach-Object { $_.final_test_acc })
        std_test_acc = StdSample ($rows | ForEach-Object { $_.final_test_acc })
        mean_cost_usd = Mean ($rows | ForEach-Object { $_.total_estimated_cost_usd })
        total_cost_usd = ($rows | ForEach-Object { [double]$_.total_estimated_cost_usd } | Measure-Object -Sum).Sum
        mean_api_attempts = Mean ($rows | ForEach-Object { $_.api_attempt_count })
        all_rounds_accepted = $acceptedAll
    }
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$logPath = Join-Path $OutRoot "run.log"
"[$(Get-Date -Format o)] start dry_run=$DryRun seeds=$($Seeds -join ',') jobs=$($Jobs -join ',')" | Tee-Object -FilePath $logPath -Append

foreach ($jobName in $Jobs) {
    if (-not $jobDefs.ContainsKey($jobName)) {
        throw "Unknown job '$jobName'. Known jobs: $($jobDefs.Keys -join ', ')"
    }
    $jobDef = $jobDefs[$jobName]
    "[$(Get-Date -Format o)] job_start name=$jobName T=$($jobDef.T) repair_rounds=$($jobDef.RepairRounds)" | Tee-Object -FilePath $logPath -Append

    foreach ($seed in $Seeds) {
        $seedOut = Join-Path $OutRoot (Join-Path $jobName ("T{0}\seed{1}" -f $jobDef.T, $seed))
        $summaryPath = Join-Path $seedOut "summary.csv"
        if (Test-Path $summaryPath) {
            "[$(Get-Date -Format o)] seed_skip job=$jobName seed=$seed summary_exists=$summaryPath" | Tee-Object -FilePath $logPath -Append
            continue
        }

        New-Item -ItemType Directory -Force -Path $seedOut | Out-Null
        $cmdArgs = @(
            "LLM_PV\program_synthesis\boosted\boosted_runner.py",
            "--provider", "openai",
            "--api-mode", "chat_completions",
            "--functions", "fn_o",
            "--lengths", "21",
            "--train-size", "10000",
            "--val-size", "0",
            "--test-size", "60000",
            "--seed", "$seed",
            "--batch-sizes", "256",
            "--boost-rounds", "$($jobDef.T)",
            "--num-trials", "1",
            "--round-retries", "1",
            "--sample-without-replacement",
            "--cdc-representation", "semantic",
            "--repair-rounds", "$($jobDef.RepairRounds)",
            "--reasoning-effort", "medium",
            "--max-output-tokens", "20000",
            "--no-tools",
            "--output-dir", $seedOut
        ) + $jobDef.RepairArgs

        "[$(Get-Date -Format o)] seed_start job=$jobName seed=$seed out=$seedOut" | Tee-Object -FilePath $logPath -Append
        if ($DryRun) {
            "$python $($cmdArgs -join ' ')" | Tee-Object -FilePath $logPath -Append
        } else {
            & $python @cmdArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Run failed: job=$jobName seed=$seed exit=$LASTEXITCODE"
            }
        }
        "[$(Get-Date -Format o)] seed_done job=$jobName seed=$seed" | Tee-Object -FilePath $logPath -Append
    }
}

$comparisonRows = @()
foreach ($jobName in ($jobDefs.Keys | Sort-Object)) {
    $summary = Summarize-Job $jobName $jobDefs[$jobName]
    if ($summary -ne $null) {
        $comparisonRows += $summary
    }
}

$priorPath = "LLM_PV\program_synthesis\boosted\runs\repair_vs_control_medium_b256_s5\comparison.csv"
if (Test-Path $priorPath) {
    $comparisonRows += Import-Csv $priorPath | Where-Object {
        $_.run_name -in @("t8_control", "t8_repair", "prior_T8_b256", "prior_T4_b256")
    }
}

$comparisonCsv = Join-Path $OutRoot "comparison.csv"
$comparisonTxt = Join-Path $OutRoot "comparison.txt"
$comparisonRows | Export-Csv -NoTypeInformation -Path $comparisonCsv
$comparisonRows |
    Sort-Object @{ Expression = { [double]$_.mean_test_acc }; Descending = $true } |
    Format-Table -AutoSize |
    Out-String |
    Set-Content -Path $comparisonTxt

"[$(Get-Date -Format o)] done comparison=$comparisonCsv" | Tee-Object -FilePath $logPath -Append
Get-Content $comparisonTxt
