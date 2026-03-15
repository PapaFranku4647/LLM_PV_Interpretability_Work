$ErrorActionPreference = "Continue"

$OUT_ROOT = "program_synthesis/runs_code0_prompt_comparison/20260303_115540"
$SHARED_DS = "$OUT_ROOT/shared_datasets"

$VARIANTS = @(
    "standard",
    "explain",
    "interview",
    "preview",
    "multipath",
    "subgroups",
    "thesis_aware",
    "regional",
    "ensemble"
)

foreach ($V in $VARIANTS) {
    Write-Host "=== $V ==="
    $cmd = @(
        "program_synthesis/run_step23_live_matrix.py",
        "--functions", "fn_n", "fn_o",
        "--seeds", "2201", "2202",
        "--samples-per-seed", "10",
        "--attempts", "5",
        "--prompt-variant", $V,
        "--thesis-prompt-version", "v2",
        "--model", "gpt-5-mini",
        "--reasoning-effort", "medium",
        "--max-output-tokens", "8000",
        "--code0-max-output-tokens", "16000",
        "--auto-split",
        "--dataset-dir", $SHARED_DS,
        "--out-root", "$OUT_ROOT/medium/$V"
    )
    python @cmd
    Write-Host ""
}

Write-Host "Done! Both seeds in one run per variant."
Write-Host "Results in: $OUT_ROOT/medium/"
