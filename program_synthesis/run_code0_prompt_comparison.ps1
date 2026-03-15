$ErrorActionPreference = "Continue"

$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$OUT_ROOT = "program_synthesis/runs_code0_prompt_comparison/$TIMESTAMP"
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
        "--seeds", "2201",
        "--samples-per-seed", "10",
        "--attempts", "5",
        "--prompt-variant", $V,
        "--thesis-prompt-version", "v2",
        "--model", "gpt-5-mini",
        "--reasoning-effort", "low",
        "--code0-max-output-tokens", "16000",
        "--auto-split",
        "--dataset-dir", $SHARED_DS,
        "--out-root", "$OUT_ROOT/low/$V"
    )
    python @cmd
    Write-Host ""
}

Write-Host "Done! Analyze with:"
Write-Host "  python program_synthesis/analyze_code0_prompt_comparison.py $OUT_ROOT"
