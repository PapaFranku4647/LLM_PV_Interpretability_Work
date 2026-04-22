"""
Overnight batch runner for Code0 prompt variant experiments.
Runs: multipath, regional, subgroups, ensemble
(thesis_aware already completed as 20260225_224446)

All runs use:
  - Same seed (2201), same dataset (fn_o), same splits (200/2300/7500)
  - Same thesis prompt (V2), same reasoning (high), same model (gpt-5-mini)
  - EXACT same training/val/test data (copied from thesis_aware run)
  - Only --prompt-variant changes between runs

Usage:
  python _run_overnight_code0_prompts.py
"""

import subprocess
import sys
import shutil
import time
from pathlib import Path

REPO = Path(r"C:\Users\lucas\Desktop\Coding\TomerResearch\LLM_PV_Working_Copy")
VENV_PYTHON = REPO / ".venv-3-11" / "Scripts" / "python.exe"

# Source of verified datasets (byte-identical to all prior runs)
DATASET_SOURCE = REPO / "program_synthesis" / "runs_step23_live_matrix" / "20260225_fn_o_diabetes_25s_code0_thesis_aware_high" / "datasets"

VARIANTS = ["multipath", "regional", "subgroups", "ensemble"]

BASE_ARGS = [
    str(VENV_PYTHON), "-m", "program_synthesis.thesis_runner",
    "--functions", "fn_o",
    "--seeds", "2201",
    "--samples-per-seed", "25",
    "--attempts", "10",
    "--num-trials", "1",
    "--train-size", "200",
    "--val-size", "2300",
    "--test-size", "7500",
    "--compute-baselines",
    "--thesis-prompt-version", "v2",
    "--reasoning-effort", "high",
    "--max-output-tokens", "16000",
    "--auto-split",
    "--total-cap", "10000",
]


def run_variant(variant: str):
    print(f"\n{'='*70}")
    print(f"STARTING: --prompt-variant {variant}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    cmd = BASE_ARGS + ["--prompt-variant", variant]

    start = time.time()
    result = subprocess.run(cmd, cwd=str(REPO))
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n{'='*70}")
    print(f"DONE: {variant} — {status} — {elapsed/60:.1f} min")
    print(f"{'='*70}\n")

    return result.returncode


def main():
    print(f"Overnight Code0 Prompt Experiment Runner")
    print(f"Variants to run: {VARIANTS}")
    print(f"Dataset source: {DATASET_SOURCE}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if not DATASET_SOURCE.is_dir():
        print(f"ERROR: Dataset source not found: {DATASET_SOURCE}")
        sys.exit(1)

    results = {}
    for variant in VARIANTS:
        rc = run_variant(variant)
        results[variant] = rc

    print(f"\n{'='*70}")
    print(f"ALL DONE — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    for v, rc in results.items():
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  {v:20s} {status}")


if __name__ == "__main__":
    main()
