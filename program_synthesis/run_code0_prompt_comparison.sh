#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="program_synthesis/runs_code0_prompt_comparison/${TIMESTAMP}"
SHARED_DATASET_DIR="${OUT_ROOT}/shared_datasets"

DATASETS="fn_n fn_p fn_o"
SEEDS="2201"
SAMPLES=10
MODEL="gpt-5-mini"
THESIS_VERSION="v2"
ATTEMPTS=5
CODE0_MAX_TOKENS=16000

VARIANTS=(
    standard
    explain
    interview
    preview
    multipath
    subgroups
    thesis_aware
    regional
    ensemble
)

REASONING_LEVELS=(
    low
    medium
    high
)

echo "=== Code0 Prompt Comparison Experiment ==="
echo "Output root: ${OUT_ROOT}"
echo "Datasets: ${DATASETS}"
echo "Seeds: ${SEEDS}"
echo "Samples/seed: ${SAMPLES}"
echo "Model: ${MODEL}"
echo "Reasoning levels: ${REASONING_LEVELS[*]}"
echo "Code0 attempts: ${ATTEMPTS}"
echo "Code0 max tokens: ${CODE0_MAX_TOKENS}"
echo "Thesis prompt: ${THESIS_VERSION}"
echo "Variants: ${VARIANTS[*]}"
echo "Shared dataset dir: ${SHARED_DATASET_DIR}"
echo ""

for REASONING in "${REASONING_LEVELS[@]}"; do
    echo "====== Reasoning level: ${REASONING} ======"
    for VARIANT in "${VARIANTS[@]}"; do
        VARIANT_DIR="${OUT_ROOT}/${REASONING}/${VARIANT}"
        echo "--- Running variant: ${VARIANT} (reasoning=${REASONING}) ---"
        python program_synthesis/run_step23_live_matrix.py \
            --functions ${DATASETS} \
            --seeds ${SEEDS} \
            --samples-per-seed ${SAMPLES} \
            --attempts ${ATTEMPTS} \
            --prompt-variant "${VARIANT}" \
            --thesis-prompt-version "${THESIS_VERSION}" \
            --model "${MODEL}" \
            --reasoning-effort "${REASONING}" \
            --code0-max-output-tokens ${CODE0_MAX_TOKENS} \
            --auto-split \
            --dataset-dir "${SHARED_DATASET_DIR}" \
            --out-root "${VARIANT_DIR}" \
        2>&1 | tee "${VARIANT_DIR}/run.log" || {
            echo "WARNING: variant ${VARIANT} (reasoning=${REASONING}) failed, continuing..."
            continue
        }
        echo "--- Done: ${VARIANT} (reasoning=${REASONING}) ---"
        echo ""
    done
done

echo "=== All variants done ==="
echo "Results in: ${OUT_ROOT}"
echo ""
echo "Run analysis with:"
echo "  python program_synthesis/analyze_code0_prompt_comparison.py ${OUT_ROOT}"
