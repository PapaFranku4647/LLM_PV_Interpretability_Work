#!/usr/bin/env python3
"""Analyze results from run_code0_prompt_comparison.sh.

Usage:
    python analyze_code0_prompt_comparison.py <results_root>

Where <results_root> has structure:
    <results_root>/<reasoning_level>/<variant>/  (with reasoning levels)
    <results_root>/<variant>/                     (flat, no reasoning levels)

Each leaf folder should contain an overall_summary.json from run_step23_live_matrix.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


METRICS = [
    ("val_acc", "mean_val_acc", ".3f"),
    ("test_acc", "mean_test_acc", ".3f"),
    ("train_acc", "mean_train_acc", ".3f"),
    ("cov_eq", "mean_coverage_eq_all", ".3f"),
    ("faith_c0", "mean_faithfulness_code0_all_zero", ".3f"),
    ("faith_gt", "mean_faithfulness_gt_all_zero", ".3f"),
    ("accepted", "accepted_rate", ".1%"),
    ("compile", "compile_ok_rate", ".1%"),
]

REASONING_LEVELS = {"low", "medium", "high", "minimal", "none"}


def load_variant(variant_dir: Path) -> dict | None:
    candidates = list(variant_dir.rglob("overall_summary.json"))
    if not candidates:
        return None
    with open(candidates[0], "r", encoding="utf-8") as f:
        return json.load(f)


def compute_composite(summary: dict) -> float:
    cov = summary.get("mean_coverage_eq_all", 0) or 0
    faith = summary.get("mean_faithfulness_code0_all_zero", 0) or 0
    acc = summary.get("mean_test_acc", 0) or 0
    return 0.4 * cov + 0.4 * faith + 0.2 * acc


def print_table(rows: list[tuple[str, dict]], label_width: int = 24) -> list[tuple[str, float, str]]:
    header_labels = [m[0] for m in METRICS]
    col_w = 10
    header = f"{'variant':<{label_width}}" + "".join(f"{h:>{col_w}}" for h in header_labels)
    print(header)
    print("-" * len(header))

    scored: list[tuple[str, float, str]] = []
    for name, summary in rows:
        parts = [f"{name:<{label_width}}"]
        for _, key, fmt in METRICS:
            val = summary.get(key)
            if val is not None:
                parts.append(f"{val:{fmt}:>{col_w}}")
            else:
                parts.append(f"{'n/a':>{col_w}}")
        print("".join(parts))

        cov = summary.get("mean_coverage_eq_all", 0) or 0
        faith = summary.get("mean_faithfulness_code0_all_zero", 0) or 0
        acc = summary.get("mean_test_acc", 0) or 0
        composite = compute_composite(summary)
        scored.append((name, composite, f"cov={cov:.3f} faith={faith:.3f} acc={acc:.3f}"))

    return scored


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        sys.exit(1)

    subdirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda d: d.name)
    has_reasoning_levels = any(d.name in REASONING_LEVELS for d in subdirs)

    all_scored: list[tuple[str, float, str]] = []

    if has_reasoning_levels:
        for reasoning_dir in subdirs:
            if not reasoning_dir.is_dir():
                continue
            reasoning = reasoning_dir.name
            variant_dirs = sorted([d for d in reasoning_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
            rows: list[tuple[str, dict]] = []
            for vdir in variant_dirs:
                summary = load_variant(vdir)
                if summary is None:
                    print(f"WARNING: no overall_summary.json in {reasoning}/{vdir.name}, skipping")
                    continue
                rows.append((vdir.name, summary))

            if not rows:
                continue

            print(f"\n{'='*60}")
            print(f"  Reasoning level: {reasoning}")
            print(f"{'='*60}\n")
            scored = print_table(rows)
            for name, score, detail in scored:
                all_scored.append((f"{reasoning}/{name}", score, detail))

            print()
            scored.sort(key=lambda x: -x[1])
            print(f"  Ranking ({reasoning}):")
            for rank, (name, score, detail) in enumerate(scored, 1):
                print(f"    {rank}. {name:<16} composite={score:.4f}  ({detail})")
    else:
        rows = []
        for vdir in subdirs:
            summary = load_variant(vdir)
            if summary is None:
                print(f"WARNING: no overall_summary.json in {vdir.name}, skipping")
                continue
            rows.append((vdir.name, summary))

        if not rows:
            print("No results found.")
            sys.exit(1)

        scored = print_table(rows)
        all_scored = scored

    if all_scored:
        print(f"\n{'='*60}")
        print("  OVERALL RANKING (0.4*cov + 0.4*faith + 0.2*acc)")
        print(f"{'='*60}\n")
        all_scored.sort(key=lambda x: -x[1])
        for rank, (name, score, detail) in enumerate(all_scored, 1):
            print(f"  {rank:>2}. {name:<28} composite={score:.4f}  ({detail})")

        if len(all_scored) >= 3:
            print()
            top3 = [s[0] for s in all_scored[:3]]
            print(f"Top 3 for confirmation on gpt-5.2: {', '.join(top3)}")


if __name__ == "__main__":
    main()
