"""Analysis utilities for thesis evaluation results.

Provides functions for loading cases.jsonl, computing trivial baselines,
per-function summaries, failure diagnostics, and prompt version comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

try:
    from program_synthesis.live_eval_common import parse_tabular_line, predict_code0_label
except ModuleNotFoundError:
    try:
        from live_eval_common import parse_tabular_line, predict_code0_label  # type: ignore
    except ModuleNotFoundError:
        parse_tabular_line = None  # type: ignore
        predict_code0_label = None  # type: ignore


def load_cases(cases_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with cases_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_trivial_baselines(
    train_lines: Sequence[str],
    code0_fn: Callable[..., Any],
) -> dict[str, dict[str, float]]:
    if parse_tabular_line is None:
        raise ImportError("parse_tabular_line not available")

    parsed = []
    for line in train_lines:
        sample, true_label = parse_tabular_line(line)
        try:
            pred_label, _ = predict_code0_label(code0_fn, sample)
        except Exception:
            pred_label = None
        parsed.append((sample, true_label, pred_label))

    n = len(parsed)
    if n == 0:
        empty: dict[str, float] = {"coverage_eq": 0.0, "faithfulness_code0": 0.0, "faithfulness_gt": 0.0}
        return {"always_1": empty, "always_0": empty}

    baselines: dict[str, dict[str, float]] = {}
    for strategy_label in [0, 1]:
        strategy_name = f"always_{strategy_label}"
        coverage = 1.0
        agree_code0 = sum(1 for _, _, p in parsed if p == strategy_label)
        agree_gt = sum(1 for _, gt, _ in parsed if gt == strategy_label)
        faithfulness_code0 = agree_code0 / n if n > 0 else 0.0
        faithfulness_gt = agree_gt / n if n > 0 else 0.0
        baselines[strategy_name] = {
            "coverage_eq": coverage,
            "faithfulness_code0": faithfulness_code0,
            "faithfulness_gt": faithfulness_gt,
        }

    return baselines


def per_function_summary(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        fn = case.get("fn", "unknown")
        groups.setdefault(fn, []).append(case)

    summaries = []
    for fn in sorted(groups):
        rows = groups[fn]
        n = len(rows)
        cov_values = [float(r.get("coverage_eq", 0.0)) for r in rows]

        # faithfulness_code0 with fallback to old "faithfulness" key
        faith_code0_values = []
        for r in rows:
            v = r.get("faithfulness_code0")
            if v is None:
                v = r.get("faithfulness")
            if v is not None:
                faith_code0_values.append(float(v))

        faith_gt_values = [float(r["faithfulness_gt"]) for r in rows if r.get("faithfulness_gt") is not None]

        x_in_a_count = sum(1 for r in rows if r.get("x_in_A"))
        accepted_count = sum(1 for r in rows if r.get("code1_accepted"))
        c1_errors = sum(int(r.get("code1_eval_errors", 0)) for r in rows)

        summaries.append({
            "fn": fn,
            "n_cases": n,
            "mean_coverage_eq": sum(cov_values) / n if n else 0.0,
            "mean_faithfulness": sum(faith_code0_values) / len(faith_code0_values) if faith_code0_values else None,
            "mean_faithfulness_gt": sum(faith_gt_values) / len(faith_gt_values) if faith_gt_values else None,
            "x_in_A_rate": x_in_a_count / n if n else 0.0,
            "accepted_rate": accepted_count / n if n else 0.0,
            "code1_eval_errors": c1_errors,
        })
    return summaries


def coverage_faith_pairs(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for case in cases:
        # faithfulness_code0 with fallback to old "faithfulness" key
        faith_code0 = case.get("faithfulness_code0")
        if faith_code0 is None:
            faith_code0 = case.get("faithfulness")
        if faith_code0 is None:
            continue
        entry: dict[str, Any] = {
            "fn": case.get("fn"),
            "seed": case.get("seed"),
            "sample_index": case.get("sample_index"),
            "coverage_eq": float(case.get("coverage_eq", 0.0)),
            "faithfulness": float(faith_code0),
            "faithfulness_gt": float(case["faithfulness_gt"]) if case.get("faithfulness_gt") is not None else None,
            "x_in_A": bool(case.get("x_in_A")),
        }
        pairs.append(entry)
    return pairs


_CATEGORICAL_RE = re.compile(r"c\d+")


def diagnose_failures(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for case in cases:
        if case.get("x_in_A"):
            continue
        if case.get("code0_pred_error"):
            continue

        reason = "unknown"
        conditions = case.get("thesis_conditions", "") or ""
        code1_errors = int(case.get("code1_eval_errors", 0))

        if code1_errors > 0:
            reason = "categorical_mismatch"
        elif not case.get("code1_compile_ok"):
            reason = "code1_compile_failure"
        elif ">" in conditions and ">=" not in conditions:
            reason = "boundary_exclusive"
        else:
            reason = "wrong_code_path"

        failures.append({
            "fn": case.get("fn"),
            "seed": case.get("seed"),
            "sample_index": case.get("sample_index"),
            "reason": reason,
            "thesis_conditions": conditions,
            "code1_eval_errors": code1_errors,
        })
    return failures


def thesis_complexity_stats(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    stats = []
    for case in cases:
        conditions = case.get("thesis_conditions", "") or ""
        n_conditions = conditions.upper().count(" AND ") + conditions.upper().count(" OR ") + 1 if conditions.strip() else 0
        stats.append({
            "fn": case.get("fn"),
            "seed": case.get("seed"),
            "sample_index": case.get("sample_index"),
            "n_conditions": n_conditions,
            "conditions_length": len(conditions),
            "coverage_eq": float(case.get("coverage_eq", 0.0)),
            "faithfulness": case.get("faithfulness"),
        })
    return stats


def compare_prompt_versions(
    v1_cases: Sequence[dict[str, Any]],
    v2_cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    def _aggregate(cases: Sequence[dict[str, Any]]) -> dict[str, float]:
        n = len(cases)
        if n == 0:
            return {"n": 0, "mean_coverage_eq": 0.0, "mean_faithfulness": 0.0, "mean_faithfulness_gt": 0.0, "x_in_A_rate": 0.0}
        cov = sum(float(c.get("coverage_eq", 0)) for c in cases) / n
        # faithfulness_code0 with fallback to old "faithfulness"
        faith_code0_vals = []
        for c in cases:
            v = c.get("faithfulness_code0")
            if v is None:
                v = c.get("faithfulness")
            if v is not None:
                faith_code0_vals.append(float(v))
        faith = sum(faith_code0_vals) / len(faith_code0_vals) if faith_code0_vals else 0.0
        faith_gt_vals = [float(c["faithfulness_gt"]) for c in cases if c.get("faithfulness_gt") is not None]
        faith_gt = sum(faith_gt_vals) / len(faith_gt_vals) if faith_gt_vals else 0.0
        x_in_a = sum(1 for c in cases if c.get("x_in_A")) / n
        return {"n": n, "mean_coverage_eq": cov, "mean_faithfulness": faith, "mean_faithfulness_gt": faith_gt, "x_in_A_rate": x_in_a}

    v1_agg = _aggregate(v1_cases)
    v2_agg = _aggregate(v2_cases)

    delta = {}
    for key in ["mean_coverage_eq", "mean_faithfulness", "mean_faithfulness_gt", "x_in_A_rate"]:
        delta[f"delta_{key}"] = v2_agg[key] - v1_agg[key]

    return {"v1": v1_agg, "v2": v2_agg, "delta": delta}


def print_markdown_table(rows: Sequence[dict[str, Any]], keys: Optional[Sequence[str]] = None) -> str:
    if not rows:
        return "(no data)"
    if keys is None:
        keys = list(rows[0].keys())

    def _fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    lines = [header, sep]
    for row in rows:
        line = "| " + " | ".join(_fmt(row.get(k)) for k in keys) + " |"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Thesis analysis and comparison tool.")
    parser.add_argument("--results-dir", action="append", required=True,
        help="Path to results directory containing cases.jsonl (can specify multiple).")
    parser.add_argument("--output-csv", default="",
        help="Optional CSV output path for per-function summary.")
    parser.add_argument("--baselines", action="store_true",
        help="Compute trivial baselines (requires code0 and train data).")
    args = parser.parse_args()

    all_dirs = [Path(d) for d in args.results_dir]
    all_cases_by_dir: list[tuple[Path, list[dict[str, Any]]]] = []
    for d in all_dirs:
        cases_path = d / "cases.jsonl"
        if not cases_path.exists():
            print(f"WARNING: {cases_path} not found, skipping.", file=sys.stderr)
            continue
        cases = load_cases(cases_path)
        all_cases_by_dir.append((d, cases))
        print(f"\n=== Results from {d} ({len(cases)} cases) ===")

        pf = per_function_summary(cases)
        print("\nPer-function summary:")
        print(print_markdown_table(pf))

        failures = diagnose_failures(cases)
        if failures:
            reason_counts: dict[str, int] = {}
            for f in failures:
                r = f["reason"]
                reason_counts[r] = reason_counts.get(r, 0) + 1
            print(f"\nFailure diagnosis (x_in_A=False): {len(failures)} cases")
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")

        complexity = thesis_complexity_stats(cases)
        if complexity:
            avg_n = sum(c["n_conditions"] for c in complexity) / len(complexity)
            avg_len = sum(c["conditions_length"] for c in complexity) / len(complexity)
            print(f"\nThesis complexity: avg_conditions={avg_n:.1f}, avg_length={avg_len:.1f}")

    if len(all_cases_by_dir) == 2:
        (d1, c1), (d2, c2) = all_cases_by_dir
        print(f"\n=== Comparison: {d1.name} vs {d2.name} ===")
        comparison = compare_prompt_versions(c1, c2)
        print(json.dumps(comparison, indent=2))

    if args.output_csv and all_cases_by_dir:
        all_cases = []
        for d, cases in all_cases_by_dir:
            for c in cases:
                c["results_dir"] = str(d)
            all_cases.extend(cases)

        pf_all = per_function_summary(all_cases)
        output_path = Path(args.output_csv)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            if pf_all:
                writer = csv.DictWriter(f, fieldnames=list(pf_all[0].keys()))
                writer.writeheader()
                for row in pf_all:
                    writer.writerow(row)
        print(f"\nWrote per-function summary to {output_path}")


if __name__ == "__main__":
    main()
