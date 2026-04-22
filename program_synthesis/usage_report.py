#!/usr/bin/env python
"""Summarize token usage and estimated API cost from runner CSV outputs."""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, Iterable, List

from llm_client import estimate_usage_cost, resolve_model_pricing


def _to_int(value) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(value))
    except Exception:
        return 0


def _load_rows(patterns: Iterable[str]) -> List[Dict[str, str]]:
    paths: List[str] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            paths.extend(matches)
        elif os.path.exists(pat):
            paths.append(pat)
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize usage and estimate spend from runner CSVs")
    p.add_argument(
        "csv",
        nargs="+",
        help="One or more CSV paths or globs (example: program_synthesis/runs*/llm_*.csv)",
    )
    p.add_argument(
        "--input-rate",
        type=float,
        default=None,
        help="Override USD per 1M input tokens for all rows.",
    )
    p.add_argument(
        "--output-rate",
        type=float,
        default=None,
        help="Override USD per 1M output tokens for all rows.",
    )
    args = p.parse_args()

    rows = _load_rows(args.csv)
    if not rows:
        raise SystemExit("No CSV rows found for the provided path(s).")

    attempt_rows = [
        r for r in rows
        if (r.get("attempt") not in (None, "")) and (str(r.get("is_summary", "")).lower() not in {"1", "true", "yes"})
    ]

    prompt_tokens = sum(_to_int(r.get("prompt_tokens")) for r in attempt_rows)
    completion_tokens = sum(_to_int(r.get("completion_tokens")) for r in attempt_rows)
    reasoning_tokens = sum(_to_int(r.get("reasoning_tokens")) for r in attempt_rows)
    cached_tokens = sum(_to_int(r.get("cached_tokens")) for r in attempt_rows)
    input_cost = 0.0
    output_cost = 0.0
    total_cost = 0.0
    unresolved_models: set[str] = set()
    priced_rows = 0
    for row in attempt_rows:
        usage = {
            "prompt_tokens": _to_int(row.get("prompt_tokens")),
            "completion_tokens": _to_int(row.get("completion_tokens")),
            "reasoning_tokens": _to_int(row.get("reasoning_tokens")),
            "cached_tokens": _to_int(row.get("cached_tokens")),
        }
        cost = estimate_usage_cost(
            usage,
            row.get("returned_model") or row.get("model"),
            input_rate=args.input_rate,
            output_rate=args.output_rate,
        )
        if cost.get("estimated_total_cost_usd") is None:
            unresolved_models.add(str(row.get("returned_model") or row.get("model") or "<missing-model>"))
            continue
        priced_rows += 1
        input_cost += float(cost.get("estimated_input_cost_usd") or 0.0)
        output_cost += float(cost.get("estimated_output_cost_usd") or 0.0)
        total_cost += float(cost.get("estimated_total_cost_usd") or 0.0)

    print(f"rows_total={len(rows)}")
    print(f"rows_attempt={len(attempt_rows)}")
    print(f"rows_priced={priced_rows}")
    print(f"prompt_tokens={prompt_tokens}")
    print(f"completion_tokens={completion_tokens}")
    print(f"reasoning_tokens={reasoning_tokens}")
    print(f"cached_tokens={cached_tokens}")
    print(f"est_input_cost_usd={input_cost:.6f}")
    print(f"est_output_cost_usd={output_cost:.6f}")
    print(f"est_total_cost_usd={total_cost:.6f}")
    if args.input_rate is None and args.output_rate is None:
        known_models = sorted({resolve_model_pricing(r.get("returned_model") or r.get("model")).canonical_name for r in attempt_rows if resolve_model_pricing(r.get("returned_model") or r.get("model"))})
        if known_models:
            print(f"priced_models={','.join(known_models)}")
    if unresolved_models:
        print(f"unpriced_models={','.join(sorted(unresolved_models))}")


if __name__ == "__main__":
    main()

