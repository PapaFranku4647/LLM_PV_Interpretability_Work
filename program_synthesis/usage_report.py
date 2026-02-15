#!/usr/bin/env python
"""Summarize token usage and estimated API cost from runner CSV outputs."""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, Iterable, List


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
        default=1.75,
        help="USD per 1M input tokens (default: GPT-5.2 price 1.75)",
    )
    p.add_argument(
        "--output-rate",
        type=float,
        default=14.0,
        help="USD per 1M output tokens (default: GPT-5.2 price 14.0)",
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

    input_cost = (prompt_tokens / 1_000_000.0) * args.input_rate
    output_cost = (completion_tokens / 1_000_000.0) * args.output_rate
    total_cost = input_cost + output_cost

    print(f"rows_total={len(rows)}")
    print(f"rows_attempt={len(attempt_rows)}")
    print(f"prompt_tokens={prompt_tokens}")
    print(f"completion_tokens={completion_tokens}")
    print(f"reasoning_tokens={reasoning_tokens}")
    print(f"cached_tokens={cached_tokens}")
    print(f"est_input_cost_usd={input_cost:.6f}")
    print(f"est_output_cost_usd={output_cost:.6f}")
    print(f"est_total_cost_usd={total_cost:.6f}")


if __name__ == "__main__":
    main()

