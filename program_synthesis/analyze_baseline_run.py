from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class LeakageRow:
    fn: str
    length: int
    dataset_seed: int
    dataset_root: str
    train_n: int
    val_n: int
    test_n: int
    overlap_line_train_val: int
    overlap_line_train_test: int
    overlap_line_val_test: int
    overlap_input_train_val: int
    overlap_input_train_test: int
    overlap_input_val_test: int
    duplicate_inputs_train: int
    duplicate_inputs_val: int
    duplicate_inputs_test: int


def _read_lines(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _canonical_input_from_line(line: str) -> str:
    if "->" not in line:
        return line.strip()
    return line.split("->", 1)[0].strip()


def _dataset_name_from_fn(fn: str) -> str:
    mapping = {"fn_p": "htru2", "fn_o": "cdc_diabetes"}
    if fn not in mapping:
        raise ValueError(f"Unsupported function id for leakage audit: {fn}")
    return mapping[fn]


def _default_dataset_root(run_dir: Path) -> Path:
    try:
        return run_dir.parents[1] / "datasets"
    except IndexError:
        return Path("program_synthesis") / "datasets"


def _resolve_dataset_root(dataset_dir_value: Any, run_dir: Path) -> Path:
    if dataset_dir_value is None:
        return _default_dataset_root(run_dir)
    if isinstance(dataset_dir_value, float) and pd.isna(dataset_dir_value):
        return _default_dataset_root(run_dir)

    text = str(dataset_dir_value).strip()
    if not text:
        return _default_dataset_root(run_dir)

    path = Path(text)
    if path.is_absolute():
        return path

    # Try likely anchors for relative dataset_dir values.
    candidates = [
        Path.cwd() / path,
        run_dir / path,
        run_dir.parent / path,
        _default_dataset_root(run_dir).parents[1] / path if len(_default_dataset_root(run_dir).parents) > 1 else None,
    ]
    for cand in candidates:
        if cand is not None and cand.exists():
            return cand

    # Fall back to cwd-relative path for clear downstream FileNotFoundError.
    return Path.cwd() / path


def _compute_leakage(dataset_root: Path, fn: str, length: int, dataset_seed: int) -> LeakageRow:
    dataset_name = _dataset_name_from_fn(fn)
    base = dataset_root / dataset_name / f"L{length}" / f"seed{dataset_seed}"
    train_lines = _read_lines(base / "train.txt")
    val_lines = _read_lines(base / "val.txt")
    test_lines = _read_lines(base / "test.txt")

    train_line_set, val_line_set, test_line_set = set(train_lines), set(val_lines), set(test_lines)
    train_inputs = [_canonical_input_from_line(x) for x in train_lines]
    val_inputs = [_canonical_input_from_line(x) for x in val_lines]
    test_inputs = [_canonical_input_from_line(x) for x in test_lines]
    train_input_set, val_input_set, test_input_set = set(train_inputs), set(val_inputs), set(test_inputs)

    return LeakageRow(
        fn=fn,
        length=length,
        dataset_seed=dataset_seed,
        dataset_root=str(dataset_root),
        train_n=len(train_lines),
        val_n=len(val_lines),
        test_n=len(test_lines),
        overlap_line_train_val=len(train_line_set & val_line_set),
        overlap_line_train_test=len(train_line_set & test_line_set),
        overlap_line_val_test=len(val_line_set & test_line_set),
        overlap_input_train_val=len(train_input_set & val_input_set),
        overlap_input_train_test=len(train_input_set & test_input_set),
        overlap_input_val_test=len(val_input_set & test_input_set),
        duplicate_inputs_train=len(train_inputs) - len(train_input_set),
        duplicate_inputs_val=len(val_inputs) - len(val_input_set),
        duplicate_inputs_test=len(test_inputs) - len(test_input_set),
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _plot_seed_test_acc(summary_by_seed: pd.DataFrame, out_path: Path) -> None:
    pivot = summary_by_seed.pivot_table(index="global_seed", columns="fn", values="test_acc")
    seeds = list(pivot.index.astype(int))
    fns = list(pivot.columns)
    x = np.arange(len(seeds))
    width = 0.35 if len(fns) <= 2 else 0.8 / max(1, len(fns))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    for i, fn in enumerate(fns):
        vals = pivot[fn].values
        ax.bar(x + (i - (len(fns) - 1) / 2) * width, vals, width=width, label=fn)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Global Seed")
    ax.set_title("Test Accuracy by Seed and Function")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_attempt_progression(attempt_rows: pd.DataFrame, out_path: Path) -> None:
    stats = (
        attempt_rows.groupby(["fn", "attempt"], as_index=False)
        .agg(mean_test=("test_acc", "mean"), std_test=("test_acc", "std"), n=("test_acc", "count"))
        .sort_values(["fn", "attempt"])
    )
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    for fn, grp in stats.groupby("fn"):
        ax.errorbar(
            grp["attempt"],
            grp["mean_test"],
            yerr=grp["std_test"].fillna(0.0),
            marker="o",
            capsize=3,
            label=fn,
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Attempt Index")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Attempt Progression (Mean ± Std)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_clean_vs_raw(raw_combined: pd.DataFrame, clean_combined: pd.DataFrame, out_path: Path) -> None:
    raw = raw_combined.set_index("fn")
    clean = clean_combined.set_index("fn")
    fns = sorted(set(raw.index) | set(clean.index))
    x = np.arange(len(fns))
    raw_vals = [float(raw.loc[fn, "mean_test_acc"]) if fn in raw.index else np.nan for fn in fns]
    clean_vals = [float(clean.loc[fn, "mean_test_acc"]) if fn in clean.index else np.nan for fn in fns]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    width = 0.35
    ax.bar(x - width / 2, raw_vals, width=width, label="Raw mean test acc")
    ax.bar(x + width / 2, clean_vals, width=width, label="Leakage-clean mean test acc")
    ax.set_xticks(x)
    ax.set_xticklabels(fns)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Raw vs Leakage-Clean Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_leakage_counts(leakage: pd.DataFrame, out_path: Path) -> None:
    leak = leakage.copy()
    leak["total_input_overlap"] = (
        leak["overlap_input_train_val"] + leak["overlap_input_train_test"] + leak["overlap_input_val_test"]
    )
    labels = [f"{fn}-s{seed}" for fn, seed in zip(leak["fn"], leak["dataset_seed"])]
    x = np.arange(len(leak))

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    ax.bar(x, leak["total_input_overlap"], label="Total input overlap across split pairs")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Overlap Count")
    ax.set_title("Leakage Audit (Input Overlap)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_analysis(run_dir: Path) -> Dict[str, Any]:
    baseline_csvs = sorted(run_dir.glob("baseline_seed*.csv"))
    if not baseline_csvs:
        raise FileNotFoundError(f"No baseline_seed*.csv files found in {run_dir}")

    df_all = pd.concat([pd.read_csv(p) for p in baseline_csvs], ignore_index=True)
    summary = df_all[df_all["is_summary"] == True].copy()
    attempts = df_all[(df_all["attempt"].notna()) & (df_all["is_summary"] != True)].copy()
    attempts["test_acc"] = pd.to_numeric(attempts["test_acc"], errors="coerce")
    attempts["val_acc"] = pd.to_numeric(attempts["val_acc"], errors="coerce")
    attempts["attempt"] = attempts["attempt"].astype(int)
    attempts["compile_ok"] = attempts["compile_error"].isna()

    summary_by_seed = summary[
        [
            "run_id",
            "global_seed",
            "fn",
            "length",
            "dataset_seed",
            "val_acc",
            "val_acc_std",
            "test_acc",
            "test_acc_std",
            "num_trials",
        ]
    ].copy()
    summary_by_seed = summary_by_seed.sort_values(["fn", "global_seed"]).reset_index(drop=True)

    combined_raw = (
        summary_by_seed.groupby(["fn", "length"], as_index=False)
        .agg(
            mean_val_acc=("val_acc", "mean"),
            mean_test_acc=("test_acc", "mean"),
            min_test_acc=("test_acc", "min"),
            max_test_acc=("test_acc", "max"),
            n_seeds=("global_seed", "nunique"),
        )
        .sort_values(["fn", "length"])
        .reset_index(drop=True)
    )
    combined_raw["test_seed_spread"] = combined_raw["max_test_acc"] - combined_raw["min_test_acc"]

    compile_stats = (
        attempts.groupby(["fn", "length"], as_index=False)
        .agg(
            attempt_rows=("attempt", "count"),
            compile_ok_rows=("compile_ok", "sum"),
            mean_prompt_tokens=("prompt_tokens", "mean"),
            mean_completion_tokens=("completion_tokens", "mean"),
            mean_duration_ms=("duration_ms", "mean"),
            mean_test_acc=("test_acc", "mean"),
        )
        .sort_values(["fn", "length"])
        .reset_index(drop=True)
    )
    compile_stats["compile_ok_rate"] = compile_stats["compile_ok_rows"] / compile_stats["attempt_rows"]

    # Leakage audit only for dataset seeds actually used in attempts.
    # Resolve dataset root from row metadata when available (supports custom --dataset-dir).
    if "dataset_dir" not in attempts.columns:
        attempts = attempts.copy()
        attempts["dataset_dir"] = str(_default_dataset_root(run_dir))

    used_ds = (
        attempts[["fn", "length", "dataset_seed", "dataset_dir"]]
        .dropna(subset=["fn", "length", "dataset_seed"])
        .drop_duplicates()
        .sort_values(["fn", "dataset_seed"])
    )
    leak_rows = []
    for rec in used_ds.itertuples(index=False):
        dataset_root = _resolve_dataset_root(rec.dataset_dir, run_dir)
        leak_rows.append(
            _compute_leakage(
                dataset_root=dataset_root,
                fn=str(rec.fn),
                length=int(rec.length),
                dataset_seed=int(rec.dataset_seed),
            ).__dict__
        )
    leakage = pd.DataFrame(leak_rows).sort_values(["fn", "dataset_seed"]).reset_index(drop=True)
    leakage["total_input_overlap"] = (
        leakage["overlap_input_train_val"] + leakage["overlap_input_train_test"] + leakage["overlap_input_val_test"]
    )
    leakage["has_input_overlap"] = leakage["total_input_overlap"] > 0

    with_flags = summary_by_seed.merge(
        leakage[["fn", "length", "dataset_seed", "has_input_overlap", "total_input_overlap"]],
        on=["fn", "length", "dataset_seed"],
        how="left",
    )
    with_flags["has_input_overlap"] = with_flags["has_input_overlap"].fillna(False)
    clean_summary = with_flags[~with_flags["has_input_overlap"]].copy()
    combined_clean = (
        clean_summary.groupby(["fn", "length"], as_index=False)
        .agg(
            n_clean_seeds=("global_seed", "nunique"),
            mean_val_acc=("val_acc", "mean"),
            mean_test_acc=("test_acc", "mean"),
        )
        .sort_values(["fn", "length"])
        .reset_index(drop=True)
    )

    # Improvement: best test in trial versus attempt1
    first_attempt = attempts[attempts["attempt"] == 1][["global_seed", "fn", "length", "trial", "test_acc"]].rename(
        columns={"test_acc": "attempt1_test_acc"}
    )
    best_attempt = (
        attempts.sort_values(
            ["global_seed", "fn", "length", "trial", "test_acc"],
            ascending=[True, True, True, True, False],
        )
        .groupby(["global_seed", "fn", "length", "trial"], as_index=False)
        .first()[["global_seed", "fn", "length", "trial", "attempt", "test_acc"]]
        .rename(columns={"attempt": "best_attempt", "test_acc": "best_test_acc"})
    )
    improvement = first_attempt.merge(best_attempt, on=["global_seed", "fn", "length", "trial"], how="inner")
    improvement["delta_test_vs_attempt1"] = improvement["best_test_acc"] - improvement["attempt1_test_acc"]
    improvement["improved"] = improvement["delta_test_vs_attempt1"] > 0
    improvement_agg = (
        improvement.groupby(["fn", "length"], as_index=False)
        .agg(
            trials=("trial", "count"),
            improved_trials=("improved", "sum"),
            mean_delta_test=("delta_test_vs_attempt1", "mean"),
            median_delta_test=("delta_test_vs_attempt1", "median"),
            max_delta_test=("delta_test_vs_attempt1", "max"),
            min_delta_test=("delta_test_vs_attempt1", "min"),
        )
        .sort_values(["fn", "length"])
        .reset_index(drop=True)
    )
    improvement_agg["improved_trial_rate"] = improvement_agg["improved_trials"] / improvement_agg["trials"]

    research_standard_ok = bool(
        (leakage["total_input_overlap"].sum() == 0)
        and (compile_stats["compile_ok_rate"].min() >= 0.99)
    )

    return {
        "summary_by_seed": summary_by_seed,
        "combined_raw": combined_raw,
        "compile_stats": compile_stats,
        "leakage": leakage,
        "summary_with_flags": with_flags,
        "combined_clean": combined_clean,
        "improvement": improvement,
        "improvement_agg": improvement_agg,
        "research_standard_ok": research_standard_ok,
    }


def write_outputs(run_dir: Path, data: Dict[str, Any]) -> None:
    prefix = "analysis_v2"
    _save_df(data["summary_by_seed"], run_dir / f"{prefix}_summary_by_seed_fn.csv")
    _save_df(data["combined_raw"], run_dir / f"{prefix}_summary_combined_raw.csv")
    _save_df(data["compile_stats"], run_dir / f"{prefix}_compile_stats.csv")
    _save_df(data["leakage"], run_dir / f"{prefix}_leakage_check.csv")
    _save_df(data["summary_with_flags"], run_dir / f"{prefix}_summary_with_leak_flags.csv")
    _save_df(data["combined_clean"], run_dir / f"{prefix}_summary_combined_clean.csv")
    _save_df(data["improvement"], run_dir / f"{prefix}_improvement_by_trial.csv")
    _save_df(data["improvement_agg"], run_dir / f"{prefix}_improvement_agg.csv")

    plot_seed = run_dir / f"{prefix}_plot_seed_test_acc.png"
    plot_attempt = run_dir / f"{prefix}_plot_attempt_progression.png"
    plot_raw_clean = run_dir / f"{prefix}_plot_raw_vs_clean.png"
    plot_leak = run_dir / f"{prefix}_plot_leakage.png"

    _plot_seed_test_acc(data["summary_by_seed"], plot_seed)
    # Attempt progression should use the full attempt-level rows from raw csvs.
    baseline_csvs = sorted(run_dir.glob("baseline_seed*.csv"))
    attempts = pd.concat([pd.read_csv(p) for p in baseline_csvs], ignore_index=True)
    attempts = attempts[(attempts["attempt"].notna()) & (attempts["is_summary"] != True)].copy()
    attempts["attempt"] = attempts["attempt"].astype(int)
    attempts["test_acc"] = pd.to_numeric(attempts["test_acc"], errors="coerce")
    _plot_attempt_progression(attempts, plot_attempt)

    _plot_clean_vs_raw(data["combined_raw"], data["combined_clean"], plot_raw_clean)
    _plot_leakage_counts(data["leakage"], plot_leak)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "BASELINE RUN ANALYSIS (v2)",
        f"Generated: {ts}",
        f"Run dir: {run_dir}",
        "",
        f"Research standard gate (strict): {data['research_standard_ok']}",
        "Gate definition: zero input-overlap leakage across all dataset seeds and compile_ok_rate >= 0.99.",
        "",
        "Combined raw results:",
    ]
    for _, row in data["combined_raw"].iterrows():
        report_lines.append(
            f"- {row['fn']} L={int(row['length'])}: mean_test={row['mean_test_acc']:.4f}, "
            f"mean_val={row['mean_val_acc']:.4f}, test_seed_spread={row['test_seed_spread']:.4f}"
        )
    report_lines.append("")
    report_lines.append("Combined leakage-clean results:")
    for _, row in data["combined_clean"].iterrows():
        report_lines.append(
            f"- {row['fn']} L={int(row['length'])}: mean_test={row['mean_test_acc']:.4f}, "
            f"mean_val={row['mean_val_acc']:.4f}, clean_seeds={int(row['n_clean_seeds'])}"
        )
    report_lines.append("")
    report_lines.append("Leakage rows:")
    for _, row in data["leakage"].iterrows():
        report_lines.append(
            f"- {row['fn']} seed={int(row['dataset_seed'])}: total_input_overlap={int(row['total_input_overlap'])} "
            f"(tv={int(row['overlap_input_train_val'])}, tt={int(row['overlap_input_train_test'])}, vt={int(row['overlap_input_val_test'])})"
        )
    report_lines.append("")
    report_lines.append("Improvement from attempt1 to best in trial:")
    for _, row in data["improvement_agg"].iterrows():
        report_lines.append(
            f"- {row['fn']} L={int(row['length'])}: improved_trials={int(row['improved_trials'])}/{int(row['trials'])}, "
            f"mean_delta_test={row['mean_delta_test']:.4f}"
        )
    report_lines.append("")
    report_lines.append("Graph files:")
    report_lines.append(f"- {plot_seed.name}")
    report_lines.append(f"- {plot_attempt.name}")
    report_lines.append(f"- {plot_raw_clean.name}")
    report_lines.append(f"- {plot_leak.name}")

    (run_dir / f"{prefix}_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Baseline Analysis v2</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; }}
    .badge-ok {{ color: #065f46; background: #d1fae5; padding: 4px 8px; border-radius: 6px; }}
    .badge-bad {{ color: #7f1d1d; background: #fee2e2; padding: 4px 8px; border-radius: 6px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0 22px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; margin-bottom: 22px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #f6f6f6; text-align: left; }}
    code {{ background: #f3f3f3; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Baseline Analysis v2</h1>
  <p>Generated: {ts}</p>
  <p>Run directory: <code>{run_dir}</code></p>
  <p>Research standard gate: {"<span class='badge-ok'>PASS</span>" if data["research_standard_ok"] else "<span class='badge-bad'>FAIL</span>"}</p>
  <p>Gate definition: zero input-overlap leakage and compile_ok_rate >= 0.99.</p>

  <h2>Seed-Level Test Accuracy</h2>
  <img src="{plot_seed.name}" alt="seed test accuracy"/>

  <h2>Attempt Progression</h2>
  <img src="{plot_attempt.name}" alt="attempt progression"/>

  <h2>Raw vs Leakage-Clean Mean Test Accuracy</h2>
  <img src="{plot_raw_clean.name}" alt="raw vs clean"/>

  <h2>Leakage Counts</h2>
  <img src="{plot_leak.name}" alt="leakage"/>

  <h2>Combined Raw Table</h2>
  {data["combined_raw"].to_html(index=False)}

  <h2>Combined Leakage-Clean Table</h2>
  {data["combined_clean"].to_html(index=False)}

  <h2>Leakage Details</h2>
  {data["leakage"].to_html(index=False)}

  <h2>Compile Stats</h2>
  {data["compile_stats"].to_html(index=False)}

  <h2>Improvement Table</h2>
  {data["improvement_agg"].to_html(index=False)}
</body>
</html>"""
    (run_dir / f"{prefix}_results_page.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze baseline run artifacts and generate charts.")
    parser.add_argument("--run-dir", required=True, help="Directory containing baseline_seed*.csv artifacts.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    _ensure_dir(run_dir)

    data = build_analysis(run_dir)
    write_outputs(run_dir, data)
    print(f"Analysis complete: {run_dir}")


if __name__ == "__main__":
    main()
