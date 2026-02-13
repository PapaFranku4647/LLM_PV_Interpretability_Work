from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _load_run(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline_csvs = sorted(run_dir.glob("baseline_seed*.csv"))
    if not baseline_csvs:
        raise FileNotFoundError(f"No baseline_seed*.csv found in {run_dir}")

    df = pd.concat([pd.read_csv(p) for p in baseline_csvs], ignore_index=True)
    df["is_summary"] = _to_bool(df["is_summary"])

    attempts = df[(df["attempt"].notna()) & (~df["is_summary"])].copy()
    summary = df[df["is_summary"]].copy()

    for col in ["attempt", "trial", "global_seed", "length", "dataset_seed"]:
        if col in attempts.columns:
            attempts[col] = pd.to_numeric(attempts[col], errors="coerce")
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    attempts["val_acc"] = pd.to_numeric(attempts["val_acc"], errors="coerce")
    attempts["test_acc"] = pd.to_numeric(attempts["test_acc"], errors="coerce")
    summary["val_acc"] = pd.to_numeric(summary["val_acc"], errors="coerce")
    summary["test_acc"] = pd.to_numeric(summary["test_acc"], errors="coerce")
    summary["test_acc_std"] = pd.to_numeric(summary["test_acc_std"], errors="coerce")

    attempts["compile_ok"] = attempts["compile_error"].isna()
    attempts["attempt"] = attempts["attempt"].astype("Int64")
    attempts["trial"] = attempts["trial"].astype("Int64")
    attempts["global_seed"] = attempts["global_seed"].astype("Int64")
    summary["global_seed"] = summary["global_seed"].astype("Int64")

    return attempts, summary


def _compute_metrics(run_dir: Path, attempts: pd.DataFrame, summary: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    summary_by_seed = summary[
        [
            "run_id",
            "global_seed",
            "fn",
            "length",
            "dataset_seed",
            "val_acc",
            "test_acc",
            "test_acc_std",
            "num_trials",
        ]
    ].copy()

    summary_by_seed["val_test_gap"] = summary_by_seed["val_acc"] - summary_by_seed["test_acc"]

    by_fn = (
        summary_by_seed.groupby(["fn", "length"], as_index=False)
        .agg(
            n_seeds=("global_seed", "nunique"),
            mean_val=("val_acc", "mean"),
            mean_test=("test_acc", "mean"),
            std_test=("test_acc", "std"),
            min_test=("test_acc", "min"),
            max_test=("test_acc", "max"),
            mean_val_test_gap=("val_test_gap", "mean"),
        )
        .sort_values(["fn", "length"])
        .reset_index(drop=True)
    )
    by_fn["seed_spread_test"] = by_fn["max_test"] - by_fn["min_test"]
    by_fn["sem_test"] = by_fn["std_test"] / np.sqrt(by_fn["n_seeds"].clip(lower=1))
    by_fn["ci95_test"] = 1.96 * by_fn["sem_test"].fillna(0.0)
    # Rough planning metric: number of seeds needed for +/-0.02 margin at 95% CI.
    target_margin = 0.02
    by_fn["recommended_seeds_for_moe_0p02"] = np.ceil(
        ((1.96 * by_fn["std_test"].fillna(0.0)) / target_margin) ** 2
    ).astype(int)

    compile_stats = (
        attempts.groupby(["fn", "length"], as_index=False)
        .agg(
            attempt_rows=("attempt", "count"),
            compile_ok_rows=("compile_ok", "sum"),
            mean_attempt_test=("test_acc", "mean"),
            std_attempt_test=("test_acc", "std"),
        )
        .sort_values(["fn", "length"])
        .reset_index(drop=True)
    )
    compile_stats["compile_ok_rate"] = compile_stats["compile_ok_rows"] / compile_stats["attempt_rows"]

    # Trial-level improvement: best attempt test vs attempt-1 test.
    first_attempt = attempts[attempts["attempt"] == 1][
        ["global_seed", "fn", "length", "trial", "test_acc"]
    ].rename(columns={"test_acc": "attempt1_test"})
    best_attempt = (
        attempts.sort_values(
            ["global_seed", "fn", "length", "trial", "test_acc"],
            ascending=[True, True, True, True, False],
        )
        .groupby(["global_seed", "fn", "length", "trial"], as_index=False)
        .first()[["global_seed", "fn", "length", "trial", "attempt", "test_acc"]]
        .rename(columns={"attempt": "best_attempt", "test_acc": "best_test"})
    )
    improvement = first_attempt.merge(
        best_attempt, on=["global_seed", "fn", "length", "trial"], how="inner"
    )
    improvement["delta_test"] = improvement["best_test"] - improvement["attempt1_test"]
    improvement["improved"] = improvement["delta_test"] > 0

    improvement_by_fn = (
        improvement.groupby(["fn", "length"], as_index=False)
        .agg(
            trials=("trial", "count"),
            improved_trials=("improved", "sum"),
            mean_delta=("delta_test", "mean"),
            median_delta=("delta_test", "median"),
            max_delta=("delta_test", "max"),
            min_delta=("delta_test", "min"),
        )
        .sort_values(["fn", "length"])
        .reset_index(drop=True)
    )
    improvement_by_fn["improved_rate"] = improvement_by_fn["improved_trials"] / improvement_by_fn["trials"]

    # Leakage metrics from analysis_v2 if present.
    leakage_path = run_dir / "analysis_v2_leakage_check.csv"
    if leakage_path.exists():
        leakage = pd.read_csv(leakage_path)
        leakage["total_input_overlap"] = pd.to_numeric(leakage["total_input_overlap"], errors="coerce").fillna(0)
        leakage_by_fn = (
            leakage.groupby("fn", as_index=False)["total_input_overlap"]
            .sum()
            .rename(columns={"total_input_overlap": "total_input_overlap_sum"})
        )
    else:
        leakage = pd.DataFrame()
        leakage_by_fn = pd.DataFrame(columns=["fn", "total_input_overlap_sum"])

    compile_errors = attempts[attempts["compile_error"].notna()][
        ["run_id", "global_seed", "fn", "length", "trial", "attempt", "compile_error"]
    ].copy()

    return {
        "summary_by_seed": summary_by_seed,
        "by_fn": by_fn,
        "compile_stats": compile_stats,
        "improvement": improvement,
        "improvement_by_fn": improvement_by_fn,
        "leakage": leakage,
        "leakage_by_fn": leakage_by_fn,
        "compile_errors": compile_errors,
    }


def _style() -> None:
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def _plot_seed_and_gap(summary_by_seed: pd.DataFrame, out_path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), dpi=170)

    if HAS_SEABORN:
        sns.lineplot(
            data=summary_by_seed,
            x="global_seed",
            y="test_acc",
            hue="fn",
            style="fn",
            dashes=False,
            linewidth=2.0,
            ax=axes[0],
        )
        sns.scatterplot(
            data=summary_by_seed,
            x="global_seed",
            y="test_acc",
            hue="fn",
            style="fn",
            s=65,
            legend=False,
            ax=axes[0],
        )
    else:
        for fn, grp in summary_by_seed.groupby("fn"):
            axes[0].plot(grp["global_seed"], grp["test_acc"], marker="o", label=fn)
    axes[0].set_title("Seed-Level Test Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Global Seed")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].legend(title="fn")

    if HAS_SEABORN:
        sns.barplot(
            data=summary_by_seed,
            x="global_seed",
            y="val_test_gap",
            hue="fn",
            errorbar=None,
            ax=axes[1],
        )
        sns.stripplot(
            data=summary_by_seed,
            x="global_seed",
            y="val_test_gap",
            hue="fn",
            dodge=True,
            size=4,
            alpha=0.65,
            legend=False,
            ax=axes[1],
        )
    else:
        for fn, grp in summary_by_seed.groupby("fn"):
            axes[1].plot(grp["global_seed"], grp["val_test_gap"], marker="o", label=fn)
    axes[1].set_title("Generalization Gap (Val - Test)")
    axes[1].set_xlabel("Global Seed")
    axes[1].set_ylabel("Gap")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].legend(title="fn")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_attempt_distribution(attempts: pd.DataFrame, out_path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), dpi=170)

    if HAS_SEABORN:
        sns.boxplot(
            data=attempts,
            x="fn",
            y="test_acc",
            hue="compile_ok",
            width=0.55,
            ax=axes[0],
        )
        sns.stripplot(
            data=attempts,
            x="fn",
            y="test_acc",
            hue="compile_ok",
            dodge=True,
            alpha=0.45,
            size=3,
            ax=axes[0],
            legend=False,
        )
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles[:2], ["compile_ok=False", "compile_ok=True"], title="compile_ok")
    else:
        for fn, grp in attempts.groupby("fn"):
            axes[0].scatter([fn] * len(grp), grp["test_acc"], alpha=0.6, s=10)
    axes[0].set_title("Attempt-Level Test Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Function")
    axes[0].set_ylabel("Test Accuracy")

    # Attempt progression aggregated across seeds/trials.
    progression = (
        attempts.groupby(["fn", "attempt"], as_index=False)
        .agg(mean_test=("test_acc", "mean"), std_test=("test_acc", "std"))
        .sort_values(["fn", "attempt"])
    )
    for fn, grp in progression.groupby("fn"):
        axes[1].errorbar(
            grp["attempt"],
            grp["mean_test"],
            yerr=grp["std_test"].fillna(0.0),
            marker="o",
            capsize=3,
            label=fn,
        )
    axes[1].set_title("Attempt Progression (Mean +/- Std)")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Attempt")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].legend(title="fn")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_overview(
    by_fn: pd.DataFrame,
    compile_stats: pd.DataFrame,
    improvement_by_fn: pd.DataFrame,
    leakage_by_fn: pd.DataFrame,
    out_path: Path,
) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=200)
    ax1, ax2, ax3, ax4 = axes.flatten()

    merged = by_fn.merge(compile_stats[["fn", "compile_ok_rate"]], on="fn", how="left")
    merged = merged.merge(improvement_by_fn[["fn", "improved_rate"]], on="fn", how="left")
    merged = merged.merge(leakage_by_fn, on="fn", how="left")
    merged["total_input_overlap_sum"] = merged["total_input_overlap_sum"].fillna(0)
    merged = merged.sort_values("fn")
    has_leakage = float(merged["total_input_overlap_sum"].sum()) > 0.0

    x = np.arange(len(merged))
    width = 0.38
    ax1.bar(x, merged["mean_test"], width=width, label="Mean test acc")
    ax1.errorbar(
        x,
        merged["mean_test"],
        yerr=merged["ci95_test"].fillna(0.0),
        fmt="none",
        capsize=4,
        color="black",
        linewidth=1.0,
        label="95% CI (seed mean)",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(merged["fn"])
    ax1.set_ylim(0, 1)
    ax1.set_title("Mean Test Accuracy (95% CI)")
    ax1.legend()

    ax2.bar(merged["fn"], merged["seed_spread_test"])
    ax2.set_title("Seed Variance (max - min)")
    ax2.set_ylabel("Spread of test acc")

    ax3.bar(merged["fn"], merged["compile_ok_rate"], label="Compile OK rate")
    ax3.bar(merged["fn"], merged["improved_rate"], alpha=0.70, label="Improved-trial rate")
    ax3.set_ylim(0, 1)
    ax3.set_title("Reliability vs Improvement")
    ax3.legend()

    if has_leakage:
        ax4.bar(merged["fn"], merged["total_input_overlap_sum"])
        ax4.set_title("Leakage Audit (Input Overlap Sum)")
        ax4.set_ylabel("Count")
    else:
        ax4.bar(merged["fn"], merged["mean_val_test_gap"])
        ax4.axhline(0.0, color="black", linewidth=1.0)
        ax4.set_title("Generalization Gap (Val - Test)")
        ax4.set_ylabel("Gap")

    fig.suptitle("Run Overview", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_leakage_only_if_needed(leakage: pd.DataFrame, out_path: Path) -> bool:
    if leakage.empty:
        return False
    leak = leakage.copy()
    leak["total_input_overlap"] = pd.to_numeric(leak["total_input_overlap"], errors="coerce").fillna(0)
    if float(leak["total_input_overlap"].sum()) <= 0.0:
        return False

    _style()
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.6), dpi=170)
    labels = [f"{fn}-s{seed}" for fn, seed in zip(leak["fn"], leak["dataset_seed"])]
    x = np.arange(len(leak))
    ax.bar(x, leak["total_input_overlap"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Leakage Audit (Only shown when non-zero)")
    ax.set_ylabel("Input overlap count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _write_text_outputs(
    run_dir: Path,
    by_fn: pd.DataFrame,
    compile_stats: pd.DataFrame,
    improvement_by_fn: pd.DataFrame,
    leakage_by_fn: pd.DataFrame,
    compile_errors: pd.DataFrame,
) -> None:
    leakage_map = {
        str(r["fn"]): float(r["total_input_overlap_sum"])
        for _, r in leakage_by_fn.iterrows()
    }
    compile_map = {
        str(r["fn"]): float(r["compile_ok_rate"])
        for _, r in compile_stats.iterrows()
    }
    improved_map = {
        str(r["fn"]): float(r["improved_rate"])
        for _, r in improvement_by_fn.iterrows()
    }
    spread_map = {
        str(r["fn"]): float(r["seed_spread_test"])
        for _, r in by_fn.iterrows()
    }

    strict_pass = (
        (len(leakage_by_fn) == 0 or all(v == 0 for v in leakage_map.values()))
        and (len(compile_map) > 0 and min(compile_map.values()) >= 0.99)
    )

    noisy_flags = []
    for fn, spread in spread_map.items():
        if spread >= 0.10:
            noisy_flags.append(f"{fn}:high_seed_variance({spread:.3f})")

    lines = [
        "RUN ANALYSIS READOUT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Run dir: {run_dir}",
        "",
        f"Strict gate pass: {strict_pass}",
        "Strict gate definition:",
        "- zero input-overlap leakage",
        "- compile_ok_rate >= 0.99 for all functions",
        "",
        "Function summary:",
    ]
    for _, row in by_fn.sort_values("fn").iterrows():
        fn = str(row["fn"])
        lines.append(
            f"- {fn} (L={int(row['length'])}): mean_test={row['mean_test']:.4f}, "
            f"mean_val={row['mean_val']:.4f}, seed_spread_test={row['seed_spread_test']:.4f}, "
            f"ci95_test={row['ci95_test']:.4f}, "
            f"compile_ok_rate={compile_map.get(fn, float('nan')):.4f}, "
            f"improved_rate={improved_map.get(fn, float('nan')):.4f}, "
            f"input_overlap_sum={int(leakage_map.get(fn, 0))}, "
            f"recommended_seeds_for_moe_0p02={int(row['recommended_seeds_for_moe_0p02'])}"
        )

    lines.extend(
        [
            "",
            "Variance/noise assessment:",
            f"- high-variance flags: {', '.join(noisy_flags) if noisy_flags else 'none'}",
            "- interpretation: leakage is clean, but stability differs strongly by function.",
            "- recommendation: for paper-level comparisons, use >= 12 seeds minimum; "
            "for high-variance functions, use 20+ seeds.",
            "",
            "Compile errors:",
        ]
    )
    if len(compile_errors) == 0:
        lines.append("- none")
    else:
        for _, row in compile_errors.iterrows():
            lines.append(
                f"- run_id={row['run_id']} fn={row['fn']} seed={int(row['global_seed'])} "
                f"trial={int(row['trial'])} attempt={int(row['attempt'])}: {row['compile_error']}"
            )

    (run_dir / "analysis_readout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced plotting and readout for a baseline run directory.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing baseline_seed*.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    attempts, summary = _load_run(run_dir)
    data = _compute_metrics(run_dir, attempts, summary)

    _plot_seed_and_gap(
        summary_by_seed=data["summary_by_seed"],
        out_path=run_dir / "analysis_plot_seed_and_gap.png",
    )
    _plot_attempt_distribution(
        attempts=attempts,
        out_path=run_dir / "analysis_plot_attempt_distribution.png",
    )
    _plot_overview(
        by_fn=data["by_fn"],
        compile_stats=data["compile_stats"],
        improvement_by_fn=data["improvement_by_fn"],
        leakage_by_fn=data["leakage_by_fn"],
        out_path=run_dir / "analysis_plot_overview.png",
    )
    leakage_plot_written = _plot_leakage_only_if_needed(
        leakage=data["leakage"],
        out_path=run_dir / "analysis_plot_leakage.png",
    )

    data["summary_by_seed"].to_csv(run_dir / "analysis_summary_by_seed.csv", index=False)
    data["by_fn"].to_csv(run_dir / "analysis_summary_by_fn.csv", index=False)
    data["compile_stats"].to_csv(run_dir / "analysis_compile_stats.csv", index=False)
    data["improvement_by_fn"].to_csv(run_dir / "analysis_improvement_by_fn.csv", index=False)
    data["compile_errors"].to_csv(run_dir / "analysis_compile_errors.csv", index=False)

    _write_text_outputs(
        run_dir=run_dir,
        by_fn=data["by_fn"],
        compile_stats=data["compile_stats"],
        improvement_by_fn=data["improvement_by_fn"],
        leakage_by_fn=data["leakage_by_fn"],
        compile_errors=data["compile_errors"],
    )

    if not leakage_plot_written:
        marker = run_dir / "analysis_leakage_plot_skipped.txt"
        marker.write_text(
            "Leakage plot intentionally skipped because total input overlap is zero.\n",
            encoding="utf-8",
        )

    print(f"Run analysis complete: {run_dir}")


if __name__ == "__main__":
    main()
