from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


TRAIN_COLOR = "#d17c00"
TEST_COLOR = "#0f8b8d"
DEFAULT_BATCH_COLORS = [
    "#1d5f8a",
    "#b85c38",
    "#2a9d8f",
    "#687d3a",
    "#c45a5a",
    "#6b7280",
]


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _std(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    mu = _mean(vals)
    return float(math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)))


def _corr(xs: Iterable[float], ys: Iterable[float]) -> float:
    xv = list(xs)
    yv = list(ys)
    if len(xv) < 2 or len(yv) < 2:
        return float("nan")
    x_mu = _mean(xv)
    y_mu = _mean(yv)
    num = sum((x - x_mu) * (y - y_mu) for x, y in zip(xv, yv))
    den_x = math.sqrt(sum((x - x_mu) ** 2 for x in xv))
    den_y = math.sqrt(sum((y - y_mu) ** 2 for y in yv))
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return float(num / (den_x * den_y))


def _sorted_ints(values: Iterable[object]) -> list[int]:
    return sorted({int(v) for v in values})


def _batch_palette(batch_sizes: Iterable[object]) -> dict[int, str]:
    return {
        batch_size: DEFAULT_BATCH_COLORS[idx % len(DEFAULT_BATCH_COLORS)]
        for idx, batch_size in enumerate(_sorted_ints(batch_sizes))
    }


def _format_set(values: Iterable[object]) -> str:
    ints = _sorted_ints(values)
    return "{" + ", ".join(str(v) for v in ints) + "}"


def _parse_run_sizes(run_dir: Path) -> tuple[int | None, int | None]:
    match = re.search(r"train(\d+)_test(\d+)", run_dir.name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _is_monotonic_increasing(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    return all(curr > prev for prev, curr in zip(values, values[1:]))


def summarize_attempt_health(run_dir: Path) -> dict[str, int]:
    total_attempt_rows = 0
    compile_error_rows = 0
    eval_error_rows = 0

    for attempts_path in sorted(run_dir.glob("T*/seed*/attempts.jsonl")):
        frame = pd.read_json(attempts_path, lines=True)
        if frame.empty:
            continue
        total_attempt_rows += int(len(frame))
        if "compile_error" in frame.columns:
            compile_error_rows += int(frame["compile_error"].notna().sum())
        if "eval_errors" in frame.columns:
            eval_error_rows += int(
                pd.to_numeric(frame["eval_errors"], errors="coerce").fillna(0).sum()
            )

    return {
        "total_attempt_rows": total_attempt_rows,
        "compile_error_rows": compile_error_rows,
        "eval_error_rows": eval_error_rows,
    }


def load_seed_rows(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(run_dir.glob("T*/seed*/summary.csv")):
        rel = summary_path.relative_to(run_dir)
        if len(rel.parts) != 3:
            continue
        t_value = int(rel.parts[0].replace("T", ""))
        seed_value = int(rel.parts[1].replace("seed", ""))
        frame = pd.read_csv(summary_path)
        for row in frame.to_dict(orient="records"):
            rows.append(
                {
                    "T": t_value,
                    "seed": seed_value,
                    "batch_size": int(row["batch_size"]),
                    "trial": int(row["trial"]),
                    "final_train_acc": float(row["final_train_acc"]),
                    "final_test_acc": float(row["final_test_acc"]),
                    "accepted_rounds": int(row["accepted_rounds"]),
                    "requested_rounds": int(row["requested_rounds"]),
                    "api_attempt_count": int(row["api_attempt_count"]),
                    "total_prompt_tokens": int(float(row["total_prompt_tokens"] or 0)),
                    "total_completion_tokens": int(float(row["total_completion_tokens"] or 0)),
                    "total_reasoning_tokens": int(float(row["total_reasoning_tokens"] or 0)),
                    "total_estimated_cost_usd": float(row["total_estimated_cost_usd"]),
                    "summary_path": str(summary_path),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No T*/seed*/summary.csv files found under {run_dir}")
    return pd.DataFrame(rows).sort_values(["T", "batch_size", "seed"]).reset_index(drop=True)


def build_aggregate(seed_df: pd.DataFrame) -> pd.DataFrame:
    agg_rows: list[dict[str, object]] = []
    for (t_value, batch_size), group in seed_df.groupby(["T", "batch_size"], sort=True):
        train_vals = group["final_train_acc"].tolist()
        test_vals = group["final_test_acc"].tolist()
        cost_vals = group["total_estimated_cost_usd"].tolist()
        agg_rows.append(
            {
                "T": int(t_value),
                "batch_size": int(batch_size),
                "seeds": int(len(group)),
                "mean_train_acc": _mean(train_vals),
                "std_train_acc": _std(train_vals),
                "mean_test_acc": _mean(test_vals),
                "std_test_acc": _std(test_vals),
                "mean_gap_train_minus_test": _mean([a - b for a, b in zip(train_vals, test_vals)]),
                "train_test_corr_across_seeds": _corr(train_vals, test_vals),
                "mean_cost_usd": _mean(cost_vals),
                "total_cost_usd": float(sum(cost_vals)),
                "mean_api_attempt_count": _mean(group["api_attempt_count"].tolist()),
                "all_rounds_accepted": bool(
                    (group["accepted_rounds"] == group["requested_rounds"]).all()
                ),
                "mean_prompt_tokens": _mean(group["total_prompt_tokens"].tolist()),
                "mean_completion_tokens": _mean(group["total_completion_tokens"].tolist()),
                "mean_reasoning_tokens": _mean(group["total_reasoning_tokens"].tolist()),
            }
        )
    return pd.DataFrame(agg_rows).sort_values(["T", "batch_size"]).reset_index(drop=True)


def build_long_accuracy_df(seed_df: pd.DataFrame) -> pd.DataFrame:
    long_df = seed_df.melt(
        id_vars=["T", "seed", "batch_size", "total_estimated_cost_usd"],
        value_vars=["final_train_acc", "final_test_acc"],
        var_name="metric",
        value_name="accuracy",
    )
    long_df["split"] = long_df["metric"].map(
        {
            "final_train_acc": "Train",
            "final_test_acc": "Test",
        }
    )
    seed_order = {seed: idx for idx, seed in enumerate(sorted(long_df["seed"].unique()))}
    split_offset = {"Train": -0.08, "Test": 0.08}
    seed_offset = {
        idx: offset
        for idx, offset in enumerate(np.linspace(-0.028, 0.028, len(seed_order)))
    }
    long_df["x_plot"] = long_df.apply(
        lambda row: float(row["T"]) + split_offset[str(row["split"])] + seed_offset[seed_order[int(row["seed"])]],
        axis=1,
    )
    return long_df.sort_values(["batch_size", "seed", "split", "T"]).reset_index(drop=True)


def build_long_aggregate_df(agg_df: pd.DataFrame) -> pd.DataFrame:
    long_rows: list[dict[str, object]] = []
    for row in agg_df.to_dict(orient="records"):
        long_rows.append(
            {
                "T": int(row["T"]),
                "batch_size": int(row["batch_size"]),
                "split": "Train",
                "mean_accuracy": float(row["mean_train_acc"]),
                "std_accuracy": float(row["std_train_acc"]),
            }
        )
        long_rows.append(
            {
                "T": int(row["T"]),
                "batch_size": int(row["batch_size"]),
                "split": "Test",
                "mean_accuracy": float(row["mean_test_acc"]),
                "std_accuracy": float(row["std_test_acc"]),
            }
        )
    return pd.DataFrame(long_rows).sort_values(["batch_size", "split", "T"]).reset_index(drop=True)


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def make_main_figure(seed_df: pd.DataFrame, agg_df: pd.DataFrame, out_path: Path) -> None:
    _style()
    long_seed = build_long_accuracy_df(seed_df)
    long_agg = build_long_aggregate_df(agg_df)
    batch_sizes = _sorted_ints(long_seed["batch_size"])
    t_values = _sorted_ints(long_seed["T"])
    batch_palette = _batch_palette(batch_sizes)

    y_min = min(long_seed["accuracy"].min(), long_agg["mean_accuracy"].min()) - 0.025
    y_max = max(long_seed["accuracy"].max(), long_agg["mean_accuracy"].max()) + 0.025

    fig = plt.figure(figsize=(max(14, 5.7 * len(batch_sizes)), 10), dpi=220)
    gs = fig.add_gridspec(2, len(batch_sizes), height_ratios=[1.05, 0.95], hspace=0.25, wspace=0.16)
    top_axes = {
        batch_size: fig.add_subplot(gs[0, idx])
        for idx, batch_size in enumerate(batch_sizes)
    }
    frontier_ax = fig.add_subplot(gs[1, :])
    split_palette = {"Train": TRAIN_COLOR, "Test": TEST_COLOR}

    for batch_size in batch_sizes:
        ax = top_axes[batch_size]
        seed_batch = long_seed[long_seed["batch_size"] == batch_size]
        agg_batch = long_agg[long_agg["batch_size"] == batch_size]

        for split_name in ("Train", "Test"):
            split_seed = seed_batch[seed_batch["split"] == split_name]
            for seed_value, seed_group in split_seed.groupby("seed", sort=True):
                ax.plot(
                    seed_group["T"],
                    seed_group["accuracy"],
                    color=split_palette[split_name],
                    alpha=0.14,
                    linewidth=1.2,
                    zorder=1,
                )
            ax.scatter(
                split_seed["x_plot"],
                split_seed["accuracy"],
                color=split_palette[split_name],
                alpha=0.28,
                s=85,
                edgecolors="none",
                zorder=2,
            )

            split_agg = agg_batch[agg_batch["split"] == split_name].sort_values("T")
            ax.plot(
                split_agg["T"],
                split_agg["mean_accuracy"],
                color=split_palette[split_name],
                linewidth=3.3,
                marker="o" if split_name == "Train" else "s",
                markersize=8,
                zorder=4,
            )
            ax.fill_between(
                split_agg["T"],
                split_agg["mean_accuracy"] - split_agg["std_accuracy"],
                split_agg["mean_accuracy"] + split_agg["std_accuracy"],
                color=split_palette[split_name],
                alpha=0.10,
                zorder=3,
            )

        ax.set_title(f"Batch {batch_size}", fontsize=15, pad=10)
        ax.set_xlabel("Boosting Rounds (T)")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(t_values)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.22)

    frontier_data = agg_df.sort_values(["batch_size", "T"]).copy()

    for batch_size, group in frontier_data.groupby("batch_size", sort=True):
        group = group.sort_values("T")
        frontier_ax.plot(
            group["mean_cost_usd"],
            group["mean_test_acc"],
            color=batch_palette[int(batch_size)],
            linewidth=2.4,
            alpha=0.9,
            zorder=2,
        )
        frontier_ax.scatter(
            group["mean_cost_usd"],
            group["mean_test_acc"],
            color=batch_palette[int(batch_size)],
            s=180,
            alpha=0.95,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        for row in group.itertuples(index=False):
            frontier_ax.annotate(
                f"T={int(row.T)}",
                (row.mean_cost_usd, row.mean_test_acc),
                textcoords="offset points",
                xytext=(7, 7),
                fontsize=11,
                color=batch_palette[int(batch_size)],
            )

    frontier_ax.set_title("Cost vs Mean Test Accuracy", fontsize=15, pad=10)
    frontier_ax.set_xlabel("Mean Estimated Cost per Run (USD)")
    frontier_ax.set_ylabel("Mean Test Accuracy Across 3 Seeds")
    frontier_ax.grid(alpha=0.22)

    top_legend_handles = [
        Line2D([0], [0], color=TRAIN_COLOR, lw=3.0, marker="o", markersize=8, label="Mean Train"),
        Line2D([0], [0], color=TEST_COLOR, lw=3.0, marker="s", markersize=8, label="Mean Test"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TRAIN_COLOR, alpha=0.28, markersize=9, label="Seed Runs"),
    ]
    fig.legend(
        handles=top_legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.928),
    )

    frontier_handles = [
        Line2D([0], [0], color=batch_palette[int(batch_size)], lw=2.4, label=f"batch {int(batch_size)} frontier")
        for batch_size in batch_sizes
    ]
    frontier_ax.legend(handles=frontier_handles, frameon=False, loc="lower right")

    fig.suptitle("CodeBoost on CDC Diabetes at Large Scale", y=0.988, fontsize=19)
    fig.text(0.5, 0.952, "train=10,000, test=60,000, 3 seeds", ha="center", fontsize=12)
    fig.subplots_adjust(top=0.885, bottom=0.08, left=0.07, right=0.985, hspace=0.28, wspace=0.16)
    fig.savefig(out_path)
    plt.close(fig)


def make_seed_trend_figure(seed_df: pd.DataFrame, agg_df: pd.DataFrame, out_path: Path) -> None:
    _style()
    long_seed = build_long_accuracy_df(seed_df)
    long_agg = build_long_aggregate_df(agg_df)
    batch_sizes = _sorted_ints(long_seed["batch_size"])
    t_values = _sorted_ints(long_seed["T"])

    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(max(8.5, 5.9 * len(batch_sizes)), 6.4), dpi=240, sharey=True)
    axes = np.atleast_1d(axes)
    split_palette = {"Train": TRAIN_COLOR, "Test": TEST_COLOR}
    y_min = min(long_seed["accuracy"].min(), long_agg["mean_accuracy"].min()) - 0.025
    y_max = max(long_seed["accuracy"].max(), long_agg["mean_accuracy"].max()) + 0.025

    for ax, batch_size in zip(axes, batch_sizes):
        seed_batch = long_seed[long_seed["batch_size"] == batch_size]
        agg_batch = long_agg[long_agg["batch_size"] == batch_size]

        for split_name in ("Train", "Test"):
            split_seed = seed_batch[seed_batch["split"] == split_name]
            for _seed_value, seed_group in split_seed.groupby("seed", sort=True):
                ax.plot(
                    seed_group["T"],
                    seed_group["accuracy"],
                    color=split_palette[split_name],
                    alpha=0.16,
                    linewidth=1.1,
                    zorder=1,
                )
            ax.scatter(
                split_seed["x_plot"],
                split_seed["accuracy"],
                color=split_palette[split_name],
                alpha=0.30,
                s=92,
                edgecolors="none",
                zorder=2,
            )

            split_agg = agg_batch[agg_batch["split"] == split_name].sort_values("T")
            ax.plot(
                split_agg["T"],
                split_agg["mean_accuracy"],
                color=split_palette[split_name],
                linewidth=3.5,
                marker="o" if split_name == "Train" else "s",
                markersize=8,
                zorder=4,
            )
            ax.fill_between(
                split_agg["T"],
                split_agg["mean_accuracy"] - split_agg["std_accuracy"],
                split_agg["mean_accuracy"] + split_agg["std_accuracy"],
                color=split_palette[split_name],
                alpha=0.11,
                zorder=3,
            )

        ax.text(
            0.03,
            0.97,
            f"batch {batch_size}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="#2c2c2c",
        )
        ax.set_xlabel("Boosting Rounds (T)")
        ax.set_xticks(t_values)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.22)

    axes[0].set_ylabel("Accuracy")
    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(labelleft=True)

    legend_handles = [
        Line2D([0], [0], color=TRAIN_COLOR, lw=3.2, marker="o", markersize=8, label="Mean Train"),
        Line2D([0], [0], color=TEST_COLOR, lw=3.2, marker="s", markersize=8, label="Mean Test"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TRAIN_COLOR, alpha=0.30, markersize=9, label="Seed Runs"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.938),
    )
    fig.suptitle("CodeBoost Accuracy vs Boosting Rounds", y=0.978, fontsize=18)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.985, wspace=0.18)
    fig.savefig(out_path)
    plt.close(fig)


def make_parity_figure(seed_df: pd.DataFrame, out_path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.8, 7.6), dpi=220)
    batch_sizes = _sorted_ints(seed_df["batch_size"])
    t_values = _sorted_ints(seed_df["T"])
    batch_palette = _batch_palette(batch_sizes)
    marker_cycle = ["o", "s", "D", "^", "P", "X"]
    marker_map = {
        t_value: marker_cycle[idx % len(marker_cycle)]
        for idx, t_value in enumerate(t_values)
    }

    for batch_size, batch_group in seed_df.groupby("batch_size", sort=True):
        for t_value, t_group in batch_group.groupby("T", sort=True):
            ax.scatter(
                t_group["final_train_acc"],
                t_group["final_test_acc"],
                s=130,
                alpha=0.92,
                color=batch_palette[int(batch_size)],
                marker=marker_map[int(t_value)],
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )

    lo = min(seed_df["final_train_acc"].min(), seed_df["final_test_acc"].min()) - 0.02
    hi = max(seed_df["final_train_acc"].max(), seed_df["final_test_acc"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.6, color="#5c5c5c", alpha=0.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Train Accuracy")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Train vs Test Accuracy by Seed", fontsize=16, pad=10)
    ax.grid(alpha=0.22)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=batch_palette[int(batch_size)], markeredgecolor="white", markersize=10, label=f"batch {int(batch_size)}")
        for batch_size in batch_sizes
    ] + [
        Line2D([0], [0], marker=marker_map[int(t_value)], color="#444444", linestyle="none", markersize=9, label=f"T={int(t_value)}")
        for t_value in t_values
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_slack_message(
    run_dir: Path,
    seed_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    overall_corr: float,
    total_cost: float,
    attempt_health: dict[str, int],
) -> None:
    best_row = agg_df.sort_values("mean_test_acc", ascending=False).iloc[0]
    train_size, test_size = _parse_run_sizes(run_dir)
    seed_count = int(seed_df["seed"].nunique())
    t_values = _format_set(seed_df["T"])
    batch_values = _format_set(seed_df["batch_size"])
    monotonic_batches: list[int] = []
    for batch_size, group in agg_df.groupby("batch_size", sort=True):
        if _is_monotonic_increasing(group.sort_values("T")["mean_test_acc"].tolist()):
            monotonic_batches.append(int(batch_size))
    fully_accepted_mask = seed_df["accepted_rounds"] == seed_df["requested_rounds"]
    fully_accepted_runs = int(fully_accepted_mask.sum())
    total_runs = int(len(seed_df))

    lines = []
    if train_size is not None and test_size is not None:
        lines.append(
            f"- Ran CDC Diabetes on balanced sampled splits with train={train_size:,}, test={test_size:,}, {seed_count} seeds, T in {t_values}, and batch in {batch_values}."
        )
    else:
        lines.append(
            f"- Ran CDC Diabetes on balanced sampled splits with {seed_count} seeds, T in {t_values}, and batch in {batch_values}."
        )

    lines.append(
        f"- Best result was T={int(best_row['T'])}, batch={int(best_row['batch_size'])} with mean test accuracy {best_row['mean_test_acc']:.4f} across {seed_count} seeds and test sd {best_row['std_test_acc']:.4f}."
    )
    if monotonic_batches:
        lines.append(
            f"- Mean test accuracy increased monotonically with T for batch {', '.join(str(v) for v in monotonic_batches)}."
        )
    lines.append(f"- Overall train/test correlation across final runs was {overall_corr:.4f}.")
    if attempt_health["compile_error_rows"] == 0 and attempt_health["eval_error_rows"] == 0:
        if fully_accepted_runs == total_runs:
            lines.append(
                f"- All {attempt_health['total_attempt_rows']} attempt rows compiled and evaluated cleanly, and every final run reached its requested number of accepted rounds."
            )
        else:
            lines.append(
                f"- All {attempt_health['total_attempt_rows']} attempt rows compiled and evaluated cleanly; {fully_accepted_runs}/{total_runs} final runs reached their requested number of accepted rounds."
            )
    else:
        lines.append(
            f"- Sweep health: {attempt_health['total_attempt_rows']} attempt rows, {attempt_health['compile_error_rows']} compile-error rows, {attempt_health['eval_error_rows']} eval-error rows."
        )
    lines.append(f"- Total estimated spend for this sweep was ${total_cost:.6f}.")
    (run_dir / "slack_message.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_claims_file(
    run_dir: Path,
    seed_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    overall_corr: float,
    total_cost: float,
    attempt_health: dict[str, int],
) -> None:
    best = agg_df.sort_values("mean_test_acc", ascending=False).iloc[0]
    most_stable = agg_df.sort_values("std_test_acc", ascending=True).iloc[0]
    train_size, test_size = _parse_run_sizes(run_dir)
    seed_count = int(seed_df["seed"].nunique())
    batch_sizes = _sorted_ints(seed_df["batch_size"])
    t_values = _sorted_ints(seed_df["T"])
    gap_min = float(agg_df["mean_gap_train_minus_test"].min())
    gap_max = float(agg_df["mean_gap_train_minus_test"].max())
    fully_accepted_mask = seed_df["accepted_rounds"] == seed_df["requested_rounds"]
    fully_accepted_runs = int(fully_accepted_mask.sum())
    total_runs = int(len(seed_df))

    lines = [
        "- This repo's CDC Diabetes path does not use a native train/test split. It fetches the raw UCI dataset, samples a balanced pool, and writes stratified cached splits for each requested size and seed.",
    ]
    if train_size is not None and test_size is not None:
        lines.append(
            f"- The current run used train={train_size:,}, test={test_size:,}, val=0 with {seed_count} seeds and batch sizes {batch_sizes}."
        )
    lines.append(
        f"- Across {len(seed_df)} final runs, overall train/test correlation was {overall_corr:.4f}."
    )
    lines.append(
        f"- Across the {len(agg_df)} configuration means, the average train-minus-test gap stayed between {gap_min:+.4f} and {gap_max:+.4f}."
    )

    for batch_size, group in agg_df.groupby("batch_size", sort=True):
        group = group.sort_values("T")
        test_values = group["mean_test_acc"].tolist()
        if _is_monotonic_increasing(test_values):
            joined = ", ".join(
                f"{acc:.4f} at T={int(t_value)}"
                for t_value, acc in zip(group["T"].tolist(), test_values)
            )
            lines.append(
                f"- For batch {int(batch_size)}, mean test accuracy increased monotonically with T: {joined}."
            )

    for t_value, group in agg_df.groupby("T", sort=True):
        group = group.sort_values("mean_test_acc", ascending=False)
        if len(group) >= 2:
            top = group.iloc[0]
            second = group.iloc[1]
            lines.append(
                f"- At T={int(t_value)}, the best batch was {int(top['batch_size'])} with mean test accuracy {top['mean_test_acc']:.4f}; the next best was batch {int(second['batch_size'])} at {second['mean_test_acc']:.4f}."
            )

    lines.extend(
        [
            f"- The best mean test configuration was T={int(best['T'])}, batch={int(best['batch_size'])} with mean test accuracy {best['mean_test_acc']:.4f}, test sd {best['std_test_acc']:.4f}, and mean cost ${best['mean_cost_usd']:.6f} per run.",
            f"- The lowest seed-level test variance in this sweep was T={int(most_stable['T'])}, batch={int(most_stable['batch_size'])} with test sd {most_stable['std_test_acc']:.4f}.",
            f"- Attempt health for this sweep: {attempt_health['total_attempt_rows']} attempt rows, {attempt_health['compile_error_rows']} compile-error rows, and {attempt_health['eval_error_rows']} eval-error rows.",
            f"- {fully_accepted_runs}/{total_runs} final runs accepted all requested rounds for their T setting.",
            f"- The estimated total spend for this full sweep was ${total_cost:.6f}.",
            "- These results support conclusions only for this sampled-split CDC setup, this prompt/model path, and this AdaBoost-style runner configuration.",
        ]
    )
    (run_dir / "data_backed_claims.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(run_dir: Path) -> None:
    seed_df = load_seed_rows(run_dir)
    agg_df = build_aggregate(seed_df)

    overall_corr = _corr(seed_df["final_train_acc"].tolist(), seed_df["final_test_acc"].tolist())
    total_cost = float(seed_df["total_estimated_cost_usd"].sum())
    attempt_health = summarize_attempt_health(run_dir)

    seed_df.to_csv(run_dir / "seed_level_summary.csv", index=False)
    agg_df.to_csv(run_dir / "aggregate_seed_summary.csv", index=False)

    make_seed_trend_figure(
        seed_df=seed_df,
        agg_df=agg_df,
        out_path=run_dir / "figure_accuracy_seed_trends.png",
    )
    make_main_figure(
        seed_df=seed_df,
        agg_df=agg_df,
        out_path=run_dir / "figure_accuracy_cost_overview.png",
    )
    make_parity_figure(
        seed_df=seed_df,
        out_path=run_dir / "figure_train_test_parity.png",
    )
    write_slack_message(run_dir, seed_df, agg_df, overall_corr, total_cost, attempt_health)
    write_claims_file(run_dir, seed_df, agg_df, overall_corr, total_cost, attempt_health)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a multi-seed boosted scale sweep directory.")
    parser.add_argument("--run-dir", required=True, help="Directory like outputs_scale_matrix_*/train10000_test60000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(Path(args.run_dir).resolve())


if __name__ == "__main__":
    main()
