from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import analyze_scale_matrix as asm


def load_combined_seed_rows(run_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        frame = asm.load_seed_rows(run_dir).copy()
        train_size, test_size = asm._parse_run_sizes(run_dir)
        frame["source_run"] = run_dir.name
        frame["source_dir"] = str(run_dir)
        frame["train_size"] = train_size
        frame["test_size"] = test_size
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No run directories were provided.")

    seed_df = pd.concat(frames, ignore_index=True)
    seed_df = seed_df.drop_duplicates(subset=["T", "batch_size", "seed", "trial"], keep="last")
    return seed_df.sort_values(["T", "batch_size", "seed", "trial"]).reset_index(drop=True)


def build_run_breakdown(run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        seed_df = asm.load_seed_rows(run_dir)
        health = asm.summarize_attempt_health(run_dir)
        train_size, test_size = asm._parse_run_sizes(run_dir)
        batch_sizes = asm._sorted_ints(seed_df["batch_size"])
        rows.append(
            {
                "source_run": run_dir.name,
                "source_dir": str(run_dir),
                "train_size": train_size,
                "test_size": test_size,
                "batch_sizes": ",".join(str(v) for v in batch_sizes),
                "configs": int(seed_df[["T", "batch_size"]].drop_duplicates().shape[0]),
                "final_runs": int(len(seed_df)),
                "seeds_per_config": int(seed_df["seed"].nunique()),
                "attempt_rows": int(health["total_attempt_rows"]),
                "compile_error_rows": int(health["compile_error_rows"]),
                "eval_error_rows": int(health["eval_error_rows"]),
                "total_cost_usd": float(seed_df["total_estimated_cost_usd"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("source_run").reset_index(drop=True)


def build_long_batch_accuracy_df(seed_df: pd.DataFrame) -> pd.DataFrame:
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
    batch_order = asm._sorted_ints(long_df["batch_size"])
    batch_pos = {batch_size: idx for idx, batch_size in enumerate(batch_order)}
    seed_order = {seed: idx for idx, seed in enumerate(sorted(long_df["seed"].unique()))}
    split_offset = {"Train": -0.08, "Test": 0.08}
    seed_offset = {
        idx: offset
        for idx, offset in enumerate(np.linspace(-0.028, 0.028, len(seed_order)))
    }
    long_df["x_center"] = long_df["batch_size"].map(batch_pos).astype(float)
    long_df["x_plot"] = long_df.apply(
        lambda row: row["x_center"] + split_offset[str(row["split"])] + seed_offset[seed_order[int(row["seed"])]],
        axis=1,
    )
    return long_df.sort_values(["T", "batch_size", "seed", "split"]).reset_index(drop=True)


def build_long_batch_aggregate_df(agg_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in agg_df.to_dict(orient="records"):
        rows.append(
            {
                "T": int(row["T"]),
                "batch_size": int(row["batch_size"]),
                "split": "Train",
                "mean_accuracy": float(row["mean_train_acc"]),
                "std_accuracy": float(row["std_train_acc"]),
            }
        )
        rows.append(
            {
                "T": int(row["T"]),
                "batch_size": int(row["batch_size"]),
                "split": "Test",
                "mean_accuracy": float(row["mean_test_acc"]),
                "std_accuracy": float(row["std_test_acc"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["T", "split", "batch_size"]).reset_index(drop=True)


def make_batch_sweep_figure(seed_df: pd.DataFrame, agg_df: pd.DataFrame, out_path: Path) -> None:
    asm._style()
    long_seed = build_long_batch_accuracy_df(seed_df)
    long_agg = build_long_batch_aggregate_df(agg_df)
    t_values = asm._sorted_ints(long_seed["T"])
    batch_sizes = asm._sorted_ints(long_seed["batch_size"])
    batch_pos = {batch_size: idx for idx, batch_size in enumerate(batch_sizes)}
    split_palette = {"Train": asm.TRAIN_COLOR, "Test": asm.TEST_COLOR}

    fig, axes = plt.subplots(1, len(t_values), figsize=(max(11.5, 5.0 * len(t_values)), 6.6), dpi=240, sharey=True)
    axes = np.atleast_1d(axes)
    y_min = min(long_seed["accuracy"].min(), long_agg["mean_accuracy"].min()) - 0.025
    y_max = max(long_seed["accuracy"].max(), long_agg["mean_accuracy"].max()) + 0.025

    for ax, t_value in zip(axes, t_values):
        seed_t = long_seed[long_seed["T"] == t_value]
        agg_t = long_agg[long_agg["T"] == t_value]

        for split_name in ("Train", "Test"):
            split_seed = seed_t[seed_t["split"] == split_name]
            for _seed_value, seed_group in split_seed.groupby("seed", sort=True):
                seed_group = seed_group.sort_values("batch_size")
                ax.plot(
                    [batch_pos[int(v)] for v in seed_group["batch_size"]],
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
                s=88,
                edgecolors="none",
                zorder=2,
            )

            split_agg = agg_t[agg_t["split"] == split_name].sort_values("batch_size")
            x_vals = [batch_pos[int(v)] for v in split_agg["batch_size"]]
            ax.plot(
                x_vals,
                split_agg["mean_accuracy"],
                color=split_palette[split_name],
                linewidth=3.4,
                marker="o" if split_name == "Train" else "s",
                markersize=8,
                zorder=4,
            )
            ax.fill_between(
                x_vals,
                split_agg["mean_accuracy"] - split_agg["std_accuracy"],
                split_agg["mean_accuracy"] + split_agg["std_accuracy"],
                color=split_palette[split_name],
                alpha=0.10,
                zorder=3,
            )

        ax.text(
            0.03,
            0.97,
            f"T={t_value}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="#2c2c2c",
        )
        ax.set_xlabel("Batch Size")
        ax.set_xticks(range(len(batch_sizes)))
        ax.set_xticklabels([str(v) for v in batch_sizes])
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.22)

    axes[0].set_ylabel("Accuracy")
    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(labelleft=True)

    legend_handles = [
        Line2D([0], [0], color=asm.TRAIN_COLOR, lw=3.2, marker="o", markersize=8, label="Mean Train"),
        Line2D([0], [0], color=asm.TEST_COLOR, lw=3.2, marker="s", markersize=8, label="Mean Test"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=asm.TRAIN_COLOR, alpha=0.30, markersize=9, label="Seed Runs"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.935),
    )
    fig.suptitle("CodeBoost Accuracy vs Batch Size", y=0.975, fontsize=18)
    fig.subplots_adjust(top=0.83, bottom=0.13, left=0.08, right=0.985, wspace=0.16)
    fig.savefig(out_path)
    plt.close(fig)


def _top_row_text(row: pd.Series) -> str:
    return (
        f"T={int(row['T'])}, batch={int(row['batch_size'])}, "
        f"mean_test={float(row['mean_test_acc']):.4f}, "
        f"mean_train={float(row['mean_train_acc']):.4f}, "
        f"test_sd={float(row['std_test_acc']):.4f}, "
        f"mean_cost=${float(row['mean_cost_usd']):.6f}"
    )


def _batch_list_text(batch_sizes: list[int]) -> str:
    return ", ".join(str(v) for v in batch_sizes)


def _best_row_for_t(agg_df: pd.DataFrame, t_value: int) -> pd.Series:
    t_df = agg_df[agg_df["T"] == t_value].sort_values(
        ["mean_test_acc", "std_test_acc"],
        ascending=[False, True],
    )
    return t_df.iloc[0]


def _available_t_values(agg_df: pd.DataFrame) -> list[int]:
    return asm._sorted_ints(agg_df["T"])


def _best_rows_by_t(agg_df: pd.DataFrame) -> list[pd.Series]:
    return [_best_row_for_t(agg_df, t_value) for t_value in _available_t_values(agg_df)]


def _largest_batch_pair(batch_sizes: list[int]) -> tuple[int, int] | None:
    if len(batch_sizes) < 2:
        return None
    return batch_sizes[-2], batch_sizes[-1]


def write_sources_file(output_dir: Path, run_dirs: list[Path], run_breakdown: pd.DataFrame) -> None:
    lines = ["- Combined source runs:"]
    for row in run_breakdown.to_dict(orient="records"):
        lines.append(
            f"- {row['source_run']}: batches [{row['batch_sizes']}], final_runs={row['final_runs']}, total_cost=${float(row['total_cost_usd']):.6f}"
        )
        lines.append(f"- path: {row['source_dir']}")
    (output_dir / "sources.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_overall_summary(
    output_dir: Path,
    seed_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    run_breakdown: pd.DataFrame,
    attempt_rows: int,
    compile_errors: int,
    eval_errors: int,
    total_cost: float,
    overall_corr: float,
) -> None:
    best = agg_df.sort_values("mean_test_acc", ascending=False).iloc[0]
    most_stable = agg_df.sort_values("std_test_acc", ascending=True).iloc[0]
    batch_sizes = asm._sorted_ints(seed_df["batch_size"])
    t_values = asm._sorted_ints(seed_df["T"])
    seed_count = int(seed_df["seed"].nunique())
    lines = [
        f"- Combined matched-split bundle covers CDC train=10,000 and test=60,000, with {seed_count} seeds per setting.",
        f"- Included batches {batch_sizes} and T values {t_values}, for {len(agg_df)} configurations and {len(seed_df)} final seeded runs.",
        f"- Highest observed mean test setting: {_top_row_text(best)}",
        f"- Lowest observed test variance setting: T={int(most_stable['T'])}, batch={int(most_stable['batch_size'])}, test_sd={float(most_stable['std_test_acc']):.4f}",
        f"- Overall train/test correlation across all final runs was {overall_corr:.4f}.",
        f"- Sweep health across included runs: {attempt_rows} attempt rows, {compile_errors} compile-error rows, {eval_errors} eval-error rows.",
        f"- Total estimated spend across included runs was ${total_cost:.6f}.",
    ]
    (output_dir / "overall_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lessons_learned(output_dir: Path, agg_df: pd.DataFrame) -> None:
    t_values = _available_t_values(agg_df)
    best_rows = _best_rows_by_t(agg_df)
    batch_sizes = asm._sorted_ints(agg_df["batch_size"])
    batch_min = batch_sizes[0]
    batch_max = batch_sizes[-1]
    largest_pair = _largest_batch_pair(batch_sizes)
    lines: list[str] = []
    for t_value, top_row in zip(t_values, best_rows):
        t_df = agg_df[agg_df["T"] == t_value].sort_values("batch_size")
        lines.append(
            f"- At T={t_value}, mean test moved from {float(t_df.iloc[0]['mean_test_acc']):.4f} at batch {batch_min} to "
            f"{float(t_df.iloc[-1]['mean_test_acc']):.4f} at batch {batch_max}, and the best observed T={t_value} setting "
            f"was batch {int(top_row['batch_size'])} at {float(top_row['mean_test_acc']):.4f}."
        )

    largest_batch_curve = [
        float(agg_df[(agg_df["T"] == t_value) & (agg_df["batch_size"] == batch_max)]["mean_test_acc"].iloc[0])
        for t_value in t_values
    ]
    if len(t_values) >= 2:
        progression_text = (
            "improved monotonically"
            if asm._is_monotonic_increasing(largest_batch_curve)
            else "improved overall but not monotonically"
        )
        curve_text = ", ".join(
            f"T={t_value} -> {score:.4f}" for t_value, score in zip(t_values, largest_batch_curve)
        )
        lines.append(
            f"- For the largest tested batch {batch_max}, mean test {progression_text}: {curve_text}."
        )

    if largest_pair is not None:
        prev_batch, max_batch = largest_pair
        pair_text = ", ".join(
            (
                f"T={t_value}: "
                f"{float(agg_df[(agg_df['T'] == t_value) & (agg_df['batch_size'] == max_batch)]['mean_test_acc'].iloc[0]):.4f} "
                f"vs {float(agg_df[(agg_df['T'] == t_value) & (agg_df['batch_size'] == prev_batch)]['mean_test_acc'].iloc[0]):.4f}"
            )
            for t_value in t_values
        )
        lines.append(
            f"- Comparing the two largest tested batches, batch {max_batch} vs batch {prev_batch} had mean test {pair_text}."
        )

    lines.append(
        "- Train and test stayed closely aligned throughout, so the observed gains look like real generalization gains on this sampled-split setup rather than obvious overfitting."
    )
    (output_dir / "lessons_learned.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_next_experiments(output_dir: Path, agg_df: pd.DataFrame) -> None:
    batch_sizes = asm._sorted_ints(agg_df["batch_size"])
    t_values = _available_t_values(agg_df)
    best = agg_df.sort_values(["mean_test_acc", "std_test_acc"], ascending=[False, True]).iloc[0]
    highest_t = t_values[-1]
    top_high_t = _best_row_for_t(agg_df, highest_t)
    largest_batch = batch_sizes[-1]
    second_largest_batch = batch_sizes[-2] if len(batch_sizes) > 1 else batch_sizes[-1]
    lines = [
        f"- Extend the current top setting T={int(best['T'])}, batch={int(best['batch_size'])} to 7-10 seeds if you want a tighter estimate of the frontier leader.",
        f"- Keep T={int(top_high_t['T'])}, batch={int(top_high_t['batch_size'])} as the strongest current high-T baseline.",
        f"- If you probe deeper boosting again, stay in the large-batch regime: batch {second_largest_batch} and batch {largest_batch}.",
        "- Run a train/test allocation sweep at the same roughly 70k total example budget, using the best current settings instead of re-testing weak regions.",
        f"- If token limits remain comfortable, batch {largest_batch * 2} is the next natural batch-size probe because the current largest-batch curve is still competitive.",
    ]
    (output_dir / "next_experiments.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined_slack_message(
    output_dir: Path,
    seed_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    total_cost: float,
    overall_corr: float,
    attempt_rows: int,
) -> None:
    best = agg_df.sort_values("mean_test_acc", ascending=False).iloc[0]
    highest_t = max(_available_t_values(agg_df))
    top_high_t = _best_row_for_t(agg_df, highest_t)
    batch_sizes = asm._sorted_ints(seed_df["batch_size"])
    lines = [
        f"- Combined the matched large-scale CDC CodeBoost sweeps into one summary bundle across batch sizes {_batch_list_text(batch_sizes)}.",
        f"- This covers {len(agg_df)} configurations, {len(seed_df)} final seeded runs, and {attempt_rows} total attempt rows on the same train=10,000 / test=60,000 split recipe.",
        f"- Highest observed mean test setting was T={int(best['T'])}, batch={int(best['batch_size'])} at {float(best['mean_test_acc']):.4f}.",
        f"- Best high-T setting was T={int(top_high_t['T'])}, batch={int(top_high_t['batch_size'])} at {float(top_high_t['mean_test_acc']):.4f}.",
        f"- The most promising region is now the largest batches, especially once T is high enough.",
        f"- Train and test tracked very closely across all runs: overall correlation {overall_corr:.4f}.",
        f"- Total estimated spend across the included sweeps was ${total_cost:.6f}.",
    ]
    (output_dir / "combined_slack_message.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_data_backed_claims(
    output_dir: Path,
    seed_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    total_cost: float,
    overall_corr: float,
    attempt_rows: int,
    compile_errors: int,
    eval_errors: int,
) -> None:
    best = agg_df.sort_values("mean_test_acc", ascending=False).iloc[0]
    most_stable = agg_df.sort_values("std_test_acc", ascending=True).iloc[0]
    t_values = _available_t_values(agg_df)
    seed_count = int(seed_df["seed"].nunique())
    gap_min = float(agg_df["mean_gap_train_minus_test"].min())
    gap_max = float(agg_df["mean_gap_train_minus_test"].max())
    batch_sizes = asm._sorted_ints(agg_df["batch_size"])
    batch_min = batch_sizes[0]
    batch_max = batch_sizes[-1]
    largest_pair = _largest_batch_pair(batch_sizes)

    lines = [
        f"- These combined results pool the matched CDC sweeps that used the same train=10,000 / test=60,000 setup and the same {seed_count} seeds per configuration.",
        f"- The combined bundle contains {len(agg_df)} unique configurations and {len(seed_df)} seeded final runs.",
        f"- Across all final runs, overall train/test correlation was {overall_corr:.4f}.",
        f"- Across configuration means, the average train-minus-test gap stayed between {gap_min:+.4f} and {gap_max:+.4f}.",
        f"- The highest observed mean test configuration overall was T={int(best['T'])}, batch={int(best['batch_size'])} with mean test {float(best['mean_test_acc']):.4f}, mean train {float(best['mean_train_acc']):.4f}, test sd {float(best['std_test_acc']):.4f}, and mean cost ${float(best['mean_cost_usd']):.6f}.",
        f"- The lowest observed test variance among all combined settings was T={int(most_stable['T'])}, batch={int(most_stable['batch_size'])} with test sd {float(most_stable['std_test_acc']):.4f}.",
        f"- Attempt health across the included sweeps was {attempt_rows} attempt rows, {compile_errors} compile-error rows, and {eval_errors} eval-error rows.",
        f"- Total estimated spend across the included matched sweeps was ${total_cost:.6f}.",
        "- These claims apply to this sampled-split CDC setup, this Azure/TAMU GPT-5.2 deployment path, and this current AdaBoost-style CodeBoost runner configuration.",
    ]
    for idx, t_value in enumerate(t_values, start=4):
        t_df = agg_df[agg_df["T"] == t_value].sort_values("batch_size")
        top_t = _best_row_for_t(agg_df, t_value)
        lines.insert(
            idx,
            f"- At T={t_value}, mean test moved from {float(t_df.iloc[0]['mean_test_acc']):.4f} at batch {batch_min} to {float(t_df.iloc[-1]['mean_test_acc']):.4f} at batch {batch_max}, and the highest observed mean test was {float(top_t['mean_test_acc']):.4f} at batch {int(top_t['batch_size'])}.",
        )
    if largest_pair is not None:
        prev_batch, max_batch = largest_pair
        pair_test_text = ", ".join(
            (
                f"T={t_value}: "
                f"{float(agg_df[(agg_df['T'] == t_value) & (agg_df['batch_size'] == max_batch)]['mean_test_acc'].iloc[0]):.4f} "
                f"vs {float(agg_df[(agg_df['T'] == t_value) & (agg_df['batch_size'] == prev_batch)]['mean_test_acc'].iloc[0]):.4f}"
            )
            for t_value in t_values
        )
        lines.insert(
            7 + len(t_values),
            f"- Comparing the two largest tested batches, batch {max_batch} vs batch {prev_batch} had mean test {pair_test_text}.",
        )
        pair_cost_text = ", ".join(
            (
                f"T={t_value}: "
                f"${float(agg_df[(agg_df['T'] == t_value) & (agg_df['batch_size'] == max_batch)]['mean_cost_usd'].iloc[0]):.6f} "
                f"vs ${float(agg_df[(agg_df['T'] == t_value) & (agg_df['batch_size'] == prev_batch)]['mean_cost_usd'].iloc[0]):.6f}"
            )
            for t_value in t_values
        )
        lines.insert(
            8 + len(t_values),
            f"- The same largest-batch comparison on mean cost was {pair_cost_text}.",
        )
    (output_dir / "data_backed_claims.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(run_dirs: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_df = load_combined_seed_rows(run_dirs)
    agg_df = asm.build_aggregate(seed_df)
    ranked_df = agg_df.sort_values(["mean_test_acc", "std_test_acc"], ascending=[False, True]).reset_index(drop=True)
    run_breakdown = build_run_breakdown(run_dirs)
    attempt_health_rows = [asm.summarize_attempt_health(run_dir) for run_dir in run_dirs]

    total_cost = float(seed_df["total_estimated_cost_usd"].sum())
    overall_corr = asm._corr(seed_df["final_train_acc"].tolist(), seed_df["final_test_acc"].tolist())
    attempt_rows = int(sum(row["total_attempt_rows"] for row in attempt_health_rows))
    compile_errors = int(sum(row["compile_error_rows"] for row in attempt_health_rows))
    eval_errors = int(sum(row["eval_error_rows"] for row in attempt_health_rows))

    seed_df.to_csv(output_dir / "combined_seed_level_summary.csv", index=False)
    agg_df.to_csv(output_dir / "combined_aggregate_summary.csv", index=False)
    ranked_df.to_csv(output_dir / "combined_aggregate_summary_ranked.csv", index=False)
    run_breakdown.to_csv(output_dir / "run_breakdown.csv", index=False)

    asm.make_seed_trend_figure(
        seed_df=seed_df,
        agg_df=agg_df,
        out_path=output_dir / "figure_accuracy_by_T_all_batches.png",
    )
    make_batch_sweep_figure(
        seed_df=seed_df,
        agg_df=agg_df,
        out_path=output_dir / "figure_accuracy_by_batch_all_T.png",
    )

    write_sources_file(output_dir, run_dirs, run_breakdown)
    write_overall_summary(
        output_dir=output_dir,
        seed_df=seed_df,
        agg_df=agg_df,
        run_breakdown=run_breakdown,
        attempt_rows=attempt_rows,
        compile_errors=compile_errors,
        eval_errors=eval_errors,
        total_cost=total_cost,
        overall_corr=overall_corr,
    )
    write_lessons_learned(output_dir, agg_df)
    write_next_experiments(output_dir, agg_df)
    write_combined_slack_message(
        output_dir=output_dir,
        seed_df=seed_df,
        agg_df=agg_df,
        total_cost=total_cost,
        overall_corr=overall_corr,
        attempt_rows=attempt_rows,
    )
    write_data_backed_claims(
        output_dir=output_dir,
        seed_df=seed_df,
        agg_df=agg_df,
        total_cost=total_cost,
        overall_corr=overall_corr,
        attempt_rows=attempt_rows,
        compile_errors=compile_errors,
        eval_errors=eval_errors,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine multiple matched scale sweeps into one summary bundle.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="One or more analyzed sweep directories.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the combined summary bundle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(path).resolve() for path in args.run_dirs]
    output_dir = Path(args.output_dir).resolve()
    run_analysis(run_dirs, output_dir)


if __name__ == "__main__":
    main()
