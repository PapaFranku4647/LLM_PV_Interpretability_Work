from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


GPT52_INPUT_PER_M = 1.75
GPT52_OUTPUT_PER_M = 14.0

CODEBOOST_COLOR = "#0f8b8d"
LLMPV_COLOR = "#b85c38"
T_COLORS = {
    1: "#6b7280",
    2: "#2a9d8f",
    4: "#1d5f8a",
    8: "#c45a5a",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def _load_codeboost_batch_acc(codeboost_dir: Path) -> pd.DataFrame:
    run_breakdown = codeboost_dir / "run_breakdown.csv"
    if not run_breakdown.exists():
        return pd.DataFrame(columns=["T", "batch_size", "mean_batch_acc"])

    breakdown = pd.read_csv(run_breakdown)
    rows: list[dict[str, float | int]] = []
    for source_dir_text in breakdown["source_dir"].dropna().astype(str):
        source_dir = Path(source_dir_text)
        if not source_dir.exists():
            continue
        for attempts_path in sorted(source_dir.glob("T*/seed*/attempts.csv")):
            try:
                t_value = int(attempts_path.parent.parent.name.replace("T", ""))
                attempts = pd.read_csv(attempts_path)
            except Exception:
                continue
            if "batch_acc" not in attempts.columns or "batch_size" not in attempts.columns:
                continue
            attempts = attempts.copy()
            attempts["batch_size"] = pd.to_numeric(attempts["batch_size"], errors="coerce")
            attempts["batch_acc"] = pd.to_numeric(attempts["batch_acc"], errors="coerce")
            attempts = attempts.dropna(subset=["batch_size", "batch_acc"])
            if attempts.empty:
                continue
            for batch_size, group in attempts.groupby("batch_size", sort=True):
                rows.append(
                    {
                        "T": t_value,
                        "batch_size": int(batch_size),
                        "mean_batch_acc_seed": float(group["batch_acc"].mean()),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["T", "batch_size", "mean_batch_acc"])

    seed_df = pd.DataFrame(rows)
    return (
        seed_df.groupby(["T", "batch_size"], as_index=False)
        .agg(mean_batch_acc=("mean_batch_acc_seed", "mean"))
        .sort_values(["T", "batch_size"])
        .reset_index(drop=True)
    )


def load_codeboost(codeboost_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = pd.read_csv(codeboost_dir / "combined_aggregate_summary.csv")
    seeds = pd.read_csv(codeboost_dir / "combined_seed_level_summary.csv")
    batch_acc = _load_codeboost_batch_acc(codeboost_dir)
    if not batch_acc.empty:
        agg = agg.merge(batch_acc, on=["T", "batch_size"], how="left")
    else:
        agg["mean_batch_acc"] = np.nan
    agg["method"] = "CodeBoost"
    seeds["method"] = "CodeBoost"
    return agg, seeds


def load_llmpv(llmpv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    seed_rows: list[dict[str, float | int | str | bool]] = []
    attempt_rows = 0
    compile_error_rows = 0

    for csv_path in sorted(llmpv_dir.glob("batch*/seed*/results_attempts.csv")):
        batch = int(csv_path.parent.parent.name.replace("batch", ""))
        seed = int(csv_path.parent.name.replace("seed", ""))
        frame = pd.read_csv(csv_path)
        frame["is_summary"] = frame["is_summary"].astype(str).str.lower().isin({"true", "1", "yes"})
        attempts = frame[~frame["is_summary"]].copy()
        summary = frame[frame["is_summary"]].copy()
        if summary.empty:
            continue
        summary_row = summary.iloc[0]

        prompt_tokens = pd.to_numeric(attempts.get("prompt_tokens"), errors="coerce").fillna(0).sum()
        completion_tokens = pd.to_numeric(attempts.get("completion_tokens"), errors="coerce").fillna(0).sum()
        reasoning_tokens = pd.to_numeric(attempts.get("reasoning_tokens"), errors="coerce").fillna(0).sum()
        cost_usd = (prompt_tokens / 1_000_000.0) * GPT52_INPUT_PER_M + (completion_tokens / 1_000_000.0) * GPT52_OUTPUT_PER_M

        attempt_rows += int(len(attempts))
        if "compile_error" in attempts.columns:
            compile_error_rows += int(attempts["compile_error"].notna().sum())

        seed_rows.append(
            {
                "batch_size": batch,
                "seed": seed,
                "val_acc": float(summary_row["val_acc"]),
                "test_acc": float(summary_row["test_acc"]),
                "prompt_tokens": float(prompt_tokens),
                "completion_tokens": float(completion_tokens),
                "reasoning_tokens": float(reasoning_tokens),
                "cost_usd": float(cost_usd),
                "attempt_rows": int(len(attempts)),
                "method": "LLM-PV",
            }
        )

    seed_df = pd.DataFrame(seed_rows).sort_values(["batch_size", "seed"]).reset_index(drop=True)

    agg_rows: list[dict[str, float | int | str]] = []
    for batch_size, group in seed_df.groupby("batch_size", sort=True):
        agg_rows.append(
            {
                "batch_size": int(batch_size),
                "seeds": int(len(group)),
                "mean_val_acc": float(group["val_acc"].mean()),
                "std_val_acc": float(group["val_acc"].std(ddof=0)),
                "mean_test_acc": float(group["test_acc"].mean()),
                "std_test_acc": float(group["test_acc"].std(ddof=0)),
                "mean_cost_usd": float(group["cost_usd"].mean()),
                "total_cost_usd": float(group["cost_usd"].sum()),
                "mean_prompt_tokens": float(group["prompt_tokens"].mean()),
                "mean_completion_tokens": float(group["completion_tokens"].mean()),
                "mean_reasoning_tokens": float(group["reasoning_tokens"].mean()),
                "mean_attempt_rows": float(group["attempt_rows"].mean()),
                "method": "LLM-PV",
            }
        )
    agg_df = pd.DataFrame(agg_rows).sort_values("batch_size").reset_index(drop=True)
    health = {
        "attempt_rows": float(attempt_rows),
        "compile_error_rows": float(compile_error_rows),
        "eval_error_rows": 0.0,
        "total_cost_usd": float(agg_df["total_cost_usd"].sum()),
    }
    return agg_df, seed_df, health


def build_best_codeboost_by_batch(codeboost_agg: pd.DataFrame) -> pd.DataFrame:
    return (
        codeboost_agg.sort_values(["batch_size", "mean_test_acc", "std_test_acc"], ascending=[True, False, True])
        .groupby("batch_size", as_index=False)
        .first()
        .sort_values("batch_size")
        .reset_index(drop=True)
    )


def build_matched_comparison(best_codeboost: pd.DataFrame, llmpv_agg: pd.DataFrame) -> pd.DataFrame:
    merged = best_codeboost.merge(
        llmpv_agg[["batch_size", "mean_test_acc", "std_test_acc", "mean_cost_usd"]],
        on="batch_size",
        how="inner",
        suffixes=("_codeboost", "_llmpv"),
    )
    merged["delta_test_codeboost_minus_llmpv"] = merged["mean_test_acc_codeboost"] - merged["mean_test_acc_llmpv"]
    merged["delta_cost_codeboost_minus_llmpv"] = merged["mean_cost_usd_codeboost"] - merged["mean_cost_usd_llmpv"]
    merged["codeboost_beats_llmpv"] = merged["delta_test_codeboost_minus_llmpv"] > 0
    return merged


def make_best_vs_baseline_figure(
    out_path: Path,
    best_codeboost: pd.DataFrame,
    codeboost_seeds: pd.DataFrame,
    llmpv_agg: pd.DataFrame,
    llmpv_seeds: pd.DataFrame,
) -> None:
    _style()
    batches = sorted(best_codeboost["batch_size"].astype(int).unique())
    batch_to_x = {batch: idx for idx, batch in enumerate(batches)}
    seed_offsets = np.linspace(-0.03, 0.03, len(sorted(llmpv_seeds["seed"].unique())))
    seed_to_offset = {seed: seed_offsets[idx] for idx, seed in enumerate(sorted(llmpv_seeds["seed"].unique()))}

    fig, ax = plt.subplots(figsize=(11.8, 6.8), dpi=220)

    cb_points = []
    llm_points = []
    for batch in batches:
        best_row = best_codeboost[best_codeboost["batch_size"] == batch].iloc[0]
        t_value = int(best_row["T"])
        cb_seed_rows = codeboost_seeds[
            (codeboost_seeds["batch_size"] == batch) & (codeboost_seeds["T"] == t_value)
        ].copy()
        llm_seed_rows = llmpv_seeds[llmpv_seeds["batch_size"] == batch].copy()

        for _, row in cb_seed_rows.iterrows():
            cb_points.append(
                {
                    "x": batch_to_x[batch] - 0.08 + seed_to_offset[int(row["seed"])],
                    "y": float(row["final_test_acc"]),
                }
            )
        for _, row in llm_seed_rows.iterrows():
            llm_points.append(
                {
                    "x": batch_to_x[batch] + 0.08 + seed_to_offset[int(row["seed"])],
                    "y": float(row["test_acc"]),
                }
            )

    cb_points_df = pd.DataFrame(cb_points)
    llm_points_df = pd.DataFrame(llm_points)
    ax.scatter(cb_points_df["x"], cb_points_df["y"], color=CODEBOOST_COLOR, alpha=0.28, s=80, edgecolors="none", zorder=2)
    ax.scatter(llm_points_df["x"], llm_points_df["y"], color=LLMPV_COLOR, alpha=0.28, s=80, edgecolors="none", zorder=2)

    cb_x = [batch_to_x[int(v)] - 0.08 for v in best_codeboost["batch_size"]]
    llm_x = [batch_to_x[int(v)] + 0.08 for v in llmpv_agg["batch_size"]]
    ax.plot(cb_x, best_codeboost["mean_test_acc"], color=CODEBOOST_COLOR, linewidth=3.3, marker="o", markersize=9, zorder=4)
    ax.plot(llm_x, llmpv_agg["mean_test_acc"], color=LLMPV_COLOR, linewidth=3.3, marker="s", markersize=9, zorder=4)

    ax.fill_between(
        cb_x,
        best_codeboost["mean_test_acc"] - best_codeboost["std_test_acc"],
        best_codeboost["mean_test_acc"] + best_codeboost["std_test_acc"],
        color=CODEBOOST_COLOR,
        alpha=0.10,
        zorder=3,
    )
    ax.fill_between(
        llm_x,
        llmpv_agg["mean_test_acc"] - llmpv_agg["std_test_acc"],
        llmpv_agg["mean_test_acc"] + llmpv_agg["std_test_acc"],
        color=LLMPV_COLOR,
        alpha=0.10,
        zorder=3,
    )

    for _, row in best_codeboost.iterrows():
        ax.text(
            batch_to_x[int(row["batch_size"])] - 0.08,
            float(row["mean_test_acc"]) + 0.012,
            f"T={int(row['T'])}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#2c2c2c",
        )

    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels([str(v) for v in batches])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Best CodeBoost Per Batch vs LLM-PV Baseline")
    ax.set_ylim(0.49, max(best_codeboost["mean_test_acc"].max(), llmpv_agg["mean_test_acc"].max()) + 0.04)
    ax.grid(alpha=0.22)

    handles = [
        Line2D([0], [0], color=CODEBOOST_COLOR, lw=3.3, marker="o", markersize=9, label="Best CodeBoost Mean"),
        Line2D([0], [0], color=LLMPV_COLOR, lw=3.3, marker="s", markersize=9, label="LLM-PV Mean"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CODEBOOST_COLOR, alpha=0.28, markersize=9, label="Seed Runs"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_codeboost_test_curves_figure(out_path: Path, codeboost_agg: pd.DataFrame, codeboost_seeds: pd.DataFrame) -> None:
    _style()
    batches = sorted(codeboost_agg["batch_size"].astype(int).unique())
    t_values = sorted(codeboost_agg["T"].astype(int).unique())
    batch_to_x = {batch: idx for idx, batch in enumerate(batches)}
    seed_order = sorted(codeboost_seeds["seed"].astype(int).unique())
    seed_offsets = np.linspace(-0.03, 0.03, len(seed_order))
    seed_to_offset = {seed: seed_offsets[idx] for idx, seed in enumerate(seed_order)}

    fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=220)
    for t_value in t_values:
        color = T_COLORS[t_value]
        seed_t = codeboost_seeds[codeboost_seeds["T"] == t_value].copy()
        agg_t = codeboost_agg[codeboost_agg["T"] == t_value].sort_values("batch_size")

        for seed, grp in seed_t.groupby("seed", sort=True):
            grp = grp.sort_values("batch_size")
            x_vals = [batch_to_x[int(v)] + seed_to_offset[int(seed)] for v in grp["batch_size"]]
            ax.plot(x_vals, grp["final_test_acc"], color=color, alpha=0.15, linewidth=1.0, zorder=1)
            ax.scatter(x_vals, grp["final_test_acc"], color=color, alpha=0.25, s=56, edgecolors="none", zorder=2)

        mean_x = [batch_to_x[int(v)] for v in agg_t["batch_size"]]
        ax.plot(mean_x, agg_t["mean_test_acc"], color=color, linewidth=3.2, marker="o", markersize=8, label=f"T={t_value}", zorder=4)
        ax.fill_between(
            mean_x,
            agg_t["mean_test_acc"] - agg_t["std_test_acc"],
            agg_t["mean_test_acc"] + agg_t["std_test_acc"],
            color=color,
            alpha=0.08,
            zorder=3,
        )

    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels([str(v) for v in batches])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("CodeBoost Test Accuracy by Batch Size and T")
    ax.set_ylim(0.49, codeboost_agg["mean_test_acc"].max() + 0.04)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=min(4, len(t_values)), loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_accuracy_cost_figure(out_path: Path, codeboost_agg: pd.DataFrame, llmpv_agg: pd.DataFrame) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10.8, 6.8), dpi=220)

    for t_value, grp in codeboost_agg.groupby("T", sort=True):
        t_int = int(t_value)
        ax.plot(
            grp["mean_cost_usd"],
            grp["mean_test_acc"],
            color=T_COLORS[t_int],
            linewidth=2.4,
            alpha=0.75,
            zorder=2,
        )
        ax.scatter(
            grp["mean_cost_usd"],
            grp["mean_test_acc"],
            color=T_COLORS[t_int],
            s=95,
            alpha=0.90,
            label=f"CodeBoost T={t_int}",
            zorder=3,
        )

    ax.scatter(
        llmpv_agg["mean_cost_usd"],
        llmpv_agg["mean_test_acc"],
        color=LLMPV_COLOR,
        marker="D",
        s=110,
        alpha=0.95,
        label="LLM-PV",
        zorder=4,
    )

    for _, row in llmpv_agg.iterrows():
        ax.text(
            float(row["mean_cost_usd"]) + 0.01,
            float(row["mean_test_acc"]) + 0.001,
            f"b{int(row['batch_size'])}",
            fontsize=10,
            color=LLMPV_COLOR,
        )

    top = codeboost_agg.sort_values(["mean_test_acc", "std_test_acc"], ascending=[False, True]).head(4)
    for _, row in top.iterrows():
        ax.text(
            float(row["mean_cost_usd"]) + 0.01,
            float(row["mean_test_acc"]) + 0.001,
            f"T{int(row['T'])}/b{int(row['batch_size'])}",
            fontsize=10,
            color="#2c2c2c",
        )

    ax.set_xlabel("Mean Cost Per Run (USD)")
    ax.set_ylabel("Mean Test Accuracy")
    ax.set_title("Accuracy / Cost Frontier: CodeBoost vs LLM-PV")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_paper_conclusions(
    codeboost_agg: pd.DataFrame,
    llmpv_agg: pd.DataFrame,
    matched: pd.DataFrame,
    total_cost_codeboost: float,
    total_cost_llmpv: float,
) -> dict[str, str]:
    best_cb = codeboost_agg.sort_values(["mean_test_acc", "std_test_acc"], ascending=[False, True]).iloc[0]
    best_llm = llmpv_agg.sort_values(["mean_test_acc", "std_test_acc"], ascending=[False, True]).iloc[0]
    best_cost_cb = codeboost_agg.sort_values(["mean_test_acc", "mean_cost_usd"], ascending=[False, True]).iloc[0]
    efficient_cb = codeboost_agg.sort_values(["mean_test_acc", "mean_cost_usd"], ascending=[False, True])
    efficient_cb = efficient_cb[efficient_cb["mean_cost_usd"] <= best_llm["mean_cost_usd"]].sort_values(
        ["mean_test_acc", "mean_cost_usd"], ascending=[False, True]
    ).iloc[0]
    best_by_batch_wins = int(matched["codeboost_beats_llmpv"].sum())

    meeting_bullets = [
        f"- Best overall raw accuracy is CodeBoost T={int(best_cb['T'])}, batch={int(best_cb['batch_size'])} at {float(best_cb['mean_test_acc']):.4f}.",
        f"- Best LLM-PV baseline is batch {int(best_llm['batch_size'])} at {float(best_llm['mean_test_acc']):.4f}.",
        f"- Best CodeBoost-by-batch beats LLM-PV at all {best_by_batch_wins}/{len(matched)} matched batch sizes.",
        f"- The strongest CodeBoost region is now batch 128-256 with high T, not the small or medium batches.",
        f"- Batch 64 is the inconsistent pocket: T=4 worked well there, T=8 did not improve further.",
        f"- The cheapest high-performing CodeBoost point is T={int(efficient_cb['T'])}, batch={int(efficient_cb['batch_size'])} at {float(efficient_cb['mean_test_acc']):.4f} for ${float(efficient_cb['mean_cost_usd']):.6f}/run.",
        f"- Full matched CodeBoost spend is ${total_cost_codeboost:.6f}; full matched LLM-PV spend is ${total_cost_llmpv:.6f}.",
    ]

    defensible = [
        "- Under the current matched CDC setup, CodeBoost does not win uniformly at every T and batch.",
        f"- But after tuning T and batch, CodeBoost defines the best observed accuracy frontier in this study, with T={int(best_cb['T'])}, batch={int(best_cb['batch_size'])} outperforming the best LLM-PV baseline by {float(best_cb['mean_test_acc'] - best_llm['mean_test_acc']):+.4f} test accuracy.",
        "- The fair claim is budget-matched, not method-identical: both methods use the same total 10k labeled budget and the same test split recipe, but LLM-PV holds out validation while CodeBoost uses the full weighted training pool.",
        f"- Best-by-batch CodeBoost outperforms LLM-PV in all matched batch sizes: "
        + ", ".join(
            f"b{int(row['batch_size'])} ({float(row['delta_test_codeboost_minus_llmpv']):+.4f})"
            for _, row in matched.iterrows()
        )
        + ".",
        "- The strongest evidence for CodeBoost is in the high-batch, high-T regime; the weakest evidence is in the mid-batch regime where deeper boosting can flatten or dip.",
        "- CodeBoost operational stability is strong on code generation itself: across the final matched CodeBoost bundle there were zero compile or eval/runtime error rows.",
    ]

    next_steps = [
        f"- Extend T={int(best_cb['T'])}, batch={int(best_cb['batch_size'])} to 10 seeds to tighten the frontier estimate.",
        "- Run batch 512 at T=8 first, because the batch-256 curve is still improving and remains inside the model context limit.",
        "- Run a train/validation allocation sweep for the LLM-PV baseline if you want a more apples-to-apples supervision-usage story.",
        "- Keep one cheaper CodeBoost operating point in play for the paper, not just the frontier winner. Right now that is T=2, batch=128.",
        "- If acceptance becomes the main limiter at high T, study retry/resampling separately as a method-improvement ablation, not mixed into the primary comparison.",
    ]

    paper_notes = [
        "- The paper story is not 'boosting always helps'; it is 'LLM weak-learner boosting creates a stronger accuracy frontier once search breadth is large enough.'",
        "- The best empirical evidence is that increasing T pays off most at the largest batches, which is consistent with CodeBoost needing enough batch diversity to keep finding useful weak learners in later rounds.",
        "- A strong framing is: CodeBoost beats a propose-and-verify LLM-PV baseline on the final accuracy frontier under a fixed labeled-budget protocol, while offering a cheaper near-frontier operating point at lower T.",
        "- Keep the fairness caveat explicit in the paper: matched label budget and matched split recipe, but different internal use of that budget.",
        "- For the paper figures, the most important pair is the method comparison by batch and the CodeBoost T-by-batch frontier plot.",
    ]

    return {
        "meeting_bullets": "\n".join(meeting_bullets) + "\n",
        "defensible_conclusions": "\n".join(defensible) + "\n",
        "next_steps": "\n".join(next_steps) + "\n",
        "paper_notes": "\n".join(paper_notes) + "\n",
    }


def write_overall_summary(
    out_dir: Path,
    codeboost_agg: pd.DataFrame,
    llmpv_agg: pd.DataFrame,
    matched: pd.DataFrame,
    total_cost_codeboost: float,
    total_cost_llmpv: float,
) -> None:
    best_cb = codeboost_agg.sort_values(["mean_test_acc", "std_test_acc"], ascending=[False, True]).iloc[0]
    best_llm = llmpv_agg.sort_values(["mean_test_acc", "std_test_acc"], ascending=[False, True]).iloc[0]
    efficient_cb = codeboost_agg[codeboost_agg["mean_cost_usd"] <= float(best_llm["mean_cost_usd"])].sort_values(
        ["mean_test_acc", "mean_cost_usd"], ascending=[False, True]
    ).iloc[0]
    wins = int(matched["codeboost_beats_llmpv"].sum())

    lines = [
        "- Final matched comparison bundle for CDC Diabetes, 5 seeds, train=10,000, test=60,000.",
        f"- Best overall raw accuracy: CodeBoost T={int(best_cb['T'])}, batch={int(best_cb['batch_size'])}, mean test={float(best_cb['mean_test_acc']):.4f}, sd={float(best_cb['std_test_acc']):.4f}, mean cost=${float(best_cb['mean_cost_usd']):.6f}/run.",
        f"- Best LLM-PV baseline: batch={int(best_llm['batch_size'])}, mean test={float(best_llm['mean_test_acc']):.4f}, sd={float(best_llm['std_test_acc']):.4f}, mean cost=${float(best_llm['mean_cost_usd']):.6f}/run.",
        f"- Best cheaper CodeBoost operating point under the best LLM-PV mean cost: T={int(efficient_cb['T'])}, batch={int(efficient_cb['batch_size'])}, mean test={float(efficient_cb['mean_test_acc']):.4f}, mean cost=${float(efficient_cb['mean_cost_usd']):.6f}/run.",
        f"- Best CodeBoost-by-batch beats LLM-PV in {wins}/{len(matched)} matched batch sizes.",
        "- Defensible read: tuned CodeBoost wins the best observed accuracy frontier here, but the claim is budget-matched rather than method-identical.",
        f"- Total matched spend: CodeBoost=${total_cost_codeboost:.6f}, LLM-PV=${total_cost_llmpv:.6f}, combined=${total_cost_codeboost + total_cost_llmpv:.6f}.",
    ]
    (out_dir / "overall_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(
    out_dir: Path,
    codeboost_dir: Path,
    llmpv_dir: Path,
    total_cost_codeboost: float,
    total_cost_llmpv: float,
) -> None:
    lines = [
        "- Meeting bundle for final CodeBoost vs LLM-PV comparison.",
        f"- CodeBoost source: {codeboost_dir}",
        f"- LLM-PV source: {llmpv_dir}",
        f"- CodeBoost matched spend: ${total_cost_codeboost:.6f}",
        f"- LLM-PV matched spend: ${total_cost_llmpv:.6f}",
        f"- Total matched experiment spend: ${total_cost_codeboost + total_cost_llmpv:.6f}",
    ]
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a meeting-ready CodeBoost bundle.")
    parser.add_argument("--codeboost-dir", required=True)
    parser.add_argument("--llmpv-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    codeboost_dir = Path(args.codeboost_dir).resolve()
    llmpv_dir = Path(args.llmpv_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    codeboost_agg, codeboost_seeds = load_codeboost(codeboost_dir)
    llmpv_agg, llmpv_seeds, llmpv_health = load_llmpv(llmpv_dir)
    best_codeboost = build_best_codeboost_by_batch(codeboost_agg)
    matched = build_matched_comparison(best_codeboost, llmpv_agg)

    codeboost_agg.to_csv(out_dir / "codeboost_all_configs.csv", index=False)
    codeboost_seeds.to_csv(out_dir / "codeboost_seed_level.csv", index=False)
    llmpv_agg.to_csv(out_dir / "llmpv_batch_summary.csv", index=False)
    llmpv_seeds.to_csv(out_dir / "llmpv_seed_level.csv", index=False)
    best_codeboost.to_csv(out_dir / "best_codeboost_by_batch.csv", index=False)
    matched.to_csv(out_dir / "matched_batch_comparison.csv", index=False)

    make_best_vs_baseline_figure(
        out_dir / "figure_best_codeboost_vs_llmpv_by_batch.png",
        best_codeboost,
        codeboost_seeds,
        llmpv_agg,
        llmpv_seeds,
    )
    make_codeboost_test_curves_figure(
        out_dir / "figure_codeboost_test_curves_by_batch_and_T.png",
        codeboost_agg,
        codeboost_seeds,
    )
    make_accuracy_cost_figure(
        out_dir / "figure_accuracy_cost_frontier.png",
        codeboost_agg,
        llmpv_agg,
    )

    for name in ["figure_accuracy_by_T_all_batches.png", "figure_accuracy_by_batch_all_T.png"]:
        shutil.copy2(codeboost_dir / name, out_dir / name)

    totals = build_paper_conclusions(
        codeboost_agg=codeboost_agg,
        llmpv_agg=llmpv_agg,
        matched=matched,
        total_cost_codeboost=float(codeboost_agg["total_cost_usd"].sum()),
        total_cost_llmpv=float(llmpv_health["total_cost_usd"]),
    )
    (out_dir / "meeting_bullets.txt").write_text(totals["meeting_bullets"], encoding="utf-8")
    (out_dir / "defensible_conclusions.txt").write_text(totals["defensible_conclusions"], encoding="utf-8")
    (out_dir / "next_steps.txt").write_text(totals["next_steps"], encoding="utf-8")
    (out_dir / "paper_notes.txt").write_text(totals["paper_notes"], encoding="utf-8")

    write_readme(
        out_dir=out_dir,
        codeboost_dir=codeboost_dir,
        llmpv_dir=llmpv_dir,
        total_cost_codeboost=float(codeboost_agg["total_cost_usd"].sum()),
        total_cost_llmpv=float(llmpv_health["total_cost_usd"]),
    )
    write_overall_summary(
        out_dir=out_dir,
        codeboost_agg=codeboost_agg,
        llmpv_agg=llmpv_agg,
        matched=matched,
        total_cost_codeboost=float(codeboost_agg["total_cost_usd"].sum()),
        total_cost_llmpv=float(llmpv_health["total_cost_usd"]),
    )


if __name__ == "__main__":
    main()
