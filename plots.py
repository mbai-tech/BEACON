"""SCHOLAR result plots (Figures 1–4 from the paper).

Usage
-----
    python plots.py                                   # reads scholar_metrics.csv
    python plots.py --csv path/to/custom.csv

Outputs (saved to NewProject/outputs/figures/)
----------------------------------------------
    fig1_success_battery_vs_density.png
    fig2_uniform_vs_mixed_medium.png
    fig3_battery_over_time.png       (single SCHOLAR Dense/Mixed trial)
    fig4_planning_time_table.png     (per-algorithm planning time)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from NewProject.constants import OUTPUT_DIR

FIG_DIR = OUTPUT_DIR / "figures"
CSV_DEFAULT = OUTPUT_DIR / "scholar_metrics.csv"

ALGO_ORDER  = ["SCHOLAR", "Bug1", "Bug2", "Greedy"]
ALGO_COLORS = {
    "SCHOLAR": "#2196F3",
    "Bug1":    "#FF9800",
    "Bug2":    "#4CAF50",
    "Greedy":  "#E91E63",
}
DENSITY_ORDER = ["sparse", "medium", "dense"]
DENSITY_LABELS = {"sparse": "Sparse\n(5–10)", "medium": "Medium\n(15–25)", "dense": "Dense\n(30–50)"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["success"] = df["success"].astype(bool)
    return df


def _success_rate(df: pd.DataFrame) -> float:
    return df["success"].mean() * 100.0


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")
    return path


# ── Figure 1: success rate & battery vs density (Mixed profile) ───────────────

def plot_fig1(df: pd.DataFrame) -> Path:
    """Success rate (left) and battery consumed (right) vs clutter density
    for the Mixed fragility profile, all four algorithms."""
    mixed = df[df["fragility_profile"] == "mixed"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(DENSITY_ORDER))
    width = 0.18

    for i, alg in enumerate(ALGO_ORDER):
        sub = mixed[mixed["algorithm"] == alg]
        sr, sr_err, bat, bat_err = [], [], [], []
        for density in DENSITY_ORDER:
            g = sub[sub["density"] == density]
            sr.append(_success_rate(g) if len(g) else 0.0)
            sr_err.append(g["success"].std() * 100 if len(g) else 0.0)
            suc = g[g["success"]]
            bat.append(suc["battery_consumed"].mean() if len(suc) else 0.0)
            bat_err.append(suc["battery_consumed"].std() if len(suc) else 0.0)

        offset = (i - 1.5) * width
        axes[0].bar(x + offset, sr, width, yerr=sr_err, capsize=3,
                    label=alg, color=ALGO_COLORS[alg], alpha=0.85)
        axes[1].bar(x + offset, bat, width, yerr=bat_err, capsize=3,
                    label=alg, color=ALGO_COLORS[alg], alpha=0.85)

    for ax, ylabel, title in zip(
        axes,
        ["Success Rate (%)", "Battery Consumed (units)"],
        ["Success Rate vs Density (Mixed Profile)",
         "Battery Consumed vs Density (Mixed Profile)"],
    ):
        ax.set_xticks(x)
        ax.set_xticklabels([DENSITY_LABELS[d] for d in DENSITY_ORDER])
        ax.set_xlabel("Clutter Density")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylim(0, 110)
    fig.tight_layout()
    return _save(fig, "fig1_success_battery_vs_density.png")


# ── Figure 2: Uniform vs Mixed at medium density ──────────────────────────────

def plot_fig2(df: pd.DataFrame) -> Path:
    """Compare Uniform and Mixed fragility profiles at medium density for each
    algorithm across three metrics: success rate, path length, contact cost."""
    medium = df[df["density"] == "medium"]

    metrics = [
        ("success",          "Success Rate (%)",      True),
        ("path_length_m",    "Path Length (m)",       False),
        ("contact_cost",     "Contact Cost Σc(Ok)",   False),
    ]
    profiles = ["uniform", "mixed"]
    profile_labels = {"uniform": "Uniform", "mixed": "Mixed"}
    profile_hatches = {"uniform": "", "mixed": "//"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(ALGO_ORDER))
    width = 0.35

    for ax, (metric, ylabel, is_rate) in zip(axes, metrics):
        for j, profile in enumerate(profiles):
            vals, errs = [], []
            for alg in ALGO_ORDER:
                g = medium[(medium["algorithm"] == alg) &
                           (medium["fragility_profile"] == profile)]
                if is_rate:
                    vals.append(_success_rate(g) if len(g) else 0.0)
                    errs.append(g["success"].std() * 100 if len(g) else 0.0)
                else:
                    suc = g[g["success"]]
                    vals.append(suc[metric].mean() if len(suc) else 0.0)
                    errs.append(suc[metric].std() if len(suc) else 0.0)

            offset = (j - 0.5) * width
            bars = ax.bar(
                x + offset, vals, width, yerr=errs, capsize=3,
                label=profile_labels[profile],
                color=[ALGO_COLORS[a] for a in ALGO_ORDER],
                hatch=profile_hatches[profile],
                alpha=0.80,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(ALGO_ORDER, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\n(Medium Density)")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylim(0, 110)
    # shared legend for profile hatching
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="grey", label="Uniform"),
                  Patch(facecolor="grey", hatch="//", label="Mixed")]
    fig.legend(handles=legend_els, loc="upper right", fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig2_uniform_vs_mixed_medium.png")


# ── Figure 3: battery over time (single representative SCHOLAR trial) ─────────

def plot_fig3(df: pd.DataFrame) -> Path:
    """Simulate a battery-level time series for SCHOLAR under Dense/Mixed using
    the aggregate stats (since step-level data is not stored in the CSV).

    A synthetic trace is constructed from:
      battery(t) = B0 - δ_move × cumulative_path(t) - δ_time × t
    approximated by drawing from the distribution of recorded metrics.
    """
    from NewProject.constants import BATTERY_INITIAL, DELTA_MOVE, DELTA_TIME, DELTA_COL

    scholar_dm = df[
        (df["algorithm"] == "SCHOLAR") &
        (df["density"] == "dense") &
        (df["fragility_profile"] == "mixed") &
        (df["success"] == True)
    ]

    if len(scholar_dm) == 0:
        # Fall back to any SCHOLAR successful trial
        scholar_dm = df[(df["algorithm"] == "SCHOLAR") & (df["success"] == True)]

    if len(scholar_dm) == 0:
        print("  [fig3] no successful SCHOLAR trials — skipping")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No successful SCHOLAR trials in data",
                ha="center", va="center", transform=ax.transAxes)
        return _save(fig, "fig3_battery_over_time.png")

    # Use the median trial for the representative trace
    row = scholar_dm.iloc[len(scholar_dm) // 2]
    steps      = int(row["step_count"])
    path_len   = float(row["path_length_m"])
    push_count = int(row["push_count"])

    # Assume constant speed → distance proportional to step
    dist_per_step = path_len / max(steps, 1)
    push_steps = sorted(
        np.random.choice(range(steps), size=min(push_count, steps), replace=False)
    ) if push_count > 0 else []
    push_set = set(push_steps)

    battery = [BATTERY_INITIAL]
    for t in range(steps):
        b = battery[-1]
        b -= DELTA_MOVE * dist_per_step
        b -= DELTA_TIME
        if t in push_set:
            b -= DELTA_COL
        battery.append(b)

    t_axis = np.arange(len(battery))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_axis, battery, color=ALGO_COLORS["SCHOLAR"], lw=2,
            label="SCHOLAR (Dense/Mixed)")
    ax.axhline(0, color="red", lw=1, linestyle="--", label="Battery depleted")
    ax.axhline(0.3 * BATTERY_INITIAL, color="orange", lw=1, linestyle=":",
               label=f"B_thresh = 0.3 B₀")

    # Mark push events
    for t in push_steps:
        if t < len(battery):
            ax.axvline(t, color="grey", lw=0.6, alpha=0.5)
    if push_steps:
        ax.axvline(push_steps[0], color="grey", lw=0.6, alpha=0.5,
                   label="Push contact")

    ax.set_xlabel("Control step")
    ax.set_ylabel("Battery (units)")
    ax.set_title("Battery Level Over Time — SCHOLAR representative trial (Dense/Mixed)")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(min(0, min(battery)) - 20, BATTERY_INITIAL + 20)
    fig.tight_layout()
    return _save(fig, "fig3_battery_over_time.png")


# ── Figure 4: planning time table (Table VI equivalent) ──────────────────────

def plot_fig4(df: pd.DataFrame) -> Path:
    """Bar chart of mean per-step planning time (ms) per algorithm, mirroring
    Table VI.  Also draws the T_MAP = 200 ms real-time budget line."""
    fig, ax = plt.subplots(figsize=(7, 4))

    means, stds = [], []
    for alg in ALGO_ORDER:
        g = df[df["algorithm"] == alg]
        means.append(g["planning_time_ms"].mean())
        stds.append(g["planning_time_ms"].std())

    x = np.arange(len(ALGO_ORDER))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.5,
                  color=[ALGO_COLORS[a] for a in ALGO_ORDER], alpha=0.85)
    ax.axhline(200, color="red", lw=1.5, linestyle="--",
               label="T_map = 200 ms real-time budget")

    ax.set_xticks(x)
    ax.set_xticklabels(ALGO_ORDER)
    ax.set_ylabel("Mean per-step planning time (ms)")
    ax.set_title("Computational Overhead per Algorithm")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Annotate bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return _save(fig, "fig4_planning_time.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main(csv_path: Path) -> None:
    print(f"Loading {csv_path} ...")
    df = _load(csv_path)
    print(f"  {len(df)} rows, configs: {sorted(df['config'].unique())}, "
          f"algorithms: {sorted(df['algorithm'].unique())}")

    print("\nFigure 1: success rate & battery vs density (Mixed)")
    plot_fig1(df)

    print("Figure 2: Uniform vs Mixed at medium density")
    plot_fig2(df)

    print("Figure 3: battery over time (representative SCHOLAR trial)")
    plot_fig3(df)

    print("Figure 4: planning time per algorithm")
    plot_fig4(df)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT,
                        help="path to scholar_metrics.csv")
    args = parser.parse_args()
    main(args.csv)
