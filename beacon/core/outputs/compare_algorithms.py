"""Algorithm comparison: BEACON vs Bug1, Bug2, Greedy.

Usage:
    python beacon/core/outputs/compare_algorithms.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CSV = Path(__file__).parent / "beacon_metrics.csv"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

ALG_ORDER   = ["BEACON", "Bug2", "Greedy", "Bug1"]
ALG_COLORS  = {"BEACON": "#2196F3", "Bug1": "#F44336", "Bug2": "#FF9800", "Greedy": "#4CAF50"}
DENSITY_ORDER = ["sparse", "medium", "dense"]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["success"] = df["success"].astype(bool)
    df["density_cat"] = pd.Categorical(df["density"], categories=DENSITY_ORDER, ordered=True)
    return df


# ── 1. Overall summary table ──────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n" + "═"*80)
    print("OVERALL SUMMARY  (mean ± std across all configs)")
    print("═"*80)
    metrics = {
        "Success Rate":        ("success",           lambda s: f"{s.mean():.1%}"),
        "Path Efficiency":     ("path_efficiency",   lambda s: f"{s.mean():.3f} ± {s.std():.3f}"),
        "Battery Remaining":   ("battery_remaining", lambda s: f"{s.mean():.1f} ± {s.std():.1f}"),
        "Dangerous Contacts":  ("dangerous_contacts",lambda s: f"{s.mean():.2f} ± {s.std():.2f}"),
        "Planning Time (ms)":  ("planning_time_ms",  lambda s: f"{s.mean():.0f} ± {s.std():.0f}"),
        "Push Count":          ("push_count",        lambda s: f"{s.mean():.1f} ± {s.std():.1f}"),
    }
    header = f"{'Metric':<22}" + "".join(f"{a:>18}" for a in ALG_ORDER)
    print(header)
    print("-"*80)
    for label, (col, fmt) in metrics.items():
        row = f"{label:<22}"
        for alg in ALG_ORDER:
            s = df[df["algorithm"] == alg][col]
            row += f"{fmt(s):>18}"
        print(row)
    print()


# ── 2. Success rate by density ────────────────────────────────────────────────

def print_by_density(df: pd.DataFrame):
    print("SUCCESS RATE BY DENSITY")
    print("-"*60)
    header = f"{'Density':<10}" + "".join(f"{a:>12}" for a in ALG_ORDER)
    print(header)
    for dens in DENSITY_ORDER:
        sub = df[df["density"] == dens]
        row = f"{dens:<10}"
        for alg in ALG_ORDER:
            sr = sub[sub["algorithm"] == alg]["success"].mean()
            row += f"{sr:>11.1%} "
        print(row)
    print()

    print("PATH EFFICIENCY BY DENSITY")
    print("-"*60)
    print(header)
    for dens in DENSITY_ORDER:
        sub = df[df["density"] == dens]
        row = f"{dens:<10}"
        for alg in ALG_ORDER:
            eff = sub[sub["algorithm"] == alg]["path_efficiency"].mean()
            row += f"{eff:>11.3f} "
        print(row)
    print()


# ── 3. Figures ────────────────────────────────────────────────────────────────

def fig_success_by_config(df: pd.DataFrame):
    configs = df["config"].unique()
    x = np.arange(len(configs))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, alg in enumerate(ALG_ORDER):
        sr = [df[(df["algorithm"]==alg) & (df["config"]==c)]["success"].mean()
              for c in configs]
        ax.bar(x + i*width, sr, width, label=alg,
               color=ALG_COLORS[alg], alpha=0.85, edgecolor="white")
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Success Rate by Scene Config")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = OUT / "success_by_config.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def fig_efficiency_boxplot(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, dens in zip(axes, DENSITY_ORDER):
        sub = df[df["density"] == dens]
        data = [sub[sub["algorithm"]==alg]["path_efficiency"].values for alg in ALG_ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch, alg in zip(bp["boxes"], ALG_ORDER):
            patch.set_facecolor(ALG_COLORS[alg])
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(ALG_ORDER)+1))
        ax.set_xticklabels(ALG_ORDER, rotation=15, ha="right")
        ax.set_title(f"{dens.capitalize()} scenes")
        ax.set_ylabel("Path Efficiency" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Path Efficiency by Density", fontsize=13, y=1.01)
    plt.tight_layout()
    path = OUT / "efficiency_boxplot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def fig_dangerous_contacts(df: pd.DataFrame):
    configs = df["config"].unique()
    x = np.arange(len(configs))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, alg in enumerate(ALG_ORDER):
        dc = [df[(df["algorithm"]==alg) & (df["config"]==c)]["dangerous_contacts"].mean()
              for c in configs]
        ax.bar(x + i*width, dc, width, label=alg,
               color=ALG_COLORS[alg], alpha=0.85, edgecolor="white")
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Avg Dangerous Contacts")
    ax.set_title("Dangerous Contacts by Scene Config")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = OUT / "dangerous_contacts.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def fig_radar(df: pd.DataFrame):
    """Radar chart comparing algorithms across 4 normalised metrics."""
    from matplotlib.patches import FancyArrowPatch
    labels = ["Success\nRate", "Path\nEfficiency", "Battery\nSaved", "Safety\n(1-danger)"]
    n = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for alg in ALG_ORDER:
        sub = df[df["algorithm"]==alg]
        sr  = sub["success"].mean()
        eff = sub["path_efficiency"].mean()
        bat = sub["battery_remaining"].mean() / 1000
        # normalise dangerous contacts: 0 contacts = 1.0, scale by observed max
        max_dc = df["dangerous_contacts"].max() or 1
        saf = 1 - sub["dangerous_contacts"].mean() / max_dc
        vals = [sr, min(eff, 1.0), bat, saf]
        vals += vals[:1]
        ax.plot(angles, vals, "-o", linewidth=2, label=alg, color=ALG_COLORS[alg])
        ax.fill(angles, vals, alpha=0.07, color=ALG_COLORS[alg])

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title("Algorithm Comparison (normalised)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    path = OUT / "radar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    df = load()
    print_summary(df)
    print_by_density(df)
    fig_success_by_config(df)
    fig_efficiency_boxplot(df)
    fig_dangerous_contacts(df)
    fig_radar(df)
    print("\nDone.")
