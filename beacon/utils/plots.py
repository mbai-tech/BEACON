import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from beacon.core.constants import OUTPUT_DIR

FIG_DIR = OUTPUT_DIR / "figures"
CSV_DEFAULT = OUTPUT_DIR / "beacon_metrics.csv"

ALGO_ORDER  = ["BEACON", "Bug1", "Bug2", "Greedy"]
ALGO_COLORS = {
    "BEACON": "#2196F3",
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


# ── Figure 3: battery over time (single representative BEACON trial) ─────────

def plot_fig3(df: pd.DataFrame) -> Path:
    """Simulate a battery-level time series for BEACON under Dense/Mixed using
    the aggregate stats (since step-level data is not stored in the CSV).

    A synthetic trace is constructed from:
      battery(t) = B0 - δ_move × cumulative_path(t) - δ_time × t
    approximated by drawing from the distribution of recorded metrics.
    """
    from beacon.core.constants import BATTERY_INITIAL, DELTA_MOVE, DELTA_TIME, DELTA_COL

    beacon_dm = df[
        (df["algorithm"] == "BEACON") &
        (df["density"] == "dense") &
        (df["fragility_profile"] == "mixed") &
        (df["success"] == True)
    ]

    if len(beacon_dm) == 0:
        # Fall back to any BEACON successful trial
        beacon_dm = df[(df["algorithm"] == "BEACON") & (df["success"] == True)]

    if len(beacon_dm) == 0:
        print("  [fig3] no successful BEACON trials — skipping")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No successful BEACON trials in data",
                ha="center", va="center", transform=ax.transAxes)
        return _save(fig, "fig3_battery_over_time.png")

    # Use the median trial for the representative trace
    row = beacon_dm.iloc[len(beacon_dm) // 2]
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
    ax.plot(t_axis, battery, color=ALGO_COLORS["BEACON"], lw=2,
            label="BEACON (Dense/Mixed)")
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
    ax.set_title("Battery Level Over Time — BEACON representative trial (Dense/Mixed)")
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


# ── Figure 5: path-length vs success-rate trade-off scatter ──────────────────

def plot_tradeoff_scatter() -> Path:
    """Path length vs success rate trade-off for all planners, with Pareto frontier."""
    planners = [
        ("Bug",               9.604, 82.25),
        ("Bug2",              8.055, 91.25),
        ("D* Lite",           8.387, 97.25),
        ("Baseline Semantic", 7.898, 89.25),
        ("BEACON",            5.053, 96.75),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    pareto_x = [4.0, 5.053, 5.053, 8.387, 8.387, 8.387]
    pareto_y = [96.75, 96.75, 97.25, 97.25, 97.25, 100.5]
    ax.step(pareto_x, pareto_y, where='post', color='#D85A30', linestyle='--',
            linewidth=1.8, label='Pareto frontier', zorder=2)

    shade_x = [4.0, 5.053, 5.053, 8.387, 8.387, 4.0, 4.0]
    shade_y = [96.75, 96.75, 97.25, 97.25, 100.5, 100.5, 96.75]
    ax.fill(shade_x, shade_y, color='#1D9E75', alpha=0.08, zorder=1)
    ax.plot([4.0, 5.053, 5.053, 8.387, 8.387],
            [96.75, 96.75, 97.25, 97.25, 100.5],
            color='#1D9E75', linewidth=1.0, alpha=0.4, zorder=1)

    for name, x, y in planners:
        if name == "BEACON":
            color, edge, size, zorder = '#1D9E75', '#0F6E56', 120, 5
            marker = '*'
        elif name == "Baseline Semantic":
            color, edge, size, zorder = '#888780', '#5F5E5A', 70, 4
            marker = 'o'
        else:
            color, edge, size, zorder = '#378ADD', '#185FA5', 70, 4
            marker = 'o'
        ax.scatter(x, y, color=color, edgecolors=edge, s=size,
                   marker=marker, linewidths=1.5, zorder=zorder)

    offsets = {
        "Bug":               {"dx":  0.1,  "dy": -2.2, "ha": "left"},
        "Bug2":              {"dx":  0.1,  "dy":  1.5, "ha": "left"},
        "D* Lite":           {"dx":  0.1,  "dy": -2.2, "ha": "left"},
        "Baseline Semantic": {"dx":  0.1,  "dy": -2.2, "ha": "left"},
        "BEACON":            {"dx": -0.2,  "dy": -2.2, "ha": "right"},
    }
    connector = dict(arrowstyle='-', color='#BBBBBB', lw=0.7, shrinkA=0, shrinkB=4)

    for name, x, y in planners:
        cfg = offsets[name]
        color = '#0F6E56' if name == "BEACON" else '#444441'
        weight = 'bold' if name == "BEACON" else 'normal'
        ax.annotate(name, xy=(x, y), xytext=(x + cfg["dx"], y + cfg["dy"]),
                    fontsize=10, color=color, fontweight=weight, ha=cfg["ha"],
                    arrowprops=connector)

    ax.set_xlim(10.8, 4.0)
    ax.set_ylim(78, 100)
    ax.set_xlabel('Avg. path length (m)', fontsize=11)
    ax.set_ylabel('Success rate (%)', fontsize=11)
    ax.grid(True, color='#D3D1C7', linewidth=0.5, linestyle='-', alpha=0.7)
    ax.set_axisbelow(True)

    legend_elements = [
        mpatches.Patch(facecolor='#1D9E75', alpha=0.15, edgecolor='#1D9E75', label='Optimal region'),
        plt.Line2D([0], [0], color='#D85A30', linestyle='--', linewidth=1.8, label='Pareto frontier'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#378ADD',
                   markeredgecolor='#185FA5', markersize=8, label='Classical planners'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#888780',
                   markeredgecolor='#5F5E5A', markersize=8, label='Baseline semantic'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#1D9E75',
                   markeredgecolor='#0F6E56', markersize=12, label='BEACON (proposed)'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower left',
              framealpha=0.9, edgecolor='#D3D1C7')

    fig.tight_layout()
    return _save(fig, "fig5_tradeoff_scatter.png")


# ── Figure 6: algorithm illustration ─────────────────────────────────────────

def plot_algorithm_illustration() -> Path:
    """Three-panel overview: robot scene, per-step decision loop, CIBP belief update."""
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 3, wspace=0.42)
    ax_s = fig.add_subplot(gs[0])
    ax_f = fig.add_subplot(gs[1])
    ax_b = fig.add_subplot(gs[2])

    for ax in (ax_s, ax_f, ax_b):
        ax.set_facecolor('#FAFAF8')

    # ── helpers ──────────────────────────────────────────────────────────────

    def fbox(ax, cx, cy, w, h, text, fc='#E3F2FD', ec='#1565C0', fs=8.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle='round,pad=0.12', facecolor=fc, edgecolor=ec,
            linewidth=1.5, zorder=3))
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='#1A1A1A', multialignment='center', zorder=4)

    def diam(ax, cx, cy, w, h, text, fc='#FFF9C4', ec='#F9A825', fs=8):
        verts = [(cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2), (cx - w/2, cy)]
        ax.add_patch(mpatches.Polygon(
            verts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3))
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='#1A1A1A', multialignment='center', zorder=4)

    def varr(ax, x, y1, y2):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5), zorder=2)

    def harr(ax, x1, x2, y, label='', above=True):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5), zorder=2)
        if label:
            ax.text((x1 + x2) / 2, y + (0.28 if above else -0.28), label,
                    ha='center', fontsize=8, color='#555', fontweight='bold')

    # ── Panel 1: Scene ───────────────────────────────────────────────────────
    ax = ax_s
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.add_patch(mpatches.Rectangle(
        (0.2, 0.2), 9.6, 8.6, fill=False, edgecolor='#555', linewidth=2.5))

    # goal
    ax.scatter([8.5], [7.5], marker='*', s=400, color='#FFB300',
               edgecolors='#E65100', linewidths=1.5, zorder=6)
    ax.text(8.5, 8.1, 'Goal', ha='center', fontsize=9,
            color='#E65100', fontweight='bold')

    # wall (fixed)
    ax.add_patch(mpatches.Rectangle(
        (3.8, 0.4), 0.45, 4.0, facecolor='#9E9E9E', edgecolor='#555', linewidth=1.5))
    ax.text(4.02, 4.7, 'wall\n(fixed)', ha='center', fontsize=7.5, color='#555')

    # movable obstacle
    ax.add_patch(mpatches.Rectangle(
        (5.2, 2.8), 1.3, 1.3, facecolor='#C8E6C9', edgecolor='#2E7D32',
        linewidth=2.0, zorder=2))
    ax.text(5.85, 3.45, 'movable', ha='center', fontsize=8,
            color='#1B5E20', fontweight='bold')

    # epsilon contact band
    ax.add_patch(mpatches.FancyBboxPatch(
        (4.98, 2.58), 1.74, 1.74, boxstyle='round,pad=0.20',
        fill=False, edgecolor='#2E7D32', linewidth=1.0, linestyle=':', zorder=3))
    ax.text(4.97, 2.45, 'ε contact band', fontsize=6.5, color='#2E7D32')

    # fragile obstacle
    ax.add_patch(mpatches.Rectangle(
        (7.0, 5.8), 1.1, 1.1, facecolor='#FFCDD2', edgecolor='#C62828',
        linewidth=2.0, zorder=2))
    ax.text(7.55, 6.45, 'fragile', ha='center', fontsize=8,
            color='#B71C1C', fontweight='bold')

    # path — goal-seek (solid blue)
    p1 = np.array([[0.8, 0.8], [2.0, 1.2], [3.4, 2.2]])
    ax.plot(p1[:, 0], p1[:, 1], color='#1565C0', lw=2.5, solid_capstyle='round', zorder=2)

    # path — boundary-follow (dashed orange)
    p2 = np.array([[3.4, 2.2], [3.7, 3.2], [3.9, 4.4], [4.3, 5.3]])
    ax.plot(p2[:, 0], p2[:, 1], color='#E65100', lw=2.5,
            linestyle='--', solid_capstyle='round', zorder=2)

    # path — goal-seek resumes
    p3 = np.array([[4.3, 5.3], [4.9, 4.5], [5.4, 3.9]])
    ax.plot(p3[:, 0], p3[:, 1], color='#1565C0', lw=2.5, solid_capstyle='round', zorder=2)

    # robot
    rx, ry, sr = 5.4, 3.9, 1.8
    for ang in np.linspace(0, 2 * np.pi, 20, endpoint=False):
        ax.plot([rx, rx + sr * np.cos(ang)], [ry, ry + sr * np.sin(ang)],
                color='#90CAF9', lw=0.5, alpha=0.45, zorder=1)
    ax.add_patch(mpatches.Circle(
        (rx, ry), sr, fill=False, edgecolor='#42A5F5',
        linewidth=1.3, linestyle='--', alpha=0.65, zorder=3))
    ax.text(rx + sr + 0.12, ry, 'sensing\nrange', fontsize=7,
            color='#1565C0', va='center')
    ax.add_patch(mpatches.Circle(
        (rx, ry), 0.28, facecolor='#1565C0', edgecolor='#0D47A1',
        linewidth=2.0, zorder=5))
    ax.annotate('', xy=(rx + 0.5, ry + 0.5), xytext=(rx, ry),
                arrowprops=dict(arrowstyle='->', color='#E3F2FD', lw=2.5), zorder=6)

    # push arrow
    ax.annotate('', xy=(6.9, 3.45), xytext=(6.5, 3.45),
                arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=2.5), zorder=5)
    ax.text(7.05, 3.45, 'push!', fontsize=9, color='#1B5E20',
            va='center', fontweight='bold')

    ax.legend(handles=[
        Line2D([0], [0], color='#1565C0', lw=2.5, label='Goal-seek mode'),
        Line2D([0], [0], color='#E65100', lw=2.5, linestyle='--',
               label='Boundary-follow mode'),
    ], fontsize=8, loc='upper left', framealpha=0.92, edgecolor='#CCC')

    ax_s.set_title('Robot–Environment Interaction', fontsize=11,
                   fontweight='bold', pad=10)

    # ── Panel 2: Flowchart ───────────────────────────────────────────────────
    ax = ax_f
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')

    CX = 5.0

    fbox(ax, CX, 15.0, 6.5, 0.9, 'PERCEIVE\nsense obstacles within range',
         fc='#E8F5E9', ec='#2E7D32')
    varr(ax, CX, 14.55, 13.85)
    fbox(ax, CX, 13.4, 6.5, 0.9, 'PLAN\nD* Lite replan (new obstacles)',
         fc='#E3F2FD', ec='#1565C0')
    varr(ax, CX, 12.95, 12.25)
    fbox(ax, CX, 11.8, 6.5, 0.9, 'CLASSIFY\npartition push set / avoid set',
         fc='#F3E5F5', ec='#6A1B9A')
    varr(ax, CX, 11.35, 10.6)
    diam(ax, CX, 10.0, 6.2, 1.3, 'Pushable obstacle\nblocking path?')

    # YES branch — right side
    harr(ax, 8.1, 9.2, 10.0, label='yes')
    ax.plot([9.2, 9.2], [10.0, 8.1], color='#555', lw=1.5)
    ax.annotate('', xy=(7.8, 8.1), xytext=(9.2, 8.1),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    fbox(ax, 6.5, 8.1, 2.6, 0.9, 'J_push vs J_avoid\n→ best action',
         fc='#E8F5E9', ec='#2E7D32', fs=8)

    # NO branch — left side
    harr(ax, 1.9, 0.8, 10.0, label='no', above=False)
    ax.plot([0.8, 0.8], [10.0, 8.1], color='#555', lw=1.5)
    ax.annotate('', xy=(2.2, 8.1), xytext=(0.8, 8.1),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    fbox(ax, 3.5, 8.1, 2.6, 0.9, 'Boundary-follow\n(Bug mode)',
         fc='#FBE9E7', ec='#BF360C', fs=8)

    # converge both branches to EXECUTE
    ax.plot([3.5, 3.5], [7.65, 7.15], color='#555', lw=1.5)
    ax.plot([6.5, 6.5], [7.65, 7.15], color='#555', lw=1.5)
    ax.plot([3.5, 6.5], [7.15, 7.15], color='#555', lw=1.5)
    ax.annotate('', xy=(CX, 6.8), xytext=(CX, 7.15),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

    fbox(ax, CX, 6.35, 6.5, 0.9, 'EXECUTE step + update battery',
         fc='#E3F2FD', ec='#1565C0')
    varr(ax, CX, 5.90, 5.20)
    fbox(ax, CX, 4.75, 6.5, 0.9, 'CIBP UPDATE\nobserve outcome → Bayesian belief',
         fc='#FFF3E0', ec='#E65100')
    varr(ax, CX, 4.30, 3.60)
    fbox(ax, CX, 3.15, 6.5, 0.9, 'KL(post ‖ prior) > τ  →  replan',
         fc='#FFF9C4', ec='#F9A825')

    # loop-back indicator
    ax.plot([1.5, 1.5], [2.7, 15.0], color='#BDBDBD', lw=1.1, linestyle=':')
    ax.plot([1.5, CX - 3.25], [15.0, 15.0], color='#BDBDBD', lw=1.1, linestyle=':')
    ax.plot([1.5, CX - 3.25], [2.7, 2.7], color='#BDBDBD', lw=1.1, linestyle=':')
    ax.text(1.0, 8.8, 'next\nstep', fontsize=7.5, color='#BDBDBD',
            ha='center', va='center', rotation=90)

    ax_f.set_title('Per-Step Decision Loop', fontsize=11, fontweight='bold', pad=10)

    # ── Panel 3: CIBP Belief Update ──────────────────────────────────────────
    ax = ax_b

    classes = ['safe', 'movable', 'fragile', 'forbidden']
    clr = ['#90A4AE', '#66BB6A', '#EF9A9A', '#B71C1C']
    prior             = [0.25, 0.25, 0.25, 0.25]
    after_displace    = [0.05, 0.76, 0.12, 0.07]
    after_no_displace = [0.62, 0.11, 0.14, 0.13]

    x = np.arange(len(classes))
    w = 0.26
    ax.bar(x - w, prior, w, label='Prior (uniform)',
           color='#CFD8DC', edgecolor='#607D8B', linewidth=1.2)
    ax.bar(x, after_displace, w,
           label='After displacement\n(object moved → movable)',
           color=clr, edgecolor='#333', linewidth=1.0, alpha=0.90)
    ax.bar(x + w, after_no_displace, w,
           label='After no displacement\n(object stayed → safe)',
           color=clr, edgecolor='#333', linewidth=1.0, alpha=0.55, hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9.5)
    ax.set_ylabel('Belief probability', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.35, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(0.97, 0.97, 'KL(posterior ‖ prior) > τ\n→ trigger replan',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF9C4',
                      edgecolor='#F9A825', alpha=0.95))
    ax.legend(fontsize=8, loc='upper center', framealpha=0.9, edgecolor='#CCC',
              bbox_to_anchor=(0.5, 0.99), ncol=1)

    ax_b.set_title('CIBP: Bayesian Belief Update\non Contact Outcome',
                   fontsize=11, fontweight='bold', pad=10)

    fig.suptitle('BEACON Algorithm Overview', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, 'fig_algorithm_illustration.png')


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

    print("Figure 3: battery over time (representative BEACON trial)")
    plot_fig3(df)

    print("Figure 4: planning time per algorithm")
    plot_fig4(df)

    print("Figure 5: path-length vs success-rate trade-off scatter")
    plot_tradeoff_scatter()

    print("Figure 6: algorithm illustration")
    plot_algorithm_illustration()

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT,
                        help="path to beacon_metrics.csv")
    args = parser.parse_args()
    main(args.csv)
