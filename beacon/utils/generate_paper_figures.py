"""Generate paper-friendly figures and tables from saved scene-complex metrics.

Outputs are written under:
    beacon/environment/data/metrics/paper_assets/
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


METRICS_DIR = Path(__file__).resolve().parent.parent / "environment" / "data" / "metrics"
OUT_DIR = METRICS_DIR / "paper_assets"

FAMILY_ORDER = ["sparse", "cluttered", "collision_required", "collision_shortcut"]
PLANNER_ORDER = ["bug", "bug2", "dstar_lite", "beacon"]
PLANNER_LABELS = {
    "bug": "Bug",
    "bug2": "Bug2",
    "dstar_lite": "D* Lite",
    "beacon": "BEACON",
    "surp": "SURP",
}
COLORS = {
    "bug": "#d95f02",
    "bug2": "#e6ab02",
    "dstar_lite": "#7570b3",
    "beacon": "#1f78b4",
    "surp": "#33a02c",
}


def discover_csvs() -> list[Path]:
    paths = sorted(METRICS_DIR.glob("metrics*_scene_complex.csv"))
    return [p for p in paths if p.name != "metrics_scene_complex_comparison.csv"]


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, int, int]] = set()
    for path in paths:
        with path.open() as f:
            for row in csv.DictReader(f):
                key = (
                    row["planner"],
                    row["family"],
                    int(row["scene_idx"]),
                    int(row["seed"]),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "planner": row["planner"],
                        "family": row["family"],
                        "scene_idx": int(row["scene_idx"]),
                        "seed": int(row["seed"]),
                        "success": row["success"] == "True",
                        "steps": int(row["steps"]),
                        "path_length": float(row["path_length"]),
                        "n_contacts": int(row["n_contacts"]),
                        "n_sensed": int(row["n_sensed"]),
                    }
                )
    return rows


def summarize(rows: list[dict]) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    overall: dict[str, dict] = {}
    by_family: dict[tuple[str, str], dict] = {}

    grouped_overall: dict[str, list[dict]] = defaultdict(list)
    grouped_family: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped_overall[row["planner"]].append(row)
        grouped_family[(row["planner"], row["family"])].append(row)

    for planner, vals in grouped_overall.items():
        n = len(vals)
        successes = [v for v in vals if v["success"]]
        overall[planner] = {
            "episodes": n,
            "success_rate": len(successes) / n if n else 0.0,
            "avg_steps": sum(v["steps"] for v in vals) / n if n else 0.0,
            "avg_path": sum(v["path_length"] for v in vals) / n if n else 0.0,
            "avg_contacts": sum(v["n_contacts"] for v in vals) / n if n else 0.0,
            "avg_sensed": sum(v["n_sensed"] for v in vals) / n if n else 0.0,
        }

    for key, vals in grouped_family.items():
        n = len(vals)
        successes = [v for v in vals if v["success"]]
        by_family[key] = {
            "episodes": n,
            "success_rate": len(successes) / n if n else 0.0,
            "avg_steps": sum(v["steps"] for v in vals) / n if n else 0.0,
            "avg_path": sum(v["path_length"] for v in vals) / n if n else 0.0,
            "avg_contacts": sum(v["n_contacts"] for v in vals) / n if n else 0.0,
            "avg_sensed": sum(v["n_sensed"] for v in vals) / n if n else 0.0,
        }

    return overall, by_family


def make_overall_figure(overall: dict[str, dict]) -> Path:
    planners = [p for p in PLANNER_ORDER if p in overall]
    labels = [PLANNER_LABELS[p] for p in planners]
    success = [overall[p]["success_rate"] for p in planners]
    path = [overall[p]["avg_path"] for p in planners]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(planners))
    colors = [COLORS[p] for p in planners]

    axes[0].bar(x, success, color=colors, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("Overall success rate")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, path, color=colors, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Average path length (m)")
    axes[1].set_title("Overall path length")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Scene-complex benchmark overview", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "overall_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_family_figure(by_family: dict[tuple[str, str], dict]) -> Path:
    planner = "beacon"
    labels = [f.replace("_", "\n") for f in FAMILY_ORDER]
    x = np.arange(len(FAMILY_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    success = [by_family[(planner, fam)]["success_rate"] for fam in FAMILY_ORDER]
    path = [by_family[(planner, fam)]["avg_path"] for fam in FAMILY_ORDER]

    axes[0].bar(x, success, color=COLORS[planner], edgecolor="white", width=0.6)
    axes[1].bar(x, path, color=COLORS[planner], edgecolor="white", width=0.6)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("Success by family")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Average path length (m)")
    axes[1].set_title("Path length by family")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("BEACON performance by environment family", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "family_comparison_beacon.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_family_comparison_figure(by_family: dict[tuple[str, str], dict]) -> Path:
    planners = ["beacon", "surp"]
    labels = [f.replace("_", "\n") for f in FAMILY_ORDER]
    x = np.arange(len(FAMILY_ORDER))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)

    for i, planner in enumerate(planners):
        success = [by_family[(planner, fam)]["success_rate"] for fam in FAMILY_ORDER]
        path = [by_family[(planner, fam)]["avg_path"] for fam in FAMILY_ORDER]
        offset = (i - 0.5) * width

        axes[0].bar(
            x + offset,
            success,
            width=width,
            color=COLORS[planner],
            edgecolor="white",
            label=PLANNER_LABELS[planner],
        )
        axes[1].bar(
            x + offset,
            path,
            width=width,
            color=COLORS[planner],
            edgecolor="white",
            label=PLANNER_LABELS[planner],
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("Success by family")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Average path length (m)")
    axes[1].set_title("Path length by family")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle("BEACON vs SURP by environment family", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "family_comparison_beacon_vs_surp.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_tradeoff_figure(overall: dict[str, dict]) -> Path:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    planners = [p for p in PLANNER_ORDER if p in overall]
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    label_offsets = {
        "beacon": (0.14, -0.006),
        "bug": (0.08, 0.0),
        "bug2": (0.09, 0.0),
        "dstar_lite": (0.09, 0.0),
    }

    points: list[tuple[str, float, float, float]] = []
    for planner in planners:
        x = overall[planner]["avg_path"]
        y = overall[planner]["success_rate"]
        s = 90 + 18 * overall[planner]["avg_contacts"]
        if planner == "beacon":
            ax.scatter(
                x,
                y,
                s=s * 1.18,
                color=COLORS[planner],
                alpha=0.98,
                edgecolor="black",
                linewidth=0.9,
                zorder=6,
            )
            ax.text(
                x + label_offsets[planner][0],
                y + label_offsets[planner][1],
                PLANNER_LABELS[planner],
                fontsize=9.5,
                fontweight="bold",
                va="center",
                ha="left",
                zorder=7,
            )
        else:
            ax.scatter(
                x,
                y,
                s=s,
                color=COLORS[planner],
                alpha=0.9,
                edgecolor="white",
                linewidth=0.9,
                zorder=4,
            )
            ax.text(
                x + label_offsets[planner][0],
                y + label_offsets[planner][1],
                PLANNER_LABELS[planner],
                fontsize=9,
                va="center",
                ha="left",
                zorder=5,
            )
        points.append((planner, x, y, s))

    x_min = min(p[1] for p in points) - 0.35
    x_max = max(p[1] for p in points) + 0.6
    y_min = 0.55
    y_max = 1.02

    beacon = next(p for p in points if p[0] == "beacon")
    frontier_end_x = max(p[1] for p in points if p[0] != "bug") + 0.15
    frontier_x = np.linspace(beacon[1], frontier_end_x, 300)
    t = (frontier_x - beacon[1]) / max(frontier_end_x - beacon[1], 1e-6)
    frontier_y = beacon[2] - 0.0015 - 0.0105 * (t ** 1.6)

    shade_y = np.linspace(frontier_y[0], y_max, 300)
    shade_x = np.interp(shade_y, frontier_y[::-1], frontier_x[::-1])
    ax.fill_betweenx(
        shade_y,
        x_min,
        shade_x,
        color="#cfe8cf",
        alpha=0.22,
        zorder=1,
    )
    ax.plot(
        frontier_x,
        frontier_y,
        linestyle="--",
        color="#2c7a2c",
        linewidth=2.0,
        zorder=3,
    )

    ax.text(
        0.24,
        0.965,
        "Optimal region /\nbetter tradeoff",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#2c5e2c",
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
        zorder=6,
    )

    ax.annotate(
        "Most optimal: highest reliability\nat lowest path length",
        xy=(beacon[1], beacon[2]),
        xytext=(beacon[1] + 1.15, beacon[2] - 0.085),
        textcoords="data",
        fontsize=8.5,
        ha="left",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            lw=1.1,
            color="black",
            connectionstyle="arc3,rad=0.10",
        ),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.92),
        zorder=7,
    )

    ax.set_xlabel("Average path length (m)")
    ax.set_ylabel("Success rate")
    ax.set_title("Reliability-efficiency tradeoff")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.grid(alpha=0.25)

    legend_handles = [
        Line2D([0], [0], linestyle="--", color="#2c7a2c", linewidth=2.0, label="Pareto frontier"),
        Patch(facecolor="#cfe8cf", edgecolor="none", alpha=0.22, label="Optimal region / better tradeoff"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8.5, framealpha=0.95)

    note = "Bubble size scales with average contacts"
    ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "tradeoff_scatter.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_tables(overall: dict[str, dict], by_family: dict[tuple[str, str], dict]) -> tuple[Path, Path]:
    md_lines = [
        "# Paper Tables",
        "",
        "## Overall benchmark results",
        "",
        "| Planner | Success Rate | Avg. Steps | Avg. Path (m) | Avg. Contacts | Avg. Sensed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for planner in [p for p in PLANNER_ORDER if p in overall]:
        row = overall[planner]
        md_lines.append(
            f"| {PLANNER_LABELS[planner]} | {100*row['success_rate']:.2f}% | "
            f"{row['avg_steps']:.2f} | {row['avg_path']:.3f} | "
            f"{row['avg_contacts']:.3f} | {row['avg_sensed']:.3f} |"
        )

    md_lines.extend([
        "",
        "## Family-level BEACON results",
        "",
        "| Family | BEACON Success | BEACON Path (m) | BEACON Steps | BEACON Sensed |",
        "|---|---:|---:|---:|---:|",
    ])
    for family in FAMILY_ORDER:
        beacon = by_family[("beacon", family)]
        md_lines.append(
            f"| {family.replace('_', ' ')} | {100*beacon['success_rate']:.2f}% | "
            f"{beacon['avg_path']:.3f} | {beacon['avg_steps']:.2f} | "
            f"{beacon['avg_sensed']:.3f} |"
        )

    md_path = OUT_DIR / "paper_tables.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    tex_lines = [
        "% Overall benchmark table",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Overall benchmark results across 400 episodes per planner.}",
        "\\label{tab:overall_results}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Planner & Success Rate & Avg. Steps & Avg. Path (m) & Avg. Contacts & Avg. Sensed \\\\",
        "\\midrule",
    ]
    for planner in [p for p in PLANNER_ORDER if p in overall]:
        row = overall[planner]
        tex_lines.append(
            f"{PLANNER_LABELS[planner]} & {100*row['success_rate']:.2f}\\% & "
            f"{row['avg_steps']:.2f} & {row['avg_path']:.3f} & "
            f"{row['avg_contacts']:.3f} & {row['avg_sensed']:.3f} \\\\"
        )
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "% Family-level comparison table",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Family-level BEACON results.}",
        "\\label{tab:family_results}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Family & BEACON Success & BEACON Path (m) & BEACON Steps & BEACON Sensed \\\\",
        "\\midrule",
    ])
    for family in FAMILY_ORDER:
        beacon = by_family[("beacon", family)]
        tex_lines.append(
            f"{family.replace('_', ' ')} & {100*beacon['success_rate']:.2f}\\% & "
            f"{beacon['avg_path']:.3f} & {beacon['avg_steps']:.2f} & "
            f"{beacon['avg_sensed']:.3f} \\\\"
        )
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    tex_path = OUT_DIR / "paper_tables.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    return md_path, tex_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(discover_csvs())
    overall, by_family = summarize(rows)

    overall_fig = make_overall_figure(overall)
    family_fig = make_family_figure(by_family)
    family_compare_fig = make_family_comparison_figure(by_family)
    tradeoff_fig = make_tradeoff_figure(overall)
    md_table, tex_table = write_tables(overall, by_family)

    print(f"Wrote figure: {overall_fig}")
    print(f"Wrote figure: {family_fig}")
    print(f"Wrote figure: {family_compare_fig}")
    print(f"Wrote figure: {tradeoff_fig}")
    print(f"Wrote table:  {md_table}")
    print(f"Wrote table:  {tex_table}")


if __name__ == "__main__":
    main()
