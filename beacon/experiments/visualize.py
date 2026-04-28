"""
experiments/visualize.py — Result plotting using beacon.core visualization.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from beacon.core.constants import OUTPUT_DIR
from beacon.core.models import OnlineSurpResult
from beacon.core.visualization import animate_result, plot_final_snapshot, save_scene_snapshot

from beacon.utils.metrics import EpisodeMetrics, aggregate, print_summary

__all__ = ["animate_result", "plot_final_snapshot", "save_scene_snapshot", "plot_results"]


def plot_results(episodes: list[EpisodeMetrics], show: bool = True) -> None:
    """Bar-chart summary of success rate, avg steps, and avg path length."""
    agg = aggregate(episodes)
    print_summary(agg, planner_name=episodes[0].planner_name if episodes else "")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [
        ("Success rate", agg.get("success_rate", 0.0), "%"),
        ("Avg steps",    agg.get("avg_steps", 0.0),    ""),
        ("Avg path (m)", agg.get("avg_path_length", 0.0), "m"),
    ]
    for ax, (title, value, unit) in zip(axes, metrics):
        ax.bar([title], [value * 100 if unit == "%" else value])
        ax.set_title(title)
        ax.set_ylabel(f"{unit}" if unit else "")

    plt.tight_layout()
    out = OUTPUT_DIR / "beacon_results_summary.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"Saved summary to {out}")
    if show:
        plt.show()
