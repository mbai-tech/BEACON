import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon

from beacon.core.constants import DISPLAY_COLORS, OUTPUT_DIR
from beacon.core.models import OnlineSurpResult


def save_scene_snapshot(scene: dict, family: str, seed: int) -> Path:
    """Persist the generated scene so a specific run can be reproduced later."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / f"random_scene_{family}_seed{seed}.json"
    with save_path.open("w") as handle:
        json.dump(scene, handle, indent=2)
    return save_path


def plot_final_snapshot(result: OnlineSurpResult) -> Path:
    """Save a static image of the final state and full robot path."""
    output_path = OUTPUT_DIR / f"online_surp_final_{result.family}_seed{result.seed}.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    xmin, xmax, ymin, ymax = result.scene["workspace"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    for obstacle in result.initial_scene["obstacles"]:
        ax.add_patch(
            MplPolygon(
                obstacle["vertices"],
                closed=True,
                fill=False,
                edgecolor="#444444",
                linewidth=1.0,
                linestyle="--",
                alpha=0.22,
            )
        )

    for obstacle in result.scene["obstacles"]:
        label = obstacle["true_class"]
        ax.add_patch(
            MplPolygon(
                obstacle["vertices"],
                closed=True,
                facecolor=DISPLAY_COLORS.get(label, "#d0d7de"),
                edgecolor="#111111" if obstacle["observed"] else "#666666",
                linewidth=2.4 if obstacle["observed"] else 1.0,
                alpha=0.9 if obstacle["observed"] else 0.45,
            )
        )

    path = np.array(result.path)
    ax.plot(path[:, 0], path[:, 1], color="#1d3557", linewidth=1.8)
    ax.scatter(path[:, 0], path[:, 1], s=10, color="#457b9d")
    ax.scatter([result.scene["start"][0]], [result.scene["start"][1]], s=90, color="#2a9d8f", marker="o")
    ax.scatter([result.scene["goal"][0]], [result.scene["goal"][1]], s=110, color="#d62828", marker="*")
    ax.set_title(f"Online Push Result: {result.family} (seed {result.seed})")
    ax.grid(alpha=0.2)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def animate_result(result: OnlineSurpResult) -> None:
    """Show the simulation in real time using the stored frame history."""
    fig, ax = plt.subplots(figsize=(8, 8))
    xmin, xmax, ymin, ymax = result.scene["workspace"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(f"Online Push Simulation: {result.family} (seed {result.seed})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)

    for obstacle in result.initial_scene["obstacles"]:
        ghost_patch = MplPolygon(
            obstacle["vertices"],
            closed=True,
            fill=False,
            edgecolor="#444444",
            linewidth=1.0,
            linestyle="--",
            alpha=0.22,
        )
        ax.add_patch(ghost_patch)

    obstacle_patches = []
    initial_obstacles = result.frames[0].obstacles
    for obstacle in initial_obstacles:
        label = obstacle["true_class"]
        patch = MplPolygon(
            obstacle["vertices"],
            closed=True,
            facecolor=DISPLAY_COLORS.get(label, "#d0d7de"),
            edgecolor="#111111" if obstacle["observed"] else "#666666",
            linewidth=2.4 if obstacle["observed"] else 1.0,
            alpha=0.9 if obstacle["observed"] else 0.45,
        )
        ax.add_patch(patch)
        obstacle_patches.append(patch)

    path_line, = ax.plot([], [], color="#1d3557", linewidth=1.8, zorder=3)
    trail_points = ax.scatter([], [], s=18, color="#457b9d", zorder=4, label="robot path")
    robot_point = ax.scatter([], [], s=90, color="#264653", zorder=5, label="robot")
    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    ax.scatter([result.scene["start"][0]], [result.scene["start"][1]], s=90, color="#2a9d8f", marker="o", label="start")
    ax.scatter([result.scene["goal"][0]], [result.scene["goal"][1]], s=110, color="#d62828", marker="*", label="goal")
    ax.legend(loc="upper right")

    path_array = np.array(result.path)

    def update(frame_idx: int):
        frame = result.frames[frame_idx]
        current_path = path_array[: frame_idx + 1]
        path_line.set_data(current_path[:, 0], current_path[:, 1])
        trail_points.set_offsets(current_path)
        robot_point.set_offsets(np.array([[frame.position[0], frame.position[1]]]))

        for patch, obstacle in zip(obstacle_patches, frame.obstacles):
            patch.set_xy(obstacle["vertices"])
            label = obstacle["true_class"]
            patch.set_facecolor(DISPLAY_COLORS.get(label, "#d0d7de"))
            patch.set_alpha(0.9 if obstacle["observed"] else 0.45)
            patch.set_edgecolor("#111111" if obstacle["observed"] else "#666666")
            patch.set_linewidth(2.4 if obstacle["observed"] else 1.0)

        status_text.set_text(f"Step {frame_idx + 1}/{len(result.frames)}\n{frame.message}")
        return [path_line, trail_points, robot_point, status_text, *obstacle_patches]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(result.frames),
        interval=90,
        blit=False,
        repeat=False,
    )
    fig._animation = animation
    plt.show()
