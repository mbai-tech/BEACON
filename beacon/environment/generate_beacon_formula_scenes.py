"""Create illustrative mixed-shape scenes that visually explain BEACON scoring."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import Polygon as MplPolygon


OUT_SCENES = Path(__file__).resolve().parent / "data" / "formula_scenes"
OUT_IMAGES = Path(__file__).resolve().parent / "data" / "formula_scene_images"
PAPER_IMAGES = (
    Path(__file__).resolve().parent
    / "data"
    / "metrics"
    / "paper_assets"
    / "beacon_formula_scenes"
)
WORKSPACE = [0.0, 6.0, 0.0, 6.0]
COLORS = {
    "movable": "#e0b400",
    "unmovable": "#8b9db5",
}


def circle_obstacle(idx: int, center: tuple[float, float], radius: float, cls: str) -> dict:
    return {
        "id": idx,
        "shape_type": "circle",
        "true_class": cls,
        "class_true": cls,
        "center": [center[0], center[1]],
        "radius": radius,
        "observed": False,
    }


def polygon_obstacle(idx: int, vertices: list[list[float]], cls: str) -> dict:
    cx = sum(v[0] for v in vertices) / len(vertices)
    cy = sum(v[1] for v in vertices) / len(vertices)
    return {
        "id": idx,
        "shape_type": "polygon",
        "true_class": cls,
        "class_true": cls,
        "center": [cx, cy],
        "vertices": vertices,
        "observed": False,
    }


def scene_progress_vs_risk() -> tuple[dict, dict]:
    obstacles = [
        circle_obstacle(0, (3.0, 3.0), 0.52, "unmovable"),
        polygon_obstacle(1, [[2.1, 1.7], [2.6, 1.55], [2.8, 2.0], [2.25, 2.2]], "movable"),
        polygon_obstacle(2, [[3.55, 3.75], [4.15, 3.5], [4.25, 4.0], [3.7, 4.2]], "movable"),
        circle_obstacle(3, (4.45, 2.0), 0.28, "movable"),
    ]
    scene = {
        "name": "progress_vs_risk",
        "family": "formula_demo",
        "workspace": WORKSPACE,
        "start": [0.7, 3.0, 0.0],
        "goal": [5.35, 3.0, 0.0],
        "obstacles": obstacles,
    }
    overlays = {
        "title": "Risk Term Changes the Preferred Route",
        "caption": "Direct progress is shorter, but high semantic risk makes the upper detour preferable.",
        "paths": [
            {
                "label": "Direct but risky",
                "points": [(0.7, 3.0), (2.2, 3.0), (3.8, 3.0), (5.35, 3.0)],
                "color": "#d62728",
                "style": "--",
                "text_xy": (2.2, 3.28),
            },
            {
                "label": "Safer BEACON route",
                "points": [(0.7, 3.0), (1.9, 4.0), (3.4, 4.55), (4.6, 3.95), (5.35, 3.0)],
                "color": "#1f78b4",
                "style": "-",
                "text_xy": (3.5, 4.9),
            },
        ],
    }
    return scene, overlays


def scene_smoothness_vs_progress() -> tuple[dict, dict]:
    obstacles = [
        polygon_obstacle(0, [[2.3, 2.25], [2.85, 2.1], [3.0, 2.75], [2.45, 2.95]], "movable"),
        polygon_obstacle(1, [[3.1, 3.0], [3.55, 2.7], [3.9, 3.2], [3.45, 3.55]], "movable"),
        circle_obstacle(2, (2.95, 3.95), 0.34, "unmovable"),
        circle_obstacle(3, (4.25, 2.15), 0.3, "movable"),
    ]
    scene = {
        "name": "smoothness_vs_progress",
        "family": "formula_demo",
        "workspace": WORKSPACE,
        "start": [0.8, 1.2, 0.0],
        "goal": [5.15, 4.9, 0.0],
        "obstacles": obstacles,
    }
    overlays = {
        "title": "Smoothness Favors Stable Local Motion",
        "caption": "A jagged shortcut exists, but the smoother arc yields better local action quality.",
        "paths": [
            {
                "label": "Sharp local shortcut",
                "points": [(0.8, 1.2), (2.15, 2.0), (2.65, 3.35), (3.45, 2.6), (5.15, 4.9)],
                "color": "#d62728",
                "style": "--",
                "text_xy": (2.2, 3.55),
            },
            {
                "label": "Smoother BEACON route",
                "points": [(0.8, 1.2), (1.75, 1.75), (2.9, 2.2), (4.0, 3.35), (5.15, 4.9)],
                "color": "#1f78b4",
                "style": "-",
                "text_xy": (3.55, 2.55),
            },
        ],
    }
    return scene, overlays


def scene_replan_trigger() -> tuple[dict, dict]:
    obstacles = [
        circle_obstacle(0, (2.15, 3.2), 0.26, "movable"),
        polygon_obstacle(1, [[3.1, 2.6], [3.85, 2.45], [3.95, 3.1], [3.25, 3.2]], "unmovable"),
        polygon_obstacle(2, [[4.3, 3.7], [4.85, 3.5], [5.0, 4.1], [4.45, 4.2]], "movable"),
        circle_obstacle(3, (3.25, 4.35), 0.31, "unmovable"),
    ]
    scene = {
        "name": "risk_triggered_replan",
        "family": "formula_demo",
        "workspace": WORKSPACE,
        "start": [0.75, 2.6, 0.0],
        "goal": [5.35, 3.85, 0.0],
        "obstacles": obstacles,
    }
    overlays = {
        "title": "New Risk Information Triggers Replanning",
        "caption": "Initial forward progress looks reasonable until newly revealed high-risk obstacles flip the local ranking.",
        "paths": [
            {
                "label": "Initial candidate",
                "points": [(0.75, 2.6), (1.95, 2.95), (3.15, 3.15), (5.35, 3.85)],
                "color": "#6a3d9a",
                "style": ":",
                "text_xy": (1.45, 2.2),
            },
            {
                "label": "Replanned BEACON route",
                "points": [(0.75, 2.6), (1.8, 2.1), (3.2, 1.95), (4.5, 2.65), (5.35, 3.85)],
                "color": "#1f78b4",
                "style": "-",
                "text_xy": (3.15, 1.55),
            },
        ],
        "sensor_arc": [(2.6, 2.95), 1.1],
    }
    return scene, overlays


def draw_scene(scene: dict, overlays: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    xmin, xmax, ymin, ymax = scene["workspace"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    for obs in scene["obstacles"]:
        color = COLORS[obs["true_class"]]
        if obs["shape_type"] == "circle":
            patch = MplCircle(
                tuple(obs["center"]),
                obs["radius"],
                facecolor=color,
                edgecolor="black",
                linewidth=1.0,
                alpha=0.82,
            )
        else:
            patch = MplPolygon(
                obs["vertices"],
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=1.0,
                alpha=0.82,
            )
        ax.add_patch(patch)

    if "sensor_arc" in overlays:
        center, radius = overlays["sensor_arc"]
        sense = MplCircle(center, radius, facecolor="#9ecae1", edgecolor="#3182bd", linestyle="--", alpha=0.15)
        ax.add_patch(sense)
        ax.text(center[0] - 0.15, center[1] + radius + 0.08, "new observation", fontsize=8, color="#225ea8")

    for path in overlays["paths"]:
        xs = [p[0] for p in path["points"]]
        ys = [p[1] for p in path["points"]]
        ax.plot(xs, ys, path["style"], color=path["color"], linewidth=2.3, zorder=5)
        ax.text(path["text_xy"][0], path["text_xy"][1], path["label"], color=path["color"], fontsize=8.5, weight="bold")

    sx, sy, _ = scene["start"]
    gx, gy, _ = scene["goal"]
    ax.plot(sx, sy, "o", color="#2a9d8f", markersize=8, label="Start")
    ax.plot(gx, gy, "*", color="#d62828", markersize=13, label="Goal")

    ax.text(0.02, 1.02, overlays["title"], transform=ax.transAxes, fontsize=12, weight="bold", ha="left")
    ax.text(0.02, -0.08, overlays["caption"], transform=ax.transAxes, fontsize=9, ha="left")
    ax.text(0.985, 0.02, "Blue: preferred BEACON action", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_scene(scene: dict, path: Path) -> None:
    path.write_text(json.dumps(scene, indent=2) + "\n")


def main() -> None:
    OUT_SCENES.mkdir(parents=True, exist_ok=True)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    PAPER_IMAGES.mkdir(parents=True, exist_ok=True)

    builders = [
        scene_progress_vs_risk,
        scene_smoothness_vs_progress,
        scene_replan_trigger,
    ]
    for builder in builders:
        scene, overlays = builder()
        scene_path = OUT_SCENES / f"{scene['name']}.json"
        image_path = OUT_IMAGES / f"{scene['name']}.png"
        paper_path = PAPER_IMAGES / f"{scene['name']}.png"
        save_scene(scene, scene_path)
        draw_scene(scene, overlays, image_path)
        draw_scene(scene, overlays, paper_path)
        print(f"saved {scene_path.name} and {image_path.name}")


if __name__ == "__main__":
    main()
