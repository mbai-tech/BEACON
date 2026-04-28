"""Export the standard random demo scenes used by the GIF pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from scholar.core.constants import DISPLAY_COLORS
from scholar.core.scene_setup import generate_one_random_environment


OUT_DIR = Path(__file__).resolve().parent / "data" / "metrics" / "paper_assets" / "regular_demo_scenes"


def draw_scene(scene: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    xmin, xmax, ymin, ymax = scene["workspace"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    for obstacle in scene["obstacles"]:
        label = obstacle.get("true_class", obstacle.get("class_true", "safe"))
        patch = MplPolygon(
            obstacle["vertices"],
            closed=True,
            facecolor=DISPLAY_COLORS.get(label, "#d0d7de"),
            edgecolor="#111111",
            linewidth=1.0,
            alpha=0.85,
        )
        ax.add_patch(patch)

    sx, sy, _ = scene["start"]
    gx, gy, _ = scene["goal"]
    ax.scatter([sx], [sy], s=85, color="#2a9d8f", marker="o", label="start", zorder=5)
    ax.scatter([gx], [gy], s=110, color="#d62828", marker="*", label="goal", zorder=5)
    ax.set_title(scene.get("family", "demo scene").replace("_", " "))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export regular GIF/demo scenes as JSON and PNG.")
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx in range(args.count):
        scene = generate_one_random_environment()
        seed = scene.get("seed", idx)
        stem = f"regular_demo_scene_{idx:02d}_seed{seed}"
        json_path = OUT_DIR / f"{stem}.json"
        png_path = OUT_DIR / f"{stem}.png"
        json_path.write_text(json.dumps(scene, indent=2) + "\n")
        draw_scene(scene, png_path)
        print(f"saved {json_path.name} and {png_path.name}")


if __name__ == "__main__":
    main()
