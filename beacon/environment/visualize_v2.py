import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from beacon.environment.scene_generator_pybullet import load_scene
from beacon.environment.scene_generator_v2 import FAMILIES_V2

CLASS_COLORS = {
    "movable":     "gold",
    "not_movable": "#8b9db5",
}

IMAGES_DIR = Path(__file__).resolve().parent / "data" / "images"


def draw_scene(scene, ax=None, save_path=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    xmin, xmax, ymin, ymax = scene["workspace"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    for obs in scene["obstacles"]:
        cls = obs.get("class_true", obs.get("true_class", "movable"))
        patch = MplPolygon(
            obs["vertices"],
            closed=True,
            edgecolor="black",
            facecolor=CLASS_COLORS.get(cls, "lightblue"),
            alpha=0.8,
        )
        ax.add_patch(patch)

    sx, sy, _ = scene["start"]
    gx, gy, _ = scene["goal"]
    ax.plot(sx, sy, "bo", markersize=8, label="Start")
    ax.plot(gx, gy, "r*", markersize=12, label="Goal")
    ax.set_title(f"{scene['family']}  ({len(scene['obstacles'])} circles)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    if standalone:
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)


def render_family(family, indices, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        scene = load_scene(idx, family=family)
        path  = out_dir / f"{family}_{idx:03d}.png"
        draw_scene(scene, save_path=path)
        print(f"  saved {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", nargs="*", default=None, choices=FAMILIES_V2)
    parser.add_argument("--index",  nargs="*", type=int, default=None)
    args = parser.parse_args()

    families = args.family or FAMILIES_V2
    for fam in families:
        fam_dir   = IMAGES_DIR / fam
        all_paths = sorted((Path(__file__).resolve().parent / "data" / "scenes" / fam).glob("scene_*.json"))
        indices   = args.index if args.index is not None else list(range(len(all_paths)))
        print(f"Rendering {len(indices)} '{fam}' scenes...")
        render_family(fam, indices, fam_dir)
