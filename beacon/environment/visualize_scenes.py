import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon, Circle as MplCircle
import numpy as np

from beacon.environment.scene_generator_pybullet import load_scene, load_all_scenes, _SCENES_DIR

IMAGES_DIR = Path(__file__).resolve().parent / "data" / "images"

COLORS = {
    "movable":   "#f4a261",   # orange
    "unmovable": "#6c757d",   # grey
}


def plot_scene(scene_dict: dict, ax: plt.Axes) -> None:
    xmin, xmax, ymin, ymax = scene_dict["workspace"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f9fa")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for obs in scene_dict["obstacles"]:
        color = COLORS.get(obs["true_class"], "#cccccc")
        if obs.get("shape_type") == "circle" and "center" in obs and "radius" in obs:
            cx, cy = obs["center"]
            patch = MplCircle((cx, cy), obs["radius"],
                              facecolor=color, edgecolor="#444444",
                              linewidth=0.8, alpha=0.85)
        else:
            verts = np.array(obs["vertices"])
            patch = MplPolygon(verts, closed=True,
                               facecolor=color, edgecolor="#444444",
                               linewidth=0.8, alpha=0.85)
        ax.add_patch(patch)

    # Start and goal
    sx, sy = scene_dict["start"][:2]
    gx, gy = scene_dict["goal"][:2]
    ax.plot(sx, sy, "o", color="#2196F3", markersize=7, zorder=5)
    ax.plot(gx, gy, "*", color="#4CAF50", markersize=10, zorder=5)

    # Legend info
    obs_list = scene_dict["obstacles"]
    n_mov = sum(1 for o in obs_list if o["true_class"] == "movable")
    n_unm = len(obs_list) - n_mov
    ax.set_title(
        f"{scene_dict.get('family', '')}  "
        f"mov={n_mov} unm={n_unm}",
        fontsize=7, pad=3,
    )


def render_all(
    indices: list,
    fragility: str,
    family: str = None,
    show: bool = False,
) -> None:
    out_dir = IMAGES_DIR / (family or "legacy")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        scene = load_scene(idx, family=family, fragility=fragility, seed=idx)

        fig, ax = plt.subplots(figsize=(4, 4))
        plot_scene(scene, ax)

        legend_patches = [
            mpatches.Patch(color=COLORS["movable"],   label="movable"),
            mpatches.Patch(color=COLORS["unmovable"], label="unmovable"),
            mpatches.Patch(color="#2196F3",            label="start"),
            mpatches.Patch(color="#4CAF50",            label="goal"),
        ]
        ax.legend(handles=legend_patches, loc="lower right",
                  fontsize=6, framealpha=0.7)

        plt.tight_layout()
        prefix = family if family else "scene"
        out = out_dir / f"{prefix}_{idx:03d}_{fragility}.png"
        plt.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  saved {out.name}")

    print(f"\nDone — {len(indices)} image(s) in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragility", default="mixed",
                        choices=["mixed", "uniform", "all_movable"])
    parser.add_argument("--family", default=None,
                        choices=["dense_clutter", "narrow_passage", "perturbed",
                                 "semantic_trap", "sparse_clutter"],
                        help="Scene family subdirectory; omit for legacy flat scenes.")
    parser.add_argument("--index", nargs="*", type=int, default=None,
                        help="Specific scene indices; omit for all.")
    args = parser.parse_args()

    if args.family is not None:
        family_dir = _SCENES_DIR / args.family
        all_paths = sorted(family_dir.glob("scene_*.json"))
    else:
        all_paths = sorted(_SCENES_DIR.glob("scene_*.json"))

    if args.index is not None:
        indices = args.index
    else:
        indices = list(range(len(all_paths)))

    tag = args.family or "legacy"
    print(f"Rendering {len(indices)} '{tag}' scene(s) [{args.fragility}]...")
    render_all(indices, fragility=args.fragility, family=args.family)
