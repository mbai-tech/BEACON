"""Run Bug1, Bug2, D* Lite, and BEACON on a single scene.

Usage
-----
    python beacon/experiments/run_one_scene.py
    python beacon/experiments/run_one_scene.py --scene 3 --family cluttered
    python beacon/experiments/run_one_scene.py --scene 0 --visuals out/visuals
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [str(REPO_ROOT), str(REPO_ROOT / "beacon")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from beacon.environment.scene_complex import generate_scene as _generate_scene
from beacon.core.bug_algorithm import run_bug
from beacon.core.bug2_algorithm import run_bug2
from beacon.core.dstar_lite_algorithm import run_dstar_lite
from beacon.core.planner import run_online_surp_push
from beacon.utils.metrics import compute_metrics

FAMILIES = ["sparse", "cluttered", "collision_required", "collision_shortcut"]

PLANNERS = {
    "bug1":    (run_bug,             dict(max_steps=500, step_size=0.07, sensing_range=0.55)),
    "bug2":    (run_bug2,            dict(max_steps=500, step_size=0.07, sensing_range=0.55)),
    "dstar":   (run_dstar_lite,      dict(max_steps=500, step_size=0.06, sensing_range=0.45)),
    "beacon":  (run_online_surp_push,dict(max_steps=500, step_size=0.04, sensing_range=0.35)),
}

CLASS_COLORS = {
    "movable":   "#a8d5a2",
    "fragile":   "#f4a0a0",
    "safe":      "#a0c4f4",
    "forbidden": "#c0a0f4",
}


def load_scene(scene_idx: int, family: str) -> dict:
    scene = _generate_scene(family=family, seed=scene_idx)
    scene["seed"] = scene_idx
    scene["scene_idx"] = scene_idx
    return scene


def _centroid(vertices) -> np.ndarray:
    return np.mean(np.array(vertices), axis=0)


def save_png(result, planner_name: str, scene_idx: int, family: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{planner_name}_{family}_scene{scene_idx:03d}.png"

    initial_by_id = {obs["id"]: obs for obs in result.initial_scene["obstacles"]}
    pushed_ids = {
        obs["id"] for obs in result.scene["obstacles"]
        if obs["id"] in initial_by_id
        and np.linalg.norm(_centroid(obs["vertices"]) - _centroid(initial_by_id[obs["id"]]["vertices"])) > 0.01
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    xmin, xmax, ymin, ymax = result.scene["workspace"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    for obs in result.scene["obstacles"]:
        oid = obs["id"]
        cls = obs.get("true_class", obs.get("class_true", "safe"))
        pushed = oid in pushed_ids

        if oid in initial_by_id:
            ax.add_patch(MplPolygon(
                initial_by_id[oid]["vertices"], closed=True,
                fill=pushed, facecolor="#ffcc8840" if pushed else "none",
                edgecolor="#e07b00" if pushed else "#cccccc",
                linewidth=1.4 if pushed else 0.7, linestyle="--", alpha=0.7, zorder=1,
            ))

        ax.add_patch(MplPolygon(
            obs["vertices"], closed=True,
            facecolor=CLASS_COLORS.get(cls, "#dddddd"),
            edgecolor="#e07b00" if pushed else "#333333",
            linewidth=2.0 if pushed else 1.2, alpha=0.85, zorder=2,
        ))

        if pushed and oid in initial_by_id:
            c0 = _centroid(initial_by_id[oid]["vertices"])
            c1 = _centroid(obs["vertices"])
            ax.annotate("", xy=c1, xytext=c0,
                        arrowprops=dict(arrowstyle="-|>", color="#e07b00",
                                        lw=1.5, mutation_scale=10), zorder=3)

    if result.path:
        path = np.array(result.path)
        ax.plot(path[:, 0], path[:, 1], color="#1d3557", linewidth=1.6, zorder=4)
        ax.scatter(path[:, 0], path[:, 1], s=6, color="#457b9d", zorder=4)

    sx, sy = result.scene["start"][:2]
    gx, gy = result.scene["goal"][:2]
    ax.scatter([sx], [sy], s=80, color="#2a9d8f", marker="o", zorder=5, label="start")
    ax.scatter([gx], [gy], s=120, color="#d62828", marker="*", zorder=5, label="goal")
    if pushed_ids:
        ax.scatter([], [], marker="s", color="#e07b00", alpha=0.6,
                   label=f"pushed ({len(pushed_ids)})")

    status = "SUCCESS" if result.success else "FAIL"
    ax.set_title(f"{planner_name} | {family} scene {scene_idx} | {status}", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",   type=int, default=0,          help="Scene index / seed.")
    parser.add_argument("--family",  type=str, default="cluttered", choices=FAMILIES)
    parser.add_argument("--visuals", type=str, default=None,        metavar="DIR",
                        help="Directory to save PNGs.")
    args = parser.parse_args()

    scene = load_scene(args.scene, args.family)
    visuals_dir = Path(args.visuals) if args.visuals else None

    print(f"\nScene {args.scene} | family={args.family} | "
          f"{len(scene['obstacles'])} obstacles\n")
    print(f"{'planner':<10}  {'success':<8}  {'steps':<7}  {'length (m)':<12}  {'contacts':<10}  {'sensed'}")
    print("-" * 65)

    for name, (fn, kwargs) in PLANNERS.items():
        result = fn(scene, **kwargs)
        m = compute_metrics(result, name)
        png = ""
        if visuals_dir:
            png = str(save_png(result, name, args.scene, args.family, visuals_dir))
        print(f"{name:<10}  {'yes' if m.success else 'no':<8}  {m.steps:<7}  "
              f"{m.path_length:<12.3f}  {m.n_contacts:<10}  {m.n_sensed}"
              + (f"  → {png}" if png else ""))


if __name__ == "__main__":
    main()
