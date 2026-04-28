"""Run metrics for multiple planners on the BEACON scene_complex scenes.

This script uses ``beacon.main_beacon.load_scene(...)`` so the evaluated
scenes match the same scene source used by the main BEACON demo CLI.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BEACON_ROOT = REPO_ROOT / "beacon"
if str(BEACON_ROOT) not in sys.path:
    sys.path.insert(0, str(BEACON_ROOT))
from beacon.environment.scene_complex import generate_scene as _generate_complex_scene

try:
    from beacon.planning.baselines import PLANNERS
    from beacon.utils.metrics import compute_metrics
except ModuleNotFoundError:
    from beacon.planning.baselines import PLANNERS
    from beacon.utils.metrics import compute_metrics


FAMILIES = ["sparse", "cluttered", "collision_required", "collision_shortcut"]


ALL_PLANNERS = sorted([*PLANNERS.keys(), "beacon"])

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "beacon"
    / "environment"
    / "data"
    / "metrics"
    / "metrics_scene_complex.csv"
)


def load_scene(scene_idx: int, family: str, seed: int | None = None) -> dict:
    effective_seed = scene_idx if seed is None else seed
    scene = _generate_complex_scene(family=family, seed=effective_seed)
    scene["seed"] = effective_seed
    scene["scene_idx"] = scene_idx
    return scene


def parse_scene_indices(scene_args: list[int], scene_range: str | None) -> list[int]:
    if scene_range:
        lo, hi = map(int, scene_range.split("-"))
        return list(range(lo, hi + 1))
    return scene_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more planners on the scene_complex BEACON scenes and save metrics to CSV."
    )
    parser.add_argument(
        "--scene",
        type=int,
        nargs="+",
        default=[0],
        help="One or more scene indices. Ignored when --scenes is provided.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Inclusive scene range, e.g. 0-99.",
    )
    parser.add_argument(
        "--family",
        nargs="*",
        default=None,
        choices=FAMILIES,
        help="Restrict evaluation to specific BEACON families.",
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["beacon", "bug", "rrt"],
        choices=sorted(ALL_PLANNERS),
        help="Planner names to evaluate.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum steps passed to each planner when supported.",
    )
    parser.add_argument(
        "--sense",
        type=float,
        default=0.35,
        help="Sensing radius for BEACON/SURP-style planners.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.04,
        help="Step size for BEACON/SURP-style planners.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="CSV output path.",
    )
    parser.add_argument(
        "--visuals",
        type=str,
        default=None,
        metavar="DIR",
        help="If set, save a PNG per run to this directory.",
    )
    return parser.parse_args()


def run_planner(planner_name: str, scene: dict, max_steps: int, step_size: float, sensing_range: float):
    planner_fn = PLANNERS[planner_name]

    if planner_name == "beacon":
        return planner_fn(scene, max_steps=max_steps, step_size=step_size, sensing_range=sensing_range)
    if planner_name == "dstar_lite":
        return planner_fn(
            scene,
            max_steps=max_steps,
            step_size=max(0.06, step_size),
            sensing_range=max(0.45, sensing_range),
        )
    if planner_name == "bug":
        return planner_fn(scene, max_steps=max_steps, step_size=max(0.07, step_size), sensing_range=max(0.55, sensing_range))
    if planner_name == "rrt":
        return planner_fn(scene, max_steps=max_steps, step_size=max(0.07, step_size), sensing_range=max(0.55, sensing_range))

    return planner_fn(scene)


CLASS_COLORS = {
    "movable":   "#a8d5a2",
    "fragile":   "#f4a0a0",
    "safe":      "#a0c4f4",
    "forbidden": "#c0a0f4",
}


def _centroid(vertices) -> np.ndarray:
    return np.mean(np.array(vertices), axis=0)


def save_result_png(result, planner_name: str, scene_idx: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{planner_name}_{result.family}_scene{scene_idx:03d}.png"
    save_path = out_dir / fname

    # index initial positions by obstacle id
    initial_by_id = {obs["id"]: obs for obs in result.initial_scene["obstacles"]}

    # find which obstacles moved (centroid shift > 1 cm)
    pushed_ids = set()
    for obs in result.scene["obstacles"]:
        oid = obs["id"]
        if oid in initial_by_id:
            delta = np.linalg.norm(
                _centroid(obs["vertices"]) - _centroid(initial_by_id[oid]["vertices"])
            )
            if delta > 0.01:
                pushed_ids.add(oid)

    fig, ax = plt.subplots(figsize=(6, 6))
    xmin, xmax, ymin, ymax = result.scene["workspace"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    # draw all obstacles: initial ghost + final filled
    for obs in result.scene["obstacles"]:
        oid = obs["id"]
        cls = obs.get("true_class", obs.get("class_true", "safe"))
        pushed = oid in pushed_ids

        # initial ghost (more prominent orange outline for pushed obstacles)
        if oid in initial_by_id:
            init_verts = initial_by_id[oid]["vertices"]
            ax.add_patch(MplPolygon(
                init_verts, closed=True, fill=pushed,
                facecolor="#ffcc8840" if pushed else "none",
                edgecolor="#e07b00" if pushed else "#bbbbbb",
                linewidth=1.4 if pushed else 0.7,
                linestyle="--", alpha=0.7, zorder=1,
            ))

        # final position
        ax.add_patch(MplPolygon(
            obs["vertices"], closed=True,
            facecolor=CLASS_COLORS.get(cls, "#dddddd"),
            edgecolor="#e07b00" if pushed else "#333333",
            linewidth=2.0 if pushed else 1.2,
            alpha=0.85, zorder=2,
        ))

        # displacement arrow for pushed obstacles
        if pushed and oid in initial_by_id:
            c0 = _centroid(initial_by_id[oid]["vertices"])
            c1 = _centroid(obs["vertices"])
            ax.annotate(
                "", xy=c1, xytext=c0,
                arrowprops=dict(arrowstyle="-|>", color="#e07b00",
                                lw=1.5, mutation_scale=10),
                zorder=3,
            )

    # path
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
    ax.set_title(f"{planner_name} | {result.family} scene {scene_idx} | {status}", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main() -> None:
    args = parse_args()
    families = args.family or FAMILIES
    scene_indices = parse_scene_indices(args.scene, args.scenes)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    visuals_dir = Path(args.visuals) if args.visuals else None
    rows: list[dict] = []
    total = len(scene_indices) * len(families) * len(args.planners)
    counter = 0

    print(
        f"Running {len(scene_indices)} scene(s) x {len(families)} family/families x "
        f"{len(args.planners)} planner(s) = {total} episodes"
    )

    for scene_idx in scene_indices:
        for family in families:
            base_scene = load_scene(scene_idx, family=family, seed=scene_idx)
            for planner_name in args.planners:
                counter += 1
                print(
                    f"[{counter}/{total}] planner={planner_name} family={family} "
                    f"scene={scene_idx:03d} seed={base_scene['seed']} ...",
                    end=" ",
                    flush=True,
                )
                result = run_planner(
                    planner_name,
                    base_scene,
                    max_steps=args.steps,
                    step_size=args.step,
                    sensing_range=args.sense,
                )
                metrics = compute_metrics(result, planner_name)
                png_path = ""
                if visuals_dir is not None:
                    png_path = str(save_result_png(result, planner_name, scene_idx, visuals_dir))
                row = {
                    "planner": metrics.planner,
                    "family": metrics.family,
                    "scene_idx": scene_idx,
                    "seed": metrics.seed,
                    "success": metrics.success,
                    "steps": metrics.steps,
                    "path_length": round(metrics.path_length, 6),
                    "n_contacts": metrics.n_contacts,
                    "n_sensed": metrics.n_sensed,
                    "png_path": png_path,
                }
                rows.append(row)
                print(
                    f"{'OK' if metrics.success else 'FAIL'} "
                    f"steps={metrics.steps} len={metrics.path_length:.2f}m "
                    f"contacts={metrics.n_contacts} sensed={metrics.n_sensed}"
                    + (f" → {png_path}" if png_path else "")
                )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "planner",
                "family",
                "scene_idx",
                "seed",
                "success",
                "steps",
                "path_length",
                "n_contacts",
                "n_sensed",
                "png_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved metrics CSV to: {output_path}")


if __name__ == "__main__":
    main()
