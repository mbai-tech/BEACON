"""Run metrics for multiple planners on the SCHOLAR scene_complex scenes.

This script uses ``scholar.demo_scholar.load_scene(...)`` so the evaluated
scenes match the same scene source used by the main SCHOLAR demo CLI.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCHOLAR_ROOT = REPO_ROOT / "scholar"
if str(SCHOLAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SCHOLAR_ROOT))
ENV_ROOT = REPO_ROOT / "enviornment"
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from scene_complex import generate_scene as _generate_complex_scene

from scholar.demo_scholar import FAMILIES, load_scene
from scholar.algorithms.bug import run_bug
from scholar.algorithms.bug2 import run_bug2
from scholar.algorithms.rrt import run_rrt
from scholar.algorithms.surp import run_online_surp_push
from scholar.planning.scholar import run_scholar
from scholar.utils.metrics import compute_metrics

PLANNERS = {
    "bug":  run_bug,
    "bug2": run_bug2,
    "rrt":  run_rrt,
    "surp": run_online_surp_push,
}


ALL_PLANNERS = sorted([*PLANNERS.keys(), "scholar"])

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "scholar"
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
        description="Run one or more planners on the scene_complex SCHOLAR scenes and save metrics to CSV."
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
        help="Restrict evaluation to specific SCHOLAR families.",
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["scholar", "bug", "rrt", "surp"],
        choices=ALL_PLANNERS,
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
        help="Sensing radius for SCHOLAR/SURP-style planners.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.04,
        help="Step size for SCHOLAR/SURP-style planners.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="CSV output path.",
    )
    return parser.parse_args()


def run_planner(planner_name: str, scene: dict, max_steps: int, step_size: float, sensing_range: float):
    if planner_name == "scholar":
        try:
            from scholar.planning.scholar import run_scholar
        except ModuleNotFoundError:
            from planning.scholar import run_scholar
        return run_scholar(scene, max_steps=max_steps, step_size=step_size, sensing_range=sensing_range)
    planner_fn = PLANNERS[planner_name]
    if planner_name == "surp":
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


def main() -> None:
    args = parse_args()
    families = args.family or FAMILIES
    scene_indices = parse_scene_indices(args.scene, args.scenes)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                }
                rows.append(row)
                print(
                    f"{'OK' if metrics.success else 'FAIL'} "
                    f"steps={metrics.steps} len={metrics.path_length:.2f}m "
                    f"contacts={metrics.n_contacts} sensed={metrics.n_sensed}"
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved metrics CSV to: {output_path}")


if __name__ == "__main__":
    main()
