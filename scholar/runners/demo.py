# -*- coding: utf-8 -*-
"""Single-episode demo runner.  Use --planner bug|bug2|rrt|surp."""
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scholar.core.constants import OUTPUT_DIR
from scholar.utils.scene_setup import generate_one_random_environment
from scholar.utils.visualization import animate_result, plot_final_snapshot, save_scene_snapshot

_PLANNERS = {
    "bug":  ("scholar.algorithms.bug",  "run_bug"),
    "bug2": ("scholar.algorithms.bug2", "run_bug2"),
    "rrt":  ("scholar.algorithms.rrt",  "run_rrt"),
    "surp": ("scholar.algorithms.surp", "run_online_surp_push"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-episode demo for any planner.")
    parser.add_argument("--planner", choices=list(_PLANNERS), default="surp",
                        help="Planner to run (default: surp)")
    args = parser.parse_args()

    module_path, fn_name = _PLANNERS[args.planner]
    import importlib
    run_fn = getattr(importlib.import_module(module_path), fn_name)

    scene  = generate_one_random_environment()
    result = run_fn(scene)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scene_path = save_scene_snapshot(result.scene, result.family, result.seed)
    image_path = plot_final_snapshot(result)
    animate_result(result)

    print(f"Planner: {args.planner}")
    print(f"Family:  {result.family}")
    print(f"Seed:    {result.seed}")
    print(f"Success: {result.success}")
    print(f"Steps:   {len(result.path)}")
    print(f"Sensed:  {result.sensed_ids}")
    print(f"Saved scene to: {scene_path}")
    print(f"Saved image to: {image_path}")


if __name__ == "__main__":
    main()
