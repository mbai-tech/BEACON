"""One-off cluttered-scene ablation runner for the geometric-only BEACON proxy."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCHOLAR_ROOT = REPO_ROOT / "scholar"
if str(SCHOLAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SCHOLAR_ROOT))

from scholar.experiments.run_scene_complex_metrics import load_scene
from scholar.planning.scholar import PlannerConfig, run_scholar
from scholar.utils.metrics import compute_metrics


def _run_one(scene_idx: int) -> dict:
    scene = load_scene(scene_idx, family="cluttered", seed=scene_idx)
    config = PlannerConfig(
        geo_weight=0.799999,
        sem_weight=0.000001,
        dir_weight=0.2,
    )
    result = run_scholar(
        scene,
        max_steps=500,
        step_size=0.04,
        sensing_range=0.35,
        config=config,
    )
    metrics = compute_metrics(result, "beacon_geometric_only")
    return asdict(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=99)
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()))
    args = parser.parse_args()

    scene_indices = list(range(args.start, args.end + 1))
    with Pool(processes=args.workers) as pool:
        rows = pool.map(_run_one, scene_indices)

    success_rate = 100.0 * np.mean([row["success"] for row in rows])
    avg_steps = float(np.mean([row["steps"] for row in rows]))
    avg_path = float(np.mean([row["path_length"] for row in rows]))
    avg_path_success = float(
        np.mean([row["path_length"] for row in rows if row["success"]])
    )

    print(f"episodes={len(rows)}")
    print(f"success_rate={success_rate:.2f}%")
    print(f"avg_steps={avg_steps:.2f}")
    print(f"avg_path={avg_path:.3f}")
    print(f"avg_path_success={avg_path_success:.3f}")


if __name__ == "__main__":
    main()
