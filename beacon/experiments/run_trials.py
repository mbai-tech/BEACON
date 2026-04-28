"""
experiments/run_trials.py — Trial runner across scene configurations.

Runs a named planner on N randomly generated scenes and returns
a list of EpisodeMetrics.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beacon.core.scene_setup import generate_one_random_environment

from beacon.planning.baselines import PLANNERS
from beacon.planning.beacon_planner import run_beacon
from beacon.utils.metrics import EpisodeMetrics, compute_metrics

# Register the BEACON planner alongside the baselines
ALL_PLANNERS = {**PLANNERS, "beacon": run_beacon}


def run_trials(
    n_trials: int = 10,
    planner_name: str = "beacon",
) -> list[EpisodeMetrics]:
    """Run `planner_name` on `n_trials` random scenes.

    Returns a list of EpisodeMetrics, one per trial.
    """
    planner_fn = ALL_PLANNERS.get(planner_name)
    if planner_fn is None:
        raise ValueError(
            f"Unknown planner '{planner_name}'. "
            f"Choose from: {sorted(ALL_PLANNERS)}"
        )

    results = []
    for i in range(n_trials):
        scene = generate_one_random_environment()
        print(f"[{i+1}/{n_trials}] family={scene.get('family','?')} seed={scene.get('seed',0)} ...",
              end=" ", flush=True)
        result = planner_fn(scene)
        metrics = compute_metrics(result, planner_name)
        results.append(metrics)
        status = "OK" if metrics.success else "FAIL"
        print(f"{status}  steps={metrics.steps}  len={metrics.path_length:.2f}m")

    return results
