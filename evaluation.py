"""SCHOLAR branch evaluation runner.

Same grid as the New-alg branch, but the SCHOLAR algorithm slot uses
the full ``run_scholar`` implementation from scholar/planning/scholar.py
instead of run_online_surp_push.

Usage
-----
    cd ../surp-SCHOLAR
    /Users/rheas/surp/venv/bin/python3 -u evaluation.py --trials 10
    /Users/rheas/surp/venv/bin/python3 -u evaluation.py --trials 10 --configs D-M
"""

import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

# scholar/ sub-package needs to be on the path for its internal imports
REPO_ROOT   = Path(__file__).resolve().parent
SCHOLAR_DIR = REPO_ROOT / "scholar"
for p in (str(REPO_ROOT), str(SCHOLAR_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from NewProject.baselines import run_bug1, run_greedy
from NewProject.bug_algorithm import run_bug                 # Bug2
from NewProject.constants import OUTPUT_DIR, SCENE_CONFIGS
from NewProject.scene_configs import generate_config_environment
from metrics import compute_metrics, RunMetrics
from planning.scholar import run_scholar                     # full SCHOLAR

ALGORITHMS: dict = {
    "SCHOLAR": run_scholar,
    "Bug1":    run_bug1,
    "Bug2":    run_bug,
    "Greedy":  run_greedy,
}

ALL_CONFIGS = list(SCENE_CONFIGS.keys())
N_TRIALS    = 50


def run_batch(
    n_trials: int         = N_TRIALS,
    configs:  list | None = None,
) -> list[RunMetrics]:
    configs = configs or ALL_CONFIGS
    all_metrics: list[RunMetrics] = []
    total = len(configs) * n_trials
    done  = 0

    for config in configs:
        for trial in range(n_trials):
            scene = generate_config_environment(config)

            for alg_name, alg_fn in ALGORITHMS.items():
                t0      = time.perf_counter()
                result  = alg_fn(scene)
                elapsed = time.perf_counter() - t0

                m = compute_metrics(
                    result,
                    algorithm       = alg_name,
                    config          = config,
                    trial           = trial,
                    planning_time_s = elapsed,
                )
                all_metrics.append(m)

            done += 1
            print(f"[{done}/{total}]  config={config}  trial={trial + 1}/{n_trials}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "scholar_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(all_metrics[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(m) for m in all_metrics)

    print(f"\nSaved {len(all_metrics)} rows → {csv_path}")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SCHOLAR branch evaluation grid")
    parser.add_argument("--trials",  type=int, default=N_TRIALS)
    parser.add_argument("--configs", nargs="+", default=None, metavar="CFG")
    args = parser.parse_args()
    run_batch(n_trials=args.trials, configs=args.configs)
