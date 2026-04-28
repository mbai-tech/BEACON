"""
main.py — Entry point for BEACON experiments.

Run from the repo root:
    python -m beacon.main --planner beacon --trials 10 --visualize
or:
    python beacon/main.py --planner bug --trials 5
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

from beacon.experiments.run_trials import run_trials, ALL_PLANNERS
from beacon.utils.metrics import print_summary, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BEACON planning trials.")
    parser.add_argument("--trials",    type=int, default=10,
                        help="Number of trials to run.")
    parser.add_argument("--planner",   type=str, default="beacon",
                        help=f"Planner to use: {sorted(ALL_PLANNERS)}")
    parser.add_argument("--visualize", action="store_true",
                        help="Plot aggregate results after trials.")
    args = parser.parse_args()

    episodes = run_trials(n_trials=args.trials, planner_name=args.planner)
    print_summary(aggregate(episodes), planner_name=args.planner)

    if args.visualize:
        from beacon.experiments.visualize import plot_results
        plot_results(episodes, show=True)


if __name__ == "__main__":
    main()
