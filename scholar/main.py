"""
main.py — Entry point for SCHOLAR experiments.

Run from the repo root:
    python -m scholar.main --planner scholar --trials 10 --visualize
or:
    python scholar/main.py --planner bug --trials 5
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

from experiments.run_trials import run_trials, ALL_PLANNERS
from utils.metrics import print_summary, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SCHOLAR planning trials.")
    parser.add_argument("--trials",    type=int, default=10,
                        help="Number of trials to run.")
    parser.add_argument("--planner",   type=str, default="scholar",
                        help=f"Planner to use: {sorted(ALL_PLANNERS)}")
    parser.add_argument("--visualize", action="store_true",
                        help="Plot aggregate results after trials.")
    args = parser.parse_args()

    episodes = run_trials(n_trials=args.trials, planner_name=args.planner)
    print_summary(aggregate(episodes), planner_name=args.planner)

    if args.visualize:
        from experiments.visualize import plot_results
        plot_results(episodes, show=True)


if __name__ == "__main__":
    main()
