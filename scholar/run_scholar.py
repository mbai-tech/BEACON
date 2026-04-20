import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from experiments.run_trials import run_trials
from utils.metrics import print_summary, aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SCHOLAR planning trials.")
    parser.add_argument("--trials",    type=int, default=10,
                        help="Number of trials to run.")
    parser.add_argument("--visualize", action="store_true",
                        help="Plot aggregate results after trials.")
    args = parser.parse_args()

    episodes = run_trials(n_trials=args.trials, planner_name="scholar")
    print_summary(aggregate(episodes), planner_name="scholar")

    if args.visualize:
        from experiments.visualize import plot_results
        plot_results(episodes, show=True)
