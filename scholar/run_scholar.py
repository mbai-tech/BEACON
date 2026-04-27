import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from scholar.experiments.run_trials import run_trials
from scholar.utils.metrics import print_summary, aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BEACON planning trials.")
    parser.add_argument("--trials",    type=int, default=10,
                        help="Number of trials to run.")
    parser.add_argument("--visualize", action="store_true",
                        help="Plot aggregate results after trials.")
    args = parser.parse_args()

    episodes = run_trials(n_trials=args.trials, planner_name="beacon")
    print_summary(aggregate(episodes), planner_name="beacon")

    if args.visualize:
        from scholar.experiments.visualize import plot_results
        plot_results(episodes, show=True)
