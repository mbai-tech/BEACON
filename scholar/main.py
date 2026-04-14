"""
main.py — Entry point for SCHOLAR experiments.
"""

import argparse

from experiments.run_trials import run_trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SCHOLAR planning trials.")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials to run.")
    parser.add_argument("--planner", type=str, default="scholar", help="Planner to use: scholar | bug1 | bug2 | greedy.")
    parser.add_argument("--visualize", action="store_true", help="Plot results after trials.")
    args = parser.parse_args()

    results = run_trials(n_trials=args.trials, planner_name=args.planner)

    if args.visualize:
        from experiments.visualize import plot_results
        plot_results(results)


if __name__ == "__main__":
    main()
