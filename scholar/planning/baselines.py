"""
baselines.py — Thin wrappers around existing NewProject planners.

Exposes run_bug (Bug2), run_rrt (online greedy RRT), and run_online_surp_push
(the full BEACON push planner) under a common interface for trial runners.
"""

from NewProject.bug_algorithm import run_bug
from NewProject.rrt_greedy import run_rrt
from NewProject.planner import run_online_surp_push
from NewProject.models import OnlineSurpResult

__all__ = ["run_bug", "run_rrt", "run_online_surp_push", "PLANNERS"]

# Registry used by run_trials to look up planners by name
PLANNERS = {
    "bug":    run_bug,
    "rrt":    run_rrt,
    "beacon": run_online_surp_push,
}
