"""
baselines.py — Thin wrappers around existing core planners.

Exposes run_bug (Bug2), run_rrt (online greedy RRT), and run_surp_push
(the full SURP push planner) under a common interface for trial runners.
"""

from scholar.core.bug_algorithm import run_bug
from scholar.core.bug2_algorithm import run_bug2
from scholar.core.dstar_lite_algorithm import run_dstar_lite
from scholar.core.rrt_greedy import run_rrt
from scholar.core.planner import run_online_surp_push
from scholar.core.models import OnlineSurpResult

__all__ = [
    "run_bug",
    "run_bug2",
    "run_dstar_lite",
    "run_rrt",
    "run_online_surp_push",
    "PLANNERS",
]

# Registry used by run_trials to look up planners by name
PLANNERS = {
    "bug": run_bug,
    "bug2": run_bug2,
    "dstar_lite": run_dstar_lite,
    "rrt": run_rrt,
    "surp": run_online_surp_push,
}
