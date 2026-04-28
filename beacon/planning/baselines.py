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
"""
baselines.py — Thin wrappers around existing core planners.

Exposes run_bug (Bug2), run_rrt (online greedy RRT), and run_surp_push
(the full SURP push planner) under a common interface for trial runners.
"""

from beacon.core.bug_algorithm import run_bug
from beacon.core.bug2_algorithm import run_bug2
from beacon.core.dstar_lite_algorithm import run_dstar_lite
from beacon.core.rrt_greedy import run_rrt
from beacon.core.planner import run_online_surp_push
from beacon.core.models import OnlineSurpResult

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
