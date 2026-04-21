"""
baselines.py — Planner registry for trial runners.
"""

from scholar.algorithms.bug import run_bug
from scholar.algorithms.bug2 import run_bug2
from scholar.algorithms.baselines import run_bug1, run_dstar
from scholar.algorithms.rrt import run_rrt
from scholar.algorithms.surp import run_online_surp_push

__all__ = [
    "run_bug",
    "run_bug1",
    "run_bug2",
    "run_dstar",
    "run_rrt",
    "run_online_surp_push",
    "PLANNERS",
]

PLANNERS = {
    "bug":   run_bug,
    "bug1":  run_bug1,
    "bug2":  run_bug2,
    "dstar": run_dstar,
    "rrt":   run_rrt,
    "surp":  run_online_surp_push,
}
