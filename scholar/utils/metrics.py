from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from NewProject.models import OnlineSurpResult


@dataclass
class EpisodeMetrics:
    """Per-episode summary produced by ``compute_metrics``."""

    planner:     str
    family:      str
    seed:        int
    success:     bool
    steps:       int          # total frames recorded (inc. sense/stop events)
    path_length: float        # cumulative Euclidean distance travelled (metres)
    n_contacts:  int          # number of push / contact events logged
    n_sensed:    int          # number of unique obstacles discovered


def compute_metrics(result: OnlineSurpResult, planner_name: str) -> EpisodeMetrics:
    pts = np.array(result.path, dtype=float)
    if len(pts) >= 2:
        path_length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    else:
        path_length = 0.0

    return EpisodeMetrics(
        planner=planner_name,
        family=result.family,
        seed=result.seed,
        success=result.success,
        steps=len(result.frames),
        path_length=path_length,
        n_contacts=len(result.contact_log),
        n_sensed=len(set(result.sensed_ids)),
    )
