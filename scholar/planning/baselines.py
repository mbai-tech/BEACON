"""
baselines.py — Bug1, Bug2, and Greedy (straight-line) baselines.

All three expose a common step(position, goal, scene) interface and return
(new_position, reached_goal, state_dict).
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from shapely.geometry import LineString

from env.robot import ROBOT_RADIUS


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _rotate_90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=float)


def _collides(position: np.ndarray, obstacles: list) -> bool:
    from shapely.geometry import Point
    body = Point(position[0], position[1]).buffer(ROBOT_RADIUS)
    for obs in obstacles:
        cx, cy = obs.position[0], obs.position[1]
        dx, dy = obs.half_extents[0], obs.half_extents[1]
        from shapely.geometry import box as shapely_box
        obs_poly = shapely_box(cx - dx, cy - dy, cx + dx, cy + dy)
        if body.intersects(obs_poly):
            return True
    return False


def _clear_line(position: np.ndarray, goal: np.ndarray, obstacles: list) -> bool:
    line = LineString([position.tolist(), goal.tolist()]).buffer(ROBOT_RADIUS)
    from shapely.geometry import box as shapely_box
    for obs in obstacles:
        cx, cy = obs.position[0], obs.position[1]
        dx, dy = obs.half_extents[0], obs.half_extents[1]
        obs_poly = shapely_box(cx - dx, cy - dy, cx + dx, cy + dy)
        if line.intersects(obs_poly):
            return False
    return True


# ── Greedy baseline ────────────────────────────────────────────────────────────

class GreedyPlanner:
    """Straight-line goal-directed planner with no obstacle avoidance."""

    def __init__(self, step_size: float = 0.15):
        self.step_size = step_size

    def step(
        self,
        position: np.ndarray,
        goal: np.ndarray,
        obstacles: list,
    ) -> Tuple[np.ndarray, bool]:
        direction = _normalize(goal - position)
        dist = float(np.linalg.norm(goal - position))
        if dist <= self.step_size:
            return goal.copy(), True
        new_pos = position + direction * self.step_size
        return new_pos, False


# ── Bug1 baseline ──────────────────────────────────────────────────────────────

class Bug1Planner:
    """Bug1: fully circumnavigate each obstacle, leave from the closest point to goal."""

    def __init__(self, step_size: float = 0.07):
        self.step_size = step_size
        self._follow_boundary = False
        self._hit_point: Optional[np.ndarray] = None
        self._boundary_direction: Optional[np.ndarray] = None
        self._best_point: Optional[np.ndarray] = None
        self._best_dist: float = float("inf")
        self._full_loop: bool = False

    def step(
        self,
        position: np.ndarray,
        goal: np.ndarray,
        obstacles: list,
    ) -> Tuple[np.ndarray, bool]:
        dist_to_goal = float(np.linalg.norm(goal - position))
        if dist_to_goal <= self.step_size:
            return goal.copy(), True

        if not self._follow_boundary:
            direction = _normalize(goal - position)
            next_pos = position + self.step_size * direction
            if not _collides(next_pos, obstacles):
                return next_pos, False
            # Hit an obstacle — start boundary following
            self._follow_boundary = True
            self._hit_point = position.copy()
            self._boundary_direction = _rotate_90(direction)
            self._best_point = position.copy()
            self._best_dist = dist_to_goal
            self._full_loop = False
        else:
            # Track closest point to goal
            if dist_to_goal < self._best_dist:
                self._best_dist = dist_to_goal
                self._best_point = position.copy()

            # Check if we've completed a full loop (returned to hit_point)
            if (self._best_point is not None and
                float(np.linalg.norm(position - self._hit_point)) < self.step_size * 1.5 and
                not self._full_loop):
                # Re-navigate to best point, then leave
                self._full_loop = True

            if self._full_loop and self._best_point is not None:
                direction = _normalize(self._best_point - position)
                next_pos = position + self.step_size * direction
                if float(np.linalg.norm(next_pos - self._best_point)) < self.step_size:
                    self._follow_boundary = False
                    return self._best_point.copy(), False

            moved = False
            for _ in range(4):
                next_pos = position + self.step_size * self._boundary_direction
                if not _collides(next_pos, obstacles):
                    moved = True
                    break
                self._boundary_direction = _rotate_90(self._boundary_direction)

            if not moved:
                return position.copy(), False
            return next_pos, False

        return position.copy(), False


# ── Bug2 baseline ──────────────────────────────────────────────────────────────

class Bug2Planner:
    """Bug2: follow M-line; leave obstacle when M-line is clear to goal."""

    def __init__(self, step_size: float = 0.07):
        self.step_size = step_size
        self._follow_boundary = False
        self._hit_point: Optional[np.ndarray] = None
        self._boundary_direction: np.ndarray = np.array([1.0, 0.0])

    def step(
        self,
        position: np.ndarray,
        goal: np.ndarray,
        obstacles: list,
    ) -> Tuple[np.ndarray, bool]:
        dist_to_goal = float(np.linalg.norm(goal - position))
        if dist_to_goal <= self.step_size:
            return goal.copy(), True

        if not self._follow_boundary:
            direction = _normalize(goal - position)
            next_pos = position + self.step_size * direction
            if not _collides(next_pos, obstacles):
                return next_pos, False
            # Hit obstacle
            self._follow_boundary = True
            self._hit_point = position.copy()
            self._boundary_direction = _rotate_90(direction)
            return position.copy(), False
        else:
            moved_away = (
                self._hit_point is None
                or float(np.linalg.norm(position - self._hit_point)) > self.step_size
            )
            if moved_away and _clear_line(position, goal, obstacles):
                self._follow_boundary = False
                return position.copy(), False

            for _ in range(4):
                next_pos = position + self.step_size * self._boundary_direction
                if not _collides(next_pos, obstacles):
                    return next_pos, False
                self._boundary_direction = _rotate_90(self._boundary_direction)

            return position.copy(), False
