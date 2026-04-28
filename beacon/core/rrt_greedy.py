"""
rrt_greedy.py
Online greedy RRT: plans only over currently observed obstacles,
replans whenever new obstacles are discovered.

At t=0 only the start position and obstacles within sensing_range are known.
As the robot moves, newly sensed obstacles trigger a replan from the current
position using the updated observed map.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import numpy as np

from beacon.core.constants import DEFAULT_SENSING_RANGE
from beacon.core.models import OnlineSurpResult
from beacon.core.planner import (
    clip_point_to_workspace,
    normalize,
    obstacle_polygon,
    reveal_nearby_obstacles,
    robot_body,
    snapshot_frame,
)
from beacon.core.scene_setup import normalize_scene_for_online_use

def _collides_observed(scene: dict, position: np.ndarray) -> bool:
    body = robot_body(position)
    return any(
        obs["observed"] and body.intersects(obstacle_polygon(obs))
        for obs in scene["obstacles"]
    )


def _segment_free_observed(scene: dict, a: np.ndarray, b: np.ndarray, subdivisions: int = 20) -> bool:
    for t in np.linspace(0.0, 1.0, subdivisions + 1):
        pt = clip_point_to_workspace(scene, a + (b - a) * t)
        if _collides_observed(scene, pt):
            return False
    return True


def _nearest(nodes: list[np.ndarray], target: np.ndarray) -> int:
    return int(np.argmin([float(np.linalg.norm(n - target)) for n in nodes]))


def _sample(scene: dict, goal: np.ndarray, goal_bias: float) -> np.ndarray:
    if random.random() < goal_bias:
        return goal.copy()
    xmin, xmax, ymin, ymax = scene["workspace"]
    return np.array([random.uniform(xmin, xmax), random.uniform(ymin, ymax)], dtype=float)


def _build_rrt(
    scene: dict,
    start: np.ndarray,
    goal: np.ndarray,
    step_size: float,
    goal_bias: float,
    max_samples: int,
    goal_tolerance: float,
) -> list[np.ndarray] | None:
    """Build RRT over observed obstacles only. Returns path or None."""
    nodes = [start.copy()]
    parents = [-1]

    for _ in range(max_samples):
        sample = _sample(scene, goal, goal_bias)
        nearest_idx = _nearest(nodes, sample)
        nearest = nodes[nearest_idx]

        direction = normalize(sample - nearest)
        if np.linalg.norm(direction) <= 1e-9:
            continue

        candidate = clip_point_to_workspace(scene, nearest + direction * step_size)
        if not _segment_free_observed(scene, nearest, candidate):
            continue

        nodes.append(candidate)
        parents.append(nearest_idx)
        new_idx = len(nodes) - 1

        if (
            float(np.linalg.norm(candidate - goal)) <= goal_tolerance
            and _segment_free_observed(scene, candidate, goal)
        ):
            nodes.append(goal.copy())
            parents.append(new_idx)
            goal_idx = len(nodes) - 1
            path = []
            idx = goal_idx
            while idx != -1:
                path.append(nodes[idx])
                idx = parents[idx]
            path.reverse()
            return path

    return None


def _smooth_path(scene: dict, path: list[np.ndarray], passes: int = 3) -> list[np.ndarray]:
    """Shortcut smoothing over observed obstacles."""
    for _ in range(passes):
        i = 0
        smoothed = [path[i]]
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if _segment_free_observed(scene, path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        path = smoothed
    return path


def _path_still_valid(scene: dict, path: list[np.ndarray]) -> bool:
    """Check that no newly observed obstacle blocks the remaining planned path."""
    for i in range(len(path) - 1):
        if not _segment_free_observed(scene, path[i], path[i + 1]):
            return False
    return True


# ── Main entry point ─────────────────────────────────────────────────────────

def run_rrt(
    scene: dict,
    sensing_range: float = DEFAULT_SENSING_RANGE,
    step_size: float = 0.15,
    goal_bias: float = 0.15,
    max_samples: int = 5000,
    max_steps: int = 800,
    smooth: bool = True,
) -> OnlineSurpResult:
    """Online greedy RRT: plan over observed map, replan when new obstacles appear.

    Parameters
    ----------
    scene         : standard scene dict
    sensing_range : radius within which obstacles are revealed each step
    step_size     : RRT expansion distance
    goal_bias     : probability of sampling the goal directly (greedy bias)
    max_samples   : RRT iteration budget per plan/replan
    max_steps     : total execution step limit
    smooth        : apply shortcut smoothing after each plan
    """
    working_scene = normalize_scene_for_online_use(scene)
    position = np.array(working_scene["start"][:2], dtype=float)
    goal = np.array(working_scene["goal"][:2], dtype=float)

    path_taken = [tuple(position)]
    frames = [snapshot_frame(position, working_scene, "start")]
    sensed_ids: list[int] = []
    goal_tolerance = max(step_size, 0.15)

    # ── Initial sense from start position ────────────────────────────────────
    newly_observed = reveal_nearby_obstacles(working_scene, position, sensing_range)
    if newly_observed:
        sensed_ids.extend(obs["id"] for obs in newly_observed)

    # ── Initial plan ─────────────────────────────────────────────────────────
    rrt_path = _build_rrt(working_scene, position, goal,
                          step_size, goal_bias, max_samples, goal_tolerance)
    if rrt_path and smooth:
        rrt_path = _smooth_path(working_scene, rrt_path)

    if rrt_path:
        frames.append(snapshot_frame(position, working_scene,
                                     f"initial RRT: {len(rrt_path)} waypoints"))
    else:
        frames.append(snapshot_frame(position, working_scene,
                                     "initial RRT failed — will retry as obstacles revealed"))

    waypoint_idx = 1
    success = False

    for _ in range(max_steps):
        if np.linalg.norm(goal - position) <= step_size:
            position = goal.copy()
            path_taken.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "goal reached"))
            success = True
            break

        # Reveal obstacles within sensing range
        newly_observed = reveal_nearby_obstacles(working_scene, position, sensing_range)
        if newly_observed:
            observed_ids = [obs["id"] for obs in newly_observed]
            sensed_ids.extend(observed_ids)
            path_taken.append(tuple(position))
            frames.append(snapshot_frame(
                position, working_scene, f"sensed obstacle(s) {observed_ids} — replanning",
            ))

            # Replan from current position with updated observed map
            remaining_goal = goal if rrt_path is None else goal
            new_path = _build_rrt(working_scene, position, goal,
                                   step_size, goal_bias, max_samples, goal_tolerance)
            if new_path and smooth:
                new_path = _smooth_path(working_scene, new_path)
            if new_path:
                rrt_path = new_path
                waypoint_idx = 1
                frames.append(snapshot_frame(position, working_scene,
                                             f"replan: {len(rrt_path)} waypoints"))
            continue

        # No path available — wait for more sensing
        if rrt_path is None or waypoint_idx >= len(rrt_path):
            path_taken.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "no path — waiting"))
            continue

        # Step toward current waypoint
        waypoint = rrt_path[waypoint_idx]
        direction = normalize(waypoint - position)
        move = min(step_size, float(np.linalg.norm(waypoint - position)))
        position = clip_point_to_workspace(working_scene, position + direction * move)

        path_taken.append(tuple(position))
        frames.append(snapshot_frame(position, working_scene,
                                     f"waypoint {waypoint_idx}/{len(rrt_path)-1}"))

        if float(np.linalg.norm(position - waypoint)) < step_size * 0.5:
            waypoint_idx += 1

    return OnlineSurpResult(
        family=scene.get("family", "unknown"),
        seed=scene.get("seed", 0),
        success=success,
        path=path_taken,
        frames=frames,
        scene=working_scene,
        initial_scene=normalize_scene_for_online_use(scene),
        contact_log=[],
        sensed_ids=sensed_ids,
    )
