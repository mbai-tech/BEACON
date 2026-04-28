"""Bug1 and Greedy baseline planners for the BEACON evaluation.

Bug2 is provided by the existing ``beacon.core.bug_algorithm.run_bug``.
"""

import numpy as np
from shapely.geometry import LineString, Point

from beacon.core.constants import DEFAULT_SENSING_RANGE, ROBOT_RADIUS
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


# ── shared helpers ────────────────────────────────────────────────────────────

def _collides_with_observed(scene: dict, position: np.ndarray) -> bool:
    body = robot_body(position)
    return any(
        obs["observed"] and body.intersects(obstacle_polygon(obs))
        for obs in scene["obstacles"]
    )


def _rotate_90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=float)


# ── Bug1 ──────────────────────────────────────────────────────────────────────

def run_bug1(
    scene: dict,
    sensing_range: float = DEFAULT_SENSING_RANGE,
    step_size: float = 0.07,
    max_steps: int = 1200,
) -> OnlineSurpResult:
    """Bug1: on contact, traverse the full obstacle boundary to find the closest
    point to the goal, then depart from there.  O(perimeter) per obstacle.
    """
    working_scene = normalize_scene_for_online_use(scene)
    position = np.array(working_scene["start"][:2], dtype=float)
    goal     = np.array(working_scene["goal"][:2],  dtype=float)

    path   = [tuple(position)]
    frames = [snapshot_frame(position, working_scene, "start")]
    sensed_ids: list[int] = []

    follow_boundary  = False
    hit_point: np.ndarray | None = None
    boundary_dir     = normalize(goal - position)
    boundary_steps   = 0
    MAX_BOUNDARY     = 600
    closest_dist     = float("inf")
    closest_pos: np.ndarray | None = None
    returning        = False

    success = False

    for _ in range(max_steps):
        if np.linalg.norm(goal - position) <= step_size:
            position = goal.copy()
            path.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "goal reached"))
            success = True
            break

        newly_observed = reveal_nearby_obstacles(working_scene, position, sensing_range)
        if newly_observed:
            sensed_ids.extend(obs["id"] for obs in newly_observed)
            path.append(tuple(position))
            frames.append(snapshot_frame(
                position, working_scene,
                f"sensed obstacle(s) {[o['id'] for o in newly_observed]}",
            ))
            continue

        # ── second leg: travel to closest point found during boundary scan ──
        if returning and closest_pos is not None:
            delta = closest_pos - position
            dist  = float(np.linalg.norm(delta))
            if dist <= step_size:
                position        = closest_pos.copy()
                returning       = False
                follow_boundary = False
                hit_point       = None
                path.append(tuple(position))
                frames.append(snapshot_frame(position, working_scene,
                                             "Bug1: at closest point — resuming goal mode"))
            else:
                cand = clip_point_to_workspace(
                    working_scene, position + normalize(delta) * min(step_size, dist))
                if not _collides_with_observed(working_scene, cand):
                    position = cand
                    path.append(tuple(position))
                    frames.append(snapshot_frame(position, working_scene,
                                                 "Bug1: travelling to closest point"))
                else:
                    returning       = False
                    follow_boundary = False
            continue

        # ── goal-seeking mode ────────────────────────────────────────────────
        if not follow_boundary:
            next_pos = clip_point_to_workspace(
                working_scene, position + step_size * normalize(goal - position))
            if _collides_with_observed(working_scene, next_pos):
                follow_boundary = True
                hit_point       = position.copy()
                boundary_dir    = _rotate_90(normalize(goal - position))
                boundary_steps  = 0
                closest_dist    = float(np.linalg.norm(goal - position))
                closest_pos     = position.copy()
                path.append(tuple(position))
                frames.append(snapshot_frame(position, working_scene,
                                             "Bug1: hit obstacle — scanning full boundary"))
            else:
                position = next_pos
                path.append(tuple(position))
                frames.append(snapshot_frame(position, working_scene, "goal mode"))

        # ── boundary-following (first leg: full scan) ────────────────────────
        else:
            d = float(np.linalg.norm(goal - position))
            if d < closest_dist:
                closest_dist = d
                closest_pos  = position.copy()

            # Loop completed?
            if (hit_point is not None and boundary_steps > 10
                    and float(np.linalg.norm(position - hit_point)) <= step_size * 2):
                returning = True
                path.append(tuple(position))
                frames.append(snapshot_frame(position, working_scene,
                                             "Bug1: boundary loop done — heading to closest"))
                continue

            if boundary_steps >= MAX_BOUNDARY:
                returning = True
                path.append(tuple(position))
                frames.append(snapshot_frame(position, working_scene,
                                             "Bug1: boundary cap — heading to closest"))
                continue

            moved = False
            for _ in range(4):
                next_pos = clip_point_to_workspace(
                    working_scene, position + step_size * boundary_dir)
                if not _collides_with_observed(working_scene, next_pos):
                    moved = True
                    break
                boundary_dir = _rotate_90(boundary_dir)

            if not moved:
                break

            position = next_pos
            boundary_steps += 1
            path.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "Bug1: boundary mode"))

    return OnlineSurpResult(
        family=scene.get("family", "unknown"),
        seed=scene.get("seed", 0),
        success=success,
        path=path,
        frames=frames,
        scene=working_scene,
        initial_scene=normalize_scene_for_online_use(scene),
        contact_log=[],
        sensed_ids=sensed_ids,
    )


# ── Greedy planner ────────────────────────────────────────────────────────────

def run_greedy(
    scene: dict,
    sensing_range: float = DEFAULT_SENSING_RANGE,
    step_size: float = 0.07,
    max_steps: int = 800,
) -> OnlineSurpResult:
    """Greedy: at each step pick the 8-connected action that most reduces
    Euclidean distance to the goal.  Colliding actions are rejected; if all
    8 actions collide the episode terminates with failure.
    """
    working_scene = normalize_scene_for_online_use(scene)
    position = np.array(working_scene["start"][:2], dtype=float)
    goal     = np.array(working_scene["goal"][:2],  dtype=float)

    path   = [tuple(position)]
    frames = [snapshot_frame(position, working_scene, "start")]
    sensed_ids: list[int] = []

    angles     = np.linspace(0, 2 * np.pi, 9)[:-1]
    directions = [np.array([np.cos(a), np.sin(a)]) for a in angles]

    success = False

    for _ in range(max_steps):
        if np.linalg.norm(goal - position) <= step_size:
            position = goal.copy()
            path.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "goal reached"))
            success = True
            break

        newly_observed = reveal_nearby_obstacles(working_scene, position, sensing_range)
        if newly_observed:
            sensed_ids.extend(obs["id"] for obs in newly_observed)
            path.append(tuple(position))
            frames.append(snapshot_frame(
                position, working_scene,
                f"sensed obstacle(s) {[o['id'] for o in newly_observed]}",
            ))
            continue

        def dist_after(d: np.ndarray) -> float:
            p = clip_point_to_workspace(working_scene, position + step_size * d)
            return float(np.linalg.norm(goal - p))

        moved = False
        for d in sorted(directions, key=dist_after):
            cand = clip_point_to_workspace(working_scene, position + step_size * d)
            if not _collides_with_observed(working_scene, cand):
                position = cand
                path.append(tuple(position))
                frames.append(snapshot_frame(position, working_scene, "greedy move"))
                moved = True
                break

        if not moved:
            frames.append(snapshot_frame(position, working_scene,
                                         "greedy stuck — all directions blocked"))
            break

    return OnlineSurpResult(
        family=scene.get("family", "unknown"),
        seed=scene.get("seed", 0),
        success=success,
        path=path,
        frames=frames,
        scene=working_scene,
        initial_scene=normalize_scene_for_online_use(scene),
        contact_log=[],
        sensed_ids=sensed_ids,
    )
