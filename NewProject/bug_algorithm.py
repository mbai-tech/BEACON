import numpy as np
from shapely.geometry import LineString, Point

from NewProject.constants import DEFAULT_SENSING_RANGE, ROBOT_RADIUS
from NewProject.models import OnlineSurpResult
from NewProject.planner import (
    clip_point_to_workspace,
    normalize,
    obstacle_polygon,
    reveal_nearby_obstacles,
    robot_body,
    snapshot_frame,
)
from NewProject.scene_setup import normalize_scene_for_online_use


def _collides_with_observed(scene: dict, position: np.ndarray) -> bool:
    """Check whether the robot disk intersects any currently observed obstacle."""
    body = robot_body(position)
    return any(
        obs["observed"] and body.intersects(obstacle_polygon(obs))
        for obs in scene["obstacles"]
    )


def _clear_line_to_goal(scene: dict, position: np.ndarray, goal: np.ndarray) -> bool:
    """Check line-of-sight to the goal through only the observed local map."""
    line = LineString([position.tolist(), goal.tolist()]).buffer(ROBOT_RADIUS)
    return not any(
        obs["observed"] and line.intersects(obstacle_polygon(obs))
        for obs in scene["obstacles"]
    )


def _rotate_90(v: np.ndarray) -> np.ndarray:
    """Rotate a 2D vector 90 degrees counterclockwise."""
    return np.array([-v[1], v[0]], dtype=float)


def run_bug(
    scene: dict,
    sensing_range: float = DEFAULT_SENSING_RANGE,
    step_size: float = 0.07,
    max_steps: int = 800,
) -> OnlineSurpResult:
    """Run a simple Bug-style baseline on the same scene format as the main demo.

    This gives you a classical no-push comparison algorithm:
    - move directly toward the goal,
    - when an observed obstacle blocks that motion, follow its boundary,
    - leave the boundary once a clear line to the goal is available again.
    """
    working_scene = normalize_scene_for_online_use(scene)
    position = np.array(working_scene["start"][:2], dtype=float)
    goal = np.array(working_scene["goal"][:2], dtype=float)

    path = [tuple(position)]
    frames = [snapshot_frame(position, working_scene, "start")]
    sensed_ids: list[int] = []

    follow_boundary = False
    hit_point: np.ndarray | None = None
    boundary_direction: np.ndarray = normalize(goal - position)

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
            observed_ids = [obs["id"] for obs in newly_observed]
            sensed_ids.extend(observed_ids)
            path.append(tuple(position))
            frames.append(snapshot_frame(
                position,
                working_scene,
                f"sensed obstacle(s) {observed_ids}",
            ))
            continue

        if not follow_boundary:
            goal_direction = normalize(goal - position)
            next_pos = clip_point_to_workspace(
                working_scene, position + step_size * goal_direction
            )

            if _collides_with_observed(working_scene, next_pos):
                follow_boundary = True
                hit_point = position.copy()
                boundary_direction = _rotate_90(goal_direction)
                path.append(tuple(position))
                frames.append(snapshot_frame(
                    position,
                    working_scene,
                    "obstacle hit - switching to boundary mode",
                ))
            else:
                position = next_pos
                path.append(tuple(position))
                frames.append(snapshot_frame(position, working_scene, "goal mode"))

        else:
            moved = False
            for _ in range(4):
                next_pos = clip_point_to_workspace(
                    working_scene, position + step_size * boundary_direction
                )
                if not _collides_with_observed(working_scene, next_pos):
                    moved = True
                    break
                boundary_direction = _rotate_90(boundary_direction)

            if not moved:
                break

            moved_away = float(np.linalg.norm(position - hit_point)) > step_size if hit_point is not None else True
            if moved_away and _clear_line_to_goal(working_scene, position, goal):
                follow_boundary = False
                path.append(tuple(position))
                frames.append(snapshot_frame(
                    position,
                    working_scene,
                    "clear path to goal - resuming goal mode",
                ))
                continue

            position = next_pos
            path.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "boundary mode"))

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
