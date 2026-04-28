import numpy as np

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

# How close to the m-line (perpendicular distance) counts as "on it"
MLINE_THRESHOLD = 0.15   # metres
# Must be this much closer to goal than hit_distance to leave the wall
LEAVE_MARGIN    = 0.1    # metres


def _collides_with_observed(scene: dict, position: np.ndarray) -> bool:
    body = robot_body(position)
    return any(
        obs["observed"] and body.intersects(obstacle_polygon(obs))
        for obs in scene["obstacles"]
    )


def _rotate_90(v: np.ndarray) -> np.ndarray:
    """Rotate 2-D vector 90° counter-clockwise."""
    return np.array([-v[1], v[0]], dtype=float)


def _rotate(v: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return normalize(np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]]))


def _best_boundary_direction(
    scene: dict,
    position: np.ndarray,
    current_direction: np.ndarray,
    step_size: float,
    n_sweep: int = 36,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Left-hand boundary following: sweep from -90° CCW and take first free step."""
    step_deg = 360.0 / n_sweep
    for i in range(n_sweep):
        angle = -90.0 + i * step_deg
        candidate = _rotate(current_direction, angle)
        next_pos = clip_point_to_workspace(scene, position + step_size * candidate)
        if not _collides_with_observed(scene, next_pos):
            return candidate, next_pos
    return None, None


def _dist_to_mline(position: np.ndarray, start: np.ndarray, goal: np.ndarray) -> float:
    """Perpendicular distance from position to the line through start and goal."""
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    den = np.sqrt(dx ** 2 + dy ** 2)
    if den == 0:
        return float("inf")
    num = abs(
        dy * position[0]
        - dx * position[1]
        + goal[0] * start[1]
        - goal[1] * start[0]
    )
    return num / den


def _on_mline_segment(
    position: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    threshold: float = MLINE_THRESHOLD,
) -> bool:
    """True if position is within threshold of the m-line and between start and goal."""
    if _dist_to_mline(position, start, goal) >= threshold:
        return False
    # Projection parameter t: 0 at start, 1 at goal
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    denom = dx ** 2 + dy ** 2
    if denom == 0:
        return False
    t = ((position[0] - start[0]) * dx + (position[1] - start[1]) * dy) / denom
    return 0.0 <= t <= 1.0


def run_bug2(
    scene: dict,
    sensing_range: float = DEFAULT_SENSING_RANGE,
    step_size: float = 0.07,
    max_steps: int = 800,
) -> OnlineSurpResult:
    """
    Bug2 algorithm.

    Difference from Bug1:
      Bug1 – traverse the entire obstacle boundary, note the closest point to
             the goal, then leave from that point.
      Bug2 – draw the m-line (start→goal) once. When the boundary is hit,
             wall-follow until the m-line is crossed again at a point strictly
             closer to the goal than the hit point, then leave immediately.
    """
    working_scene = normalize_scene_for_online_use(scene)
    start    = np.array(working_scene["start"][:2], dtype=float)
    goal     = np.array(working_scene["goal"][:2],  dtype=float)
    position = start.copy()

    path       = [tuple(position)]
    frames     = [snapshot_frame(position, working_scene, "start")]
    sensed_ids: list[int] = []

    follow_boundary    = False
    hit_point: np.ndarray | None = None
    hit_distance       = float("inf")  # distance to goal at the time of wall hit
    boundary_direction = normalize(goal - position)

    success = False

    for _ in range(max_steps):
        # ── Goal reached ─────────────────────────────────────────────────────
        if np.linalg.norm(goal - position) <= step_size:
            position = goal.copy()
            path.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "goal reached"))
            success = True
            break

        # ── Sense nearby obstacles ────────────────────────────────────────────
        newly_observed = reveal_nearby_obstacles(working_scene, position, sensing_range)
        if newly_observed:
            ids = [obs["id"] for obs in newly_observed]
            sensed_ids.extend(ids)
            path.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, f"sensed {ids}"))
            continue

        dist_to_goal = float(np.linalg.norm(goal - position))

        if not follow_boundary:
            # ── GOALSEEK: step directly toward goal ───────────────────────────
            goal_dir = normalize(goal - position)
            next_pos = clip_point_to_workspace(
                working_scene, position + step_size * goal_dir
            )

            if _collides_with_observed(working_scene, next_pos):
                # Hit a wall — record hit point and switch to boundary mode
                follow_boundary    = True
                hit_point          = position.copy()
                hit_distance       = dist_to_goal
                boundary_direction = _rotate_90(goal_dir)
                path.append(tuple(position))
                frames.append(snapshot_frame(
                    position, working_scene,
                    f"obstacle hit (d={hit_distance:.2f}) — boundary mode",
                ))
            else:
                position = next_pos
                path.append(tuple(position))
                frames.append(snapshot_frame(position, working_scene, "goal mode"))

        else:
            # ── WALLFOLLOW: left-hand rule ────────────────────────────────────
            boundary_direction, next_pos = _best_boundary_direction(
                working_scene, position, boundary_direction, step_size
            )
            if boundary_direction is None:
                break  # fully trapped

            # Bug2 leave-wall condition (all three must hold):
            #   1. Back on the m-line segment between start and goal
            #   2. Strictly closer to goal than when the wall was first hit
            #   3. Moved far enough from the hit point (avoids re-triggering immediately)
            moved_away = (
                hit_point is not None
                and float(np.linalg.norm(position - hit_point)) > 2 * step_size
            )
            if (
                moved_away
                and _on_mline_segment(position, start, goal)
                and dist_to_goal < hit_distance - LEAVE_MARGIN
            ):
                follow_boundary = False
                path.append(tuple(position))
                frames.append(snapshot_frame(
                    position, working_scene,
                    f"m-line crossed (d={dist_to_goal:.2f} < hit={hit_distance:.2f})"
                    " — goal mode",
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
