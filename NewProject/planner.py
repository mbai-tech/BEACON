import copy
import random
from dataclasses import dataclass

import numpy as np
from shapely.affinity import translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points

from NewProject.constants import CHAIN_ATTENUATION, DEFAULT_SENSING_RANGE, ROBOT_RADIUS, SAFE_PROB_THRESHOLD
from NewProject.models import OnlineSurpResult, SimulationFrame
from NewProject.scene_setup import normalize_scene_for_online_use


@dataclass
class TrajectoryCandidate:
    """One locally evaluated trajectory option.

    The outline in your flowchart computes two competing trajectories:
    one for avoidance and one for pushing. This container keeps both
    branches in the same format so they can be compared by a common
    reconciliation step.
    """

    mode: str
    waypoint: np.ndarray
    step_target: np.ndarray
    total_cost: float
    safety_margin: float
    progress_gain: float
    obstacle_index: int | None = None
    push_distance: float = 0.0
    corridor_gain: float = 0.0
    reason: str = ""


def obstacle_polygon(obstacle: dict) -> Polygon:
    """Return the Shapely polygon describing one obstacle footprint.

    Caches the result on the obstacle dict under '_poly' so the polygon
    is only constructed once per obstacle (or once after each push that
    updates 'vertices').  Call invalidate_polygon_cache(obstacle) after
    modifying vertices.
    """
    if "_poly" not in obstacle:
        obstacle["_poly"] = Polygon(obstacle["vertices"])
    return obstacle["_poly"]


def invalidate_polygon_cache(obstacle: dict) -> None:
    """Drop the cached polygon so the next call rebuilds it from vertices."""
    obstacle.pop("_poly", None)


def polygon_to_vertices(poly: Polygon) -> list[list[float]]:
    """Convert a polygon back into the scene's serializable vertex format."""
    coords = list(poly.exterior.coords)[:-1]
    return [[float(x), float(y)] for x, y in coords]


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return the unit vector v / ||v||, or zero if the norm is zero."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros_like(vector)
    return vector / norm


def workspace_polygon(scene: dict):
    """Return the rectangular workspace as a polygon for containment checks."""
    xmin, xmax, ymin, ymax = scene["workspace"]
    return box(xmin, ymin, xmax, ymax)


def clip_point_to_workspace(scene: dict, point: np.ndarray) -> np.ndarray:
    """Clip a point to the workspace bounds component-wise."""
    xmin, xmax, ymin, ymax = scene["workspace"]
    return np.array([
        np.clip(point[0], xmin, xmax),
        np.clip(point[1], ymin, ymax),
    ])


def robot_body(position: np.ndarray):
    """Approximate the robot as a disk of radius ``ROBOT_RADIUS``."""
    return Point(position[0], position[1]).buffer(ROBOT_RADIUS, resolution=24)


def robot_clearance_to_obstacle(position: np.ndarray, obstacle: dict) -> float:
    """Compute the signed-like body clearance from robot disk to obstacle polygon."""
    return float(robot_body(position).distance(obstacle_polygon(obstacle)))


def snapshot_frame(position: np.ndarray, scene: dict, message: str) -> SimulationFrame:
    """Freeze the current state for later animation."""
    return SimulationFrame(
        position=(float(position[0]), float(position[1])),
        obstacles=copy.deepcopy(scene["obstacles"]),
        message=message,
    )


def reveal_nearby_obstacles(scene: dict, position: np.ndarray, sensing_range: float) -> list[dict]:
    """Reveal obstacles whose body clearance is within the sensing radius."""
    newly_observed = []
    for obstacle in scene["obstacles"]:
        if obstacle["observed"]:
            continue
        if robot_clearance_to_obstacle(position, obstacle) <= sensing_range:
            obstacle["observed"] = True
            newly_observed.append(obstacle)
    return newly_observed


def cast_sensor_rays(
    scene: dict,
    position: np.ndarray,
    sensing_range: float,
    num_rays: int = 48,
    step_size: float = 0.03,
) -> list[dict]:
    """Cast short radial sensing rays around the robot.

    This mirrors the first perception block in the outline. For each angle we
    march outward along

        p_ray(s) = p + s * [cos(theta), sin(theta)],

    until either a workspace boundary or obstacle is hit. The result is a
    lightweight local free/occupied scan rather than a full grid map.
    """
    ray_results = []
    for angle in np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        hit_obstacle_idx = None
        hit_distance = sensing_range
        frontier_point = clip_point_to_workspace(scene, position + direction * sensing_range)

        for distance in np.linspace(step_size, sensing_range, max(2, int(sensing_range / step_size))):
            sample = clip_point_to_workspace(scene, position + direction * distance)
            sample_body = Point(sample[0], sample[1])
            boundary_clip = float(np.linalg.norm(sample - (position + direction * distance)))
            if boundary_clip > 1e-6:
                hit_distance = float(np.linalg.norm(sample - position))
                frontier_point = sample
                break

            for idx, obstacle in enumerate(scene["obstacles"]):
                if sample_body.distance(obstacle_polygon(obstacle)) <= ROBOT_RADIUS:
                    hit_obstacle_idx = idx
                    hit_distance = float(np.linalg.norm(sample - position))
                    frontier_point = sample
                    break
            if hit_obstacle_idx is not None:
                break

        ray_results.append({
            "angle": float(angle),
            "direction": direction,
            "hit_obstacle_idx": hit_obstacle_idx,
            "hit_distance": hit_distance,
            "frontier_point": frontier_point,
            "is_frontier": hit_obstacle_idx is None and hit_distance >= sensing_range - 1e-6,
        })

    return ray_results


def identify_frontier_cells(ray_results: list[dict]) -> list[np.ndarray]:
    """Identify local frontier points at the edge of known free space.

    In the continuous simulation, a frontier is approximated by a ray that
    reaches the sensing horizon without hitting an obstacle. Those points are
    where free space meets still-unknown space.
    """
    return [ray["frontier_point"] for ray in ray_results if ray["is_frontier"]]


def update_local_perception(
    scene: dict,
    position: np.ndarray,
    sensing_range: float,
) -> dict:
    """Run the full local perception update from the outline.

    The update sequence is:
    1. reveal nearby obstacles,
    2. cast radial rays,
    3. identify frontier points,
    4. update per-obstacle occupancy and blocking estimates.
    """
    newly_observed = reveal_nearby_obstacles(scene, position, sensing_range)
    ray_results = cast_sensor_rays(scene, position, sensing_range)
    frontier_points = identify_frontier_cells(ray_results)

    hit_counts: dict[int, int] = {}
    for ray in ray_results:
        hit_idx = ray["hit_obstacle_idx"]
        if hit_idx is None:
            continue
        hit_counts[hit_idx] = hit_counts.get(hit_idx, 0) + 1

    total_rays = max(1, len(ray_results))
    for idx, obstacle in enumerate(scene["obstacles"]):
        obstacle.setdefault("occupancy_prob", 0.0)
        obstacle.setdefault("blocking_score", 0.0)
        obstacle.setdefault("semantic_default", obstacle["true_class"])
        obstacle.setdefault("push_count", 0)
        obstacle.setdefault("last_push_step", -1)
        if not obstacle["observed"]:
            continue

        hit_ratio = hit_counts.get(idx, 0) / total_rays
        goal_alignment = 0.0
        obstacle["occupancy_prob"] = min(1.0, 0.35 + 1.8 * hit_ratio)
        obstacle["blocking_score"] = min(1.0, 0.45 * obstacle["occupancy_prob"] + 0.55 * goal_alignment)

    return {
        "newly_observed": newly_observed,
        "rays": ray_results,
        "frontiers": frontier_points,
    }


def goal_progress_is_stalled(
    goal_distance_history: list[float],
    window: int = 12,
    min_progress: float = 0.14,
) -> bool:
    """Detect stalls from insufficient drop in goal distance over a time window.

    The test is

        d(t - window) - d(t) < min_progress,

    where ``d(t) = ||g - p(t)||`` is the direct Euclidean distance to goal.
    """
    if len(goal_distance_history) < window:
        return False
    return (goal_distance_history[-window] - goal_distance_history[-1]) < min_progress


def step_until_sense_or_contact(
    scene: dict,
    start: np.ndarray,
    direction: np.ndarray,
    step_size: float,
    epsilon: float,
    subdivisions: int = 14,
) -> np.ndarray:
    """Advance until motion would violate collision or epsilon-sensing limits.

    The robot attempts the step

        p_next = p + alpha * step_size * d_hat,

    for increasing ``alpha in (0, 1]``. The last candidate that is still safe is
    returned. Observed obstacles use direct body intersection, while unseen
    obstacles are treated conservatively by stopping once the body gets within
    ``epsilon`` of the obstacle.
    """
    direction = normalize(direction)
    if np.linalg.norm(direction) <= 1e-9:
        return start.copy()

    last_safe = start.copy()
    for fraction in np.linspace(1.0 / subdivisions, 1.0, subdivisions):
        candidate = clip_point_to_workspace(scene, start + direction * step_size * fraction)
        candidate_body = robot_body(candidate)

        blocked = False
        for obstacle in scene["obstacles"]:
            poly = obstacle_polygon(obstacle)
            if obstacle["observed"]:
                if candidate_body.intersects(poly):
                    blocked = True
                    break
            else:
                if candidate_body.distance(poly) <= epsilon:
                    blocked = True
                    break

        if blocked:
            break
        last_safe = candidate

    return last_safe


def pushability_score(obstacle: dict, position: np.ndarray, goal_direction: np.ndarray) -> float:
    """Heuristic score for choosing which safe obstacle to push.

    The score prefers obstacles that are:
    - semantically pushable,
    - close to the robot,
    - and forward along the goal direction.
    """
    cls = obstacle["true_class"]
    if cls == "movable":
        class_score = 3.0
    else:
        class_score = -1000.0

    poly = obstacle_polygon(obstacle)
    distance_score = -robot_clearance_to_obstacle(position, obstacle)
    center = np.array([poly.centroid.x, poly.centroid.y])
    forward_score = float(np.dot(center - position, goal_direction))
    return class_score + 0.35 * forward_score + 0.15 * distance_score


def obstacle_is_ahead(position: np.ndarray, goal_direction: np.ndarray, obstacle: dict, epsilon: float) -> bool:
    """Check whether an obstacle occupies the robot's short forward sweep."""
    lookahead = max(0.4, epsilon + 0.15)
    segment = LineString([position, position + goal_direction * lookahead])
    return segment.buffer(ROBOT_RADIUS).intersects(obstacle_polygon(obstacle))


def choose_best_pushable_obstacle(
    scene: dict,
    position: np.ndarray,
    goal_direction: np.ndarray,
    epsilon: float,
    contact_margin: float = 0.03,
) -> int | None:
    """Choose the best observed pushable obstacle in the current contact band."""
    candidates = []
    for idx, obstacle in enumerate(scene["obstacles"]):
        if not obstacle["observed"]:
            continue
        if obstacle["true_class"] != "movable":
            continue
        distance = robot_clearance_to_obstacle(position, obstacle)
        ahead = obstacle_is_ahead(position, goal_direction, obstacle, epsilon + contact_margin)
        in_contact_band = distance <= epsilon + contact_margin
        if not (ahead and in_contact_band):
            continue
        candidates.append((pushability_score(obstacle, position, goal_direction), idx))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def sigmoid(value: float) -> float:
    """Standard logistic map sigma(x) = 1 / (1 + e^(-x))."""
    return 1.0 / (1.0 + np.exp(-value))


def estimate_obstacle_features(obstacle: dict) -> dict:
    """Return a simple semantic feature model used by the risk-aware controller."""
    cls = obstacle["true_class"]
    # Binary classification: movable (low risk) vs not_movable (high risk)
    fragility_map = {
        "movable":     0.10,
        "not_movable": 0.90,
    }
    semantic_risk_map = {
        "movable":     0.10,
        "not_movable": 0.90,
    }
    confidence_map = {
        "movable":     0.88,
        "not_movable": 0.92,
    }
    # Fallback for any legacy class labels
    f = fragility_map.get(cls, 0.50)
    c = semantic_risk_map.get(cls, 0.50)
    q = confidence_map.get(cls, 0.88)
    return {
        "m": 0.50,
        "mu": 0.45,
        "f": f,
        "c": c,
        "q": q,
        "class_label": cls,
    }


def obstacle_safety_probability(obstacle: dict) -> tuple[float, float]:
    """Convert semantic obstacle features into a push-safety probability.

    The code uses the weighted score

        S = 0.10 m + 0.15 mu + 0.45 f + 0.40 c - 0.25 q,

    then maps it into

        P_safe = sigma(-S).

    Lower risk therefore means larger ``P_safe``.
    """
    features = estimate_obstacle_features(obstacle)
    safety_score = (
        0.10 * features["m"]
        + 0.15 * features["mu"]
        + 0.45 * features["f"]
        + 0.40 * features["c"]
        - 0.25 * features["q"]
    )
    return float(sigmoid(-safety_score)), float(safety_score)


def classify_sensed_obstacles(
    scene: dict,
    position: np.ndarray,
    goal_direction: np.ndarray,
    epsilon: float,
) -> tuple[list[int], list[int]]:
    """Partition sensed obstacles into avoid and push sets.

    This corresponds to the outline stage where sensed space is classified and
    then partitioned into candidate sets. We compute a simple local blocking
    score using

        block_i = 0.55 * occupancy_i + 0.45 * ahead_i,

    where ``ahead_i`` is 1 when the obstacle lies in the short forward sweep.
    Obstacles that are safe enough and close enough enter the push set;
    everything else stays in the avoid set.
    """
    avoid_set: list[int] = []
    push_set: list[int] = []
    for idx, obstacle in enumerate(scene["obstacles"]):
        if not obstacle["observed"]:
            continue

        p_safe, _ = obstacle_safety_probability(obstacle)
        ahead = 1.0 if obstacle_is_ahead(position, goal_direction, obstacle, epsilon + 0.06) else 0.0
        distance = robot_clearance_to_obstacle(position, obstacle)
        proximity = max(0.0, 1.0 - distance / max(epsilon + 0.2, 1e-6))

        obstacle["semantic_default"] = obstacle.get("semantic_default", obstacle["true_class"])
        obstacle["safety_probability"] = p_safe
        obstacle["blocking_score"] = min(
            1.0,
            0.55 * obstacle.get("occupancy_prob", 0.35) + 0.30 * ahead + 0.15 * proximity,
        )

        avoid_set.append(idx)
        if (
            obstacle["true_class"] == "movable"
            and p_safe >= SAFE_PROB_THRESHOLD
            and distance <= epsilon + 0.08
            and ahead > 0.0
        ):
            push_set.append(idx)

    push_set.sort(
        key=lambda idx: (
            scene["obstacles"][idx].get("blocking_score", 0.0),
            scene["obstacles"][idx].get("safety_probability", 0.0),
        ),
        reverse=True,
    )
    avoid_set.sort(
        key=lambda idx: scene["obstacles"][idx].get("blocking_score", 0.0),
        reverse=True,
    )
    return avoid_set, push_set


def nearest_sensed_obstacle(scene: dict, position: np.ndarray) -> tuple[int | None, float]:
    """Return the observed obstacle with minimum robot-body clearance."""
    best_idx = None
    best_distance = float("inf")
    for idx, obstacle in enumerate(scene["obstacles"]):
        if not obstacle["observed"]:
            continue
        distance = robot_clearance_to_obstacle(position, obstacle)
        if distance < best_distance:
            best_idx = idx
            best_distance = distance
    return best_idx, best_distance


def choose_sidestep_motion(
    scene: dict,
    position: np.ndarray,
    goal: np.ndarray,
    step_size: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a small set of lateral detours around the goal direction.

    Each candidate is scored using a weighted combination of:
    - goal distance improvement,
    - local obstacle clearance,
    - and actual step length.
    """
    goal_direction = normalize(goal - position)
    if np.linalg.norm(goal_direction) <= 1e-9:
        return np.zeros(2, dtype=float), position.copy()

    perpendicular = np.array([-goal_direction[1], goal_direction[0]])
    direction_candidates = [
        normalize(perpendicular),
        normalize(-perpendicular),
        normalize(goal_direction + 0.8 * perpendicular),
        normalize(goal_direction - 0.8 * perpendicular),
    ]

    best_direction = np.zeros(2, dtype=float)
    best_candidate = position.copy()
    best_score = -float("inf")
    current_goal_distance = float(np.linalg.norm(goal - position))

    for candidate_direction in direction_candidates:
        candidate = step_until_sense_or_contact(
            scene,
            position,
            candidate_direction,
            step_size,
            epsilon,
        )
        moved_distance = float(np.linalg.norm(candidate - position))
        if moved_distance < 0.03:
            continue

        goal_improvement = current_goal_distance - float(np.linalg.norm(goal - candidate))
        obstacle_clearance = min(
            robot_clearance_to_obstacle(candidate, obstacle)
            for obstacle in scene["obstacles"]
        )
        score = goal_improvement + 0.04 * obstacle_clearance + 0.02 * moved_distance
        if score > best_score:
            best_score = score
            best_direction = candidate_direction
            best_candidate = candidate

    return best_direction, best_candidate


def rotate_direction(direction: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2D direction by ``angle_deg`` degrees."""
    direction = normalize(direction)
    if np.linalg.norm(direction) <= 1e-9:
        return direction
    angle = np.radians(angle_deg)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    return normalize(rotation @ direction)


def choose_escape_motion(
    scene: dict,
    position: np.ndarray,
    goal: np.ndarray,
    step_size: float,
    epsilon: float,
    bad_directions: list[dict] | None = None,
    num_directions: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Search over local directions when the direct goal motion is blocked.

    The local score is

        score = 0.8 * goal_improvement
              + 0.05 * clearance
              + 0.04 * step_length
              + 0.03 * alignment
              - memory_penalty.

    This favors progress and clearance while discouraging previously bad
    directions stored in the robot's local memory.
    """
    goal_direction = normalize(goal - position)
    current_goal_distance = float(np.linalg.norm(goal - position))

    best_direction = np.zeros(2, dtype=float)
    best_candidate = position.copy()
    best_score = -float("inf")

    for angle_deg in np.linspace(0.0, 360.0, num_directions, endpoint=False):
        if np.linalg.norm(goal_direction) > 1e-9:
            candidate_direction = rotate_direction(goal_direction, float(angle_deg))
        else:
            angle_rad = np.radians(angle_deg)
            candidate_direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

        candidate = step_until_sense_or_contact(
            scene,
            position,
            candidate_direction,
            step_size,
            epsilon,
        )
        moved_distance = float(np.linalg.norm(candidate - position))
        if moved_distance < 0.01:
            continue

        goal_improvement = current_goal_distance - float(np.linalg.norm(goal - candidate))
        obstacle_clearance = min(
            robot_clearance_to_obstacle(candidate, obstacle)
            for obstacle in scene["obstacles"]
        )
        alignment = float(np.dot(normalize(candidate - position), goal_direction)) if np.linalg.norm(goal_direction) > 1e-9 else 0.0
        memory_penalty = 0.0
        if bad_directions:
            for memory in bad_directions:
                local_distance = float(np.linalg.norm(position - memory["position"]))
                if local_distance > memory["radius"]:
                    continue
                direction_similarity = float(np.dot(candidate_direction, memory["direction"]))
                if direction_similarity > 0.55:
                    memory_penalty += memory["weight"] * direction_similarity

        score = (
            0.8 * goal_improvement
            + 0.05 * obstacle_clearance
            + 0.04 * moved_distance
            + 0.03 * alignment
            - memory_penalty
        )

        if score > best_score:
            best_score = score
            best_direction = candidate_direction
            best_candidate = candidate

    return best_direction, best_candidate


def estimate_avoid_cost(
    scene: dict,
    position: np.ndarray,
    goal: np.ndarray,
    step_size: float,
    epsilon: float,
    bad_directions: list[dict],
) -> tuple[float, np.ndarray]:
    """Compare a short sidestep with a broader local escape and keep the better one.

    The avoid cost is a simple weighted sum:

        J_avoid = 1.1 * move_distance + 0.7 * remaining_goal_distance + 0.25 * uncertainty.
    """
    _, sidestep_candidate = choose_sidestep_motion(
        scene,
        position,
        goal,
        step_size,
        epsilon,
    )
    _, escape_candidate = choose_escape_motion(
        scene,
        position,
        goal,
        step_size,
        epsilon,
        bad_directions=bad_directions,
    )

    sidestep_gain = float(np.linalg.norm(goal - position) - np.linalg.norm(goal - sidestep_candidate))
    escape_gain = float(np.linalg.norm(goal - position) - np.linalg.norm(goal - escape_candidate))

    if np.linalg.norm(escape_candidate - position) > 1e-6 and escape_gain >= sidestep_gain:
        chosen_candidate = escape_candidate
    else:
        chosen_candidate = sidestep_candidate

    move_distance = float(np.linalg.norm(chosen_candidate - position))
    remaining = float(np.linalg.norm(goal - chosen_candidate))
    uncertainty = 0.20
    cost = 1.1 * move_distance + 0.7 * remaining + 0.25 * uncertainty
    return cost, chosen_candidate


def _translated_obstacle_poly(obstacle: dict, direction: np.ndarray, displacement: float) -> Polygon:
    """Translate an obstacle footprint by ``displacement * direction``."""
    return translate(
        obstacle_polygon(obstacle),
        xoff=float(direction[0] * displacement),
        yoff=float(direction[1] * displacement),
    )


def obstacle_push_direction(obstacle: dict, robot_position: np.ndarray, preferred_direction: np.ndarray) -> np.ndarray:
    """Blend the robot motion direction with an outward push normal.

    If ``e_pref`` is the robot's preferred direction and ``e_out`` points from
    the robot toward the obstacle center, we use

        d_push = normalize(0.35 * e_pref + 0.65 * e_out),

    unless the preferred direction points backward into the robot, in which
    case we use the outward direction directly.
    """
    obstacle_center = np.array(obstacle_polygon(obstacle).centroid.coords[0], dtype=float)
    away_from_robot = normalize(obstacle_center - robot_position)
    preferred_direction = normalize(preferred_direction)

    if np.linalg.norm(away_from_robot) <= 1e-9:
        return preferred_direction
    if np.linalg.norm(preferred_direction) <= 1e-9:
        return away_from_robot
    if float(np.dot(preferred_direction, away_from_robot)) < 0.0:
        return away_from_robot
    return normalize(0.35 * preferred_direction + 0.65 * away_from_robot)


def feasible_push_distance(
    scene: dict,
    obstacle_index: int,
    direction: np.ndarray,
    requested_displacement: float,
    robot_position: np.ndarray | None = None,
    subdivisions: int = 16,
) -> float:
    """Return the largest collision-free push distance for a single obstacle."""
    obstacle = scene["obstacles"][obstacle_index]
    if requested_displacement <= 0:
        return 0.0

    if robot_position is not None:
        direction = obstacle_push_direction(obstacle, robot_position, direction)

    best_distance = 0.0
    for fraction in np.linspace(1.0 / subdivisions, 1.0, subdivisions):
        displacement = requested_displacement * fraction
        moved_poly = _translated_obstacle_poly(obstacle, direction, displacement)

        if not workspace_polygon(scene).contains(moved_poly):
            break

        collision_found = False
        for idx, other in enumerate(scene["obstacles"]):
            if idx == obstacle_index:
                continue
            if moved_poly.intersects(obstacle_polygon(other)):
                collision_found = True
                break

        if collision_found:
            break

        best_distance = displacement

    return best_distance


def moved_chain_levels(
    scene: dict,
    obstacle_index: int,
    direction: np.ndarray,
    displacement: float,
) -> dict[int, int]:
    """Build the obstacle-contact chain for a proposed push.

    If obstacle ``i`` hits obstacle ``j``, the secondary obstacle is assigned one
    larger chain level. The actual displacement then decays as

        delta_j = delta_0 * CHAIN_ATTENUATION^level_j.
    """
    chain_levels = {obstacle_index: 0}
    while True:
        expanded = False
        moved_polys = {
            idx: _translated_obstacle_poly(
                scene["obstacles"][idx],
                direction,
                displacement * (CHAIN_ATTENUATION ** level),
            )
            for idx, level in chain_levels.items()
        }

        for idx, moved_poly in moved_polys.items():
            for other_idx, other in enumerate(scene["obstacles"]):
                if other_idx in chain_levels:
                    continue
                if other["true_class"] != "movable":
                    continue
                if moved_poly.intersects(obstacle_polygon(other)):
                    chain_levels[other_idx] = chain_levels[idx] + 1
                    expanded = True

        if not expanded:
            return chain_levels


def feasible_chain_push_distance(
    scene: dict,
    obstacle_index: int,
    direction: np.ndarray,
    requested_displacement: float,
    robot_position: np.ndarray | None = None,
    subdivisions: int = 16,
) -> tuple[float, set[int]]:
    """Return the largest feasible push distance when chain contacts are allowed."""
    obstacle = scene["obstacles"][obstacle_index]
    if requested_displacement <= 0:
        return 0.0, {obstacle_index}

    if robot_position is not None:
        direction = obstacle_push_direction(obstacle, robot_position, direction)

    best_distance = 0.0
    best_chain = {obstacle_index}

    for fraction in np.linspace(1.0 / subdivisions, 1.0, subdivisions):
        displacement = requested_displacement * fraction
        chain_levels = moved_chain_levels(scene, obstacle_index, direction, displacement)
        moved_polys = {
            idx: _translated_obstacle_poly(
                scene["obstacles"][idx],
                direction,
                displacement * (CHAIN_ATTENUATION ** level),
            )
            for idx, level in chain_levels.items()
        }

        if not all(workspace_polygon(scene).contains(poly) for poly in moved_polys.values()):
            break

        collision_found = False
        # Check chain members against all non-chain obstacles
        for idx, moved_poly in moved_polys.items():
            for other_idx, other in enumerate(scene["obstacles"]):
                if other_idx in chain_levels:
                    continue
                if moved_poly.intersects(obstacle_polygon(other)):
                    collision_found = True
                    break
            if collision_found:
                break

        # Also check chain members against each other — differential attenuation
        # means the lead obstacle moves more than secondaries and can overlap them.
        if not collision_found:
            chain_ids = sorted(chain_levels.keys())
            for i in range(len(chain_ids)):
                for j in range(i + 1, len(chain_ids)):
                    if moved_polys[chain_ids[i]].intersects(moved_polys[chain_ids[j]]):
                        collision_found = True
                        break
                if collision_found:
                    break

        if collision_found:
            break

        best_distance = displacement
        best_chain = set(chain_levels.keys())

    return best_distance, best_chain


def estimate_push_cost(
    scene: dict,
    position: np.ndarray,
    obstacle_index: int,
    goal_direction: np.ndarray,
    push_distance: float,
) -> tuple[float, float, float]:
    """Estimate whether a push is worthwhile using effort, risk, and corridor gain.

    The implemented push cost is

        J_push = 0.6 * time + 0.5 * effort + 1.2 * risk - 1.4 * corridor_gain.

    ``corridor_gain`` is the change in robot-to-obstacle clearance after a
    predicted push; larger corridor gain lowers the push cost.
    """
    obstacle = scene["obstacles"][obstacle_index]
    features = estimate_obstacle_features(obstacle)
    actual_displacement, chain = feasible_chain_push_distance(
        scene,
        obstacle_index,
        goal_direction,
        requested_displacement=push_distance,
        robot_position=position,
    )
    before_clearance = robot_clearance_to_obstacle(position, obstacle)
    if actual_displacement > 1e-6:
        effective_direction = obstacle_push_direction(obstacle, position, goal_direction)
        moved_poly = _translated_obstacle_poly(obstacle, effective_direction, actual_displacement)
        after_clearance = float(robot_body(position).distance(moved_poly))
    else:
        after_clearance = before_clearance
    corridor_gain = after_clearance - before_clearance

    chain_load = max(0, len(chain) - 1)
    risk = (
        0.70 * features["f"]
        + 0.20 * (1.0 - features["q"])
        + 0.90 * (1.0 if features["class_label"] in {"fragile", "forbidden"} else 0.0)
    )
    effort = 1.0 + 0.35 * chain_load
    time_term = 1.0
    cost = 0.6 * time_term + 0.5 * effort + 1.2 * risk - 1.4 * corridor_gain
    return cost, actual_displacement, corridor_gain


def smooth_candidate_step(
    position: np.ndarray,
    waypoint: np.ndarray,
    goal: np.ndarray,
    scene: dict,
    step_size: float,
) -> np.ndarray:
    """Apply a tiny one-step smoothing pass before execution.

    The flowchart mentions path smoothing after search. Here we approximate that
    by blending the waypoint direction with the direct goal direction:

        d_smooth = normalize(0.7 * d_waypoint + 0.3 * d_goal).

    We then move only one feasible step along that blended direction.
    """
    waypoint_direction = normalize(waypoint - position)
    goal_direction = normalize(goal - position)
    blended = normalize(0.7 * waypoint_direction + 0.3 * goal_direction)
    if np.linalg.norm(blended) <= 1e-9:
        blended = waypoint_direction
    return safe_step_position(scene, position, blended, step_size, observed_only=True)


def compute_avoid_trajectory(
    scene: dict,
    position: np.ndarray,
    goal: np.ndarray,
    step_size: float,
    epsilon: float,
    bad_directions: list[dict],
) -> TrajectoryCandidate:
    """Compute the avoidance branch from the current local map.

    This is the blue branch in the outline. The returned cost is based on the
    avoid cost model, while the safety margin is the minimum robot-body
    clearance at the resulting step target.
    """
    total_cost, waypoint = estimate_avoid_cost(
        scene,
        position,
        goal,
        step_size,
        epsilon,
        bad_directions,
    )
    step_target = smooth_candidate_step(position, waypoint, goal, scene, step_size)
    safety_margin = min(
        robot_clearance_to_obstacle(step_target, obstacle)
        for obstacle in scene["obstacles"]
    ) if scene["obstacles"] else float("inf")
    progress_gain = float(np.linalg.norm(goal - position) - np.linalg.norm(goal - step_target))
    return TrajectoryCandidate(
        mode="avoid",
        waypoint=waypoint,
        step_target=step_target,
        total_cost=total_cost,
        safety_margin=safety_margin,
        progress_gain=progress_gain,
        reason="CFP branch: local avoid trajectory",
    )


def compute_push_trajectory(
    scene: dict,
    position: np.ndarray,
    goal: np.ndarray,
    goal_direction: np.ndarray,
    push_candidates: list[int],
    step_size: float,
    epsilon: float,
    push_distance: float,
) -> TrajectoryCandidate | None:
    """Compute the pushing branch from the current local map.

    This is the orange branch in the outline. We rank pushable obstacles, score
    the predicted push corridor, and keep the lowest-cost push branch.
    """
    best_candidate: TrajectoryCandidate | None = None
    for obstacle_index in push_candidates:
        cost, feasible_distance, corridor_gain = estimate_push_cost(
            scene,
            position,
            obstacle_index,
            goal_direction,
            push_distance,
        )
        if feasible_distance <= 1e-6:
            continue

        predicted_position = step_until_sense_or_contact(
            scene,
            position,
            goal_direction,
            step_size,
            epsilon,
        )
        safety_margin = min(
            robot_clearance_to_obstacle(predicted_position, obstacle)
            for obstacle in scene["obstacles"]
        ) if scene["obstacles"] else float("inf")
        progress_gain = float(np.linalg.norm(goal - position) - np.linalg.norm(goal - predicted_position))
        candidate = TrajectoryCandidate(
            mode="push",
            waypoint=predicted_position,
            step_target=predicted_position,
            total_cost=cost,
            safety_margin=safety_margin,
            progress_gain=progress_gain,
            obstacle_index=obstacle_index,
            push_distance=feasible_distance,
            corridor_gain=corridor_gain,
            reason="CPP branch: local push trajectory",
        )
        if best_candidate is None or candidate.total_cost < best_candidate.total_cost:
            best_candidate = candidate

    return best_candidate


def reconcile_trajectory_decision(
    avoid_candidate: TrajectoryCandidate,
    push_candidate: TrajectoryCandidate | None,
    safety_margin_threshold: float,
) -> TrajectoryCandidate:
    """Reconcile the avoid and push branches into one action.

    The outline's decision block is implemented as a simple normalized
    comparison:

        score = cost - 0.35 * progress_gain - 0.25 * safety_margin.

    A push branch is only allowed if its safety margin exceeds the threshold.
    """
    def branch_score(candidate: TrajectoryCandidate) -> float:
        return candidate.total_cost - 0.35 * candidate.progress_gain - 0.25 * candidate.safety_margin

    selected = avoid_candidate
    avoid_score = branch_score(avoid_candidate)
    if push_candidate is None:
        return selected

    push_score = branch_score(push_candidate) - 0.20 * push_candidate.corridor_gain
    if push_candidate.safety_margin >= safety_margin_threshold and push_score <= avoid_score:
        selected = push_candidate
    return selected


def remember_bad_direction(
    bad_directions: list[dict],
    position: np.ndarray,
    direction: np.ndarray,
    radius: float = 0.5,
    weight: float = 0.8,
) -> None:
    """Store a local direction that recently led to poor progress."""
    bad_directions.append({
        "position": position.copy(),
        "direction": normalize(direction),
        "radius": radius,
        "weight": weight,
    })


def choose_backtrack_target(path: list[tuple[float, float]], lookback: int = 12) -> np.ndarray | None:
    """Return an earlier point on the executed path for simple backtracking."""
    if len(path) < 3:
        return None
    idx = max(0, len(path) - 1 - lookback)
    target = np.array(path[idx], dtype=float)
    current = np.array(path[-1], dtype=float)
    if np.linalg.norm(target - current) < 1e-6:
        return None
    return target


def choose_deep_backtrack_target(
    path: list[tuple[float, float]],
    position: np.ndarray,
    stuck_level: int,
) -> np.ndarray | None:
    """Escalate backtracking depth as repeated stall events accumulate."""
    if len(path) < 3:
        return None

    lookbacks = [12, 24, 40, 64, 96]
    max_idx = min(len(lookbacks) - 1, max(0, stuck_level - 1))
    for lookback in reversed(lookbacks[: max_idx + 1]):
        target = choose_backtrack_target(path, lookback=lookback)
        if target is None:
            continue
        if np.linalg.norm(target - position) > 0.08:
            return target
    return choose_backtrack_target(path, lookback=lookbacks[0])


def point_collides(scene: dict, point: np.ndarray, observed_only: bool = False) -> bool:
    """Check whether the robot disk at ``point`` intersects any selected obstacle."""
    point_geom = robot_body(point)
    for obstacle in scene["obstacles"]:
        if observed_only and not obstacle["observed"]:
            continue
        if point_geom.intersects(obstacle_polygon(obstacle)):
            return True
    return False


def segment_is_collision_free(
    scene: dict,
    start: np.ndarray,
    end: np.ndarray,
    observed_only: bool = False,
    subdivisions: int = 20,
) -> bool:
    """Check collision freedom by discretizing a segment into short samples."""
    for fraction in np.linspace(0.0, 1.0, subdivisions + 1):
        candidate = clip_point_to_workspace(scene, start + (end - start) * fraction)
        if point_collides(scene, candidate, observed_only=observed_only):
            return False
    return True


def safe_step_position(
    scene: dict,
    start: np.ndarray,
    direction: np.ndarray,
    step_size: float,
    observed_only: bool = False,
    subdivisions: int = 12,
) -> np.ndarray:
    """Advance in a direction until collision would occur, then stop at last safe point."""
    direction = normalize(direction)
    if np.linalg.norm(direction) == 0:
        return start.copy()

    last_safe = start.copy()
    for fraction in np.linspace(1.0 / subdivisions, 1.0, subdivisions):
        candidate = clip_point_to_workspace(scene, start + direction * step_size * fraction)
        if point_collides(scene, candidate, observed_only=observed_only):
            break
        last_safe = candidate
    return last_safe


def sample_free_point(scene: dict, observed_only: bool = True, max_tries: int = 40) -> np.ndarray | None:
    """Sample a random collision-free point from the workspace."""
    xmin, xmax, ymin, ymax = scene["workspace"]
    for _ in range(max_tries):
        candidate = np.array(
            [random.uniform(xmin, xmax), random.uniform(ymin, ymax)],
            dtype=float,
        )
        if not point_collides(scene, candidate, observed_only=observed_only):
            return candidate
    return None


def nearest_tree_index(nodes: list[np.ndarray], target: np.ndarray) -> int:
    """Return the index of the tree node with minimum Euclidean distance to target."""
    distances = [float(np.linalg.norm(node - target)) for node in nodes]
    return int(np.argmin(distances))


def build_rrt_path(nodes: list[np.ndarray], parents: list[int], goal_idx: int) -> list[np.ndarray]:
    """Recover a root-to-goal path from the RRT parent array."""
    path: list[np.ndarray] = []
    idx = goal_idx
    while idx != -1:
        path.append(nodes[idx])
        idx = parents[idx]
    path.reverse()
    return path


def rrt_escape_step(
    scene: dict,
    position: np.ndarray,
    goal: np.ndarray,
    step_size: float,
    bad_directions: list[dict] | None = None,
    max_samples: int = 180,
    expand_distance: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build a lightweight local RRT over sensed free space and return one step.

    The tree expands by repeatedly:
    1. sampling a free workspace point,
    2. finding the nearest tree node,
    3. stepping a fixed distance toward the sample,
    4. rejecting branches that collide or repeat locally bad directions.

    Every fifth sample is the goal itself, which biases the tree toward useful
    escape branches without assuming global map knowledge.
    """
    expand_distance = expand_distance or max(0.24, 2.8 * step_size)
    nodes = [position.copy()]
    parents = [-1]

    xmin, xmax, ymin, ymax = scene["workspace"]
    bbox_diag = float(np.hypot(xmax - xmin, ymax - ymin))
    goal_tolerance = max(0.26, 3.2 * step_size)

    for sample_idx in range(max_samples):
        if sample_idx % 5 == 0:
            sample = goal.copy()
        else:
            sample = sample_free_point(scene, observed_only=True)
            if sample is None:
                continue

        nearest_idx = nearest_tree_index(nodes, sample)
        nearest = nodes[nearest_idx]
        direction = normalize(sample - nearest)
        if np.linalg.norm(direction) <= 1e-9:
            continue

        candidate = clip_point_to_workspace(scene, nearest + direction * expand_distance)
        if not segment_is_collision_free(scene, nearest, candidate, observed_only=True):
            continue

        memory_penalty = 0.0
        if bad_directions:
            for memory in bad_directions:
                local_distance = float(np.linalg.norm(nearest - memory["position"]))
                if local_distance > memory["radius"] * 1.35:
                    continue
                direction_similarity = float(np.dot(direction, memory["direction"]))
                if direction_similarity > 0.45:
                    memory_penalty += memory["weight"] * direction_similarity

        if memory_penalty > 0.9:
            continue

        nodes.append(candidate)
        parents.append(nearest_idx)
        new_idx = len(nodes) - 1

        if (
            float(np.linalg.norm(candidate - goal)) <= goal_tolerance
            and segment_is_collision_free(scene, candidate, goal, observed_only=True)
        ):
            nodes.append(goal.copy())
            parents.append(new_idx)
            goal_idx = len(nodes) - 1
            rrt_path = build_rrt_path(nodes, parents, goal_idx)
            if len(rrt_path) >= 2:
                next_waypoint = rrt_path[1]
                next_candidate = safe_step_position(
                    scene,
                    position,
                    normalize(next_waypoint - position),
                    min(step_size * 2.8, float(np.linalg.norm(next_waypoint - position))),
                    observed_only=True,
                )
                if np.linalg.norm(next_candidate - position) > 1e-6:
                    return normalize(next_candidate - position), next_candidate

    scored_nodes = []
    goal_direction = normalize(goal - position)
    current_goal_distance = float(np.linalg.norm(goal - position))
    for idx, node in enumerate(nodes[1:], start=1):
        direction = normalize(node - position)
        if np.linalg.norm(direction) <= 1e-9:
            continue
        memory_penalty = 0.0
        if bad_directions:
            for memory in bad_directions:
                local_distance = float(np.linalg.norm(position - memory["position"]))
                if local_distance > memory["radius"] * 1.35:
                    continue
                direction_similarity = float(np.dot(direction, memory["direction"]))
                if direction_similarity > 0.45:
                    memory_penalty += memory["weight"] * direction_similarity
        goal_improvement = current_goal_distance - float(np.linalg.norm(goal - node))
        alignment = float(np.dot(direction, goal_direction)) if np.linalg.norm(goal_direction) > 1e-9 else 0.0
        clearance_bonus = min(
            robot_clearance_to_obstacle(node, obstacle)
            for obstacle in scene["obstacles"]
            if obstacle["observed"]
        ) if any(obstacle["observed"] for obstacle in scene["obstacles"]) else bbox_diag
        score = 1.1 * goal_improvement + 0.22 * alignment + 0.05 * clearance_bonus - memory_penalty
        scored_nodes.append((score, idx, node))

    if not scored_nodes:
        return None, None

    scored_nodes.sort(reverse=True, key=lambda item: item[0])
    waypoint = scored_nodes[0][2]
    fallback_candidate = safe_step_position(
        scene,
        position,
        normalize(waypoint - position),
        min(step_size * 2.4, float(np.linalg.norm(waypoint - position))),
        observed_only=True,
    )
    if np.linalg.norm(fallback_candidate - position) <= 1e-6:
        return None, None
    return normalize(fallback_candidate - position), fallback_candidate


def _bug_rotate(v: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2D vector by angle_deg degrees."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return normalize(np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]]))


def _scholar_best_boundary_direction(
    scene: dict,
    position: np.ndarray,
    current_direction: np.ndarray,
    step_size: float,
    n_sweep: int = 36,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fine-sweep boundary follower implementing the Bug1 left-hand rule.

    Starts from -90° (right turn toward the wall) and sweeps CCW through a
    full 360°, accepting the first collision-free candidate. This correctly
    re-acquires curved boundaries: right → forward → left → back.
    Uses observed-only collision checking so undetected obstacles are ignored.
    """
    step_deg = 360.0 / n_sweep
    for i in range(n_sweep):
        angle = -90.0 + i * step_deg
        candidate_dir = _bug_rotate(current_direction, angle)
        next_pos = clip_point_to_workspace(scene, position + step_size * candidate_dir)
        if not point_collides(scene, next_pos, observed_only=True):
            return candidate_dir, next_pos
    return None, None



def detect_boundary_oscillation(
    boundary_stop_events: list[int],
    min_alternations: int = 3,
) -> tuple[int, int] | None:
    """Return the two oscillating obstacle IDs if the boundary-stop history alternates between exactly two.

    Checks whether the last ``2 * min_alternations`` boundary-stop events form
    a strict A-B-A-B-... pattern. Three full alternations (6 events) are
    required before triggering to avoid false positives in the early steps.
    """
    n = 2 * min_alternations
    if len(boundary_stop_events) < n:
        return None
    recent = boundary_stop_events[-n:]
    if len(set(recent)) != 2:
        return None
    for i in range(len(recent) - 1):
        if recent[i] == recent[i + 1]:
            return None
    ids = list(set(recent))
    return (ids[0], ids[1])


def semantic_corridor_response(
    scene: dict,
    position: np.ndarray,
    goal: np.ndarray,
    obs_id_a: int,
    obs_id_b: int,
    step_size: float,
    frontier_points: list[np.ndarray],
) -> tuple[np.ndarray | None, str, int | None]:
    """Semantic response to oscillation between two corridor walls.

    SCHOLAR's philosophy: ask "can I change this environment?" before asking
    "how do I fit through it?" The two outcomes are mutually exclusive:

    1. **Corridor widening** — if either wall is semantically pushable, return
       the obstacle index so the caller can push it aside and step through.
       The scene is NOT modified here; the caller does the push and bookkeeping.

    2. **Frontier detour** — if neither wall is pushable, find a frontier point
       that heads toward the goal while steering *away* from the blocked
       corridor, and return one step toward it. This routes through unexplored
       space rather than threading known tight geometry.

    Return value: ``(candidate_position, label, push_obs_idx)``

    - ``push_obs_idx is not None``: caller should push that obstacle index,
      then step toward the goal.
    - ``push_obs_idx is None and candidate_position is not None``: caller
      should move to ``candidate_position`` directly.
    - ``(None, "", None)``: neither strategy found a viable action.
    """
    goal_direction = normalize(goal - position)

    # --- 1. Widening push: prefer semantics over geometry ---
    for obs_id in [obs_id_a, obs_id_b]:
        obs = next((o for o in scene["obstacles"] if o["id"] == obs_id), None)
        if obs is None:
            continue
        p_safe, _ = obstacle_safety_probability(obs)
        if obs["true_class"] != "movable" or p_safe < SAFE_PROB_THRESHOLD:
            continue
        obs_idx = next(
            (i for i, o in enumerate(scene["obstacles"]) if o["id"] == obs_id), None
        )
        if obs_idx is not None:
            return None, f"corridor widening: push obs {obs_id} to widen gap", obs_idx

    # --- 2. Frontier detour: exploit unexplored space, not known geometry ---
    if not frontier_points:
        return None, "", None

    # Find the direction toward the corridor bottleneck so we can steer away.
    obs_a = next((o for o in scene["obstacles"] if o["id"] == obs_id_a), None)
    obs_b = next((o for o in scene["obstacles"] if o["id"] == obs_id_b), None)
    blocked_dir = np.zeros(2, dtype=float)
    if obs_a is not None and obs_b is not None:
        pt_a, pt_b = nearest_points(obstacle_polygon(obs_a), obstacle_polygon(obs_b))
        bottleneck = np.array([(pt_a.x + pt_b.x) / 2.0, (pt_a.y + pt_b.y) / 2.0])
        blocked_dir = normalize(bottleneck - position)

    best_frontier: np.ndarray | None = None
    best_score = -float("inf")
    for fp in frontier_points:
        fp_dir = normalize(fp - position)
        goal_alignment = float(np.dot(fp_dir, goal_direction))
        # Penalize frontiers that point directly into the blocked corridor.
        corridor_avoidance = -float(np.dot(fp_dir, blocked_dir))
        score = 0.6 * goal_alignment + 0.4 * corridor_avoidance
        if score > best_score:
            best_score = score
            best_frontier = fp

    if best_frontier is None or best_score <= 0.0:
        return None, "", None

    fp_dir = normalize(best_frontier - position)
    candidate = safe_step_position(
        scene, position, fp_dir, step_size * 1.3, observed_only=True
    )
    if np.linalg.norm(candidate - position) > 1e-6:
        return (
            candidate,
            f"frontier detour: around blocked corridor (obs {obs_id_a}/{obs_id_b})",
            None,
        )

    return None, "", None


def move_obstacle_in_direction(
    scene: dict,
    obstacle_index: int,
    direction: np.ndarray,
    displacement: float,
    robot_position: np.ndarray | None = None,
) -> tuple[float, list[int]]:
    """Move a contacted obstacle and any chain-contact neighbors.

    The lead obstacle moves in the robot push direction. Any obstacle in the
    induced contact chain moves less according to

        delta_level = delta_0 * CHAIN_ATTENUATION^level.

    This is a simple quasi-static energy-loss model rather than a full rigid-body
    simulator, but it preserves the direction of the push and weakens secondary
    motion in a physically sensible way.
    """
    obstacle = scene["obstacles"][obstacle_index]
    if robot_position is not None:
        direction = obstacle_push_direction(obstacle, robot_position, direction)
    actual_displacement, _ = feasible_chain_push_distance(
        scene,
        obstacle_index,
        direction,
        requested_displacement=displacement,
        robot_position=robot_position,
    )
    if actual_displacement <= 1e-6:
        return 0.0, [obstacle_index]

    chain_levels = moved_chain_levels(scene, obstacle_index, direction, actual_displacement)
    for idx, level in chain_levels.items():
        moved_poly = _translated_obstacle_poly(
            scene["obstacles"][idx],
            direction,
            actual_displacement * (CHAIN_ATTENUATION ** level),
        )
        scene["obstacles"][idx]["vertices"] = polygon_to_vertices(moved_poly)
        invalidate_polygon_cache(scene["obstacles"][idx])
    return actual_displacement, sorted(chain_levels.keys())


def run_online_surp_push(
    scene: dict,
    epsilon: float = 0.10,
    sensing_range: float = DEFAULT_SENSING_RANGE,
    step_size: float = 0.07,
    push_distance: float = 0.12,
    max_steps: int = 500,
) -> OnlineSurpResult:
    """Run the online goal-seeking / avoid / push controller on one scene.

    The controller is hybrid:
    - goal mode: move straight toward the goal,
    - avoid mode: choose a local detour or escape maneuver,
    - contact mode: push a semantically safe obstacle if that is cheaper.

    The basic goal command is the normalized direct-to-goal vector

        u_goal = v_max * (g - p) / ||g - p||,

    implemented here with fixed step length rather than continuous dynamics.
    """
    working_scene = normalize_scene_for_online_use(scene)
    position = np.array(working_scene["start"][:2], dtype=float)
    goal = np.array(working_scene["goal"][:2], dtype=float)

    path = [tuple(position)]
    frames = [snapshot_frame(position, working_scene, "start")]
    contact_log: list[str] = []
    sensed_ids: list[int] = []
    goal_distance_history = [float(np.linalg.norm(goal - position))]
    bad_directions: list[dict] = []
    push_history: list[dict] = []
    last_boundary_stop_id: int | None = None
    boundary_stop_events: list[int] = []
    stuck_events = 0
    consecutive_free_steps = 0
    bug_follow_boundary: bool = False
    bug_hit_point: np.ndarray | None = None
    bug_boundary_direction: np.ndarray | None = None
    bug_follow_obs_id: int | None = None
    bug_boundary_steps: int = 0
    bug_last_exited_obs_id: int | None = None
    bug_bounce_count: int = 0
    bug_recent_obs_ids: list[int] = []   # pinball detector: last N boundary entries

    success = False
    for _ in range(max_steps):
        if np.linalg.norm(goal - position) <= step_size:
            position = goal.copy()
            path.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "goal reached"))
            success = True
            break

        perception = update_local_perception(working_scene, position, sensing_range)
        newly_observed = perception["newly_observed"]
        if newly_observed:
            observed_ids = [obstacle["id"] for obstacle in newly_observed]
            sensed_ids.extend(observed_ids)
            path.append(tuple(position))
            frames.append(
                snapshot_frame(
                    position,
                    working_scene,
                    f"sensed obstacle(s) {observed_ids}; frontiers={len(perception['frontiers'])}",
                )
            )
            continue

        goal_direction = normalize(goal - position)
        if np.linalg.norm(goal_direction) <= 1e-9:
            break

        avoid_set, push_set = classify_sensed_obstacles(
            working_scene,
            position,
            goal_direction,
            epsilon,
        )
        nearest_idx, nearest_distance = nearest_sensed_obstacle(working_scene, position)
        stalled = goal_progress_is_stalled(goal_distance_history)

        if nearest_idx is None or nearest_distance > epsilon:
            last_boundary_stop_id = None
            bug_follow_boundary = False
            bug_recent_obs_ids.clear()
            direct_candidate = step_until_sense_or_contact(
                working_scene,
                position,
                goal_direction,
                step_size,
                epsilon,
            )
            if np.linalg.norm(direct_candidate - position) > 1e-6:
                position = direct_candidate
                consecutive_free_steps += 1
                if consecutive_free_steps >= 8:
                    stuck_events = max(0, stuck_events - 1)
                    consecutive_free_steps = 0
                goal_distance_history.append(float(np.linalg.norm(goal - position)))
                path.append(tuple(position))
                frames.append(snapshot_frame(position, working_scene, "goal mode: moving directly toward goal"))
                continue

        if nearest_idx is None:
            goal_distance_history.append(float(np.linalg.norm(goal - position)))
            path.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "no sensed obstacle but direct motion unavailable"))
            continue

        nearest_obstacle = working_scene["obstacles"][nearest_idx]

        if last_boundary_stop_id != nearest_obstacle["id"]:
            last_boundary_stop_id = nearest_obstacle["id"]
            boundary_stop_events.append(nearest_obstacle["id"])
            if len(boundary_stop_events) > 14:
                boundary_stop_events = boundary_stop_events[-14:]

            # Semantic response to oscillation: widen the corridor or detour via frontiers.
            osc_pair = detect_boundary_oscillation(boundary_stop_events)
            if osc_pair is not None:
                sc_pos, sc_label, sc_push_idx = semantic_corridor_response(
                    working_scene,
                    position,
                    goal,
                    osc_pair[0],
                    osc_pair[1],
                    step_size,
                    perception["frontiers"],
                )
                if sc_push_idx is not None:
                    # Widen the corridor by pushing the pushable wall aside.
                    moved_distance, moved_chain = move_obstacle_in_direction(
                        working_scene,
                        sc_push_idx,
                        goal_direction,
                        push_distance,
                        robot_position=position,
                    )
                    if moved_distance > 1e-6:
                        position = step_until_sense_or_contact(
                            working_scene, position, goal_direction, step_size, epsilon
                        )
                        pushed_obs = working_scene["obstacles"][sc_push_idx]
                        pushed_obs["push_count"] = pushed_obs.get("push_count", 0) + 1
                        push_history.append({
                            "obstacle_id": pushed_obs["id"],
                            "distance": moved_distance,
                            "chain": moved_chain,
                        })
                        stuck_events = 0
                        goal_distance_history.append(float(np.linalg.norm(goal - position)))
                        chain_text = f" chain {moved_chain}" if len(moved_chain) > 1 else ""
                        message = (
                            f"corridor widening: pushed obs {pushed_obs['id']} "
                            f"by {moved_distance:.2f}{chain_text}"
                        )
                        path.append(tuple(position))
                        frames.append(snapshot_frame(position, working_scene, message))
                        contact_log.append(message)
                        continue
                elif sc_pos is not None and np.linalg.norm(sc_pos - position) > 1e-6:
                    position = sc_pos
                    goal_distance_history.append(float(np.linalg.norm(goal - position)))
                    path.append(tuple(position))
                    frames.append(snapshot_frame(position, working_scene, sc_label))
                    continue

            path.append(tuple(position))
            frames.append(
                snapshot_frame(
                    position,
                    working_scene,
                    f"stopped at epsilon boundary near obstacle {nearest_obstacle['id']}",
                )
            )
            continue

        p_safe, safety_score = obstacle_safety_probability(nearest_obstacle)

        # Determine whether pushing this obstacle is both viable and optimal.
        is_pushable = (
            nearest_obstacle["true_class"] == "movable"
            and p_safe >= SAFE_PROB_THRESHOLD
        )
        push_candidate = None
        push_is_optimal = False
        avoid_step_target = position  # safe default — avoids NameError below
        J_avoid = 0.0
        J_push = float("inf")
        available_push_distance = 0.0
        corridor_gain = 0.0

        if is_pushable:
            avoid_candidate = compute_avoid_trajectory(
                working_scene,
                position,
                goal,
                step_size,
                epsilon,
                bad_directions,
            )
            push_candidate = compute_push_trajectory(
                working_scene,
                position,
                goal,
                goal_direction,
                push_set,
                step_size,
                epsilon,
                push_distance,
            )
            selected_candidate = reconcile_trajectory_decision(
                avoid_candidate,
                push_candidate,
                safety_margin_threshold=max(0.02, epsilon * 0.2),
            )
            J_avoid = avoid_candidate.total_cost
            avoid_step_target = avoid_candidate.step_target
            if push_candidate is not None:
                J_push = push_candidate.total_cost
                available_push_distance = push_candidate.push_distance
                corridor_gain = push_candidate.corridor_gain
            push_is_optimal = (
                selected_candidate.mode == "push"
                and available_push_distance > 1e-6
                and push_candidate is not None
            )

        # ── Boundary mode ─────────────────────────────────────────────────────
        # Activated whenever pushing is not the chosen action — covers both
        # unmovable obstacles (can't push) and movable obstacles where the cost
        # model prefers avoidance (not optimal to push).  Uses the Bug1
        # left-hand rule: turn 90° left on first contact, then fine-sweep CCW
        # to hug the surface until line-of-sight to the goal reopens.
        _boundary_trapped = False
        if not push_is_optimal:
            obs_id = nearest_obstacle["id"]
            obs_class = nearest_obstacle["true_class"]

            # Primary exit: check whether a direct step toward the goal is
            # collision-free right now.  Do this before any sweep so the robot
            # exits and moves in the same iteration — preventing immediate
            # re-entry on the very next step (which caused the oscillation loop).
            direct_goal_pos = clip_point_to_workspace(
                working_scene, position + step_size * goal_direction
            )
            if not point_collides(working_scene, direct_goal_pos, observed_only=True):
                if bug_follow_boundary:
                    bug_last_exited_obs_id = obs_id
                    bug_follow_boundary = False
                    path.append(tuple(position))
                    frames.append(snapshot_frame(
                        position, working_scene,
                        f"boundary mode: direct path clear — leaving obs {obs_id}",
                    ))
                position = direct_goal_pos
                goal_distance_history.append(float(np.linalg.norm(goal - position)))
                path.append(tuple(position))
                frames.append(snapshot_frame(
                    position, working_scene, "goal mode: moving directly toward goal"
                ))
                continue

            # Direct path is blocked — boundary following is needed.
            if not bug_follow_boundary or bug_follow_obs_id != obs_id:
                # Track recent boundary entries for pinball detection.
                bug_recent_obs_ids.append(obs_id)
                if len(bug_recent_obs_ids) > 6:
                    bug_recent_obs_ids.pop(0)

                # Same-obstacle bounce: re-entering an obstacle we just exited
                # means the direct-path exit leads into a dead-end corridor.
                if obs_id == bug_last_exited_obs_id:
                    bug_bounce_count += 1
                else:
                    bug_bounce_count = 0
                    bug_last_exited_obs_id = None

                # Pinball detection: cycling between ≤2 obstacles with no
                # progress (two-obstacle corridor that same-obstacle bounce
                # counting cannot detect since the IDs always differ).
                pinball = (
                    len(bug_recent_obs_ids) >= 6
                    and len(set(bug_recent_obs_ids)) <= 2
                )

                # After enough bounces or a confirmed pinball, the cost model
                # is wrong: push the obstacle if possible, otherwise fall
                # through to stall recovery.
                if (bug_bounce_count >= 3 or pinball) and is_pushable:
                    push_idx_override = next(
                        (i for i, o in enumerate(working_scene["obstacles"])
                         if o["id"] == obs_id),
                        None,
                    )
                    if push_idx_override is not None:
                        moved_distance, moved_chain = move_obstacle_in_direction(
                            working_scene,
                            push_idx_override,
                            goal_direction,
                            push_distance,
                            robot_position=position,
                        )
                        if moved_distance > 1e-6:
                            position = step_until_sense_or_contact(
                                working_scene, position, goal_direction, step_size, epsilon
                            )
                            pushed_obs = working_scene["obstacles"][push_idx_override]
                            pushed_obs["push_count"] = pushed_obs.get("push_count", 0) + 1
                            push_history.append({
                                "obstacle_id": pushed_obs["id"],
                                "distance": moved_distance,
                                "chain": moved_chain,
                            })
                            stuck_events = 0
                            n_bounces = bug_bounce_count
                            bug_bounce_count = 0
                            bug_last_exited_obs_id = None
                            bug_recent_obs_ids.clear()
                            goal_distance_history.append(float(np.linalg.norm(goal - position)))
                            chain_text = f" chain {moved_chain}" if len(moved_chain) > 1 else ""
                            reason = "pinball" if pinball else f"{n_bounces} bounces"
                            message = (
                                f"boundary bounce override: pushed obs {pushed_obs['id']} "
                                f"by {moved_distance:.2f}{chain_text} ({reason})"
                            )
                            path.append(tuple(position))
                            frames.append(snapshot_frame(position, working_scene, message))
                            contact_log.append(message)
                            continue
                    # Push failed or not pushable — treat as trapped
                    bug_bounce_count = 0
                    bug_recent_obs_ids.clear()
                    _boundary_trapped = True

                if not _boundary_trapped:
                    # First contact (or new obstacle) — enter boundary mode.
                    bug_follow_boundary = True
                    bug_follow_obs_id = obs_id
                    bug_hit_point = position.copy()
                    bug_boundary_direction = _bug_rotate(goal_direction, 90.0)
                    bug_boundary_steps = 0
                    path.append(tuple(position))
                    frames.append(snapshot_frame(
                        position, working_scene,
                        f"boundary mode: entered near obs {obs_id} "
                        f"({obs_class}, P_safe={p_safe:.2f})",
                    ))
                    continue

            # Already in boundary mode — sweep for the best tangential step.
            new_dir, next_pos = _scholar_best_boundary_direction(
                working_scene, position, bug_boundary_direction, step_size
            )
            if new_dir is not None:
                bug_boundary_direction = new_dir
                bug_boundary_steps += 1
                stuck_events = max(0, stuck_events - 1)
                consecutive_free_steps = 0
                moved_away = (
                    float(np.linalg.norm(position - bug_hit_point)) > step_size * 1.5
                    if bug_hit_point is not None else True
                )
                # Circuit detection: returned near hit_point after a full orbit,
                # or absolute step cap for very large obstacles.
                circuit_done = (not moved_away and bug_boundary_steps >= 4)
                orbit_timeout = (bug_boundary_steps >= 120)
                if circuit_done or orbit_timeout:
                    reason = "circuit complete" if circuit_done else "orbit timeout"
                    bug_follow_boundary = False
                    _boundary_trapped = True
                    path.append(tuple(position))
                    frames.append(snapshot_frame(
                        position, working_scene,
                        f"boundary mode: {reason} for obs {obs_id} — escalating",
                    ))
                else:
                    position = next_pos
                    goal_distance_history.append(float(np.linalg.norm(goal - position)))
                    path.append(tuple(position))
                    frames.append(snapshot_frame(
                        position, working_scene,
                        f"boundary mode: following obs {obs_id} ({obs_class})",
                    ))
                if not _boundary_trapped:
                    continue

            if not _boundary_trapped:
                # 360° sweep found nothing — completely trapped.
                bug_follow_boundary = False
                _boundary_trapped = True

        # ── Stall-recovery cascade ─────────────────────────────────────────────
        # Reached when: (a) boundary mode is fully trapped, or
        #               (b) push was chosen but goal progress stalled.
        if stalled or _boundary_trapped:
            stuck_events += 1
            consecutive_free_steps = 0
            remember_bad_direction(bad_directions, position, goal_direction)
            if len(bad_directions) > 24:
                bad_directions = bad_directions[-24:]

            push_idx = choose_best_pushable_obstacle(
                working_scene,
                position,
                goal_direction,
                epsilon,
            )
            if push_idx is not None and nearest_idx is not None and push_idx != nearest_idx:
                nearest_obstacle = working_scene["obstacles"][nearest_idx]
                nearest_distance = robot_clearance_to_obstacle(position, nearest_obstacle)
                if nearest_obstacle["true_class"] != "movable" or nearest_distance <= epsilon + 0.03:
                    push_idx = None
            if push_idx is not None:
                pushed_obstacle = working_scene["obstacles"][push_idx]
                moved_distance, moved_chain = move_obstacle_in_direction(
                    working_scene,
                    push_idx,
                    goal_direction,
                    push_distance,
                    robot_position=position,
                )
                if moved_distance > 1e-6:
                    position = step_until_sense_or_contact(
                        working_scene,
                        position,
                        goal_direction,
                        step_size,
                        epsilon,
                    )
                    stuck_events = 0
                    goal_distance_history.append(float(np.linalg.norm(goal - position)))
                    chain_text = f" with chain {moved_chain}" if len(moved_chain) > 1 else ""
                    message = (
                        f"contact mode (stuck override): pushed obstacle "
                        f"{pushed_obstacle['id']} by {moved_distance:.2f}{chain_text}"
                    )
                    pushed_obstacle["push_count"] = pushed_obstacle.get("push_count", 0) + 1
                    push_history.append({
                        "obstacle_id": pushed_obstacle["id"],
                        "distance": moved_distance,
                        "chain": moved_chain,
                    })
                    path.append(tuple(position))
                    frames.append(snapshot_frame(position, working_scene, message))
                    contact_log.append(message)
                    continue

            # Try RRT first — backtrack oscillates when the path revisits the
            # same cluster, but RRT can find genuinely new free-space routes.
            _, rrt_candidate = rrt_escape_step(
                working_scene,
                position,
                goal,
                step_size,
                bad_directions=bad_directions,
                max_samples=120 + 40 * min(stuck_events, 4),
            )
            if rrt_candidate is not None and np.linalg.norm(rrt_candidate - position) > 1e-6:
                position = rrt_candidate
                goal_distance_history.append(float(np.linalg.norm(goal - position)))
                path.append(tuple(position))
                frames.append(
                    snapshot_frame(
                        position,
                        working_scene,
                        "avoid mode (RRT escape): retreating and replanning through sensed free space",
                    )
                )
                continue

            # RRT failed — fall back to backtrack along the executed path.
            backtrack_target = choose_deep_backtrack_target(path, position, stuck_events)
            if backtrack_target is not None:
                retreat_direction = normalize(backtrack_target - position)
                retreat_step_size = min(
                    max(step_size * (1.8 + 0.35 * min(stuck_events, 5)), step_size),
                    float(np.linalg.norm(backtrack_target - position)),
                )
                retreat_candidate = safe_step_position(
                    working_scene,
                    position,
                    retreat_direction,
                    retreat_step_size,
                    observed_only=True,
                )
                if np.linalg.norm(retreat_candidate - position) > 1e-6:
                    position = retreat_candidate
                    goal_distance_history.append(float(np.linalg.norm(goal - position)))
                    path.append(tuple(position))
                    frames.append(
                        snapshot_frame(
                            position,
                            working_scene,
                            f"avoid mode (memory replan): deeper backtrack level {stuck_events}",
                        )
                    )
                    continue

            _, escape_candidate = choose_escape_motion(
                working_scene,
                position,
                goal,
                step_size,
                epsilon,
                bad_directions=bad_directions,
            )
            if np.linalg.norm(escape_candidate - position) > 1e-6:
                position = escape_candidate
                goal_distance_history.append(float(np.linalg.norm(goal - position)))
                path.append(tuple(position))
                frames.append(
                    snapshot_frame(
                        position,
                        working_scene,
                        "avoid mode (memory replan): local escape from non-pushable trap",
                    )
                )
                continue
        else:
            consecutive_free_steps += 1
            if consecutive_free_steps >= 8:
                stuck_events = max(0, stuck_events - 1)
                consecutive_free_steps = 0

        # ── Push execution ─────────────────────────────────────────────────────
        # Only reached when push_is_optimal is True and stall recovery did not
        # fire or did not find a move.
        if push_is_optimal and push_candidate is not None and push_candidate.obstacle_index is not None:
            moved_distance, moved_chain = move_obstacle_in_direction(
                working_scene,
                push_candidate.obstacle_index,
                goal_direction,
                push_candidate.push_distance,
                robot_position=position,
            )
            if moved_distance > 1e-6:
                position = step_until_sense_or_contact(
                    working_scene,
                    position,
                    goal_direction,
                    step_size,
                    epsilon,
                )
                stuck_events = 0
                goal_distance_history.append(float(np.linalg.norm(goal - position)))
                chain_text = f" with chain {moved_chain}" if len(moved_chain) > 1 else ""
                pushed_obstacle = working_scene["obstacles"][push_candidate.obstacle_index]
                pushed_obstacle["push_count"] = pushed_obstacle.get("push_count", 0) + 1
                push_history.append({
                    "obstacle_id": pushed_obstacle["id"],
                    "distance": moved_distance,
                    "chain": moved_chain,
                })
                message = (
                    f"contact mode: pushed obstacle {pushed_obstacle['id']} by "
                    f"{moved_distance:.2f}{chain_text}; push_history={len(push_history)}"
                )
                path.append(tuple(position))
                frames.append(snapshot_frame(position, working_scene, message))
                contact_log.append(message)
                continue

        # Push was selected but failed to move the obstacle — fall back to the
        # avoid trajectory computed during reconciliation.
        if np.linalg.norm(avoid_step_target - position) > 1e-6:
            position = avoid_step_target
            consecutive_free_steps += 1
            if consecutive_free_steps >= 8:
                stuck_events = max(0, stuck_events - 1)
                consecutive_free_steps = 0
            goal_distance_history.append(float(np.linalg.norm(goal - position)))
            path.append(tuple(position))
            frames.append(
                snapshot_frame(
                    position,
                    working_scene,
                    (
                        f"avoid fallback: push failed for obs {nearest_obstacle['id']} "
                        f"(P_safe={p_safe:.2f}, J_push={J_push:.2f}, J_avoid={J_avoid:.2f})"
                    ),
                )
            )
            continue

        goal_distance_history.append(float(np.linalg.norm(goal - position)))
        path.append(tuple(position))
        frames.append(
            snapshot_frame(
                position,
                working_scene,
                (
                    f"all strategies exhausted: obstacle {nearest_obstacle['id']} "
                    f"(P_safe={p_safe:.2f}, score={safety_score:.2f}, "
                    f"J_push={J_push:.2f}, J_avoid={J_avoid:.2f}, gain={corridor_gain:.2f})"
                ),
            )
        )

    return OnlineSurpResult(
        family=working_scene["family"],
        seed=working_scene["seed"],
        success=success,
        path=path,
        frames=frames,
        scene=working_scene,
        initial_scene=copy.deepcopy(scene),
        contact_log=contact_log,
        sensed_ids=sensed_ids,
    )
