import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scipy.interpolate import make_interp_spline
from shapely.geometry import LineString, Point

from NewProject.constants import DEFAULT_SENSING_RANGE
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

from planning.cost_map import (
    SEMANTIC_COSTS,
    CostMap,
    build_voxel_grid,
    compute_anisotropic_map,
    update as update_cost_map,
)
from utils.geometry import ray_cast, visibility_range

V_MAX            = 0.15    # max robot speed (m/step)
N_RAYS           = 36      # rays for visibility polygon
N_FRONTIER_POS   = 10      # frontier positions sampled per tick
C_MAX            = 7.0     # max obstacle cost allowed for contact (Stage 3)
FA               = 2.0     # heading-deviation penalty factor (J_pos)
HEADING_THRESH   = np.pi / 4

# Fixed objective weights
W_P   = 1.0   # position term
W_B   = 0.5   # resource term

# Risk sub-weights  (sum = 1)
W_GEO = 0.4
W_SEM = 0.4
W_DIR = 0.2

# Resource sub-weights  (sum = 1)
W_DIST = 0.4
W_TIME = 0.3
W_COL  = 0.3

# Battery-adaptive weight schedule
W_R0    = 2.0   # w_r at full battery
W_V_MIN = 0.5   # w_v at full battery
W_V_MAX = 4.0   # w_v at empty battery

# Trajectory generation
P_SAFE = 1.2    # safety margin on traversal duration
T_MAP  = 0.2    # minimum trajectory duration (s)

# Maneuver selection
F_MIN  = 0.3    # minimum f_s to consider PUSH safe

# 8 velocity direction unit vectors
_VEL_DIRS = np.array([
    [np.cos(a), np.sin(a)]
    for a in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
])

@dataclass
class RobotState:
    position:    np.ndarray   # (2,) world coords
    velocity:    np.ndarray   # (2,) current velocity vector
    battery:     float = 1.0  # current charge, normalised [0, 1]
    battery_max: float = 1.0  # B0


@dataclass
class CandidateState:
    position:        np.ndarray          # (2,)
    velocity:        np.ndarray          # (2,)
    obstacle_contact: Optional[dict] = None  # nearest obs if position overlaps one
    contact_cost:    float = 0.0

class Scholar:
    def __init__(
        self,
        v_max:           float = V_MAX,
        n_rays:          int   = N_RAYS,
        n_frontier_pos:  int   = N_FRONTIER_POS,
        sensing_range:   float = DEFAULT_SENSING_RANGE,
    ):
        self.v_max          = v_max
        self.n_rays         = n_rays
        self.n_frontier_pos = n_frontier_pos
        self.sensing_range  = sensing_range

    def select_next_state(
        self,
        robot:    RobotState,
        scene:    dict,
        cost_map: CostMap,
        goal:     np.ndarray,
    ) -> CandidateState:
        observed = [o for o in scene["obstacles"] if o.get("observed", False)]

        # ── 1. Sample candidates ───────────────────────────────────────────
        candidates = self._sample_candidates(robot, scene, cost_map, goal, observed)

        if not candidates:
            # Fallback: step straight toward goal
            d = normalize(goal - robot.position)
            return CandidateState(
                position=robot.position + d * self.v_max,
                velocity=d * self.v_max,
            )

        # ── 2. Three-stage pruning ─────────────────────────────────────────
        candidates = self._prune_position(candidates, robot.position, goal, cost_map)
        candidates = self._prune_velocity(candidates, robot.position, scene, observed)
        candidates = self._prune_semantic(candidates)

        # ── 3. Battery-adaptive weights ────────────────────────────────────
        b_frac = min(1.0, max(0.0, robot.battery / max(robot.battery_max, 1e-9)))
        w_r = W_R0 * b_frac
        w_v = W_V_MIN + (W_V_MAX - W_V_MIN) * (1.0 - b_frac)

        # ── 4. Score and return argmin ─────────────────────────────────────
        scores = [
            self._score(q, robot, goal, cost_map, observed, w_r, w_v)
            for q in candidates
        ]
        return candidates[int(np.argmin(scores))]

    # ── Step 1: candidate sampling ─────────────────────────────────────────────

    def _sample_candidates(
        self,
        robot:    RobotState,
        scene:    dict,
        cost_map: CostMap,
        goal:     np.ndarray,
        observed: list,
    ) -> List[CandidateState]:
        candidates: List[CandidateState] = []
        pos = robot.position

        # Frontier positions: rays that reach sensing_range without obstacle
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_rays, endpoint=False)
        frontier_positions: List[np.ndarray] = []

        for angle in angles:
            d = np.array([np.cos(angle), np.sin(angle)])
            dist, obs_id = ray_cast(pos, d, observed, max_range=self.sensing_range)
            if obs_id is None:
                # Ray reached sensing limit → frontier direction
                # Step to 90% of range to stay inside workspace
                fp = clip_point_to_workspace(scene, pos + d * dist * 0.9)
                frontier_positions.append(fp)

        # Subsample to n_frontier_pos positions (evenly spaced in list)
        if len(frontier_positions) > self.n_frontier_pos:
            idx = np.round(
                np.linspace(0, len(frontier_positions) - 1, self.n_frontier_pos)
            ).astype(int)
            frontier_positions = [frontier_positions[i] for i in idx]

        # Always include the goal itself as a candidate position
        frontier_positions.append(goal.copy())

        # Build candidates: each position × 8 dirs × 3 speeds
        speeds = [0.0, self.v_max / 2.0, self.v_max]
        for fp in frontier_positions:
            # Determine contact with nearest obstacle at this position
            contact_obs, contact_cost = self._nearest_contact(fp, observed)

            for vel_dir in _VEL_DIRS:
                for speed in speeds:
                    vel = vel_dir * speed
                    candidates.append(CandidateState(
                        position=fp.copy(),
                        velocity=vel.copy(),
                        obstacle_contact=contact_obs,
                        contact_cost=contact_cost,
                    ))

        return candidates

    def _nearest_contact(
        self, position: np.ndarray, observed: list
    ) -> Tuple[Optional[dict], float]:
        body = robot_body(position)
        best_dist = float("inf")
        best_obs  = None

        for obs in observed:
            poly = obstacle_polygon(obs)
            dist = float(body.distance(poly))
            if dist < best_dist:
                best_dist = dist
                best_obs = obs

        if best_obs is None:
            return None, 0.0
        sem = best_obs.get("true_class", best_obs.get("class_true", "movable"))
        cost = float(SEMANTIC_COSTS.get(sem, 1.0))
        return best_obs, cost

    # ── Step 2a: position dominance pruning ───────────────────────────────────

    def _prune_position(
        self,
        candidates: List[CandidateState],
        robot_pos:  np.ndarray,
        goal:       np.ndarray,
        cost_map:   CostMap,
    ) -> List[CandidateState]:
        if len(candidates) <= 1:
            return candidates

        dists = [float(np.linalg.norm(q.position - goal)) for q in candidates]
        order = np.argsort(dists)           # sorted closest → farthest
        dominated = [False] * len(candidates)

        for j_idx, j in enumerate(order):  # j is closer to goal
            if dominated[j]:
                continue
            for k in order[j_idx + 1:]:    # k is farther from goal
                if dominated[k]:
                    continue
                if self._line_free(candidates[k].position,
                                   candidates[j].position, cost_map):
                    dominated[k] = True

        surviving = [q for i, q in enumerate(candidates) if not dominated[i]]
        return surviving if surviving else candidates

    def _line_free(
        self, a: np.ndarray, b: np.ndarray, cost_map: CostMap, n_checks: int = 10
    ) -> bool:
        for t in np.linspace(0.0, 1.0, n_checks):
            pt = a + t * (b - a)
            if cost_map.cost_at_world(pt) > 0.0:
                return False
        return True

    def _prune_velocity(
        self,
        candidates: List[CandidateState],
        robot_pos:  np.ndarray,
        scene:      dict,
        observed:   list,
    ) -> List[CandidateState]:
        if not observed:
            return candidates

        unobserved = [o for o in scene["obstacles"] if not o.get("observed", False)]
        if not unobserved:
            return candidates  # nothing left to discover — keep all

        surviving = []
        for q in candidates:
            speed = float(np.linalg.norm(q.velocity))
            if speed < 1e-9:
                surviving.append(q)    # stationary candidate always survives
                continue

            vel_dir = q.velocity / speed
            # Keep if velocity direction has positive component toward any
            # unobserved obstacle (would lead to new information)
            points_toward_unknown = any(
                float(np.dot(vel_dir,
                             normalize(np.array(o["vertices"]).mean(axis=0) - q.position)
                             )) > 0.0
                for o in unobserved
            )
            if points_toward_unknown:
                surviving.append(q)

        return surviving if surviving else candidates

    def _prune_semantic(
        self, candidates: List[CandidateState], c_max: float = C_MAX
    ) -> List[CandidateState]:
        safe = [q for q in candidates if q.contact_cost <= c_max]
        if safe:
            return safe
        # Relax: keep minimum-cost contact
        return [min(candidates, key=lambda q: q.contact_cost)]

    def _score(
        self,
        q:         CandidateState,
        robot:     RobotState,
        goal:      np.ndarray,
        cost_map:  CostMap,
        observed:  list,
        w_r:       float,
        w_v:       float,
    ) -> float:
        j_pos      = self._j_pos(q, goal)
        j_risk     = self._j_risk_sem(q, goal, cost_map, observed)
        j_vel      = self._j_vel_sem(q, goal, observed)
        j_resource = self._j_resource(q, robot)
        return W_P * j_pos + w_r * j_risk + w_v * j_vel + W_B * j_resource

    def _j_pos(self, q: CandidateState, goal: np.ndarray) -> float:
        dist = float(np.linalg.norm(q.position - goal))
        speed = float(np.linalg.norm(q.velocity))
        if speed > 1e-9:
            heading = q.velocity / speed
        else:
            heading = normalize(goal - q.position)

        goal_dir = normalize(goal - q.position)
        cos_dev  = float(np.clip(np.dot(heading, goal_dir), -1.0, 1.0))
        deviation = float(np.arccos(cos_dev))
        fa = FA if deviation > HEADING_THRESH else 1.0
        return dist * fa

    def _j_risk_sem(
        self,
        q:        CandidateState,
        goal:     np.ndarray,
        cost_map: CostMap,
        observed: list,
    ) -> float:
        speed     = float(np.linalg.norm(q.velocity))
        body      = robot_body(q.position)

        # Geometric risk: 1/t_c, normalised to [0,1]
        min_dist = float("inf")
        nearest_cost = 0.0
        for obs in observed:
            d = float(body.distance(obstacle_polygon(obs)))
            if d < min_dist:
                min_dist = d
                sem = obs.get("true_class", obs.get("class_true", "movable"))
                nearest_cost = float(SEMANTIC_COSTS.get(sem, 1.0))

        t_c = min_dist / max(speed, 1e-6)
        # Normalise: t_c → ∞ gives 0 risk, t_c → 0 gives 1.0
        geo_risk = 1.0 / (1.0 + t_c)

        # Semantic risk of nearest obstacle
        sem_risk = min(1.0, nearest_cost / 10.0)

        # Directional safety from anisotropic map
        aniso_val = cost_map.cost_at_world(q.position)
        # M' ∈ [0,10]: high → unsafe; f_s ≈ 1 − M'/10
        f_s = 1.0 - min(1.0, max(0.0, float(aniso_val) / 10.0))
        dir_risk = 1.0 - f_s   # = M'/10

        return W_GEO * geo_risk + W_SEM * sem_risk + W_DIR * dir_risk

    def _j_vel_sem(
        self,
        q:       CandidateState,
        goal:    np.ndarray,
        observed: list,
    ) -> float:
        speed = float(np.linalg.norm(q.velocity))
        if speed < 1e-9:
            return 0.0

        vel_dir  = q.velocity / speed
        goal_dir = normalize(goal - q.position)

        # Find nearest obstacle surface to get outward normal
        body     = robot_body(q.position)
        min_dist = float("inf")
        nearest_normal = goal_dir          # fallback
        nearest_cost   = 0.0

        for obs in observed:
            poly = obstacle_polygon(obs)
            d    = float(body.distance(poly))
            if d < min_dist:
                min_dist = d
                # Outward normal: direction from nearest poly point toward robot
                near_pt = poly.exterior.interpolate(
                    poly.exterior.project(Point(q.position[0], q.position[1]))
                )
                diff = q.position - np.array([near_pt.x, near_pt.y])
                nn   = float(np.linalg.norm(diff))
                nearest_normal = diff / nn if nn > 1e-9 else goal_dir
                sem = obs.get("true_class", obs.get("class_true", "movable"))
                nearest_cost = float(SEMANTIC_COSTS.get(sem, 1.0))

        # Garwin reflection: v_r = v − 2(v·n̂)n̂
        v_ref = vel_dir - 2.0 * float(np.dot(vel_dir, nearest_normal)) * nearest_normal
        n_ref = float(np.linalg.norm(v_ref))
        if n_ref > 1e-9:
            v_ref = v_ref / n_ref

        # Reward = how well reflected velocity aligns with goal direction, ∈ [0,1]
        reflection_reward = (float(np.dot(v_ref, goal_dir)) + 1.0) / 2.0
        j_vel = 1.0 - reflection_reward    # lower is better

        return j_vel * (1.0 - min(1.0, nearest_cost / 10.0))

    def _j_resource(
        self, q: CandidateState, robot: RobotState
    ) -> float:
        dist  = float(np.linalg.norm(q.position - robot.position))
        speed = float(np.linalg.norm(q.velocity))
        T_trav = dist / max(speed, 1e-6)

        in_contact = q.obstacle_contact is not None and q.contact_cost > 0.0
        if in_contact:
            # Kinetic energy proxy: ½mv²  (use unit mass for unknown objects)
            dE_col = 0.5 * speed ** 2
        else:
            dE_col = 0.0

        return (W_DIST * dist
                + W_TIME * T_trav
                + W_COL  * float(in_contact) * dE_col)

    # ── generate_trajectory ───────────────────────────────────────────────────

    def generate_trajectory(
        self,
        q_star: "CandidateState",
        robot:  "RobotState",
    ):
        """Fit a cubic B-spline through four waypoints.

        Waypoints
        ---------
        p0 = robot.position
        p1 = robot.position + (q_star.position - robot.position) * 1/3
        p2 = robot.position + (q_star.position - robot.position) * 2/3
        p3 = q_star.position

        Trajectory duration
        -------------------
        T = max(T_v, T_map)
        where T_v = P_safe * total_path_length / v_max
              T_map = 0.2 s  (minimum map-update horizon)

        Returns
        -------
        spline : BSpline object callable as spline(t) for t in [0, T],
                 returning (x, y) position at time t.
        T      : float, total trajectory duration.
        """
        p0 = robot.position.copy()
        p3 = q_star.position.copy()
        delta = p3 - p0
        p1 = p0 + delta * (1.0 / 3.0)
        p2 = p0 + delta * (2.0 / 3.0)

        waypoints = np.array([p0, p1, p2, p3])   # (4, 2)

        # Segment lengths for duration estimate
        seg_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        total_length = float(seg_lengths.sum())

        T_v = P_SAFE * total_length / max(self.v_max, 1e-9)
        T   = max(T_v, T_MAP)

        # Uniform parameter knots for the 4 control points
        t_knots = np.linspace(0.0, T, len(waypoints))

        # Cubic B-spline (k=3) — need at least 4 points, which we have exactly
        spline = make_interp_spline(t_knots, waypoints, k=3)
        return spline, T

    # ── select_maneuver ───────────────────────────────────────────────────────

    def select_maneuver(
        self,
        waypoints:       np.ndarray,    # (N, 2) planned path waypoints
        collision_point: Optional[np.ndarray],   # (2,) or None if no collision
        c_O:             float,         # semantic cost of colliding obstacle
        f_s:             float,         # anisotropic safety score at collision ∈ [0,1]
        battery:         float,         # current normalised battery ∈ [0, 1]
    ) -> str:
        """Algorithm 2: select a reactive maneuver at a potential collision.

        Decision tree
        -------------
        1. No collision predicted         → 'FREESPACE'
        2. c(O) > c_max (forbidden/heavy) → 'ROTATE'  (fallback: 'REPLAN')
        3. Otherwise, compute v (path direction at collision) and
           v_perp (left-perpendicular to v):
             dot(v_perp, v_collision) >= 0 AND f_s > f_min → 'PUSH'
             dot(v_perp, v_collision) <  0                 → 'BOUNDARYFOLLOW'
             else                                          → 'FLOWTHROUGH'

        Parameters
        ----------
        waypoints        : planned path as (N, 2) array; used to derive path
                           direction at the collision point.
        collision_point  : predicted (x, y) of first contact, or None.
        c_O              : semantic cost of the obstacle at the collision point.
        f_s              : directional safety score from the anisotropic map.
        battery          : normalised battery level.

        Returns
        -------
        maneuver : one of 'FREESPACE', 'ROTATE', 'REPLAN', 'PUSH',
                   'BOUNDARYFOLLOW', 'FLOWTHROUGH'
        """
        # Step 1 — no collision
        if collision_point is None:
            return "FREESPACE"

        # Step 2 — obstacle too costly to interact with
        if c_O > C_MAX:
            # REPLAN if battery is high enough to afford a detour,
            # otherwise ROTATE in place as a last-resort fallback.
            return "REPLAN" if battery > 0.2 else "ROTATE"

        # Step 3 — derive path direction v at the collision point
        v = self._path_direction_at(waypoints, collision_point)  # unit vector

        # Perpendicular: 90° CCW rotation of v
        v_perp = np.array([-v[1], v[0]])

        # Vector from path to collision point
        nearest_wp = self._nearest_waypoint(waypoints, collision_point)
        v_collision = normalize(collision_point - nearest_wp)

        dot = float(np.dot(v_perp, v_collision))

        if dot >= 0.0 and f_s > F_MIN:
            return "PUSH"
        elif dot < 0.0:
            return "BOUNDARYFOLLOW"
        else:
            return "FLOWTHROUGH"

    # ── maneuver helpers ──────────────────────────────────────────────────────

    def _path_direction_at(
        self, waypoints: np.ndarray, query: np.ndarray
    ) -> np.ndarray:
        """Return the unit tangent of the path at the waypoint nearest to query."""
        if len(waypoints) < 2:
            return np.array([1.0, 0.0])

        # Find the segment closest to query
        best_seg = 0
        best_dist = float("inf")
        for i in range(len(waypoints) - 1):
            seg_mid = (waypoints[i] + waypoints[i + 1]) / 2.0
            d = float(np.linalg.norm(query - seg_mid))
            if d < best_dist:
                best_dist = d
                best_seg = i

        tangent = waypoints[best_seg + 1] - waypoints[best_seg]
        n = float(np.linalg.norm(tangent))
        return tangent / n if n > 1e-9 else np.array([1.0, 0.0])

    def _nearest_waypoint(
        self, waypoints: np.ndarray, query: np.ndarray
    ) -> np.ndarray:
        """Return the waypoint in the array closest to query."""
        dists = np.linalg.norm(waypoints - query, axis=1)
        return waypoints[int(np.argmin(dists))]


def _astar(
    cost_map: CostMap,
    start_xy: np.ndarray,
    goal_xy:  np.ndarray,
) -> Optional[List[np.ndarray]]:
    start_cell = cost_map.world_to_cell(start_xy)
    goal_cell  = cost_map.world_to_cell(goal_xy)
    H, W = cost_map.height, cost_map.width

    def h(r: int, c: int) -> float:
        gr, gc = goal_cell
        return cost_map.resolution * ((r - gr) ** 2 + (c - gc) ** 2) ** 0.5

    heap: list = []
    heapq.heappush(heap, (h(*start_cell), 0.0, start_cell))
    came_from: Dict[Tuple, Optional[Tuple]] = {start_cell: None}
    g: Dict[Tuple, float] = {start_cell: 0.0}
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while heap:
        _, g_cur, cur = heapq.heappop(heap)
        if cur == goal_cell:
            path = []
            node = cur
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return [cost_map.cell_to_world(r, c) for r, c in path]
        r, c = cur
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            step_cost = cost_map.grid[nr, nc] * cost_map.resolution * (1.414 if dr and dc else 1.0)
            tg = g_cur + step_cost
            if tg < g.get((nr, nc), float("inf")):
                g[(nr, nc)] = tg
                came_from[(nr, nc)] = cur
                heapq.heappush(heap, (tg + h(nr, nc), tg, (nr, nc)))
    return None

def run_scholar(
    scene:              dict,
    sensing_range:      float = DEFAULT_SENSING_RANGE,
    step_size:          float = V_MAX,
    max_steps:          int   = 800,
    cost_map_resolution: float = 0.05,
    battery_budget:     float = 1.0,
) -> OnlineSurpResult:
    working_scene = normalize_scene_for_online_use(scene)
    position = np.array(working_scene["start"][:2], dtype=float)
    goal     = np.array(working_scene["goal"][:2],  dtype=float)

    robot = RobotState(
        position=position.copy(),
        velocity=np.zeros(2),
        battery=battery_budget,
        battery_max=battery_budget,
    )

    planner      = Scholar(v_max=step_size, sensing_range=sensing_range)
    path_taken   = [tuple(position)]
    frames       = [snapshot_frame(position, working_scene, "start")]
    sensed_ids:  List[int] = []
    success      = False

    # Initial sense
    newly = reveal_nearby_obstacles(working_scene, position, sensing_range)
    if newly:
        sensed_ids.extend(o["id"] for o in newly)

    # Initial cost map
    cost_map = build_voxel_grid(working_scene, resolution=cost_map_resolution)
    compute_anisotropic_map(cost_map, working_scene)
    frames.append(snapshot_frame(position, working_scene, "initial cost map built"))

    # Battery drain rate: fully drained after traversing the workspace diagonal
    xmin, xmax, ymin, ymax = working_scene["workspace"]
    _diag = float(np.linalg.norm([xmax - xmin, ymax - ymin]))
    battery_drain_per_metre = battery_budget / max(_diag, 1.0)

    for _ in range(max_steps):
        if float(np.linalg.norm(goal - position)) <= step_size:
            position = goal.copy()
            path_taken.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "goal reached"))
            success = True
            break

        # Sense new obstacles
        newly = reveal_nearby_obstacles(working_scene, position, sensing_range)
        if newly:
            ids = [o["id"] for o in newly]
            sensed_ids.extend(ids)
            # Slide and recompute cost map around robot
            update_cost_map(cost_map, working_scene, position)
            compute_anisotropic_map(cost_map, working_scene)
            frames.append(snapshot_frame(
                position, working_scene, f"sensed {ids} — cost map updated"
            ))

        # SCHOLAR tick
        robot.position = position.copy()
        q_star = planner.select_next_state(robot, working_scene, cost_map, goal)

        # Move to selected position (clamped to one step)
        delta = q_star.position - position
        dist  = float(np.linalg.norm(delta))
        move  = min(step_size, dist) if dist > 1e-9 else 0.0
        if dist > 1e-9:
            position = clip_point_to_workspace(
                working_scene, position + (delta / dist) * move
            )
        robot.velocity = q_star.velocity.copy()

        # Battery drain
        robot.battery = max(0.0, robot.battery - move * battery_drain_per_metre)

        path_taken.append(tuple(position))
        frames.append(snapshot_frame(
            position, working_scene,
            f"SCHOLAR → {tuple(np.round(q_star.position, 2))}  "
            f"B={robot.battery:.2f}"
        ))

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
