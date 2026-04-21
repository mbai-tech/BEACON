"""Bug1 and D* Lite baseline planners for the SCHOLAR evaluation."""

from __future__ import annotations

import heapq
import math

import numpy as np

from scholar.core.constants import DEFAULT_SENSING_RANGE, ROBOT_RADIUS
from scholar.core.models import OnlineSurpResult
from scholar.algorithms.surp import (
    clip_point_to_workspace,
    normalize,
    obstacle_polygon,
    reveal_nearby_obstacles,
    robot_body,
    snapshot_frame,
)
from scholar.utils.scene_setup import normalize_scene_for_online_use


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

        else:
            d = float(np.linalg.norm(goal - position))
            if d < closest_dist:
                closest_dist = d
                closest_pos  = position.copy()

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


# ── D* Lite ───────────────────────────────────────────────────────────────────

_INF = float("inf")


class _DStarLite:
    """Backward incremental planner on a 2D occupancy grid.

    Searches from the goal toward the start (backward search), so g[s] is the
    optimal cost from s to the goal.  When obstacles change, only the cells
    whose shortest-path estimates are affected are re-expanded — O(changed
    cells) rather than O(grid) per replan.

    Reference: Koenig & Likhachev, "D* Lite", AAAI 2002.
    """

    def __init__(self, workspace: list[float], resolution: float) -> None:
        xmin, xmax, ymin, ymax = workspace
        self.xmin    = xmin
        self.ymin    = ymin
        self.res     = resolution
        self.nx      = int(round((xmax - xmin) / resolution)) + 1
        self.ny      = int(round((ymax - ymin) / resolution)) + 1
        self.blocked: set[tuple[int, int]] = set()
        self.g:   dict[tuple[int, int], float] = {}
        self.rhs: dict[tuple[int, int], float] = {}
        self._heap: list                        = []
        self._in_q: dict[tuple[int, int], tuple[float, float]] = {}
        self.km:      float              = 0.0
        self.s_start: tuple[int, int]   = (0, 0)
        self.s_goal:  tuple[int, int]   = (0, 0)

    # ── initialisation ────────────────────────────────────────────────────

    def initialize(self, s_start: tuple[int, int], s_goal: tuple[int, int]) -> None:
        self.s_start = s_start
        self.s_goal  = s_goal
        self.km      = 0.0
        self.g       = {}
        self.rhs     = {s_goal: 0.0}
        self._heap   = []
        self._in_q   = {}
        self._insert(s_goal, self._key(s_goal))

    # ── priority queue (lazy-deletion heap) ───────────────────────────────

    def _h(self, s: tuple[int, int]) -> float:
        return math.hypot(s[0] - self.s_start[0], s[1] - self.s_start[1]) * self.res

    def _key(self, s: tuple[int, int]) -> tuple[float, float]:
        m = min(self.g.get(s, _INF), self.rhs.get(s, _INF))
        return (m + self._h(s) + self.km, m)

    def _insert(self, s: tuple[int, int], key: tuple[float, float]) -> None:
        heapq.heappush(self._heap, (key, s))
        self._in_q[s] = key

    def _remove(self, s: tuple[int, int]) -> None:
        self._in_q.pop(s, None)

    def _top_key(self) -> tuple[float, float]:
        while self._heap:
            key, s = self._heap[0]
            if self._in_q.get(s) == key:
                return key
            heapq.heappop(self._heap)
        return (_INF, _INF)

    def _pop(self) -> tuple[tuple[float, float] | None, tuple[int, int] | None]:
        while self._heap:
            key, s = heapq.heappop(self._heap)
            if self._in_q.get(s) == key:
                del self._in_q[s]
                return key, s
        return None, None

    # ── graph ─────────────────────────────────────────────────────────────

    def _neighbours(self, s: tuple[int, int], passthrough: bool = False) -> list[tuple[int, int]]:
        ix, iy = s
        out = []
        for dix in (-1, 0, 1):
            for diy in (-1, 0, 1):
                if dix == 0 and diy == 0:
                    continue
                nb = (ix + dix, iy + diy)
                if not (0 <= nb[0] < self.nx and 0 <= nb[1] < self.ny):
                    continue
                if not passthrough and nb in self.blocked:
                    continue
                out.append(nb)
        return out

    def _cost(self, s: tuple[int, int], t: tuple[int, int]) -> float:
        if s in self.blocked or t in self.blocked:
            return _INF
        diagonal = (s[0] != t[0]) and (s[1] != t[1])
        return self.res * (1.4142 if diagonal else 1.0)

    # ── core D* Lite ──────────────────────────────────────────────────────

    def _update_vertex(self, s: tuple[int, int]) -> None:
        if s != self.s_goal:
            succs = self._neighbours(s)
            self.rhs[s] = (
                min(self._cost(s, sp) + self.g.get(sp, _INF) for sp in succs)
                if succs else _INF
            )
        self._remove(s)
        if self.g.get(s, _INF) != self.rhs.get(s, _INF):
            self._insert(s, self._key(s))

    def compute_shortest_path(self, budget: int = 200_000) -> None:
        for _ in range(budget):
            k_top = self._top_key()
            k_s   = self._key(self.s_start)
            if k_top >= k_s and self.rhs.get(self.s_start, _INF) == self.g.get(self.s_start, _INF):
                break
            k_old, u = self._pop()
            if u is None:
                break
            if k_old < self._key(u):
                self._insert(u, self._key(u))
            elif self.g.get(u, _INF) > self.rhs.get(u, _INF):
                self.g[u] = self.rhs[u]
                for nb in self._neighbours(u, passthrough=True):
                    self._update_vertex(nb)
            else:
                self.g[u] = _INF
                for nb in self._neighbours(u, passthrough=True) + [u]:
                    self._update_vertex(nb)

    def mark_blocked(self, cells: set[tuple[int, int]]) -> None:
        """Register newly observed blocked cells and propagate cost changes."""
        newly = cells - self.blocked
        self.blocked |= newly
        for c in newly:
            self.g[c]   = _INF
            self.rhs[c] = _INF
            self._remove(c)
            for nb in self._neighbours(c, passthrough=True):
                self._update_vertex(nb)

    # ── navigation ────────────────────────────────────────────────────────

    def best_next(self, s: tuple[int, int]) -> tuple[int, int] | None:
        succs = self._neighbours(s)
        if not succs:
            return None
        best = min(succs, key=lambda sp: self._cost(s, sp) + self.g.get(sp, _INF))
        return None if self._cost(s, best) + self.g.get(best, _INF) >= _INF else best

    # ── coordinate conversion ─────────────────────────────────────────────

    def to_grid(self, x: float, y: float) -> tuple[int, int]:
        ix = max(0, min(self.nx - 1, int(round((x - self.xmin) / self.res))))
        iy = max(0, min(self.ny - 1, int(round((y - self.ymin) / self.res))))
        return (ix, iy)

    def to_pos(self, ix: int, iy: int) -> tuple[float, float]:
        return (self.xmin + ix * self.res, self.ymin + iy * self.res)


def _cells_for_obstacles(dsl: _DStarLite, obstacles: list[dict]) -> set[tuple[int, int]]:
    """Return every grid cell whose robot footprint overlaps any obstacle polygon."""
    blocked: set[tuple[int, int]] = set()
    for obs in obstacles:
        poly = obstacle_polygon(obs)
        minx, miny, maxx, maxy = poly.bounds
        pad  = ROBOT_RADIUS + dsl.res
        ix_lo = max(0, int((minx - pad - dsl.xmin) / dsl.res))
        ix_hi = min(dsl.nx - 1, int((maxx + pad - dsl.xmin) / dsl.res) + 1)
        iy_lo = max(0, int((miny - pad - dsl.ymin) / dsl.res))
        iy_hi = min(dsl.ny - 1, int((maxy + pad - dsl.ymin) / dsl.res) + 1)
        for ix in range(ix_lo, ix_hi + 1):
            for iy in range(iy_lo, iy_hi + 1):
                if robot_body(np.array(dsl.to_pos(ix, iy))).intersects(poly):
                    blocked.add((ix, iy))
    return blocked


def run_dstar(
    scene: dict,
    sensing_range: float = DEFAULT_SENSING_RANGE,
    step_size: float = 0.10,
    max_steps: int = 1000,
    grid_resolution: float = 0.10,
) -> OnlineSurpResult:
    """D* Lite: incremental optimal planner for partially-known environments.

    Maintains a backward shortest-path tree from the goal over the 2D occupancy
    grid.  When new obstacles are sensed, only the affected nodes are
    re-expanded — making each replan proportional to the size of the change
    rather than the full grid.

    Parameters
    ----------
    scene            : standard scene dict
    sensing_range    : radius within which obstacles are revealed each step
    step_size        : continuous movement distance per execution step
    max_steps        : execution step budget
    grid_resolution  : occupancy-grid cell size in metres (default 0.10 m)
    """
    working_scene = normalize_scene_for_online_use(scene)
    position = np.array(working_scene["start"][:2], dtype=float)
    goal     = np.array(working_scene["goal"][:2],  dtype=float)

    path_taken  = [tuple(position)]
    frames      = [snapshot_frame(position, working_scene, "start")]
    sensed_ids: list[int] = []

    dsl    = _DStarLite(working_scene["workspace"], grid_resolution)
    s_goal = dsl.to_grid(*goal)
    s_curr = dsl.to_grid(*position)
    s_last = s_curr
    dsl.initialize(s_curr, s_goal)

    # Initial sensing pass
    newly = reveal_nearby_obstacles(working_scene, position, sensing_range)
    if newly:
        sensed_ids.extend(obs["id"] for obs in newly)
        dsl.mark_blocked(_cells_for_obstacles(dsl, newly))
        frames.append(snapshot_frame(position, working_scene,
                                     f"initial sense: {len(newly)} obstacle(s)"))

    dsl.compute_shortest_path()
    success = False

    for step in range(max_steps):
        if float(np.linalg.norm(goal - position)) <= step_size:
            position = goal.copy()
            path_taken.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "goal reached"))
            success = True
            break

        # ── sense new obstacles ───────────────────────────────────────────
        newly = reveal_nearby_obstacles(working_scene, position, sensing_range)
        if newly:
            observed_ids = [obs["id"] for obs in newly]
            sensed_ids.extend(observed_ids)

            s_curr = dsl.to_grid(*position)
            dsl.km += math.hypot(s_last[0] - s_curr[0],
                                 s_last[1] - s_curr[1]) * grid_resolution
            s_last       = s_curr
            dsl.s_start  = s_curr

            dsl.mark_blocked(_cells_for_obstacles(dsl, newly))
            dsl.compute_shortest_path()

            path_taken.append(tuple(position))
            frames.append(snapshot_frame(
                position, working_scene,
                f"sensed {observed_ids} — incremental replan (step {step})",
            ))
            continue

        # ── move toward best next grid cell ───────────────────────────────
        s_curr = dsl.to_grid(*position)
        s_next = dsl.best_next(s_curr)

        if s_next is None:
            frames.append(snapshot_frame(position, working_scene,
                                         "D* Lite: no path to goal"))
            break

        target    = np.array(dsl.to_pos(*s_next))
        direction = target - position
        dist      = float(np.linalg.norm(direction))
        if dist > 1e-9:
            position = clip_point_to_workspace(
                working_scene,
                position + normalize(direction) * min(step_size, dist),
            )

        path_taken.append(tuple(position))
        frames.append(snapshot_frame(
            position, working_scene,
            f"D* Lite step {step} — g={dsl.g.get(s_next, _INF):.3f}",
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
