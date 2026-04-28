import heapq
from collections import defaultdict, deque

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


INF = float("inf")


class _PriorityQueue:
    """Lazy-deletion queue used by the standalone D* Lite planner."""

    def __init__(self) -> None:
        self.heap: list[tuple[tuple[float, float], tuple[int, int]]] = []
        self.active: dict[tuple[int, int], tuple[float, float]] = {}

    def push(self, state: tuple[int, int], key: tuple[float, float]) -> None:
        self.active[state] = key
        heapq.heappush(self.heap, (key, state))

    def discard(self, state: tuple[int, int]) -> None:
        self.active.pop(state, None)

    def top_key(self) -> tuple[float, float]:
        while self.heap:
            key, state = self.heap[0]
            if self.active.get(state) == key:
                return key
            heapq.heappop(self.heap)
        return (INF, INF)

    def pop(self) -> tuple[tuple[float, float], tuple[int, int] | None]:
        while self.heap:
            key, state = heapq.heappop(self.heap)
            if self.active.get(state) == key:
                del self.active[state]
                return key, state
        return (INF, INF), None


class _ObservedGrid:
    """Occupancy raster for pure D* Lite replanning on the observed map."""

    def __init__(self, scene: dict, resolution: float) -> None:
        self.scene = scene
        self.resolution = resolution
        self.xmin, self.xmax, self.ymin, self.ymax = scene["workspace"]
        self.width = max(1, int(np.ceil((self.xmax - self.xmin) / resolution)) + 1)
        self.height = max(1, int(np.ceil((self.ymax - self.ymin) / resolution)) + 1)

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def world_to_grid(self, point: np.ndarray) -> tuple[int, int]:
        clipped = clip_point_to_workspace(self.scene, point)
        gx = int(round((float(clipped[0]) - self.xmin) / self.resolution))
        gy = int(round((float(clipped[1]) - self.ymin) / self.resolution))
        return (min(max(gx, 0), self.width - 1), min(max(gy, 0), self.height - 1))

    def grid_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        return np.array(
            [
                float(self.xmin + cell[0] * self.resolution),
                float(self.ymin + cell[1] * self.resolution),
            ],
            dtype=float,
        )

    def neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = cell
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [nbr for nbr in candidates if self.in_bounds(nbr)]

    def heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return self.resolution * (abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def cell_blocked(self, cell: tuple[int, int]) -> bool:
        world = self.grid_to_world(cell)
        body = robot_body(world)
        return any(
            obstacle.get("observed", False) and body.intersects(obstacle_polygon(obstacle))
            for obstacle in self.scene["obstacles"]
        )

    def nearest_free(self, start: tuple[int, int]) -> tuple[int, int] | None:
        if not self.cell_blocked(start):
            return start

        queue: deque[tuple[int, int]] = deque([start])
        seen = {start}
        while queue:
            state = queue.popleft()
            for nbr in self.neighbors(state):
                if nbr in seen:
                    continue
                if not self.cell_blocked(nbr):
                    return nbr
                seen.add(nbr)
                queue.append(nbr)
        return None

    def observed_blocked_cells(self) -> set[tuple[int, int]]:
        blocked: set[tuple[int, int]] = set()
        padding = self.resolution + ROBOT_RADIUS

        for obstacle in self.scene["obstacles"]:
            if not obstacle.get("observed", False):
                continue

            poly = obstacle_polygon(obstacle)
            minx, miny, maxx, maxy = poly.bounds
            gx0 = max(0, int(np.floor((minx - padding - self.xmin) / self.resolution)))
            gx1 = min(self.width - 1, int(np.ceil((maxx + padding - self.xmin) / self.resolution)))
            gy0 = max(0, int(np.floor((miny - padding - self.ymin) / self.resolution)))
            gy1 = min(self.height - 1, int(np.ceil((maxy + padding - self.ymin) / self.resolution)))

            for gy in range(gy0, gy1 + 1):
                for gx in range(gx0, gx1 + 1):
                    if self.cell_blocked((gx, gy)):
                        blocked.add((gx, gy))

        return blocked


class _DStarLitePlanner:
    """Pure D* Lite graph search over the observed occupancy raster."""

    def __init__(
        self,
        grid: _ObservedGrid,
        start: tuple[int, int],
        goal: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> None:
        self.grid = grid
        self.s_start = start
        self.s_goal = goal
        self.last = start
        self.km = 0.0
        self.blocked = set(blocked)

        self.g: defaultdict[tuple[int, int], float] = defaultdict(lambda: INF)
        self.rhs: defaultdict[tuple[int, int], float] = defaultdict(lambda: INF)
        self.queue = _PriorityQueue()

        self.rhs[self.s_goal] = 0.0
        self.queue.push(self.s_goal, self.calculate_key(self.s_goal))

    def heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return self.grid.heuristic(a, b)

    def calculate_key(self, state: tuple[int, int]) -> tuple[float, float]:
        value = min(self.g[state], self.rhs[state])
        return (value + self.heuristic(self.s_start, state) + self.km, value)

    def cost(self, _a: tuple[int, int], b: tuple[int, int]) -> float:
        return INF if b in self.blocked else self.grid.resolution

    def predecessors(self, state: tuple[int, int]) -> list[tuple[int, int]]:
        return self.grid.neighbors(state)

    def successors(self, state: tuple[int, int]) -> list[tuple[int, int]]:
        return self.grid.neighbors(state)

    def update_vertex(self, state: tuple[int, int]) -> None:
        if state != self.s_goal:
            self.rhs[state] = min(
                (self.cost(state, succ) + self.g[succ] for succ in self.successors(state)),
                default=INF,
            )

        self.queue.discard(state)
        if self.g[state] != self.rhs[state]:
            self.queue.push(state, self.calculate_key(state))

    def compute_shortest_path(self) -> None:
        while (
            self.queue.top_key() < self.calculate_key(self.s_start)
            or self.rhs[self.s_start] != self.g[self.s_start]
        ):
            old_key, state = self.queue.pop()
            if state is None:
                break

            new_key = self.calculate_key(state)
            if old_key < new_key:
                self.queue.push(state, new_key)
            elif self.g[state] > self.rhs[state]:
                self.g[state] = self.rhs[state]
                for pred in self.predecessors(state):
                    self.update_vertex(pred)
            else:
                self.g[state] = INF
                self.update_vertex(state)
                for pred in self.predecessors(state):
                    self.update_vertex(pred)

    def best_successor(self, state: tuple[int, int]) -> tuple[int, int] | None:
        best = None
        best_cost = INF
        for succ in self.successors(state):
            value = self.cost(state, succ) + self.g[succ]
            if value < best_cost:
                best_cost = value
                best = succ
        return best

    def move_start(self, new_start: tuple[int, int]) -> None:
        self.last = self.s_start
        self.s_start = new_start
        self.km += self.heuristic(self.last, self.s_start)

    def update_obstacle(self, cell: tuple[int, int], blocked: bool) -> None:
        changed = False
        if blocked and cell not in self.blocked:
            self.blocked.add(cell)
            changed = True
        elif not blocked and cell in self.blocked:
            self.blocked.remove(cell)
            changed = True

        if not changed:
            return

        self.update_vertex(cell)
        for pred in self.predecessors(cell):
            self.update_vertex(pred)


def _step_toward(current: np.ndarray, target: np.ndarray, step_size: float) -> np.ndarray:
    delta = target - current
    distance = float(np.linalg.norm(delta))
    if distance <= step_size:
        return target.copy()
    return current + normalize(delta) * step_size


def run_dstar_lite(
    scene: dict,
    sensing_range: float = DEFAULT_SENSING_RANGE,
    step_size: float = 0.07,
    max_steps: int = 500,
    grid_resolution: float | None = None,
) -> OnlineSurpResult:
    """Pure D* Lite planner over an observed occupancy grid.

    Unknown space is treated as free. As obstacles are sensed, their occupied
    grid cells are inserted into the planner and the path is repaired
    incrementally rather than rebuilt from scratch.
    """
    working_scene = normalize_scene_for_online_use(scene)
    initial_scene = normalize_scene_for_online_use(scene)
    position = np.array(working_scene["start"][:2], dtype=float)
    goal = np.array(working_scene["goal"][:2], dtype=float)

    resolution = grid_resolution or max(0.08, step_size)
    grid = _ObservedGrid(working_scene, resolution=resolution)
    blocked = grid.observed_blocked_cells()

    start_cell = grid.nearest_free(grid.world_to_grid(position))
    goal_cell = grid.nearest_free(grid.world_to_grid(goal))

    path = [tuple(position)]
    frames = [snapshot_frame(position, working_scene, "start")]
    sensed_ids: list[int] = []
    contact_log: list[str] = []
    success = False

    if start_cell is None or goal_cell is None:
        frames.append(snapshot_frame(position, working_scene, "no free start/goal cell"))
        return OnlineSurpResult(
            family=scene.get("family", "unknown"),
            seed=scene.get("seed", 0),
            success=False,
            path=path,
            frames=frames,
            scene=working_scene,
            initial_scene=initial_scene,
            contact_log=contact_log,
            sensed_ids=sensed_ids,
        )

    planner = _DStarLitePlanner(grid, start_cell, goal_cell, blocked)

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

            updated_blocked = grid.observed_blocked_cells()
            changed_cells = planner.blocked.symmetric_difference(updated_blocked)
            for cell in changed_cells:
                planner.update_obstacle(cell, blocked=(cell in updated_blocked))

            start_guess = grid.world_to_grid(position)
            repaired_start = grid.nearest_free(start_guess)
            if repaired_start is not None:
                planner.move_start(repaired_start)

            path.append(tuple(position))
            frames.append(
                snapshot_frame(
                    position,
                    working_scene,
                    f"sensed obstacle(s) {observed_ids}; repairing D* Lite graph",
                )
            )
            continue

        start_guess = grid.world_to_grid(position)
        start_cell = grid.nearest_free(start_guess)
        if start_cell is None:
            frames.append(snapshot_frame(position, working_scene, "robot trapped in observed map"))
            break

        if start_cell != planner.s_start:
            planner.move_start(start_cell)

        planner.compute_shortest_path()
        if planner.g[planner.s_start] == INF:
            path.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "no known path to goal"))
            break

        next_cell = planner.best_successor(planner.s_start)
        if next_cell is None:
            path.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "no successor from current cell"))
            break

        waypoint = grid.grid_to_world(next_cell)
        target = goal.copy() if np.linalg.norm(goal - position) <= max(step_size, resolution) else waypoint
        next_position = clip_point_to_workspace(working_scene, _step_toward(position, target, step_size))

        if np.linalg.norm(next_position - position) <= 1e-9:
            path.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "stalled on local D* Lite step"))
            break

        position = next_position
        planner.move_start(grid.world_to_grid(position))
        path.append(tuple(position))
        frames.append(snapshot_frame(position, working_scene, "dstar lite step"))

    return OnlineSurpResult(
        family=scene.get("family", "unknown"),
        seed=scene.get("seed", 0),
        success=success,
        path=path,
        frames=frames,
        scene=working_scene,
        initial_scene=initial_scene,
        contact_log=contact_log,
        sensed_ids=sensed_ids,
    )
