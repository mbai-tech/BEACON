"""
scholar.py — Main SCHOLAR planner.

SCHOLAR (Semantic Cost-aware Holonomic Online Learning and Reasoning) is an
online planner that:
  1. Maintains a semantic belief over each observed obstacle.
  2. Builds a cost map from current beliefs.
  3. Plans a least-cost path via A* on the cost map.
  4. Executes one step, senses new obstacles, updates beliefs, and replans.
"""

import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from planning.cost_map import CostMap, build_cost_map, SEMANTIC_COSTS
from planning.semantic_cost import bayesian_update, uniform_prior, expected_cost


@dataclass
class PlannerState:
    position: np.ndarray
    goal: np.ndarray
    beliefs: Dict[int, Dict[str, float]] = field(default_factory=dict)  # obs_id → belief
    observed_ids: List[int] = field(default_factory=list)
    path: List[np.ndarray] = field(default_factory=list)
    cost_map: Optional[CostMap] = None


# ── A* on cost map ─────────────────────────────────────────────────────────────

def _astar(cost_map: CostMap, start_xy: np.ndarray, goal_xy: np.ndarray) -> Optional[List[np.ndarray]]:
    """A* search on the cost map grid. Returns list of world-space waypoints or None."""
    start_cell = cost_map.world_to_cell(start_xy)
    goal_cell = cost_map.world_to_cell(goal_xy)

    H, W = cost_map.height, cost_map.width

    def heuristic(r: int, c: int) -> float:
        gr, gc = goal_cell
        return cost_map.resolution * ((r - gr)**2 + (c - gc)**2) ** 0.5

    open_heap: list = []
    heapq.heappush(open_heap, (heuristic(*start_cell), 0.0, start_cell))
    came_from: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {start_cell: None}
    g_score: Dict[Tuple[int,int], float] = {start_cell: 0.0}

    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current == goal_cell:
            # Reconstruct path
            path_cells = []
            node = current
            while node is not None:
                path_cells.append(node)
                node = came_from[node]
            path_cells.reverse()
            return [cost_map.cell_to_world(r, c) for r, c in path_cells]

        r, c = current
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            step_cost = cost_map.grid[nr, nc] * cost_map.resolution * (1.414 if dr and dc else 1.0)
            tentative_g = g + step_cost
            if tentative_g < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = tentative_g
                came_from[(nr, nc)] = current
                f = tentative_g + heuristic(nr, nc)
                heapq.heappush(open_heap, (f, tentative_g, (nr, nc)))

    return None


# ── Sensing model ───────────────────────────────────────────────────────────────

def sense_obstacles(
    obstacles: list,
    position: np.ndarray,
    sensing_range: float,
) -> List[int]:
    """Return ids of obstacles within sensing_range of position."""
    newly_observed = []
    for obs in obstacles:
        dist = float(np.linalg.norm(obs.position[:2] - position[:2]))
        if dist <= sensing_range and obs.body_id not in []:
            newly_observed.append(obs.body_id)
    return newly_observed


# ── Main planner ────────────────────────────────────────────────────────────────

class ScholarPlanner:
    """Online semantic cost-aware planner with belief updates."""

    def __init__(
        self,
        scene,
        sensing_range: float = 1.5,
        step_size: float = 0.15,
        workspace_bounds: tuple = (0.0, 10.0, 0.0, 10.0),
        cost_map_resolution: float = 0.05,
    ):
        self.scene = scene
        self.sensing_range = sensing_range
        self.step_size = step_size
        self.workspace_bounds = workspace_bounds
        self.resolution = cost_map_resolution

        # Initialise beliefs (all obstacles start with uniform prior over classes)
        self.beliefs: Dict[int, Dict[str, float]] = {
            obs.body_id: uniform_prior() for obs in scene.obstacles
        }
        self.observed: Dict[int, bool] = {obs.body_id: False for obs in scene.obstacles}

        self.state = PlannerState(
            position=scene.start[:2].copy(),
            goal=scene.goal[:2].copy(),
        )
        self._replan()

    # ------------------------------------------------------------------

    def _replan(self) -> None:
        observed_obstacles = [
            obs for obs in self.scene.obstacles if self.observed.get(obs.body_id, False)
        ]
        cost_map = build_cost_map(
            observed_obstacles,
            workspace_bounds=self.workspace_bounds,
            resolution=self.resolution,
        )
        self.state.cost_map = cost_map
        path = _astar(cost_map, self.state.position, self.state.goal)
        self.state.path = path if path is not None else []

    def _sense(self) -> List[int]:
        newly_seen = []
        for obs in self.scene.obstacles:
            if self.observed.get(obs.body_id, False):
                continue
            dist = float(np.linalg.norm(obs.position[:2] - self.state.position))
            if dist <= self.sensing_range:
                self.observed[obs.body_id] = True
                newly_seen.append(obs.body_id)
        return newly_seen

    def step(self) -> Tuple[np.ndarray, bool]:
        """Execute one planning step. Returns (new_position, reached_goal)."""
        newly_seen = self._sense()
        if newly_seen:
            self._replan()

        if not self.state.path:
            return self.state.position.copy(), False

        # Find next waypoint along path
        next_wp = None
        for wp in self.state.path:
            if float(np.linalg.norm(wp - self.state.position)) > self.step_size * 0.5:
                next_wp = wp
                break

        if next_wp is None:
            next_wp = self.state.goal

        direction = next_wp - self.state.position
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            return self.state.position.copy(), False

        move = min(self.step_size, dist)
        self.state.position = self.state.position + (direction / dist) * move

        reached = float(np.linalg.norm(self.state.position - self.state.goal)) <= self.step_size
        return self.state.position.copy(), reached
