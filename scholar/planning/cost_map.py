"""
cost_map.py — Semantic anisotropic cost map.

The cost map associates each grid cell with a traversal cost that depends on
the semantic class of nearby obstacles.  Cells that are free space carry cost
1.0; cells occupied or adjacent to obstacles carry higher costs according to
the semantic class hierarchy.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

# Per-class base traversal cost
SEMANTIC_COSTS: Dict[str, float] = {
    "safe":      1.0,
    "movable":   3.0,
    "fragile":  15.0,
    "forbidden": 1000.0,
    "unknown":   1.0,   # optimistic; treat unknown as free
}


@dataclass
class CostMap:
    """2-D grid cost map over the workspace."""
    grid: np.ndarray          # shape (H, W) — traversal costs
    resolution: float         # metres per cell
    origin: np.ndarray        # (x, y) world coords of cell (0, 0)

    # ------------------------------------------------------------------
    @property
    def height(self) -> int:
        return self.grid.shape[0]

    @property
    def width(self) -> int:
        return self.grid.shape[1]

    def world_to_cell(self, xy: np.ndarray) -> tuple[int, int]:
        col = int((xy[0] - self.origin[0]) / self.resolution)
        row = int((xy[1] - self.origin[1]) / self.resolution)
        return row, col

    def cell_to_world(self, row: int, col: int) -> np.ndarray:
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return np.array([x, y])

    def cost_at_world(self, xy: np.ndarray) -> float:
        r, c = self.world_to_cell(xy)
        if 0 <= r < self.height and 0 <= c < self.width:
            return float(self.grid[r, c])
        return float("inf")


def build_cost_map(
    obstacles: list,
    workspace_bounds: tuple = (0.0, 10.0, 0.0, 10.0),
    resolution: float = 0.05,
    inflation_radius: float = 0.2,
) -> CostMap:
    """Build a cost map from a list of Obstacle objects.

    Each obstacle cell is assigned its semantic cost.  Cells within
    `inflation_radius` of an obstacle boundary carry the same cost (to
    keep the robot body away from obstacle edges).
    """
    xmin, xmax, ymin, ymax = workspace_bounds
    cols = int((xmax - xmin) / resolution) + 1
    rows = int((ymax - ymin) / resolution) + 1

    grid = np.ones((rows, cols), dtype=float)
    origin = np.array([xmin, ymin])
    cost_map = CostMap(grid=grid, resolution=resolution, origin=origin)

    inflation_cells = int(inflation_radius / resolution)

    for obs in obstacles:
        base_cost = SEMANTIC_COSTS.get(obs.semantic_class, 1.0)
        cx = obs.position[0]
        cy = obs.position[1]
        dx = obs.half_extents[0] + inflation_radius
        dy = obs.half_extents[1] + inflation_radius

        r_lo, c_lo = cost_map.world_to_cell(np.array([cx - dx, cy - dy]))
        r_hi, c_hi = cost_map.world_to_cell(np.array([cx + dx, cy + dy]))

        r_lo = max(0, r_lo)
        r_hi = min(rows - 1, r_hi)
        c_lo = max(0, c_lo)
        c_hi = min(cols - 1, c_hi)

        grid[r_lo:r_hi+1, c_lo:c_hi+1] = np.maximum(
            grid[r_lo:r_hi+1, c_lo:c_hi+1], base_cost
        )

    return cost_map
