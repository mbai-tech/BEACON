import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from beacon.core.planner import obstacle_polygon

# ── Semantic base costs ────────────────────────────────────────────────────────

SEMANTIC_COSTS: Dict[str, float] = {
    "movable":   3.0,
    "unmovable": 8.0,
    "unknown":   0.0,   # optimistic — unknown = free
}

RESOLUTION: float = 0.05   # metres per cell
R_MAP:      float = 1.5    # sliding-window radius (metres)
GOAL_SENTINEL: float = -1.0


# ── CostMap dataclass ──────────────────────────────────────────────────────────

@dataclass
class CostMap:
    """2-D grid cost map aligned to a fixed resolution."""

    grid: np.ndarray        # shape (H, W), dtype float32
    resolution: float       # metres per cell
    origin: np.ndarray      # (x, y) world coords of cell (0, 0)

    # Cached anisotropic layer (same shape as grid); None until computed.
    aniso: Optional[np.ndarray] = field(default=None, repr=False)

    # ── coordinate helpers ─────────────────────────────────────────────────────

    @property
    def height(self) -> int:
        return self.grid.shape[0]

    @property
    def width(self) -> int:
        return self.grid.shape[1]

    def world_to_cell(self, xy: np.ndarray) -> Tuple[int, int]:
        col = int((float(xy[0]) - self.origin[0]) / self.resolution)
        row = int((float(xy[1]) - self.origin[1]) / self.resolution)
        return row, col

    def cell_to_world(self, row: int, col: int) -> np.ndarray:
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return np.array([x, y], dtype=float)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def cost_at_world(self, xy: np.ndarray) -> float:
        r, c = self.world_to_cell(xy)
        if self.in_bounds(r, c):
            layer = self.aniso if self.aniso is not None else self.grid
            return float(layer[r, c])
        return float("inf")


# ── Step 1: build_voxel_grid ───────────────────────────────────────────────────

def build_voxel_grid(
    scene: dict,
    resolution: float = RESOLUTION,
    observed_only: bool = True,
    inflation_radius: float = 0.0,
) -> CostMap:
    """Build M: the base semantic cost grid.

    Each cell takes the maximum semantic cost of any obstacle that
    overlaps it (top-down max-pooling).  Free cells = 0, goal = -1.

    Parameters
    ----------
    scene            : beacon.core scene dict
    resolution       : metres per grid cell
    observed_only    : only rasterise observed obstacles (online mode)
    inflation_radius : buffer each obstacle polygon by this amount
    """
    xmin, xmax, ymin, ymax = scene["workspace"]
    cols = int(round((xmax - xmin) / resolution)) + 1
    rows = int(round((ymax - ymin) / resolution)) + 1
    origin = np.array([xmin, ymin], dtype=float)

    grid = np.zeros((rows, cols), dtype=np.float32)
    cost_map = CostMap(grid=grid, resolution=resolution, origin=origin)

    from shapely.geometry import Point as _Point

    for obs in scene["obstacles"]:
        if observed_only and not obs.get("observed", False):
            continue

        sem_class = obs.get("true_class", obs.get("class_true", "movable"))
        cost = float(SEMANTIC_COSTS.get(sem_class, 1.0))
        if cost == 0.0:
            continue

        poly = obstacle_polygon(obs)
        if inflation_radius > 0.0:
            poly = poly.buffer(inflation_radius)

        bx_min, by_min, bx_max, by_max = poly.bounds
        r_lo, c_lo = cost_map.world_to_cell(np.array([bx_min, by_min]))
        r_hi, c_hi = cost_map.world_to_cell(np.array([bx_max, by_max]))
        r_lo = max(0, r_lo - 1)
        r_hi = min(rows - 1, r_hi + 1)
        c_lo = max(0, c_lo - 1)
        c_hi = min(cols - 1, c_hi + 1)

        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                if grid[r, c] >= cost:
                    continue
                world_pt = cost_map.cell_to_world(r, c)
                if poly.contains(_Point(world_pt[0], world_pt[1])):
                    grid[r, c] = cost

    # Mark goal cell
    goal_xy = np.array(scene["goal"][:2], dtype=float)
    gr, gc = cost_map.world_to_cell(goal_xy)
    if cost_map.in_bounds(gr, gc):
        grid[gr, gc] = GOAL_SENTINEL

    return cost_map


# ── Step 2: compute_anisotropic_map ───────────────────────────────────────────

def compute_anisotropic_map(
    cost_map: CostMap,
    scene: dict,
    m_samples: int = 20,
    alpha: float = 0.5,
    c_thresh: float = 5.0,
    push_dist: float = 0.3,
    sigma_angle: float = 0.4,
) -> np.ndarray:
    """Compute M': the anisotropic cost layer (Section V-B).

    For every boundary cell of a pushable obstacle (cost in (0, c_thresh]):
      1. Compute the outward surface normal n̂ from the cell toward free space.
      2. Sample m push directions from N(-n̂, σ²I) (reverse normal = push dir).
      3. Score each direction: u_i = safety of the landing zone after a push
         of distance `push_dist`.
      4. f_s = Σ(λ_i · u_i) / Σλ_i  where λ_i is the Gaussian likelihood.

    M'[r,c] = α · M[r,c] + (1-α) · (10 - 10 · f_s[r,c])

    For non-boundary / high-cost / free cells, M' = M unchanged.

    Parameters
    ----------
    cost_map    : base grid from build_voxel_grid
    scene       : beacon.core scene dict (for workspace bounds)
    m_samples   : number of push-direction samples per boundary cell
    alpha       : blend weight between base cost and anisotropic penalty
    c_thresh    : max base cost to consider as pushable
    push_dist   : simulated push distance in metres
    sigma_angle : std-dev of angular Gaussian perturbation (radians)
    """
    M = cost_map.grid
    H, W = M.shape
    rng = np.random.default_rng(seed=0)

    f_s = np.zeros((H, W), dtype=np.float32)
    computed = np.zeros((H, W), dtype=bool)

    neighbors_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            cell_cost = float(M[r, c])
            if cell_cost <= 0.0 or cell_cost > c_thresh:
                continue

            # Check if this is a boundary cell (at least one free neighbour)
            free_dirs = []
            for dr, dc in neighbors_4:
                nr, nc = r + dr, c + dc
                if cost_map.in_bounds(nr, nc) and M[nr, nc] == 0.0:
                    free_dirs.append(np.array([dc, dr], dtype=float))  # (x, y) direction

            if not free_dirs:
                continue

            # Outward surface normal: average of free-neighbour directions
            normal = np.mean(free_dirs, axis=0)
            norm_len = float(np.linalg.norm(normal))
            if norm_len < 1e-9:
                continue
            normal /= norm_len

            # Push direction = reverse normal (push the object away from robot)
            push_dir_mean = -normal

            # Sample m push directions from angular Gaussian
            angles = rng.normal(0.0, sigma_angle, size=m_samples)
            push_dirs = np.stack([
                np.array([
                    push_dir_mean[0] * np.cos(a) - push_dir_mean[1] * np.sin(a),
                    push_dir_mean[0] * np.sin(a) + push_dir_mean[1] * np.cos(a),
                ])
                for a in angles
            ])  # (m, 2)

            # Gaussian likelihoods (proportional — use angle deviation directly)
            lambdas = np.exp(-0.5 * (angles / sigma_angle) ** 2)

            # Evaluate safety score for each push outcome
            world_pos = cost_map.cell_to_world(r, c)
            u = np.zeros(m_samples, dtype=float)
            for i, (d, lam) in enumerate(zip(push_dirs, lambdas)):
                d_norm = float(np.linalg.norm(d))
                if d_norm < 1e-9:
                    u[i] = 0.0
                    continue
                d = d / d_norm
                landing = world_pos + d * push_dist
                lr, lc = cost_map.world_to_cell(landing)
                if not cost_map.in_bounds(lr, lc):
                    u[i] = 0.0
                    continue
                landing_cost = float(M[lr, lc])
                if landing_cost == 0.0:
                    u[i] = 1.0          # landed in free space
                elif landing_cost < 0:  # goal cell
                    u[i] = 1.0
                else:
                    # Scale down by normalised path cost
                    u[i] = 1.0 / (1.0 + landing_cost)

            denom = float(lambdas.sum())
            f_s[r, c] = float((lambdas * u).sum()) / denom if denom > 1e-12 else 0.0
            computed[r, c] = True

    # M'[r,c] = α·M + (1-α)·(10 - 10·f_s)
    M_prime = M.copy().astype(np.float32)
    aniso_penalty = (10.0 - 10.0 * f_s).astype(np.float32)
    mask = computed & (M > 0)
    M_prime[mask] = (alpha * M[mask] + (1.0 - alpha) * aniso_penalty[mask]).astype(np.float32)

    cost_map.aniso = M_prime
    return M_prime


# ── Step 3: update (sliding window) ───────────────────────────────────────────

def update(
    cost_map: CostMap,
    scene: dict,
    robot_pos: np.ndarray,
    r_map: float = R_MAP,
    m_samples: int = 20,
    alpha: float = 0.5,
    c_thresh: float = 5.0,
) -> np.ndarray:
    """Recompute M' inside a sliding window centred on the robot.

    Only the cells within radius r_map of robot_pos are updated;
    the rest of cost_map.aniso is left untouched.

    Parameters
    ----------
    cost_map  : CostMap with an existing .aniso layer (or None)
    scene     : beacon.core scene dict
    robot_pos : (x, y) robot position in world coordinates
    r_map     : sliding-window radius in metres

    Returns the full updated M' array.
    """
    if cost_map.aniso is None:
        cost_map.aniso = cost_map.grid.copy().astype(np.float32)

    robot_pos = np.array(robot_pos[:2], dtype=float)
    r_cells = int(np.ceil(r_map / cost_map.resolution))
    r_center, c_center = cost_map.world_to_cell(robot_pos)

    r_lo = max(0, r_center - r_cells)
    r_hi = min(cost_map.height - 1, r_center + r_cells)
    c_lo = max(0, c_center - r_cells)
    c_hi = min(cost_map.width - 1, c_center + r_cells)

    # Build a mini scene restricted to obstacles whose bbox overlaps the window
    win_xmin = cost_map.origin[0] + c_lo * cost_map.resolution
    win_xmax = cost_map.origin[0] + (c_hi + 1) * cost_map.resolution
    win_ymin = cost_map.origin[1] + r_lo * cost_map.resolution
    win_ymax = cost_map.origin[1] + (r_hi + 1) * cost_map.resolution

    window_scene = {
        "workspace": [win_xmin, win_xmax, win_ymin, win_ymax],
        "start":     scene["start"],
        "goal":      scene["goal"],
        "obstacles": [
            obs for obs in scene["obstacles"]
            if _obs_overlaps_window(obs, win_xmin, win_xmax, win_ymin, win_ymax)
        ],
    }

    # Rebuild the base grid for the window only
    window_map = build_voxel_grid(
        window_scene,
        resolution=cost_map.resolution,
        observed_only=True,
    )

    # Compute anisotropic layer for the window
    window_prime = compute_anisotropic_map(
        window_map, window_scene,
        m_samples=m_samples, alpha=alpha, c_thresh=c_thresh,
    )

    # Write back into the full map, clipped to the window bounds
    win_H, win_W = window_prime.shape
    r_write = min(r_hi - r_lo + 1, win_H)
    c_write = min(c_hi - c_lo + 1, win_W)
    cost_map.aniso[r_lo:r_lo + r_write, c_lo:c_lo + c_write] = \
        window_prime[:r_write, :c_write]

    return cost_map.aniso


def _obs_overlaps_window(
    obs: dict,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
) -> bool:
    """Return True if the obstacle's bounding box intersects the window."""
    if not obs.get("observed", False):
        return False
    poly = obstacle_polygon(obs)
    bx_min, by_min, bx_max, by_max = poly.bounds
    return (bx_max >= xmin and bx_min <= xmax and
            by_max >= ymin and by_min <= ymax)
