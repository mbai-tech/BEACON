"""
scene_generator.py — PyBullet scene generator for BEACON experiments.

Configurations
--------------
Six canonical configs built from SceneConfig(density, fragility):
  S-U: sparse  + uniform    S-M: sparse  + mixed
  M-U: medium  + uniform    M-M: medium  + mixed
  D-U: dense   + uniform    D-M: dense   + mixed

Density → obstacle count:
  sparse  →  5–10
  medium  → 15–25
  dense   → 30–50

Fragility distributions:
  uniform : c(O) ~ Uniform{1..9} for each obstacle
  mixed   : 70% have c(O) in {1..5}, 30% have c(O) in {7..9}

Public API
----------
SceneConfig        — density + fragility spec
Scene              — PyBullet session + cost map + planner-compatible dict
generate_scene(config, seed) → Scene
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import numpy as np
import pybullet as p
import pybullet_data
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from beacon.planning.semantic_cost import (
    MOVABLE_THRESHOLD,
    _assign_cost_on_client,
)


# ── Arena constants ────────────────────────────────────────────────────────────

ARENA_SIZE   = 5.0          # metres  (square: 0..5 × 0..5)
START_POS    = (0.5, 0.5)
GOAL_POS     = (4.5, 4.5)
CLEAR_RADIUS = 0.5          # keep obstacles away from start / goal
ROBOT_RADIUS = 0.15

# ── Density ranges ─────────────────────────────────────────────────────────────

_DENSITY_RANGES: Dict[str, Tuple[int, int]] = {
    "sparse": ( 5, 10),
    "medium": (15, 25),
    "dense":  (30, 50),
}

# ── Physics templates  ─────────────────────────────────────────────────────────
# Each entry: (mass_kg, friction, restitution, half_extents_xyz)
#
# Proxy formula:  c_raw = 0.5·clip(h/min(w,d), 0,5)
#                       + 0.3·clip(mass·restitution, 0,3)
#                       + 0.2·clip(mass·friction, 0,2)
# Max achievable proxy cost is 4 (c_raw_max = 3.8 → round = 4).
# For target costs 5–9 the proxy saturates; we assign the intended cost
# directly and record the proxy separately for diagnostics.
#
_PHYSICS_TEMPLATES: Dict[int, Tuple[float, float, float, Tuple[float, float, float]]] = {
    # cost : (mass, friction, restitution, (hx, hy, hz))
    1: (0.10, 0.30, 0.02, (0.10, 0.10, 0.10)),  # cube, very light   → proxy ≈ 1
    2: (0.30, 0.40, 0.10, (0.04, 0.04, 0.12)),  # tall, light        → proxy ≈ 2
    3: (0.50, 0.35, 0.05, (0.03, 0.03, 0.15)),  # very tall, light   → proxy ≈ 3
    4: (10.0, 0.20, 0.30, (0.05, 0.05, 0.25)),  # max proxy params   → proxy ≈ 4
    5: (2.00, 0.50, 0.25, (0.08, 0.08, 0.20)),  # heavy, tall        (proxy saturates)
    6: (3.00, 0.55, 0.30, (0.07, 0.07, 0.22)),
    7: (5.00, 0.60, 0.35, (0.06, 0.06, 0.25)),
    8: (7.00, 0.65, 0.38, (0.05, 0.05, 0.28)),
    9: (9.00, 0.70, 0.40, (0.05, 0.05, 0.30)),
}


# ── SceneConfig ────────────────────────────────────────────────────────────────

@dataclass
class SceneConfig:
    """Specification for one scene configuration.

    density   : 'sparse' | 'medium' | 'dense'
    fragility : 'uniform' | 'mixed'
    """
    density:   str
    fragility: str

    # Convenience constructors for the six canonical configs
    @staticmethod
    def S_U() -> "SceneConfig": return SceneConfig("sparse",  "uniform")
    @staticmethod
    def S_M() -> "SceneConfig": return SceneConfig("sparse",  "mixed")
    @staticmethod
    def M_U() -> "SceneConfig": return SceneConfig("medium",  "uniform")
    @staticmethod
    def M_M() -> "SceneConfig": return SceneConfig("medium",  "mixed")
    @staticmethod
    def D_U() -> "SceneConfig": return SceneConfig("dense",   "uniform")
    @staticmethod
    def D_M() -> "SceneConfig": return SceneConfig("dense",   "mixed")

    def __post_init__(self):
        if self.density not in _DENSITY_RANGES:
            raise ValueError(f"density must be one of {list(_DENSITY_RANGES)}")
        if self.fragility not in ("uniform", "mixed"):
            raise ValueError("fragility must be 'uniform' or 'mixed'")

    @property
    def label(self) -> str:
        return f"{self.density[0].upper()}-{self.fragility[0].upper()}"


# ── Scene ──────────────────────────────────────────────────────────────────────

@dataclass
class Scene:
    """A generated PyBullet scene.

    Attributes
    ----------
    client_id    : PyBullet physics client id
    robot_id     : PyBullet body id of the robot disk
    goal_id      : PyBullet body id of the goal marker
    obstacle_ids : ordered list of obstacle body ids
    cost_map     : {body_id → intended_cost}  (from fragility distribution)
    proxy_costs  : {body_id → assign_cost() result} (physics proxy estimate)
    goal_pos     : (x, y) goal position
    workspace    : (xmin, xmax, ymin, ymax)
    config       : SceneConfig used to generate the scene
    seed         : RNG seed
    """
    client_id:    int
    robot_id:     int
    goal_id:      int
    obstacle_ids: List[int]
    cost_map:     Dict[int, int]   # body_id → intended cost
    proxy_costs:  Dict[int, int]   # body_id → assign_cost() result
    goal_pos:     np.ndarray
    workspace:    Tuple[float, float, float, float]
    config:       SceneConfig
    seed:         int

    # ── planner-compatible dict ────────────────────────────────────────────────

    def to_planner_dict(self) -> dict:
        """Return a scene dict in the format expected by the BEACON planner.

        Each obstacle becomes:
          {
            "id"        : int,
            "vertices"  : [[x,y], [x,y], [x,y], [x,y]],   ← 2-D AABB corners
            "true_class": "movable" | "unmovable",
            "class_true": same,
            "observed"  : False,
          }
        """
        obstacles_out = []
        for idx, body_id in enumerate(self.obstacle_ids):
            aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=self.client_id)
            xmin, ymin = aabb_min[0], aabb_min[1]
            xmax, ymax = aabb_max[0], aabb_max[1]
            vertices = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ]
            cost = self.cost_map[body_id]
            cls  = "movable" if cost < MOVABLE_THRESHOLD else "unmovable"
            obstacles_out.append({
                "id":         idx,
                "vertices":   vertices,
                "true_class": cls,
                "class_true": cls,
                "observed":   False,
                "cost":       cost,
            })

        xmin, xmax, ymin, ymax = self.workspace
        return {
            "workspace": [xmin, xmax, ymin, ymax],
            "start":     [START_POS[0], START_POS[1], 0.0],
            "goal":      [GOAL_POS[0],  GOAL_POS[1],  0.0],
            "obstacles": obstacles_out,
            "family":    self.config.label,
            "seed":      self.seed,
        }

    def disconnect(self) -> None:
        """Shut down the PyBullet session."""
        p.disconnect(physicsClientId=self.client_id)


# ── generate_scene ─────────────────────────────────────────────────────────────

def generate_scene(config: SceneConfig, seed: int = 0) -> Scene:
    """Generate a scene in a headless PyBullet session.

    Steps
    -----
    1. Open DIRECT PyBullet session, load ground plane, set gravity.
    2. Place robot cylinder at START_POS, goal sphere at GOAL_POS.
    3. Sample n_obstacles from the density range.
    4. Sample target costs from the fragility distribution.
    5. For each obstacle: pick physics template, rejection-sample a
       non-overlapping position, spawn the body.
    6. Call assign_cost() on each body; store intended and proxy costs.
    7. Return a Scene object.

    Parameters
    ----------
    config : SceneConfig
    seed   : integer RNG seed for reproducibility
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # ── 1. PyBullet session ──────────────────────────────────────────────────
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)

    workspace = (0.0, ARENA_SIZE, 0.0, ARENA_SIZE)

    # ── 2. Robot and goal marker ─────────────────────────────────────────────
    robot_id = _spawn_cylinder(
        client, position=(START_POS[0], START_POS[1], 0.15),
        radius=ROBOT_RADIUS, height=0.30, mass=5.0,
        color=[0.2, 0.4, 0.9, 1.0],
    )
    goal_id = _spawn_sphere(
        client, position=(GOAL_POS[0], GOAL_POS[1], 0.05),
        radius=0.08, mass=0.0,   # static marker
        color=[0.0, 0.9, 0.2, 0.6],
    )

    # ── 3. Obstacle count ────────────────────────────────────────────────────
    lo, hi = _DENSITY_RANGES[config.density]
    n_obstacles = rng.randint(lo, hi)

    # ── 4. Target costs from fragility distribution ──────────────────────────
    target_costs = _sample_costs(config.fragility, n_obstacles, rng)

    # ── 5. Spawn obstacles ───────────────────────────────────────────────────
    placed_aabbs: List[Tuple[np.ndarray, np.ndarray]] = []  # (min_xy, max_xy)
    obstacle_ids: List[int] = []
    cost_map:     Dict[int, int] = {}
    proxy_costs:  Dict[int, int] = {}

    for target_cost in target_costs:
        mass, friction, restitution, half_extents = _PHYSICS_TEMPLATES[target_cost]

        # Random orientation (yaw only)
        yaw = rng.uniform(0, 2 * np.pi)

        pos_xy = _sample_position(
            half_extents, yaw, placed_aabbs, workspace, rng,
            clear_start=np.array(START_POS),
            clear_goal=np.array(GOAL_POS),
            clear_radius=CLEAR_RADIUS,
        )
        if pos_xy is None:
            continue   # couldn't place — skip (dense scenes may not fit all)

        z = half_extents[2] + 0.001   # just above ground
        body_id = _spawn_box(
            client, position=(pos_xy[0], pos_xy[1], z),
            half_extents=half_extents, mass=mass,
            friction=friction, restitution=restitution, yaw=yaw,
        )

        # Record AABB for future overlap checks
        aabb_min2, aabb_max2 = _aabb_2d(half_extents, pos_xy, yaw)
        placed_aabbs.append((aabb_min2, aabb_max2))
        obstacle_ids.append(body_id)

        # ── 6. assign_cost ───────────────────────────────────────────────────
        proxy = _assign_cost_on_client(body_id, client, {}, beta=(0.5, 0.3, 0.2))
        cost_map[body_id]    = target_cost   # intended cost governs planning
        proxy_costs[body_id] = proxy

    return Scene(
        client_id=client,
        robot_id=robot_id,
        goal_id=goal_id,
        obstacle_ids=obstacle_ids,
        cost_map=cost_map,
        proxy_costs=proxy_costs,
        goal_pos=np.array(GOAL_POS, dtype=float),
        workspace=workspace,
        config=config,
        seed=seed,
    )


# ── Fragility sampling ─────────────────────────────────────────────────────────

def _sample_costs(fragility: str, n: int, rng: random.Random) -> List[int]:
    """Sample n target costs: movable=2, unmovable=8.

    mixed       — 70% movable (cost 2), 30% unmovable (cost 8)
    uniform     — 50/50 split
    all_movable — all movable (cost 2)
    """
    if fragility == "all_movable":
        return [2] * n
    if fragility == "uniform":
        return [2 if rng.random() < 0.5 else 8 for _ in range(n)]
    # mixed
    return [2 if rng.random() < 0.70 else 8 for _ in range(n)]


# ── Obstacle placement ─────────────────────────────────────────────────────────

def _sample_position(
    half_extents: Tuple[float, float, float],
    yaw: float,
    placed_aabbs: List[Tuple[np.ndarray, np.ndarray]],
    workspace:    Tuple[float, float, float, float],
    rng:          random.Random,
    clear_start:  np.ndarray,
    clear_goal:   np.ndarray,
    clear_radius: float,
    max_attempts: int = 200,
) -> Optional[np.ndarray]:
    """Rejection-sample a 2-D position for an obstacle with no overlaps."""
    xmin, xmax, ymin, ymax = workspace
    hx, hy = half_extents[0], half_extents[1]
    margin = max(hx, hy) * 1.5

    for _ in range(max_attempts):
        x = rng.uniform(xmin + margin, xmax - margin)
        y = rng.uniform(ymin + margin, ymax - margin)
        pos = np.array([x, y])

        # Clearance from start and goal
        if (np.linalg.norm(pos - clear_start) < clear_radius + margin or
                np.linalg.norm(pos - clear_goal)  < clear_radius + margin):
            continue

        # Overlap check against already-placed obstacles
        new_min, new_max = _aabb_2d(half_extents, pos, yaw)
        overlap = any(
            _aabbs_overlap(new_min, new_max, pm, pM)
            for pm, pM in placed_aabbs
        )
        if not overlap:
            return pos

    return None   # placement failed after max_attempts


def _aabb_2d(
    half_extents: Tuple[float, float, float],
    center_xy:    np.ndarray,
    yaw:          float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of a yaw-rotated box in 2-D."""
    hx, hy = float(half_extents[0]), float(half_extents[1])
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    ext_x = hx * c + hy * s
    ext_y = hx * s + hy * c
    ext = np.array([ext_x, ext_y])
    return center_xy - ext, center_xy + ext


def _aabbs_overlap(
    a_min: np.ndarray, a_max: np.ndarray,
    b_min: np.ndarray, b_max: np.ndarray,
    gap:   float = 0.02,
) -> bool:
    """Return True if two 2-D AABBs overlap (with a small gap buffer)."""
    return bool(
        a_max[0] + gap > b_min[0] and a_min[0] - gap < b_max[0] and
        a_max[1] + gap > b_min[1] and a_min[1] - gap < b_max[1]
    )


# ── PyBullet body helpers ──────────────────────────────────────────────────────

def _spawn_box(
    client:      int,
    position:    Tuple[float, float, float],
    half_extents: Tuple[float, float, float],
    mass:        float,
    friction:    float,
    restitution: float,
    yaw:         float = 0.0,
) -> int:
    col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=list(half_extents),
        physicsClientId=client,
    )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=list(half_extents),
        rgbaColor=[0.7, 0.5, 0.3, 1.0],
        physicsClientId=client,
    )
    orn = p.getQuaternionFromEuler([0, 0, yaw])
    body = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(position),
        baseOrientation=list(orn),
        physicsClientId=client,
    )
    p.changeDynamics(
        body, -1,
        lateralFriction=friction,
        restitution=restitution,
        physicsClientId=client,
    )
    return body


def _spawn_cylinder(
    client:   int,
    position: Tuple[float, float, float],
    radius:   float,
    height:   float,
    mass:     float,
    color:    List[float],
) -> int:
    col = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius, height=height,
        physicsClientId=client,
    )
    vis = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius, length=height,
        rgbaColor=color, physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(position),
        physicsClientId=client,
    )


def _spawn_sphere(
    client:   int,
    position: Tuple[float, float, float],
    radius:   float,
    mass:     float,
    color:    List[float],
) -> int:
    col = p.createCollisionShape(
        p.GEOM_SPHERE, radius=radius, physicsClientId=client,
    )
    vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=radius, rgbaColor=color,
        physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(position),
        physicsClientId=client,
    )


# ── Load existing scenes ───────────────────────────────────────────────────────

_SCENES_DIR = Path(__file__).resolve().parent / "data" / "scenes"


def load_scene(
    index_or_path,
    family: str = None,
    fragility: str = "mixed",
    seed: int = 0,
) -> dict:
    """Load an existing scene JSON and assign movable/unmovable labels.

    Parameters
    ----------
    index_or_path : int or str or Path
        If int and family is None, loads `data/scenes/scene_{index:03d}.json`.
        If int and family is given, loads `data/scenes/{family}/scene_{index:03d}.json`.
        If str/Path, loads that file directly.
    family : str or None
        One of 'dense_clutter', 'narrow_passage', 'perturbed', or None for
        the flat legacy directory.
    fragility : 'mixed' | 'uniform' | 'all_movable'
        How costs are (re-)assigned to obstacles.
    seed : int
        RNG seed for fragility sampling.

    Returns
    -------
    dict : planner-compatible scene dict with obstacles as dicts containing
           id, vertices, true_class, class_true, observed, cost.
    """
    import json

    if isinstance(index_or_path, int):
        if family is not None:
            path = _SCENES_DIR / family / f"scene_{index_or_path:03d}.json"
        else:
            path = _SCENES_DIR / f"scene_{index_or_path:03d}.json"
    else:
        path = Path(index_or_path)

    with open(path) as f:
        raw = json.load(f)

    raw_obstacles = raw["obstacles"]
    costs = _sample_costs(fragility, len(raw_obstacles), random.Random(seed))

    obstacles_out = []
    for idx, (obs, cost) in enumerate(zip(raw_obstacles, costs)):
        cls = "movable" if cost < MOVABLE_THRESHOLD else "unmovable"
        # New circle format: obs is already a dict with vertices etc.
        # Old format: obs is a list of [x, y] vertex pairs.
        if isinstance(obs, dict):
            entry = {**obs, "id": idx, "true_class": cls,
                     "class_true": cls, "observed": False, "cost": cost}
        else:
            entry = {"id": idx, "vertices": obs, "true_class": cls,
                     "class_true": cls, "observed": False, "cost": cost}
        obstacles_out.append(entry)

    workspace = raw["workspace"]
    start = raw["start"]
    goal  = raw["goal"]

    return {
        "workspace": workspace,
        "start":     [float(start[0]), float(start[1]), 0.0],
        "goal":      [float(goal[0]),  float(goal[1]),  0.0],
        "obstacles": obstacles_out,
        "family":    raw.get("family", f"loaded_{fragility}"),
        "seed":      seed,
    }


def load_all_scenes(
    family: str = None,
    fragility: str = "mixed",
    seed: int = 0,
) -> List[dict]:
    """Load every scene in data/scenes/ (or a family subdirectory) and label obstacles.

    Parameters
    ----------
    family : str or None
        If given, loads from `data/scenes/{family}/`.
        If None, loads from the flat `data/scenes/` directory.
    """
    if family is not None:
        paths = sorted((_SCENES_DIR / family).glob("scene_*.json"))
    else:
        paths = sorted(_SCENES_DIR.glob("scene_*.json"))
    return [load_scene(p, fragility=fragility, seed=seed + i)
            for i, p in enumerate(paths)]


def _label_obstacles(
    raw_obstacles: list,
    fragility: str,
    seed: int,
) -> List[int]:
    """Assign cost 2 (movable) or 8 (unmovable) to each obstacle.

    mixed       — 70% movable, 30% unmovable (random)
    uniform     — 50/50 split (random)
    all_movable — every obstacle is movable
    """
    n = len(raw_obstacles)
    if n == 0:
        return []
    rng = random.Random(seed)
    return _sample_costs(fragility, n, rng)


# ── Convenience: generate all six canonical configs ────────────────────────────

def generate_all_configs(base_seed: int = 0) -> Dict[str, Scene]:
    """Generate one scene for each of the six canonical configurations."""
    configs = [
        SceneConfig.S_U(), SceneConfig.S_M(),
        SceneConfig.M_U(), SceneConfig.M_M(),
        SceneConfig.D_U(), SceneConfig.D_M(),
    ]
    return {
        cfg.label: generate_scene(cfg, seed=base_seed + i)
        for i, cfg in enumerate(configs)
    }


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # PyBullet-generated scenes
    for cfg in [SceneConfig.S_U(), SceneConfig.M_M(), SceneConfig.D_U()]:
        scene = generate_scene(cfg, seed=42)
        n = len(scene.obstacle_ids)
        costs = list(scene.cost_map.values())
        proxies = list(scene.proxy_costs.values())
        print(
            f"{cfg.label}: {n} obstacles | "
            f"costs {min(costs)}–{max(costs)} "
            f"(proxy {min(proxies)}–{max(proxies)})"
        )
        scene.disconnect()

    # Loaded existing scenes
    print()
    for fragility in ("mixed", "uniform", "all_movable"):
        scene_dict = load_scene(0, fragility=fragility, seed=0)
        obs = scene_dict["obstacles"]
        movable   = sum(1 for o in obs if o["true_class"] == "movable")
        unmovable = len(obs) - movable
        print(
            f"scene_000 [{fragility:12s}]: {len(obs)} obstacles | "
            f"movable={movable} unmovable={unmovable}"
        )
