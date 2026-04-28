"""BEACON scene configuration generation."""

import copy
import random
import sys
from pathlib import Path

from shapely.geometry import Point, Polygon

# Ensure beacon/ sub-package is importable
_BEACON_DIR = str(Path(__file__).resolve().parent.parent / "beacon")
if _BEACON_DIR not in sys.path:
    sys.path.insert(0, _BEACON_DIR)

from beacon.environment.scene_basic import (
    generate_circle_scene,
    polygon_to_list,
    valid_candidate,
)
from beacon.core.constants import (
    DENSITY_OBSTACLE_COUNTS,
    SCENE_CONFIGS,
)


# ── semantic cost assignment ──────────────────────────────────────────────────

def assign_semantic_cost(true_class: str) -> int:
    if true_class == "forbidden":
        return 10
    if true_class == "fragile":
        return random.randint(7, 9)
    if true_class == "safe":
        return random.randint(3, 6)
    return random.randint(1, 4)  # movable


def _assign_fragility_class(fragility_profile: str) -> str:
    if fragility_profile == "uniform":
        return random.choices(
            ["movable", "safe", "fragile"],
            weights=[0.45, 0.35, 0.20],
            k=1,
        )[0]
    if random.random() < 0.70:
        return random.choice(["movable", "safe"])
    return random.choices(["fragile", "forbidden"], weights=[0.85, 0.15], k=1)[0]


# ── obstacle placement ────────────────────────────────────────────────────────

def _add_obstacles_to_scene(scene: dict, n: int, fragility_profile: str) -> dict:
    scene = copy.deepcopy(scene)
    workspace    = tuple(scene["workspace"])   # (xmin, xmax, ymin, ymax)
    start        = tuple(scene["start"][:2])
    goal         = tuple(scene["goal"][:2])
    placed       = [Polygon(obs["vertices"]) for obs in scene["obstacles"]]
    start_buffer = Point(start).buffer(0.5)
    goal_buffer  = Point(goal).buffer(0.5)

    xmin, xmax, ymin, ymax = workspace
    added, attempts = 0, 0

    while added < n and attempts < n * 150:
        attempts += 1
        radius = random.uniform(0.13, 0.30)
        cx = random.uniform(xmin + radius + 0.05, xmax - radius - 0.05)
        cy = random.uniform(ymin + radius + 0.05, ymax - radius - 0.05)
        candidate = Point(cx, cy).buffer(radius, resolution=20)
        # scene_basic.valid_candidate checks within_workspace using its hardcoded
        # WORKSPACE constant, so we do the overlap/buffer checks manually here
        if candidate.intersects(start_buffer) or candidate.intersects(goal_buffer):
            continue
        if any(candidate.intersects(p) for p in placed):
            continue
        placed.append(candidate)
        true_class = _assign_fragility_class(fragility_profile)
        scene["obstacles"].append({
            "id":            len(scene["obstacles"]),
            "shape_type":    "circle",
            "true_class":    true_class,
            "class_true":    true_class,
            "vertices":      polygon_to_list(candidate),
            "semantic_cost": assign_semantic_cost(true_class),
        })
        added += 1

    return scene


# ── public API ────────────────────────────────────────────────────────────────

def generate_config_environment(config: str) -> dict:
    """Generate one scene for the named BEACON config (e.g. "D-M")."""
    density_name, fragility = SCENE_CONFIGS[config]
    n_min, n_max = DENSITY_OBSTACLE_COUNTS[density_name]
    target_n = random.randint(n_min, n_max)

    # Use a clean base scene with no pre-placed obstacles
    base = generate_circle_scene(family="sparse")
    base["obstacles"] = []

    scene = _add_obstacles_to_scene(base, target_n, fragility)

    scene["config"]            = config
    scene["density"]           = density_name
    scene["fragility_profile"] = fragility
    scene["family"]            = f"config_{config}"
    scene["seed"]              = random.randint(0, 2 ** 31 - 1)

    return scene
