import copy
import random

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point, Polygon

from beacon.core.constants import CLUTTER_BIASED_FAMILIES, TARGET_MAX_WORKSPACE_SPAN

def _import_scene_generator():
    from beacon.environment.scene_generator_shapely import generate_scene, polygon_to_list, valid_candidate
    return generate_scene, polygon_to_list, valid_candidate


CLASSES: list[str] = ["safe", "movable", "fragile", "forbidden"]

# Confusion-model parameters
_BELIEF_CORRECT  = 0.8                              # P(true class | true class)
_BELIEF_OTHER    = (1.0 - _BELIEF_CORRECT) / 3.0   # uniform over the other three


def init_belief(raw_class: str) -> dict[str, float]:
    """Return a 4-class prior for one obstacle using a simple confusion model.

    P(correct class) = 0.8; the remaining 0.2 mass is split uniformly over the
    other three classes.  If ``raw_class`` is not in CLASSES (e.g. a legacy
    binary label such as ``not_movable``) the prior is uniform.

    The returned dict always sums to 1.0 over CLASSES.
    """
    if raw_class not in CLASSES:
        return {c: 1.0 / len(CLASSES) for c in CLASSES}
    return {c: (_BELIEF_CORRECT if c == raw_class else _BELIEF_OTHER) for c in CLASSES}


def map_class(belief: dict[str, float]) -> str:
    """Return the MAP (most-probable) class from a belief dict."""
    return max(belief, key=belief.__getitem__)


def coerce_pushable_class(raw_class: str) -> str:
    """Map any class label to the binary movable / not_movable scheme.

    This coercion is kept for backward compatibility with all motion-logic
    code in planner.py that expects a binary ``true_class``.  The full
    4-class information is carried in ``obstacle["belief"]`` and
    ``obstacle["map_class"]``.
    """
    if raw_class == "movable":
        return "movable"
    return "not_movable"


def normalize_scene_for_online_use(scene: dict) -> dict:
    """Attach the state fields needed by online sensing and visualization.

    Each obstacle gets:
      ``belief``         – 4-class probability vector (safe / movable / fragile / forbidden)
                          initialised with the confusion-model prior.
      ``initial_belief`` – immutable copy of that prior; used as the decay target so
                          belief_decay in the planner can interpolate back toward
                          the original sensor classification over time.
      ``map_class``      – argmax of ``belief``; tracks the robot's current best
                          estimate of the obstacle class.
      ``true_class``/ ``class_true`` – binary coercion (movable / not_movable)
                          retained for backward compatibility with motion-logic code.
    """
    normalized = copy.deepcopy(scene)
    for obstacle in normalized["obstacles"]:
        raw = obstacle.get("true_class", obstacle.get("class_true", "movable"))
        belief = init_belief(raw)
        obstacle["belief"]         = belief
        obstacle["initial_belief"] = copy.deepcopy(belief)
        obstacle["map_class"]      = map_class(belief)
        # Binary coercion — keeps all planner.py motion logic unchanged.
        binary_class = coerce_pushable_class(raw)
        obstacle["true_class"] = binary_class
        obstacle["class_true"] = binary_class
        obstacle["observed"]   = False
        obstacle["initial_vertices"] = copy.deepcopy(obstacle["vertices"])
        obstacle.pop("_poly", None)
    return normalized


def create_cluttered_variant(scene: dict, extra_obstacles: int = 10) -> dict:
    generate_scene, polygon_to_list, valid_candidate = _import_scene_generator()
    """Add extra circular clutter near the start-goal corridor.

    The corridor bias increases the probability that the robot must either
    detour or push. Geometrically, we sample centers near

        c(t) = s + t (g - s),   t in [0, 1],

    where ``s`` is the start point and ``g`` is the goal.
    """
    cluttered = copy.deepcopy(scene)
    family = cluttered["family"]
    workspace = tuple(cluttered["workspace"])
    start = tuple(cluttered["start"][:2])
    goal = tuple(cluttered["goal"][:2])

    placed = [Polygon(obstacle["vertices"]) for obstacle in cluttered["obstacles"]]
    start_buffer = Point(start).buffer(0.5)
    goal_buffer = Point(goal).buffer(0.5)

    attempts = 0
    max_attempts = max(400, extra_obstacles * 80)

    corridor_seed = Point(0, 0).buffer(0.36, resolution=18)
    sx, sy = start
    gx, gy = goal
    corridor_t = 0.38
    corridor_center_x = sx + corridor_t * (gx - sx)
    corridor_center_y = sy + corridor_t * (gy - sy)
    corridor_candidate = translate(corridor_seed, xoff=corridor_center_x, yoff=corridor_center_y)
    if valid_candidate(corridor_candidate, placed, start_buffer, goal_buffer, workspace):
        placed.append(corridor_candidate)
        cluttered["obstacles"].append({
            "id": len(cluttered["obstacles"]),
            "shape_type": "circle",
            "class_true": "movable",
            "true_class": "movable",
            "vertices": polygon_to_list(corridor_candidate),
        })

    while extra_obstacles > 0 and attempts < max_attempts:
        attempts += 1
        radius = random.uniform(0.18, 0.42)
        candidate = Point(0, 0).buffer(radius, resolution=24)

        if random.random() < 0.55:
            t = random.uniform(0.2, 0.8)
            corridor_x = sx + t * (gx - sx)
            corridor_y = sy + t * (gy - sy)
            cx, cy = candidate.centroid.x, candidate.centroid.y
            candidate = translate(candidate, xoff=corridor_x - cx, yoff=corridor_y - cy)

        if not valid_candidate(candidate, placed, start_buffer, goal_buffer, workspace):
            continue

        placed.append(candidate)
        true_class = random.choices(
            ["safe", "movable"],
            weights=[0.45, 0.55],
            k=1,
        )[0]
        cluttered["obstacles"].append({
            "id": len(cluttered["obstacles"]),
            "shape_type": "circle",
            "class_true": true_class,
            "true_class": true_class,
            "vertices": polygon_to_list(candidate),
        })
        extra_obstacles -= 1

    cluttered["family"] = f"{family}_extra_clutter"
    return cluttered


def convert_scene_obstacles_to_circles(scene: dict) -> dict:
    generate_scene, polygon_to_list, valid_candidate = _import_scene_generator()
    """Convert generator output into circles of roughly matched area.

    For an obstacle of area ``A``, the equivalent circle radius is chosen using

        r = sqrt(A / pi),

    then slightly shrunk so neighboring shapes do not become artificially
    larger during conversion.
    """
    converted = copy.deepcopy(scene)
    workspace = tuple(converted["workspace"])
    start = tuple(converted["start"][:2])
    goal = tuple(converted["goal"][:2])
    start_buffer = Point(start).buffer(0.5)
    goal_buffer = Point(goal).buffer(0.5)
    placed: list[Polygon] = []

    for obstacle in converted["obstacles"]:
        original = Polygon(obstacle["vertices"])
        centroid = original.centroid
        radius = max(0.08, float(np.sqrt(original.area / np.pi)) * 0.92)

        candidate = Point(centroid.x, centroid.y).buffer(radius, resolution=24)
        while radius > 0.08 and not valid_candidate(candidate, placed, start_buffer, goal_buffer, workspace):
            radius *= 0.9
            candidate = Point(centroid.x, centroid.y).buffer(radius, resolution=24)

        placed.append(candidate)
        obstacle["shape_type"] = "circle"
        true_class = coerce_pushable_class(
            obstacle.get("true_class", obstacle.get("class_true", "movable"))
        )
        obstacle["true_class"] = true_class
        obstacle["class_true"] = true_class
        obstacle["vertices"] = polygon_to_list(candidate)

    converted["family"] = f"{scene['family']}_circles"
    return converted


def shrink_scene(scene: dict, target_max_span: float = TARGET_MAX_WORKSPACE_SPAN) -> dict:
    """Uniformly rescale a scene to reduce runtime and keep displays compact."""
    shrunk = copy.deepcopy(scene)
    xmin, xmax, ymin, ymax = shrunk["workspace"]
    width = xmax - xmin
    height = ymax - ymin
    current_max_span = max(width, height)
    if current_max_span <= target_max_span:
        return shrunk

    scale = target_max_span / current_max_span

    def scale_point(point_xy):
        return [
            float((point_xy[0] - xmin) * scale),
            float((point_xy[1] - ymin) * scale),
        ]

    shrunk["workspace"] = [0.0, float(width * scale), 0.0, float(height * scale)]
    shrunk["start"] = [
        float((shrunk["start"][0] - xmin) * scale),
        float((shrunk["start"][1] - ymin) * scale),
        float(shrunk["start"][2]),
    ]
    shrunk["goal"] = [
        float((shrunk["goal"][0] - xmin) * scale),
        float((shrunk["goal"][1] - ymin) * scale),
        float(shrunk["goal"][2]),
    ]

    for obstacle in shrunk["obstacles"]:
        obstacle["vertices"] = [scale_point(vertex) for vertex in obstacle["vertices"]]

    return shrunk


def generate_one_random_environment() -> dict:
    generate_scene, polygon_to_list, valid_candidate = _import_scene_generator()
    """Generate one random circle-only environment for the demo."""
    family = random.choice(CLUTTER_BIASED_FAMILIES)
    base_scene = convert_scene_obstacles_to_circles(shrink_scene(generate_scene(family)))
    return create_cluttered_variant(base_scene, extra_obstacles=5)
