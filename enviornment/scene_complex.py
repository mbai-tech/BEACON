import json
import random
from shapely.geometry import Point, box, LineString
from validator import validate_scene

WORKSPACE = (0, 6, 0, 6)   # xmin, xmax, ymin, ymax
CLASSES = ["safe", "movable", "fragile"]


def polygon_to_list(poly):
    coords = list(poly.exterior.coords)[:-1]
    return [[float(x), float(y)] for x, y in coords]


def within_workspace(poly):
    xmin, xmax, ymin, ymax = WORKSPACE
    workspace_poly = box(xmin, ymin, xmax, ymax)
    return workspace_poly.contains(poly)


def valid_candidate(candidate, placed, start_buffer=None, goal_buffer=None, min_gap=0.05):
    if not candidate.is_valid:
        return False
    if not within_workspace(candidate):
        return False
    if start_buffer is not None and candidate.distance(start_buffer) < min_gap:
        return False
    if goal_buffer is not None and candidate.distance(goal_buffer) < min_gap:
        return False
    if any(candidate.distance(p) < min_gap for p in placed):
        return False
    return True


def make_circle_at(x, y, r):
    return Point(x, y).buffer(r, resolution=32)


def make_random_circle(r_min=0.15, r_max=0.45):
    xmin, xmax, ymin, ymax = WORKSPACE
    r = random.uniform(r_min, r_max)
    x = random.uniform(xmin + r, xmax - r)
    y = random.uniform(ymin + r, ymax - r)
    return make_circle_at(x, y, r), r


def random_start_goal(min_dist=3.5, margin=0.45):
    xmin, xmax, ymin, ymax = WORKSPACE

    for _ in range(500):
        sx = random.uniform(xmin + margin, xmax - margin)
        sy = random.uniform(ymin + margin, ymax - margin)
        gx = random.uniform(xmin + margin, xmax - margin)
        gy = random.uniform(ymin + margin, ymax - margin)

        if ((sx - gx) ** 2 + (sy - gy) ** 2) ** 0.5 >= min_dist:
            return (sx, sy), (gx, gy)

    return (0.6, 0.6), (5.4, 5.4)


def obstacle_record(poly, idx, cls):
    center = poly.centroid
    radius = ((poly.area / 3.1415926535) ** 0.5)
    return {
        "id": idx,
        "shape_type": "circle",
        "class_true": cls,
        "radius": round(radius, 4),
        "center": [round(center.x, 4), round(center.y, 4)],
        "vertices": polygon_to_list(poly)
    }

def try_add_obstacle(poly, cls, placed, obstacles, start=None, goal=None, min_gap=0.05):
    start_buffer = Point(start).buffer(0.35) if start is not None else None
    goal_buffer = Point(goal).buffer(0.35) if goal is not None else None

    if valid_candidate(poly, placed, start_buffer, goal_buffer, min_gap=min_gap):
        placed.append(poly)
        obstacles.append(obstacle_record(poly, len(obstacles), cls))
        return True
    return False


def sample_background_obstacles(
    n_min, n_max, start, goal, placed=None, class_weights=None
):
    if placed is None:
        placed = []

    if class_weights is None:
        class_weights = {
            "safe": 0.2,
            "movable": 0.6,
            "fragile": 0.2
        }

    classes = list(class_weights.keys())
    weights = list(class_weights.values())

    start_buffer = Point(start).buffer(0.35)
    goal_buffer = Point(goal).buffer(0.35)

    n_obs = random.randint(n_min, n_max)
    obstacles = []
    attempts = 0
    max_attempts = 10000

    while len(obstacles) < n_obs and attempts < max_attempts:
        attempts += 1
        candidate, _ = make_random_circle()

        if valid_candidate(candidate, placed, start_buffer, goal_buffer, min_gap=0.05):
            placed.append(candidate)
            cls = random.choices(classes, weights=weights, k=1)[0]
            obstacles.append(obstacle_record(candidate, len(obstacles), cls))

    return placed, obstacles


def generate_sparse(seed=None):
    if seed is not None:
        random.seed(seed)

    start, goal = random_start_goal()
    _, obstacles = sample_background_obstacles(8, 15, start, goal)

    return {
        "family": "sparse",
        "workspace": WORKSPACE,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles
    }


def generate_cluttered(seed=None):
    if seed is not None:
        random.seed(seed)

    start, goal = random_start_goal()
    _, obstacles = sample_background_obstacles(25, 40, start, goal)

    return {
        "family": "cluttered",
        "workspace": WORKSPACE,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles
    }


def generate_collision_required(seed=None):
    if seed is not None:
        random.seed(seed)

    start, goal = random_start_goal(min_dist=4.0)
    placed = []
    obstacles = []

    line = LineString([start, goal])
    midpoint = line.interpolate(0.5, normalized=True)

    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length == 0:
        dx, dy, length = 1.0, 0.0, 1.0
    px = -dy / length
    py = dx / length

    band_offsets = [-0.9, -0.3, 0.3, 0.9]
    band_radii = [0.34, 0.38, 0.38, 0.34]

    for offset, r in zip(band_offsets, band_radii):
        x = midpoint.x + px * offset
        y = midpoint.y + py * offset
        poly = make_circle_at(x, y, r)
        cls = "movable" if abs(offset) < 0.5 else "fragile"
        try_add_obstacle(poly, cls, placed, obstacles, start, goal, min_gap=0.05)

    extra_offsets = [-1.35, 1.35]
    for offset in extra_offsets:
        x = midpoint.x + px * offset
        y = midpoint.y + py * offset
        poly = make_circle_at(x, y, 0.30)
        try_add_obstacle(poly, "safe", placed, obstacles, start, goal, min_gap=0.05)

    placed, bg = sample_background_obstacles(
        6, 10, start, goal, placed=placed,
        class_weights={"safe": 0.2, "movable": 0.6, "fragile": 0.2}
    )

    for obs in bg:
        obs["id"] = len(obstacles)
        obstacles.append(obs)

    return {
        "family": "collision_required",
        "workspace": WORKSPACE,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles
    }


def generate_collision_shortcut(seed=None):
    if seed is not None:
        random.seed(seed)

    start, goal = random_start_goal(min_dist=4.0)
    placed = []
    obstacles = []

    line = LineString([start, goal])
    fractions = [0.35, 0.5, 0.65]

    for frac in fractions:
        p = line.interpolate(frac, normalized=True)
        r = random.uniform(0.28, 0.40)
        poly = make_circle_at(p.x, p.y, r)
        try_add_obstacle(poly, "movable", placed, obstacles, start, goal, min_gap=0.05)

    placed, bg = sample_background_obstacles(
        14, 22, start, goal, placed=placed,
        class_weights={"safe": 0.25, "movable": 0.5, "fragile": 0.25}
    )

    for obs in bg:
        obs["id"] = len(obstacles)
        obstacles.append(obs)

    return {
        "family": "collision_shortcut",
        "workspace": WORKSPACE,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles
    }

# Without validator
def generate_scene(family="sparse", seed=None):
    if family == "sparse":
        return generate_sparse(seed=seed)
    if family == "cluttered":
        return generate_cluttered(seed=seed)
    if family == "collision_required":
        return generate_collision_required(seed=seed)
    if family == "collision_shortcut":
        return generate_collision_shortcut(seed=seed)

    raise ValueError(
        "family must be one of: "
        "'sparse', 'cluttered', 'collision_required', 'collision_shortcut'"
    )


# WITH VALIDATOR.PY
# def generate_scene(family="sparse", seed=None, validate=True, max_tries=200):
#     for attempt in range(max_tries):
#         current_seed = seed if seed is not None and attempt == 0 else random.randrange(0, 10**9)

#         if family == "sparse":
#             scene = generate_sparse(seed=current_seed)
#         elif family == "cluttered":
#             scene = generate_cluttered(seed=current_seed)
#         elif family == "collision_required":
#             scene = generate_collision_required(seed=current_seed)
#         elif family == "collision_shortcut":
#             scene = generate_collision_shortcut(seed=current_seed)
#         else:
#             raise ValueError(
#                 "family must be one of: "
#                 "'sparse', 'cluttered', 'collision_required', 'collision_shortcut'"
#             )

#         scene["seed"] = current_seed

#         if not validate:
#             return scene

#         validation = validate_scene(scene)
#         scene["validation"] = validation

#         if validation["valid"]:
#             return scene

#     raise RuntimeError(f"Could not generate a valid scene for family '{family}' after {max_tries} tries")


def save_scene_json(scene, path):
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)