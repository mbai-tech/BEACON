import json
import math
import random
import sys
from pathlib import Path
from shapely.geometry import Point, Polygon, box, LineString
from shapely.affinity import translate, rotate

from beacon.environment.validator import validate_scene

WORKSPACE = (0, 6, 0, 6)   # xmin, xmax, ymin, ymax
CLASSES = ["safe", "movable", "fragile"]
SHAPE_TYPES = ["circle", "rectangle", "triangle", "trapezoid",
               "parallelogram", "pentagon", "hexagon"]


def polygon_to_list(poly):
    coords = list(poly.exterior.coords)[:-1]
    return [[float(x), float(y)] for x, y in coords]


def within_workspace(poly):
    xmin, xmax, ymin, ymax = WORKSPACE
    workspace_poly = box(xmin, ymin, xmax, ymax)
    return workspace_poly.contains(poly)


def valid_candidate(candidate, placed, start_buffer=None, goal_buffer=None):
    if not candidate.is_valid:
        return False
    if not within_workspace(candidate):
        return False

    # touching allowed, overlap forbidden
    if start_buffer is not None and candidate.intersection(start_buffer).area > 1e-9:
        return False
    if goal_buffer is not None and candidate.intersection(goal_buffer).area > 1e-9:
        return False

    for p in placed:
        if candidate.intersection(p).area > 1e-9:
            return False

    return True


def make_circle_at(x, y, r):
    return Point(x, y).buffer(r, resolution=32)


def make_rectangle(w, h):
    return Polygon([
        (-w / 2, -h / 2),
        (w / 2, -h / 2),
        (w / 2, h / 2),
        (-w / 2, h / 2)
    ])


def make_triangle(w, h):
    return Polygon([
        (-w / 2, -h / 2),
        (w / 2, -h / 2),
        (0, h / 2)
    ])


def make_trapezoid(bottom, top, h):
    return Polygon([
        (-bottom / 2, -h / 2),
        (bottom / 2, -h / 2),
        (top / 2, h / 2),
        (-top / 2, h / 2)
    ])


def make_parallelogram(w, h, skew):
    return Polygon([
        (-w / 2, -h / 2),
        (w / 2, -h / 2),
        (w / 2 + skew, h / 2),
        (-w / 2 + skew, h / 2)
    ])


def make_regular_ngon(n, r):
    pts = []
    for i in range(n):
        ang = 2 * math.pi * i / n
        pts.append((r * math.cos(ang), r * math.sin(ang)))
    return Polygon(pts)


def make_shape(shape_type, scale_min=0.15, scale_max=0.45):
    if shape_type == "circle":
        r = random.uniform(scale_min, scale_max)
        return Point(0, 0).buffer(r, resolution=32)

    if shape_type == "rectangle":
        w = random.uniform(scale_min * 1.5, scale_max * 2.2)
        h = random.uniform(scale_min * 1.2, scale_max * 1.8)
        return make_rectangle(w, h)

    if shape_type == "triangle":
        w = random.uniform(scale_min * 1.6, scale_max * 2.1)
        h = random.uniform(scale_min * 1.4, scale_max * 1.9)
        return make_triangle(w, h)

    if shape_type == "trapezoid":
        bottom = random.uniform(scale_min * 1.6, scale_max * 2.3)
        top = bottom * random.uniform(0.45, 0.8)
        h = random.uniform(scale_min * 1.2, scale_max * 1.8)
        return make_trapezoid(bottom, top, h)

    if shape_type == "parallelogram":
        w = random.uniform(scale_min * 1.6, scale_max * 2.2)
        h = random.uniform(scale_min * 1.1, scale_max * 1.7)
        skew = random.uniform(0.08, 0.28)
        return make_parallelogram(w, h, skew)

    if shape_type == "pentagon":
        r = random.uniform(scale_min, scale_max)
        return make_regular_ngon(5, r)

    if shape_type == "hexagon":
        r = random.uniform(scale_min, scale_max)
        return make_regular_ngon(6, r)

    raise ValueError(f"Unknown shape_type: {shape_type}")


def place_shape_at(poly, x, y, angle_deg=None):
    if angle_deg is None:
        angle_deg = random.uniform(0, 180)
    poly = rotate(poly, angle_deg, origin=(0, 0), use_radians=False)
    poly = translate(poly, xoff=x, yoff=y)
    return poly


def make_random_polygon(scale_min=0.15, scale_max=0.45):
    xmin, xmax, ymin, ymax = WORKSPACE

    shape_type = random.choice(SHAPE_TYPES)
    base_poly = make_shape(shape_type, scale_min=scale_min, scale_max=scale_max)

    minx, miny, maxx, maxy = base_poly.bounds
    x = random.uniform(xmin - minx, xmax - maxx)
    y = random.uniform(ymin - miny, ymax - maxy)

    poly = place_shape_at(base_poly, x, y)
    return poly, shape_type


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


def obstacle_record(poly, idx, cls, shape_type):
    center = poly.centroid
    return {
        "id": idx,
        "shape_type": shape_type,
        "class_true": cls,
        "center": [round(center.x, 4), round(center.y, 4)],
        "vertices": polygon_to_list(poly)
    }


def try_add_obstacle(poly, cls, shape_type, placed, obstacles, start=None, goal=None):
    start_buffer = Point(start).buffer(0.35) if start is not None else None
    goal_buffer = Point(goal).buffer(0.35) if goal is not None else None

    if valid_candidate(poly, placed, start_buffer, goal_buffer):
        placed.append(poly)
        obstacles.append(obstacle_record(poly, len(obstacles), cls, shape_type))
        return True
    return False


def sample_background_obstacles(
    n_min, n_max, start, goal, placed=None, class_weights=None,
    scale_min=0.15, scale_max=0.45
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
    max_attempts = 15000

    while len(obstacles) < n_obs and attempts < max_attempts:
        attempts += 1
        candidate, shape_type = make_random_polygon(
            scale_min=scale_min, scale_max=scale_max
        )

        if valid_candidate(candidate, placed, start_buffer, goal_buffer):
            placed.append(candidate)
            cls = random.choices(classes, weights=weights, k=1)[0]
            obstacles.append(
                obstacle_record(candidate, len(obstacles), cls, shape_type)
            )

    return placed, obstacles


def generate_sparse(seed=None):
    if seed is not None:
        random.seed(seed)

    start, goal = random_start_goal()
    _, obstacles = sample_background_obstacles(
        8, 15, start, goal, scale_min=0.14, scale_max=0.32
    )

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
    _, obstacles = sample_background_obstacles(
        25, 40, start, goal, scale_min=0.12, scale_max=0.24
    )

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

    start, goal = random_start_goal(min_dist=4.4)
    placed = []
    obstacles = []

    line = LineString([start, goal])
    midpoint = line.interpolate(0.5, normalized=True)

    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length == 0:
        dx, dy, length = 1.0, 0.0, 1.0

    ux = dx / length
    uy = dy / length
    px = -dy / length
    py = dx / length

    row_offsets = [-0.95, -0.60, -0.25, 0.10, 0.45, 0.80]
    col_offsets = [-2.35, -1.90, -1.45, -1.00, -0.55, -0.10,
                   0.35, 0.80, 1.25, 1.70, 2.15]

    for row in row_offsets:
        for col in col_offsets:
            jitter_u = random.uniform(-0.08, 0.08)
            jitter_p = random.uniform(-0.08, 0.08)

            x = midpoint.x + ux * (row + jitter_u) + px * (col + jitter_p)
            y = midpoint.y + uy * (row + jitter_u) + py * (col + jitter_p)

            shape_type = random.choice(SHAPE_TYPES)
            base = make_shape(shape_type, scale_min=0.16, scale_max=0.26)
            poly = place_shape_at(base, x, y)

            if not within_workspace(poly):
                continue

            if abs(col) <= 0.9 and abs(row) <= 0.55:
                cls = random.choices(
                    ["movable", "fragile", "safe"],
                    weights=[0.70, 0.20, 0.10],
                    k=1
                )[0]
            elif abs(col) <= 1.6 and abs(row) <= 0.75:
                cls = random.choices(
                    ["movable", "fragile", "safe"],
                    weights=[0.45, 0.40, 0.15],
                    k=1
                )[0]
            else:
                cls = random.choices(
                    ["fragile", "movable", "safe"],
                    weights=[0.60, 0.25, 0.15],
                    k=1
                )[0]

            try_add_obstacle(poly, cls, shape_type, placed, obstacles, start, goal)

    cap_points = [
        (-2.75, -0.55), (-2.75, -0.15), (-2.75, 0.25), (-2.75, 0.65),
        (2.55, -0.50), (2.55, -0.10), (2.55, 0.30), (2.55, 0.70)
    ]

    for col, row in cap_points:
        x = midpoint.x + ux * row + px * col
        y = midpoint.y + uy * row + py * col
        shape_type = random.choice(SHAPE_TYPES)
        base = make_shape(shape_type, scale_min=0.14, scale_max=0.22)
        poly = place_shape_at(base, x, y)

        if within_workspace(poly):
            try_add_obstacle(poly, "fragile", shape_type, placed, obstacles, start, goal)

    start_buffer = Point(start).buffer(0.35)
    goal_buffer = Point(goal).buffer(0.35)

    extra_targets = random.randint(10, 16)
    attempts = 0
    max_attempts = 16000

    while extra_targets > 0 and attempts < max_attempts:
        attempts += 1

        candidate, shape_type = make_random_polygon(scale_min=0.12, scale_max=0.22)
        c = candidate.centroid

        rel_x = c.x - midpoint.x
        rel_y = c.y - midpoint.y
        along = rel_x * ux + rel_y * uy
        across = rel_x * px + rel_y * py

        in_main_wall_region = (abs(across) < 2.6 and abs(along) < 1.1)
        if in_main_wall_region:
            continue

        if valid_candidate(candidate, placed, start_buffer, goal_buffer):
            placed.append(candidate)
            cls = random.choices(
                ["safe", "movable", "fragile"],
                weights=[0.25, 0.35, 0.40],
                k=1
            )[0]
            obstacles.append(
                obstacle_record(candidate, len(obstacles), cls, shape_type)
            )
            extra_targets -= 1

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
        shape_type = random.choice(SHAPE_TYPES)
        base = make_shape(shape_type, scale_min=0.18, scale_max=0.30)
        poly = place_shape_at(base, p.x, p.y)
        try_add_obstacle(poly, "movable", shape_type, placed, obstacles, start, goal)

    placed, bg = sample_background_obstacles(
        14, 22, start, goal, placed=placed,
        class_weights={"safe": 0.25, "movable": 0.5, "fragile": 0.25},
        scale_min=0.12, scale_max=0.24
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


def save_scene_json(scene, path):
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)
