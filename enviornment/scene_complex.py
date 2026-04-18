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


def try_add_obstacle(poly, cls, placed, obstacles, start=None, goal=None):
    start_buffer = Point(start).buffer(0.35) if start is not None else None
    goal_buffer = Point(goal).buffer(0.35) if goal is not None else None

    if valid_candidate(poly, placed, start_buffer, goal_buffer):
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

        if valid_candidate(candidate, placed, start_buffer, goal_buffer):
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

    start, goal = random_start_goal(min_dist=4.4)
    placed = []
    obstacles = []

    line = LineString([start, goal])
    midpoint = line.interpolate(0.5, normalized=True)

    # Unit direction from start to goal
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length == 0:
        dx, dy, length = 1.0, 0.0, 1.0

    ux = dx / length
    uy = dy / length

    # Unit perpendicular direction
    px = -dy / length
    py = dx / length

    # --------------------------------------------------
    # FULL EXTENDED CLUTTER WALL
    # Long across the route, thick along the route
    # --------------------------------------------------
    row_offsets = [-0.95, -0.60, -0.25, 0.10, 0.45, 0.80]
    col_offsets = [-2.35, -1.90, -1.45, -1.00, -0.55, -0.10,
                    0.35, 0.80, 1.25, 1.70, 2.15]

    for row in row_offsets:
        for col in col_offsets:
            # jitter so it looks cluttered, not like a grid
            jitter_u = random.uniform(-0.08, 0.08)
            jitter_p = random.uniform(-0.08, 0.08)

            x = midpoint.x + ux * (row + jitter_u) + px * (col + jitter_p)
            y = midpoint.y + uy * (row + jitter_u) + py * (col + jitter_p)

            r = random.uniform(0.20, 0.28)
            poly = make_circle_at(x, y, r)

            if not within_workspace(poly):
                continue

            # Make the core mostly movable so collision through the wall is useful.
            # Outer regions are more fragile to discourage just smashing anywhere.
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

            try_add_obstacle(poly, cls, placed, obstacles, start, goal)

    # --------------------------------------------------
    # SIDE EXTENSIONS / CAPS
    # Makes it harder to sneak around the ends
    # --------------------------------------------------
    cap_points = [
        (-2.75, -0.55), (-2.75, -0.15), (-2.75, 0.25), (-2.75, 0.65),
        ( 2.55, -0.50), ( 2.55, -0.10), ( 2.55, 0.30), ( 2.55, 0.70)
    ]

    for col, row in cap_points:
        x = midpoint.x + ux * row + px * col
        y = midpoint.y + uy * row + py * col
        r = random.uniform(0.18, 0.25)
        poly = make_circle_at(x, y, r)

        if within_workspace(poly):
            try_add_obstacle(poly, "fragile", placed, obstacles, start, goal)

    # --------------------------------------------------
    # EXTRA OBSTACLES OUTSIDE THE WALL
    # So the whole scene does not look empty except the barrier
    # --------------------------------------------------
    start_buffer = Point(start).buffer(0.35)
    goal_buffer = Point(goal).buffer(0.35)

    extra_targets = random.randint(10, 16)
    attempts = 0
    max_attempts = 12000

    while extra_targets > 0 and attempts < max_attempts:
        attempts += 1

        candidate, _ = make_random_circle(r_min=0.14, r_max=0.24)
        c = candidate.centroid

        # Signed coordinates relative to the wall frame
        rel_x = c.x - midpoint.x
        rel_y = c.y - midpoint.y
        along = rel_x * ux + rel_y * uy
        across = rel_x * px + rel_y * py

        # keep these extra obstacles mostly outside the main wall region
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
            obstacles.append(obstacle_record(candidate, len(obstacles), cls))
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
        r = random.uniform(0.28, 0.40)
        poly = make_circle_at(p.x, p.y, r)
        try_add_obstacle(poly, "movable", placed, obstacles, start, goal)

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