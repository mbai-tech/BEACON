import json
import math
import random
import copy
from shapely.geometry import Polygon, Point, box
from shapely.affinity import translate, rotate

DEFAULT_WORKSPACE = (0, 10, 0, 10)
WORKSPACES = {
    "narrow_passage": (0, 10, 0, 4),
    "sparse_clutter": (0, 12, 0, 12),
    "dense_clutter": (0, 9, 0, 9),
    "semantic_trap": (0, 11, 0, 11),
    "perturbed": (0, 10, 0, 10),
}
OBSTACLE_COUNT_RANGES = {
    "sparse_clutter": (2, 7),
    "dense_clutter": (14, 22),
    "narrow_passage": (9, 13),
    "semantic_trap": (7, 11),
}
SHAPE_SIZE_PROFILES = {
    "sparse_clutter": {
        "rectangle_w": (0.6, 2.2),
        "rectangle_h": (0.6, 2.2),
        "triangle_w": (0.7, 2.1),
        "triangle_h": (0.7, 2.0),
        "trapezoid_bottom": (1.0, 2.4),
        "trapezoid_top_ratio": (0.45, 0.9),
        "trapezoid_h": (0.7, 1.9),
        "parallelogram_w": (1.0, 2.4),
        "parallelogram_h": (0.7, 1.8),
        "parallelogram_skew": (0.25, 0.95),
        "circle_r": (0.35, 1.0),
    },
    "dense_clutter": {
        "rectangle_w": (0.35, 1.1),
        "rectangle_h": (0.35, 1.1),
        "triangle_w": (0.4, 1.0),
        "triangle_h": (0.4, 1.0),
        "trapezoid_bottom": (0.5, 1.3),
        "trapezoid_top_ratio": (0.5, 0.85),
        "trapezoid_h": (0.4, 1.0),
        "parallelogram_w": (0.5, 1.4),
        "parallelogram_h": (0.35, 1.0),
        "parallelogram_skew": (0.15, 0.5),
        "circle_r": (0.2, 0.55),
    },
    "narrow_passage": {
        "rectangle_w": (0.3, 0.9),
        "rectangle_h": (0.3, 1.1),
        "triangle_w": (0.35, 0.9),
        "triangle_h": (0.35, 0.9),
        "trapezoid_bottom": (0.45, 1.0),
        "trapezoid_top_ratio": (0.55, 0.9),
        "trapezoid_h": (0.35, 0.9),
        "parallelogram_w": (0.45, 1.0),
        "parallelogram_h": (0.35, 0.9),
        "parallelogram_skew": (0.1, 0.35),
        "circle_r": (0.18, 0.45),
    },
    "semantic_trap": {
        "rectangle_w": (0.45, 1.4),
        "rectangle_h": (0.45, 1.4),
        "triangle_w": (0.5, 1.3),
        "triangle_h": (0.5, 1.3),
        "trapezoid_bottom": (0.7, 1.6),
        "trapezoid_top_ratio": (0.45, 0.85),
        "trapezoid_h": (0.45, 1.3),
        "parallelogram_w": (0.7, 1.7),
        "parallelogram_h": (0.45, 1.2),
        "parallelogram_skew": (0.2, 0.6),
        "circle_r": (0.25, 0.75),
    },
    "perturbed": {
        "rectangle_w": (0.45, 1.4),
        "rectangle_h": (0.45, 1.4),
        "triangle_w": (0.5, 1.3),
        "triangle_h": (0.5, 1.3),
        "trapezoid_bottom": (0.7, 1.6),
        "trapezoid_top_ratio": (0.45, 0.85),
        "trapezoid_h": (0.45, 1.3),
        "parallelogram_w": (0.7, 1.7),
        "parallelogram_h": (0.45, 1.2),
        "parallelogram_skew": (0.2, 0.6),
        "circle_r": (0.25, 0.75),
    },
}
PERTURBATION_CHANGE_RANGE = (1, 3)
CLASSES = ["safe", "movable", "fragile", "forbidden"]


# -----------------------------
# Basic helpers
# -----------------------------
def polygon_to_list(poly):
    coords = list(poly.exterior.coords)[:-1]
    return [[float(x), float(y)] for x, y in coords]


def random_class():
    return random.choice(CLASSES)


def get_workspace(family):
    return WORKSPACES.get(family, DEFAULT_WORKSPACE)


def get_obstacle_count_range(family):
    return OBSTACLE_COUNT_RANGES[family]


def get_shape_size_profile(family):
    return SHAPE_SIZE_PROFILES[family]


def random_pose_point(workspace, margin=0.6):
    x = random.uniform(workspace[0] + margin, workspace[1] - margin)
    y = random.uniform(workspace[2] + margin, workspace[3] - margin)
    return (x, y)


def make_start_goal(workspace, family):
    xmin, xmax, ymin, ymax = workspace
    width = xmax - xmin
    height = ymax - ymin
    layout_by_family = {
        "sparse_clutter": ((0.10, 0.10), (0.90, 0.90)),
        "dense_clutter": ((0.12, 0.15), (0.88, 0.85)),
        "narrow_passage": ((0.15, 0.50), (0.85, 0.50)),
        "semantic_trap": ((0.12, 0.18), (0.88, 0.82)),
        "perturbed": ((0.10, 0.10), (0.90, 0.90)),
    }
    start_ratio, goal_ratio = layout_by_family.get(
        family, ((0.10, 0.10), (0.90, 0.90))
    )
    start = (
        xmin + start_ratio[0] * width,
        ymin + start_ratio[1] * height,
    )
    goal = (
        xmin + goal_ratio[0] * width,
        ymin + goal_ratio[1] * height,
    )
    return start, goal


def within_workspace(poly, workspace):
    xmin, xmax, ymin, ymax = workspace
    workspace_poly = box(xmin, ymin, xmax, ymax)
    return workspace_poly.contains(poly)


def valid_candidate(candidate, placed, start_buffer, goal_buffer, workspace):
    if not candidate.is_valid:
        return False
    if not within_workspace(candidate, workspace):
        return False
    if candidate.intersects(start_buffer) or candidate.intersects(goal_buffer):
        return False
    if any(candidate.intersects(p) for p in placed):
        return False
    return True


def make_rectangle(size_profile):
    w = random.uniform(*size_profile["rectangle_w"])
    h = random.uniform(*size_profile["rectangle_h"])
    poly = Polygon([(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)])
    return poly


def make_triangle(size_profile):
    w = random.uniform(*size_profile["triangle_w"])
    h = random.uniform(*size_profile["triangle_h"])
    poly = Polygon([
        (-w/2, -h/2),
        (w/2, -h/2),
        (random.uniform(-w/4, w/4), h/2)
    ])
    return poly


def make_trapezoid(size_profile):
    bottom = random.uniform(*size_profile["trapezoid_bottom"])
    top_ratio = random.uniform(*size_profile["trapezoid_top_ratio"])
    top = bottom * top_ratio
    h = random.uniform(*size_profile["trapezoid_h"])
    shift = random.uniform(-0.3, 0.3)
    poly = Polygon([
        (-bottom/2, -h/2),
        (bottom/2, -h/2),
        (shift + top/2, h/2),
        (shift - top/2, h/2)
    ])
    return poly


def make_parallelogram(size_profile):
    w = random.uniform(*size_profile["parallelogram_w"])
    h = random.uniform(*size_profile["parallelogram_h"])
    skew = random.uniform(*size_profile["parallelogram_skew"])
    poly = Polygon([
        (-w/2, -h/2),
        (w/2, -h/2),
        (w/2 + skew, h/2),
        (-w/2 + skew, h/2)
    ])
    return poly


def make_circle_polygon(size_profile):
    r = random.uniform(*size_profile["circle_r"])
    # buffered point gives a polygon approximation of a circle
    return Point(0, 0).buffer(r, resolution=24)


def make_random_shape(workspace, family):
    size_profile = get_shape_size_profile(family)
    shape_type = random.choice([
        "rectangle", "triangle", "trapezoid", "parallelogram", "circle"
    ])

    if shape_type == "rectangle":
        poly = make_rectangle(size_profile)
    elif shape_type == "triangle":
        poly = make_triangle(size_profile)
    elif shape_type == "trapezoid":
        poly = make_trapezoid(size_profile)
    elif shape_type == "parallelogram":
        poly = make_parallelogram(size_profile)
    else:
        poly = make_circle_polygon(size_profile)

    angle = random.uniform(0, 180)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)

    minx, miny, maxx, maxy = poly.bounds
    x = random.uniform(workspace[0] - minx + 0.1, workspace[1] - maxx - 0.1)
    y = random.uniform(workspace[2] - miny + 0.1, workspace[3] - maxy - 0.1)
    poly = translate(poly, xoff=x, yoff=y)

    return shape_type, poly

# -----------------------------
# Common obstacle sampler
# -----------------------------
def sample_random_obstacles(n_min, n_max, start, goal, workspace, family):
    n_obs = random.randint(n_min, n_max)
    placed = []
    obstacles = []

    start_buffer = Point(start).buffer(0.5)
    goal_buffer = Point(goal).buffer(0.5)

    attempts = 0
    max_attempts = 3000

    while len(placed) < n_obs and attempts < max_attempts:
        attempts += 1
        shape_type, candidate = make_random_shape(workspace, family)

        if valid_candidate(candidate, placed, start_buffer, goal_buffer, workspace):
            placed.append(candidate)
            obstacles.append({
                "id": len(obstacles),
                "shape_type": shape_type,
                "class_true": random_class(), # TODO: add the colors, can prob do in the method idk you're choice
                "vertices": polygon_to_list(candidate)
            })

    return placed, obstacles


# -----------------------------
# Family generators
# -----------------------------
# TODO: This is up to you, but you can add functionality to change the range for the size of shapes based on 
def generate_sparse_clutter():
    family = "sparse_clutter"
    workspace = get_workspace(family)
    start, goal = make_start_goal(workspace, family)
    obstacle_range = get_obstacle_count_range(family)
    _, obstacles = sample_random_obstacles(
        obstacle_range[0], obstacle_range[1], start, goal, workspace, family
    )

    return {
        "family": family,
        "workspace": workspace,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles
    }


def generate_dense_clutter():
    family = "dense_clutter"
    workspace = get_workspace(family)
    start, goal = make_start_goal(workspace, family)
    obstacle_range = get_obstacle_count_range(family)
    _, obstacles = sample_random_obstacles(
        obstacle_range[0], obstacle_range[1], start, goal, workspace, family
    )

    return {
        "family": family,
        "workspace": workspace,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles
    }


def generate_narrow_passage():
    family = "narrow_passage"
    workspace = get_workspace(family)
    start, goal = make_start_goal(workspace, family)
    obstacles = []

    # Two rectangles fully contained in the 10x4 workspace leave a narrow gap at y=2.
    left_wall = Polygon([(4.2, 0.4), (4.9, 0.4), (4.9, 1.8), (4.2, 1.8)])
    right_wall = Polygon([(4.2, 2.2), (4.9, 2.2), (4.9, 3.6), (4.2, 3.6)])

    fixed = [
        ("rectangle", left_wall, "forbidden"),
        ("rectangle", right_wall, "forbidden"),
    ]

    for i, (shape_type, poly, cls) in enumerate(fixed):
        obstacles.append({
            "id": i,
            "shape_type": shape_type,
            "class_true": cls,
            "vertices": polygon_to_list(poly)
        })

    # add more clutter around the passage
    placed = [left_wall, right_wall]
    start_buffer = Point(start).buffer(0.5)
    goal_buffer = Point(goal).buffer(0.5)
    obstacle_range = get_obstacle_count_range(family)
    target_total = random.randint(obstacle_range[0], obstacle_range[1])

    attempts = 0
    while len(obstacles) < target_total and attempts < 2000:
        attempts += 1
        shape_type, candidate = make_random_shape(workspace, family)
        if valid_candidate(candidate, placed, start_buffer, goal_buffer, workspace):
            placed.append(candidate)
            obstacles.append({
                "id": len(obstacles),
                "shape_type": shape_type,
                "class_true": random_class(),
                "vertices": polygon_to_list(candidate)
            })

    return {
        "family": family,
        "workspace": workspace,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles
    }


def generate_semantic_trap():
    family = "semantic_trap"
    workspace = get_workspace(family)
    start, goal = make_start_goal(workspace, family)
    obstacles = []

    # tempting central object labeled misleadingly in later versions if needed
    xmin, xmax, ymin, ymax = workspace
    trap_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    trap_poly = Point(*trap_center).buffer(0.8, resolution=24)

    obstacles.append({
        "id": 0,
        "shape_type": "circle",
        "class_true": "fragile",
        "vertices": polygon_to_list(trap_poly)
    })

    placed = [trap_poly]
    start_buffer = Point(start).buffer(0.5)
    goal_buffer = Point(goal).buffer(0.5)

    attempts = 0
    obstacle_range = get_obstacle_count_range(family)
    target_total = random.randint(obstacle_range[0], obstacle_range[1])
    while len(obstacles) < target_total and attempts < 2500:
        attempts += 1
        shape_type, candidate = make_random_shape(workspace, family)
        if valid_candidate(candidate, placed, start_buffer, goal_buffer, workspace):
            placed.append(candidate)
            obstacles.append({
                "id": len(obstacles),
                "shape_type": shape_type,
                "class_true": random_class(),
                "vertices": polygon_to_list(candidate)
            })

    return {
        "family": family,
        "workspace": workspace,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles
    }


def generate_perturbed():
    # Start from a clean base scene
    base = copy.deepcopy(random.choice([
        generate_sparse_clutter(),
        generate_dense_clutter()
    ]))

    start = tuple(base["start"][:2])
    goal = tuple(base["goal"][:2])
    start_buffer = Point(start).buffer(0.3)
    goal_buffer = Point(goal).buffer(0.3)

    num_changes = min(
        random.randint(PERTURBATION_CHANGE_RANGE[0], PERTURBATION_CHANGE_RANGE[1]),
        len(base["obstacles"])
    )
    chosen_ids = random.sample(range(len(base["obstacles"])), num_changes)

    # Convert all obstacles to shapely polygons
    polygons = [Polygon(obs["vertices"]) for obs in base["obstacles"]]

    for idx in chosen_ids:
        original_poly = polygons[idx]
        moved = False

        for _ in range(50):  # try up to 50 random shifts
            dx = random.uniform(-0.4, 0.4)
            dy = random.uniform(-0.4, 0.4)
            candidate = translate(original_poly, xoff=dx, yoff=dy)

            # validity checks
            if not candidate.is_valid:
                continue
            if not within_workspace(candidate, tuple(base["workspace"])):
                continue
            if candidate.intersects(start_buffer) or candidate.intersects(goal_buffer):
                continue

            collision = False
            for j, other_poly in enumerate(polygons):
                if j == idx:
                    continue
                if candidate.intersects(other_poly):
                    collision = True
                    break

            if collision:
                continue

            # accept move
            polygons[idx] = candidate
            base["obstacles"][idx]["vertices"] = polygon_to_list(candidate)
            moved = True
            break

        # if no valid move found, keep original obstacle unchanged

    base["family"] = "perturbed"
    return base


def generate_scene(family, seed=None):
    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**32)

    previous_state = random.getstate()
    random.seed(seed)

    try:
        if family == "sparse_clutter":
            scene = generate_sparse_clutter()
        elif family == "dense_clutter":
            scene = generate_dense_clutter()
        elif family == "narrow_passage":
            scene = generate_narrow_passage()
        elif family == "semantic_trap":
            scene = generate_semantic_trap()
        elif family == "perturbed":
            scene = generate_perturbed()
        else:
            raise ValueError(f"Unknown family: {family}")
    finally:
        random.setstate(previous_state)

    scene["seed"] = seed
    return scene


# -----------------------------
# Save JSON
# -----------------------------
def save_scene_json(scene, path):
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)
