import json
import math
import random
import copy
from shapely.geometry import Polygon, Point, box
from shapely.affinity import translate, rotate

WORKSPACE = (0, 10, 0, 10)  # TODO: write method to change this based on the various env types
CLASSES = ["safe", "movable", "fragile", "forbidden"]


# -----------------------------
# Basic helpers
# -----------------------------
def polygon_to_list(poly):
    coords = list(poly.exterior.coords)[:-1]
    return [[float(x), float(y)] for x, y in coords]


def random_class():
    return random.choice(CLASSES)


def random_pose_point(margin=0.6):
    x = random.uniform(WORKSPACE[0] + margin, WORKSPACE[1] - margin)
    y = random.uniform(WORKSPACE[2] + margin, WORKSPACE[3] - margin)
    return (x, y)

# TODO: change start, end goal based on env
def make_start_goal():
    return (1.0, 1.0), (9.0, 9.0)


def within_workspace(poly):
    xmin, xmax, ymin, ymax = WORKSPACE
    workspace_poly = box(xmin, ymin, xmax, ymax)
    return workspace_poly.contains(poly)


def valid_candidate(candidate, placed, start_buffer, goal_buffer):
    if not candidate.is_valid:
        return False
    if not within_workspace(candidate):
        return False
    if candidate.intersects(start_buffer) or candidate.intersects(goal_buffer):
        return False
    if any(candidate.intersects(p) for p in placed):
        return False
    return True


# -----------------------------
# Random shape generators
# -----------------------------
def make_rectangle():
    w = random.uniform(0.4, 1.8)
    h = random.uniform(0.4, 1.8)
    poly = Polygon([(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)])
    return poly


def make_triangle():
    w = random.uniform(0.5, 1.8)
    h = random.uniform(0.5, 1.8)
    poly = Polygon([
        (-w/2, -h/2),
        (w/2, -h/2),
        (random.uniform(-w/4, w/4), h/2)
    ])
    return poly


def make_trapezoid():
    bottom = random.uniform(0.8, 2.0)
    top = random.uniform(0.4, bottom * 0.9)
    h = random.uniform(0.5, 1.6)
    shift = random.uniform(-0.3, 0.3)
    poly = Polygon([
        (-bottom/2, -h/2),
        (bottom/2, -h/2),
        (shift + top/2, h/2),
        (shift - top/2, h/2)
    ])
    return poly


def make_parallelogram():
    w = random.uniform(0.8, 2.0)
    h = random.uniform(0.5, 1.5)
    skew = random.uniform(0.2, 0.8)
    poly = Polygon([
        (-w/2, -h/2),
        (w/2, -h/2),
        (w/2 + skew, h/2),
        (-w/2 + skew, h/2)
    ])
    return poly


def make_circle_polygon():
    r = random.uniform(0.25, 0.9)
    # buffered point gives a polygon approximation of a circle
    return Point(0, 0).buffer(r, resolution=24)


def make_random_shape():
    shape_type = random.choice([
        "rectangle", "triangle", "trapezoid", "parallelogram", "circle"
    ])

    if shape_type == "rectangle":
        poly = make_rectangle()
    elif shape_type == "triangle":
        poly = make_triangle()
    elif shape_type == "trapezoid":
        poly = make_trapezoid()
    elif shape_type == "parallelogram":
        poly = make_parallelogram()
    else:
        poly = make_circle_polygon()

    angle = random.uniform(0, 180)
    poly = rotate(poly, angle, origin=(0, 0), use_radians=False)

    minx, miny, maxx, maxy = poly.bounds
    x = random.uniform(WORKSPACE[0] - minx + 0.1, WORKSPACE[1] - maxx - 0.1)
    y = random.uniform(WORKSPACE[2] - miny + 0.1, WORKSPACE[3] - maxy - 0.1)
    poly = translate(poly, xoff=x, yoff=y)

    return shape_type, poly


# -----------------------------
# Common obstacle sampler
# -----------------------------
def sample_random_obstacles(n_min, n_max, start, goal):
    n_obs = random.randint(n_min, n_max)
    placed = []
    obstacles = []

    start_buffer = Point(start).buffer(0.5)
    goal_buffer = Point(goal).buffer(0.5)

    attempts = 0
    max_attempts = 3000

    while len(placed) < n_obs and attempts < max_attempts:
        attempts += 1
        shape_type, candidate = make_random_shape()

        if valid_candidate(candidate, placed, start_buffer, goal_buffer):
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
# TODO: Work in the workspace size part here
# TODO: For EACH of the families, change the range of obstacles, there's placeholder numbers for now but feel free to decide that
# TODO: This is up to you, but you can add functionality to change the range for the size of shapes based on 
def generate_sparse_clutter():
    start, goal = make_start_goal()
    _, obstacles = sample_random_obstacles(5, 8, start, goal) 

    return {
        "family": "sparse_clutter",
        "workspace": WORKSPACE,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles
    }


def generate_dense_clutter():
    start, goal = make_start_goal()
    _, obstacles = sample_random_obstacles(12, 20, start, goal)

    return {
        "family": "dense_clutter",
        "workspace": WORKSPACE,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles
    }


def generate_narrow_passage():
    start, goal = make_start_goal()
    obstacles = []

    # manually create passage walls
    left_wall = Polygon([(4.2, 0.5), (4.7, 0.5), (4.7, 4.2), (4.2, 4.2)])
    right_wall = Polygon([(5.3, 5.8), (5.8, 5.8), (5.8, 9.5), (5.3, 9.5)])

    # connector walls creating a narrow path between y=4.2 and y=5.8
    upper_left = Polygon([(4.2, 5.8), (4.7, 5.8), (4.7, 9.5), (4.2, 9.5)])
    lower_right = Polygon([(5.3, 0.5), (5.8, 0.5), (5.8, 4.2), (5.3, 4.2)])

    fixed = [
        ("rectangle", left_wall, "forbidden"),
        ("rectangle", right_wall, "forbidden"),
        ("rectangle", upper_left, "forbidden"),
        ("rectangle", lower_right, "forbidden"),
    ]

    for i, (shape_type, poly, cls) in enumerate(fixed):
        obstacles.append({
            "id": i,
            "shape_type": shape_type,
            "class_true": cls,
            "vertices": polygon_to_list(poly)
        })

    # add more clutter around the passage
    placed = [left_wall, right_wall, upper_left, lower_right]
    start_buffer = Point(start).buffer(0.5)
    goal_buffer = Point(goal).buffer(0.5)

    attempts = 0
    while len(obstacles) < random.randint(8, 12) and attempts < 2000:
        attempts += 1
        shape_type, candidate = make_random_shape()
        if valid_candidate(candidate, placed, start_buffer, goal_buffer):
            placed.append(candidate)
            obstacles.append({
                "id": len(obstacles),
                "shape_type": shape_type,
                "class_true": random_class(),
                "vertices": polygon_to_list(candidate)
            })

    return {
        "family": "narrow_passage",
        "workspace": WORKSPACE,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles
    }


def generate_semantic_trap():
    start, goal = make_start_goal()
    obstacles = []

    # tempting central object labeled misleadingly in later versions if needed
    trap_poly = Point(5.0, 5.0).buffer(0.8, resolution=24)

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
    target_total = random.randint(8, 12)
    while len(obstacles) < target_total and attempts < 2500:
        attempts += 1
        shape_type, candidate = make_random_shape()
        if valid_candidate(candidate, placed, start_buffer, goal_buffer):
            placed.append(candidate)
            obstacles.append({
                "id": len(obstacles),
                "shape_type": shape_type,
                "class_true": random_class(),
                "vertices": polygon_to_list(candidate)
            })

    return {
        "family": "semantic_trap",
        "workspace": WORKSPACE,
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

    num_changes = min(random.randint(1, 3), len(base["obstacles"]))
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
            if not within_workspace(candidate):
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


def generate_scene(family):
    if family == "sparse_clutter":
        return generate_sparse_clutter()
    if family == "dense_clutter":
        return generate_dense_clutter()
    if family == "narrow_passage":
        return generate_narrow_passage()
    if family == "semantic_trap":
        return generate_semantic_trap()
    if family == "perturbed":
        return generate_perturbed()
    raise ValueError(f"Unknown family: {family}")


# -----------------------------
# Save JSON
# -----------------------------
def save_scene_json(scene, path):
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)