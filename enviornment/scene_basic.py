import json
import random
from shapely.geometry import Point, box

WORKSPACE = (0, 6, 0, 6)   # xmin, xmax, ymin, ymax
CLASSES = ["safe", "movable", "fragile"]   # no immovable / forbidden


def polygon_to_list(poly):
    coords = list(poly.exterior.coords)[:-1]
    return [[float(x), float(y)] for x, y in coords]


def within_workspace(poly):
    xmin, xmax, ymin, ymax = WORKSPACE
    workspace_poly = box(xmin, ymin, xmax, ymax)
    return workspace_poly.contains(poly)


def make_start_goal():
    xmin, xmax, ymin, ymax = WORKSPACE

    # same idea as original: fixed, well-separated corners
    start = (xmin + 0.6, ymin + 0.6)
    goal = (xmax - 0.6, ymax - 0.6)

    return start, goal


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


def make_circle():
    r = random.uniform(0.15, 0.45)
    xmin, xmax, ymin, ymax = WORKSPACE

    x = random.uniform(xmin + r, xmax - r)
    y = random.uniform(ymin + r, ymax - r)

    poly = Point(x, y).buffer(r, resolution=32)
    return poly, r


def generate_circle_scene(family="sparse", seed=None):
    if seed is not None:
        random.seed(seed)

    if family == "sparse":
        n_min, n_max = 8, 15
    elif family == "cluttered":
        n_min, n_max = 25, 45
    else:
        raise ValueError("family must be 'sparse' or 'cluttered'")

    start, goal = make_start_goal()
    start_buffer = Point(start).buffer(0.35)
    goal_buffer = Point(goal).buffer(0.35)

    n_obs = random.randint(n_min, n_max)

    placed = []
    obstacles = []

    attempts = 0
    max_attempts = 10000

    while len(obstacles) < n_obs and attempts < max_attempts:
        attempts += 1
        candidate, radius = make_circle()

        if valid_candidate(candidate, placed, start_buffer, goal_buffer):
            placed.append(candidate)
            center = candidate.centroid

            obstacles.append({
                "id": len(obstacles),
                "shape_type": "circle",
                "class_true": random.choice(CLASSES),
                "radius": round(radius, 4),
                "center": [round(center.x, 4), round(center.y, 4)],
                "vertices": polygon_to_list(candidate)
            })

    return {
        "family": family,
        "workspace": WORKSPACE,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles
    }


def save_scene_json(scene, path):
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)