import random
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
import json

WORKSPACE = (0, 10, 0, 10)

def random_rectangle():
    w = random.uniform(0.6, 1.5)
    h = random.uniform(0.6, 1.5)
    x = random.uniform(w/2, 10 - w/2)
    y = random.uniform(h/2, 10 - h/2)

    rect = Polygon([(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)])
    rect = translate(rect, xoff=x, yoff=y)
    return rect

def random_circle():
    radius = random.uniform(0.35, 0.9)
    x = random.uniform(radius, 10 - radius)
    y = random.uniform(radius, 10 - radius)
    return Point(x, y).buffer(radius, resolution=32)

def generate_scene(num_obstacles=6, obstacle_shape="rectangle"):
    obstacles = []
    start = (1, 1)
    goal = (8, 8)

    start_buffer = Point(start).buffer(0.4)
    goal_buffer = Point(goal).buffer(0.4)
    obstacle_generators = {
        "rectangle": random_rectangle,
        "circle": random_circle
    }

    if obstacle_shape not in obstacle_generators:
        raise ValueError(
            f"Unsupported obstacle_shape '{obstacle_shape}'. "
            "Use 'rectangle' or 'circle'."
        )

    attempts = 0
    while len(obstacles) < num_obstacles and attempts < 500:
        attempts += 1
        candidate = obstacle_generators[obstacle_shape]()

        if candidate.intersects(start_buffer) or candidate.intersects(goal_buffer):
            continue

        if any(candidate.intersects(obs) for obs in obstacles):
            continue

        obstacles.append(candidate)

    return {
        "workspace": WORKSPACE,
        "start": start,
        "goal": goal,
        "obstacles": obstacles,
        "obstacle_shape": obstacle_shape
    }

def polygon_to_list(poly):
    coords = list(poly.exterior.coords)[:-1]
    return [[float(x), float(y)] for x, y in coords]

def save_scene_json(scene, path):
    data = {
        "workspace": list(scene["workspace"]),
        "start": list(scene["start"]),
        "goal": list(scene["goal"]),
        "obstacle_shape": scene.get("obstacle_shape", "rectangle"),
        "obstacles": [polygon_to_list(obs) for obs in scene["obstacles"]]
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
