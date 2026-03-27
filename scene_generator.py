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

def generate_scene(num_obstacles=6):
    obstacles = []
    start = (1, 1)
    goal = (8, 8)

    start_buffer = Point(start).buffer(0.4)
    goal_buffer = Point(goal).buffer(0.4)

    attempts = 0
    while len(obstacles) < num_obstacles and attempts < 500:
        attempts += 1
        candidate = random_rectangle()

        if candidate.intersects(start_buffer) or candidate.intersects(goal_buffer):
            continue

        if any(candidate.intersects(obs) for obs in obstacles):
            continue

        obstacles.append(candidate)

    return {
        "workspace": WORKSPACE,
        "start": start,
        "goal": goal,
        "obstacles": obstacles
    }

def polygon_to_list(poly):
    coords = list(poly.exterior.coords)[:-1]
    return [[float(x), float(y)] for x, y in coords]

def save_scene_json(scene, path):
    data = {
        "workspace": list(scene["workspace"]),
        "start": list(scene["start"]),
        "goal": list(scene["goal"]),
        "obstacles": [polygon_to_list(obs) for obs in scene["obstacles"]]
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
