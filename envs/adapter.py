import json
import numpy as np

CLASSES = ["safe", "movable", "fragile", "forbidden"]

def load_and_adapt_scene(json_path, noise_level="medium"):
    """
    Load one of your existing scene JSONs and convert it
    to the format our SURP planners expect.
    """
    with open(json_path) as f:
        raw = json.load(f)

    correct_prob = {"low": 0.9, "medium": 0.75, "high": 0.6}[noise_level]

    obstacles = []
    for verts in raw["obstacles"]:
        # Randomly assign a true class
        true_cls_idx = np.random.randint(0, 4)
        prior = [(1 - correct_prob) / 3] * 4
        prior[true_cls_idx] = correct_prob

        obstacles.append({
            "vertices": verts,
            "true_class": CLASSES[true_cls_idx],
            "prior": prior
        })

    # Your scenes have 2D start/goal — add theta=0
    sx, sy = raw["start"]
    gx, gy = raw["goal"]

    return {
        "workspace": raw["workspace"],
        "start": [sx, sy, 0],
        "goal":  [gx, gy, 0],
        "obstacles": obstacles,
        "noise_level": noise_level
    }