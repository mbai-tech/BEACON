import json, numpy as np
from shapely.geometry import Polygon

CLASSES = ["safe", "movable", "fragile", "forbidden"]
PLAN_COSTS = {"safe": 1, "movable": 3, "fragile": 10, "forbidden": 1000}

# The 4x4 likelihood table: p(outcome | true_class)
# Rows = outcomes (0=null,1=displacement,2=damage,3=hardflag)
# Columns = classes (safe, movable, fragile, forbidden)
DEFAULT_LIKELIHOOD = np.array([
    [0.90, 0.05, 0.04, 0.01],  # null outcome
    [0.05, 0.85, 0.05, 0.05],  # displacement
    [0.04, 0.08, 0.85, 0.03],  # damage flag
    [0.01, 0.02, 0.06, 0.91],  # hard event
])

def make_prior(true_class_idx, noise_level="medium"):
    """Create noisy prior distribution over classes."""
    correct_prob = {"low": 0.9, "medium": 0.75, "high": 0.6}[noise_level]
    prior = np.full(4, (1 - correct_prob) / 3)
    prior[true_class_idx] = correct_prob
    return prior.tolist()

def generate_scene(n_obstacles=8, noise_level="medium", seed=42):
    np.random.seed(seed)
    obstacles = []
    for i in range(n_obstacles):
        # Random small rectangle obstacle
        cx, cy = np.random.uniform(1, 9), np.random.uniform(1, 9)
        w, h = np.random.uniform(0.3, 0.8), np.random.uniform(0.3, 0.8)
        verts = [(cx-w/2, cy-h/2),(cx+w/2, cy-h/2),
                 (cx+w/2, cy+h/2),(cx-w/2, cy+h/2)]
        true_cls = np.random.choice(len(CLASSES))
        obstacles.append({
            "id": i,
            "vertices": verts,
            "true_class": CLASSES[true_cls],
            "prior": make_prior(true_cls, noise_level)
        })
    return {
        "workspace": [0, 0, 10, 10],
        "start": [0.5, 0.5, 0],
        "goal": [9.5, 9.5, 0],
        "obstacles": obstacles,
        "likelihood_table": DEFAULT_LIKELIHOOD.tolist(),
        "noise_level": noise_level
    }

def save_scene(scene, path):
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)