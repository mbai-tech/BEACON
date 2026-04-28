"""
generate_circle_scenes.py — Generate 100 circle-obstacle scenes per family.

Families
--------
dense_clutter  : 25–45 small circles randomly packed across the workspace
narrow_passage : circles form two walls crossing the diagonal, each with one gap
perturbed      : regular grid layout with Gaussian position noise

Obstacle classes are assigned per the fragility distribution:
  mixed       — 70% movable, 30% unmovable  (default)
  uniform     — 50/50 split
  all_movable — every obstacle is movable

Usage
-----
    python beacon/environment/generate_circle_scenes.py
    python beacon/environment/generate_circle_scenes.py --family dense_clutter
    python beacon/environment/generate_circle_scenes.py --n 100 --fragility mixed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import random
import argparse
import numpy as np
from shapely.geometry import Point, box

# ── Constants ──────────────────────────────────────────────────────────────────

WORKSPACE    = (0.0, 6.0, 0.0, 6.0)   # xmin, xmax, ymin, ymax
START        = (0.6, 0.6)
GOAL         = (5.4, 5.4)
START_BUFFER = 0.50    # clear radius around start
GOAL_BUFFER  = 0.50    # clear radius around goal
RESOLUTION   = 32      # circle polygon approximation facets

FAMILIES = ["dense_clutter", "narrow_passage", "perturbed", "semantic_trap", "sparse_clutter"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _polygon_to_list(poly):
    coords = list(poly.exterior.coords)[:-1]
    return [[float(x), float(y)] for x, y in coords]


def _within_workspace(poly):
    xmin, xmax, ymin, ymax = WORKSPACE
    ws = box(xmin, ymin, xmax, ymax)
    return ws.contains(poly)


def _valid_placement(candidate, placed, start_buf, goal_buf):
    if not candidate.is_valid:
        return False
    if not _within_workspace(candidate):
        return False
    if candidate.intersects(start_buf) or candidate.intersects(goal_buf):
        return False
    return not any(candidate.intersects(p) for p in placed)


def _sample_class(rng, fragility="mixed"):
    if fragility == "all_movable":
        return "movable"
    if fragility == "uniform":
        return "movable" if rng.random() < 0.5 else "unmovable"
    return "movable" if rng.random() < 0.70 else "unmovable"   # mixed: 70%


def _make_obs(cx, cy, r, obs_id, true_class):
    poly = Point(cx, cy).buffer(r, resolution=RESOLUTION)
    return {
        "id":         obs_id,
        "shape_type": "circle",
        "true_class": true_class,
        "class_true": true_class,
        "radius":     round(r, 4),
        "center":     [round(cx, 4), round(cy, 4)],
        "vertices":   _polygon_to_list(poly),
        "observed":   False,
    }


def _make_scene_dict(family, obstacles, seed):
    return {
        "family":    family,
        "workspace": list(WORKSPACE),
        "start":     [START[0], START[1], 0.0],
        "goal":      [GOAL[0],  GOAL[1],  0.0],
        "seed":      seed,
        "obstacles": obstacles,
    }


# ── Family: dense_clutter ──────────────────────────────────────────────────────

def generate_dense_clutter(seed: int, fragility: str = "mixed") -> dict:
    """25–45 small-medium circles placed uniformly at random."""
    rng = random.Random(seed)
    start_buf = Point(START).buffer(START_BUFFER)
    goal_buf  = Point(GOAL).buffer(GOAL_BUFFER)
    xmin, xmax, ymin, ymax = WORKSPACE

    n_target  = rng.randint(25, 45)
    placed    = []
    obstacles = []
    attempts  = 0

    while len(obstacles) < n_target and attempts < 30_000:
        attempts += 1
        r  = rng.uniform(0.10, 0.30)
        cx = rng.uniform(xmin + r, xmax - r)
        cy = rng.uniform(ymin + r, ymax - r)
        candidate = Point(cx, cy).buffer(r, resolution=RESOLUTION)
        if _valid_placement(candidate, placed, start_buf, goal_buf):
            placed.append(candidate)
            cls = _sample_class(rng, fragility)
            obstacles.append(_make_obs(cx, cy, r, len(obstacles), cls))

    return _make_scene_dict("dense_clutter", obstacles, seed)


# ── Family: narrow_passage ─────────────────────────────────────────────────────

def generate_narrow_passage(seed: int, fragility: str = "mixed") -> dict:
    """Two walls of circles across the start→goal diagonal, each with a gap."""
    rng  = random.Random(seed)
    nrng = np.random.default_rng(seed)
    start_buf = Point(START).buffer(START_BUFFER)
    goal_buf  = Point(GOAL).buffer(GOAL_BUFFER)
    xmin, xmax, ymin, ymax = WORKSPACE

    sx, sy = START
    gx, gy = GOAL
    diag     = np.array([gx - sx, gy - sy], dtype=float)
    diag_len = float(np.linalg.norm(diag))
    diag_hat = diag / diag_len
    perp_hat = np.array([-diag_hat[1], diag_hat[0]])  # perpendicular unit vec

    placed    = []
    obstacles = []

    # Two walls at 30% and 65% along the diagonal
    for frac in [0.30, 0.65]:
        wall_cx = sx + frac * diag_len * diag_hat[0]
        wall_cy = sy + frac * diag_len * diag_hat[1]

        gap_center = rng.uniform(-1.5, 1.5)   # gap position on perp axis
        gap_half   = rng.uniform(0.20, 0.40)  # half-width of passage
        r_wall     = rng.uniform(0.18, 0.30)
        spacing    = r_wall * 2.1              # circle spacing along wall

        t = -3.5
        while t <= 3.5:
            # Skip the gap
            if gap_center - gap_half <= t <= gap_center + gap_half:
                t += gap_half * 2
                continue

            cx = float(wall_cx + t * perp_hat[0])
            cy = float(wall_cy + t * perp_hat[1])
            cx = float(np.clip(cx, xmin + r_wall, xmax - r_wall))
            cy = float(np.clip(cy, ymin + r_wall, ymax - r_wall))

            candidate = Point(cx, cy).buffer(r_wall, resolution=RESOLUTION)
            if _valid_placement(candidate, placed, start_buf, goal_buf):
                placed.append(candidate)
                cls = _sample_class(rng, fragility)
                obstacles.append(_make_obs(cx, cy, r_wall, len(obstacles), cls))

            t += spacing

    # Scatter extra circles to fill the scene
    scatter_target = len(obstacles) + rng.randint(5, 15)
    attempts = 0
    while len(obstacles) < scatter_target and attempts < 10_000:
        attempts += 1
        r  = rng.uniform(0.12, 0.25)
        cx = rng.uniform(xmin + r, xmax - r)
        cy = rng.uniform(ymin + r, ymax - r)
        candidate = Point(cx, cy).buffer(r, resolution=RESOLUTION)
        if _valid_placement(candidate, placed, start_buf, goal_buf):
            placed.append(candidate)
            cls = _sample_class(rng, fragility)
            obstacles.append(_make_obs(cx, cy, r, len(obstacles), cls))

    return _make_scene_dict("narrow_passage", obstacles, seed)


# ── Family: perturbed ──────────────────────────────────────────────────────────

def generate_perturbed(seed: int, fragility: str = "mixed") -> dict:
    """Regular grid of circles with Gaussian position perturbations."""
    rng  = random.Random(seed)
    nrng = np.random.default_rng(seed)
    start_buf = Point(START).buffer(START_BUFFER)
    goal_buf  = Point(GOAL).buffer(GOAL_BUFFER)
    xmin, xmax, ymin, ymax = WORKSPACE

    n_cols   = rng.randint(4, 6)
    n_rows   = rng.randint(4, 6)
    x_step   = (xmax - xmin) / (n_cols + 1)
    y_step   = (ymax - ymin) / (n_rows + 1)
    sigma    = rng.uniform(0.15, 0.40)   # perturbation std dev

    placed    = []
    obstacles = []

    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            r  = rng.uniform(0.15, 0.40)
            bx = xmin + col * x_step
            by = ymin + row * y_step
            dx, dy = nrng.normal(0.0, sigma, 2)
            cx = float(np.clip(bx + dx, xmin + r, xmax - r))
            cy = float(np.clip(by + dy, ymin + r, ymax - r))

            candidate = Point(cx, cy).buffer(r, resolution=RESOLUTION)
            if _valid_placement(candidate, placed, start_buf, goal_buf):
                placed.append(candidate)
                cls = _sample_class(rng, fragility)
                obstacles.append(_make_obs(cx, cy, r, len(obstacles), cls))

    return _make_scene_dict("perturbed", obstacles, seed)


# ── Family: semantic_trap ──────────────────────────────────────────────────────

def generate_semantic_trap(seed: int, fragility: str = "mixed") -> dict:
    """Unmovable circles surround the path; movable circles fill the interior.

    The idea: the direct path from start to goal is blocked by a ring of
    unmovable obstacles.  Inside and outside the ring are movable objects.
    A planner that ignores semantics will try to push through the unmovable
    barrier; a semantic-aware planner will route around it or push the
    cheaper movable objects instead.
    """
    rng = random.Random(seed)
    start_buf = Point(START).buffer(START_BUFFER)
    goal_buf  = Point(GOAL).buffer(GOAL_BUFFER)
    xmin, xmax, ymin, ymax = WORKSPACE

    placed    = []
    obstacles = []

    # ── Ring of unmovable circles around the midpoint of start→goal ──────────
    sx, sy = START
    gx, gy = GOAL
    mid    = np.array([(sx + gx) / 2, (sy + gy) / 2])
    ring_r = rng.uniform(0.9, 1.4)          # ring radius
    r_obs  = rng.uniform(0.18, 0.28)        # obstacle radius
    # Place obstacles around the ring; gap near both ends so path enters/exits
    gap_angle = rng.uniform(0.35, 0.55)     # half-angle of each gap (radians)
    # Path angle from mid: atan2(gy-sy, gx-sx) and its reverse
    path_angle = float(np.arctan2(gy - sy, gx - sx))
    gap_angles = {path_angle, path_angle + np.pi}   # gaps at entry and exit

    circumference = 2 * np.pi * ring_r
    n_ring = max(6, int(circumference / (r_obs * 2.2)))
    for k in range(n_ring):
        angle = 2 * np.pi * k / n_ring
        # Skip if within a gap
        if any(abs(((angle - ga + np.pi) % (2 * np.pi)) - np.pi) < gap_angle
               for ga in gap_angles):
            continue
        cx = float(mid[0] + ring_r * np.cos(angle))
        cy = float(mid[1] + ring_r * np.sin(angle))
        cx = float(np.clip(cx, xmin + r_obs, xmax - r_obs))
        cy = float(np.clip(cy, ymin + r_obs, ymax - r_obs))
        candidate = Point(cx, cy).buffer(r_obs, resolution=RESOLUTION)
        if _valid_placement(candidate, placed, start_buf, goal_buf):
            placed.append(candidate)
            obstacles.append(_make_obs(cx, cy, r_obs, len(obstacles), "unmovable"))

    # ── Scatter movable circles inside and outside the ring ───────────────────
    n_movable = rng.randint(10, 20)
    attempts  = 0
    while len([o for o in obstacles if o["true_class"] == "movable"]) < n_movable \
            and attempts < 15_000:
        attempts += 1
        r  = rng.uniform(0.10, 0.25)
        cx = rng.uniform(xmin + r, xmax - r)
        cy = rng.uniform(ymin + r, ymax - r)
        candidate = Point(cx, cy).buffer(r, resolution=RESOLUTION)
        if _valid_placement(candidate, placed, start_buf, goal_buf):
            placed.append(candidate)
            obstacles.append(_make_obs(cx, cy, r, len(obstacles), "movable"))

    return _make_scene_dict("semantic_trap", obstacles, seed)


# ── Family: sparse_clutter ─────────────────────────────────────────────────────

def generate_sparse_clutter(seed: int, fragility: str = "mixed") -> dict:
    """8–15 circles scattered lightly across the workspace (easy baseline)."""
    rng = random.Random(seed)
    start_buf = Point(START).buffer(START_BUFFER)
    goal_buf  = Point(GOAL).buffer(GOAL_BUFFER)
    xmin, xmax, ymin, ymax = WORKSPACE

    n_target  = rng.randint(8, 15)
    placed    = []
    obstacles = []
    attempts  = 0

    while len(obstacles) < n_target and attempts < 10_000:
        attempts += 1
        r  = rng.uniform(0.15, 0.45)
        cx = rng.uniform(xmin + r, xmax - r)
        cy = rng.uniform(ymin + r, ymax - r)
        candidate = Point(cx, cy).buffer(r, resolution=RESOLUTION)
        if _valid_placement(candidate, placed, start_buf, goal_buf):
            placed.append(candidate)
            cls = _sample_class(rng, fragility)
            obstacles.append(_make_obs(cx, cy, r, len(obstacles), cls))

    return _make_scene_dict("sparse_clutter", obstacles, seed)


# ── Dispatch ───────────────────────────────────────────────────────────────────

_GENERATORS = {
    "dense_clutter":  generate_dense_clutter,
    "narrow_passage": generate_narrow_passage,
    "perturbed":      generate_perturbed,
    "semantic_trap":  generate_semantic_trap,
    "sparse_clutter": generate_sparse_clutter,
}


# ── Generate and save ──────────────────────────────────────────────────────────

def generate_all(
    n: int = 100,
    families: list = None,
    fragility: str = "mixed",
) -> None:
    if families is None:
        families = FAMILIES

    scenes_dir = Path(__file__).resolve().parent / "data" / "scenes"

    for family in families:
        out_dir = scenes_dir / family
        out_dir.mkdir(parents=True, exist_ok=True)
        gen_fn  = _GENERATORS[family]

        print(f"Generating {n} '{family}' scenes [{fragility}]...")
        for i in range(n):
            scene = gen_fn(seed=i, fragility=fragility)
            path  = out_dir / f"scene_{i:03d}.json"
            with open(path, "w") as f:
                json.dump(scene, f, indent=2)

        n_obs_list = []
        for i in range(min(n, 5)):
            with open(out_dir / f"scene_{i:03d}.json") as f:
                s = json.load(f)
            n_obs_list.append(len(s["obstacles"]))
        print(f"  saved to {out_dir}  (sample sizes: {n_obs_list}...)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate circle-obstacle scenes for BEACON experiments."
    )
    parser.add_argument("--n", type=int, default=100,
                        help="Number of scenes per family (default: 100)")
    parser.add_argument("--family", nargs="*", default=None,
                        choices=FAMILIES,
                        help="Families to generate (default: all five)")
    parser.add_argument("--fragility", default="mixed",
                        choices=["mixed", "uniform", "all_movable"])
    args = parser.parse_args()

    generate_all(n=args.n, families=args.family, fragility=args.fragility)
