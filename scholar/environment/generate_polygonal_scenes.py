"""Generate BEACON-style scenes with mixed polygon and circle obstacles."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import Polygon as MplPolygon
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box


WORKSPACE = (0.0, 6.0, 0.0, 6.0)
FAMILIES = ["sparse", "cluttered", "collision_required", "collision_shortcut"]
SCENES_DIR = Path(__file__).resolve().parent / "data" / "scenes_polygonal"
IMAGES_DIR = Path(__file__).resolve().parent / "data" / "images_polygonal"
CLASS_COLORS = {
    "movable": "#e0b400",
    "not_movable": "#8b9db5",
}


def polygon_to_vertices(poly: Polygon) -> list[list[float]]:
    coords = list(poly.exterior.coords)[:-1]
    return [[round(float(x), 4), round(float(y), 4)] for x, y in coords]


def within_workspace(poly: Polygon) -> bool:
    xmin, xmax, ymin, ymax = WORKSPACE
    return box(xmin, ymin, xmax, ymax).contains(poly)


def obstacle_record(poly: Polygon, idx: int, cls: str) -> dict:
    center = poly.centroid
    return {
        "id": idx,
        "shape_type": "polygon",
        "class_true": cls,
        "true_class": cls,
        "center": [round(center.x, 4), round(center.y, 4)],
        "vertices": polygon_to_vertices(poly),
        "observed": False,
    }


def circle_record(center: tuple[float, float], radius: float, idx: int, cls: str) -> dict:
    return {
        "id": idx,
        "shape_type": "circle",
        "class_true": cls,
        "true_class": cls,
        "center": [round(center[0], 4), round(center[1], 4)],
        "radius": round(radius, 4),
        "vertices": polygon_to_vertices(Point(center).buffer(radius, resolution=24)),
        "observed": False,
    }


def make_random_polygon(rng: random.Random, center: tuple[float, float] | None = None) -> Polygon:
    xmin, xmax, ymin, ymax = WORKSPACE
    if center is None:
        center = (
            rng.uniform(xmin + 0.55, xmax - 0.55),
            rng.uniform(ymin + 0.55, ymax - 0.55),
        )

    n_vertices = rng.randint(4, 7)
    base_radius = rng.uniform(0.18, 0.42)
    angles = sorted(rng.uniform(0.0, 2.0 * math.pi) for _ in range(n_vertices))
    pts: list[tuple[float, float]] = []
    for angle in angles:
        radius = base_radius * rng.uniform(0.7, 1.3)
        pts.append((
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        ))

    poly = Polygon(pts).convex_hull
    stretch_x = rng.uniform(0.8, 1.35)
    stretch_y = rng.uniform(0.8, 1.35)
    poly = affinity.scale(poly, xfact=stretch_x, yfact=stretch_y, origin="center")
    poly = affinity.rotate(poly, rng.uniform(0.0, 180.0), origin="center")
    return poly


def make_random_circle(rng: random.Random, center: tuple[float, float] | None = None) -> tuple[Polygon, tuple[float, float], float]:
    xmin, xmax, ymin, ymax = WORKSPACE
    radius = rng.uniform(0.16, 0.34)
    if center is None:
        center = (
            rng.uniform(xmin + radius + 0.2, xmax - radius - 0.2),
            rng.uniform(ymin + radius + 0.2, ymax - radius - 0.2),
        )
    return Point(center).buffer(radius, resolution=32), center, radius


def random_start_goal(rng: random.Random, min_dist: float = 3.5, margin: float = 0.5) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, xmax, ymin, ymax = WORKSPACE
    for _ in range(500):
        start = (
            rng.uniform(xmin + margin, xmax - margin),
            rng.uniform(ymin + margin, ymax - margin),
        )
        goal = (
            rng.uniform(xmin + margin, xmax - margin),
            rng.uniform(ymin + margin, ymax - margin),
        )
        if math.dist(start, goal) >= min_dist:
            return start, goal
    return (0.6, 0.6), (5.4, 5.4)


def valid_candidate(poly: Polygon, placed: list[Polygon], start: tuple[float, float], goal: tuple[float, float], min_gap: float = 0.08) -> bool:
    if not poly.is_valid or not within_workspace(poly):
        return False
    start_buffer = Point(start).buffer(0.35)
    goal_buffer = Point(goal).buffer(0.35)
    if poly.distance(start_buffer) < min_gap or poly.distance(goal_buffer) < min_gap:
        return False
    return all(poly.distance(existing) >= min_gap for existing in placed)


def add_random_obstacles(
    rng: random.Random,
    placed: list[Polygon],
    obstacles: list[dict],
    start: tuple[float, float],
    goal: tuple[float, float],
    n_min: int,
    n_max: int,
) -> None:
    target = rng.randint(n_min, n_max)
    attempts = 0
    while len(obstacles) < target and attempts < 15000:
        attempts += 1
        use_circle = rng.random() < 0.4
        if use_circle:
            poly, center, radius = make_random_circle(rng)
        else:
            poly = make_random_polygon(rng)
        if valid_candidate(poly, placed, start, goal):
            placed.append(poly)
            cls = "movable" if rng.random() < 0.6 else "not_movable"
            if use_circle:
                obstacles.append(circle_record(center, radius, len(obstacles), cls))
            else:
                obstacles.append(obstacle_record(poly, len(obstacles), cls))


def generate_sparse(seed: int) -> dict:
    rng = random.Random(seed)
    start, goal = random_start_goal(rng)
    placed: list[Polygon] = []
    obstacles: list[dict] = []
    add_random_obstacles(rng, placed, obstacles, start, goal, 8, 14)
    return scene_dict("sparse", seed, start, goal, obstacles)


def generate_cluttered(seed: int) -> dict:
    rng = random.Random(seed)
    start, goal = random_start_goal(rng)
    placed: list[Polygon] = []
    obstacles: list[dict] = []
    add_random_obstacles(rng, placed, obstacles, start, goal, 24, 36)
    return scene_dict("cluttered", seed, start, goal, obstacles)


def generate_collision_required(seed: int) -> dict:
    rng = random.Random(seed)
    start, goal = random_start_goal(rng, min_dist=4.0)
    placed: list[Polygon] = []
    obstacles: list[dict] = []
    line = LineString([start, goal])
    midpoint = line.interpolate(0.5, normalized=True)

    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    px, py = -dy / length, dx / length

    specs = [
        (-0.95, "not_movable"),
        (-0.32, "movable"),
        (0.32, "movable"),
        (0.95, "not_movable"),
    ]
    for offset, cls in specs:
        center = (midpoint.x + px * offset, midpoint.y + py * offset)
        if abs(offset) < 0.5:
            poly, circle_center, radius = make_random_circle(rng, center=center)
            record = circle_record(circle_center, radius, len(obstacles), cls)
        else:
            poly = make_random_polygon(rng, center=center)
            record = obstacle_record(poly, len(obstacles), cls)
        if valid_candidate(poly, placed, start, goal, min_gap=0.05):
            placed.append(poly)
            obstacles.append(record)

    add_random_obstacles(rng, placed, obstacles, start, goal, 6, 10)
    return scene_dict("collision_required", seed, start, goal, obstacles)


def generate_collision_shortcut(seed: int) -> dict:
    rng = random.Random(seed)
    start, goal = random_start_goal(rng, min_dist=4.0)
    placed: list[Polygon] = []
    obstacles: list[dict] = []
    line = LineString([start, goal])
    for frac in (0.35, 0.5, 0.65):
        p = line.interpolate(frac, normalized=True)
        if frac == 0.5:
            poly, circle_center, radius = make_random_circle(rng, center=(p.x, p.y))
            record = circle_record(circle_center, radius, len(obstacles), "movable")
        else:
            poly = make_random_polygon(rng, center=(p.x, p.y))
            record = obstacle_record(poly, len(obstacles), "movable")
        if valid_candidate(poly, placed, start, goal, min_gap=0.05):
            placed.append(poly)
            obstacles.append(record)

    add_random_obstacles(rng, placed, obstacles, start, goal, 14, 20)
    return scene_dict("collision_shortcut", seed, start, goal, obstacles)


def scene_dict(family: str, seed: int, start: tuple[float, float], goal: tuple[float, float], obstacles: list[dict]) -> dict:
    return {
        "family": family,
        "workspace": list(WORKSPACE),
        "start": [round(start[0], 4), round(start[1], 4), 0.0],
        "goal": [round(goal[0], 4), round(goal[1], 4), 0.0],
        "seed": seed,
        "obstacles": obstacles,
    }


def draw_scene(scene: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    xmin, xmax, ymin, ymax = scene["workspace"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    for obs in scene["obstacles"]:
        color = CLASS_COLORS.get(obs.get("true_class", "movable"), "#cccccc")
        if obs.get("shape_type") == "circle":
            cx, cy = obs["center"]
            patch = MplCircle(
                (cx, cy),
                obs["radius"],
                edgecolor="black",
                facecolor=color,
                linewidth=0.9,
                alpha=0.82,
            )
        else:
            patch = MplPolygon(
                obs["vertices"],
                closed=True,
                edgecolor="black",
                facecolor=color,
                linewidth=0.9,
                alpha=0.82,
            )
        ax.add_patch(patch)

    sx, sy, _ = scene["start"]
    gx, gy, _ = scene["goal"]
    ax.plot(sx, sy, "o", color="#1f78b4", markersize=8, label="Start")
    ax.plot(gx, gy, "*", color="#d62728", markersize=12, label="Goal")
    ax.set_title(f"{scene['family']} mixed-shape scene")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_scene(family: str, seed: int) -> dict:
    generators = {
        "sparse": generate_sparse,
        "cluttered": generate_cluttered,
        "collision_required": generate_collision_required,
        "collision_shortcut": generate_collision_shortcut,
    }
    return generators[family](seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mixed polygon-and-circle BEACON scenes and preview PNGs.")
    parser.add_argument("--family", nargs="*", default=FAMILIES, choices=FAMILIES)
    parser.add_argument("--index", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    for family in args.family:
        scene_dir = SCENES_DIR / family
        image_dir = IMAGES_DIR / family
        scene_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating {family} mixed-shape scenes...")
        for idx in args.index:
            scene = generate_scene(family, idx)
            scene_path = scene_dir / f"scene_{idx:03d}.json"
            image_path = image_dir / f"{family}_{idx:03d}.png"
            scene_path.write_text(json.dumps(scene, indent=2) + "\n")
            draw_scene(scene, image_path)
            print(f"  saved {scene_path.name} and {image_path.name}")


if __name__ == "__main__":
    main()
