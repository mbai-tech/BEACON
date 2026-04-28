"""
utils/geometry.py — Geometry helpers built on top of beacon.core.planner primitives.
"""

import numpy as np
from typing import Optional, Tuple

from beacon.core.planner import (
    normalize,
    obstacle_polygon,
    robot_body,
    clip_point_to_workspace,
)

__all__ = [
    "normalize",
    "obstacle_polygon",
    "robot_body",
    "clip_point_to_workspace",
    "ray_cast",
    "visibility_range",
]


def ray_cast(
    origin: np.ndarray,
    direction: np.ndarray,
    obstacles: list,
    max_range: float = 20.0,
) -> Tuple[float, Optional[int]]:
    """Cast a ray from origin in direction and return (distance, obs_id) of nearest hit.

    Uses Shapely intersection on each observed obstacle's polygon.
    """
    from shapely.geometry import LineString
    direction = normalize(direction)
    endpoint = origin + direction * max_range
    ray = LineString([origin.tolist(), endpoint.tolist()])

    best_dist = max_range
    best_id: Optional[int] = None

    for obs in obstacles:
        poly = obstacle_polygon(obs)
        if ray.intersects(poly):
            intersection = ray.intersection(poly)
            pt = intersection.centroid
            dist = float(np.linalg.norm(np.array([pt.x, pt.y]) - origin))
            if dist < best_dist:
                best_dist = dist
                best_id = obs.get("id")

    return best_dist, best_id


def visibility_range(
    position: np.ndarray,
    obstacles: list,
    sensing_range: float,
    n_rays: int = 36,
) -> np.ndarray:
    """Approximate visible boundary by casting n_rays uniformly around position.

    Returns an (n_rays, 2) array of boundary points.
    """
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    boundary = np.zeros((n_rays, 2))
    for i, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)])
        dist, _ = ray_cast(position, direction, obstacles, max_range=sensing_range)
        boundary[i] = position + direction * dist
    return boundary
