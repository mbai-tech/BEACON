"""
geometry.py — AABB, ray casting, and visibility polygon helpers.
"""

import numpy as np
from typing import List, Optional, Tuple


# ── AABB ───────────────────────────────────────────────────────────────────────

def aabb_contains(
    point: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
) -> bool:
    """Return True if point lies within the axis-aligned bounding box."""
    return bool(np.all(np.abs(point - center[:2]) <= half_extents[:2]))


def aabb_overlap(
    c1: np.ndarray, h1: np.ndarray,
    c2: np.ndarray, h2: np.ndarray,
) -> bool:
    """Return True if two AABBs overlap."""
    return bool(np.all(np.abs(c1[:2] - c2[:2]) <= h1[:2] + h2[:2]))


# ── Ray casting ────────────────────────────────────────────────────────────────

def ray_aabb_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
) -> Optional[float]:
    """Return distance to first intersection of ray with AABB, or None."""
    d = direction[:2]
    o = origin[:2]
    c = center[:2]
    h = half_extents[:2]

    t_min, t_max = -np.inf, np.inf
    for i in range(2):
        if abs(d[i]) < 1e-10:
            if abs(o[i] - c[i]) > h[i]:
                return None
        else:
            t1 = (c[i] - h[i] - o[i]) / d[i]
            t2 = (c[i] + h[i] - o[i]) / d[i]
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))

    if t_max < 0 or t_min > t_max:
        return None
    return float(t_min) if t_min >= 0 else float(t_max)


def cast_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    obstacles: list,
    max_range: float = 20.0,
) -> Tuple[float, Optional[int]]:
    """Cast a ray and return (distance, obstacle_body_id) of nearest hit."""
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    best_dist = max_range
    best_id: Optional[int] = None

    for obs in obstacles:
        dist = ray_aabb_intersect(origin, direction, obs.position[:2], obs.half_extents[:2])
        if dist is not None and dist < best_dist:
            best_dist = dist
            best_id = obs.body_id

    return best_dist, best_id


# ── Visibility polygon (approximate) ──────────────────────────────────────────

def visibility_polygon(
    position: np.ndarray,
    obstacles: list,
    sensing_range: float,
    n_rays: int = 180,
) -> np.ndarray:
    """Approximate visibility polygon by casting n_rays uniformly around position.

    Returns an (n_rays, 2) array of visible boundary points.
    """
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    boundary = np.zeros((n_rays, 2))
    for i, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)])
        dist, _ = cast_ray(position[:2], direction, obstacles, max_range=sensing_range)
        boundary[i] = position[:2] + direction * dist
    return boundary
