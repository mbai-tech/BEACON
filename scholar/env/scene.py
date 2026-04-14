"""
scene.py — Scene generation wrappers around NewProject.scene_setup.

Scenes are dict-based with Shapely polygon obstacles (not PyBullet bodies).
Use generate_scene() to get a random environment in the standard format.
"""

from NewProject.scene_setup import (
    generate_one_random_environment,
    normalize_scene_for_online_use,
    create_cluttered_variant,
    shrink_scene,
)

__all__ = [
    "generate_one_random_environment",
    "normalize_scene_for_online_use",
    "create_cluttered_variant",
    "shrink_scene",
]


def generate_scene(family: str | None = None, extra_clutter: int = 5) -> dict:
    """Generate one random environment, optionally with extra clutter."""
    if family is None:
        return generate_one_random_environment()
    from enviornment.scene_generator import generate_scene as _gen
    from NewProject.scene_setup import convert_scene_obstacles_to_circles
    base = convert_scene_obstacles_to_circles(shrink_scene(_gen(family)))
    return create_cluttered_variant(base, extra_obstacles=extra_clutter)
