#!/usr/bin/env python3

import sys
from pathlib import Path

from beacon.environment.scene_generator_shapely import generate_scene, save_scene_json


VALID_FAMILIES = [
    "narrow_passage",
    "sparse_clutter",
    "dense_clutter",
    "semantic_trap",
    "perturbed",
]


def clear_output_dir(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def next_available_path(directory, base_name, suffix):
    candidate = directory / f"{base_name}{suffix}"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = directory / f"{base_name}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in VALID_FAMILIES:
        print("Usage: python3 beacon/environment/run_family.py <family> [seed]")
        print("Valid families:", ", ".join(VALID_FAMILIES))
        sys.exit(1)

    family = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) == 3 else None
    scene = generate_scene(family, seed=seed)

    base_dir = Path(__file__).resolve().parent
    if seed is None:
        output_dir = base_dir / "data"
        images_dir = output_dir / "images"
        scenes_dir = output_dir / "scenes"
        clear_output_dir(images_dir)
        clear_output_dir(scenes_dir)
        json_path = scenes_dir / f"{family}_scene.json"
        image_path = images_dir / f"{family}_scene.png"
    else:
        output_dir = base_dir / "saved_enviornments"
        images_dir = output_dir / "images"
        scenes_dir = output_dir / "scenes"
        images_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{family}_seed{scene['seed']}"
        json_path = next_available_path(scenes_dir, base_name, ".json")
        image_path = next_available_path(images_dir, base_name, ".png")

    save_scene_json(scene, json_path)

    try:
        from beacon.environment.draw_scene import draw_scene
    except Exception:
        draw_scene = None

    if callable(draw_scene):
        draw_scene(scene, image_path)
        print(f"Saved image to {image_path}")

    print(f"Seed: {scene['seed']}")
    print(f"Saved scene JSON to {json_path}")


if __name__ == "__main__":
    main()
