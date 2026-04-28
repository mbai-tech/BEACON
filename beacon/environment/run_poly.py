from pathlib import Path

from beacon.environment.draw_scene import draw_scene
from beacon.environment.scene_generator_shapely import generate_scene, save_scene_json


FAMILIES = [
    "sparse_clutter",
    "dense_clutter",
    "narrow_passage",
    "semantic_trap",
    "perturbed",
]
SCENES_PER_FAMILY = 100


def clear_output_dir(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def main():
    base_dir = Path(__file__).resolve().parent
    images_dir = base_dir / "data" / "images"
    scenes_dir = base_dir / "data" / "scenes"
    clear_output_dir(images_dir)
    clear_output_dir(scenes_dir)

    for family in FAMILIES:
        for i in range(SCENES_PER_FAMILY):
            scene = generate_scene(family)
            seed = scene["seed"]
            base_name = f"{family}_{i:03d}_seed{seed}"
            image_path = images_dir / f"{base_name}.png"
            json_path = scenes_dir / f"{base_name}.json"

            draw_scene(scene, image_path)
            save_scene_json(scene, json_path)

    print("Done generating scenes.")


if __name__ == "__main__":
    main()
