from scene_generator import generate_scene, save_scene_json
from draw_scene import draw_scene

families = [
    # "sparse_clutter",
    # "dense_clutter",
    # "narrow_passage",
    # "semantic_trap",
    "perturbed"
]

scenes_per_family = 5 # TODO: change this after you finish testing

for family in families:
    for i in range(scenes_per_family):
        scene = generate_scene(family)

        # print(f"\nFamily: {family}, scene {i}")
        # print("Obstacle count:", len(scene["obstacles"]))
        for obs in scene["obstacles"][:5]:
        # print("  ", obs["shape_type"], obs["class_true"])

            image_path = f"data/images_new/{family}_{i:03d}.png"
            json_path = f"data/scenes_new/{family}_{i:03d}.json"

            draw_scene(scene, image_path)
            save_scene_json(scene, json_path)

print("Done generating scenes.")