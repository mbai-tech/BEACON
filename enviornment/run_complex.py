import sys
import os
from scene_complex import generate_scene, save_scene_json
from draw_complex import draw_scene

os.makedirs("images", exist_ok=True)
os.makedirs("scenes", exist_ok=True)

NUM_SCENES = 5
if len(sys.argv) > 1:
    NUM_SCENES = int(sys.argv[1])

families = [
    "sparse",
    "cluttered",
    "collision_required",
    "collision_shortcut"
]

for family in families:
    for i in range(NUM_SCENES):
        scene = generate_scene(family=family)

        draw_scene(scene, f"images_c/{family}_{i:03d}.png")
        save_scene_json(scene, f"json_c/{family}_{i:03d}.json")

print(f"Generated {NUM_SCENES} scenes per family.")