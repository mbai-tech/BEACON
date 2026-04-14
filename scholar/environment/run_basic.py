import sys
import os
from scene_basic import generate_circle_scene, save_scene_json
from draw_basic import draw_scene

os.makedirs("images_basic", exist_ok=True)
os.makedirs("json_basic", exist_ok=True)

NUM_SCENES = 10

if len(sys.argv) > 1:
    NUM_SCENES = int(sys.argv[1])

families = ["sparse", "cluttered"]

for family in families:
    for i in range(NUM_SCENES):
        scene = generate_circle_scene(family)

        draw_scene(scene, f"images_basic/{family}_{i:03d}.png")
        save_scene_json(scene, f"json_basic/{family}_{i:03d}.json")

print(f"Generated {NUM_SCENES} scenes per family.")