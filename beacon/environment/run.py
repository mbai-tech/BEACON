from scene_generator import generate_scene
from draw_scene import draw_scene
from scene_generator import save_scene_json

for i in range(100):
    scene = generate_scene()
    draw_scene(scene, f"data/images/scene_{i:03d}.png")
    save_scene_json(scene, f"data/scenes/scene_{i:03d}.json")