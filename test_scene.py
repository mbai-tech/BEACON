import json
from envs.generator import generate_scene, save_scene

scene = generate_scene(n_obstacles=6)
save_scene(scene, "data/scenes/test_scene.json")
print(json.dumps(scene, indent=2))