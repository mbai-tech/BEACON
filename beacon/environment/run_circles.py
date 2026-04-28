#!/usr/bin/env python3

from beacon.environment.draw_scene import draw_scene
from beacon.environment.scene_generator_shapely import generate_scene, save_scene_json


for i in range(100):
    scene = generate_scene(obstacle_shape="circle")
    draw_scene(scene, f"data/images/circle_scene_{i:03d}.png")
    save_scene_json(scene, f"data/scenes/circle_scene_{i:03d}.json")
