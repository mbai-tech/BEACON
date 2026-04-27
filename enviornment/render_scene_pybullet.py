import time
import math
import sys
from pathlib import Path

import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np

# Your folder is named "enviornment", so import from there.
sys.path.append("enviornment")
from scene_complex import generate_scene, save_scene_json


ENV_DIR = Path(__file__).resolve().parent
JSON_DIR = ENV_DIR / "json_c"
IMAGES_DIR = ENV_DIR / "images_c"
FAMILIES = [
    "sparse",
    "cluttered",
    "collision_required",
    "collision_shortcut",
]

HEIGHT_BY_CLASS = {
    "safe": 0.18,
    "movable": 0.28,
    "fragile": 0.38,
}

COLOR_BY_CLASS = {
    "safe": [0.2, 0.8, 0.35, 1.0],      # green
    "movable": [0.2, 0.45, 1.0, 1.0],   # blue
    "fragile": [1.0, 0.35, 0.25, 1.0],  # red
}


def make_extruded_polygon_mesh(vertices_2d, height):
    """
    Converts a 2D polygon into a 3D prism mesh.

    vertices_2d: [[x, y], ...]
    height: z height of obstacle
    """
    n = len(vertices_2d)

    bottom = [[x, y, 0.0] for x, y in vertices_2d]
    top = [[x, y, height] for x, y in vertices_2d]
    vertices_3d = bottom + top

    indices = []

    # Bottom face
    for i in range(1, n - 1):
        indices.extend([0, i + 1, i])

    # Top face
    for i in range(1, n - 1):
        indices.extend([n, n + i, n + i + 1])

    # Side faces
    for i in range(n):
        j = (i + 1) % n

        bottom_i = i
        bottom_j = j
        top_i = n + i
        top_j = n + j

        indices.extend([bottom_i, bottom_j, top_j])
        indices.extend([bottom_i, top_j, top_i])

    return vertices_3d, indices


def add_polygon_obstacle(obs):
    cls = obs["class_true"]
    height = HEIGHT_BY_CLASS.get(cls, 0.25)
    color = COLOR_BY_CLASS.get(cls, [0.7, 0.7, 0.7, 1.0])

    vertices, indices = make_extruded_polygon_mesh(obs["vertices"], height)

    collision_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        vertices=vertices,
        indices=indices,
    )

    visual_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        vertices=vertices,
        indices=indices,
        rgbaColor=color,
    )

    body_id = p.createMultiBody(
        baseMass=0.0 if cls in ["safe", "fragile"] else 1.0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=[0, 0, 0],
    )

    p.changeDynamics(
        body_id,
        -1,
        lateralFriction=0.8,
        restitution=0.1,
    )

    return body_id


def add_marker(position, color, radius=0.12, height=0.05):
    x, y, _ = position

    collision_id = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height,
    )

    visual_id = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color,
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=[x, y, height / 2],
    )


def setup_pybullet_scene(scene):
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")

    xmin, xmax, ymin, ymax = scene["workspace"]

    # Draw workspace boundary
    z = 0.02
    corners = [
        [xmin, ymin, z],
        [xmax, ymin, z],
        [xmax, ymax, z],
        [xmin, ymax, z],
    ]

    for i in range(4):
        p.addUserDebugLine(
            corners[i],
            corners[(i + 1) % 4],
            lineColorRGB=[1, 1, 1],
            lineWidth=3,
        )

    # Start and goal markers
    add_marker(scene["start"], [0.0, 1.0, 0.0, 1.0])
    add_marker(scene["goal"], [1.0, 0.0, 1.0, 1.0])

    # Obstacles
    for obs in scene["obstacles"]:
        add_polygon_obstacle(obs)

    # Camera
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    p.resetDebugVisualizerCamera(
        cameraDistance=7.5,
        cameraYaw=45,
        cameraPitch=-55,
        cameraTargetPosition=[center_x, center_y, 0],
    )

    return scene


def build_pybullet_scene(family="collision_required", seed=1):
    scene = generate_scene(family=family, seed=seed)
    setup_pybullet_scene(scene)
    return scene


def save_pybullet_screenshot(scene, path, width=1000, height=800):
    xmin, xmax, ymin, ymax = scene["workspace"]
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[center_x, center_y, 0],
        distance=7.5,
        yaw=45,
        pitch=-55,
        roll=0,
        upAxisIndex=2,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.1,
        farVal=100,
    )

    _, _, rgb, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    rgb = np.reshape(rgb, (height, width, 4)).astype(np.uint8)
    plt.imsave(path, rgb[:, :, :3])


if __name__ == "__main__":
    JSON_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)

    p.connect(p.GUI)

    scene_keys = []
    for family in FAMILIES:
        for i in range(5):
            scene = generate_scene(family=family, seed=i)
            save_scene_json(scene, JSON_DIR / f"{family}_{i:03d}.json")
            setup_pybullet_scene(scene)
            save_pybullet_screenshot(scene, IMAGES_DIR / f"{family}_{i:03d}.png")
            scene_keys.append((family, i))

    scene_index = 0
    family_to_view, seed_to_view = scene_keys[scene_index]
    scene = build_pybullet_scene(family=family_to_view, seed=seed_to_view)

    save_scene_json(scene, JSON_DIR / f"{family_to_view}_{seed_to_view:03d}.json")

    print(f"Loaded scene family: {scene['family']}")
    print(f"Obstacles: {len(scene['obstacles'])}")
    print(f"Saved 5 scenes per family in: {JSON_DIR}")
    print(f"Saved PyBullet screenshots in: {IMAGES_DIR}")
    print("Controls: press N for next scene, P for previous scene, Q to quit.")

    while True:
        keys = p.getKeyboardEvents()

        if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
            break

        if ord("n") in keys and keys[ord("n")] & p.KEY_WAS_TRIGGERED:
            scene_index = (scene_index + 1) % len(scene_keys)
            family_to_view, seed_to_view = scene_keys[scene_index]
            scene = build_pybullet_scene(family=family_to_view, seed=seed_to_view)
            print(f"Loaded scene: {family_to_view}_{seed_to_view:03d}")

        if ord("p") in keys and keys[ord("p")] & p.KEY_WAS_TRIGGERED:
            scene_index = (scene_index - 1) % len(scene_keys)
            family_to_view, seed_to_view = scene_keys[scene_index]
            scene = build_pybullet_scene(family=family_to_view, seed=seed_to_view)
            print(f"Loaded scene: {family_to_view}_{seed_to_view:03d}")

        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()
