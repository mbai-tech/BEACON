import time
from envs.adapter import load_family_scene
from planners.astar import astar_collision_free
from planners.semantic_astar import semantic_astar

scene = load_family_scene(
    "/Users/rheas/SURP/enviornment/data/scenes/sparse_clutter_000_seed1060967585.json",
    noise_level="medium"
)

print("Workspace:", scene["workspace"])
print("Start:", scene["start"])
print("Goal:", scene["goal"])
print("Num obstacles:", len(scene["obstacles"]))
print("Obstacle vertex counts:", [len(o["vertices"]) for o in scene["obstacles"]])

print("\nTrying collision-free A*...")
t0 = time.time()
path = astar_collision_free(scene)
print(f"Done in {time.time()-t0:.1f}s — {'found' if path else 'NO PATH'}")