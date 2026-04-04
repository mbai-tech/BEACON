from envs.generator import generate_scene
from planners.astar import astar_collision_free

scene = generate_scene(n_obstacles=6, seed=42)
path = astar_collision_free(scene)

if path:
    print(f"Path found! {len(path)} steps")
    print("First pose:", path[0])
    print("Last pose:", path[-1])
else:
    print("No path found")