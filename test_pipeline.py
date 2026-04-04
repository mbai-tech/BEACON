from envs.generator import generate_scene
from planners.astar import astar_collision_free
from eval.plots import plot_scene

scene = generate_scene(n_obstacles=8, seed=1)
path = astar_collision_free(scene)
plot_scene(scene, path, title="Baseline 1: Collision-free A*")