from envs.generator import generate_scene
from planners.semantic_astar import semantic_astar
from eval.plots import plot_scene

scene = generate_scene(n_obstacles=8, seed=5)

# Baseline 2: deterministic (argmax of prior)
path_b2 = semantic_astar(scene, use_expected=False)
plot_scene(scene, path_b2, title="Baseline 2: Deterministic semantic A*")
input("Press Enter for next plot...")

# Baseline 3: expected cost (SURP-NoUpdate)
path_b3 = semantic_astar(scene, use_expected=True)
plot_scene(scene, path_b3, title="Baseline 3: SURP-NoUpdate")