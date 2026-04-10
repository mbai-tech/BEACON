from envs.generator import generate_trap_scene
from planners.astar import astar_collision_free
from planners.semantic_astar import semantic_astar
from planners.surp import run_surp
from eval.plots import plot_scene

# Use semantic planner to visualize — B1 won't go through any obstacle
scene0 = generate_trap_scene(seed=0)
path0 = semantic_astar(scene0, use_expected=False)  # B2 will go through safe-labeled obstacles
plot_scene(scene0, path0, title="Trap scene layout (B2 path through trap)",
           save_path="results/trap_layout.png")
print("Saved trap layout to results/trap_layout.png")

trap_results = {"b1": [], "b2": [], "surp": []}

for seed in range(20):
    scene = generate_trap_scene(seed=seed)

    p1 = astar_collision_free(scene)
    p2 = semantic_astar(scene, use_expected=False)
    sr = run_surp(scene)

    trap_results["b1"].append(p1 is not None)
    trap_results["b2"].append(p2 is not None)
    trap_results["surp"].append(sr["replans"] > 0)

    print(f"Trap {seed:02d}  B1={'OK' if p1 else 'FAIL'}  "
          f"B2={'OK' if p2 else 'FAIL'}  "
          f"SURP rerouted={'YES' if sr['replans'] > 0 else 'NO'}  "
          f"damage={sr['damage']}")

print(f"\nSURP trap recovery rate: "
      f"{sum(trap_results['surp'])}/20 = "
      f"{sum(trap_results['surp'])/20:.1%}")