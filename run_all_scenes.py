import os
import json
from envs.adapter import load_and_adapt_scene
from planners.astar import astar_collision_free
from planners.semantic_astar import semantic_astar
from planners.surp import run_surp
from eval.plots import plot_scene

SCENES_DIR = "env/data/scenes"   # path inside your SURP-main folder
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

scene_files = sorted(f for f in os.listdir(SCENES_DIR) if f.endswith(".json"))

results = []

for fname in scene_files:
    path = os.path.join(SCENES_DIR, fname)
    scene = load_and_adapt_scene(path, noise_level="medium")
    scene_id = fname.replace(".json", "")

    # Run all 3 planners
    path_b1 = astar_collision_free(scene)
    path_b2 = semantic_astar(scene, use_expected=False)
    surp_result = run_surp(scene)

    row = {
        "scene": scene_id,
        "b1_success": path_b1 is not None,
        "b2_success": path_b2 is not None,
        "surp_success": surp_result["success"],
        "surp_damage": surp_result["damage"],
        "surp_replans": surp_result["replans"],
    }
    results.append(row)
    print(f"{scene_id}  B1={'OK' if row['b1_success'] else 'FAIL'}  "
          f"B2={'OK' if row['b2_success'] else 'FAIL'}  "
          f"SURP={'OK' if row['surp_success'] else 'FAIL'}  "
          f"damage={row['surp_damage']}  replans={row['surp_replans']}")

    # Save a plot of the SURP path for this scene
    plot_scene(scene, surp_result["path"],
               title=f"SURP — {scene_id}",
               save_path=os.path.join(OUTPUT_DIR, f"{scene_id}_surp.png"))

# Save summary CSV
import csv
with open(os.path.join(OUTPUT_DIR, "results_summary.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone. Results saved to {OUTPUT_DIR}/results_summary.csv")