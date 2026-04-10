import os
import numpy as np
import csv
from envs.adapter import load_and_adapt_scene
from planners.astar import astar_collision_free
from planners.semantic_astar import semantic_astar
from planners.surp import run_surp
from envs.generator import generate_trap_scene
from eval.metrics import (success_rate, avg_path_length, avg_rotation_cost,
                           belief_error, trap_recovery_rate, forbidden_rate,
                           fragile_rate, replan_count, print_summary,
                           measure_planning_time, delta_noise)
from eval.ablations import (ablation_cost_function, ablation_belief_update,
                             ablation_replan_trigger, ablation_noise_level,
                             ablation_likelihood_misspec)

SCENES_DIR = "env/data/scenes"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

scene_files = sorted(f for f in os.listdir(SCENES_DIR) if f.endswith(".json"))

# ── Load all scenes ──────────────────────────────────────────────
print("Loading scenes...")
scenes, true_classes_list = [], []
for fname in scene_files:
    seed = int(fname.replace("scene_", "").replace(".json", ""))
    s = load_and_adapt_scene(os.path.join(SCENES_DIR, fname),
                             noise_level="medium", seed=seed)
    scenes.append(s)
    true_classes_list.append([o["true_class"] for o in s["obstacles"]])

# ── Run all 4 planners ───────────────────────────────────────────
print("Running planners on 100 scenes...")
b1, b2, b3, surp = [], [], [], []

for i, (scene, true_classes) in enumerate(zip(scenes, true_classes_list)):
    p1 = astar_collision_free(scene)
    p2 = semantic_astar(scene, use_expected=False)
    p3 = semantic_astar(scene, use_expected=True)
    sr = run_surp(scene)

    b1.append({"success": p1 is not None, "damage": 0,
                "path": p1 or [], "contact_log": [], "replans": 0})
    b2.append({"success": p2 is not None, "damage": 0,
                "path": p2 or [], "contact_log": [], "replans": 0})
    b3.append({"success": p3 is not None, "damage": 0,
                "path": p3 or [], "contact_log": [], "replans": 0})
    surp.append(sr)

    print(f"  scene_{i:03d}  SURP={'OK' if sr['success'] else 'FAIL'}  "
          f"damage={sr['damage']}  replans={sr['replans']}")

# ── Print main results ───────────────────────────────────────────
print("\n\n" + "="*60)
print("MAIN RESULTS TABLE")
print("="*60)
print_summary(b1,   "Baseline 1: Collision-free A*")
print_summary(b2,   "Baseline 2: Deterministic semantic")
print_summary(b3,   "Baseline 3: SURP-NoUpdate")
print_summary(surp, "SURP: Full closed-loop", true_classes_list)

# ── Planning time ────────────────────────────────────────────────
print("\n\n" + "="*60)
print("PLANNING TIME (first 10 scenes)")
print("="*60)
times = [measure_planning_time(astar_collision_free, s) for s in scenes[:10]]
print(f"  B1 mean: {np.mean([t['mean'] for t in times]):.3f}s  "
      f"p95: {np.max([t['p95'] for t in times]):.3f}s")
times = [measure_planning_time(lambda s: run_surp(s), s) for s in scenes[:10]]
print(f"  SURP mean: {np.mean([t['mean'] for t in times]):.3f}s  "
      f"p95: {np.max([t['p95'] for t in times]):.3f}s")

# ── Trap recovery ────────────────────────────────────────────────
print("\n\n" + "="*60)
print("SEMANTIC TRAP RECOVERY RATE")
print("="*60)
trap_scenes = [generate_trap_scene(seed=i) for i in range(20)]
trap_results = [run_surp(s) for s in trap_scenes]
print(f"  SURP trap recovery: {trap_recovery_rate(trap_results):.1%}")

# ── Noise comparison ─────────────────────────────────────────────
print("\n\nNOISE LEVEL COMPARISON:")
low_results, high_results = [], []
for fname in scene_files:
    seed = int(fname.replace("scene_", "").replace(".json", ""))
    sl = load_and_adapt_scene(os.path.join(SCENES_DIR, fname), "low", seed)
    sh = load_and_adapt_scene(os.path.join(SCENES_DIR, fname), "high", seed)
    low_results.append(run_surp(sl))
    high_results.append(run_surp(sh))
print(f"  Δnoise = {delta_noise(low_results, high_results):.1%}")

# ── Ablations ────────────────────────────────────────────────────
ablation_cost_function(scenes[:30], true_classes_list[:30])
ablation_belief_update(scenes[:30], true_classes_list[:30])
ablation_replan_trigger(scenes[:30], true_classes_list[:30])
ablation_noise_level(scene_files, SCENES_DIR)
ablation_likelihood_misspec(scenes[:30], true_classes_list[:30])

# ── Save CSV ─────────────────────────────────────────────────────
rows = []
for i, fname in enumerate(scene_files):
    rows.append({
        "scene": fname.replace(".json", ""),
        "b1_success": b1[i]["success"],
        "b2_success": b2[i]["success"],
        "b3_success": b3[i]["success"],
        "surp_success": surp[i]["success"],
        "surp_damage": surp[i]["damage"],
        "surp_replans": surp[i]["replans"],
        "surp_path_len": round(avg_path_length([surp[i]]), 3),
        "belief_err": round(belief_error(
            surp[i]["posteriors"], true_classes_list[i]), 4)
        if surp[i]["posteriors"] else "N/A"
    })

with open(os.path.join(OUTPUT_DIR, "full_results.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\n\nDone. Full results saved to {OUTPUT_DIR}/full_results.csv")