import os
import numpy as np
import csv
from envs.adapter import load_family_scene
from planners.astar import astar_collision_free
from planners.semantic_astar import semantic_astar
from planners.surp import run_surp
from eval.metrics import (success_rate, avg_path_length, belief_error,
                           forbidden_rate, fragile_rate, replan_count,
                           print_summary, trap_recovery_rate)
from eval.ablations import (ablation_cost_function, ablation_belief_update, ablation_replan_trigger,
                             ablation_noise_level, ablation_likelihood_misspec)

SCENES_DIR = "/Users/rheas/SURP/enviornment/data/scenes"
OUTPUT_DIR = "results/family_experiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAMILIES = ["sparse_clutter", "dense_clutter", "narrow_passage",
            "semantic_trap", "perturbed"]

all_files = sorted(f for f in os.listdir(SCENES_DIR) if f.endswith(".json"))

# Group files by family
family_files = {fam: [] for fam in FAMILIES}
for fname in all_files:
    for fam in FAMILIES:
        if fname.startswith(fam):
            family_files[fam].append(fname)
            break

# ── Run all planners per family ──────────────────────────────────
all_rows = []

for family in FAMILIES:
    files = family_files[family]
    print(f"\n{'='*60}")
    print(f"FAMILY: {family.upper()} ({len(files)} scenes)")
    print(f"{'='*60}")

    scenes, true_classes_list = [], []
    for fname in files:
        s = load_family_scene(os.path.join(SCENES_DIR, fname),
                              noise_level="medium")
        scenes.append(s)
        true_classes_list.append([o["true_class"] for o in s["obstacles"]])

    b1, b2, b3, surp = [], [], [], []

    scenes = scenes[:20]
    true_classes_list = true_classes_list[:20]
    files = files[:20]
    
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

        print(f"  {files[i][:40]:40s}  "
              f"SURP={'OK' if sr['success'] else 'FAIL'}  "
              f"damage={sr['damage']:.4f}  replans={sr['replans']}")

    # Print summary for this family
    print_summary(b1,   f"  B1 ({family})")
    print_summary(b2,   f"  B2 ({family})")
    print_summary(b3,   f"  B3 ({family})")
    print_summary(surp, f"  SURP ({family})", true_classes_list)

    # Trap recovery rate for semantic_trap family
    if family == "semantic_trap":
        rate = trap_recovery_rate(surp)
        print(f"\n  *** SEMANTIC TRAP RECOVERY RATE: {rate:.1%} ***")

    # Save per-family CSV
    rows = []
    for i, fname in enumerate(files):
        rows.append({
            "family": family,
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
    all_rows.extend(rows)

# ── Ablations (on sparse_clutter for speed) ──────────────────────
print("\n\nRunning ablations on sparse_clutter scenes...")
sparse_files = family_files["sparse_clutter"][:30]
sparse_scenes, sparse_classes = [], []
for fname in sparse_files:
    s = load_family_scene(os.path.join(SCENES_DIR, fname), "medium")
    sparse_scenes.append(s)
    sparse_classes.append([o["true_class"] for o in s["obstacles"]])

ablation_cost_function(sparse_scenes, sparse_classes)
ablation_belief_update(sparse_scenes, sparse_classes)
ablation_replan_trigger(sparse_scenes, sparse_classes)
ablation_likelihood_misspec(sparse_scenes, sparse_classes)

# Noise ablation
print("\n\n" + "="*60)
print("ABLATION: Noise level (sparse_clutter)")
print("="*60)
for noise in ["low", "medium", "high", "adversarial"]:
    ns, nc = [], []
    for fname in sparse_files:
        s = load_family_scene(os.path.join(SCENES_DIR, fname), noise)
        ns.append(s)
        nc.append([o["true_class"] for o in s["obstacles"]])
    results = [run_surp(s) for s in ns]
    sr = success_rate(results)
    dmg = np.mean([r["damage"] for r in results])
    be = np.mean([belief_error(r["posteriors"], tc)
                  for r, tc in zip(results, nc) if r["posteriors"]])
    print(f"  {noise:12s}  success={sr:.1%}  "
          f"damage={dmg:.4f}  belief_err={be:.4f}")

# ── Save master CSV ──────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "family_results.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\n\nDone. Results saved to {OUTPUT_DIR}/family_results.csv")