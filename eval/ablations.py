import numpy as np
import time
from envs.adapter import load_and_adapt_scene
from planners.surp import run_surp
from planners.cibp import cibp_update
from eval.metrics import success_rate, avg_path_length, belief_error, print_summary
import os, copy

CLASSES = ["safe", "movable", "fragile", "forbidden"]

# ─── Ablation 1: Cost function variant ─────────────────────────

def ablation_cost_function(scenes, true_classes_list):
    """Compare no-semantic vs deterministic vs expected vs robust cost."""
    from planners.semantic_astar import semantic_astar
    from planners.astar import astar_collision_free

    print("\n\n" + "="*60)
    print("ABLATION 1: Cost function variant")
    print("="*60)

    # No semantic cost (collision-free)
    results = [{"success": astar_collision_free(s) is not None,
                "damage": 0, "path": astar_collision_free(s) or [],
                "contact_log": [], "replans": 0}
               for s in scenes]
    print_summary(results, "No semantic cost (collision-free)")

    # Deterministic
    results = [{"success": semantic_astar(s, use_expected=False) is not None,
                "damage": 0,
                "path": semantic_astar(s, use_expected=False) or [],
                "contact_log": [], "replans": 0}
               for s in scenes]
    print_summary(results, "Deterministic (argmax of prior)")

    # Expected
    results = [run_surp(s, use_cibp=False) for s in scenes]
    print_summary(results, "Expected cost (SURP-NoUpdate)", true_classes_list)

    # Full SURP
    results = [run_surp(s, use_cibp=True) for s in scenes]
    print_summary(results, "Full SURP (expected + CIBP)", true_classes_list)

# ─── Ablation 2: Belief update on vs off ───────────────────────

def ablation_belief_update(scenes, true_classes_list):
    print("\n\n" + "="*60)
    print("ABLATION 2: Belief update (SURP-NoUpdate vs SURP)")
    print("="*60)

    no_update = [run_surp(s, use_cibp=False) for s in scenes]
    with_update = [run_surp(s, use_cibp=True) for s in scenes]

    print_summary(no_update, "SURP-NoUpdate (no CIBP)", true_classes_list)
    print_summary(with_update, "SURP (with CIBP)", true_classes_list)

    be_no = np.mean([belief_error(r["posteriors"], tc)
                     for r, tc in zip(no_update, true_classes_list) if r["posteriors"]])
    be_yes = np.mean([belief_error(r["posteriors"], tc)
                      for r, tc in zip(with_update, true_classes_list) if r["posteriors"]])
    print(f"\n  BeliefErr without CIBP: {be_no:.4f}")
    print(f"  BeliefErr with CIBP:    {be_yes:.4f}")
    print(f"  Improvement:            {be_no - be_yes:.4f}")

# ─── Ablation 3: Replanning trigger ────────────────────────────

def ablation_replan_trigger(scenes, true_classes_list):
    print("\n\n" + "="*60)
    print("ABLATION 3: Replanning trigger strategy")
    print("="*60)

    # KL threshold (default SURP)
    r_kl = [run_surp(s, kl_threshold=0.1) for s in scenes]
    print_summary(r_kl, "KL-threshold (default, δ=0.1)", true_classes_list)

    # Very sensitive trigger
    r_sensitive = [run_surp(s, kl_threshold=0.01) for s in scenes]
    print_summary(r_sensitive, "Very sensitive (δ=0.01)", true_classes_list)

    # Very conservative trigger
    r_conservative = [run_surp(s, kl_threshold=0.5) for s in scenes]
    print_summary(r_conservative, "Conservative (δ=0.5)", true_classes_list)

    # No replan (fixed: only plan once)
    r_none = [run_surp(s, kl_threshold=999) for s in scenes]
    print_summary(r_none, "No replanning (δ=999)", true_classes_list)

# ─── Ablation 4: Noise level ────────────────────────────────────

def ablation_noise_level(scene_files, scenes_dir):
    from envs.adapter import load_family_scene
    print("\n\n" + "="*60)
    print("ABLATION 4: Noise level degradation")
    print("="*60)

    noise_levels = ["low", "medium", "high", "adversarial"]
    for noise in noise_levels:
        scenes = []
        true_classes_list = []
        for fname in scene_files:
            path = os.path.join(scenes_dir, fname)
            s = load_family_scene(path, noise_level=noise)
            scenes.append(s)
            true_classes_list.append([o["true_class"] for o in s["obstacles"]])

        results = [run_surp(s) for s in scenes]
        sr = success_rate(results)
        dmg = np.mean([r["damage"] for r in results])
        be = np.mean([belief_error(r["posteriors"], tc)
                      for r, tc in zip(results, true_classes_list)
                      if r["posteriors"]])
        print(f"  {noise:12s}  success={sr:.1%}  "
              f"damage={dmg:.4f}  belief_err={be:.4f}")
# ─── Ablation 5: Likelihood misspecification ────────────────────

def ablation_likelihood_misspec(scenes, true_classes_list):
    print("\n\n" + "="*60)
    print("ABLATION 5: Likelihood misspecification")
    print("="*60)

    import planners.cibp as cibp_module
    original_L = cibp_module.LIKELIHOOD.copy()

    off_diagonal_masses = [0.02, 0.10, 0.20, 0.35]
    for mass in off_diagonal_masses:
        # Build a new likelihood table with more off-diagonal mass
        L = np.full((4, 4), mass / 3)
        np.fill_diagonal(L, 1.0 - mass)
        cibp_module.LIKELIHOOD = L

        results = [run_surp(s) for s in scenes]
        sr = success_rate(results)
        dmg = np.mean([r["damage"] for r in results])
        be = np.mean([belief_error(r["posteriors"], tc)
                      for r, tc in zip(results, true_classes_list) if r["posteriors"]])
        print(f"  off-diag mass={mass:.2f}  "
              f"success={sr:.1%}  damage={dmg:.4f}  belief_err={be:.4f}")

    # Restore original
    cibp_module.LIKELIHOOD = original_L
    print("  (likelihood restored to original)")