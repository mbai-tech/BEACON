"""Compare BEACON with handcrafted reconciler vs trained ML model.

Runs the same scenes under both conditions and prints a summary table.

Usage:
    python -m beacon.core.ml.evaluate_ml --trials 20 --configs D-M D-U M-M M-U
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
BEACON_DIR = REPO_ROOT / "beacon"
for p in (str(REPO_ROOT), str(BEACON_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

import beacon.core.planner as _planner_mod
from beacon.core.planner import run_online_surp_push
from beacon.core.scene_configs import generate_config_environment
from beacon.core.constants import SCENE_CONFIGS
from beacon.core.ml.push_policy import PushAvoidPolicy, COLLECT


def _compute_metrics(result, elapsed: float) -> dict:
    path = np.array(result.path)
    goal = np.array(result.scene["goal"][:2])
    start = np.array(result.scene["start"][:2])
    path_len = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    straight = float(np.linalg.norm(goal - start))
    efficiency = straight / path_len if path_len > 0 else 0.0
    dangerous = sum(
        1 for e in result.contact_log
        if "fragile" in e or "forbidden" in e
    )
    return {
        "success":           int(result.success),
        "path_efficiency":   efficiency,
        "dangerous_contacts": dangerous,
        "steps":             len(result.path),
        "planning_ms":       elapsed * 1000,
    }


def _summarise(rows: list[dict], label: str) -> dict:
    n = len(rows)
    sr  = sum(r["success"] for r in rows) / n
    eff = np.mean([r["path_efficiency"] for r in rows])
    dc  = np.mean([r["dangerous_contacts"] for r in rows])
    ms  = np.mean([r["planning_ms"] for r in rows])
    print(f"  {label:<20}  success={sr:.1%}  efficiency={eff:.3f}"
          f"  dangerous={dc:.2f}  ms/run={ms:.0f}  (n={n})")
    return {"label": label, "success_rate": sr, "efficiency": eff,
            "dangerous": dc, "n": n}


def run_condition(
    condition: str,
    configs: list[str],
    n_trials: int,
    seeds: list[int],
) -> list[dict]:
    """Run one condition (baseline or ml) and return per-run metrics."""
    policy: PushAvoidPolicy = _planner_mod._push_policy
    if condition == "baseline":
        saved_model = policy._model
        policy._model = None   # force handcrafted formula

    rows = []
    total = len(configs) * n_trials
    done  = 0
    for cfg in configs:
        for t in range(n_trials):
            seed = seeds[done % len(seeds)]
            import random; random.seed(seed + done)
            scene = generate_config_environment(cfg)
            t0 = time.perf_counter()
            result = run_online_surp_push(scene, max_steps=300)
            elapsed = time.perf_counter() - t0
            m = _compute_metrics(result, elapsed)
            m["config"] = cfg
            m["trial"]  = t
            rows.append(m)
            done += 1
            if done % 10 == 0:
                print(f"    [{condition}] {done}/{total}")

    if condition == "baseline":
        policy._model = saved_model   # restore

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int, default=15)
    parser.add_argument("--configs", nargs="+",
                        default=["D-M", "D-U", "M-M", "M-U"])
    args = parser.parse_args()

    import beacon.core.ml.push_policy as pp_mod
    import importlib, builtins
    # disable logging during eval so we don't pollute the training log
    pp_mod.COLLECT = False

    seeds = list(range(1000, 1000 + args.trials * len(args.configs)))

    print(f"\nRunning {args.trials} trials × {len(args.configs)} configs "
          f"× 2 conditions = "
          f"{args.trials * len(args.configs) * 2} total runs\n")

    print("── Baseline (handcrafted formula) ──────────────────────────────")
    base_rows = run_condition("baseline", args.configs, args.trials, seeds)

    print("\n── ML model (trained push policy) ──────────────────────────────")
    ml_rows = run_condition("ml", args.configs, args.trials, seeds)

    print("\n═══ Summary ════════════════════════════════════════════════════")
    b = _summarise(base_rows, "Baseline")
    m = _summarise(ml_rows,   "ML model")

    # per-config breakdown
    print("\nPer-config efficiency:")
    for cfg in args.configs:
        b_eff = np.mean([r["path_efficiency"] for r in base_rows if r["config"] == cfg])
        m_eff = np.mean([r["path_efficiency"] for r in ml_rows   if r["config"] == cfg])
        delta = m_eff - b_eff
        arrow = "▲" if delta > 0 else "▼"
        print(f"  {cfg}  baseline={b_eff:.3f}  ml={m_eff:.3f}  {arrow}{abs(delta):.3f}")

    pp_mod.COLLECT = True   # restore


if __name__ == "__main__":
    main()
