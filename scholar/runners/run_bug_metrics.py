# -*- coding: utf-8 -*-
"""Run Bug1 and Bug2 on the pre-generated scholar scenes and report all metrics."""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scholar.algorithms.bug  import run_bug
from scholar.algorithms.bug2 import run_bug2
from scholar.utils.episode_metrics        import run_all_metrics, validate_metrics
from scholar.core.models         import ContactEvent, ReplanEvent, RolloutRecord

SCENES_DIR = Path(__file__).resolve().parent.parent / "scholar" / "environment" / "data" / "scenes"
FAMILIES   = ("sparse", "cluttered", "collision_required", "collision_shortcut")


# ── Scene loader ──────────────────────────────────────────────────────────────

def _load_scene(family: str, idx: int) -> dict:
    path = SCENES_DIR / family / f"scene_{idx:03d}.json"
    with path.open() as f:
        return json.load(f)


# ── OnlineSurpResult → RolloutRecord adapter ─────────────────────────────────

def _infer_headings_deg(path: list[tuple]) -> list[float]:
    """Infer robot heading at each step from consecutive (x, y) displacements."""
    if not path:
        return []
    headings: list[float] = []
    last = 0.0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        if dx != 0.0 or dy != 0.0:
            last = math.degrees(math.atan2(dy, dx))
        headings.append(last)
    headings.append(last)
    return headings


def _adapt(result, scene: dict, planner: str) -> RolloutRecord:
    """Convert a bug OnlineSurpResult + source scene to a RolloutRecord.

    Bug algorithms have no belief model, no battery, and no replan events.
    Fields that cannot be populated are set to their empty / neutral defaults.
    """
    path     = result.path          # list[(x, y)]
    headings = _infer_headings_deg(path)
    trajectory = [(x, y, h) for (x, y), h in zip(path, headings)]

    true_classes: dict[int, str] = {
        obs["id"]: obs.get("true_class", obs.get("class_true", "safe"))
        for obs in scene.get("obstacles", [])
    }

    total_steps = max(len(path) - 1, 0)
    # Bug algorithms traverse obstacles but don't record contacts semantically.
    # We infer contacts from sensed obstacle IDs: any sensed obstacle whose
    # polygon the robot path passes through is a contact candidate, but the
    # algorithms do not expose collision normals, areas, or per-step battery.
    # We therefore leave contact_events empty and rely on path/success metrics.
    contact_events: list[ContactEvent] = []
    replan_events:  list[ReplanEvent]  = []

    gx, gy, gtheta = (scene["goal"] + [0.0])[:3]
    fx, fy = path[-1] if path else (0.0, 0.0)
    ftheta = headings[-1] if headings else 0.0

    return RolloutRecord(
        scene_family   = result.family,
        planner        = planner,
        success        = result.success,
        trajectory     = trajectory,
        contact_events = contact_events,
        replan_events  = replan_events,
        final_beliefs  = {},
        true_classes   = true_classes,
        battery_history = [1.0] * (total_steps + 1),   # no battery model
        stuck_events   = 0,
        total_steps    = total_steps,
        goal_reached   = result.success,
        final_pose     = (fx, fy, ftheta),
        goal_pose      = (gx, gy, gtheta),
    )


# ── Batch runner ──────────────────────────────────────────────────────────────

def _run_one(planner_fn, planner_name: str, family: str, idx: int) -> RolloutRecord:
    scene  = _load_scene(family, idx)
    result = planner_fn(scene)
    return _adapt(result, scene, planner_name)


def collect_records(
    n_scenes: int,
    workers: int,
) -> list[RolloutRecord]:
    planners = [("bug1", run_bug), ("bug2", run_bug2)]
    tasks: list[tuple] = [
        (fn, name, family, idx)
        for name, fn  in planners
        for family    in FAMILIES
        for idx       in range(n_scenes)
    ]

    records: list[RolloutRecord] = [None] * len(tasks)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one, fn, name, family, idx): i
            for i, (fn, name, family, idx) in enumerate(tasks)
        }
        done = 0
        for fut in as_completed(futures):
            records[futures[fut]] = fut.result()
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} episodes complete", flush=True)

    return records


# ── Pretty printer ────────────────────────────────────────────────────────────

def _fmt(v, decimals: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def print_summary(results: dict) -> None:
    by_planner = results["by_planner"]
    planners   = sorted(by_planner)

    print("\n" + "═" * 60)
    print("BUG ALGORITHM METRICS SUMMARY")
    print("═" * 60)

    # Per-planner aggregate
    agg_keys = [
        ("success_rate",            "Success rate"),
        ("mean_path_length",        "Mean path length"),
        ("mean_rotation_cost",      "Mean rotation cost"),
        ("mean_semantic_damage",    "Mean semantic damage"),
        ("mean_replan_count",       "Mean replan count"),
        ("forbidden_rate",          "Forbidden contact rate"),
        ("fragile_rate",            "Fragile contact rate"),
    ]
    col_w = 26
    header = f"{'Metric':<{col_w}}" + "".join(f"{p:>14}" for p in planners)
    print("\n" + header)
    print("─" * len(header))
    for key, label in agg_keys:
        row = f"{label:<{col_w}}"
        for p in planners:
            v = by_planner[p]["aggregate"].get(key)
            row += f"{_fmt(v):>14}"
        print(row)

    # Per-family success rate
    all_families = sorted({f for p in planners for f in by_planner[p]["by_family"]})
    print(f"\n{'Family success rates':}")
    print(f"  {'Family':<22}" + "".join(f"{p:>14}" for p in planners))
    print("  " + "─" * (22 + 14 * len(planners)))
    for fam in all_families:
        row = f"  {fam:<22}"
        for p in planners:
            v = by_planner[p]["by_family"].get(fam, {}).get("success_rate")
            row += f"{_fmt(v):>14}"
        print(row)

    # Robustness block
    rob = results["robustness"]
    print(f"\n{'Robustness':}")
    for k, v in rob.items():
        print(f"  {k:<30} {_fmt(v)}")

    print("═" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bug1/Bug2 with full metrics pipeline.")
    parser.add_argument("--scenes",  type=int, default=20,  help="Scenes per family (max 100)")
    parser.add_argument("--workers", type=int, default=8,   help="Parallel workers")
    args = parser.parse_args()

    n = min(args.scenes, 100)
    print(f"Running Bug1 + Bug2 on {n} scenes × {len(FAMILIES)} families "
          f"= {2 * n * len(FAMILIES)} episodes ({args.workers} workers)")

    records = collect_records(n_scenes=n, workers=args.workers)

    print("\nComputing metrics …")
    results = run_all_metrics(records, weight_histories={})

    print_summary(results)

    print("\nValidation checks:")
    validate_metrics(results)


if __name__ == "__main__":
    main()
