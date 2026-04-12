import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import copy
from shapely.geometry import Polygon, Point
from envs.adapter import load_family_scene
from geometry.polygons import make_robot_polygon, transform_polygon, contact_area
from planners.semantic_astar import semantic_astar
from planners.cibp import cibp_update
from envs.contact_simulator import simulate_contact

SCENES_DIR = "enviornment/data/scenes"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

scene_files = sorted(f for f in os.listdir(SCENES_DIR) if f.endswith(".json"))[:N]

for fname in scene_files:
    scene = load_family_scene(os.path.join(SCENES_DIR, fname), noise_level="medium")
    scene_id = fname.replace(".json", "")

    robot_base = make_robot_polygon("rectangle")
    obstacles = [Polygon(o["vertices"]) for o in scene["obstacles"]]
    true_classes = [o["true_class"] for o in scene["obstacles"]]
    posteriors = [list(o["prior"]) for o in scene["obstacles"]]
    TRUE_COSTS = {"safe": 1, "movable": 3, "fragile": 15, "forbidden": 1000}

    gx, gy, _ = scene["goal"]
    current_pose = tuple(scene["start"])
    known_indices = set()
    r_sense = 2.0

    def sense(pose):
        cx, cy, _ = pose
        pt = Point(cx, cy)
        return [i for i, obs in enumerate(obstacles)
                if i not in known_indices and obs.distance(pt) <= r_sense]

    known_indices.update(sense(current_pose))

    def make_scene():
        s = copy.deepcopy(scene)
        s["obstacles"] = [{**s["obstacles"][i], "prior": posteriors[i]}
                          for i in sorted(known_indices)]
        return s

    plan = semantic_astar(make_scene(), use_expected=True)
    if plan is None:
        print(f"{scene_id}: initial plan FAILED")
        continue

    plan_idx = 0
    step = 0
    exit_reason = "max_steps"

    while step < 2000:
        cx, cy, _ = current_pose
        if np.hypot(cx - gx, cy - gy) < 0.2:
            exit_reason = "success"
            break

        if plan_idx >= len(plan):
            exit_reason = "plan_exhausted"
            break

        current_pose = plan[plan_idx]
        plan_idx += 1
        step += 1

        newly_seen = sense(current_pose)
        triggered_replan = bool(newly_seen)
        if newly_seen:
            known_indices.update(newly_seen)

        robot_poly = transform_polygon(robot_base, *current_pose)
        forbidden_contact = False
        for i, obs in enumerate(obstacles):
            area = contact_area(robot_poly, obs)
            if area > 0:
                cls = true_classes[i]
                if cls == "forbidden":
                    forbidden_contact = True
                if cls == "fragile":
                    print(f"  step {step}: FRAGILE contact (obs {i}), plan_idx={plan_idx}/{len(plan)}")
                if i not in known_indices:
                    known_indices.add(i)
                    triggered_replan = True
                outcome = simulate_contact(cls)
                new_post, replan = cibp_update(posteriors[i], outcome, 0.1)
                posteriors[i] = new_post
                if replan:
                    triggered_replan = True

        if forbidden_contact:
            exit_reason = "forbidden_contact"
            break

        if triggered_replan:
            new_plan = semantic_astar(make_scene(), use_expected=True)
            if new_plan is None:
                print(f"  step {step}: REPLAN FAILED (plan_idx={plan_idx}/{len(plan)}, known={len(known_indices)})")
            else:
                plan = new_plan
                plan_idx = 0

    print(f"{scene_id}: exit={exit_reason}  steps={step}  known={len(known_indices)}/{len(obstacles)}")
