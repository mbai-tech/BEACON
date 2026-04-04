import numpy as np
import copy
from shapely.geometry import Polygon
from geometry.polygons import make_robot_polygon, transform_polygon, contact_area
from planners.semantic_astar import semantic_astar
from planners.cibp import cibp_update
from envs.contact_simulator import simulate_contact

def run_surp(scene, kl_threshold=0.1, max_steps=2000, robot_shape="rectangle"):
    robot_base = make_robot_polygon(robot_shape)
    obstacles = [Polygon(o["vertices"]) for o in scene["obstacles"]]
    true_classes = [o["true_class"] for o in scene["obstacles"]]
    posteriors = [list(o["prior"]) for o in scene["obstacles"]]

    gx, gy, _ = scene["goal"]
    current_pose = tuple(scene["start"])

    path_taken = [current_pose]
    total_damage = 0.0
    replan_count = 0
    success = False

    working_scene = _scene_with_posteriors(scene, posteriors)
    plan = semantic_astar(working_scene, use_expected=True)

    if plan is None:
        return {"success": False, "damage": 0, "replans": 0, "path": []}

    plan_idx = 0
    step = 0

    while step < max_steps:
        cx, cy, _ = current_pose
        if np.hypot(cx - gx, cy - gy) < 0.2:
            success = True
            break

        if plan_idx >= len(plan):
            break

        next_pose = plan[plan_idx]
        plan_idx += 1
        current_pose = next_pose
        path_taken.append(current_pose)
        step += 1

        robot_poly = transform_polygon(robot_base, *current_pose)
        triggered_replan = False

        for i, obs in enumerate(obstacles):
            area = contact_area(robot_poly, obs)
            if area > 0:
                outcome = simulate_contact(true_classes[i])
                true_costs = {"safe": 1, "movable": 3, "fragile": 15, "forbidden": 1000}
                total_damage += true_costs[true_classes[i]] * area
                new_post, replan = cibp_update(posteriors[i], outcome, kl_threshold)
                posteriors[i] = new_post
                if replan:
                    triggered_replan = True

        if triggered_replan:
            replan_count += 1
            working_scene = _scene_with_posteriors(scene, posteriors)
            new_plan = semantic_astar(working_scene, use_expected=True)
            if new_plan:
                plan = new_plan
                plan_idx = 0

    return {
        "success": success,
        "damage": round(total_damage, 4),
        "replans": replan_count,
        "path": path_taken,
        "posteriors": posteriors
    }

def _scene_with_posteriors(scene, posteriors):
    s = copy.deepcopy(scene)
    for i, obs in enumerate(s["obstacles"]):
        obs["prior"] = posteriors[i]
    return s