"""
surp_push.py
Online SURP push controller: hybrid goal-seek / avoid / push algorithm.
"""

import copy
import numpy as np

from surp_motion_planning.constants import DEFAULT_SENSING_RANGE, SAFE_PROB_THRESHOLD
from surp_motion_planning.models import OnlineSurpResult
from surp_motion_planning.scene_setup import normalize_scene_for_online_use
from surp_motion_planning.utils import (
    normalize,
    snapshot_frame,
    update_local_perception,
    classify_sensed_obstacles,
    nearest_sensed_obstacle,
    goal_progress_is_stalled,
    step_until_sense_or_contact,
    obstacle_safety_probability,
    robot_clearance_to_obstacle,
    compute_avoid_trajectory,
    compute_push_trajectory,
    reconcile_trajectory_decision,
    remember_bad_direction,
    choose_best_pushable_obstacle,
    move_obstacle_in_direction,
    choose_deep_backtrack_target,
    safe_step_position,
    rrt_escape_step,
    choose_escape_motion,
)
def run_online_surp_push(
    scene: dict,
    epsilon: float = 0.10,
    sensing_range: float = DEFAULT_SENSING_RANGE,
    step_size: float = 0.07,
    push_distance: float = 0.12,
    max_steps: int = 500,
) -> OnlineSurpResult:
    """Run the online goal-seeking / avoid / push controller on one scene.

    The controller is hybrid:
    - goal mode: move straight toward the goal,
    - avoid mode: choose a local detour or escape maneuver,
    - contact mode: push a semantically safe obstacle if that is cheaper.

    The basic goal command is the normalized direct-to-goal vector

        u_goal = v_max * (g - p) / ||g - p||,

    implemented here with fixed step length rather than continuous dynamics.
    """
    working_scene = normalize_scene_for_online_use(scene)
    position = np.array(working_scene["start"][:2], dtype=float)
    goal = np.array(working_scene["goal"][:2], dtype=float)

    path = [tuple(position)]
    frames = [snapshot_frame(position, working_scene, "start")]
    contact_log: list[str] = []
    sensed_ids: list[int] = []
    goal_distance_history = [float(np.linalg.norm(goal - position))]
    bad_directions: list[dict] = []
    push_history: list[dict] = []
    last_boundary_stop_id: int | None = None
    stuck_events = 0

    success = False
    for _ in range(max_steps):
        if np.linalg.norm(goal - position) <= step_size:
            position = goal.copy()
            path.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "goal reached"))
            success = True
            break

        perception = update_local_perception(working_scene, position, sensing_range)
        newly_observed = perception["newly_observed"]
        if newly_observed:
            observed_ids = [obstacle["id"] for obstacle in newly_observed]
            sensed_ids.extend(observed_ids)
            path.append(tuple(position))
            frames.append(
                snapshot_frame(
                    position,
                    working_scene,
                    f"sensed obstacle(s) {observed_ids}; frontiers={len(perception['frontiers'])}",
                )
            )
            continue

        goal_direction = normalize(goal - position)
        if np.linalg.norm(goal_direction) <= 1e-9:
            break

        avoid_set, push_set = classify_sensed_obstacles(
            working_scene,
            position,
            goal_direction,
            epsilon,
        )
        nearest_idx, nearest_distance = nearest_sensed_obstacle(working_scene, position)
        stalled = goal_progress_is_stalled(goal_distance_history)

        if nearest_idx is None or nearest_distance > epsilon:
            last_boundary_stop_id = None
            direct_candidate = step_until_sense_or_contact(
                working_scene,
                position,
                goal_direction,
                step_size,
                epsilon,
            )
            if np.linalg.norm(direct_candidate - position) > 1e-6:
                position = direct_candidate
                stuck_events = 0
                goal_distance_history.append(float(np.linalg.norm(goal - position)))
                path.append(tuple(position))
                frames.append(snapshot_frame(position, working_scene, "goal mode: moving directly toward goal"))
                continue

        if nearest_idx is None:
            goal_distance_history.append(float(np.linalg.norm(goal - position)))
            path.append(tuple(position))
            frames.append(snapshot_frame(position, working_scene, "no sensed obstacle but direct motion unavailable"))
            continue

        nearest_obstacle = working_scene["obstacles"][nearest_idx]

        if last_boundary_stop_id != nearest_obstacle["id"]:
            last_boundary_stop_id = nearest_obstacle["id"]
            path.append(tuple(position))
            frames.append(
                snapshot_frame(
                    position,
                    working_scene,
                    f"stopped at epsilon boundary near obstacle {nearest_obstacle['id']}",
                )
            )
            continue

        p_safe, safety_score = obstacle_safety_probability(nearest_obstacle)
        avoid_candidate = compute_avoid_trajectory(
            working_scene,
            position,
            goal,
            step_size,
            epsilon,
            bad_directions,
        )
        push_candidate = compute_push_trajectory(
            working_scene,
            position,
            goal,
            goal_direction,
            push_set,
            step_size,
            epsilon,
            push_distance,
        )
        selected_candidate = reconcile_trajectory_decision(
            avoid_candidate,
            push_candidate,
            safety_margin_threshold=max(0.02, epsilon * 0.2),
        )
        J_avoid = avoid_candidate.total_cost
        avoid_step_target = avoid_candidate.step_target
        if push_candidate is not None:
            J_push = push_candidate.total_cost
            available_push_distance = push_candidate.push_distance
            corridor_gain = push_candidate.corridor_gain
        else:
            J_push = float("inf")
            available_push_distance = 0.0
            corridor_gain = 0.0

        if stalled:
            stuck_events += 1
            remember_bad_direction(bad_directions, position, goal_direction)

            push_idx = choose_best_pushable_obstacle(
                working_scene,
                position,
                goal_direction,
                epsilon,
            )
            if push_idx is not None and nearest_idx is not None and push_idx != nearest_idx:
                nearest_obstacle = working_scene["obstacles"][nearest_idx]
                nearest_distance = robot_clearance_to_obstacle(position, nearest_obstacle)
                if nearest_obstacle["true_class"] not in {"safe", "movable"} or nearest_distance <= epsilon + 0.03:
                    push_idx = None
            if push_idx is not None:
                pushed_obstacle = working_scene["obstacles"][push_idx]
                moved_distance, moved_chain = move_obstacle_in_direction(
                    working_scene,
                    push_idx,
                    goal_direction,
                    push_distance,
                    robot_position=position,
                )
                if moved_distance > 1e-6:
                    position = step_until_sense_or_contact(
                        working_scene,
                        position,
                        goal_direction,
                        step_size,
                        epsilon,
                    )
                    stuck_events = 0
                    goal_distance_history.append(float(np.linalg.norm(goal - position)))
                    chain_text = f" with chain {moved_chain}" if len(moved_chain) > 1 else ""
                    message = (
                        f"contact mode (stuck override): pushed obstacle "
                        f"{pushed_obstacle['id']} by {moved_distance:.2f}{chain_text}"
                    )
                    pushed_obstacle["push_count"] = pushed_obstacle.get("push_count", 0) + 1
                    push_history.append({
                        "obstacle_id": pushed_obstacle["id"],
                        "distance": moved_distance,
                        "chain": moved_chain,
                    })
                    path.append(tuple(position))
                    frames.append(snapshot_frame(position, working_scene, message))
                    contact_log.append(message)
                    continue

            backtrack_target = choose_deep_backtrack_target(path, position, stuck_events)
            if backtrack_target is not None:
                retreat_direction = normalize(backtrack_target - position)
                retreat_step_size = min(
                    max(step_size * (1.8 + 0.35 * min(stuck_events, 5)), step_size),
                    float(np.linalg.norm(backtrack_target - position)),
                )
                retreat_candidate = safe_step_position(
                    working_scene,
                    position,
                    retreat_direction,
                    retreat_step_size,
                    observed_only=True,
                )
                if np.linalg.norm(retreat_candidate - position) > 1e-6:
                    position = retreat_candidate
                    goal_distance_history.append(float(np.linalg.norm(goal - position)))
                    path.append(tuple(position))
                    frames.append(
                        snapshot_frame(
                            position,
                            working_scene,
                            f"avoid mode (memory replan): deeper backtrack level {stuck_events}",
                        )
                    )
                    continue

            if stuck_events >= 2:
                _, rrt_candidate = rrt_escape_step(
                    working_scene,
                    position,
                    goal,
                    step_size,
                    bad_directions=bad_directions,
                    max_samples=120 + 40 * min(stuck_events, 4),
                )
                if rrt_candidate is not None and np.linalg.norm(rrt_candidate - position) > 1e-6:
                    position = rrt_candidate
                    goal_distance_history.append(float(np.linalg.norm(goal - position)))
                    path.append(tuple(position))
                    frames.append(
                        snapshot_frame(
                            position,
                            working_scene,
                            "avoid mode (RRT escape): retreating and replanning through sensed free space",
                        )
                    )
                    continue

            _, escape_candidate = choose_escape_motion(
                working_scene,
                position,
                goal,
                step_size,
                epsilon,
                bad_directions=bad_directions,
            )
            if np.linalg.norm(escape_candidate - position) > 1e-6:
                position = escape_candidate
                goal_distance_history.append(float(np.linalg.norm(goal - position)))
                path.append(tuple(position))
                frames.append(
                    snapshot_frame(
                        position,
                        working_scene,
                        "avoid mode (memory replan): local escape from non-pushable trap",
                    )
                )
                continue
        else:
            stuck_events = 0

        push_allowed = (
            selected_candidate.mode == "push"
            and p_safe >= SAFE_PROB_THRESHOLD
            and available_push_distance > 1e-6
            and push_candidate is not None
        )
        if push_allowed and push_candidate is not None and push_candidate.obstacle_index is not None:
            moved_distance, moved_chain = move_obstacle_in_direction(
                working_scene,
                push_candidate.obstacle_index,
                goal_direction,
                push_candidate.push_distance,
                robot_position=position,
            )
            if moved_distance > 1e-6:
                position = step_until_sense_or_contact(
                    working_scene,
                    position,
                    goal_direction,
                    step_size,
                    epsilon,
                )
                stuck_events = 0
                goal_distance_history.append(float(np.linalg.norm(goal - position)))
                chain_text = f" with chain {moved_chain}" if len(moved_chain) > 1 else ""
                pushed_obstacle = working_scene["obstacles"][push_candidate.obstacle_index]
                pushed_obstacle["push_count"] = pushed_obstacle.get("push_count", 0) + 1
                push_history.append({
                    "obstacle_id": pushed_obstacle["id"],
                    "distance": moved_distance,
                    "chain": moved_chain,
                })
                message = (
                    f"contact mode: pushed obstacle {pushed_obstacle['id']} by "
                    f"{moved_distance:.2f}{chain_text}; push_history={len(push_history)}"
                )
                path.append(tuple(position))
                frames.append(snapshot_frame(position, working_scene, message))
                contact_log.append(message)
                continue

        if np.linalg.norm(avoid_step_target - position) > 1e-6:
            position = avoid_step_target
            stuck_events = 0
            goal_distance_history.append(float(np.linalg.norm(goal - position)))
            path.append(tuple(position))
            frames.append(
                snapshot_frame(
                    position,
                    working_scene,
                    (
                        f"avoid mode: obstacle {nearest_obstacle['id']} "
                        f"(P_safe={p_safe:.2f}, J_push={J_push:.2f}, J_avoid={J_avoid:.2f}, "
                        f"frontiers={len(perception['frontiers'])})"
                    ),
                )
            )
            continue

        goal_distance_history.append(float(np.linalg.norm(goal - position)))
        path.append(tuple(position))
        frames.append(
            snapshot_frame(
                position,
                working_scene,
                (
                    f"boundary decision stalled: obstacle {nearest_obstacle['id']} "
                    f"(P_safe={p_safe:.2f}, score={safety_score:.2f}, "
                    f"J_push={J_push:.2f}, J_avoid={J_avoid:.2f}, gain={corridor_gain:.2f})"
                ),
            )
        )

    return OnlineSurpResult(
        family=working_scene["family"],
        seed=working_scene["seed"],
        success=success,
        path=path,
        frames=frames,
        scene=working_scene,
        initial_scene=copy.deepcopy(scene),
        contact_log=contact_log,
        sensed_ids=sensed_ids,
    )
