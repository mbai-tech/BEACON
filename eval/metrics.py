import numpy as np
import time

CLASSES = ["safe", "movable", "fragile", "forbidden"]
EVAL_COSTS = {"safe": 1, "movable": 3, "fragile": 15, "forbidden": 1000}

def success_rate(results):
    return sum(r["success"] for r in results) / len(results)

def path_length(path):
    if not path or len(path) < 2:
        return 0.0
    return sum(
        np.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        for i in range(len(path) - 1)
    )

def rotation_cost(path):
    if not path or len(path) < 2:
        return 0.0
    return sum(abs(path[i+1][2] - path[i][2]) for i in range(len(path) - 1))

def avg_path_length(results):
    return np.mean([path_length(r["path"]) for r in results if r["path"]])

def avg_rotation_cost(results):
    return np.mean([rotation_cost(r["path"]) for r in results if r["path"]])

def measure_planning_time(plan_fn, scene, n_runs=3):
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        plan_fn(scene)
        times.append(time.time() - t0)
    return {"mean": float(np.mean(times)), "p95": float(np.percentile(times, 95))}

def contact_count_by_class(contact_log):
    counts = {c: 0 for c in CLASSES}
    for cls, area in contact_log:
        if area > 0:
            counts[cls] += 1
    return counts

def contact_severity(contact_log):
    return sum(area for _, area in contact_log)

def semantic_damage(contact_log):
    return sum(EVAL_COSTS[cls] * area for cls, area in contact_log)

def forbidden_rate(results):
    return sum(
        1 for r in results
        if any(cls == "forbidden" and area > 0
               for cls, area in r.get("contact_log", []))
    ) / len(results)

def fragile_rate(results):
    return sum(
        1 for r in results
        if any(cls == "fragile" and area > 0
               for cls, area in r.get("contact_log", []))
    ) / len(results)

def kl_divergence(p, q):
    p = np.array(p, dtype=float) + 1e-9
    q = np.array(q, dtype=float) + 1e-9
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))

def belief_error(final_posteriors, true_classes):
    total = 0.0
    for post, cls in zip(final_posteriors, true_classes):
        idx = CLASSES.index(cls)
        one_hot = np.full(4, 1e-9)
        one_hot[idx] = 1.0
        total += kl_divergence(one_hot, post)
    return total / len(true_classes)

def trap_recovery_rate(trap_results):
    recovered = sum(1 for r in trap_results if r.get("replans", 0) > 0)
    return recovered / len(trap_results)

def delta_noise(low_noise_results, high_noise_results):
    return success_rate(low_noise_results) - success_rate(high_noise_results)

def replan_count(results):
    return np.mean([r.get("replans", 0) for r in results])

def print_summary(results, label, true_classes_per_episode=None):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Success rate:      {success_rate(results):.1%}")
    print(f"  Avg path length:   {avg_path_length(results):.2f}")
    print(f"  Avg rotation cost: {avg_rotation_cost(results):.2f}")
    print(f"  Avg damage:        {np.mean([r.get('damage',0) for r in results]):.4f}")
    print(f"  Avg replans:       {replan_count(results):.2f}")
    print(f"  Forbidden rate:    {forbidden_rate(results):.1%}")
    print(f"  Fragile rate:      {fragile_rate(results):.1%}")
    if true_classes_per_episode:
        be = np.mean([
            belief_error(r["posteriors"], tc)
            for r, tc in zip(results, true_classes_per_episode)
            if "posteriors" in r and r["posteriors"]
        ])
        print(f"  Belief error:      {be:.4f}")