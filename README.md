## SCHOLAR — Semantic, Contact-aware, Hybrid Online planner for Local obstacle Avoidance and Rearrangement

SCHOLAR is a real-time, sensor-based motion planner for a disk robot navigating an unknown environment containing obstacles of mixed manipulability.  It unifies three complementary strategies — direct goal pursuit, Bug1-style boundary following, and physics-based pushing — into a single per-step decision loop.

---

### How it works

**Sensing**

Each step the robot reveals all obstacles within `sensing_range`.  Newly observed obstacles are classified by `true_class` (`movable` / `not_movable`) and a safety probability `P_safe` derived from their fragility score.

**Per-step decision**

```
goal reached?          → done
new obstacles sensed?  → record and re-plan
no obstacle within ε?  → GOAL MODE  (move directly toward goal)
obstacle within ε?     →
    movable & P_safe ≥ threshold & J_push < J_avoid?
        → PUSH MODE    (translate obstacle, follow through)
    otherwise
        → BOUNDARY MODE (Bug1 left-hand rule, see below)
progress stalled or boundary trapped?
        → STALL RECOVERY cascade
```

**Boundary mode** (Bug1 left-hand rule)

When an obstacle is unmovable, too fragile to push, or when avoidance is cheaper than pushing, the robot follows the obstacle surface:

1. On first contact, turn 90° CCW (left) and begin hugging the surface.
2. At each step, sweep 36 candidate directions (10° increments, starting from −90° toward the wall) and take the first collision-free step.
3. **Exit**: as soon as a direct step toward the goal is collision-free, take that step immediately and return to goal mode.
4. **Circuit detection**: if the robot returns near its entry point after ≥ 4 boundary steps without finding an exit, it has completed a full orbit → escalate to stall recovery.
5. **Bounce detection**: if the robot exits boundary mode for an obstacle then immediately re-enters it 3+ times (indicating the exit leads into a dead-end corridor), override the cost model and push the obstacle if it is pushable; otherwise escalate to stall recovery.

**Push mode**

Pushes are evaluated by a cost function:

```
J_push = 0.6·t + 0.5·e + 1.2·r − 1.4·Δc
```

where `t` = travel time, `e` = push effort, `r` = fragility risk, `Δc` = corridor gain from widening the gap.  Chain reactions are propagated with `CHAIN_ATTENUATION = 0.7` per link.

**Stall recovery cascade**

Triggered when boundary mode is fully trapped or goal progress has stalled:

1. **Push override** — push the best reachable movable obstacle toward the goal.
2. **RRT escape** — build a local rapidly-exploring random tree through sensed free space.
3. **Deep backtrack** — retreat along the executed path by a depth proportional to `stuck_events`.
4. **Local escape** — random-walk style perturbation away from blocked directions.

---

### Baselines

`scholar/planning/baselines.py` registers three baselines for comparative evaluation:

| Name | File | Description |
|------|------|-------------|
| `bug` | `NewProject/bug_algorithm.py` | Bug1: goal mode + left-hand boundary follow, fine 36-direction sweep |
| `rrt` | `NewProject/rrt_baseline.py` | Global RRT over the fully revealed map |
| `surp` | `NewProject/planner.py` | SCHOLAR (online SURP push) |

Metrics are computed via `scholar/utils/metrics.py` (`EpisodeMetrics`: success, steps, path length, contacts, sensed obstacles).

---

### Scene families

Scenes live in `scholar/environment/data/scenes/{family}/scene_{N:03d}.json`.

| Family | Description |
|--------|-------------|
| `sparse` | Few well-separated obstacles; direct path usually available |
| `cluttered` | Dense mixed obstacles; frequent boundary following and pushing |
| `collision_required` | At least one not_movable obstacle fully blocks the direct path |
| `collision_shortcut` | A movable obstacle blocks the direct path; pushing it creates a shortcut |

---

### Running simulations

**SCHOLAR (all families in parallel, live animation):**
```bash
python scholar/demo.py --scene 15
python scholar/demo.py --scene 15 --steps 400 --save
```

**Bug1 baseline:**
```bash
python scholar/test_bug1.py --scene 15
python scholar/test_bug1.py --scene 15 --family cluttered collision_required
```

**Algorithm illustration:**
```bash
python scholar/utils/algorithm_illustration.py
```
Output saved to `scholar/environment/data/images/algorithm_illustration.png`.

---

### Module map

```
NewProject/
  planner.py          — SCHOLAR main loop: sensing, boundary mode, push, stall recovery
  bug_algorithm.py    — Bug1 baseline planner
  scene_setup.py      — scene loading and normalisation for online use
  models.py           — SimulationFrame, OnlineSurpResult dataclasses
  constants.py        — ROBOT_RADIUS, SAFE_PROB_THRESHOLD, CHAIN_ATTENUATION, …

scholar/
  demo.py             — live multi-family SCHOLAR simulation
  test_bug1.py        — live multi-family Bug1 simulation
  planning/
    baselines.py      — PLANNERS registry
  experiments/
    run_trials.py     — batch evaluation across all planners and scenes
  utils/
    metrics.py        — EpisodeMetrics, compute_metrics
    algorithm_illustration.py — publication-quality 4-panel figure
  environment/
    scene_generator.py
    visualize_v2.py
    data/scenes/      — JSON scene files
    data/logs/        — simulation logs
    data/videos/      — saved mp4/gif outputs
    data/images/      — saved figures
```
