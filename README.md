## SURP / SCHOLAR

SCHOLAR is an online navigation and manipulation planner for a disk robot in partially known environments. The current repo combines:

- a SCHOLAR demo and batch runner under `scholar/`
- the main online planner under `NewProject/`
- a newer top-level environment generator under `enviornment/`

This README is the best place to start if you want to run the code on your own machine.

## Repo layout

```text
SURP/
  NewProject/
    planner.py                # Current SCHOLAR-style online planner
    bug_algorithm.py          # Bug-style baseline
    bug2_algorithm.py         # Bug2-style baseline
    rrt_greedy.py             # RRT baseline
    scene_setup.py            # Random scene generation + normalization helpers
    online_surp_push.py       # Run the planner once on a random scene
    run_bug.py                # Run Bug baseline once on a random scene
    run_rrt.py                # Run RRT baseline once on a random scene
    visualization.py          # Animation and snapshot utilities

  scholar/
    demo_scholar.py           # Main SCHOLAR demo and batch runner
    demo_bug1.py              # Bug1 demo and batch runner
    demo_bug2.py              # Bug2 demo and batch runner
    experiments/run_trials.py # Programmatic experiment runner
    environment/              # Legacy SCHOLAR data/log/video/image directories

  enviornment/
    scene_generator.py        # New scene generator used by demo_scholar.py
    run.py                    # Generate 100 scenes for each new environment family
    run_family.py             # Generate one environment family
    README.md                 # Extra details for environment generation
```

## What the main demo uses

The current `scholar/demo_scholar.py` now generates scenes through the top-level `enviornment/scene_generator.py`.

The SCHOLAR demo keeps these four user-facing family names:

- `sparse`
- `cluttered`
- `collision_required`
- `collision_shortcut`

Internally they map to the new environment generator families like this:

- `sparse` -> `sparse_clutter`
- `cluttered` -> `dense_clutter`
- `collision_required` -> `narrow_passage`
- `collision_shortcut` -> `semantic_trap`

So when you run `scholar/demo_scholar.py`, the scene source is the new `enviornment` generator, not the old `scholar/environment/data/scenes/...` JSON loader.

## Requirements

You should use Python 3.11 if possible. The code has been run with:

- `python3`
- `/opt/anaconda3/bin/python3`

Core Python packages used in the repo include:

- `numpy`
- `matplotlib`
- `shapely`
- `pybullet`

Depending on your machine and workflow, you may also want:

- `pip`
- `venv`

## Recommended setup

From the repo root:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy matplotlib shapely pybullet
```

If you are using Anaconda and want to use the same interpreter as recent runs in this repo:

```bash
cd /Users/ishita/Documents/GitHub/SURP
/opt/anaconda3/bin/python3 -m pip install numpy matplotlib shapely pybullet
```

For the new environment generator specifically, there is also a local requirements file:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 -m pip install -r enviornment/requirements.txt
```

## Quick start

Run SCHOLAR on one scene interactively:

```bash
cd /Users/ishita/Documents/GitHub/SURP
/opt/anaconda3/bin/python3 scholar/demo_scholar.py --scene 0
```

Run SCHOLAR on scenes `0-99`, save outputs, and clear old saved simulations first:

```bash
cd /Users/ishita/Documents/GitHub/SURP
/opt/anaconda3/bin/python3 scholar/demo_scholar.py --scenes 0-99 --save --clear-past
```

That is the main batch command currently used in this repo.

## Main SCHOLAR command

File:

- [scholar/demo_scholar.py](/Users/ishita/Documents/GitHub/SURP/scholar/demo_scholar.py:432)

CLI arguments:

- `--scene 0`
  Run one or more specific scene indices. Example: `--scene 0 1 2`
- `--scenes 0-99`
  Run an inclusive range of scene indices. This implies batch mode.
- `--family sparse cluttered`
  Restrict which SCHOLAR family names to run.
- `--steps 500`
  Maximum simulation steps. Default is now `500`.
- `--sense 0.35`
  Sensing radius in meters.
- `--step 0.04`
  Robot step size in meters.
- `--save`
  Save logs and video or gif outputs.
- `--clear-past`
  Delete previously saved simulation logs and videos before running.
- `--speedup 3`
  Playback speed multiplier for saved output.
- `--workers 8`
  Maximum parallel workers in batch mode.

Useful examples:

```bash
cd /Users/ishita/Documents/GitHub/SURP
/opt/anaconda3/bin/python3 scholar/demo_scholar.py --scene 0
```

```bash
cd /Users/ishita/Documents/GitHub/SURP
/opt/anaconda3/bin/python3 scholar/demo_scholar.py --scene 0 --family sparse
```

```bash
cd /Users/ishita/Documents/GitHub/SURP
/opt/anaconda3/bin/python3 scholar/demo_scholar.py --scenes 0-9 --save
```

```bash
cd /Users/ishita/Documents/GitHub/SURP
/opt/anaconda3/bin/python3 scholar/demo_scholar.py --scenes 0-99 --save --clear-past --steps 700
```

## SCHOLAR outputs

Saved outputs go under:

- `scholar/environment/data/logs/`
- `scholar/environment/data/videos/`

Typical output types:

- text logs such as `simulation_scene000_*.txt`
- gifs such as `simulation_scene000.gif`
- sometimes mp4 files depending on the run path

The `--clear-past` flag only deletes saved simulation logs and videos. It does not delete source scene assets.

## Baseline demos

### Bug1

File:

- [scholar/demo_bug1.py](/Users/ishita/Documents/GitHub/SURP/scholar/demo_bug1.py:384)

Example commands:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 scholar/demo_bug1.py --scene 0
```

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 scholar/demo_bug1.py --scenes 0-19 --save
```

Important defaults:

- `--steps 500`
- `--sense 0.55`
- `--step 0.07`

### Bug2

File:

- [scholar/demo_bug2.py](/Users/ishita/Documents/GitHub/SURP/scholar/demo_bug2.py:386)

Example commands:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 scholar/demo_bug2.py --scene 0
```

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 scholar/demo_bug2.py --scenes 0-19 --save
```

Important defaults:

- `--steps 500`
- `--sense 0.55`
- `--step 0.07`

## One-off local runs from `NewProject/`

These are useful if you want to debug one randomly generated scene without using the SCHOLAR batch UI.

### SCHOLAR planner once

File:

- [NewProject/online_surp_push.py](/Users/ishita/Documents/GitHub/SURP/NewProject/online_surp_push.py:1)

Command:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 NewProject/online_surp_push.py
```

This:

- generates one random environment
- runs `run_online_surp_push(...)`
- saves a final snapshot
- opens the animation path

### Bug baseline once

File:

- [NewProject/run_bug.py](/Users/ishita/Documents/GitHub/SURP/NewProject/run_bug.py:1)

Command:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 NewProject/run_bug.py
```

### RRT baseline once

File:

- [NewProject/run_rrt.py](/Users/ishita/Documents/GitHub/SURP/NewProject/run_rrt.py:1)

Command:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 NewProject/run_rrt.py
```

## Programmatic experiments

File:

- [scholar/experiments/run_trials.py](/Users/ishita/Documents/GitHub/SURP/scholar/experiments/run_trials.py:1)

Example:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 -c "from scholar.experiments.run_trials import run_trials; run_trials(n_trials=3, planner_name='scholar')"
```

Available planner names are defined from:

- `planning.baselines.PLANNERS`
- plus `"scholar"`

So valid names generally include:

- `scholar`
- baseline names registered in `scholar/planning/baselines.py`

## New environment generator

The new environment generator lives in:

- [enviornment/scene_generator.py](/Users/ishita/Documents/GitHub/SURP/enviornment/scene_generator.py:1)

Its native family names are:

- `sparse_clutter`
- `dense_clutter`
- `narrow_passage`
- `semantic_trap`
- `perturbed`

### Generate all new environment families

File:

- [enviornment/run.py](/Users/ishita/Documents/GitHub/SURP/enviornment/run.py:1)

Command:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 enviornment/run.py
```

This generates 100 scenes for each family and writes them to:

- `enviornment/data/images/`
- `enviornment/data/scenes/`

Important:

- `run.py` clears those output folders first
- the generated files are named with their seed

### Generate one family

File:

- [enviornment/run_family.py](/Users/ishita/Documents/GitHub/SURP/enviornment/run_family.py:1)

Command without fixed seed:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 enviornment/run_family.py sparse_clutter
```

Command with fixed seed:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 enviornment/run_family.py sparse_clutter 12345
```

Behavior:

- without a seed, output goes to `enviornment/data/...` and old files there are cleared
- with a seed, output goes to `enviornment/saved_enviornments/...` and is preserved

## Current planner behavior summary

The main online planner is:

- [NewProject/planner.py](/Users/ishita/Documents/GitHub/SURP/NewProject/planner.py:2123)

Key characteristics of the current version:

- D* Lite-inspired local path repair on the observed map
- pushing only when strong blockage is detected ahead
- Bug-style boundary following as a fallback
- D* Lite-style backtracking during deeper recovery
- near-goal direct-step fallback so the robot does not stop just short of the goal

The current default max step limit in the planner is:

- [NewProject/planner.py](/Users/ishita/Documents/GitHub/SURP/NewProject/planner.py:2129) -> `max_steps: int = 500`

The current default SCHOLAR CLI step limit is:

- [scholar/demo_scholar.py](/Users/ishita/Documents/GitHub/SURP/scholar/demo_scholar.py:438) -> `--steps` default `500`

## Common workflows

### 1. Just run SCHOLAR on all main scenes

```bash
cd /Users/ishita/Documents/GitHub/SURP
/opt/anaconda3/bin/python3 scholar/demo_scholar.py --scenes 0-99 --save --clear-past
```

### 2. Run one family only

```bash
cd /Users/ishita/Documents/GitHub/SURP
/opt/anaconda3/bin/python3 scholar/demo_scholar.py --scenes 0-19 --family cluttered --save
```

### 3. Give the robot more time

```bash
cd /Users/ishita/Documents/GitHub/SURP
/opt/anaconda3/bin/python3 scholar/demo_scholar.py --scenes 0-99 --save --clear-past --steps 700
```

### 4. Rebuild environment-generator outputs

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 enviornment/run.py
```

### 5. Reproduce one generated environment by seed

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 enviornment/run_family.py dense_clutter 12345
```

## Notes for macOS

Batch saving on macOS can fail if Matplotlib tries to open GUI windows from worker threads. `scholar/demo_scholar.py` now switches to the `Agg` backend automatically for batch `--save` runs, which avoids the common `NSWindow should only be instantiated on the main thread` crash.

If you still see Matplotlib cache warnings, you can set:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-cache
mkdir -p "$MPLCONFIGDIR"
```

before running the demos.

## Troubleshooting

### `ModuleNotFoundError: No module named 'enviornment.scene_generator'`

Run from the repo root:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 scholar/demo_scholar.py --scene 0
```

The repo root needs to be on the Python path for the top-level `enviornment/` package to resolve.

### `ValueError: not enough values to unpack` with `--scenes`

`--scenes` currently expects a range like:

```bash
--scenes 0-99
```

not:

```bash
--scenes 0
```

If you want one scene, use:

```bash
--scene 0
```

or:

```bash
--scenes 0-0
```

### The robot seems to stop too early

Check the step limit first:

```bash
python3 scholar/demo_scholar.py --scene 0 --steps 700
```

The current defaults are already `500`, but long cluttered scenes may still need more.

## Related READMEs

For generator-specific details, also see:

- [enviornment/README.md](/Users/ishita/Documents/GitHub/SURP/enviornment/README.md)
