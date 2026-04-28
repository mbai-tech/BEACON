# BEACON

Online motion planning for a disk robot in partially known environments.
BEACON navigates by sensing nearby obstacles, learning their semantic class through contact (Bayesian belief update), and choosing between pushing, avoiding, or boundary-following based on a battery-aware cost function.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib shapely pybullet
```

## Running

### Live BEACON demo (animated)

```bash
python3 beacon/main_beacon.py --scene 0
python3 beacon/main_beacon.py --scene 0 --family cluttered
python3 beacon/main_beacon.py --scenes 0-9 --save
python3 beacon/main_beacon.py --scenes 0-99 --save --clear-past --steps 700
```

Flags: `--scene N` (seed), `--family sparse|cluttered|collision_required|collision_shortcut`, `--steps N`, `--sense R`, `--step S`, `--save`, `--clear-past`

Saved outputs go to `beacon/environment/data/logs/` and `beacon/environment/data/videos/`.

### Compare all baselines on one scene

```bash
python3 beacon/experiments/run_one_scene.py --scene 0 --family cluttered
python3 beacon/experiments/run_one_scene.py --scene 0 --family cluttered --visuals out/visuals
```

Runs Bug1, Bug2, D\* Lite, and BEACON on the same scene and prints a metrics table.
With `--visuals DIR`, saves a PNG per planner showing the path and any pushed obstacles.

### Benchmark across many scenes

```bash
python3 beacon/utils/run_scene_complex_metrics.py --scene 0 --planners beacon bug bug2 dstar_lite
python3 beacon/utils/run_scene_complex_metrics.py --scenes 0-99 --planners beacon bug bug2 dstar_lite
python3 beacon/utils/run_scene_complex_metrics.py --scenes 0-99 --planners beacon bug bug2 dstar_lite --visuals out/visuals
```

Writes CSV to `beacon/environment/data/metrics/metrics_scene_complex.csv`.

### Summarize and plot results

```bash
python3 beacon/utils/summarize_scene_complex_metrics.py
python3 beacon/utils/compare_scene_complex_metrics.py
python3 beacon/utils/generate_paper_figures.py
python3 beacon/utils/plots.py --csv path/to/metrics.csv
```

## Repo Map

```
SURP/
  beacon/
    main_beacon.py              Live BEACON demo with animation and video saving
    main.py                     General CLI (planner selection, batch trials)
    main_bug1.py                Live Bug1 animated demo
    main_bug2.py                Live Bug2 animated demo

    core/                       Planner implementations
      planner.py                BEACON (run_online_surp_push)
      bug_algorithm.py          Bug1
      bug2_algorithm.py         Bug2
      dstar_lite_algorithm.py   D* Lite
      rrt_greedy.py             Greedy RRT baseline
      cibp.py                   Bayesian belief updater
      models.py                 Shared dataclasses
      constants.py              Shared constants (battery, sensing range, etc.)
      visualization.py          Plot and animation helpers

    planning/
      baselines.py              Planner registry (PLANNERS dict)
      cost_map.py               Anisotropic cost map
      semantic_cost.py          Semantic class cost tables
      vlm_updater.py            VLM-based belief updater

    experiments/
      run_one_scene.py          Run all baselines on one scene, print table + PNGs

    utils/
      run_scene_complex_metrics.py  Batch benchmark, writes CSV
      summarize_scene_complex_metrics.py
      compare_scene_complex_metrics.py
      generate_paper_figures.py
      plots.py                  Paper figures (trade-off scatter, algorithm illustration)

    environment/
      data/logs/                Simulation step logs
      data/videos/              Saved MP4/GIF outputs
      data/metrics/             Benchmark CSVs
      data/scenes/              Pre-saved scene JSONs (100 per family)

  beacon/environment/
    scene_complex.py            Primary generator — sparse, cluttered, collision_required, collision_shortcut
    scene_generator_shapely.py  Shapely-based polygon generator
    scene_generator_pybullet.py PyBullet-based scene generator
    validator.py                Scene validity checks
    run_family.py               Generate and save scenes for one family

```

## Scenes

Scenes are procedurally generated from a seed — `--scene N` is both the index and the random seed, so any integer gives a reproducible scene. There is no fixed dataset limit.

Four families:
- `sparse` — few obstacles, open space
- `cluttered` — many obstacles, navigation requires interaction
- `collision_required` — a push is necessary to reach the goal
- `collision_shortcut` — a push creates a shorter path

## Notes

- `main_beacon.py` runs only BEACON. Use `run_one_scene.py` to compare against baselines.
- Benchmark CSVs record: `planner, family, scene_idx, seed, success, steps, path_length, n_contacts, n_sensed, png_path`.
