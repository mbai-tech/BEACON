# SURP / SCHOLAR

Online motion planning experiments for a disk robot in partially known environments.

## Quick Start

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy matplotlib shapely pybullet
```

Main commands:

```bash
# Run the main SCHOLAR demo on one scene
python3 scholar/demo_scholar.py --scene 0

# Run SCHOLAR on scenes 0-99 and save outputs
python3 scholar/demo_scholar.py --scenes 0-99 --save --clear-past

# Run Bug baselines
python3 scholar/demo_bug1.py --scene 0
python3 scholar/demo_bug2.py --scene 0

# Run the shared scene-complex benchmark
python3 scholar/experiments/run_scene_complex_metrics.py --scenes 0-99 --planners scholar bug bug2 dstar_lite

# Summarize and compare saved metrics
python3 scholar/experiments/summarize_scene_complex_metrics.py
python3 scholar/experiments/compare_scene_complex_metrics.py
```

## Repo Map

```text
SURP/
  scholar/                Main demos, planner code, and experiment scripts
    demo_scholar.py       Best entry point for SCHOLAR runs
    demo_bug1.py          Bug1 demo entry point
    demo_bug2.py          Bug2 demo entry point
    planning/             SCHOLAR planner and planner-side utilities
    experiments/          Benchmark runners, summaries, and paper figures
    environment/data/     Saved logs, videos, and metrics

  NewProject/             Core planner implementations and baseline algorithms
    planner.py            Main online planner logic
    bug_algorithm.py      Bug baseline
    bug2_algorithm.py     Bug2 baseline
    dstar_lite_algorithm.py
    scene_setup.py        Scene normalization and setup helpers

  enviornment/            Top-level scene generators used by SCHOLAR
    scene_complex.py      Main scene family generator
    scene_generator.py    Alternate generator utilities
```

## What To Run

Use `scholar/demo_scholar.py` for most interactive or saved SCHOLAR runs.
It generates scenes from `enviornment/scene_complex.py` and supports:

- `sparse`
- `cluttered`
- `collision_required`
- `collision_shortcut`

Useful examples:

```bash
python3 scholar/demo_scholar.py --scene 0 --family sparse
python3 scholar/demo_scholar.py --scenes 0-9 --save
python3 scholar/demo_scholar.py --scenes 0-99 --save --clear-past --steps 700
```

Saved outputs go under:

- `scholar/environment/data/logs/`
- `scholar/environment/data/videos/`

## Benchmarking

Run one or more planners on the shared `scene_complex` benchmark:

```bash
python3 scholar/experiments/run_scene_complex_metrics.py \
  --scenes 0-99 \
  --planners scholar bug bug2 dstar_lite
```

This writes CSV metrics under:

- `scholar/environment/data/metrics/`

Then summarize:

```bash
python3 scholar/experiments/summarize_scene_complex_metrics.py
```

Or compare multiple saved planner CSVs:

```bash
python3 scholar/experiments/compare_scene_complex_metrics.py
```

Paper-friendly plots and tables:

```bash
python3 scholar/experiments/generate_paper_figures.py
```

## Environment Generation

Generate top-level environment assets with:

```bash
python3 enviornment/run.py
python3 enviornment/run_family.py sparse_clutter
python3 enviornment/run_family.py sparse_clutter 12345
```

See [enviornment/README.md](/Users/ishita/Documents/GitHub/SURP/enviornment/README.md) for environment-generator details.

## Notes

- The historical folder name `enviornment/` is intentionally preserved because the existing code imports it directly.
- `scholar/main_*.py` still hold the full implementations; the `scholar/demo_*.py` wrappers are the cleaner user-facing entry points.
- The saved data folders contain useful outputs, but they are not the best place to start reading the code.
