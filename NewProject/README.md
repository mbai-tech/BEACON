## NewProject

This folder contains the current robotics motion-planning sandbox for the new project.

The main demo:

- uses the repo's existing environment generator in `enviornment/scene_generator.py`
- converts the generated obstacles into circles for this project
- runs a point robot with local sensing only
- stops at an `epsilon` distance before obstacle contact
- chooses between goal motion, local avoidance, and pushing safe obstacles
- propagates pushes through obstacle chains with reduced motion for downstream contacts
- falls back to deeper backtracking and a local RRT-style escape when it gets stuck

The code is now split into:

- `online_surp_push.py`: thin entrypoint
- `scene_setup.py`: environment generation and preprocessing
- `planner.py`: sensing, motion, risk-aware pushing, and stuck recovery
- `visualization.py`: saved plots and live animation
- `models.py` and `constants.py`: shared data structures and configuration

Run the demo from the repository root:

```bash
./NewProject/run_demo.sh
```

This opens a live animation window and also saves outputs to:

```text
NewProject/outputs/
```
