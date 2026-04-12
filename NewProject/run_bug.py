import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NewProject.bug_algorithm import run_bug
from NewProject.constants import OUTPUT_DIR
from NewProject.scene_setup import generate_one_random_environment
from NewProject.visualization import animate_result, plot_final_snapshot, save_scene_snapshot


def main() -> None:
    """Run the Bug-style baseline in one random environment."""
    scene = generate_one_random_environment()
    result = run_bug(scene)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scene_path = save_scene_snapshot(result.scene, result.family, result.seed)
    image_path = plot_final_snapshot(result)
    animate_result(result)

    print(f"Family:  {result.family}")
    print(f"Seed:    {result.seed}")
    print(f"Success: {result.success}")
    print(f"Steps:   {len(result.path)}")
    print(f"Sensed:  {result.sensed_ids}")
    print(f"Saved scene to: {scene_path}")
    print(f"Saved image to: {image_path}")


if __name__ == "__main__":
    main()
