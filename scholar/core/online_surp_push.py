import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scholar.core.constants import OUTPUT_DIR
from scholar.core.planner import run_online_surp_push
from scholar.core.scene_setup import generate_one_random_environment
from scholar.core.visualization import animate_result, plot_final_snapshot, save_scene_snapshot


def main() -> None:
    """Run one random environment, save outputs, and open the live simulation."""
    random_scene = generate_one_random_environment()
    result = run_online_surp_push(random_scene, epsilon=0.10, step_size=0.07, max_steps=320)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scene_path = save_scene_snapshot(result.scene, result.family, result.seed)
    image_path = plot_final_snapshot(result)
    animate_result(result)

    print(f"Family: {result.family}")
    print(f"Seed: {result.seed}")
    print(f"Success: {result.success}")
    print(f"Sensed obstacle ids: {result.sensed_ids}")
    print(f"Contacts: {len(result.contact_log)}")
    print(f"Saved scene JSON to: {scene_path}")
    print(f"Saved final image to: {image_path}")


if __name__ == "__main__":
    main()
