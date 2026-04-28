import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from beacon.core.bug_algorithm import run_bug
from beacon.core.scene_setup import generate_one_random_environment


def main() -> None:
    """Small local debug runner for the Bug baseline in the cleaned repo."""
    scene = generate_one_random_environment()
    result = run_bug(scene, max_steps=1000)

    print(f"Family: {result.family}")
    print(f"Seed: {result.seed}")
    print(f"Success: {result.success}")
    print(f"Steps: {len(result.path)}")
    print(f"Sensed ids: {result.sensed_ids}")
    print(f"Contacts logged: {len(result.contact_log)}")


if __name__ == "__main__":
    main()
