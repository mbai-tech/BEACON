from pathlib import Path


DISPLAY_COLORS = {
    "safe": "#7bc96f",
    "movable": "#f4a261",
    "fragile": "#e76f51",
    "forbidden": "#6c757d",
}

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CLUTTER_BIASED_FAMILIES = ["sparse_clutter"]
TARGET_MAX_WORKSPACE_SPAN = 6.0
DEFAULT_SENSING_RANGE = 0.55
SAFE_PROB_THRESHOLD = 0.45
ROBOT_RADIUS = 0.14
CHAIN_ATTENUATION = 0.7
