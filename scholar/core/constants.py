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
ROBOT_RADIUS           = 0.14
ROBOT_BODY_RESOLUTION  = 8
RAY_COUNT              = 24
RAY_STEP_SIZE          = 0.06
CHAIN_ATTENUATION      = 0.7

# BEACON paper battery model
BATTERY_INITIAL   = 1000.0
DELTA_MOVE        = 1.0
DELTA_TIME        = 0.1
DELTA_COL         = 2.0
T_MAP             = 0.2

# Semantic cost thresholds
C_MAX    = 7
C_THRESH = 5

# Scene configuration catalogue
DENSITY_OBSTACLE_COUNTS = {
    "sparse":  (5,  10),
    "medium":  (15, 25),
    "dense":   (30, 50),
}

SCENE_CONFIGS = {
    "S-U": ("sparse",  "uniform"),
    "S-M": ("sparse",  "mixed"),
    "M-U": ("medium",  "uniform"),
    "M-M": ("medium",  "mixed"),
    "D-U": ("dense",   "uniform"),
    "D-M": ("dense",   "mixed"),
}
