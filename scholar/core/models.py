from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ContactEvent:
    obstacle_id:       int
    true_class:        str
    belief_at_contact: dict        # class → probability
    outcome:           str         # e.g. "push", "avoid", "blocked"
    contact_area:      float
    speed_at_contact:  float
    battery_at_contact: float
    step:              int


@dataclass
class ReplanEvent:
    step:           int
    kl_divergence:  float
    trigger:        str            # e.g. "stuck", "belief_update", "forced"
    duration_ms:    float = 0.0   # wall-clock time spent replanning


@dataclass
class RolloutRecord:
    """Complete episode record; every metric function takes one of these."""

    scene_family:   str
    planner:        str
    success:        bool
    trajectory:     list[tuple[float, float, float]]   # (x, y, θ) per step
    contact_events: list[ContactEvent]
    replan_events:  list[ReplanEvent]
    final_beliefs:  dict   # obstacle_id → belief dict
    true_classes:   dict   # obstacle_id → str
    battery_history: list[float]                        # one entry per step
    stuck_events:   int
    total_steps:    int
    goal_reached:   bool
    final_pose:     tuple[float, float, float]
    goal_pose:      tuple[float, float, float]


@dataclass
class SceneSummary:
    """Per-scene diagnostic bundle collected by run_scholar."""

    family:                  str
    success:                 bool
    final_battery:           float
    total_semantic_damage:   float   # Σ contact_cost · Δdist over trajectory
    forbidden_contact_rate:  float   # forbidden-class contact steps / total steps
    fragile_contact_rate:    float   # fragile-class contact steps / total steps
    mean_j_risk:             float
    mean_j_vel:              float
    mean_j_resource:         float
    n_cibp_replans:          int     # steps where semantic pruning was relaxed
    n_stuck_events:              int
    mean_speed_at_contact:       float           # battery-coupled: speed when robot touches obstacle
    dominant_j:                  str             # J component with highest weighted cost most often
    battery_at_first_stuck:      Optional[float] # None if never stuck
    battery_contact_log:         list            # list[dict] — one entry per contact/stuck event
    low_battery_contact_fraction: float          # fraction of contact events where b < 0.3


@dataclass
class SimulationFrame:
    """One animation frame for the online simulation."""

    position: tuple[float, float]
    obstacles: list[dict]
    message: str


@dataclass
class OnlineSurpResult:
    """Final simulation bundle used by the visualizer and CLI output."""

    family: str
    seed: int
    success: bool
    path: list[tuple[float, float]]
    frames: list[SimulationFrame]
    scene: dict
    initial_scene: dict
    contact_log: list[str]
    sensed_ids: list[int]
    scene_summary: Optional[SceneSummary] = None
