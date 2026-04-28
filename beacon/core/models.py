from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationFrame:
    """One animation frame for the online simulation."""

    position: tuple[float, float]
    obstacles: list[dict]
    message: str


@dataclass
class SceneSummary:
    """Post-run BEACON diagnostics used for analysis and adaptation."""

    family: str
    success: bool
    final_battery: float
    total_semantic_damage: float
    forbidden_contact_rate: float
    fragile_contact_rate: float
    mean_j_risk: float
    mean_j_vel: float
    mean_j_resource: float
    n_cibp_replans: int
    n_stuck_events: int
    mean_speed_at_contact: float
    dominant_j: str
    battery_at_first_stuck: float | None
    battery_contact_log: list[dict]
    low_battery_contact_fraction: float


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
