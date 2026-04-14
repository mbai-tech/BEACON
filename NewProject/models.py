from dataclasses import dataclass


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
