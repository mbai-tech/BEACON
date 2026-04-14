"""
metrics.py — Episode metric logging and aggregation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class EpisodeMetrics:
    planner_name: str
    seed: int
    success: bool
    steps: int
    path_length: float              # total distance travelled
    damage: float                   # cumulative contact penalty
    n_contacts: int                 # number of obstacle contacts
    n_replans: int                  # number of replanning events
    contact_log: List[Dict] = field(default_factory=list)


def path_length(path: List[np.ndarray]) -> float:
    """Compute the total Euclidean length of a path."""
    if len(path) < 2:
        return 0.0
    return float(sum(
        np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
        for i in range(len(path) - 1)
    ))


def aggregate_metrics(episodes: List[EpisodeMetrics]) -> Dict:
    """Compute aggregate statistics over a list of episodes."""
    if not episodes:
        return {}

    success_rate = float(np.mean([e.success for e in episodes]))
    avg_steps = float(np.mean([e.steps for e in episodes]))
    avg_length = float(np.mean([e.path_length for e in episodes]))
    avg_damage = float(np.mean([e.damage for e in episodes]))
    avg_contacts = float(np.mean([e.n_contacts for e in episodes]))

    return {
        "n_episodes": len(episodes),
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_path_length": avg_length,
        "avg_damage": avg_damage,
        "avg_contacts": avg_contacts,
    }


def print_summary(agg: Dict) -> None:
    print(f"Episodes:      {agg.get('n_episodes', 0)}")
    print(f"Success rate:  {agg.get('success_rate', 0.0):.1%}")
    print(f"Avg steps:     {agg.get('avg_steps', 0.0):.1f}")
    print(f"Avg path len:  {agg.get('avg_path_length', 0.0):.3f} m")
    print(f"Avg damage:    {agg.get('avg_damage', 0.0):.3f}")
    print(f"Avg contacts:  {agg.get('avg_contacts', 0.0):.1f}")
