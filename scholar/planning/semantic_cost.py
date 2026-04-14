"""
semantic_cost.py — Physics-informed cost assignment pipeline.

Given an obstacle and the current belief over its semantic class,
compute the expected traversal cost and the expected damage/penalty
incurred by contact.
"""

import numpy as np
from typing import Dict

# Base costs for each semantic class
SEMANTIC_COSTS: Dict[str, float] = {
    "safe":      1.0,
    "movable":   3.0,
    "fragile":  15.0,
    "forbidden": 1000.0,
}

# Damage penalties incurred on contact (on top of traversal cost)
CONTACT_PENALTIES: Dict[str, float] = {
    "safe":      0.0,
    "movable":   1.0,
    "fragile":  10.0,
    "forbidden": 500.0,
}


def expected_cost(belief: Dict[str, float]) -> float:
    """Expected traversal cost given a probability distribution over classes.

    Parameters
    ----------
    belief : dict mapping semantic class name → probability (should sum to 1)
    """
    return sum(belief.get(cls, 0.0) * cost for cls, cost in SEMANTIC_COSTS.items())


def expected_penalty(belief: Dict[str, float]) -> float:
    """Expected contact penalty given a class belief distribution."""
    return sum(belief.get(cls, 0.0) * pen for cls, pen in CONTACT_PENALTIES.items())


def bayesian_update(
    prior: Dict[str, float],
    contact_observed: bool,
    likelihood_contact: Dict[str, float],
    likelihood_no_contact: Dict[str, float],
) -> Dict[str, float]:
    """Update a semantic class belief with a contact observation.

    Parameters
    ----------
    prior                  : prior class probabilities
    contact_observed       : True if contact occurred, False otherwise
    likelihood_contact     : P(contact | class) for each class
    likelihood_no_contact  : P(no_contact | class) for each class
    """
    likelihoods = likelihood_contact if contact_observed else likelihood_no_contact
    unnorm = {cls: prior.get(cls, 0.0) * likelihoods.get(cls, 1.0) for cls in prior}
    total = sum(unnorm.values())
    if total < 1e-12:
        return {cls: 1.0 / len(prior) for cls in prior}
    return {cls: v / total for cls, v in unnorm.items()}


def uniform_prior(classes=("safe", "movable", "fragile", "forbidden")) -> Dict[str, float]:
    """Uniform distribution over semantic classes."""
    n = len(classes)
    return {cls: 1.0 / n for cls in classes}


def physics_informed_cost(
    obstacle_mass: float,
    obstacle_size: float,
    belief: Dict[str, float],
) -> float:
    """Combine semantic expected cost with physics-derived difficulty.

    Heavier and larger obstacles are harder to push, so their effective
    traversal cost is scaled up relative to pure semantic cost.
    """
    base = expected_cost(belief)
    # Physics scaling: heavier / larger = harder to move
    physics_scale = 1.0 + 0.1 * obstacle_mass + 0.05 * obstacle_size
    return base * physics_scale
