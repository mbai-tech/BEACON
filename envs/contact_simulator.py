import numpy as np

OUTCOME_NULL = 0
OUTCOME_DISPLACEMENT = 1
OUTCOME_DAMAGE = 2
OUTCOME_HARDFLAG = 3

LIKELIHOOD = np.array([
    [0.90, 0.05, 0.04, 0.01],
    [0.05, 0.85, 0.05, 0.05],
    [0.04, 0.08, 0.85, 0.03],
    [0.01, 0.02, 0.06, 0.91],
])

def simulate_contact(true_class_str):
    """Given the true class, sample what outcome the robot observes."""
    class_idx = ["safe","movable","fragile","forbidden"].index(true_class_str)
    probs = LIKELIHOOD[:, class_idx]
    return np.random.choice(4, p=probs)

def outcome_name(outcome_int):
    return ["null","displacement","damage","hardflag"][outcome_int]