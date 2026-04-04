import numpy as np

LIKELIHOOD = np.array([
    [0.90, 0.05, 0.04, 0.01],
    [0.05, 0.85, 0.05, 0.05],
    [0.04, 0.08, 0.85, 0.03],
    [0.01, 0.02, 0.06, 0.91],
])

def kl_divergence(p, q):
    p = np.array(p) + 1e-9
    q = np.array(q) + 1e-9
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))

def cibp_update(prior, outcome, kl_threshold=0.1):
    prior = np.array(prior)
    likelihood_col = LIKELIHOOD[outcome, :]
    unnorm = likelihood_col * prior
    new_posterior = unnorm / unnorm.sum()
    kl = kl_divergence(new_posterior, prior)
    replan = kl > kl_threshold
    return new_posterior.tolist(), replan