import math


CLASSES  = ["safe", "movable", "fragile", "forbidden"]
OUTCOMES = ["no_displacement", "displacement", "damage", "hard_flag"]

_N    = len(CLASSES)           # 4
_DIAG = 0.7
_OFF  = 0.1 / (_N - 1)        # 0.1 / 3 ≈ 0.03333


class CIBP:

    CLASSES  = CLASSES
    OUTCOMES = OUTCOMES

    # L[outcome_idx][class_idx] = P(outcome | class)
    # Row order matches OUTCOMES; column order matches CLASSES.
    L: list[list[float]] = [
        #  safe    movable  fragile  forbidden
        [_DIAG,  _OFF,    _OFF,    _OFF   ],   # no_displacement
        [_OFF,   _DIAG,   _OFF,    _OFF   ],   # displacement
        [_OFF,   _OFF,    _DIAG,   _OFF   ],   # damage
        [_OFF,   _OFF,    _OFF,    _DIAG  ],   # hard_flag
    ]

    def __init__(self, tau: float = 0.15) -> None:
        self.tau = tau

    def update(
        self,
        prior: dict[str, float],
        outcome: str,
    ) -> tuple[dict[str, float], float]:
        if outcome not in self.OUTCOMES:
            raise ValueError(
                f"Unknown outcome {outcome!r}. Must be one of {self.OUTCOMES}."
            )

        o_idx = self.OUTCOMES.index(outcome)
        likelihood_row = self.L[o_idx]

        # ── Unnormalized posterior: L(outcome | class) × prior(class) ────────
        unnorm = {
            cls: likelihood_row[c_idx] * prior.get(cls, 0.0)
            for c_idx, cls in enumerate(self.CLASSES)
        }

        total = sum(unnorm.values())
        if total <= 0.0:
            raise ValueError(
                "Unnormalized posterior sums to zero — prior may be degenerate."
            )
            
        posterior = {cls: v / total for cls, v in unnorm.items()}

        # ── KL(posterior ∥ prior) in nats ─────────────────────────────────────
        # Guard against zero prior entries with a tiny epsilon so that classes
        # with near-zero prior don't produce -inf log terms when the posterior
        # also rounds to near-zero.  Classes where posterior[c] == 0 are skipped
        # (0 · ln(0) = 0 by convention).
        _EPS = 1e-12
        kl = 0.0
        for cls in self.CLASSES:
            p = posterior[cls]
            if p <= 0.0:
                continue
            q = prior.get(cls, 0.0)
            kl += p * math.log(p / max(q, _EPS))
        kl = max(0.0, kl)   # clamp floating-point underflow to zero

        return posterior, kl

    def likelihood_vector(self, outcome: str) -> dict[str, float]:
        if outcome not in self.OUTCOMES:
            raise ValueError(
                f"Unknown outcome {outcome!r}. Must be one of {self.OUTCOMES}."
            )
        o_idx = self.OUTCOMES.index(outcome)
        return {cls: self.L[o_idx][c_idx] for c_idx, cls in enumerate(self.CLASSES)}

    def map_class(self, belief: dict[str, float]) -> str:
        """Return the MAP (most probable) class from a belief dict."""
        return max(belief, key=belief.__getitem__)
