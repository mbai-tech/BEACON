"""ML push/avoid policy for the BEACON planner.

Phase 1: COLLECT=True logs every reconcile decision to decision_log.pkl.
Phase 2: after running train_push_policy.py, loads the model and overrides
         the handcrafted formula in reconcile_trajectory_decision.

Feature vector (8-d):
  0  j_avoid              avoid branch total cost
  1  j_push               push branch total cost
  2  cost_diff            j_avoid - j_push  (positive = push is cheaper)
  3  push_belief_risk     E[risk | CIBP belief] for the target obstacle
  4  corridor_gain        predicted clearance gain from the push
  5  margin_excess        push safety_margin - safety_margin_threshold
  6  dist_to_goal         ||goal - position||
  7  is_stuck             1.0 if stuck_events > 0
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from beacon.core.planner import TrajectoryCandidate

_HERE = Path(__file__).parent
MODEL_PATH = _HERE / "push_policy_model.pkl"
LOG_PATH   = _HERE / "decision_log.pkl"

COLLECT = True   # flip to False to disable data collection without other changes


def extract_features(
    j_avoid: float,
    j_push: float,
    push_belief_risk: float,
    corridor_gain: float,
    safety_margin_push: float,
    safety_margin_threshold: float,
    dist_to_goal: float,
    stuck_events: int,
) -> np.ndarray:
    return np.array([
        j_avoid,
        j_push,
        j_avoid - j_push,
        push_belief_risk,
        corridor_gain,
        safety_margin_push - safety_margin_threshold,
        dist_to_goal,
        float(stuck_events > 0),
    ], dtype=float)


class PushAvoidPolicy:
    """Optional ML override for reconcile_trajectory_decision.

    Load once at module level in planner.py:
        _push_policy = PushAvoidPolicy()

    Then at each reconcile call site, after calling the original function:
        selected = _push_policy.maybe_override(
            selected, avoid_candidate, push_candidate,
            safety_margin_threshold, push_belief_risk,
            decision_log=decision_log, dist_to_goal=...,
            stuck_events=..., step=...,
        )
    """

    def __init__(self) -> None:
        self._model = None
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                print(f"[PushAvoidPolicy] model loaded from {MODEL_PATH.name}")
            except Exception as exc:
                self._model = None
                print(
                    "[PushAvoidPolicy] failed to load model "
                    f"({exc.__class__.__name__}: {exc}); using handcrafted policy"
                )
        else:
            print("[PushAvoidPolicy] no model yet — collecting decisions for future training")

    def maybe_override(
        self,
        selected: "TrajectoryCandidate",
        avoid_candidate: "TrajectoryCandidate",
        push_candidate: "TrajectoryCandidate | None",
        safety_margin_threshold: float,
        push_belief_risk: float = 0.0,
        decision_log: list[dict] | None = None,
        dist_to_goal: float = 0.0,
        stuck_events: int = 0,
        step: int = 0,
        site: str = "main",
    ) -> "TrajectoryCandidate":
        """Log the decision, then optionally override it with the ML model."""
        if push_candidate is None:
            return selected

        if COLLECT and decision_log is not None:
            decision_log.append({
                "j_avoid":                 avoid_candidate.total_cost,
                "j_push":                  push_candidate.total_cost,
                "push_belief_risk":        push_belief_risk,
                "corridor_gain":           push_candidate.corridor_gain,
                "safety_margin_push":      push_candidate.safety_margin,
                "safety_margin_threshold": safety_margin_threshold,
                "dist_to_goal":            dist_to_goal,
                "stuck_events":            stuck_events,
                "action":                  selected.mode,
                "step":                    step,
                "site":                    site,
            })

        if self._model is None:
            return selected

        features = extract_features(
            avoid_candidate.total_cost,
            push_candidate.total_cost,
            push_belief_risk,
            push_candidate.corridor_gain,
            push_candidate.safety_margin,
            safety_margin_threshold,
            dist_to_goal,
            stuck_events,
        ).reshape(1, -1)
        ml_push = bool(self._model.predict(features)[0])
        return push_candidate if ml_push else avoid_candidate

    def save_run(
        self,
        decision_log: list[dict],
        path: list[tuple[float, float]],
        goal: list[float],
    ) -> None:
        """Retrospectively label decisions and append them to the persistent log."""
        if not decision_log:
            return
        path_arr = np.array(path)
        goal_arr = np.array(goal[:2])
        labeled: list[dict] = []
        for rec in decision_log:
            s      = rec["step"]
            future = min(s + 10, len(path_arr) - 1)
            if s < len(path_arr) and future > s:
                d_now    = float(np.linalg.norm(goal_arr - path_arr[s]))
                d_future = float(np.linalg.norm(goal_arr - path_arr[future]))
                outcome  = int((d_now - d_future) > 0.02)
            else:
                outcome = 0
            labeled.append({**rec, "outcome": outcome})

        existing: list[dict] = []
        if LOG_PATH.exists():
            with open(LOG_PATH, "rb") as f:
                try:
                    existing = pickle.load(f)
                except Exception:
                    existing = []
        existing.extend(labeled)
        LOG_PATH.parent.mkdir(exist_ok=True, parents=True)
        with open(LOG_PATH, "wb") as f:
            pickle.dump(existing, f)
        print(f"[PushAvoidPolicy] +{len(labeled)} decisions saved "
              f"(total={len(existing)}) → {LOG_PATH.name}")
