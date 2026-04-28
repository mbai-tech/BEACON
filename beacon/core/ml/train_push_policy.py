"""Train the push/avoid policy from collected decision logs.

After running enough simulations with COLLECT=True in push_policy.py:

    python -m beacon.core.ml.train_push_policy

The trained model is saved to beacon/core/ml/push_policy_model.pkl and will
be loaded automatically by PushAvoidPolicy on the next planner import.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from beacon.core.ml.push_policy import LOG_PATH, MODEL_PATH, extract_features

FEATURE_NAMES = [
    "j_avoid",
    "j_push",
    "cost_diff",
    "push_belief_risk",
    "corridor_gain",
    "margin_excess",
    "dist_to_goal",
    "is_stuck",
]


def build_dataset(log_path: Path = LOG_PATH):
    if not log_path.exists():
        raise FileNotFoundError(
            f"No decision log at {log_path}.\n"
            "Run simulations first with COLLECT=True in push_policy.py."
        )
    with open(log_path, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} decision records from {log_path.name}")

    X, y = [], []
    for r in records:
        # Label: should push?
        #   push chosen + outcome good  → push was right (y=1)
        #   avoid chosen + outcome bad  → avoid failed, push might have helped (y=1)
        #   push chosen + outcome bad   → push was wrong (y=0)
        #   avoid chosen + outcome good → avoid worked fine (y=0)
        chose_push = int(r["action"] == "push")
        outcome    = r["outcome"]
        label = int(chose_push == outcome)   # XOR-complement: 1 when action matched outcome sense
        X.append(extract_features(
            r["j_avoid"], r["j_push"], r["push_belief_risk"],
            r["corridor_gain"], r["safety_margin_push"],
            r["safety_margin_threshold"], r["dist_to_goal"], r["stuck_events"],
        ))
        y.append(label)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def train(output_path: Path = MODEL_PATH) -> GradientBoostingClassifier:
    X, y = build_dataset()
    if len(X) < 30:
        raise ValueError(
            f"Only {len(X)} labeled decisions — need at least 30. "
            "Run more simulations."
        )

    push_n = int(y.sum())
    avoid_n = len(y) - push_n
    print(f"Class balance: push={push_n} ({push_n/len(y):.1%})  "
          f"avoid={avoid_n} ({avoid_n/len(y):.1%})")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    model.fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)
    print(f"Test accuracy: {acc:.1%}  (n_train={len(y_tr)}  n_test={len(y_te)})")

    print("\nFeature importances:")
    for name, imp in sorted(
        zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {name:<20} {imp:.3f}")

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {output_path}")
    return model


if __name__ == "__main__":
    train()
