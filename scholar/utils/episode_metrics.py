from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.special import rel_entr

from scholar.core.models import ContactEvent, RolloutRecord

_CLASSES = ("safe", "movable", "fragile", "forbidden")
_EVAL_COSTS: dict[str, float] = {"safe": 1, "movable": 3, "fragile": 15, "forbidden": 1000}


def compute_success(r: RolloutRecord) -> bool:
    x_T, y_T, theta_T = r.final_pose
    x_g, y_g, theta_g = r.goal_pose
    pos_dist = math.hypot(x_T - x_g, y_T - y_g)
    angle_diff = abs((theta_T - theta_g + 180.0) % 360.0 - 180.0)
    return r.goal_reached and pos_dist <= 0.15 and angle_diff <= 15.0


def compute_path_length(r: RolloutRecord) -> float:
    return sum(
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1, _), (x2, y2, _) in zip(r.trajectory, r.trajectory[1:])
    )


def compute_rotation_cost(r: RolloutRecord) -> float:
    return sum(
        abs((t2 - t1 + 180.0) % 360.0 - 180.0)
        for (_, _, t1), (_, _, t2) in zip(r.trajectory, r.trajectory[1:])
    )


def compute_contact_count_by_class(r: RolloutRecord) -> dict[str, int]:
    counts: dict[str, int] = {c: 0 for c in _CLASSES}
    for event in r.contact_events:
        counts[event.true_class] += 1
    return counts


def compute_contact_severity(r: RolloutRecord) -> float:
    return sum(e.contact_area for e in r.contact_events)


def compute_semantic_damage(r: RolloutRecord) -> float:
    return sum(_EVAL_COSTS[e.true_class] * e.contact_area for e in r.contact_events)


def compute_belief_error(r: RolloutRecord) -> float:
    observed_ids = {e.obstacle_id for e in r.contact_events}
    kl_values: list[float] = []
    for obs_id in observed_ids:
        if obs_id not in r.final_beliefs or obs_id not in r.true_classes:
            continue
        belief = r.final_beliefs[obs_id]
        true_class = r.true_classes[obs_id]
        p = np.array([belief.get(c, 0.0) for c in _CLASSES])
        q = np.array([1.0 if c == true_class else 0.0 for c in _CLASSES])
        kl_values.append(float(np.sum(rel_entr(p, q))))
    return float(np.mean(kl_values)) if kl_values else 0.0


def compute_trap_recovery_rate(r: RolloutRecord) -> float | None:
    if r.scene_family != "semantic_trap":
        return None

    # Group events per obstacle in chronological order
    events_by_id: dict[int, list[ContactEvent]] = {}
    for e in sorted(r.contact_events, key=lambda e: e.step):
        events_by_id.setdefault(e.obstacle_id, []).append(e)

    # Trap obstacle: first (by initial-contact step) fragile obstacle
    # whose MAP class at first contact was 'safe'
    trap_id: int | None = None
    trap_first_step: int = -1
    for obs_id, obs_events in sorted(events_by_id.items(), key=lambda kv: kv[1][0].step):
        if r.true_classes.get(obs_id) != "fragile":
            continue
        first = obs_events[0]
        if max(first.belief_at_contact, key=first.belief_at_contact.__getitem__) == "safe":
            trap_id = obs_id
            trap_first_step = first.step
            break

    if trap_id is None:
        return None

    recovered = not any(
        e.obstacle_id == trap_id and e.step > trap_first_step
        for e in r.contact_events
    )
    return 1.0 if recovered else 0.0


def compute_low_battery_contact_fraction(r: RolloutRecord) -> float:
    if not r.contact_events:
        return 0.0
    return sum(e.battery_at_contact < 0.3 for e in r.contact_events) / len(r.contact_events)


def compute_mean_speed_at_contact(r: RolloutRecord) -> float:
    if not r.contact_events:
        return 0.0
    return sum(e.speed_at_contact for e in r.contact_events) / len(r.contact_events)


def compute_battery_coupling_diagnostic(
    r: RolloutRecord, w_v_floor: float, w_v_range: float
) -> list[dict]:
    return [
        {
            "battery":      e.battery_at_contact,
            "speed":        e.speed_at_contact,
            "w_v_effective": w_v_floor + w_v_range * (1.0 - e.battery_at_contact),
        }
        for e in r.contact_events
    ]


def compute_forbidden_rate(records: list[RolloutRecord]) -> float:
    return sum(
        any(e.true_class == "forbidden" for e in r.contact_events)
        for r in records
    ) / len(records)


def compute_fragile_rate(records: list[RolloutRecord]) -> float:
    return sum(
        any(e.true_class == "fragile" for e in r.contact_events)
        for r in records
    ) / len(records)


def compute_delta_noise(records_low: list[RolloutRecord], records_high: list[RolloutRecord]) -> float:
    low_rate  = sum(compute_success(r) for r in records_low)  / len(records_low)
    high_rate = sum(compute_success(r) for r in records_high) / len(records_high)
    return low_rate - high_rate


def compute_delta_perturb(records_static: list[RolloutRecord], records_perturbed: list[RolloutRecord]) -> float:
    static_rate    = sum(compute_success(r) for r in records_static)    / len(records_static)
    perturbed_rate = sum(compute_success(r) for r in records_perturbed) / len(records_perturbed)
    return static_rate - perturbed_rate


def compute_recovery_rate(records_perturbed: list[RolloutRecord]) -> float:
    return sum(compute_success(r) for r in records_perturbed) / len(records_perturbed)


def compute_replan_count(records: list[RolloutRecord]) -> float:
    return sum(len(r.replan_events) for r in records) / len(records)


def compute_planning_time(r: RolloutRecord) -> dict[str, float]:
    if not r.replan_events:
        return {"mean_replan_ms": 0.0, "p95_replan_ms": 0.0}
    times = np.array([e.duration_ms for e in r.replan_events], dtype=float)
    return {
        "mean_replan_ms": float(np.mean(times)),
        "p95_replan_ms":  float(np.percentile(times, 95)),
    }


def _per_record_metrics(r: RolloutRecord) -> dict:
    return {
        "scene_family":                  r.scene_family,
        "planner":                       r.planner,
        "success":                       compute_success(r),
        "path_length":                   compute_path_length(r),
        "rotation_cost":                 compute_rotation_cost(r),
        "contact_count_by_class":        compute_contact_count_by_class(r),
        "contact_severity":              compute_contact_severity(r),
        "semantic_damage":               compute_semantic_damage(r),
        "belief_error":                  compute_belief_error(r),
        "trap_recovery_rate":            compute_trap_recovery_rate(r),
        "low_battery_contact_fraction":  compute_low_battery_contact_fraction(r),
        "mean_speed_at_contact":         compute_mean_speed_at_contact(r),
        "planning_time":                 compute_planning_time(r),
    }


def _aggregate_over(records: list[RolloutRecord], metrics: list[dict]) -> dict:
    trap_rates = [m["trap_recovery_rate"] for m in metrics if m["trap_recovery_rate"] is not None]
    n = len(metrics)
    return {
        "n":                                  n,
        "success_rate":                       sum(m["success"] for m in metrics) / n,
        "mean_path_length":                   sum(m["path_length"] for m in metrics) / n,
        "mean_rotation_cost":                 sum(m["rotation_cost"] for m in metrics) / n,
        "mean_contact_severity":              sum(m["contact_severity"] for m in metrics) / n,
        "mean_semantic_damage":               sum(m["semantic_damage"] for m in metrics) / n,
        "mean_belief_error":                  sum(m["belief_error"] for m in metrics) / n,
        "mean_low_battery_contact_fraction":  sum(m["low_battery_contact_fraction"] for m in metrics) / n,
        "mean_speed_at_contact":              sum(m["mean_speed_at_contact"] for m in metrics) / n,
        "mean_replan_count":                  compute_replan_count(records),
        "forbidden_rate":                     compute_forbidden_rate(records),
        "fragile_rate":                       compute_fragile_rate(records),
        "trap_recovery_rate":                 sum(trap_rates) / len(trap_rates) if trap_rates else None,
    }


def run_all_metrics(
    rollout_records: list[RolloutRecord],
    weight_histories: dict[str, list[dict]],
) -> dict:
    per_record = [_per_record_metrics(r) for r in rollout_records]

    # Group by planner → family, collecting (records, metrics) pairs
    grouped: dict[str, dict[str, tuple[list, list]]] = defaultdict(
        lambda: defaultdict(lambda: ([], []))
    )
    for r, m in zip(rollout_records, per_record):
        recs, mets = grouped[r.planner][r.scene_family]
        recs.append(r)
        mets.append(m)

    by_planner: dict[str, dict] = {}
    for planner, families in grouped.items():
        all_r: list[RolloutRecord] = []
        all_m: list[dict] = []
        fam_results: dict[str, dict] = {}
        for family, (recs, mets) in families.items():
            fam_results[family] = _aggregate_over(recs, mets)
            all_r.extend(recs)
            all_m.extend(mets)
        by_planner[planner] = {
            "by_family": fam_results,
            "aggregate": _aggregate_over(all_r, all_m),
        }

    # Robustness — partition by family name; delta_noise uses sparse vs cluttered
    # as the low-noise / high-noise proxy; delta_perturb uses "perturbed" family
    by_family: dict[str, list[RolloutRecord]] = defaultdict(list)
    for r in rollout_records:
        by_family[r.scene_family].append(r)

    sparse    = by_family.get("sparse", [])
    cluttered = by_family.get("cluttered", [])
    perturbed = by_family.get("perturbed", [])
    static    = [r for r in rollout_records if r.scene_family != "perturbed"]

    robustness: dict[str, float | None] = {
        "delta_noise":   compute_delta_noise(sparse, cluttered)   if sparse and cluttered else None,
        "delta_perturb": compute_delta_perturb(static, perturbed) if static and perturbed else None,
        "recovery_rate": compute_recovery_rate(perturbed)         if perturbed            else None,
        "replan_count":  compute_replan_count(rollout_records),
    }

    return {
        "per_record":         per_record,
        "by_planner":         by_planner,
        "robustness":         robustness,
        "weight_convergence": {
            fam: compute_weight_convergence(hist)
            for fam, hist in weight_histories.items()
        },
    }


def validate_metrics(results: dict) -> None:
    def _check(label: str, passed: bool, offending: object = None) -> None:
        if passed:
            print(f"PASS  {label}")
        else:
            print(f"WARN  {label} — offending value: {offending}")

    def _deep(d: dict, *keys: str):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    # 1. Semantic damage is non-negative for all records
    bad_damage = [
        (m["planner"], m["scene_family"], m["semantic_damage"])
        for m in results["per_record"]
        if m["semantic_damage"] < 0
    ]
    _check("semantic_damage >= 0 for all records", not bad_damage, bad_damage or None)

    # 2. Belief error: proposed < no_update on semantic_trap scenes
    proposed_be  = _deep(results, "by_planner", "proposed",   "by_family", "semantic_trap", "mean_belief_error")
    no_update_be = _deep(results, "by_planner", "no_update",  "by_family", "semantic_trap", "mean_belief_error")
    if proposed_be is None or no_update_be is None:
        print("WARN  belief_error(proposed) < belief_error(no_update) on semantic_trap — insufficient data")
    else:
        _check(
            "belief_error(proposed) < belief_error(no_update) on semantic_trap",
            proposed_be < no_update_be,
            f"proposed={proposed_be:.4f}, no_update={no_update_be:.4f}",
        )

    # 3. Forbidden rate is 0.0 for collision-free planner
    cf_forbidden = _deep(results, "by_planner", "collision_free", "aggregate", "forbidden_rate")
    if cf_forbidden is None:
        print("WARN  forbidden_rate(collision_free) == 0.0 — no collision_free planner data")
    else:
        _check(
            "forbidden_rate == 0.0 for collision_free planner",
            cf_forbidden == 0.0,
            f"{cf_forbidden:.4f}",
        )

    # 4. low_battery_contact_fraction in [0, 1] for all records
    bad_lbcf = [
        (m["planner"], m["scene_family"], m["low_battery_contact_fraction"])
        for m in results["per_record"]
        if not (0.0 <= m["low_battery_contact_fraction"] <= 1.0)
    ]
    _check("low_battery_contact_fraction in [0, 1] for all records", not bad_lbcf, bad_lbcf or None)


def analyze_battery_vs_damage(records: list[RolloutRecord]) -> dict:
    def _fragile_event_rate(r: RolloutRecord) -> float:
        if not r.contact_events:
            return 0.0
        return sum(e.true_class == "fragile" for e in r.contact_events) / len(r.contact_events)

    def _forbidden_event_rate(r: RolloutRecord) -> float:
        if not r.contact_events:
            return 0.0
        return sum(e.true_class == "forbidden" for e in r.contact_events) / len(r.contact_events)

    batteries = np.array([r.battery_history[-1] for r in records])
    q25, q50, q75 = float(np.percentile(batteries, 25)), float(np.percentile(batteries, 50)), float(np.percentile(batteries, 75))
    boundaries = [q25, q50, q75]

    def _quartile(b: float) -> str:
        if b <= q25: return "Q1"
        if b <= q50: return "Q2"
        if b <= q75: return "Q3"
        return "Q4"

    bins: dict[str, list[tuple[float, float, float, float, float]]] = {q: [] for q in ("Q1", "Q2", "Q3", "Q4")}
    for r in records:
        battery = float(r.battery_history[-1])
        bins[_quartile(battery)].append((
            battery,
            compute_semantic_damage(r),
            _fragile_event_rate(r),
            _forbidden_event_rate(r),
            compute_path_length(r),
        ))

    battery_ranges = {
        "Q1": (float(batteries.min()), q25),
        "Q2": (q25, q50),
        "Q3": (q50, q75),
        "Q4": (q75, float(batteries.max())),
    }

    quartiles: dict[str, dict] = {}
    for q, entries in bins.items():
        if not entries:
            quartiles[q] = {"battery_range": battery_ranges[q], "n": 0}
            continue
        bats, damages, frags, forbids, paths = zip(*entries)
        n = len(entries)
        quartiles[q] = {
            "battery_range":           battery_ranges[q],
            "n":                       n,
            "mean_final_battery":      sum(bats)    / n,
            "mean_semantic_damage":    sum(damages) / n,
            "mean_fragile_contact_rate":  sum(frags)   / n,
            "mean_forbidden_contact_rate": sum(forbids) / n,
            "mean_path_length":        sum(paths)   / n,
        }

    return {"quartile_boundaries": boundaries, "quartiles": quartiles}


def analyze_velocity_penalty_vs_speed(
    records: list[RolloutRecord], w_v_floor: float, w_v_range: float
) -> dict:
    events: list[dict] = []
    for r in records:
        events.extend(compute_battery_coupling_diagnostic(r, w_v_floor, w_v_range))

    if not events:
        return {"n_contact_events": 0, "correlation": float("nan"), "correlation_positive": False, "quintile_boundaries": [], "quintiles": {}}

    w_v_eff = np.array([e["w_v_effective"]  for e in events], dtype=float)
    speeds  = np.array([e["speed"]          for e in events], dtype=float)

    if w_v_eff.std() == 0 or speeds.std() == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(w_v_eff, speeds)[0, 1])

    # Quintile bins on w_v_effective — 4 cut points → 5 bins
    boundaries   = [float(np.percentile(w_v_eff, p)) for p in range(20, 100, 20)]
    quintile_idx = np.digitize(w_v_eff, boundaries)   # values 0..4

    wv_ranges: list[tuple[float, float]] = []
    prev = float(w_v_eff.min())
    for b in boundaries:
        wv_ranges.append((prev, b))
        prev = b
    wv_ranges.append((prev, float(w_v_eff.max())))

    bins: dict[int, list[float]] = {i: [] for i in range(5)}
    for i, idx in enumerate(quintile_idx):
        bins[int(idx)].append(float(speeds[i]))

    quintiles: dict[str, dict] = {}
    for i in range(5):
        label   = f"Q{i + 1}"
        entries = bins[i]
        if not entries:
            quintiles[label] = {"w_v_effective_range": wv_ranges[i], "n": 0}
            continue
        arr = np.array(entries, dtype=float)
        quintiles[label] = {
            "w_v_effective_range": wv_ranges[i],
            "n":                   len(entries),
            "mean_speed":          float(arr.mean()),
            "std_speed":           float(arr.std()),
        }

    return {
        "n_contact_events":    len(events),
        "correlation":         correlation,
        "correlation_positive": not (correlation != correlation) and correlation > 0,  # NaN-safe
        "quintile_boundaries": boundaries,
        "quintiles":           quintiles,
    }


def analyze_lowbattery_contact_vs_semantic_risk(records: list[RolloutRecord]) -> dict:
    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        return float("nan") if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])

    lbcf     = np.array([compute_low_battery_contact_fraction(r) for r in records], dtype=float)
    fragile  = np.array([float(any(e.true_class == "fragile"  for e in r.contact_events)) for r in records])
    forbidden = np.array([float(any(e.true_class == "forbidden" for e in r.contact_events)) for r in records])
    damage   = np.array([compute_semantic_damage(r) for r in records], dtype=float)

    correlations = {
        "lbcf_vs_fragile_rate":    _pearson(lbcf, fragile),
        "lbcf_vs_forbidden_rate":  _pearson(lbcf, forbidden),
        "lbcf_vs_semantic_damage": _pearson(lbcf, damage),
    }

    # Quintile bins — 4 cut points → 5 bins
    boundaries = [float(np.percentile(lbcf, p)) for p in range(20, 100, 20)]
    quintile_idx = np.digitize(lbcf, boundaries)  # values 0..4

    lbcf_ranges: list[tuple[float, float]] = []
    prev = float(lbcf.min())
    for b in boundaries:
        lbcf_ranges.append((prev, b))
        prev = b
    lbcf_ranges.append((prev, float(lbcf.max())))

    bins: dict[int, list[tuple]] = {i: [] for i in range(5)}
    for i, idx in enumerate(quintile_idx):
        bins[int(idx)].append((float(lbcf[i]), float(damage[i])))

    quintiles: dict[str, dict] = {}
    for i in range(5):
        label   = f"Q{i + 1}"
        entries = bins[i]
        if not entries:
            quintiles[label] = {"lbcf_range": lbcf_ranges[i], "n": 0}
            continue
        lbcfs, damages = zip(*entries)
        n = len(entries)
        quintiles[label] = {
            "lbcf_range":          lbcf_ranges[i],
            "n":                   n,
            "mean_lbcf":           sum(lbcfs)   / n,
            "mean_semantic_damage": sum(damages) / n,
        }

    return {"correlations": correlations, "quintile_boundaries": boundaries, "quintiles": quintiles}


def analyze_battery_vs_success(records: list[RolloutRecord]) -> dict:
    batteries = np.array([r.battery_history[-1] for r in records], dtype=float)
    successes  = np.array([compute_success(r)    for r in records], dtype=float)

    if batteries.std() == 0 or successes.std() == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(batteries, successes)[0, 1])

    # 9 cut points → 10 decile bins
    boundaries   = [float(np.percentile(batteries, p)) for p in range(10, 100, 10)]
    decile_idx   = np.digitize(batteries, boundaries)   # values 0..9

    battery_ranges: list[tuple[float, float]] = []
    prev = float(batteries.min())
    for b in boundaries:
        battery_ranges.append((prev, b))
        prev = b
    battery_ranges.append((prev, float(batteries.max())))

    bins: dict[int, list[tuple]] = {i: [] for i in range(10)}
    for i, (r, idx) in enumerate(zip(records, decile_idx)):
        bins[int(idx)].append((float(batteries[i]), float(successes[i]), r.stuck_events, len(r.replan_events)))

    deciles: dict[str, dict] = {}
    for i in range(10):
        label   = f"D{i + 1:02d}"
        entries = bins[i]
        if not entries:
            deciles[label] = {"battery_range": battery_ranges[i], "n": 0, "at_risk": None}
            continue
        bats, succs, stucks, replans = zip(*entries)
        n            = len(entries)
        success_rate = sum(succs) / n
        deciles[label] = {
            "battery_range":      battery_ranges[i],
            "n":                  n,
            "mean_final_battery": sum(bats)    / n,
            "success_rate":       success_rate,
            "mean_stuck_events":  sum(stucks)  / n,
            "mean_replan_count":  sum(replans) / n,
            "at_risk":            success_rate < 0.5,
        }

    return {"decile_boundaries": boundaries, "correlation": correlation, "deciles": deciles}


_REPLAN_BIN_LABELS  = ("0", "1-2", "3-5", "6-10", "10+")
_REPLAN_BIN_RANGES  = ((0, 0), (1, 2), (3, 5), (6, 10), (11, None))


def _replan_bin(n: int) -> int:
    if n == 0:  return 0
    if n <= 2:  return 1
    if n <= 5:  return 2
    if n <= 10: return 3
    return 4


def analyze_cibp_frequency_vs_belief(records: list[RolloutRecord]) -> dict:
    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        return float("nan") if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])

    def _quintile_bins(
        updates: np.ndarray,
        belief_errors: np.ndarray,
        damages: np.ndarray,
        trap_recoveries: list[float | None],
    ) -> tuple[list[float], dict[str, dict]]:
        if len(updates) < 2:
            return [], {}
        boundaries   = [float(np.percentile(updates, p)) for p in range(20, 100, 20)]
        quintile_idx = np.digitize(updates, boundaries)
        ranges: list[tuple[float, float]] = []
        prev = float(updates.min())
        for b in boundaries:
            ranges.append((prev, b))
            prev = b
        ranges.append((prev, float(updates.max())))
        bins: dict[int, list] = {i: [] for i in range(5)}
        for i, idx in enumerate(quintile_idx):
            bins[int(idx)].append((float(updates[i]), float(belief_errors[i]), float(damages[i]), trap_recoveries[i]))
        quintiles: dict[str, dict] = {}
        for i in range(5):
            label, entries = f"Q{i + 1}", bins[i]
            if not entries:
                quintiles[label] = {"update_count_range": ranges[i], "n": 0}
                continue
            upds, bes, dmgs, traps = zip(*entries)
            valid_traps = [v for v in traps if v is not None]
            n = len(entries)
            quintiles[label] = {
                "update_count_range":      ranges[i],
                "n":                       n,
                "mean_update_count":       sum(upds) / n,
                "mean_belief_error":       sum(bes)  / n,
                "mean_semantic_damage":    sum(dmgs) / n,
                "mean_trap_recovery_rate": sum(valid_traps) / len(valid_traps) if valid_traps else None,
            }
        return boundaries, quintiles

    updates_all = np.array([len(r.contact_events)     for r in records], dtype=float)
    be_all      = np.array([compute_belief_error(r)    for r in records], dtype=float)
    dmg_all     = np.array([compute_semantic_damage(r) for r in records], dtype=float)
    trap_all: list[float | None] = [compute_trap_recovery_rate(r) for r in records]

    global_bounds, global_quintiles = _quintile_bins(updates_all, be_all, dmg_all, trap_all)

    # Per-family: collect indices, then slice the global arrays
    family_indices: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        family_indices[r.scene_family].append(i)

    by_family: dict[str, dict] = {}
    for family, idxs in family_indices.items():
        u    = updates_all[idxs]
        be   = be_all[idxs]
        dmg  = dmg_all[idxs]
        traps = [trap_all[i] for i in idxs]
        fam_bounds, fam_quintiles = _quintile_bins(u, be, dmg, traps)
        by_family[family] = {
            "n":                   len(idxs),
            "correlation":         _pearson(u, be),
            "quintile_boundaries": fam_bounds,
            "quintiles":           fam_quintiles,
        }

    return {
        "n":                   len(records),
        "global_correlation":  _pearson(updates_all, be_all),
        "quintile_boundaries": global_bounds,
        "quintiles":           global_quintiles,
        "by_family":           by_family,
    }


def analyze_replan_tradeoff(records: list[RolloutRecord]) -> dict:
    bins: dict[int, list[tuple]] = {i: [] for i in range(5)}
    for r in records:
        rc  = len(r.replan_events)
        tpt = sum(e.duration_ms for e in r.replan_events)
        bins[_replan_bin(rc)].append((
            rc,
            tpt,
            float(compute_success(r)),
            compute_semantic_damage(r),
            compute_path_length(r),
        ))

    bin_stats: dict[str, dict] = {}
    for i, label in enumerate(_REPLAN_BIN_LABELS):
        entries = bins[i]
        lo, hi  = _REPLAN_BIN_RANGES[i]
        if not entries:
            bin_stats[label] = {
                "replan_range": (lo, hi),
                "n":            0,
            }
            continue
        rcs, tpts, succs, dmgs, pls = zip(*entries)
        n = len(entries)
        bin_stats[label] = {
            "replan_range":               (lo, hi),
            "n":                          n,
            "mean_replan_count":          sum(rcs)   / n,
            "mean_success_rate":          sum(succs) / n,
            "mean_total_planning_time_ms": sum(tpts)  / n,
            "mean_semantic_damage":       sum(dmgs)  / n,
            "mean_path_length":           sum(pls)   / n,
        }

    # Marginal return between consecutive non-empty bins
    populated = [
        (label, bin_stats[label])
        for label in _REPLAN_BIN_LABELS
        if bin_stats[label]["n"] > 0
    ]

    marginal_returns: list[dict] = []
    diminishing_threshold: str | None = None

    for (label_a, stats_a), (label_b, stats_b) in zip(populated, populated[1:]):
        d_success = stats_b["mean_success_rate"]  - stats_a["mean_success_rate"]
        d_replan  = stats_b["mean_replan_count"]  - stats_a["mean_replan_count"]
        marginal  = (d_success / d_replan) if d_replan != 0 else None
        diminishing = marginal is not None and marginal < 0.02
        marginal_returns.append({
            "from_bin":       label_a,
            "to_bin":         label_b,
            "delta_success":  d_success,
            "delta_replan":   d_replan,
            "marginal_return": marginal,
            "diminishing":    diminishing,
        })
        if diminishing and diminishing_threshold is None:
            diminishing_threshold = label_b

    return {
        "bins":                           bin_stats,
        "marginal_returns":               marginal_returns,
        "diminishing_return_threshold_bin": diminishing_threshold,
    }


def analyze_path_damage_pareto(records_by_planner: dict[str, list[RolloutRecord]]) -> dict:
    # Collect (mean_path_length, mean_semantic_damage) per planner per family
    planner_family_stats: dict[str, dict[str, tuple[float, float]]] = {}
    for planner, records in records_by_planner.items():
        by_family: dict[str, list[RolloutRecord]] = defaultdict(list)
        for r in records:
            by_family[r.scene_family].append(r)
        planner_family_stats[planner] = {
            family: (
                sum(compute_path_length(r)    for r in fam_records) / len(fam_records),
                sum(compute_semantic_damage(r) for r in fam_records) / len(fam_records),
            )
            for family, fam_records in by_family.items()
        }

    # Per-family Pareto analysis
    all_families: set[str] = {f for stats in planner_family_stats.values() for f in stats}
    by_family_out: dict[str, dict] = {}

    for family in all_families:
        points = [
            (planner, *planner_family_stats[planner][family])
            for planner in planner_family_stats
            if family in planner_family_stats[planner]
        ]  # list of (planner, path_length, damage)

        dominated: list[str] = []
        frontier:  list[str] = []
        for planner, pl, dmg in points:
            others = [(opl, odmg) for op, opl, odmg in points if op != planner]
            is_dominated = any(
                opl <= pl and odmg <= dmg and (opl < pl or odmg < dmg)
                for opl, odmg in others
            )
            (dominated if is_dominated else frontier).append(planner)

        by_family_out[family] = {
            "planners": [
                {"planner": p, "mean_path_length": pl, "mean_semantic_damage": dmg}
                for p, pl, dmg in points
            ],
            "dominated": dominated,
            "frontier":  frontier,
        }

    # Area under path_length vs. semantic_damage curve per planner
    # Points are one per family; sort by path_length before trapz
    area_under_curve: dict[str, float | None] = {}
    for planner, fam_stats in planner_family_stats.items():
        if len(fam_stats) < 2:
            area_under_curve[planner] = None
            continue
        sorted_pts = sorted(fam_stats.values(), key=lambda t: t[0])  # sort by path_length
        path_lengths = np.array([t[0] for t in sorted_pts], dtype=float)
        damages      = np.array([t[1] for t in sorted_pts], dtype=float)
        area_under_curve[planner] = float(np.trapz(damages, path_lengths))

    return {"by_family": by_family_out, "area_under_curve": area_under_curve}


def analyze_kl_threshold_tradeoff(
    records_by_threshold: dict[float, list[RolloutRecord]],
    replan_penalty: float = 0.1,
) -> dict:
    by_threshold: dict[float, dict] = {}

    for threshold, records in records_by_threshold.items():
        if not records:
            by_threshold[threshold] = {"n": 0}
            continue

        # Mean replan count
        mean_replan = compute_replan_count(records)

        # Planning time from the flat pool of all replan event durations
        all_durations = [e.duration_ms for r in records for e in r.replan_events]
        if all_durations:
            dur = np.array(all_durations, dtype=float)
            planning_time: dict[str, float] = {
                "mean_replan_ms": float(dur.mean()),
                "p95_replan_ms":  float(np.percentile(dur, 95)),
            }
        else:
            planning_time = {"mean_replan_ms": 0.0, "p95_replan_ms": 0.0}

        # Trap-only metrics
        trap_records = [r for r in records if r.scene_family == "semantic_trap"]
        if trap_records:
            recovery_vals = [compute_trap_recovery_rate(r) for r in trap_records]
            recovery_vals = [v for v in recovery_vals if v is not None]
            trap_recovery: float | None = (
                sum(recovery_vals) / len(recovery_vals) if recovery_vals else None
            )
            success_rate_trap: float | None = (
                sum(compute_success(r) for r in trap_records) / len(trap_records)
            )
        else:
            trap_recovery = None
            success_rate_trap = None

        efficiency_score: float | None = (
            trap_recovery - replan_penalty * mean_replan
            if trap_recovery is not None
            else None
        )

        by_threshold[threshold] = {
            "n":                   len(records),
            "mean_replan_count":   mean_replan,
            "mean_planning_time":  planning_time,
            "trap_recovery_rate":  trap_recovery,
            "success_rate_trap":   success_rate_trap,
            "efficiency_score":    efficiency_score,
        }

    # Optimal threshold: highest efficiency_score; None if no threshold has trap data
    scored = {
        t: s["efficiency_score"]
        for t, s in by_threshold.items()
        if s.get("efficiency_score") is not None
    }
    optimal_threshold: float | None = max(scored, key=scored.__getitem__) if scored else None

    return {
        "by_threshold":     by_threshold,
        "optimal_threshold": optimal_threshold,
        "replan_penalty":   replan_penalty,
    }


def analyze_semweight_tradeoff(
    records: list[RolloutRecord],
    weight_histories: dict[str, list[dict]],
) -> dict:
    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        return float("nan") if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])

    by_family: dict[str, list[RolloutRecord]] = defaultdict(list)
    for r in records:
        by_family[r.scene_family].append(r)

    # Collect all per-scene tuples across families for global stats
    all_weights:  list[float] = []
    all_damages:  list[float] = []
    all_paths:    list[float] = []
    all_ratios:   list[float] = []  # damage / path_length, nan when path == 0

    per_family: dict[str, dict] = {}

    for family, hist in weight_histories.items():
        family_records = by_family.get(family, [])
        n = min(len(family_records), len(hist))
        if n == 0:
            per_family[family] = {"n": 0}
            continue

        sw  = np.array([hist[i]["sem_weight"]                    for i in range(n)], dtype=float)
        dmg = np.array([compute_semantic_damage(family_records[i]) for i in range(n)], dtype=float)
        pl  = np.array([compute_path_length(family_records[i])     for i in range(n)], dtype=float)
        ratio = np.where(pl != 0, dmg / pl, np.nan)

        all_weights.extend(sw.tolist())
        all_damages.extend(dmg.tolist())
        all_paths.extend(pl.tolist())
        all_ratios.extend(ratio.tolist())

        per_family[family] = {
            "n": n,
            "correlations": {
                "sem_weight_vs_damage":      _pearson(sw, dmg),
                "sem_weight_vs_path_length": _pearson(sw, pl),
            },
            "mean_efficiency_ratio": float(np.nanmean(ratio)) if not np.all(np.isnan(ratio)) else float("nan"),
        }

    # Global correlations
    sw_all  = np.array(all_weights, dtype=float)
    dmg_all = np.array(all_damages, dtype=float)
    pl_all  = np.array(all_paths,   dtype=float)
    rat_all = np.array(all_ratios,  dtype=float)

    correlations = {
        "sem_weight_vs_damage":      _pearson(sw_all, dmg_all),
        "sem_weight_vs_path_length": _pearson(sw_all, pl_all),
    }

    # Quintile bins on sem_weight — 4 cut points → 5 bins
    boundaries   = [float(np.percentile(sw_all, p)) for p in range(20, 100, 20)] if len(sw_all) >= 5 else []
    quintile_idx = np.digitize(sw_all, boundaries) if boundaries else np.zeros(len(sw_all), dtype=int)

    sw_ranges: list[tuple[float, float]] = []
    if boundaries:
        prev = float(sw_all.min())
        for b in boundaries:
            sw_ranges.append((prev, b))
            prev = b
        sw_ranges.append((prev, float(sw_all.max())))
    else:
        sw_ranges = [(float(sw_all.min()), float(sw_all.max()))] if len(sw_all) else []

    bins: dict[int, list[tuple]] = {i: [] for i in range(max(5, len(sw_ranges)))}
    for i, idx in enumerate(quintile_idx):
        bins[int(idx)].append((float(sw_all[i]), float(dmg_all[i]), float(pl_all[i]), float(rat_all[i])))

    quintiles: dict[str, dict] = {}
    for i in range(len(sw_ranges)):
        label   = f"Q{i + 1}"
        entries = bins[i]
        if not entries:
            quintiles[label] = {"sem_weight_range": sw_ranges[i], "n": 0}
            continue
        sws, dmgs, pls, rats = zip(*entries)
        rat_arr = np.array(rats, dtype=float)
        n = len(entries)
        quintiles[label] = {
            "sem_weight_range":      sw_ranges[i],
            "n":                     n,
            "mean_sem_weight":       sum(sws)  / n,
            "mean_semantic_damage":  sum(dmgs) / n,
            "mean_path_length":      sum(pls)  / n,
            "mean_efficiency_ratio": float(np.nanmean(rat_arr)) if not np.all(np.isnan(rat_arr)) else float("nan"),
        }

    return {
        "n":                   len(all_weights),
        "correlations":        correlations,
        "quintile_boundaries": boundaries,
        "quintiles":           quintiles,
        "by_family":           per_family,
    }


def analyze_wr_wv_tradeoff(
    records: list[RolloutRecord],
    weight_histories: dict[str, list[dict]],
) -> dict:
    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        return float("nan") if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])

    by_family: dict[str, list[RolloutRecord]] = defaultdict(list)
    for r in records:
        by_family[r.scene_family].append(r)

    family_results: dict[str, dict] = {}
    for family, hist in weight_histories.items():
        family_records = by_family.get(family, [])
        n = min(len(family_records), len(hist))
        if n == 0:
            family_results[family] = {"n": 0}
            continue

        w_r  = np.array([hist[i]["w_r_scale"] for i in range(n)], dtype=float)
        w_v  = np.array([hist[i]["w_v_range"]  for i in range(n)], dtype=float)
        dmg  = np.array([compute_semantic_damage(family_records[i])             for i in range(n)], dtype=float)
        lbcf = np.array([compute_low_battery_contact_fraction(family_records[i]) for i in range(n)], dtype=float)

        ratios     = np.where(w_v != 0, w_r / w_v, np.nan)
        mean_ratio = float(np.nanmean(ratios)) if not np.all(np.isnan(ratios)) else float("nan")

        family_results[family] = {
            "n":                        n,
            "correlation_wr_vs_damage": _pearson(w_r, dmg),
            "correlation_wv_vs_lbcf":   _pearson(w_v, lbcf),
            "mean_wr_wv_ratio":         mean_ratio,
        }

    return {"by_family": family_results}


def compute_weight_convergence(family_weight_history: list[dict]) -> dict:
    if len(family_weight_history) < 2:
        return {}

    early_window = family_weight_history[:5]
    late_window  = family_weight_history[-5:]

    def _mean_abs_change(window: list[dict], key: str) -> float:
        diffs = [abs(window[t + 1][key] - window[t][key]) for t in range(len(window) - 1)]
        return sum(diffs) / len(diffs)

    result: dict[str, dict] = {}
    for key in family_weight_history[0]:
        early_change = _mean_abs_change(early_window, key)
        late_change  = _mean_abs_change(late_window,  key)

        if early_change == 0.0:
            ratio = 0.0 if late_change == 0.0 else float("inf")
        else:
            ratio = late_change / early_change

        result[key] = {
            "early_change": early_change,
            "late_change":  late_change,
            "ratio":        ratio,
            "converged":    ratio < 0.3,
            "oscillating":  late_change > early_change,
        }

    return result
