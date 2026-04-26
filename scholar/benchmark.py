from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
import threading
import concurrent.futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from enviornment.scene_generator import generate_scene as _gen_scene
from planning.scholar import PlannerConfig, run_scholar
from planning.vlm_updater import VLMWeightUpdater
from utils.analysis import SceneRecord

_DEFAULT_FAMILIES = ["sparse", "cluttered", "collision_required", "collision_shortcut"]

_ENV_MAP = {
    "sparse":             "sparse_clutter",
    "cluttered":          "dense_clutter",
    "collision_required": "narrow_passage",
    "collision_shortcut": "semantic_trap",
}

_COND_LABELS = {
    "A_legacy": "A0 — legacy SURP baseline",
    "A": "A — fixed defaults",
    "B": "B — LLM (all weights)",
    "C": "C — LLM (battery terms)",
}
_COND_COLORS = {"A_legacy": "#6c757d", "A": "#264653", "B": "#2a9d8f", "C": "#e76f51"}
_ALL_CONDITIONS = ("A_legacy", "A", "B", "C")
_UPDATE_ACCEPT_ABS_TOL = 1e-4
_UPDATE_ACCEPT_REL_TOL = 1e-3

def _load_scene(family: str, scene_idx: int) -> dict:
    mapped = _ENV_MAP.get(family, family)
    scene  = _gen_scene(mapped, seed=scene_idx)
    scene["family"]    = family
    scene["scene_idx"] = scene_idx
    return scene

def _mean_delta_e(record: SceneRecord) -> float:
    """Mean dissipated kinetic energy per contact event: δ·v² averaged over contacts."""
    contacts = [e for e in record.summary.battery_contact_log
                if e["event"] == "contact"]
    if not contacts:
        return 0.0
    return record.config.delta_E_coeff * float(
        np.mean([e["speed"] ** 2 for e in contacts])
    )


def _condition_stats(records: list) -> dict:
    sem = [r.summary.total_semantic_damage        for r in records]
    suc = [float(r.summary.success)               for r in records]
    lb  = [r.summary.low_battery_contact_fraction for r in records]
    de  = [_mean_delta_e(r)                        for r in records]
    return {
        "n":               len(records),
        "semantic_damage": (float(np.mean(sem)), float(np.std(sem))),
        "success_rate":    float(np.mean(suc)),
        "lb_frac":         (float(np.mean(lb)),  float(np.std(lb))),
        "mean_delta_e":    (float(np.mean(de)),  float(np.std(de))),
        # raw lists for hypothesis tests
        "_lb_raw":         lb,
        "_sem_raw":        sem,
    }

def _ttest_less(x: list, y: list) -> tuple:
    """One-tailed Welch t-test: H₁  mean(x) < mean(y).  Returns (t, p, method)."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if len(xa) < 2 or len(ya) < 2:
        return float("nan"), float("nan"), "n/a"
    try:
        from scipy import stats as _ss
        r = _ss.ttest_ind(xa, ya, equal_var=False, alternative="less")
        return float(r.statistic), float(r.pvalue), "Welch t"
    except ImportError:
        pass
    # Manual Welch t-test fallback
    nx, ny   = len(xa), len(ya)
    mx, my   = xa.mean(), ya.mean()
    sx2, sy2 = xa.var(ddof=1), ya.var(ddof=1)
    se       = math.sqrt(sx2 / nx + sy2 / ny)
    if se < 1e-12:
        return 0.0, 1.0, "Welch t (manual)"
    t  = (mx - my) / se
    df = (sx2 / nx + sy2 / ny) ** 2 / (
        (sx2 / nx) ** 2 / (nx - 1) + (sy2 / ny) ** 2 / (ny - 1)
    )
    try:
        from scipy.special import stdtr
        p = float(stdtr(df, t))          # P(T ≤ t) = one-tailed p
    except ImportError:
        p = float("nan")
    return t, p, "Welch t (manual)"


def _config_as_json_dict(cfg: PlannerConfig) -> dict:
    return {k: float(v) for k, v in dataclasses.asdict(cfg).items()}


def _score_improved(candidate_score: float, baseline_score: float) -> bool:
    tol = max(_UPDATE_ACCEPT_ABS_TOL, _UPDATE_ACCEPT_REL_TOL * abs(baseline_score))
    return candidate_score <= baseline_score + tol


def _summarize_config_delta(
    current: PlannerConfig,
    proposed: PlannerConfig,
    max_items: int = 6,
) -> str:
    cur = _config_as_json_dict(current)
    new = _config_as_json_dict(proposed)
    deltas: list[tuple[float, str]] = []
    for key, cur_val in cur.items():
        new_val = new[key]
        delta = new_val - cur_val
        if abs(delta) <= 1e-12:
            continue
        rel = abs(delta) / max(abs(cur_val), 1e-9)
        deltas.append((rel, f"{key}: {cur_val:.4f}->{new_val:.4f}"))
    deltas.sort(key=lambda item: item[0], reverse=True)
    if not deltas:
        return "no effective parameter change"
    return ", ".join(text for _, text in deltas[:max_items])


def _stats_path(save_dir) -> Path | None:
    return Path(save_dir) / "condition_stats.json" if save_dir else None


def _load_saved_stats(save_dir) -> dict:
    path = _stats_path(save_dir)
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_merged_stats(save_dir, stats_by_condition: dict) -> None:
    path = _stats_path(save_dir)
    if path is None:
        return
    path.write_text(json.dumps(stats_by_condition, indent=2))
    print(f"Saved → {path}")


def _update_trace_path(save_dir, label: str) -> Path | None:
    return Path(save_dir) / f"llm_update_trace_{label.lower()}.json" if save_dir else None


def _batch_objective(records: list[SceneRecord], battery_only: bool = False) -> float:
    if not records:
        return float("inf")
    summaries = [r.summary for r in records]
    success_penalty = 1.0 - float(np.mean([float(s.success) for s in summaries]))
    mean_semantic = float(np.mean([s.total_semantic_damage for s in summaries]))
    mean_lb_frac = float(np.mean([s.low_battery_contact_fraction for s in summaries]))
    mean_stuck = float(np.mean([s.n_stuck_events for s in summaries]))
    mean_forbidden = float(np.mean([s.forbidden_contact_rate for s in summaries]))
    if battery_only:
        return (
            4.0 * mean_lb_frac
            + 2.0 * success_penalty
            + 0.35 * mean_semantic
            + 0.25 * mean_stuck
            + 0.5 * mean_forbidden
        )
    return (
        1.5 * mean_semantic
        + 2.0 * success_penalty
        + 0.75 * mean_lb_frac
        + 0.25 * mean_stuck
        + 0.75 * mean_forbidden
    )


def _batch_metrics(records: list[SceneRecord], battery_only: bool = False) -> dict:
    if not records:
        return {
            "n": 0,
            "objective": float("inf"),
            "success_rate": 0.0,
            "semantic_damage": 0.0,
            "low_battery_contact_fraction": 0.0,
            "n_stuck_events": 0.0,
            "forbidden_contact_rate": 0.0,
            "mean_speed_at_contact": 0.0,
            "mean_final_battery": 0.0,
            "dominant_j_counts": {},
        }
    summaries = [r.summary for r in records]
    dominant_counts: dict[str, int] = {}
    for s in summaries:
        dominant_counts[s.dominant_j] = dominant_counts.get(s.dominant_j, 0) + 1
    return {
        "n": len(records),
        "objective": _batch_objective(records, battery_only),
        "success_rate": float(np.mean([float(s.success) for s in summaries])),
        "semantic_damage": float(np.mean([s.total_semantic_damage for s in summaries])),
        "low_battery_contact_fraction": float(np.mean([s.low_battery_contact_fraction for s in summaries])),
        "n_stuck_events": float(np.mean([s.n_stuck_events for s in summaries])),
        "forbidden_contact_rate": float(np.mean([s.forbidden_contact_rate for s in summaries])),
        "mean_speed_at_contact": float(np.mean([s.mean_speed_at_contact for s in summaries])),
        "mean_final_battery": float(np.mean([s.final_battery for s in summaries])),
        "dominant_j_counts": dominant_counts,
    }


def _format_batch_metrics(metrics: dict) -> str:
    return (
        f"obj={metrics['objective']:.4f}, "
        f"success={metrics['success_rate']:.1%}, "
        f"sem={metrics['semantic_damage']:.3f}, "
        f"lb_frac={metrics['low_battery_contact_fraction']:.3f}, "
        f"stuck={metrics['n_stuck_events']:.2f}, "
        f"forbidden={metrics['forbidden_contact_rate']:.3f}, "
        f"speed@contact={metrics['mean_speed_at_contact']:.3f}, "
        f"final_batt={metrics['mean_final_battery']:.3f}"
    )


def _run_scene_batch(
    scenes: list[dict],
    family: str,
    config: PlannerConfig,
    run_kw: dict,
) -> list[SceneRecord]:
    batch_records: list[SceneRecord] = []
    for scene in scenes:
        result = run_scholar(scene, config=config, **run_kw)
        batch_records.append(SceneRecord(
            scene_idx=scene["scene_idx"],
            family=family,
            config=config,
            summary=result.scene_summary,
        ))
    return batch_records


def _init_config_for_family(prev_frozen: "PlannerConfig | None") -> PlannerConfig:
    # delta_E_coeff and w_v_floor are grounded in kinetic energy budget literature
    # and should be invariant to scene structure; resource_d and resource_T reflect
    # motor/sensor cost priors that transfer across environments.  Semantic and
    # geometric weights (sem_weight, geo_weight, dir_weight, w_r_scale, w_v_range,
    # kl_threshold, f_alpha_threshold) are family-specific and must be re-learned.
    if prev_frozen is None:
        return PlannerConfig()
    defaults = PlannerConfig()
    # resource_contact is derived from the simplex constraint so the sum stays 1.
    return dataclasses.replace(
        defaults,
        delta_E_coeff    = prev_frozen.delta_E_coeff,
        w_v_floor        = prev_frozen.w_v_floor,
        resource_d       = prev_frozen.resource_d,
        resource_T       = prev_frozen.resource_T,
        resource_contact = 1.0 - prev_frozen.resource_d - prev_frozen.resource_T,
    )

def _run_fixed(
    scenes_by_family: dict,
    run_kw:           dict,
    n_workers:        int = 8,
    label:            str = "A",
    use_legacy_baseline: bool = False,
) -> list:
    """Fixed-config condition, all episodes in parallel."""
    config     = PlannerConfig()
    flat       = [(fam, s) for fam, ss in scenes_by_family.items() for s in ss]
    total      = len(flat)
    done_count = [0]
    done_lock  = threading.Lock()
    results: list = [None] * total

    def _one(idx_pair):
        idx, (fam, scene) = idx_pair
        result = run_scholar(scene, config=config, use_legacy_baseline=use_legacy_baseline, **run_kw)
        rec = SceneRecord(
            scene_idx=scene["scene_idx"],
            family=fam,
            config=config,
            summary=result.scene_summary,
        )
        with done_lock:
            done_count[0] += 1
            n = done_count[0]
            if n % 100 == 0 or n == total:
                print(f"  {label}: {n}/{total}")
        return idx, rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        for idx, rec in ex.map(_one, enumerate(flat)):
            results[idx] = rec

    return results


def _maybe_same_scene_tune(
    scene: dict,
    family: str,
    config: PlannerConfig,
    summary,
    history: list[tuple[PlannerConfig, object]],
    updater: VLMWeightUpdater,
    run_kw: dict,
    battery_only: bool,
    scenes_remaining: int,
    passes: int,
) -> PlannerConfig:
    tuned_config = config
    tuned_summary = summary

    for _ in range(passes):
        proposed = updater.update(
            tuned_config,
            tuned_summary,
            history,
            family=family,
            scene_idx_in_family=scene["scene_idx"],
            scenes_remaining=scenes_remaining,
            battery_only=battery_only,
        )
        if proposed == tuned_config:
            break

        baseline_records = _run_scene_batch([scene], family, tuned_config, run_kw)
        candidate_records = _run_scene_batch([scene], family, proposed, run_kw)
        baseline_score = _batch_objective(baseline_records, battery_only)
        candidate_score = _batch_objective(candidate_records, battery_only)
        if _score_improved(candidate_score, baseline_score):
            tuned_config = proposed
            tuned_summary = candidate_records[0].summary
        else:
            break

    return tuned_config


def _run_llm(
    scenes_by_family: dict,
    updater:          VLMWeightUpdater,
    run_kw:           dict,
    battery_only:     bool       = False,
    label:            str        = "B",
    update_batch_size: int       = 8,
    accept_batch_size: int       = 2,
    same_scene_tuning_passes: int = 0,
    save_dir                     = None,
) -> tuple[list, dict]:
    """
    Conditions B/C — batched LLM-updated config. Scenes are still evaluated on
    a shared fixed set across A/B/C, but updates are proposed from aggregated
    batches rather than single-scene noise. Each proposed update is accepted
    only if it improves a short calibration rollout.
    """
    records:        list                           = []
    history:        list[tuple[PlannerConfig, object]] = []
    family_configs: dict[str, PlannerConfig]       = {}
    update_trace:   list[dict]                     = []
    update_batch_size = max(1, int(update_batch_size))
    accept_batch_size = max(1, int(accept_batch_size))

    prev_frozen: PlannerConfig | None = None
    for fam, scenes in scenes_by_family.items():
        history.clear()
        config = _init_config_for_family(prev_frozen)
        total = len(scenes)
        idx = 0

        while idx < total:
            batch_start = idx
            batch_scenes = scenes[idx: idx + update_batch_size]
            batch_records = _run_scene_batch(batch_scenes, fam, config, run_kw)
            batch_metrics = _batch_metrics(batch_records, battery_only)
            records.extend(batch_records)
            history.extend((config, rec.summary) for rec in batch_records)
            idx += len(batch_records)
            print(
                f"  {label} [{fam}] batch {batch_start}-{idx - 1} summary: "
                f"{_format_batch_metrics(batch_metrics)}"
            )

            if same_scene_tuning_passes > 0 and batch_records:
                config = _maybe_same_scene_tune(
                    batch_scenes[-1],
                    fam,
                    config,
                    batch_records[-1].summary,
                    history,
                    updater,
                    run_kw,
                    battery_only=battery_only,
                    scenes_remaining=total - idx,
                    passes=same_scene_tuning_passes,
                )

            if idx < total:
                proposed = updater.update_family(
                    config,
                    [rec.summary for rec in batch_records],
                    fam,
                    history,
                    battery_only=battery_only,
                )

                if proposed != config:
                    calib_scenes = scenes[idx: idx + accept_batch_size]
                    if calib_scenes:
                        baseline_records = _run_scene_batch(calib_scenes, fam, config, run_kw)
                        candidate_records = _run_scene_batch(calib_scenes, fam, proposed, run_kw)
                        baseline_metrics = _batch_metrics(baseline_records, battery_only)
                        candidate_metrics = _batch_metrics(candidate_records, battery_only)
                        baseline_score = baseline_metrics["objective"]
                        candidate_score = candidate_metrics["objective"]
                        delta_summary = _summarize_config_delta(config, proposed)
                        accepted = _score_improved(candidate_score, baseline_score)
                        update_trace.append({
                            "family": fam,
                            "condition": label,
                            "battery_only": battery_only,
                            "scene_range": [batch_start, idx - 1],
                            "batch_metrics": batch_metrics,
                            "accept_scenes": [scene["scene_idx"] for scene in calib_scenes],
                            "baseline_metrics": baseline_metrics,
                            "candidate_metrics": candidate_metrics,
                            "accepted": accepted,
                            "config_delta": delta_summary,
                        })
                        if _score_improved(candidate_score, baseline_score):
                            config = proposed
                            print(
                                f"  {label} [{fam}] accepted update after scenes {max(0, idx - len(batch_records))}-{idx - 1}: "
                                f"{baseline_score:.4f} → {candidate_score:.4f} "
                                f"({delta_summary})"
                            )
                            print(
                                f"    baseline accept metrics: {_format_batch_metrics(baseline_metrics)}"
                            )
                            print(
                                f"    candidate accept metrics: {_format_batch_metrics(candidate_metrics)}"
                            )
                        else:
                            print(
                                f"  {label} [{fam}] rejected update after scenes {max(0, idx - len(batch_records))}-{idx - 1}: "
                                f"{baseline_score:.4f} → {candidate_score:.4f} "
                                f"({delta_summary})"
                            )
                            print(
                                f"    baseline accept metrics: {_format_batch_metrics(baseline_metrics)}"
                            )
                            print(
                                f"    candidate accept metrics: {_format_batch_metrics(candidate_metrics)}"
                            )
                    else:
                        config = proposed

            if idx % 50 == 0 or idx == total:
                print(f"  {label} [{fam}]: {idx}/{total}")

        family_configs[fam] = copy.deepcopy(config)
        prev_frozen = family_configs[fam]

    if save_dir:
        p = Path(save_dir) / f"family_configs_{label.lower()}.json"
        payload = {fam: _config_as_json_dict(cfg) for fam, cfg in family_configs.items()}
        p.write_text(json.dumps(payload, indent=2))
        print(f"  {label} family configs → {p}")
        trace_path = _update_trace_path(save_dir, label)
        if trace_path is not None:
            trace_path.write_text(json.dumps(update_trace, indent=2))
            print(f"  {label} update trace → {trace_path}")

    return records, family_configs


# ── Family-config diagnostics ─────────────────────────────────────────────────

_DIFFERENTIATION_PARAMS = ("sem_weight", "kl_threshold", "w_r_scale")


def _print_family_configs_table(family_configs: dict, label: str) -> None:
    """Print a side-by-side table: rows = parameters, columns = families."""
    if not family_configs:
        return
    families = list(family_configs.keys())
    params   = list(dataclasses.asdict(next(iter(family_configs.values()))).keys())

    row_w = max(len(p) for p in params) + 2
    col_w = max(12, max(len(f) for f in families) + 2)
    div   = "─" * (row_w + col_w * len(families))

    print(f"\n── Condition {label} — optimized family configs ──")
    print(div)
    print(f"{'Parameter':<{row_w}}" + "".join(f"{f:>{col_w}}" for f in families))
    print(div)
    for p in params:
        vals = [getattr(family_configs[f], p) for f in families]
        print(f"{p:<{row_w}}" + "".join(f"{v:>{col_w}.4f}" for v in vals))
    print(div)


def _check_family_differentiation(
    family_configs: dict,
    label:          str,
    tol:            float = 0.01,
) -> None:
    """Warn if the LLM failed to differentiate configs across families."""
    if len(family_configs) < 2:
        return
    for param in _DIFFERENTIATION_PARAMS:
        vals  = [getattr(cfg, param) for cfg in family_configs.values()]
        spread = max(vals) - min(vals)
        if spread <= tol:
            print(
                f"  WARNING [{label}] '{param}' is identical across all families "
                f"(max\u2212min = {spread:.4f} \u2264 {tol}) \u2014 "
                "family-scoping may not be working; LLM is not differentiating."
            )


# ── Report ────────────────────────────────────────────────────────────────────

def _print_report(stats_by_condition: dict, n_total: int) -> None:
    title = f"── BEACON Benchmark — {n_total} episodes"
    div   = "─" * 80

    print(f"\n{title}\n{div}")
    hdr = (f"{'Condition':<28}  {'Sem. Damage':>16}  {'Success':>7}"
           f"  {'LB Contact Frac':>17}  {'ΔE':>14}")
    print(hdr)
    print(div)

    def _row(label, s):
        sd_m, sd_s = s["semantic_damage"]
        lb_m, lb_s = s["lb_frac"]
        de_m, de_s = s["mean_delta_e"]
        return (
            f"{label:<28}  "
            f"{sd_m:>7.4f} ± {sd_s:<6.4f}  "
            f"{s['success_rate']:>6.1%}  "
            f"{lb_m:>7.4f} ± {lb_s:<6.4f}  "
            f"{de_m:>6.4f} ± {de_s:<5.4f}"
        )

    for cond in [c for c in _ALL_CONDITIONS if c in stats_by_condition]:
        print(_row(_COND_LABELS[cond], stats_by_condition[cond]))
    print(div)

    if "A" in stats_by_condition and "C" in stats_by_condition:
        t, p, method = _ttest_less(stats_by_condition["C"]["_lb_raw"], stats_by_condition["A"]["_lb_raw"])
        verdict = "SUPPORTED" if (not math.isnan(p) and p < 0.05) else "not supported"
        print(f"\n  H₁  low_battery_contact_fraction  C < A")
        print(f"      {method}:  t = {t:+.3f}   p = {p:.4g}   [{verdict} at α = 0.05]")

        lb_a = np.array(stats_by_condition["A"]["_lb_raw"], float)
        lb_c = np.array(stats_by_condition["C"]["_lb_raw"], float)
        pooled_sd = math.sqrt((lb_a.var(ddof=1) + lb_c.var(ddof=1)) / 2)
        if pooled_sd > 1e-12:
            d = (lb_a.mean() - lb_c.mean()) / pooled_sd
            print(f"\n  Cohen's d (A − C on LB frac): {d:.3f}"
                  f"  ({'large' if abs(d) >= 0.8 else 'medium' if abs(d) >= 0.5 else 'small'})")

    if "A" in stats_by_condition and "B" in stats_by_condition:
        t2, p2, method2 = _ttest_less(stats_by_condition["B"]["_sem_raw"], stats_by_condition["A"]["_sem_raw"])
        v2 = "SUPPORTED" if (not math.isnan(p2) and p2 < 0.05) else "not supported"
        print(f"\n  H₂  semantic_damage  B < A  (full LLM drives damage reduction)")
        print(f"      {method2}:  t = {t2:+.3f}   p = {p2:.4g}   [{v2} at α = 0.05]")

    if "A_legacy" in stats_by_condition and "A" in stats_by_condition:
        legacy_sr = stats_by_condition["A_legacy"]["success_rate"]
        new_sr = stats_by_condition["A"]["success_rate"]
        print(f"\n  Baseline shift  legacy A0 → new A: {legacy_sr:.1%} → {new_sr:.1%}")

    print(div + "\n")


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_comparison(
    stats_by_condition: dict,
    save_dir=None,
    show: bool = True,
) -> None:
    cond_keys  = [c for c in _ALL_CONDITIONS if c in stats_by_condition]
    stats_list = [stats_by_condition[c] for c in cond_keys]
    colors     = [_COND_COLORS[k] for k in cond_keys]
    xs         = np.arange(len(cond_keys))

    metrics = [
        ("success_rate",    "Success rate",                     True),
        ("semantic_damage", "Mean semantic damage",             False),
        ("lb_frac",         "Mean low-battery contact frac.",   False),
        ("mean_delta_e",    "Mean ΔE at contact  (δ·v²)",       False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fig.suptitle("BEACON benchmark comparison", fontsize=12)

    for ax, (key, title, is_rate) in zip(axes.flatten(), metrics):
        if is_rate:
            vals = [s[key]    for s in stats_list]
            errs = None
        else:
            vals = [s[key][0] for s in stats_list]
            errs = [s[key][1] for s in stats_list]

        bars = ax.bar(xs, vals, color=colors, width=0.55, alpha=0.88,
                      yerr=errs, capsize=5,
                      error_kw={"linewidth": 1.2, "ecolor": "#333333"})
        ax.set_title(title, fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(cond_keys, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.22)
        if is_rate:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(vals) * 0.01),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    if save_dir:
        p = Path(save_dir) / "benchmark_comparison.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved → {p}")
    if show:
        plt.show()
    plt.close(fig)

    # Distribution: low_battery_contact_fraction  A vs C
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
    ax2.set_title(
        "low_battery_contact_fraction  —  A vs C\n"
        "Hypothesis: C shifts distribution left (lower fraction)",
        fontsize=9,
    )
    ax2.set_xlabel("low_battery_contact_fraction", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.grid(alpha=0.2)

    if not ("A" in stats_by_condition and "C" in stats_by_condition):
        plt.close(fig2)
        return

    for cond, raw in [("A", stats_by_condition["A"]["_lb_raw"]), ("C", stats_by_condition["C"]["_lb_raw"])]:
        arr = np.array(raw, float)
        col = _COND_COLORS[cond]
        lbl = _COND_LABELS[cond]
        try:
            from scipy.stats import gaussian_kde
            if arr.std() > 1e-9:
                kde = gaussian_kde(arr, bw_method="scott")
                xs  = np.linspace(max(0.0, arr.min() - 0.05),
                                  min(1.0, arr.max() + 0.05), 300)
                ax2.plot(xs, kde(xs), color=col, linewidth=2.0, label=lbl)
                ax2.fill_between(xs, kde(xs), alpha=0.12, color=col)
        except ImportError:
            ax2.hist(arr, bins=25, density=True, alpha=0.45,
                     color=col, label=lbl, edgecolor="white", linewidth=0.4)

    # Annotate means
    for cond, raw in [("A", stats_by_condition["A"]["_lb_raw"]), ("C", stats_by_condition["C"]["_lb_raw"])]:
        m = float(np.mean(raw))
        ax2.axvline(m, color=_COND_COLORS[cond],
                    linewidth=1.4, linestyle="--", alpha=0.8)
        ax2.text(m, ax2.get_ylim()[1] * 0.92, f"μ={m:.3f}",
                 color=_COND_COLORS[cond], fontsize=7,
                 ha="left" if cond == "A" else "right")

    ax2.legend(fontsize=8, framealpha=0.9)

    if save_dir:
        p2 = Path(save_dir) / "lb_frac_distribution.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        print(f"Saved → {p2}")
    if show:
        plt.show()
    plt.close(fig2)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_benchmark(
    n_scenes_per_family: int   = 375,
    families:            list  = None,
    llm_model:           str   = "Qwen/Qwen2.5-7B-Instruct",
    max_steps:           int   = 500,
    step_size:           float = 0.04,
    sensing_range:       float = 0.35,
    n_workers_a:         int   = 8,
    llm_batch_size:      int   = 64,
    update_batch_size:   int   = 8,
    accept_batch_size:   int   = 2,
    same_scene_tuning_passes: int = 0,
    conditions:          list  = None,
    save_dir                   = None,
    show_plots:          bool  = True,
) -> dict:
    families = families or _DEFAULT_FAMILIES
    conditions = list(dict.fromkeys(conditions or list(_ALL_CONDITIONS)))
    invalid_conditions = [cond for cond in conditions if cond not in _ALL_CONDITIONS]
    if invalid_conditions:
        raise ValueError(f"Invalid conditions requested: {invalid_conditions}")
    n_total  = n_scenes_per_family * len(families)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Generate the shared scene set (same episodes for all three conditions)
    print(f"Generating {n_scenes_per_family} scenes × {len(families)} families"
          f" ({n_total} total)...")
    scenes_by_family = {
        fam: [_load_scene(fam, i) for i in range(n_scenes_per_family)]
        for fam in families
    }
    print("Scene generation done.\n")

    run_kw = {
        "max_steps":     max_steps,
        "step_size":     step_size,
        "sensing_range": sensing_range,
    }
    records_by_condition: dict[str, list] = {}
    family_configs_by_condition: dict[str, dict] = {}

    if "A_legacy" in conditions:
        print(f"── Condition A0 — legacy SURP baseline ({n_total} episodes, {n_workers_a} workers) ──")
        records_by_condition["A_legacy"] = _run_fixed(
            scenes_by_family, run_kw, n_workers=n_workers_a, label="A0", use_legacy_baseline=True
        )

    if "A" in conditions:
        print(f"── Condition A — fixed defaults ({n_total} episodes, {n_workers_a} workers) ──")
        records_by_condition["A"] = _run_fixed(
            scenes_by_family, run_kw, n_workers=n_workers_a, label="A", use_legacy_baseline=False
        )

    if any(cond in conditions for cond in ("B", "C")):
        # Only initialize vLLM if we actually need LLM-backed conditions.
        updater_b = None
        updater_c = None
        if "B" in conditions:
            updater_b = VLMWeightUpdater(model=llm_model, max_num_seqs=llm_batch_size)
        if "C" in conditions:
            updater_c = VLMWeightUpdater(model=llm_model, max_num_seqs=llm_batch_size)

        if "B" in conditions:
            print(f"\n── Condition B — LLM all weights ({n_total} episodes) ──")
            records_b, family_configs_b = _run_llm(scenes_by_family, updater_b, run_kw,
                                                    battery_only=False, label="B",
                                                    update_batch_size=update_batch_size,
                                                    accept_batch_size=accept_batch_size,
                                                    same_scene_tuning_passes=same_scene_tuning_passes,
                                                    save_dir=save_dir)
            records_by_condition["B"] = records_b
            family_configs_by_condition["B"] = family_configs_b

        if "C" in conditions:
            print(f"\n── Condition C — LLM battery terms only ({n_total} episodes) ──")
            records_c, family_configs_c = _run_llm(scenes_by_family, updater_c, run_kw,
                                                    battery_only=True,  label="C",
                                                    update_batch_size=update_batch_size,
                                                    accept_batch_size=accept_batch_size,
                                                    same_scene_tuning_passes=same_scene_tuning_passes,
                                                    save_dir=save_dir)
            records_by_condition["C"] = records_c
            family_configs_by_condition["C"] = family_configs_c

    for _lbl, _fc in family_configs_by_condition.items():
        _print_family_configs_table(_fc, _lbl)
        _check_family_differentiation(_fc, _lbl)

    current_stats = {cond: _condition_stats(records) for cond, records in records_by_condition.items()}
    merged_stats = _load_saved_stats(save_dir)
    merged_stats.update(current_stats)
    _save_merged_stats(save_dir, merged_stats)

    required_for_full = ("A", "B", "C")
    if all(cond in merged_stats for cond in required_for_full):
        _print_report(merged_stats, n_total)
        _plot_comparison(merged_stats, save_dir=save_dir, show=show_plots)
    else:
        missing = [cond for cond in required_for_full if cond not in merged_stats]
        print(f"\nPartial run complete. Saved stats for {sorted(current_stats)}.")
        print(f"Still missing conditions for full comparison: {missing}")

    return {
        "records":        records_by_condition,
        "stats":          merged_stats,
        "family_configs": family_configs_by_condition,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BEACON three-condition benchmark (A=fixed, B=LLM-full, C=LLM-battery)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenes", type=int, default=375,
        help="Scenes per family (375 × 4 = 1500 total)",
    )
    parser.add_argument(
        "--families", nargs="*", default=None, choices=_DEFAULT_FAMILIES,
    )
    parser.add_argument("--model",   type=str,   default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID for vllm")
    parser.add_argument("--steps",   type=int,   default=500)
    parser.add_argument("--step",    type=float, default=0.04)
    parser.add_argument("--sense",   type=float, default=0.35)
    parser.add_argument("--workers", type=int,   default=8,
                        help="Thread-pool size for condition A")
    parser.add_argument("--llm-batch-size", type=int, default=64,
                        help="vLLM engine max_num_seqs batch capacity")
    parser.add_argument("--update-batch-scenes", type=int, default=8,
                        help="Number of official scenes to aggregate before each LLM update")
    parser.add_argument("--accept-batch-scenes", type=int, default=2,
                        help="Number of future scenes used to accept/reject a proposed update")
    parser.add_argument("--same-scene-tuning-passes", type=int, default=0,
                        help="Optional number of same-scene local adaptation passes after each batch")
    parser.add_argument("--conditions", nargs="+", default=list(_ALL_CONDITIONS),
                        choices=_ALL_CONDITIONS,
                        help="Subset of conditions to run. Use 'A' on cheap compute, then 'B C' on A100 with the same --save dir.")
    parser.add_argument("--save",    type=str,   default=None,
                        metavar="DIR", help="Directory for saved plots")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip plt.show() — use with --save in headless mode")
    args = parser.parse_args()

    run_benchmark(
        n_scenes_per_family = args.scenes,
        families            = args.families,
        llm_model           = args.model,
        max_steps           = args.steps,
        step_size           = args.step,
        sensing_range       = args.sense,
        n_workers_a         = args.workers,
        llm_batch_size      = args.llm_batch_size,
        update_batch_size   = args.update_batch_scenes,
        accept_batch_size   = args.accept_batch_scenes,
        same_scene_tuning_passes = args.same_scene_tuning_passes,
        conditions          = args.conditions,
        save_dir            = args.save,
        show_plots          = not args.no_show,
    )
