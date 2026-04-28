"""
BEACON three-condition benchmark.

Conditions
----------
A  Fixed default PlannerConfig for all episodes.
B  VLM updates all 16 PlannerConfig weights after each scene.
C  VLM updates only battery-coupled weights (w_r_scale, w_v_floor, w_v_range);
   all other weights stay at their per-family defaults.

Hypothesis (rover planning + kinetic energy budget literature)
--------------------------------------------------------------
Condition C should reduce low_battery_contact_fraction relative to A even
without full weight tuning, because the VLM learns to raise w_v_range in
scenes where the robot reaches low battery states — making high-speed contacts
at depleted battery increasingly costly, thereby slowing the robot before they
occur.

Usage
-----
python benchmark.py --scenes 375 [--families sparse cluttered ...]
                    [--save data/benchmark] [--no-show]
"""
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

from beacon.environment.scene_generator_shapely import generate_scene as _gen_scene
from beacon.planning.beacon_planner import PlannerConfig, run_beacon
from beacon.planning.vlm_updater import VLMWeightUpdater
from beacon.utils.analysis import SceneRecord


# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_FAMILIES = ["sparse", "cluttered", "collision_required", "collision_shortcut"]

_ENV_MAP = {
    "sparse":             "sparse_clutter",
    "cluttered":          "dense_clutter",
    "collision_required": "narrow_passage",
    "collision_shortcut": "semantic_trap",
}

_COND_LABELS = {
    "A": "A — fixed defaults",
    "B": "B — VLM (all weights)",
    "C": "C — VLM (battery terms)",
}
_COND_COLORS = {"A": "#264653", "B": "#2a9d8f", "C": "#e76f51"}


# ── Scene loading ─────────────────────────────────────────────────────────────

def _load_scene(family: str, scene_idx: int) -> dict:
    mapped = _ENV_MAP.get(family, family)
    scene  = _gen_scene(mapped, seed=scene_idx)
    scene["family"]    = family
    scene["scene_idx"] = scene_idx
    return scene


# ── Per-record metric helpers ─────────────────────────────────────────────────

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


# ── Statistical helpers ───────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _config_as_json_dict(cfg: PlannerConfig) -> dict:
    return {k: float(v) for k, v in dataclasses.asdict(cfg).items()}


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


# ── Condition runners ─────────────────────────────────────────────────────────

def _run_fixed(
    scenes_by_family: dict,
    run_kw:           dict,
    n_workers:        int = 8,
) -> list:
    """Condition A — shared default PlannerConfig, all episodes in parallel."""
    config     = PlannerConfig()
    flat       = [(fam, s) for fam, ss in scenes_by_family.items() for s in ss]
    total      = len(flat)
    done_count = [0]
    done_lock  = threading.Lock()
    results: list = [None] * total

    def _one(idx_pair):
        idx, (fam, scene) = idx_pair
        result = run_beacon(scene, config=config, **run_kw)
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
                print(f"  A: {n}/{total}")
        return idx, rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        for idx, rec in ex.map(_one, enumerate(flat)):
            results[idx] = rec

    return results


def _run_vlm(
    scenes_by_family: dict,
    updater:          VLMWeightUpdater,
    run_kw:           dict,
    battery_only:     bool       = False,
    label:            str        = "B",
    save_dir                     = None,
) -> tuple[list, dict]:
    """
    Conditions B/C — VLM-updated config, sequential within each family so
    history accumulates correctly.  Families are run one after another.
    History resets at the start of each new family.  After the VLM returns
    its final update for each family the resulting config is frozen into
    family_configs and never modified again.  family_configs is serialized
    to <save_dir>/family_configs_<label>.json on completion.
    """
    records:        list                           = []
    history:        list[tuple[PlannerConfig, object]] = []
    config:         PlannerConfig                  = PlannerConfig()
    current_family: str | None                     = None
    fam_counts:     dict[str, int]                 = {f: 0 for f in scenes_by_family}
    fam_totals:     dict[str, int]                 = {f: len(ss) for f, ss in scenes_by_family.items()}
    family_configs: dict[str, PlannerConfig]       = {}

    def is_last_scene_in_family(fam_: str) -> bool:
        return fam_counts[fam_] == fam_totals[fam_] - 1

    flat = [(fam, scene) for fam, scenes in scenes_by_family.items() for scene in scenes]

    for fam, scene in flat:
        if fam != current_family:
            history.clear()
            prev_frozen    = family_configs.get(current_family) if current_family is not None else None
            config         = _init_config_for_family(prev_frozen)
            current_family = fam

        result  = run_beacon(scene, config=config, **run_kw)
        summary = result.scene_summary
        records.append(SceneRecord(
            scene_idx=scene["scene_idx"],
            family=fam,
            config=config,
            summary=summary,
        ))
        new_config = updater.update(
            config, summary, history,
            family=fam,
            scene_idx_in_family=fam_counts[fam],
            scenes_remaining=fam_totals[fam] - fam_counts[fam] - 1,
            battery_only=battery_only,
        )
        history.append((config, summary))
        config = new_config

        if is_last_scene_in_family(fam):
            family_configs[fam] = copy.deepcopy(config)

        fam_counts[fam] += 1
        n = fam_counts[fam]
        if n % 50 == 0 or n == fam_totals[fam]:
            print(f"  {label} [{fam}]: {n}/{fam_totals[fam]}")

    if save_dir:
        p = Path(save_dir) / f"family_configs_{label.lower()}.json"
        payload = {fam: _config_as_json_dict(cfg) for fam, cfg in family_configs.items()}
        p.write_text(json.dumps(payload, indent=2))
        print(f"  {label} family configs → {p}")

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
    """Warn if the VLM failed to differentiate configs across families."""
    if len(family_configs) < 2:
        return
    for param in _DIFFERENTIATION_PARAMS:
        vals  = [getattr(cfg, param) for cfg in family_configs.values()]
        spread = max(vals) - min(vals)
        if spread <= tol:
            print(
                f"  WARNING [{label}] '{param}' is identical across all families "
                f"(max\u2212min = {spread:.4f} \u2264 {tol}) \u2014 "
                "family-scoping may not be working; VLM is not differentiating."
            )


# ── Report ────────────────────────────────────────────────────────────────────

def _print_report(sa: dict, sb: dict, sc: dict, n_total: int) -> None:
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

    print(_row(_COND_LABELS["A"], sa))
    print(_row(_COND_LABELS["B"], sb))
    print(_row(_COND_LABELS["C"], sc))
    print(div)

    # Primary hypothesis: low_battery_contact_fraction  C < A
    t, p, method = _ttest_less(sc["_lb_raw"], sa["_lb_raw"])
    verdict = "SUPPORTED" if (not math.isnan(p) and p < 0.05) else "not supported"
    print(f"\n  H₁  low_battery_contact_fraction  C < A")
    print(f"      {method}:  t = {t:+.3f}   p = {p:.4g}   [{verdict} at α = 0.05]")

    # Secondary: semantic damage  B < A
    t2, p2, _ = _ttest_less(sb["_sem_raw"], sa["_sem_raw"])
    v2 = "SUPPORTED" if (not math.isnan(p2) and p2 < 0.05) else "not supported"
    print(f"\n  H₂  semantic_damage  B < A  (full VLM drives damage reduction)")
    print(f"      {method}:  t = {t2:+.3f}   p = {p2:.4g}   [{v2} at α = 0.05]")

    # Effect size (Cohen's d) for H₁
    lb_a = np.array(sa["_lb_raw"], float)
    lb_c = np.array(sc["_lb_raw"], float)
    pooled_sd = math.sqrt((lb_a.var(ddof=1) + lb_c.var(ddof=1)) / 2)
    if pooled_sd > 1e-12:
        d = (lb_a.mean() - lb_c.mean()) / pooled_sd
        print(f"\n  Cohen's d (A − C on LB frac): {d:.3f}"
              f"  ({'large' if abs(d) >= 0.8 else 'medium' if abs(d) >= 0.5 else 'small'})")

    print(div + "\n")


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_comparison(
    sa: dict,
    sb: dict,
    sc: dict,
    save_dir=None,
    show: bool = True,
) -> None:
    cond_keys  = ["A", "B", "C"]
    stats_list = [sa, sb, sc]
    colors     = [_COND_COLORS[k] for k in cond_keys]
    xs         = np.arange(3)

    metrics = [
        ("success_rate",    "Success rate",                     True),
        ("semantic_damage", "Mean semantic damage",             False),
        ("lb_frac",         "Mean low-battery contact frac.",   False),
        ("mean_delta_e",    "Mean ΔE at contact  (δ·v²)",       False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fig.suptitle("BEACON three-condition benchmark", fontsize=12)

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
        ax.set_xticklabels(["A", "B", "C"], fontsize=9)
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

    for cond, raw in [("A", sa["_lb_raw"]), ("C", sc["_lb_raw"])]:
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
    for cond, raw in [("A", sa["_lb_raw"]), ("C", sc["_lb_raw"])]:
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
    vlm_model:           str   = "qwen-plus",
    max_steps:           int   = 500,
    step_size:           float = 0.04,
    sensing_range:       float = 0.35,
    n_workers_a:         int   = 8,
    save_dir                   = None,
    show_plots:          bool  = True,
) -> dict:
    """
    Run the three-condition BEACON benchmark.

    Parameters
    ----------
    n_scenes_per_family : scenes per family; 375 × 4 families = 1 500 total
    families            : subset of _DEFAULT_FAMILIES, or None for all four
    vlm_model           : Qwen model name passed to VLMWeightUpdater
    n_workers_a         : thread-pool size for the parallelised condition-A run
    save_dir            : if given, save plots there
    show_plots          : call plt.show() (False in headless contexts)

    Returns
    -------
    {
        "A":     list[SceneRecord],
        "B":     list[SceneRecord],
        "C":     list[SceneRecord],
        "stats": {"A": dict, "B": dict, "C": dict},
    }
    """
    families = families or _DEFAULT_FAMILIES
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

    # Separate updater instances so conditions B and C accumulate independent histories
    updater_b = VLMWeightUpdater(model=vlm_model)
    updater_c = VLMWeightUpdater(model=vlm_model)

    print(f"── Condition A — fixed defaults ({n_total} episodes, {n_workers_a} workers) ──")
    records_a = _run_fixed(scenes_by_family, run_kw, n_workers=n_workers_a)

    print(f"\n── Condition B — VLM all weights ({n_total} episodes) ──")
    records_b, family_configs_b = _run_vlm(scenes_by_family, updater_b, run_kw,
                                            battery_only=False, label="B",
                                            save_dir=save_dir)

    print(f"\n── Condition C — VLM battery terms only ({n_total} episodes) ──")
    records_c, family_configs_c = _run_vlm(scenes_by_family, updater_c, run_kw,
                                            battery_only=True,  label="C",
                                            save_dir=save_dir)

    for _lbl, _fc in [("B", family_configs_b), ("C", family_configs_c)]:
        _print_family_configs_table(_fc, _lbl)
        _check_family_differentiation(_fc, _lbl)

    stats_a = _condition_stats(records_a)
    stats_b = _condition_stats(records_b)
    stats_c = _condition_stats(records_c)

    _print_report(stats_a, stats_b, stats_c, n_total)
    _plot_comparison(stats_a, stats_b, stats_c,
                     save_dir=save_dir, show=show_plots)

    return {
        "A":              records_a,
        "B":              records_b,
        "C":              records_c,
        "stats":          {"A": stats_a, "B": stats_b, "C": stats_c},
        "family_configs": {"B": family_configs_b, "C": family_configs_c},
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BEACON three-condition benchmark (A=fixed, B=VLM-full, C=VLM-battery)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenes", type=int, default=375,
        help="Scenes per family (375 × 4 = 1500 total)",
    )
    parser.add_argument(
        "--families", nargs="*", default=None, choices=_DEFAULT_FAMILIES,
    )
    parser.add_argument("--model",   type=str,   default="qwen-plus")
    parser.add_argument("--steps",   type=int,   default=500)
    parser.add_argument("--step",    type=float, default=0.04)
    parser.add_argument("--sense",   type=float, default=0.35)
    parser.add_argument("--workers", type=int,   default=8,
                        help="Thread-pool size for condition A")
    parser.add_argument("--save",    type=str,   default=None,
                        metavar="DIR", help="Directory for saved plots")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip plt.show() — use with --save in headless mode")
    args = parser.parse_args()

    run_benchmark(
        n_scenes_per_family = args.scenes,
        families            = args.families,
        vlm_model           = args.model,
        max_steps           = args.steps,
        step_size           = args.step,
        sensing_range       = args.sense,
        n_workers_a         = args.workers,
        save_dir            = args.save,
        show_plots          = not args.no_show,
    )
