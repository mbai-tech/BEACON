"""Post-run adaptation diagnostics for BEACON VLM weight tuning.

Usage
-----
from beacon.utils.analysis import SceneRecord, plot_adaptation_diagnostics

records = [SceneRecord(scene_idx=i, family=fam, config=cfg, summary=summ), ...]
stats   = plot_adaptation_diagnostics(records, save_dir=Path("data/plots"))

The function returns a dict with Pearson correlations and a list of
oscillating weight names. It also prints a text summary to stdout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from beacon.core.models import SceneSummary
from beacon.planning.beacon_planner import PlannerConfig


# ── Data type ─────────────────────────────────────────────────────────────────

@dataclass
class SceneRecord:
    """Config used and result produced for one scene, used for post-run analysis."""
    scene_idx: int
    family:    str
    config:    PlannerConfig
    summary:   SceneSummary


# ── Constants ─────────────────────────────────────────────────────────────────

_WEIGHT_FIELDS = [f.name for f in fields(PlannerConfig)]

_FAMILY_COLORS = {
    "sparse":             "#2a9d8f",
    "sparse_clutter":     "#2a9d8f",
    "cluttered":          "#e76f51",
    "dense_clutter":      "#e76f51",
    "semantic_trap":      "#9b5de5",
    "collision_shortcut": "#9b5de5",
    "perturbed":          "#f4a261",
    "perturbed_scenes":   "#f4a261",
}
_DEFAULT_COLOR = "#264653"

# Oscillation: flag a weight if any window of _OSC_WINDOW consecutive per-family
# values contains >= _OSC_MIN_CHANGES sign reversals in the diff sequence AND
# the value range within the window exceeds _OSC_MIN_RANGE (avoids noise flags).
_OSC_WINDOW      = 5
_OSC_MIN_CHANGES = 3
_OSC_MIN_RANGE   = 0.02


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_oscillation(values: list) -> bool:
    """True if any 5-scene window has >=3 sign reversals without converging."""
    if len(values) < _OSC_WINDOW:
        return False
    arr = np.array(values, dtype=float)
    for start in range(len(arr) - _OSC_WINDOW + 1):
        w = arr[start : start + _OSC_WINDOW]
        if w.max() - w.min() < _OSC_MIN_RANGE:
            continue
        diffs  = np.diff(w)
        signs  = np.sign(diffs)
        signs  = signs[signs != 0]
        if len(signs) < 2:
            continue
        n_flips = int(np.sum(np.diff(signs) != 0))
        if n_flips >= _OSC_MIN_CHANGES:
            return True
    return False


def _pearsonr(x: list, y: list):
    """Pearson r and two-tailed p-value; falls back to manual calculation."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    try:
        from scipy import stats as _ss
        r, p = _ss.pearsonr(x, y)
        return float(r), float(p)
    except ImportError:
        pass
    # Manual fallback
    xm, ym = x - x.mean(), y - y.mean()
    denom = math.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if denom < 1e-12:
        return float("nan"), float("nan")
    r = float((xm * ym).sum() / denom)
    # t-statistic → two-tailed p via normal approximation (adequate for n > 10)
    t = r * math.sqrt(n - 2) / max(math.sqrt(1 - r ** 2), 1e-12)
    try:
        from scipy.special import stdtr as _stdtr
        p = 2.0 * float(_stdtr(n - 2, -abs(t)))
    except ImportError:
        p = float("nan")
    return r, p


def _canonical_family(family: str) -> str:
    """Return the short display name for a family."""
    _MAP = {
        "sparse_clutter":     "sparse",
        "dense_clutter":      "cluttered",
        "collision_shortcut": "semantic_trap",
        "perturbed_scenes":   "perturbed",
    }
    return _MAP.get(family, family)


# ── Main entry point ──────────────────────────────────────────────────────────

def plot_adaptation_diagnostics(
    records:  list,
    save_dir: Optional[Path] = None,
    show:     bool           = True,
) -> dict:
    """
    Produce two diagnostic plots and print a text report.

    Parameters
    ----------
    records   : list[SceneRecord]
    save_dir  : if given, save PNGs there (weight_trajectories.png,
                wv_speed_scatter.png)
    show      : call plt.show() (set False in headless/batch contexts)

    Returns
    -------
    {
        "pearson_r_wv_speed":       float,   # w_v_effective vs. mean_speed_at_contact
        "pearson_p_wv_speed":       float,
        "pearson_r_delta_E_damage": float,   # delta_E_coeff vs. semantic_damage
        "pearson_p_delta_E_damage": float,
        "oscillating_weights":      list[str],  # "param (family)" tags
        "oscillating_params":       list[str],  # unique param names only
    }
    """
    if not records:
        return {}

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Group by family, sorted by scene_idx
    by_family: dict = {}
    for r in records:
        by_family.setdefault(r.family, []).append(r)
    for fam in by_family:
        by_family[fam].sort(key=lambda r: r.scene_idx)

    # ── Plot 1: weight trajectories ───────────────────────────────────────────
    n_w   = len(_WEIGHT_FIELDS)
    ncols = 4
    nrows = math.ceil(n_w / ncols)

    fig1, axes1 = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.6, nrows * 2.8),
        constrained_layout=True,
    )
    fig1.suptitle("BEACON — weight adaptation across scenes", fontsize=11, y=1.01)
    flat = axes1.flatten() if hasattr(axes1, "flatten") else [axes1]

    for ax in flat[n_w:]:
        ax.set_visible(False)

    for i, wname in enumerate(_WEIGHT_FIELDS):
        ax = flat[i]
        ax.set_title(wname, fontsize=8, pad=3)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18)

        for fam, recs in by_family.items():
            color = _FAMILY_COLORS.get(fam, _DEFAULT_COLOR)
            xs    = [r.scene_idx for r in recs]
            ys    = [getattr(r.config, wname) for r in recs]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.1,
                    color=color, alpha=0.85)

        ax.set_xlabel("scene", fontsize=6)

    # Shared legend
    legend_handles = [
        mlines.Line2D([], [], color=_FAMILY_COLORS.get(fam, _DEFAULT_COLOR),
                      marker="o", markersize=4, linewidth=1.2,
                      label=_canonical_family(fam))
        for fam in by_family
    ]
    fig1.legend(handles=legend_handles,
                loc="lower center", ncol=len(by_family),
                fontsize=8, bbox_to_anchor=(0.5, -0.03),
                framealpha=0.9)

    if save_dir:
        p1 = Path(save_dir) / "weight_trajectories.png"
        fig1.savefig(p1, dpi=150, bbox_inches="tight")
        print(f"Saved → {p1}")
    if show:
        plt.show()
    plt.close(fig1)

    # ── Plot 2: w_v_effective vs. mean_speed_at_contact ───────────────────────
    fig2, ax2 = plt.subplots(figsize=(6.2, 5.0))
    ax2.set_xlabel(r"$w_v^{\rm eff} = w_{v,\rm floor} + w_{v,\rm range}\cdot(1-b)$"
                   "  at contact events", fontsize=9)
    ax2.set_ylabel("mean speed at contact  (m/step)", fontsize=9)
    ax2.set_title(
        "Velocity-penalty effectiveness\n"
        r"Hypothesis ($H_-$): higher $w_v^{\rm eff}$ $\Rightarrow$ lower contact speed",
        fontsize=9,
    )
    ax2.grid(alpha=0.18)

    all_wv:  list = []
    all_spd: list = []

    for fam, recs in by_family.items():
        color    = _FAMILY_COLORS.get(fam, _DEFAULT_COLOR)
        xs_fam, ys_fam = [], []

        for r in recs:
            contact_log = [e for e in r.summary.battery_contact_log
                           if e["event"] == "contact"]
            if not contact_log:
                continue
            wv_eff = float(np.mean([e["w_v"] for e in contact_log]))
            spd    = r.summary.mean_speed_at_contact
            xs_fam.append(wv_eff)
            ys_fam.append(spd)

        if xs_fam:
            ax2.scatter(xs_fam, ys_fam, color=color,
                        label=_canonical_family(fam),
                        s=48, alpha=0.82, edgecolors="white", linewidths=0.5)
            all_wv.extend(xs_fam)
            all_spd.extend(ys_fam)

    r_wv, p_wv = _pearsonr(all_wv, all_spd)

    if not math.isnan(r_wv) and len(all_wv) >= 3:
        xs_line = np.linspace(min(all_wv), max(all_wv), 120)
        m, b_fit = np.polyfit(all_wv, all_spd, 1)
        ax2.plot(xs_line, m * xs_line + b_fit,
                 color="#333333", linewidth=1.2, linestyle="--", alpha=0.65)

    sig_label = " *" if (not math.isnan(p_wv) and p_wv < 0.05) else ""
    r_str = f"r = {r_wv:.3f}{sig_label}  (p = {p_wv:.3g})" \
            if not math.isnan(r_wv) else "r = n/a (no contact data)"
    ax2.annotate(r_str, xy=(0.97, 0.97), xycoords="axes fraction",
                 ha="right", va="top", fontsize=8,
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    h2, l2 = ax2.get_legend_handles_labels()
    if h2:
        ax2.legend(h2, l2, fontsize=8, framealpha=0.9)

    if save_dir:
        p2 = Path(save_dir) / "wv_speed_scatter.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        print(f"Saved → {p2}")
    if show:
        plt.show()
    plt.close(fig2)

    # ── Pearson r: delta_E_coeff vs. total_semantic_damage ───────────────────
    de_vals  = [r.config.delta_E_coeff          for r in records]
    dmg_vals = [r.summary.total_semantic_damage  for r in records]
    r_de, p_de = _pearsonr(de_vals, dmg_vals)

    print("\n── VLM adaptation diagnostics ───────────────────────────────────────────")
    print(f"  w_v_eff vs. contact speed   r = {r_wv:.3f}  p = {p_wv:.3g}"
          f"   ({'✓ negative — correct' if r_wv < 0 else '✗ not negative'})"
          if not math.isnan(r_wv) else "  w_v_eff vs. contact speed: insufficient contact data")

    if not math.isnan(r_de):
        verdict = "✓ positive — VLM correctly raises contact cost" \
                  if r_de > 0 else "✗ not positive — VLM not responding to damage"
        print(f"  delta_E_coeff vs. damage    r = {r_de:.3f}  p = {p_de:.3g}   ({verdict})")
    else:
        print("  delta_E_coeff vs. damage: insufficient data")

    # ── Oscillation detection ─────────────────────────────────────────────────
    oscillating_tags: list = []    # "param (family)" strings
    oscillating_params: set = set()

    for wname in _WEIGHT_FIELDS:
        for fam, recs in by_family.items():
            vals = [getattr(r.config, wname) for r in recs]
            if _detect_oscillation(vals):
                oscillating_tags.append(f"{wname} ({_canonical_family(fam)})")
                oscillating_params.add(wname)

    if oscillating_tags:
        print("\n  ⚠ Oscillating weights (consider tightening 20% change cap):")
        for tag in sorted(oscillating_tags):
            print(f"    • {tag}")
    else:
        print("\n  ✓ No oscillating weights detected.")

    print("─" * 72)

    return {
        "pearson_r_wv_speed":       r_wv,
        "pearson_p_wv_speed":       p_wv,
        "pearson_r_delta_E_damage": r_de,
        "pearson_p_delta_E_damage": p_de,
        "oscillating_weights":      sorted(oscillating_tags),
        "oscillating_params":       sorted(oscillating_params),
    }
