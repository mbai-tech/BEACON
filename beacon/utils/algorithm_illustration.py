import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Wedge, Rectangle,
    Polygon as MplPolygon, FancyArrowPatch,
)
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

C_MOVABLE     = "#f4b942"   # movable
C_NOT_MOVABLE = "#8b9db5"   # not movable
C_ROBOT   = "#264653"
C_GOAL    = "#d62828"
C_PATH    = "#1d3557"
C_SENSE   = "#4895ef"
C_PUSH    = "#f4a261"
C_AVOID   = "#4895ef"
C_RRT     = "#aaaaaa"
C_ESCAPE  = "#2a9d8f"
C_BAD     = "#e07070"
C_GHOST   = "#bbbbbb"
BG        = "#f8f9fa"
ROBOT_R   = 0.14


# ── Geometry helpers ───────────────────────────────────────────────────────────

def circle_verts(cx, cy, r, n=40):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def obs_patch(cx, cy, r, color, alpha=0.78, ghost=False, lw=1.0, ls="-"):
    verts = circle_verts(cx, cy, r)
    if ghost:
        return MplPolygon(verts, closed=True, facecolor="none",
                          edgecolor=C_GHOST, linewidth=1.2,
                          linestyle="--", alpha=0.65, zorder=2)
    return MplPolygon(verts, closed=True, facecolor=color,
                      edgecolor="#555555", linewidth=lw,
                      linestyle=ls, alpha=alpha, zorder=3)


def place_obs(ax, cx, cy, r, color, label=None, observed=True, ghost=False):
    p = obs_patch(cx, cy, r, color if observed else "#c8c8c8",
                  alpha=0.78 if observed else 0.30, ghost=ghost)
    ax.add_patch(p)
    if label and observed and not ghost:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=7, fontweight="bold", color="#333", zorder=5)


def robot(ax, cx, cy, zo=7):
    ax.add_patch(Circle((cx, cy), ROBOT_R, facecolor=C_ROBOT,
                        edgecolor="white", linewidth=1.5, zorder=zo))
    ax.text(cx, cy, "R", ha="center", va="center",
            fontsize=5.5, color="white", fontweight="bold", zorder=zo + 1)


def sense_ring(ax, cx, cy, r):
    ax.add_patch(Circle((cx, cy), r, facecolor=C_SENSE,
                        edgecolor="none", alpha=0.06, zorder=1))
    ax.add_patch(Circle((cx, cy), r, facecolor="none",
                        edgecolor=C_SENSE, linewidth=1.2,
                        linestyle="--", alpha=0.55, zorder=1))


def ax_setup(ax, title, xlim=(0, 6), ylim=(0, 6)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=7)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.grid(alpha=0.10, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0); sp.set_color("#cccccc")


# ══════════════════════════════════════════════════════════════════════════════
# Panel 1 — Sensing & Classification
# ══════════════════════════════════════════════════════════════════════════════

def panel_sensing(ax):
    ax_setup(ax, "① Sensing & Semantic Classification")

    rp   = (2.0, 2.0)
    goal = (5.3, 5.3)
    sr   = 1.55

    # Dotted line to goal
    ax.plot([rp[0], goal[0]], [rp[1], goal[1]],
            color=C_PATH, lw=0.9, ls=":", alpha=0.35, zorder=1)

    # Sensing disc
    sense_ring(ax, *rp, sr)

    # ── Unobserved obstacles (outside radius) ──
    place_obs(ax, 4.6, 4.0, 0.38, C_MOVABLE,     observed=False)
    place_obs(ax, 0.9, 4.6, 0.30, C_NOT_MOVABLE, observed=False)
    place_obs(ax, 5.1, 1.6, 0.26, C_NOT_MOVABLE, observed=False)
    ax.text(4.6,  3.52, "unobserved", fontsize=5.8, ha="center",
            color="#999", style="italic")
    ax.text(0.9,  4.22, "unobserved", fontsize=5.8, ha="center",
            color="#999", style="italic")

    # ── Observed obstacles (inside radius) ──
    place_obs(ax, 3.0, 2.6, 0.36, C_MOVABLE,     label="M")
    place_obs(ax, 2.7, 1.3, 0.26, C_NOT_MOVABLE, label="NM")
    place_obs(ax, 1.2, 3.1, 0.28, C_NOT_MOVABLE, label="NM")
    place_obs(ax, 3.3, 1.5, 0.20, C_MOVABLE,     label="M")

    # Sensing radius label
    lx, ly = rp[0] + sr * 0.68, rp[1] + sr * 0.68
    ax.text(lx + 0.08, ly + 0.08, "sensing\nradius  r",
            fontsize=6.2, color=C_SENSE, ha="center",
            alpha=0.85, style="italic")
    ax.annotate("", xy=(lx - 0.12, ly - 0.12), xytext=(lx + 0.05, ly + 0.02),
                arrowprops=dict(arrowstyle="-", color=C_SENSE,
                                lw=0.8, alpha=0.55))

    # Robot
    robot(ax, *rp)

    # Goal
    ax.scatter(*goal, s=180, color=C_GOAL, marker="*", zorder=8)
    ax.text(goal[0] + 0.12, goal[1] + 0.18, "Goal",
            fontsize=7, color=C_GOAL, fontweight="bold")

    # Class annotations with callout lines
    callouts = [
        ((3.0, 2.6), (3.65, 3.30), "movable:\npushable", C_MOVABLE),
        ((1.2, 3.1), (0.45, 3.85), "not movable:\navoid only", C_NOT_MOVABLE),
        ((2.7, 1.3), (1.85, 0.60), "not movable:\navoid only", C_NOT_MOVABLE),
    ]
    for (ox, oy), (tx, ty), lbl, col in callouts:
        ax.annotate(lbl, xy=(ox, oy), xytext=(tx, ty),
                    fontsize=6.2, color=col, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white", edgecolor=col,
                              alpha=0.88, linewidth=0.9),
                    arrowprops=dict(arrowstyle="-",
                                    color=col, lw=0.8, alpha=0.7))

    # Legend
    leg = [
        mpatches.Patch(fc=C_MOVABLE,     ec="#555", label="movable"),
        mpatches.Patch(fc=C_NOT_MOVABLE, ec="#555", label="not movable"),
        mpatches.Patch(fc="#c8c8c8",     ec="#aaa", label="unobserved"),
    ]
    ax.legend(handles=leg, loc="upper left", fontsize=6.2,
              framealpha=0.88, edgecolor="#ccc", handlelength=1.2)


# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Decision Flowchart
# ══════════════════════════════════════════════════════════════════════════════

def panel_flowchart(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")
    ax.set_title("② Per-Step Decision Logic", fontsize=9.5,
                 fontweight="bold", pad=7)

    def box(x, y, w, h, text, fc="#e8f4f8", ec="#4895ef",
            fs=7.2, bold=False, tc="#111"):
        r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.15",
                           facecolor=fc, edgecolor=ec,
                           linewidth=1.3, zorder=3)
        ax.add_patch(r)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color=tc, zorder=4, multialignment="center")

    def arrow(x1, y1, x2, y2, col="#555555", lab=None, lab_dx=0.18):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=col, lw=1.15))
        if lab:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + lab_dx, my, lab, fontsize=6.0, color=col, va="center")

    CX = 5.0  # centre x

    # ── START ──
    box(CX, 19.2, 4.8, 0.85, "robot at position  p",
        fc=C_ROBOT, ec=C_ROBOT, bold=True, tc="white")

    arrow(CX, 18.77, CX, 18.15)

    # ── Goal reached? ──
    box(CX, 17.8, 4.8, 0.65, "goal reached?",
        fc="#fff3cd", ec="#e0a800")
    ax.annotate("", xy=(9.2, 17.8), xytext=(CX + 2.4, 17.8),
                arrowprops=dict(arrowstyle="-|>", color=C_ESCAPE, lw=1.15))
    ax.text(9.25, 17.8, "✓ done", fontsize=6.5, color=C_ESCAPE,
            fontweight="bold", va="center")
    ax.text(CX + 2.4, 17.57, "yes", fontsize=6, color="#888", ha="center")

    arrow(CX, 17.47, CX, 16.85, lab="no")

    # ── Sense obstacles ──
    box(CX, 16.5, 4.8, 0.65, "sense all obstacles\nwithin radius  r",
        fc="#dceeff", ec=C_SENSE)

    arrow(CX, 16.17, CX, 15.55)

    # ── Any new? ──
    box(CX, 15.2, 4.8, 0.65, "newly observed obstacles?",
        fc="#fff3cd", ec="#e0a800")
    ax.annotate("", xy=(9.2, 15.2), xytext=(CX + 2.4, 15.2),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.0))
    box(9.5, 14.6, 1.8, 0.65, "record &\ncontinue", fc="#f0f0f0", ec="#aaa", fs=6.5)
    ax.annotate("", xy=(9.5, 14.93), xytext=(9.5, 15.20),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.0))
    ax.text(CX + 2.4, 14.97, "yes", fontsize=6, color="#888", ha="center")

    arrow(CX, 14.87, CX, 14.25, lab="no")

    # ── Obstacle within ε? ──
    box(CX, 13.9, 4.8, 0.65, "obstacle within reach  ε?",
        fc="#fff3cd", ec="#e0a800")
    # No → GOAL MODE
    ax.annotate("", xy=(0.85, 13.9), xytext=(CX - 2.4, 13.9),
                arrowprops=dict(arrowstyle="-|>", color="#2a9d8f", lw=1.15))
    ax.text(CX - 2.4, 13.67, "no", fontsize=6, color="#888", ha="center")
    box(0.5, 13.25, 1.8, 0.75, "GOAL\nMODE", fc="#d4edda", ec="#28a745",
        bold=True, fs=7.5)
    ax.annotate("", xy=(0.5, 13.62), xytext=(0.5, 13.9),
                arrowprops=dict(arrowstyle="-|>", color="#2a9d8f", lw=1.15))
    ax.text(0.5, 12.85, "move directly\ntoward goal",
            fontsize=5.8, color="#555", ha="center", style="italic")

    arrow(CX, 13.57, CX, 12.95, lab="yes")

    # ── Classify ──
    box(CX, 12.6, 4.8, 0.65, "classify: movable / not movable",
        fc="#dceeff", ec=C_SENSE)

    arrow(CX, 12.27, CX, 11.65)

    # ── Push optimal? ──
    box(CX, 11.3, 4.8, 0.65,
        r"movable & $P_{safe}\!\geq\!\theta$ & $J_{push}\!<\!J_{avoid}$?",
        fc="#fff3cd", ec="#e0a800")
    # Yes → PUSH MODE
    ax.annotate("", xy=(9.2, 11.3), xytext=(CX + 2.4, 11.3),
                arrowprops=dict(arrowstyle="-|>", color=C_PUSH, lw=1.15))
    ax.text(CX + 2.4, 11.07, "yes", fontsize=6, color="#888", ha="center")
    box(9.5, 10.65, 1.8, 0.75, "PUSH\nMODE", fc="#fff3e0", ec=C_PUSH,
        bold=True, fs=7.5)
    ax.annotate("", xy=(9.5, 11.02), xytext=(9.5, 11.3),
                arrowprops=dict(arrowstyle="-|>", color=C_PUSH, lw=1.15))
    ax.text(9.5, 10.25, "push obstacle\ntoward goal",
            fontsize=5.8, color="#555", ha="center", style="italic")

    arrow(CX, 10.97, CX, 10.35, lab="no")

    # ── BOUNDARY MODE ──
    C_BOUNDARY = "#7b4fa5"
    box(CX, 10.0, 4.8, 0.75, "BOUNDARY MODE  (Bug1)",
        fc="#ede7f6", ec=C_BOUNDARY, bold=True, fs=7.5)

    # Exit sub-note
    ax.text(CX, 9.58,
            "sweep 36 dirs · exit when direct step to goal is free",
            fontsize=5.6, color="#555", ha="center", style="italic")

    # Bounce detection side note
    ax.annotate("", xy=(0.85, 10.0), xytext=(CX - 2.4, 10.0),
                arrowprops=dict(arrowstyle="-|>", color=C_BOUNDARY, lw=1.0,
                                linestyle="dashed"))
    ax.text(CX - 2.4, 9.77, "bounce\n≥ 3×", fontsize=6, color=C_BOUNDARY,
            ha="center")
    box(0.5, 9.3, 1.8, 0.65, "bounce\noverride:\npush obs",
        fc="#f3e5f5", ec=C_BOUNDARY, fs=6.0)
    ax.annotate("", xy=(0.5, 9.62), xytext=(0.5, 9.95),
                arrowprops=dict(arrowstyle="-|>", color=C_BOUNDARY, lw=1.0))

    arrow(CX, 9.62, CX, 9.00, lab="circuit\ncomplete", lab_dx=0.22)

    # ── Stalled / trapped? ──
    box(CX, 8.65, 4.8, 0.65, "stalled or boundary trapped?",
        fc="#fff3cd", ec="#e0a800")
    ax.annotate("", xy=(9.2, 8.65), xytext=(CX + 2.4, 8.65),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.0))
    ax.text(9.25, 8.65, "no →\nnext step", fontsize=6, color="#888", va="center")
    ax.text(CX + 2.4, 8.42, "no", fontsize=6, color="#888", ha="center")

    arrow(CX, 8.32, CX, 7.70, lab="yes")

    # ── STALL RECOVERY ──
    box(CX, 7.35, 4.8, 0.65, "STALL RECOVERY",
        fc="#ffe0e0", ec=C_NOT_MOVABLE, bold=True, fs=7.5)
    ax.text(CX, 6.97,
            "push override  →  RRT escape  →  backtrack  →  local escape",
            fontsize=5.6, color="#555", ha="center", style="italic")

    # Sub-bullets for stall recovery
    stall_steps = [
        ("1.", "push any movable obstacle toward goal"),
        ("2.", "RRT-sample through sensed free space"),
        ("3.", "deep backtrack along executed path"),
        ("4.", "random-walk local escape"),
    ]
    for i, (num, txt) in enumerate(stall_steps):
        y = 6.55 - i * 0.42
        ax.text(CX - 2.3, y, num, fontsize=5.8, color=C_NOT_MOVABLE,
                fontweight="bold", va="center")
        ax.text(CX - 2.0, y, txt, fontsize=5.8, color="#444", va="center")


# ══════════════════════════════════════════════════════════════════════════════
# Panel 3 — Push Mode
# ══════════════════════════════════════════════════════════════════════════════

def panel_push(ax):
    ax_setup(ax, "③ Push Mode — Corridor Clearance & Chain Reaction")

    rp   = (1.2, 3.0)
    goal = (5.6, 3.0)

    # Dotted goal line
    ax.plot([rp[0], goal[0]], [rp[1], goal[1]],
            color=C_PATH, lw=0.8, ls=":", alpha=0.30, zorder=1)
    ax.scatter(*goal, s=180, color=C_GOAL, marker="*", zorder=8)
    ax.text(goal[0] + 0.08, goal[1] + 0.22, "Goal",
            fontsize=7, color=C_GOAL, fontweight="bold")

    # ── Obstacle A ──
    Ax, Ay, Ar = 2.9, 3.0, 0.40
    push_d = 0.90
    # Ghost (before)
    place_obs(ax, Ax, Ay, Ar, C_MOVABLE, ghost=True)
    ax.text(Ax, Ay + Ar + 0.20, "before\npush",
            fontsize=6, ha="center", color="#aaa", style="italic")
    # Solid (after)
    place_obs(ax, Ax + push_d, Ay, Ar, C_MOVABLE, label="M")
    ax.text(Ax + push_d, Ay + Ar + 0.20, "after\npush",
            fontsize=6, ha="center", color=C_MOVABLE, fontweight="bold")

    # Push arrow on obstacle
    ax.annotate(
        "", xy=(Ax + push_d - Ar - 0.06, Ay), xytext=(Ax + Ar + 0.06, Ay),
        arrowprops=dict(arrowstyle="-|>", color=C_PUSH, lw=2.2),
    )
    ax.text((2 * Ax + push_d) / 2, Ay + 0.12,
            f"push Δ = {push_d:.2f} m",
            fontsize=6.8, ha="center", color=C_PUSH, fontweight="bold")

    # Corridor shading
    cw = Ax - Ar - rp[0] - ROBOT_R - 0.06   # width before push
    half = ROBOT_R + 0.14
    ax.add_patch(Rectangle((rp[0] + ROBOT_R + 0.04, Ay - half),
                            cw, 2 * half,
                            facecolor=C_ESCAPE, alpha=0.13, zorder=1))
    ax.text(rp[0] + ROBOT_R + cw / 2 + 0.04, Ay - half - 0.25,
            "clearance\ncorridor",
            fontsize=6, ha="center", color=C_ESCAPE, style="italic")

    # Robot: original + moved
    robot(ax, *rp)
    robot_new = (Ax - Ar - ROBOT_R - 0.06, rp[1])
    robot(ax, *robot_new)
    ax.annotate(
        "", xy=(robot_new[0] - ROBOT_R - 0.02, rp[1]),
        xytext=(rp[0] + ROBOT_R + 0.02, rp[1]),
        arrowprops=dict(arrowstyle="-|>", color=C_ROBOT, lw=1.4,
                        linestyle="dashed"),
    )

    # ── Obstacle B — chain reaction ──
    Bx, By, Br = Ax + push_d + Ar + 0.32 + 0.30, 3.30, 0.28
    place_obs(ax, Bx, By, Br, C_MOVABLE, ghost=True)
    place_obs(ax, Bx + 0.42, By, Br, C_MOVABLE, label="M")
    ax.annotate(
        "", xy=(Bx + 0.42 - Br - 0.04, By), xytext=(Bx + Br + 0.04, By),
        arrowprops=dict(arrowstyle="-|>", color=C_PUSH, lw=1.3, alpha=0.70),
    )
    ax.text(Bx + 0.21, By - Br - 0.22, "chain push",
            fontsize=6, ha="center", color="#aaa", style="italic")

    # ── Non-pushable obstacle — avoided ──
    place_obs(ax, 2.5, 1.55, 0.30, C_NOT_MOVABLE, label="NM")
    ax.annotate("avoid", xy=(2.5, 1.55), xytext=(1.7, 1.10),
                fontsize=6.2, color=C_NOT_MOVABLE,
                arrowprops=dict(arrowstyle="-", color=C_NOT_MOVABLE,
                                lw=0.8, alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=C_NOT_MOVABLE, alpha=0.85, linewidth=0.9))

    # Cost formula
    ax.text(3.0, 5.65,
            r"$J_{push}=0.6\,t + 0.5\,e + 1.2\,r - 1.4\,\Delta c$",
            fontsize=7.5, ha="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#ddd", alpha=0.90))
    ax.text(3.0, 5.22,
            r"$t$ = time,  $e$ = effort,  $r$ = risk,  $\Delta c$ = corridor gain",
            fontsize=6.0, ha="center", color="#555")

    # Legend
    leg = [
        mpatches.Patch(fc=C_MOVABLE,     ec="#555", label="movable (pushed)"),
        mpatches.Patch(fc=C_NOT_MOVABLE, ec="#555", label="not movable (avoided)"),
        Line2D([0], [0], color=C_PUSH, lw=2, label="push direction"),
        mpatches.Patch(fc=C_ESCAPE,  alpha=0.3, ec="none", label="corridor"),
    ]
    ax.legend(handles=leg, loc="lower left", fontsize=6.2,
              framealpha=0.88, edgecolor="#ccc")


# ══════════════════════════════════════════════════════════════════════════════
# Panel 4 — Stuck Recovery
# ══════════════════════════════════════════════════════════════════════════════

def panel_escape(ax):
    ax_setup(ax, "④ Stuck Recovery — Backtrack & RRT Escape")

    stuck = np.array([2.6, 2.3])
    goal  = np.array([5.2, 5.2])

    # Goal
    ax.scatter(*goal, s=180, color=C_GOAL, marker="*", zorder=8)
    ax.text(goal[0] + 0.12, goal[1] + 0.18, "Goal",
            fontsize=7, color=C_GOAL, fontweight="bold")

    # Blocking cluster
    blockers = [
        (3.5, 2.1, 0.37, C_NOT_MOVABLE, "NM"),
        (3.7, 2.9, 0.34, C_NOT_MOVABLE, "NM"),
        (3.4, 3.7, 0.31, C_NOT_MOVABLE, "NM"),
        (4.3, 2.4, 0.27, C_MOVABLE,     "M"),
    ]
    for bx, by, br, bc, lb in blockers:
        place_obs(ax, bx, by, br, bc, label=lb)

    # ── Bad-direction memory (shaded wedge) ──
    gd  = goal - stuck
    ang = np.degrees(np.arctan2(gd[1], gd[0]))
    ax.add_patch(Wedge(stuck, 1.9, ang - 38, ang + 38,
                       facecolor=C_BAD, alpha=0.12, zorder=1,
                       edgecolor=C_BAD, linewidth=0.5, linestyle="--"))
    ax.text(stuck[0] + 1.35, stuck[1] + 1.25,
            "bad-direction\nmemory zone",
            fontsize=6, color=C_BAD, ha="center", style="italic")

    # ── Executed path leading to stuck point ──
    path_pts = [(0.9, 0.9), (1.3, 1.2), (1.8, 1.6), (2.2, 2.0), stuck]
    xs, ys = zip(*path_pts)
    ax.plot(xs, ys, color=C_PATH, lw=1.6, alpha=0.55, zorder=2, solid_capstyle="round")
    ax.scatter(*path_pts[0], s=70, color=C_ESCAPE, marker="o", zorder=6)
    ax.text(path_pts[0][0], path_pts[0][1] - 0.32, "start",
            fontsize=6.5, ha="center", color=C_ESCAPE)

    # ── Backtrack target ──
    bt = np.array(path_pts[-3])
    ax.scatter(*bt, s=90, color="#6c757d", marker="D", zorder=6)
    ax.annotate(
        "backtrack\ntarget", xy=bt, xytext=(bt[0] - 0.9, bt[1] - 0.5),
        fontsize=6, color="#6c757d", ha="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor="#6c757d", alpha=0.85, linewidth=0.8),
        arrowprops=dict(arrowstyle="-", color="#6c757d",
                        lw=0.8, alpha=0.7),
    )
    ax.annotate(
        "", xy=bt + np.array([0.04, 0.04]),
        xytext=stuck - np.array([0.04, 0.04]),
        arrowprops=dict(arrowstyle="-|>", color="#6c757d",
                        lw=1.5, linestyle="dashed"),
    )

    # ── RRT tree ──
    np.random.seed(7)
    tree = [stuck.copy()]
    edges = []
    free_targets = [
        np.array([1.7, 3.6]), np.array([1.4, 4.3]), np.array([1.1, 3.1]),
        np.array([2.1, 4.1]), np.array([0.9, 4.9]), np.array([1.9, 4.9]),
        np.array([2.7, 4.6]), np.array([1.4, 5.3]), np.array([2.4, 5.1]),
        np.array([3.1, 5.0]),
    ]
    for t in free_targets:
        dists   = [np.linalg.norm(t - n) for n in tree]
        nearest = tree[int(np.argmin(dists))]
        edges.append((nearest.copy(), t.copy()))
        tree.append(t.copy())

    for n1, n2 in edges:
        ax.plot([n1[0], n2[0]], [n1[1], n2[1]],
                color=C_RRT, lw=0.8, alpha=0.65, zorder=2)
    rrt_pts = np.array(tree[1:])
    ax.scatter(rrt_pts[:, 0], rrt_pts[:, 1],
               s=12, color=C_RRT, zorder=3, alpha=0.8)

    # ── Highlighted escape path ──
    esc = [stuck, np.array([1.7, 3.6]), np.array([2.1, 4.1]),
           np.array([2.7, 4.6]), np.array([3.1, 5.0])]
    ex, ey = zip(*esc)
    ax.plot(ex, ey, color=C_ESCAPE, lw=2.3, zorder=4, alpha=0.92,
            solid_capstyle="round")
    ax.annotate(
        "", xy=esc[-1], xytext=esc[-2],
        arrowprops=dict(arrowstyle="-|>", color=C_ESCAPE, lw=2.0),
    )
    ax.text(2.1, 5.35, "RRT escape path",
            fontsize=6.5, color=C_ESCAPE, fontweight="bold", ha="center")

    # Robot at stuck
    robot(ax, *stuck)
    ax.text(stuck[0], stuck[1] - 0.35, "stuck here",
            fontsize=6, ha="center", color="#666", style="italic")

    # Legend
    leg = [
        Line2D([0], [0], color=C_PATH,   lw=1.6, alpha=0.55, label="executed path"),
        Line2D([0], [0], color="#6c757d", lw=1.5, ls="--",   label="backtrack"),
        Line2D([0], [0], color=C_RRT,    lw=1.0,             label="RRT tree"),
        Line2D([0], [0], color=C_ESCAPE, lw=2.3,             label="escape path"),
        mpatches.Patch(fc=C_BAD, alpha=0.25, ec="none",      label="blocked directions"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=6.2,
              framealpha=0.88, edgecolor="#ccc")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fig = plt.figure(figsize=(15, 11.5))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(
        "BEACON — How the Algorithm Reasons at Each Step",
        fontsize=14, fontweight="bold", y=0.995,
    )

    gs = fig.add_gridspec(
        2, 2, hspace=0.32, wspace=0.22,
        left=0.03, right=0.98, top=0.955, bottom=0.025,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    panel_sensing(ax1)
    panel_flowchart(ax2)
    panel_push(ax3)
    panel_escape(ax4)

    out_dir  = Path(__file__).resolve().parent.parent / "environment" / "data" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "algorithm_illustration.png"

    plt.savefig(str(out_path), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    plt.show()
    plt.close(fig)
