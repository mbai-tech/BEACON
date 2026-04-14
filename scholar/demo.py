"""
demo.py — Real-time SCHOLAR simulation across all 5 scene families.

Each family runs in a background thread. Frames are intercepted as they
are produced and displayed live — no waiting for the full simulation to finish.

Usage
-----
    python scholar/demo.py
    python scholar/demo.py --scene 3
    python scholar/demo.py --family dense_clutter narrow_passage
    python scholar/demo.py --steps 400
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon

import NewProject.planner as _planner_module
from NewProject.planner import run_online_surp_push
from environment.visualize_v2 import CLASS_COLORS as DISPLAY_COLORS
from environment.scene_generator import load_scene

FAMILIES = ["sparse", "cluttered", "collision_required", "collision_shortcut"]


# ── Real-time frame interception ───────────────────────────────────────────────

_original_snapshot = _planner_module.snapshot_frame
_frame_queues: dict = {}   # thread_id → list of frames
_frame_lock = threading.Lock()


def _patched_snapshot(position, scene, message):
    frame = _original_snapshot(position, scene, message)
    tid = threading.get_ident()
    with _frame_lock:
        if tid in _frame_queues:
            _frame_queues[tid].append(frame)
    return frame

_planner_module.snapshot_frame = _patched_snapshot


# ── Per-family simulation thread ───────────────────────────────────────────────

class SimThread(threading.Thread):
    def __init__(self, family, scene_idx, max_steps):
        super().__init__(daemon=True)
        self.family     = family
        self.scene_idx  = scene_idx
        self.max_steps  = max_steps
        self.frames     = []   # filled in real time by patched snapshot_frame
        self.path       = []
        self.scene      = None
        self.initial_scene = None
        self.success    = None
        self.done       = False

    def run(self):
        with _frame_lock:
            _frame_queues[threading.get_ident()] = self.frames

        raw_scene = load_scene(self.scene_idx, family=self.family,
                               fragility="mixed", seed=self.scene_idx)
        result = run_online_surp_push(raw_scene, max_steps=self.max_steps)

        self.path          = result.path
        self.scene         = result.scene
        self.initial_scene = result.initial_scene
        self.success       = result.success
        self.done          = True

        with _frame_lock:
            _frame_queues.pop(threading.get_ident(), None)


# ── Build figure and animate ───────────────────────────────────────────────────

def run_realtime(families, scene_idx, max_steps, save=False, speedup=3):
    # Start one sim thread per family
    threads = [SimThread(fam, scene_idx, max_steps) for fam in families]
    for t in threads:
        t.start()

    # Wait just long enough for threads to load their scenes and emit first frame
    import time
    while not any(len(t.frames) > 0 for t in threads):
        time.sleep(0.05)

    n   = len(threads)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"SCHOLAR push simulation — scene {scene_idx:03d}  (live)", fontsize=10)

    panels = []
    for ax, t in zip(axes, threads):
        ax.set_facecolor("#f8f9fa")
        ax.set_aspect("equal")
        ax.set_title(t.family.replace("_", " "), fontsize=8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.grid(alpha=0.15)

        # We don't know workspace yet for all threads — use default and update later
        ax.set_xlim(0, 6); ax.set_ylim(0, 6)

        patches      = []     # obstacle patches, built once first frame arrives
        ghost_built  = [False]
        path_line,   = ax.plot([], [], color="#1d3557", linewidth=1.4, zorder=3)
        robot_dot    = ax.scatter([], [], s=80, color="#264653", marker="o", zorder=5)
        status_text  = ax.text(0.02, 0.98, "waiting...", transform=ax.transAxes,
                               va="top", ha="left", fontsize=6,
                               bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        frame_cursor = [0]

        panels.append((t, ax, patches, ghost_built, path_line, robot_dot,
                       status_text, frame_cursor))

    def update(_tick):
        artists = []
        for (t, ax, patches, ghost_built, path_line,
             robot_dot, status_text, cursor) in panels:

            frames = t.frames   # live list

            if not frames:
                artists += [path_line, robot_dot, status_text]
                continue

            # First frame: set up workspace, ghost outlines, and obstacle patches
            if not ghost_built[0]:
                frame0 = frames[0]
                xmin, xmax, ymin, ymax = 0, 6, 0, 6
                if t.scene:
                    xmin, xmax, ymin, ymax = t.scene["workspace"]
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                # Dashed ghost outlines — initial obstacle positions (static)
                for obs in frame0.obstacles:
                    ax.add_patch(MplPolygon(
                        obs["vertices"], closed=True,
                        fill=False, edgecolor="#999999",
                        linewidth=0.8, linestyle="--", alpha=0.45, zorder=1,
                    ))

                # Live obstacle patches (updated each frame)
                for obs in frame0.obstacles:
                    p = MplPolygon(
                        obs["vertices"], closed=True,
                        facecolor=DISPLAY_COLORS.get(obs.get("class_true", obs.get("true_class", "movable")), "lightblue"),
                        edgecolor="#666666", linewidth=0.9, alpha=0.5, zorder=2,
                    )
                    ax.add_patch(p)
                    patches.append(p)

                # Start / goal markers
                if t.scene:
                    ax.scatter(*t.scene["start"][:2], s=80, color="#2a9d8f",
                               marker="o", zorder=6, label="start")
                    ax.scatter(*t.scene["goal"][:2], s=110, color="#d62828",
                               marker="*", zorder=6, label="goal")
                    ax.legend(fontsize=6, loc="upper right")
                ghost_built[0] = True

            # Advance to the latest available frame
            idx   = min(len(frames) - 1, len(frames) - 1)
            frame = frames[idx]
            cursor[0] = idx

            # Update obstacle patches
            for patch, obs in zip(patches, frame.obstacles):
                patch.set_xy(obs["vertices"])
                patch.set_facecolor(DISPLAY_COLORS.get(obs.get("class_true", obs.get("true_class", "movable")), "lightblue"))
                patch.set_alpha(0.92 if obs["observed"] else 0.42)
                patch.set_edgecolor("#111111" if obs["observed"] else "#666666")
                patch.set_linewidth(2.0 if obs["observed"] else 0.9)
                artists.append(patch)

            # Path trail — built from frame positions (available in real time)
            positions = [(f.position[0], f.position[1]) for f in frames[:idx + 1]]
            arr = np.array(positions)
            path_line.set_data(arr[:, 0], arr[:, 1])
            robot_dot.set_offsets([[frame.position[0], frame.position[1]]])

            # Status
            if t.done:
                label = "✓ SUCCESS" if t.success else "✗ FAILED"
            else:
                label = f"step {idx} / {max_steps}"
            status_text.set_text(f"{label}\n{frame.message[:55]}")

            artists += [path_line, robot_dot, status_text]

        return artists

    anim = FuncAnimation(fig, update, interval=60, blit=False, cache_frame_data=False)
    fig._anim = anim
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close(fig)

    # Wait for any still-running threads
    for t in threads:
        t.join()

    if save:
        _save_video(threads, scene_idx)


def _save_video(threads, scene_idx, speedup=3, fps=30):
    import shutil

    total_frames = max(len(t.frames) for t in threads)
    step         = max(1, speedup)
    frame_idxs   = list(range(0, total_frames, step))
    n            = len(threads)

    print(f"Saving {len(frame_idxs)} frames at {fps} fps ({speedup}× speed)...")

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"SCHOLAR — scene {scene_idx:03d}", fontsize=10)

    # Build static elements
    panels = []
    for ax, t in zip(axes, threads):
        frame0 = t.frames[0]
        xmin, xmax, ymin, ymax = t.scene["workspace"] if t.scene else (0,6,0,6)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal"); ax.set_facecolor("#f8f9fa")
        ax.set_title(t.family.replace("_", " "), fontsize=8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        for obs in frame0.obstacles:
            ax.add_patch(MplPolygon(obs["vertices"], closed=True, fill=False,
                                    edgecolor="#999999", linewidth=0.8,
                                    linestyle="--", alpha=0.45, zorder=1))
        patches = []
        for obs in frame0.obstacles:
            p = MplPolygon(obs["vertices"], closed=True,
                           facecolor=DISPLAY_COLORS.get(
                               obs.get("class_true", obs.get("true_class","movable")),
                               "lightblue"),
                           edgecolor="#666666", linewidth=0.9, alpha=0.5, zorder=2)
            ax.add_patch(p); patches.append(p)

        if t.scene:
            ax.scatter(*t.scene["start"][:2], s=80, color="#2a9d8f", marker="o", zorder=6)
            ax.scatter(*t.scene["goal"][:2],  s=110, color="#d62828", marker="*", zorder=6)

        path_line,  = ax.plot([], [], color="#1d3557", linewidth=1.4, zorder=3)
        robot_dot   = ax.scatter([], [], s=80, color="#264653", zorder=5)
        status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                              ha="left", fontsize=6,
                              bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        panels.append((t, patches, path_line, robot_dot, status_text))

    def update(fi):
        real_idx = frame_idxs[fi]
        for t, patches, path_line, robot_dot, status_text in panels:
            idx   = min(real_idx, len(t.frames) - 1)
            frame = t.frames[idx]
            for patch, obs in zip(patches, frame.obstacles):
                patch.set_xy(obs["vertices"])
                patch.set_facecolor(DISPLAY_COLORS.get(
                    obs.get("class_true", obs.get("true_class","movable")), "lightblue"))
                patch.set_alpha(0.92 if obs["observed"] else 0.42)
                patch.set_edgecolor("#111111" if obs["observed"] else "#666666")
            positions = [(f.position[0], f.position[1]) for f in t.frames[:idx+1]]
            arr = np.array(positions)
            path_line.set_data(arr[:, 0], arr[:, 1])
            robot_dot.set_offsets([[frame.position[0], frame.position[1]]])
            done = idx >= len(t.frames) - 1
            status_text.set_text(("✓ SUCCESS" if t.success else "✗ FAILED") if done
                                 else f"step {idx}")

    save_anim = FuncAnimation(fig, update, frames=len(frame_idxs),
                              interval=1000 // fps, blit=False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir  = Path(__file__).resolve().parent / "environment" / "data" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"simulation_scene{scene_idx:03d}.mp4"

    if shutil.which("ffmpeg"):
        from matplotlib.animation import FFMpegWriter
        save_anim.save(str(out_path), writer=FFMpegWriter(fps=fps, bitrate=1800))
    else:
        out_path = out_path.with_suffix(".gif")
        from matplotlib.animation import PillowWriter
        save_anim.save(str(out_path), writer=PillowWriter(fps=fps))

    plt.close(fig)
    print(f"Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",  type=int, default=0)
    parser.add_argument("--family", nargs="*", default=None, choices=FAMILIES)
    parser.add_argument("--steps",  type=int, default=300)
    parser.add_argument("--save",   action="store_true",
                        help="Save video after simulation ends")
    parser.add_argument("--speedup", type=int, default=3)
    args = parser.parse_args()

    families = args.family or FAMILIES
    print(f"Starting real-time simulation — scene {args.scene}, "
          f"families: {', '.join(families)}")
    run_realtime(families, scene_idx=args.scene, max_steps=args.steps,
                 save=args.save, speedup=args.speedup)
