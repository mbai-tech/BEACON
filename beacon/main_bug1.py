import sys
import json
import time
import threading
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon

import beacon.core.bug_algorithm as _bug_module
from beacon.core.bug_algorithm import run_bug
from beacon.environment.visualize_v2 import CLASS_COLORS as DISPLAY_COLORS

FAMILIES   = ["sparse", "cluttered", "collision_required", "collision_shortcut"]
SCENES_DIR = Path(__file__).resolve().parent / "environment" / "data" / "scenes"


# ── Scene loading ─────────────────────────────────────────────────────────────

def load_scene(scene_idx: int, family: str) -> dict:
    path = SCENES_DIR / family / f"scene_{scene_idx:03d}.json"
    with open(path) as f:
        return json.load(f)


# ── Real-time frame interception ──────────────────────────────────────────────

_original_snapshot = _bug_module.snapshot_frame
_frame_queues      = {}
_frame_lock        = threading.Lock()


def _patched_snapshot(position, scene, message):
    frame = _original_snapshot(position, scene, message)
    tid   = threading.get_ident()
    with _frame_lock:
        if tid in _frame_queues:
            _frame_queues[tid].append(frame)
    return frame

_bug_module.snapshot_frame = _patched_snapshot


# ── Per-family simulation thread ──────────────────────────────────────────────

class SimThread(threading.Thread):
    def __init__(self, family, scene_idx, max_steps, step_size=0.07, sensing_range=0.55):
        super().__init__(daemon=True)
        self.family        = family
        self.scene_idx     = scene_idx
        self.max_steps     = max_steps
        self.step_size     = step_size
        self.sensing_range = sensing_range
        self.frames        = []
        self.path          = []
        self.scene         = None
        self.initial_scene = None
        self.workspace     = None
        self.success       = None
        self.done          = False

    def run(self):
        with _frame_lock:
            _frame_queues[threading.get_ident()] = self.frames

        raw_scene = load_scene(self.scene_idx, self.family)
        self.workspace = raw_scene.get("workspace", [0, 6, 0, 6])
        result    = run_bug(
            raw_scene,
            max_steps     = self.max_steps,
            step_size     = self.step_size,
            sensing_range = self.sensing_range,
        )

        self.path          = result.path
        self.scene         = result.scene
        self.initial_scene = result.initial_scene
        self.success       = result.success
        self.done          = True

        with _frame_lock:
            _frame_queues.pop(threading.get_ident(), None)


# ── Live visualisation ────────────────────────────────────────────────────────

def run_realtime(families, scene_idx, max_steps,
                 step_size=0.07, sensing_range=0.55,
                 save=False, speedup=3):

    threads = [
        SimThread(fam, scene_idx, max_steps,
                  step_size=step_size, sensing_range=sensing_range)
        for fam in families
    ]
    for t in threads:
        t.start()

    while not any(len(t.frames) > 0 for t in threads):
        time.sleep(0.05)

    n = len(threads)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Bug1 — scene {scene_idx:03d}  (live)", fontsize=10)

    panels = []
    for ax, t in zip(axes, threads):
        ax.set_facecolor("#f8f9fa")
        ax.set_aspect("equal")
        ax.set_title(t.family.replace("_", " "), fontsize=8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.grid(alpha=0.15)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)

        patches     = []
        ghost_built = [False]
        path_line,  = ax.plot([], [], color="#1d3557", linewidth=1.4, zorder=3)
        robot_dot   = ax.scatter([], [], s=80, color="#264653", marker="o", zorder=5)
        status_text = ax.text(
            0.02, 0.98, "waiting...", transform=ax.transAxes,
            va="top", ha="left", fontsize=6,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        frame_cursor = [0]
        panels.append((t, ax, patches, ghost_built,
                       path_line, robot_dot, status_text, frame_cursor))

    def update(_tick):
        artists = []
        for (t, ax, patches, ghost_built,
             path_line, robot_dot, status_text, cursor) in panels:

            frames = t.frames

            if not frames:
                artists += [path_line, robot_dot, status_text]
                continue

            if not ghost_built[0]:
                frame0 = frames[0]
                xmin, xmax, ymin, ymax = t.workspace or [0, 6, 0, 6]
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                for obs in frame0.obstacles:
                    ax.add_patch(MplPolygon(
                        obs["vertices"], closed=True,
                        fill=False, edgecolor="#999999",
                        linewidth=0.8, linestyle="--", alpha=0.45, zorder=1,
                    ))

                for obs in frame0.obstacles:
                    cls = obs.get("true_class", obs.get("class_true", "movable"))
                    p = MplPolygon(
                        obs["vertices"], closed=True,
                        facecolor=DISPLAY_COLORS.get(cls, "lightblue"),
                        edgecolor="#666666", linewidth=0.9, alpha=0.5, zorder=2,
                    )
                    ax.add_patch(p)
                    patches.append(p)

                if t.scene:
                    ax.scatter(*t.scene["start"][:2], s=80,  color="#2a9d8f",
                               marker="o", zorder=6, label="start")
                    ax.scatter(*t.scene["goal"][:2],  s=110, color="#d62828",
                               marker="*", zorder=6, label="goal")
                    ax.legend(fontsize=6, loc="upper right")

                ghost_built[0] = True

            idx   = len(frames) - 1
            frame = frames[idx]
            cursor[0] = idx

            for patch, obs in zip(patches, frame.obstacles):
                patch.set_xy(obs["vertices"])
                cls = obs.get("true_class", obs.get("class_true", "movable"))
                patch.set_facecolor(DISPLAY_COLORS.get(cls, "lightblue"))
                patch.set_alpha(0.92 if obs["observed"] else 0.42)
                patch.set_edgecolor("#111111" if obs["observed"] else "#666666")
                patch.set_linewidth(2.0 if obs["observed"] else 0.9)
                artists.append(patch)

            positions = [(f.position[0], f.position[1]) for f in frames[:idx + 1]]
            arr = np.array(positions)
            path_line.set_data(arr[:, 0], arr[:, 1])
            robot_dot.set_offsets([[frame.position[0], frame.position[1]]])

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

    for t in threads:
        t.join()

    _save_log(threads, scene_idx)

    if save:
        _save_video(threads, scene_idx, speedup=speedup)


# ── Batch mode ────────────────────────────────────────────────────────────────

def run_batch(families, scene_indices, max_steps,
              step_size=0.07, sensing_range=0.55,
              save=False, speedup=3, max_workers=8):
    import concurrent.futures

    def _run_scene(scene_idx):
        threads = [
            SimThread(fam, scene_idx, max_steps,
                      step_size=step_size, sensing_range=sensing_range)
            for fam in families
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        _save_log(threads, scene_idx)
        if save:
            _save_video(threads, scene_idx, speedup=speedup)
        return scene_idx, {t.family: t.success for t in threads}

    print(f"Running {len(scene_indices)} scenes × {len(families)} families "
          f"({min(max_workers, len(scene_indices))} workers)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_scene, idx): idx for idx in scene_indices}
        for fut in concurrent.futures.as_completed(futures):
            scene_idx, results = fut.result()
            summary = "  ".join(
                f"{fam[:3]}={'✓' if ok else '✗'}" for fam, ok in results.items()
            )
            print(f"  scene {scene_idx:03d}  {summary}")

    print("Done.")


# ── Log / video saving ────────────────────────────────────────────────────────

def _save_log(threads, scene_idx):
    log_dir   = Path(__file__).resolve().parent / "environment" / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"bug1_scene{scene_idx:03d}_{timestamp}.txt"

    with open(log_path, "w") as f:
        f.write("Bug1 Simulation Log\n")
        f.write(f"Scene: {scene_idx:03d}   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for t in threads:
            f.write(f"Family: {t.family}\n")
            f.write(f"Result: {'SUCCESS' if t.success else 'FAILED'}  "
                    f"({len(t.frames)} steps)\n")
            f.write("-" * 50 + "\n")
            for i, frame in enumerate(t.frames):
                x, y = frame.position
                f.write(f"  step {i:4d} | ({x:.4f}, {y:.4f}) | {frame.message}\n")
            f.write("\n")

    print(f"Log saved → {log_path}")


def _save_video(threads, scene_idx, speedup=3, fps=30):
    import shutil

    total_frames = max(len(t.frames) for t in threads)
    frame_idxs   = list(range(0, total_frames, max(1, speedup)))
    n            = len(threads)

    print(f"Saving {len(frame_idxs)} frames at {fps} fps ({speedup}x speed)...")

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Bug1 — scene {scene_idx:03d}", fontsize=10)

    panels = []
    for ax, t in zip(axes, threads):
        frame0 = t.frames[0]
        xmin, xmax, ymin, ymax = t.workspace or [0, 6, 0, 6]
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_facecolor("#f8f9fa")
        ax.set_title(t.family.replace("_", " "), fontsize=8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        for obs in frame0.obstacles:
            ax.add_patch(MplPolygon(
                obs["vertices"], closed=True, fill=False,
                edgecolor="#999999", linewidth=0.8, linestyle="--", alpha=0.45, zorder=1,
            ))

        patches = []
        for obs in frame0.obstacles:
            cls = obs.get("true_class", obs.get("class_true", "movable"))
            p = MplPolygon(
                obs["vertices"], closed=True,
                facecolor=DISPLAY_COLORS.get(cls, "lightblue"),
                edgecolor="#666666", linewidth=0.9, alpha=0.5, zorder=2,
            )
            ax.add_patch(p)
            patches.append(p)

        if t.scene:
            ax.scatter(*t.scene["start"][:2], s=80,  color="#2a9d8f", marker="o", zorder=6)
            ax.scatter(*t.scene["goal"][:2],  s=110, color="#d62828",  marker="*", zorder=6)

        path_line,  = ax.plot([], [], color="#1d3557", linewidth=1.4, zorder=3)
        robot_dot   = ax.scatter([], [], s=80, color="#264653", marker="o", zorder=5)
        status_text = ax.text(
            0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=6,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        panels.append((t, patches, path_line, robot_dot, status_text))

    def update(fi):
        real_idx = frame_idxs[fi]
        for t, patches, path_line, robot_dot, status_text in panels:
            idx   = min(real_idx, len(t.frames) - 1)
            frame = t.frames[idx]
            for patch, obs in zip(patches, frame.obstacles):
                cls = obs.get("true_class", obs.get("class_true", "movable"))
                patch.set_xy(obs["vertices"])
                patch.set_facecolor(DISPLAY_COLORS.get(cls, "lightblue"))
                patch.set_alpha(0.92 if obs["observed"] else 0.42)
                patch.set_edgecolor("#111111" if obs["observed"] else "#666666")
            positions = [(f.position[0], f.position[1]) for f in t.frames[:idx + 1]]
            arr = np.array(positions)
            path_line.set_data(arr[:, 0], arr[:, 1])
            robot_dot.set_offsets([[frame.position[0], frame.position[1]]])
            done = idx >= len(t.frames) - 1
            status_text.set_text(
                ("✓ SUCCESS" if t.success else "✗ FAILED") if done else f"step {idx}"
            )

    save_anim = FuncAnimation(fig, update, frames=len(frame_idxs),
                              interval=1000 // fps, blit=False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir  = Path(__file__).resolve().parent / "environment" / "data" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bug1_scene{scene_idx:03d}.mp4"

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
    parser.add_argument("--scene",    type=int,   nargs="+", default=[0],
                        help="One or more scene indices. Multiple → batch mode (no live viz).")
    parser.add_argument("--scenes",   type=str,   default=None,
                        help="Range of scenes, e.g. '0-99'. Implies batch mode.")
    parser.add_argument("--family",   nargs="*",  default=None, choices=FAMILIES)
    parser.add_argument("--steps",    type=int,   default=500)
    parser.add_argument("--sense",    type=float, default=0.55,
                        help="Sensing radius in metres (default: 0.55)")
    parser.add_argument("--step",     type=float, default=0.07,
                        help="Robot step size in metres (default: 0.07)")
    parser.add_argument("--save",     action="store_true",
                        help="Save video after simulation ends")
    parser.add_argument("--speedup",  type=int,   default=3)
    parser.add_argument("--workers",  type=int,   default=8,
                        help="Max parallel workers in batch mode (default: 8)")
    args = parser.parse_args()

    families = args.family or FAMILIES

    if args.scenes:
        lo, hi = map(int, args.scenes.split("-"))
        scene_indices = list(range(lo, hi + 1))
    else:
        scene_indices = args.scene

    if len(scene_indices) > 1 or args.scenes:
        print(f"Bug1 batch — {len(scene_indices)} scenes  families: {', '.join(families)}")
        run_batch(
            families,
            scene_indices = scene_indices,
            max_steps     = args.steps,
            step_size     = args.step,
            sensing_range = args.sense,
            save          = args.save,
            speedup       = args.speedup,
            max_workers   = args.workers,
        )
    else:
        scene_idx = scene_indices[0]
        print(f"Bug1 — scene {scene_idx:03d}  families: {', '.join(families)}")
        run_realtime(
            families,
            scene_idx     = scene_idx,
            max_steps     = args.steps,
            step_size     = args.step,
            sensing_range = args.sense,
            save          = args.save,
            speedup       = args.speedup,
        )
