import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon

CLASS_COLORS = {"safe":"#aed6f1","movable":"#a9dfbf","fragile":"#f9e79f","forbidden":"#f1948a"}

def plot_scene(scene, path=None, title="Scene", save_path="output.png"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_aspect("equal"); ax.set_title(title)

    for obs in scene["obstacles"]:
        poly = plt.Polygon(obs["vertices"],
                           color=CLASS_COLORS[obs["true_class"]], alpha=0.7)
        ax.add_patch(poly)
        cx = sum(v[0] for v in obs["vertices"]) / len(obs["vertices"])
        cy = sum(v[1] for v in obs["vertices"]) / len(obs["vertices"])
        ax.text(cx, cy, obs["true_class"][0].upper(), ha="center", va="center", fontsize=8)

    sx, sy, _ = scene["start"]
    gx, gy, _ = scene["goal"]
    ax.plot(sx, sy, "go", markersize=10, label="start")
    ax.plot(gx, gy, "r*", markersize=14, label="goal")

    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, "b-", linewidth=1.5, alpha=0.7, label="path")

    ax.legend(); plt.tight_layout()
    plt.savefig("save_path", dpi=150)
    plt.show()