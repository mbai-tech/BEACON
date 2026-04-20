import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

CLASS_COLORS = {
    "safe": "lightgreen",
    "movable": "gold",
    "fragile": "tomato"
}


def draw_scene(scene, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    xmin, xmax, ymin, ymax = scene["workspace"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    for obs in scene["obstacles"]:
        patch = MplPolygon(
            obs["vertices"],
            closed=True,
            edgecolor="black",
            facecolor=CLASS_COLORS.get(obs["class_true"], "lightblue"),
            alpha=0.8
        )
        ax.add_patch(patch)

    sx, sy, _ = scene["start"]
    gx, gy, _ = scene["goal"]

    ax.plot(sx, sy, "bo", markersize=8, label="Start")
    ax.plot(gx, gy, "r*", markersize=12, label="Goal")

    ax.set_title(f"{scene['family']} ({len(scene['obstacles'])} circles)")
    ax.legend(loc="upper right")

    # no grid lines
    ax.set_xticks([])
    ax.set_yticks([])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)