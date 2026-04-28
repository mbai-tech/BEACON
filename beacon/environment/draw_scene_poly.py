import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


CLASS_COLORS = {
    "safe": "#A8D5BA",
    "movable": "#F6D186",
    "fragile": "#F4A6A6",
    "forbidden": "#9AA5B1",
}


def draw_scene(scene, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    xmin, xmax, ymin, ymax = scene["workspace"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    for obstacle in scene["obstacles"]:
        vertices = obstacle["vertices"]
        color = CLASS_COLORS.get(obstacle.get("class_true"), "lightgray")
        patch = MplPolygon(
            vertices,
            closed=True,
            edgecolor="black",
            facecolor=color,
            alpha=0.9,
        )
        ax.add_patch(patch)

    sx, sy, *_ = scene["start"]
    gx, gy, *_ = scene["goal"]
    ax.plot(sx, sy, "go", markersize=8, label="Start")
    ax.plot(gx, gy, "ro", markersize=8, label="Goal")

    ax.set_title(f"Environment: {scene['family']}")
    ax.legend(loc="upper right")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
