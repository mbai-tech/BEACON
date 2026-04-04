import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# TODO : fix colors NOTE: forbidden and the passageways in the narrow passage function should be the same label/color
CLASS_COLORS = {
    "safe": "lightgreen",
    "movable": "gold",
    "fragile": "tomato",
    "forbidden": "lightgray"
}


def draw_scene(scene, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    xmin, xmax, ymin, ymax = scene["workspace"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    for obs in scene["obstacles"]:
        verts = obs["vertices"]
        patch = MplPolygon(
            verts,
            closed=True,
            edgecolor="black",
            facecolor=CLASS_COLORS.get(obs["class_true"], "lightblue"),
            alpha=0.8
        )
        ax.add_patch(patch)

    sx, sy, _ = scene["start"]
    gx, gy, _ = scene["goal"]
    ax.plot(sx, sy, "bo", markersize=8)
    ax.plot(gx, gy, "r*", markersize=12)

    ax.set_title(scene["family"])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)