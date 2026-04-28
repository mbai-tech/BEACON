import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

def draw_scene(scene, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Workspace
    ax.set_xlim(scene["workspace"][0], scene["workspace"][1])
    ax.set_ylim(scene["workspace"][2], scene["workspace"][3])
    ax.set_aspect("equal")

    # Obstacles
    for obs in scene["obstacles"]:
        x, y = obs.exterior.xy
        patch = MplPolygon(list(zip(x, y)), closed=True,
                           edgecolor="black", facecolor="lightgray")
        ax.add_patch(patch)

    # Start and goal
    sx, sy = scene["start"]
    gx, gy = scene["goal"]
    ax.plot(sx, sy, "go", markersize=8, label="Start")
    ax.plot(gx, gy, "ro", markersize=8, label="Goal")

    ax.set_title("Generated Environment")
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    # plt.show()
