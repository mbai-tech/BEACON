import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

fig, ax = plt.subplots(figsize=(6, 6))

# Workspace boundary
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")

# One obstacle
obstacle = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)],
                   closed=True, edgecolor="black", facecolor="lightgray")
ax.add_patch(obstacle)

# Start and goal
ax.plot(1, 1, "go", markersize=8, label="Start")
ax.plot(8, 8, "ro", markersize=8, label="Goal")

ax.set_title("My First Environment")
ax.legend()
plt.savefig("data/images/test_scene.png", dpi=200, bbox_inches="tight")
plt.show()