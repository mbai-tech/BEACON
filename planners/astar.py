import heapq
import numpy as np
from geometry.polygons import transform_polygon, make_robot_polygon, min_distance

# Lattice parameters
D_STEP = 0.1
THETA_STEP = 15  # degrees
THETAS = list(range(0, 360, THETA_STEP))

def get_neighbors(x, y, theta):
    """Return neighboring poses from motion primitives."""
    neighbors = []
    rad = np.radians(theta)
    # Forward/back
    neighbors.append((x + D_STEP*np.cos(rad), y + D_STEP*np.sin(rad), theta, D_STEP))
    neighbors.append((x - D_STEP*np.cos(rad), y - D_STEP*np.sin(rad), theta, D_STEP))
    # Rotate
    neighbors.append((x, y, (theta + THETA_STEP) % 360, 0.01))
    neighbors.append((x, y, (theta - THETA_STEP) % 360, 0.01))
    return neighbors

def snap(x, y, theta):
    """Snap continuous pose to nearest lattice node."""
    return (round(x/D_STEP)*D_STEP, round(y/D_STEP)*D_STEP,
            round(theta/THETA_STEP)*THETA_STEP % 360)

def heuristic(x, y, gx, gy):
    return np.hypot(gx - x, gy - y)

def astar_collision_free(scene, robot_shape="rectangle"):
    from shapely.geometry import Polygon
    robot_base = make_robot_polygon(robot_shape)
    obstacles = [Polygon(o["vertices"]) for o in scene["obstacles"]]
    
    ws = scene["workspace"]
    xmin, xmax, ymin, ymax = ws[0], ws[1], ws[2], ws[3]
    
    sx, sy, st = scene["start"]
    gx, gy, gt = scene["goal"]
    start = snap(sx, sy, st)

    def in_bounds(x, y):
        return xmin + 0.3 <= x <= xmax - 0.3 and ymin + 0.3 <= y <= ymax - 0.3

    def is_collision(x, y, theta):
        r = transform_polygon(robot_base, x, y, theta)
        return any(r.intersects(obs) for obs in obstacles)

    open_heap = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        cx, cy, ct = current
        if np.hypot(cx - gx, cy - gy) < D_STEP * 1.5:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))

        for nx, ny, nt, cost in get_neighbors(cx, cy, ct):
            if not in_bounds(nx, ny):
                continue
            if is_collision(nx, ny, nt):
                continue
            ns = snap(nx, ny, nt)
            new_g = g_score[current] + cost
            if ns not in g_score or new_g < g_score[ns]:
                g_score[ns] = new_g
                f = new_g + heuristic(nx, ny, gx, gy)
                heapq.heappush(open_heap, (f, ns))
                came_from[ns] = current
    return None