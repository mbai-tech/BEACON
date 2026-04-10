import heapq
import numpy as np
from geometry.polygons import transform_polygon, make_robot_polygon, contact_area

D_STEP = 0.1
THETA_STEP = 15
PLAN_COSTS = {"safe": 1, "movable": 3, "fragile": 10, "forbidden": 1000}

def get_neighbors(x, y, theta):
    neighbors = []
    rad = np.radians(theta)
    neighbors.append((x + D_STEP*np.cos(rad), y + D_STEP*np.sin(rad), theta, D_STEP))
    neighbors.append((x - D_STEP*np.cos(rad), y - D_STEP*np.sin(rad), theta, D_STEP))
    neighbors.append((x, y, (theta + THETA_STEP) % 360, 0.01))
    neighbors.append((x, y, (theta - THETA_STEP) % 360, 0.01))
    return neighbors

def snap(x, y, theta):
    return (round(x/D_STEP)*D_STEP, round(y/D_STEP)*D_STEP,
            round(theta/THETA_STEP)*THETA_STEP % 360)

def heuristic(x, y, gx, gy):
    return np.hypot(gx - x, gy - y)

def semantic_cost_at_pose(robot_poly, obstacles, posteriors, use_expected=True):
    """
    Compute semantic contact cost at this pose.
    use_expected=True  → Baseline 3 / SURP style (weighted sum over beliefs)
    use_expected=False → Baseline 2 style (argmax of prior, deterministic)
    """
    COSTS = [1, 3, 10, 1000]  # safe, movable, fragile, forbidden
    total = 0.0
    for i, obs in enumerate(obstacles):
        area = contact_area(robot_poly, obs)
        if area > 0:
            post = posteriors[i]
            if use_expected:
                cost = sum(post[c] * COSTS[c] for c in range(4))
            else:
                cost = COSTS[int(np.argmax(post))]
            total += cost * area
    return total

def semantic_astar(scene, robot_shape="rectangle", use_expected=True):
    from shapely.geometry import Polygon
    robot_base = make_robot_polygon(robot_shape)
    obstacles = [Polygon(o["vertices"]) for o in scene["obstacles"]]
    posteriors = [o["prior"] for o in scene["obstacles"]]

    ws = scene["workspace"]
    xmin, xmax, ymin, ymax = ws[0], ws[1], ws[2], ws[3]

    sx, sy, st = scene["start"]
    gx, gy, gt = scene["goal"]
    start = snap(sx, sy, st)

    def in_bounds(x, y):
        return xmin + 0.3 <= x <= xmax - 0.3 and ymin + 0.3 <= y <= ymax - 0.3

    def edge_cost(x, y, theta, move_cost):
        r = transform_polygon(robot_base, x, y, theta)
        sem = semantic_cost_at_pose(r, obstacles, posteriors, use_expected)
        return move_cost + sem

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

        for nx, ny, nt, move_cost in get_neighbors(cx, cy, ct):
            if not in_bounds(nx, ny):
                continue
            ns = snap(nx, ny, nt)
            new_g = g_score[current] + edge_cost(nx, ny, nt, move_cost)
            if ns not in g_score or new_g < g_score[ns]:
                g_score[ns] = new_g
                f = new_g + heuristic(nx, ny, gx, gy)
                heapq.heappush(open_heap, (f, ns))
                came_from[ns] = current
    return None