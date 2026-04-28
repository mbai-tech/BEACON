import math
from collections import deque
from shapely.geometry import Point, Polygon


def obstacle_polygons(scene):
    return [Polygon(obs["vertices"]) for obs in scene["obstacles"]]


def point_is_blocked(x, y, scene, mode="collision_free", robot_radius=0.12):
    """
    mode:
      - 'collision_free': all obstacles block motion
      - 'contact_allowed': only fragile obstacles block motion
    """
    p = Point(x, y).buffer(robot_radius, resolution=16)

    for obs in scene["obstacles"]:
        poly = Polygon(obs["vertices"])
        cls = obs["class_true"]

        if mode == "collision_free":
            if p.intersects(poly):
                return True

        elif mode == "contact_allowed":
            # movable and safe can be contacted/pushed through
            if cls == "fragile" and p.intersects(poly):
                return True

        else:
            raise ValueError("mode must be 'collision_free' or 'contact_allowed'")

    return False


def build_grid(scene, step=0.15, mode="collision_free", robot_radius=0.12):
    xmin, xmax, ymin, ymax = scene["workspace"]

    xs = []
    x = xmin
    while x <= xmax + 1e-9:
        xs.append(round(x, 6))
        x += step

    ys = []
    y = ymin
    while y <= ymax + 1e-9:
        ys.append(round(y, 6))
        y += step

    blocked = set()
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if point_is_blocked(x, y, scene, mode=mode, robot_radius=robot_radius):
                blocked.add((ix, iy))

    return xs, ys, blocked


def nearest_free_index(xs, ys, blocked, point):
    px, py = point
    best = None
    best_dist = float("inf")

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if (ix, iy) in blocked:
                continue
            d = (x - px) ** 2 + (y - py) ** 2
            if d < best_dist:
                best_dist = d
                best = (ix, iy)

    return best


def shortest_path_length(scene, mode="collision_free", step=0.15, robot_radius=0.12):
    xs, ys, blocked = build_grid(
        scene, step=step, mode=mode, robot_radius=robot_radius
    )

    start_xy = tuple(scene["start"][:2])
    goal_xy = tuple(scene["goal"][:2])

    start = nearest_free_index(xs, ys, blocked, start_xy)
    goal = nearest_free_index(xs, ys, blocked, goal_xy)

    if start is None or goal is None:
        return None

    q = deque()
    q.append(start)
    dist = {start: 0}

    neighbors = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    while q:
        cur = q.popleft()
        if cur == goal:
            return dist[cur] * step

        cx, cy = cur
        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < len(xs) and 0 <= ny < len(ys)):
                continue
            if (nx, ny) in blocked:
                continue
            if (nx, ny) in dist:
                continue

            dist[(nx, ny)] = dist[cur] + (math.sqrt(2) if dx != 0 and dy != 0 else 1)
            q.append((nx, ny))

    return None


def validate_scene(scene, step=0.15, robot_radius=0.12, shortcut_margin=0.5):
    """
    Returns a dictionary with path info and family validity.
    """
    free_len = shortest_path_length(
        scene, mode="collision_free", step=step, robot_radius=robot_radius
    )
    contact_len = shortest_path_length(
        scene, mode="contact_allowed", step=step, robot_radius=robot_radius
    )

    family = scene["family"]

    result = {
        "family": family,
        "collision_free_path_exists": free_len is not None,
        "contact_allowed_path_exists": contact_len is not None,
        "collision_free_length": None if free_len is None else round(free_len, 3),
        "contact_allowed_length": None if contact_len is None else round(contact_len, 3),
        "valid": False,
    }

    if family in ["sparse", "cluttered"]:
        result["valid"] = (free_len is not None)

    elif family == "collision_required":
        result["valid"] = (free_len is None and contact_len is not None)

    elif family == "collision_shortcut":
        result["valid"] = (
            free_len is not None
            and contact_len is not None
            and contact_len + shortcut_margin < free_len
        )

    return result