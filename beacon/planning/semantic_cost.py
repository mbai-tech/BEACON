from typing import Dict, Tuple

import pybullet as p
import pybullet_data

LOOKUP_TABLE: Dict[str, int] = {
    "movable_object":   2,   # light, easy to displace
    "unmovable_object": 8,   # heavy / fixed
}

# Cost threshold: cost < threshold → movable, cost >= threshold → unmovable.
# With mixed distribution: costs 1–5 are movable, costs 7–9 are unmovable.
MOVABLE_THRESHOLD: int = 6

def assign_cost(
    body_id: int,
    lookup_table: Dict[str, int] = LOOKUP_TABLE,
    beta: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    eps: float = 1e-6,
) -> int:
    body_info = p.getBodyInfo(body_id)
    name = body_info[1].decode("utf-8") if isinstance(body_info[1], bytes) else str(body_info[1])
    if name in lookup_table:
        return lookup_table[name]

    aabb_min, aabb_max = p.getAABB(body_id)
    w = aabb_max[0] - aabb_min[0]
    d = aabb_max[1] - aabb_min[1]
    h = aabb_max[2] - aabb_min[2]

    dyn = p.getDynamicsInfo(body_id, -1)
    mass        = dyn[0]
    friction    = dyn[1]
    restitution = dyn[5]

    phi_top  = h / (min(w, d) + eps)
    phi_mom  = mass * restitution
    phi_disp = mass * friction

    b1, b2, b3 = beta
    c_raw = (
        b1 * _clip(phi_top,  0.0, 5.0)
      + b2 * _clip(phi_mom,  0.0, 3.0)
      + b3 * _clip(phi_disp, 0.0, 2.0)
    )
    return min(10, max(0, round(c_raw)))


def is_movable(
    body_id: int,
    lookup_table: Dict[str, int] = LOOKUP_TABLE,
    threshold: int = MOVABLE_THRESHOLD,
    beta: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    eps: float = 1e-6,
) -> bool:
    return assign_cost(body_id, lookup_table, beta, eps) < threshold


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

# (mass_kg, friction, restitution, half_extents_xyz_m)
# Movable: flat/wide box → phi_top = h/(min(w,d)) = 0.05/0.10 = 0.5 → low cost
# Unmovable: tall heavy box → phi_top = 0.30/0.08 = 3.75, phi_mom = 8*0.4 = 3.2 → high cost
_PROXY_PARAMS: Dict[str, Tuple[float, float, float, Tuple[float, float, float]]] = {
    "movable_object":   (0.30, 0.40, 0.10, (0.10, 0.10, 0.05)),   # flat, light
    "unmovable_object": (8.00, 0.70, 0.40, (0.08, 0.08, 0.30)),   # tall, heavy
}


def validate_proxy(beta: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> None:
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)

    header = f"{'Object':<20} {'Lookup':>6}  {'Proxy':>5}  {'|Δ|':>4}  {'Movable?':>8}"
    print(header)
    print("-" * len(header))

    for name, lookup_cost in LOOKUP_TABLE.items():
        mass, friction, restitution, half_extents = _PROXY_PARAMS[name]

        col_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=list(half_extents),
            physicsClientId=client,
        )
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            basePosition=[0, 0, 0.5],
            physicsClientId=client,
        )
        p.changeDynamics(
            body_id, -1,
            lateralFriction=friction,
            restitution=restitution,
            physicsClientId=client,
        )

        proxy_cost = _assign_cost_on_client(body_id, client, {}, beta)
        diff = abs(lookup_cost - proxy_cost)
        movable = proxy_cost < MOVABLE_THRESHOLD
        print(f"{name:<20} {lookup_cost:>6}  {proxy_cost:>5}  {diff:>4}  {'yes' if movable else 'no':>8}")

        p.removeBody(body_id, physicsClientId=client)

    p.disconnect(client)


def _assign_cost_on_client(
    body_id: int,
    client: int,
    lookup_table: Dict[str, int],
    beta: Tuple[float, float, float],
    eps: float = 1e-6,
) -> int:
    body_info = p.getBodyInfo(body_id, physicsClientId=client)
    name = body_info[1].decode("utf-8") if isinstance(body_info[1], bytes) else str(body_info[1])
    if name in lookup_table:
        return lookup_table[name]

    aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=client)
    w = aabb_max[0] - aabb_min[0]
    d = aabb_max[1] - aabb_min[1]
    h = aabb_max[2] - aabb_min[2]

    dyn = p.getDynamicsInfo(body_id, -1, physicsClientId=client)
    mass        = dyn[0]
    friction    = dyn[1]
    restitution = dyn[5]

    phi_top  = h / (min(w, d) + eps)
    phi_mom  = mass * restitution
    phi_disp = mass * friction

    b1, b2, b3 = beta
    c_raw = (
        b1 * _clip(phi_top,  0.0, 5.0)
      + b2 * _clip(phi_mom,  0.0, 3.0)
      + b3 * _clip(phi_disp, 0.0, 2.0)
    )
    return min(10, max(0, round(c_raw)))


if __name__ == "__main__":
    validate_proxy()
