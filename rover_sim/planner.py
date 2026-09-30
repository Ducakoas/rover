from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np

from .paths import resample_path


@dataclass
class PlanResult:
    path_xy: np.ndarray
    raw_path_xy: np.ndarray
    explored_nodes: int
    path_length_m: float
    max_slope_deg_on_path: float


def _path_length(path_xy: np.ndarray) -> float:
    p = np.asarray(path_xy, dtype=float)
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _bilinear_grid(terrain, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    z = np.empty((len(ys), len(xs)), dtype=float)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            z[iy, ix] = terrain.height_at(float(x), float(y))
    return z


def _line_free(a: np.ndarray, b: np.ndarray, blocked: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> bool:
    d = float(np.linalg.norm(b - a))
    if d <= 1e-9:
        return True
    step = 0.45 * min(float(xs[1] - xs[0]), float(ys[1] - ys[0]))
    n = max(2, int(math.ceil(d / max(step, 1e-3))))
    for u in np.linspace(0.0, 1.0, n):
        q = (1.0 - u) * a + u * b
        ix = int(np.clip(np.searchsorted(xs, q[0]) - 1, 0, len(xs) - 1))
        iy = int(np.clip(np.searchsorted(ys, q[1]) - 1, 0, len(ys) - 1))
        if blocked[iy, ix]:
            return False
    return True


def simplify_line_of_sight(path_xy: np.ndarray, blocked: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    p = np.asarray(path_xy, dtype=float)
    if len(p) <= 2:
        return p.copy()
    out = [p[0]]
    i = 0
    while i < len(p) - 1:
        j = len(p) - 1
        while j > i + 1 and not _line_free(p[i], p[j], blocked, xs, ys):
            j -= 1
        out.append(p[j])
        i = j
    return np.asarray(out, dtype=float)


def _nearest_index(value: float, grid: np.ndarray) -> int:
    return int(np.argmin(np.abs(grid - float(value))))


def _obstacle_mask(
    xs: np.ndarray,
    ys: np.ndarray,
    obstacle_xy: Iterable[tuple[float, float]],
    clearance_m: float,
) -> np.ndarray:
    X, Y = np.meshgrid(xs, ys)
    mask = np.zeros_like(X, dtype=bool)
    r2 = float(clearance_m) ** 2
    for ox, oy in obstacle_xy:
        mask |= (X - float(ox)) ** 2 + (Y - float(oy)) ** 2 <= r2
    return mask


def choose_approach_point(
    terrain,
    start_xy: np.ndarray,
    sample_xy: np.ndarray,
    approach_distance_m: float,
    max_slope_deg: float,
    obstacle_xy: Iterable[tuple[float, float]] = (),
    obstacle_clearance_m: float = 0.70,
    n_candidates: int = 32,
) -> np.ndarray:
    start = np.asarray(start_xy, dtype=float)
    sample = np.asarray(sample_xy, dtype=float)
    obstacles = list(obstacle_xy)
    candidates: list[tuple[float, np.ndarray]] = []

    for ang in np.linspace(-np.pi, np.pi, n_candidates, endpoint=False):
        q = sample + approach_distance_m * np.array([math.cos(ang), math.sin(ang)], dtype=float)
        if not (terrain.xs[1] < q[0] < terrain.xs[-2] and terrain.ys[1] < q[1] < terrain.ys[-2]):
            continue
        gx, gy = terrain.gradient_at(float(q[0]), float(q[1]))
        slope = math.degrees(math.atan(math.hypot(gx, gy)))
        if slope > max_slope_deg:
            continue
        if any(np.linalg.norm(q - np.asarray(o, dtype=float)) < obstacle_clearance_m for o in obstacles):
            continue
        # Prefer short travel, gentle ground, and an approach direction that does not
        # require a sharp final turn relative to the initial target bearing.
        travel = float(np.linalg.norm(q - start))
        score = travel + 0.22 * slope
        candidates.append((score, q))

    if not candidates:
        # Conservative fallback: stand off from the sample along the start->sample line.
        vec = sample - start
        n = float(np.linalg.norm(vec))
        if n < 1e-9:
            vec = np.array([1.0, 0.0], dtype=float)
            n = 1.0
        return sample - approach_distance_m * vec / n

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def plan_terrain_astar(
    terrain,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    resolution_m: float = 0.45,
    max_slope_deg: float = 24.0,
    slope_weight: float = 7.0,
    climb_weight: float = 2.0,
    obstacle_xy: Iterable[tuple[float, float]] = (),
    obstacle_clearance_m: float = 0.70,
    path_spacing_m: float = 0.30,
    final_heading_target_xy: np.ndarray | None = None,
) -> PlanResult:
    """Terrain-aware 8-connected A* on a heightfield.

    This is a mission/path-planning layer, not the low-level rover controller.
    Cells steeper than ``max_slope_deg`` are treated as impassable. Traversal
    cost increases with local slope and uphill height change. Discrete rock
    locations can be inflated by ``obstacle_clearance_m``.
    """

    start = np.asarray(start_xy, dtype=float)
    goal = np.asarray(goal_xy, dtype=float)
    res = max(float(resolution_m), 0.15)

    margin = max(0.7, obstacle_clearance_m)
    xmin = float(terrain.xs[0] + margin)
    xmax = float(terrain.xs[-1] - margin)
    ymin = float(terrain.ys[0] + margin)
    ymax = float(terrain.ys[-1] - margin)

    xs = np.arange(xmin, xmax + 0.5 * res, res, dtype=float)
    ys = np.arange(ymin, ymax + 0.5 * res, res, dtype=float)
    Z = _bilinear_grid(terrain, xs, ys)

    dzdy, dzdx = np.gradient(Z, ys, xs, edge_order=1)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    slope_deg = np.degrees(slope_rad)
    blocked = slope_deg > float(max_slope_deg)
    blocked |= _obstacle_mask(xs, ys, obstacle_xy, obstacle_clearance_m)

    sx, sy = _nearest_index(start[0], xs), _nearest_index(start[1], ys)
    gx, gy = _nearest_index(goal[0], xs), _nearest_index(goal[1], ys)
    blocked[sy, sx] = False
    blocked[gy, gx] = False

    start_node = (sy, sx)
    goal_node = (gy, gx)
    ny, nx = blocked.shape

    g_cost = np.full((ny, nx), np.inf, dtype=float)
    g_cost[start_node] = 0.0
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    closed = np.zeros((ny, nx), dtype=bool)
    heap: list[tuple[float, float, tuple[int, int]]] = []

    def heuristic(node: tuple[int, int]) -> float:
        iy, ix = node
        return math.hypot(xs[ix] - xs[gx], ys[iy] - ys[gy])

    heapq.heappush(heap, (heuristic(start_node), 0.0, start_node))
    explored = 0
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]

    found = False
    while heap:
        _, g_here, node = heapq.heappop(heap)
        iy, ix = node
        if closed[iy, ix]:
            continue
        closed[iy, ix] = True
        explored += 1
        if node == goal_node:
            found = True
            break

        for dy, dx in neighbors:
            jy, jx = iy + dy, ix + dx
            if not (0 <= jy < ny and 0 <= jx < nx):
                continue
            if blocked[jy, jx] or closed[jy, jx]:
                continue

            step_len = math.hypot(xs[jx] - xs[ix], ys[jy] - ys[iy])
            local_slope = 0.5 * (slope_rad[iy, ix] + slope_rad[jy, jx])
            dz = Z[jy, jx] - Z[iy, ix]
            uphill = max(0.0, dz) / max(step_len, 1e-6)
            step_cost = step_len * (
                1.0
                + slope_weight * abs(math.tan(float(local_slope)))
                + climb_weight * uphill
            )
            ng = g_here + step_cost
            if ng + 1e-12 < g_cost[jy, jx]:
                g_cost[jy, jx] = ng
                parent[(jy, jx)] = node
                heapq.heappush(heap, (ng + heuristic((jy, jx)), ng, (jy, jx)))

    if not found:
        raise RuntimeError(
            "Terrain-aware A* could not find a feasible route. "
            "Try increasing mission.planner_max_slope_deg or reducing obstacle clearance."
        )

    nodes = [goal_node]
    while nodes[-1] != start_node:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    raw = np.asarray([[xs[ix], ys[iy]] for iy, ix in nodes], dtype=float)
    raw[0] = start
    raw[-1] = goal

    simple = simplify_line_of_sight(raw, blocked, xs, ys)

    # Encourage the final rover heading to face the sample so the arm starts from
    # a sensible pose. Insert a pre-goal waypoint behind the approach point.
    if final_heading_target_xy is not None and len(simple) >= 2:
        target = np.asarray(final_heading_target_xy, dtype=float)
        direction = target - goal
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction /= norm
            pre_goal = goal - min(0.75, 0.60 * float(np.linalg.norm(goal - simple[-2]))) * direction
            if _line_free(simple[-2], pre_goal, blocked, xs, ys) and _line_free(pre_goal, goal, blocked, xs, ys):
                simple = np.vstack([simple[:-1], pre_goal, goal])

    path = resample_path(simple, spacing=path_spacing_m)
    path[0] = start
    path[-1] = goal

    max_slope = 0.0
    for q in path:
        gx_t, gy_t = terrain.gradient_at(float(q[0]), float(q[1]))
        max_slope = max(max_slope, math.degrees(math.atan(math.hypot(gx_t, gy_t))))

    return PlanResult(
        path_xy=path,
        raw_path_xy=raw,
        explored_nodes=explored,
        path_length_m=_path_length(path),
        max_slope_deg_on_path=float(max_slope),
    )
