from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np

from .config import TerrainConfig


@dataclass
class TerrainField:
    body_id: int
    scenario: str
    cfg: TerrainConfig
    xs: np.ndarray
    ys: np.ndarray
    heights: np.ndarray
    rock_ids: list[int] = field(default_factory=list)
    decorative_ids: list[int] = field(default_factory=list)

    def height_at(self, x: float, y: float) -> float:
        x = float(np.clip(x, self.xs[0], self.xs[-1]))
        y = float(np.clip(y, self.ys[0], self.ys[-1]))
        ix = int(np.clip(np.searchsorted(self.xs, x) - 1, 0, len(self.xs) - 2))
        iy = int(np.clip(np.searchsorted(self.ys, y) - 1, 0, len(self.ys) - 2))
        x0, x1 = float(self.xs[ix]), float(self.xs[ix + 1])
        y0, y1 = float(self.ys[iy]), float(self.ys[iy + 1])
        tx = (x - x0) / max(x1 - x0, 1e-12)
        ty = (y - y0) / max(y1 - y0, 1e-12)
        z00 = float(self.heights[iy, ix])
        z10 = float(self.heights[iy, ix + 1])
        z01 = float(self.heights[iy + 1, ix])
        z11 = float(self.heights[iy + 1, ix + 1])
        z0 = (1.0 - tx) * z00 + tx * z10
        z1 = (1.0 - tx) * z01 + tx * z11
        return float((1.0 - ty) * z0 + ty * z1)

    def gradient_at(self, x: float, y: float) -> tuple[float, float]:
        dx = max(float(self.xs[1] - self.xs[0]), 1e-4)
        dy = max(float(self.ys[1] - self.ys[0]), 1e-4)
        dzdx = (self.height_at(x + dx, y) - self.height_at(x - dx, y)) / (2.0 * dx)
        dzdy = (self.height_at(x, y + dy) - self.height_at(x, y - dy)) / (2.0 * dy)
        return float(dzdx), float(dzdy)

    def slope_along_heading(self, x: float, y: float, yaw: float) -> float:
        gx, gy = self.gradient_at(x, y)
        grade = gx * math.cos(yaw) + gy * math.sin(yaw)
        return float(math.atan(grade))

    def friction_at(self, x: float, y: float) -> float:
        c = self.cfg
        if self.scenario == "low_mu":
            return float(c.low_friction)
        if self.scenario == "split_mu":
            return float(c.split_left_friction if y >= 0.0 else c.split_right_friction)
        if self.scenario == "patchy_mu":
            mu = (
                0.55
                + 0.31 * np.sin(0.52 * x + 0.35) * np.cos(0.78 * y - 0.4)
                + 0.11 * np.sin(1.7 + 0.21 * x - 0.57 * y)
            )
            return float(np.clip(mu, c.patch_min_friction, c.patch_max_friction))
        return float(c.base_friction)


def _smooth_random_modes(
    X: np.ndarray,
    Y: np.ndarray,
    rng: np.random.Generator,
    amp: float,
) -> np.ndarray:
    z = np.zeros_like(X, dtype=float)
    bands = [
        (0.045, 0.085, 0.52),
        (0.090, 0.180, 0.27),
        (0.180, 0.340, 0.14),
        (0.340, 0.600, 0.07),
    ]
    for fmin, fmax, weight in bands:
        for _ in range(5):
            fx = rng.uniform(fmin, fmax)
            fy = rng.uniform(fmin, fmax)
            phase = rng.uniform(-np.pi, np.pi)
            z += weight * rng.normal() * np.sin(fx * X + fy * Y + phase)
    z -= np.mean(z)
    std = float(np.std(z))
    if std > 1e-9:
        z *= float(amp) / std
    return z


def _gaussian_hill(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cx: float,
    cy: float,
    amp: float,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    return amp * np.exp(
        -0.5
        * (
            ((X - cx) / max(sigma_x, 1e-6)) ** 2
            + ((Y - cy) / max(sigma_y, 1e-6)) ** 2
        )
    )


def _cross_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cx: float,
    cy: float = 0.0,
    amp: float,
    width: float,
    angle_deg: float = 0.0,
) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    ca = np.cos(angle)
    sa = np.sin(angle)
    xr = ca * (X - cx) + sa * (Y - cy)
    return amp * np.exp(-(xr / max(width, 1e-6)) ** 2)


def _add_crater(
    Z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cx: float,
    cy: float,
    radius: float,
    depth: float,
) -> None:
    radius = max(float(radius), 0.05)
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    bowl = -depth * np.exp(-(r / (0.68 * radius)) ** 2)
    rim = 0.40 * depth * np.exp(-((r - radius) / (0.22 * radius)) ** 2)
    Z += bowl + rim


def _apply_safe_start_zone(
    Z: np.ndarray,
    X: np.ndarray,
    *,
    flat_until_x: float = -9.8,
    rough_from_x: float = -8.0,
) -> np.ndarray:
    if rough_from_x <= flat_until_x:
        raise ValueError("rough_from_x must exceed flat_until_x")
    u = np.clip(
        (X - flat_until_x) / (rough_from_x - flat_until_x),
        0.0,
        1.0,
    )
    gate = u * u * (3.0 - 2.0 * u)
    return Z * gate


def _terrain_heights(
    scenario: str,
    cfg: TerrainConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(cfg.grid_n)
    xs = np.linspace(-cfg.size_x / 2.0, cfg.size_x / 2.0, n)
    ys = np.linspace(-cfg.size_y / 2.0, cfg.size_y / 2.0, n)
    X, Y = np.meshgrid(xs, ys)

    if scenario == "flat":
        return xs, ys, np.zeros_like(X, dtype=np.float32)

    rng = np.random.default_rng(seed)
    if scenario == "rough":
        base_amp = float(np.clip(cfg.rough_roughness, 0.025, 0.045))
    elif scenario == "extreme_rough":
        base_amp = float(np.clip(1.20 * cfg.rough_roughness, 0.040, 0.060))
    else:
        base_amp = float(np.clip(cfg.nominal_roughness, 0.015, 0.030))

    Z = _smooth_random_modes(X, Y, rng, base_amp)

    # Small regolith-scale undulation.
    Z += 0.018 * np.sin(0.42 * X + 0.16 * Y + 0.40)
    Z += 0.011 * np.sin(0.90 * X - 0.35 * Y - 1.20)
    Z += 0.007 * np.sin(1.45 * Y + 0.27 * X)

    if scenario in ("rough", "extreme_rough"):
        scale = 1.22 if scenario == "extreme_rough" else 1.0

        # Broad macro-features intentionally placed after the flat launch zone.
        Z += _gaussian_hill(X, Y, cx=-5.8, cy=0.0, amp=0.24 * scale, sigma_x=2.20, sigma_y=4.50)
        Z += _gaussian_hill(X, Y, cx=-1.5, cy=0.10, amp=-0.17 * scale, sigma_x=2.00, sigma_y=4.20)
        Z += _gaussian_hill(X, Y, cx=2.8, cy=-0.10, amp=0.30 * scale, sigma_x=2.30, sigma_y=4.50)
        Z += _gaussian_hill(X, Y, cx=6.8, cy=0.15, amp=-0.20 * scale, sigma_x=2.10, sigma_y=4.20)
        Z += _gaussian_hill(X, Y, cx=10.2, cy=-0.10, amp=0.25 * scale, sigma_x=2.00, sigma_y=4.10)

        # Oblique ridges excite left/right rocker-bogie motion without sharp steps.
        Z += _cross_ridge(X, Y, cx=-3.8, amp=0.050 * scale, width=1.10, angle_deg=12.0)
        Z -= _cross_ridge(X, Y, cx=4.7, amp=0.040 * scale, width=1.20, angle_deg=-13.0)
        Z += _cross_ridge(X, Y, cx=8.6, amp=0.050 * scale, width=1.05, angle_deg=15.0)

        corridor = np.exp(-(Y / 3.5) ** 2)
        Z += 0.025 * scale * corridor * np.sin(0.90 * X + 0.75 * Y + 0.40)
        Z += 0.014 * scale * corridor * np.sin(1.40 * X - 0.95 * Y - 0.70)
    else:
        # Moderate geometry for traction/disturbance scenarios.
        Z += _gaussian_hill(X, Y, cx=-4.5, cy=0.0, amp=0.11, sigma_x=2.6, sigma_y=4.5)
        Z += _gaussian_hill(X, Y, cx=2.5, cy=0.0, amp=-0.08, sigma_x=2.4, sigma_y=4.3)
        Z += _gaussian_hill(X, Y, cx=8.0, cy=-0.1, amp=0.12, sigma_x=2.5, sigma_y=4.4)

    # Decorative craters outside the nominal path corridor.
    _add_crater(Z, X, Y, cx=-8.5, cy=+7.0, radius=1.25 * cfg.crater_radius, depth=0.90 * cfg.crater_depth)
    _add_crater(Z, X, Y, cx=+5.5, cy=-7.2, radius=1.45 * cfg.crater_radius, depth=0.80 * cfg.crater_depth)
    _add_crater(Z, X, Y, cx=+15.0, cy=+6.4, radius=0.95 * cfg.crater_radius, depth=0.65 * cfg.crater_depth)
    _add_crater(Z, X, Y, cx=-17.0, cy=-6.5, radius=0.82 * cfg.crater_radius, depth=0.55 * cfg.crater_depth)

    if scenario in ("rough", "extreme_rough"):
        _add_crater(
            Z,
            X,
            Y,
            cx=12.5,
            cy=1.20,
            radius=1.15,
            depth=0.14 if scenario == "rough" else 0.18,
        )

    if scenario == "slope":
        Z += cfg.slope_grade * (X + cfg.size_x / 2.0)
        Z += 0.025 * np.sin(0.58 * X + 0.30 * Y)

    # Completely flat launch area around x=-12, then smooth transition.
    Z = _apply_safe_start_zone(Z, X, flat_until_x=-9.8, rough_from_x=-8.0)

    ix0 = int(np.argmin(np.abs(xs + 12.0)))
    iy0 = int(np.argmin(np.abs(ys)))
    Z -= float(Z[iy0, ix0])
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return xs, ys, Z.astype(np.float32)


def _spawn_collision_rocks(pb, field: TerrainField, seed: int) -> list[int]:
    if field.scenario not in ("rock_field", "extreme_rough"):
        return []

    rng = np.random.default_rng(seed + 9182)
    ids: list[int] = []
    count = int(field.cfg.rock_count)
    if field.scenario == "extreme_rough":
        count = max(6, min(count, 10))

    for i in range(count):
        x = float(rng.uniform(-18.0, 18.0))
        ymag = float(rng.uniform(2.2, 12.0))
        y = ymag if rng.random() > 0.5 else -ymag

        if i < 4:
            x, y = [
                (-5.0, +1.35),
                (0.0, -1.35),
                (5.8, +1.40),
                (10.0, -1.35),
            ][i]

        radius = float(rng.uniform(0.07, 0.11) if i < 4 else rng.uniform(0.09, 0.24))
        z = field.height_at(x, y) + 0.52 * radius
        col = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius)
        vis = pb.createVisualShape(
            pb.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[0.30, 0.15, 0.085, 1.0],
        )
        bid = pb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, z],
        )
        pb.changeDynamics(bid, -1, lateralFriction=0.85, restitution=0.0)
        ids.append(bid)
    return ids


def _spawn_decorative_rocks(pb, field: TerrainField, seed: int) -> list[int]:
    if field.scenario == "flat":
        return []

    rng = np.random.default_rng(seed + 32491)
    ids: list[int] = []
    for _ in range(int(field.cfg.decorative_rock_count)):
        x = float(rng.uniform(field.xs[0] + 1.5, field.xs[-1] - 1.5))
        y = float(rng.uniform(field.ys[0] + 1.5, field.ys[-1] - 1.5))

        if abs(y) < 2.4 and -15.0 < x < 15.0:
            side = 1.0 if rng.random() > 0.5 else -1.0
            y = side * (2.7 + rng.uniform(0.0, 4.5))

        sx = float(rng.uniform(0.08, 0.28))
        sy = float(rng.uniform(0.07, 0.23))
        sz = float(rng.uniform(0.05, 0.16))
        z = field.height_at(x, y) + 0.50 * sz
        vis = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=[sx, sy, sz],
            rgbaColor=[
                float(rng.uniform(0.25, 0.38)),
                float(rng.uniform(0.12, 0.20)),
                float(rng.uniform(0.07, 0.12)),
                1.0,
            ],
        )
        orn = pb.getQuaternionFromEuler([
            rng.uniform(-0.25, 0.25),
            rng.uniform(-0.25, 0.25),
            rng.uniform(-np.pi, np.pi),
        ])
        bid = pb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, z],
            baseOrientation=orn,
        )
        ids.append(bid)
    return ids


def _apply_texture(pb, terrain_id: int, cfg: TerrainConfig) -> None:
    texture = Path(__file__).resolve().parents[1] / "assets" / "visual" / "mars_regolith_texture.png"
    if not texture.exists():
        pb.changeVisualShape(terrain_id, -1, rgbaColor=[0.52, 0.255, 0.115, 1.0])
        return

    try:
        tid = pb.loadTexture(str(texture))
        pb.changeVisualShape(
            terrain_id,
            -1,
            textureUniqueId=tid,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
        )
    except Exception:
        pb.changeVisualShape(terrain_id, -1, rgbaColor=[0.52, 0.255, 0.115, 1.0])


def create_terrain(pb, scenario: str, cfg: TerrainConfig, seed: int = 7) -> TerrainField:
    xs, ys, Z = _terrain_heights(scenario, cfg, seed)
    sx = cfg.size_x / max(int(cfg.grid_n) - 1, 1)
    sy = cfg.size_y / max(int(cfg.grid_n) - 1, 1)

    shape = pb.createCollisionShape(
        shapeType=pb.GEOM_HEIGHTFIELD,
        meshScale=[float(sx), float(sy), 1.0],
        heightfieldTextureScaling=float(cfg.texture_repeat),
        heightfieldData=Z.flatten(order="C").tolist(),
        numHeightfieldRows=int(cfg.grid_n),
        numHeightfieldColumns=int(cfg.grid_n),
    )
    terrain_id = pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape)
    _apply_texture(pb, terrain_id, cfg)
    pb.changeDynamics(
        terrain_id,
        -1,
        lateralFriction=float(cfg.base_friction),
        restitution=0.0,
        rollingFriction=0.003,
        spinningFriction=0.003,
    )

    field = TerrainField(
        body_id=terrain_id,
        scenario=scenario,
        cfg=cfg,
        xs=xs,
        ys=ys,
        heights=Z,
    )
    field.decorative_ids = _spawn_decorative_rocks(pb, field, seed)
    field.rock_ids = _spawn_collision_rocks(pb, field, seed)
    return field
