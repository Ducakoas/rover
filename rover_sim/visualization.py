from __future__ import annotations

import base64
import math
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Global visual-only sky combat manager. It never creates collision shapes,
# so the research physics/controller results are unaffected.
_SPACE_COMBAT = None


@dataclass
class CameraConfig:
    # Orbit camera state.
    distance: float = 5.6
    pitch_deg: float = -24.0
    yaw_offset_deg: float = -42.0
    target_height_offset: float = 0.24
    smoothing: float = 0.20

    min_distance: float = 1.15
    max_distance: float = 35.0
    min_pitch_deg: float = -82.0
    max_pitch_deg: float = 20.0

    # Mouse sensitivity: plain LEFT-drag rotates around the rover.
    mouse_yaw_deg_per_px: float = 0.34
    mouse_pitch_deg_per_px: float = 0.26

    # Keyboard backup controls.
    yaw_step_deg: float = 2.5
    pitch_step_deg: float = 2.0
    zoom_factor: float = 0.93


class CameraManager:
    """Interactive orbit camera for the native PyBullet window.

    V12 controls
    ------------
    LEFT MOUSE DRAG  : orbit around rover (no Ctrl/Alt required)
    Mouse wheel      : zoom in/out (native PyBullet zoom is preserved)
    Left/Right arrow : orbit
    Up/Down arrow    : raise/lower viewing angle
    U / O            : zoom in/out
    R                : reset chase view

    ``enabled`` only controls whether the TARGET follows the rover.  Camera
    yaw/pitch/distance remain interactive in both follow and free modes.
    """

    def __init__(self, pb, cfg: CameraConfig | None = None):
        self.pb = pb
        self.cfg = cfg or CameraConfig()
        self.target: Optional[np.ndarray] = None
        self.enabled = True

        self._initialized = False
        self._yaw: float | None = None
        self._pitch: float | None = None
        self._distance: float | None = None

        # Custom mouse orbit state.  PyBullet's default GUI normally needs
        # Ctrl+left-drag; this controller intentionally makes plain left-drag
        # work so it behaves like a normal 3-D model viewer.
        self._mouse_left_down = False
        self._mouse_last_xy: tuple[float, float] | None = None

    def _read_native_camera(self) -> tuple[float, float, float] | None:
        try:
            info = self.pb.getDebugVisualizerCamera()
            if len(info) >= 11:
                yaw = float(info[8])
                pitch = float(info[9])
                distance = float(info[10])
                if np.isfinite(yaw) and np.isfinite(pitch) and np.isfinite(distance):
                    return yaw, pitch, distance
        except Exception:
            pass
        return None

    @staticmethod
    def _event_down(pb, state: int) -> bool:
        try:
            return bool(int(state) & (int(pb.KEY_IS_DOWN) | int(pb.KEY_WAS_TRIGGERED)))
        except Exception:
            return bool(state)

    @staticmethod
    def _event_released(pb, state: int) -> bool:
        try:
            return bool(int(state) & int(pb.KEY_WAS_RELEASED))
        except Exception:
            return False

    def _key_pressed(self, keys, code: int | None) -> bool:
        if code is None:
            return False
        try:
            state = int(keys.get(int(code), 0))
            return self._event_down(self.pb, state)
        except Exception:
            return False

    def _key_triggered(self, keys, code: int | None) -> bool:
        if code is None:
            return False
        try:
            state = int(keys.get(int(code), 0))
            return bool(state & int(self.pb.KEY_WAS_TRIGGERED))
        except Exception:
            return False

    def _process_keyboard(self, rover_yaw_deg: float) -> None:
        try:
            keys = self.pb.getKeyboardEvents()
        except Exception:
            return

        left = getattr(self.pb, 'B3G_LEFT_ARROW', None)
        right = getattr(self.pb, 'B3G_RIGHT_ARROW', None)
        up = getattr(self.pb, 'B3G_UP_ARROW', None)
        down = getattr(self.pb, 'B3G_DOWN_ARROW', None)

        if self._key_pressed(keys, left):
            self._yaw = float(self._yaw or 0.0) - self.cfg.yaw_step_deg
        if self._key_pressed(keys, right):
            self._yaw = float(self._yaw or 0.0) + self.cfg.yaw_step_deg
        if self._key_pressed(keys, up):
            self._pitch = float(self._pitch or self.cfg.pitch_deg) - self.cfg.pitch_step_deg
        if self._key_pressed(keys, down):
            self._pitch = float(self._pitch or self.cfg.pitch_deg) + self.cfg.pitch_step_deg

        if self._key_pressed(keys, ord('u')) or self._key_pressed(keys, ord('U')):
            self._distance = float(self._distance or self.cfg.distance) * self.cfg.zoom_factor
        if self._key_pressed(keys, ord('o')) or self._key_pressed(keys, ord('O')):
            self._distance = float(self._distance or self.cfg.distance) / self.cfg.zoom_factor

        if self._key_triggered(keys, ord('r')) or self._key_triggered(keys, ord('R')):
            self._yaw = rover_yaw_deg + self.cfg.yaw_offset_deg
            self._pitch = self.cfg.pitch_deg
            self._distance = self.cfg.distance

    def _process_mouse(self) -> None:
        """Turn plain left-button drag into an orbit gesture.

        getMouseEvents returns tuples
        (eventType, mouseX, mouseY, buttonIndex, buttonState).
        We deliberately do not depend on a particular numeric MOUSE_MOVE_EVENT
        constant; button events are identified by buttonIndex and all other
        cursor events can update the drag delta while the button is held.
        """
        try:
            events = self.pb.getMouseEvents()
        except Exception:
            return

        for event in events:
            if len(event) < 5:
                continue

            _, mx, my, button, state = event[:5]
            mx = float(mx)
            my = float(my)
            button = int(button)
            state = int(state)

            # In Bullet the left mouse button is index 0.
            if button == 0:
                if self._event_released(self.pb, state):
                    self._mouse_left_down = False
                    self._mouse_last_xy = None
                elif self._event_down(self.pb, state):
                    self._mouse_left_down = True
                    if self._mouse_last_xy is None:
                        self._mouse_last_xy = (mx, my)

            # Bullet uses eventType==1 for mouse motion in the standard GUI.
            # Use the exported constant when available, otherwise fall back to 1.
            move_event = int(getattr(self.pb, 'MOUSE_MOVE_EVENT', 1))
            is_move = int(event[0]) == move_event

            # Some builds report buttonIndex=-1 for motion; others retain the
            # active button index.  Therefore use eventType rather than button.
            if self._mouse_left_down and is_move:
                if self._mouse_last_xy is not None:
                    dx = mx - self._mouse_last_xy[0]
                    dy = my - self._mouse_last_xy[1]

                    if abs(dx) < 500.0 and abs(dy) < 500.0:
                        self._yaw = float(self._yaw or 0.0) + dx * self.cfg.mouse_yaw_deg_per_px
                        self._pitch = float(self._pitch or self.cfg.pitch_deg) + dy * self.cfg.mouse_pitch_deg_per_px

                self._mouse_last_xy = (mx, my)

    def update(self, rover, force: bool = False) -> None:
        pos, rpy, _ = rover.pose()
        rover_yaw_deg = math.degrees(float(rpy[2]))

        wanted_target = np.asarray(
            [float(pos[0]), float(pos[1]), float(pos[2]) + self.cfg.target_height_offset],
            dtype=float,
        )

        if self.target is None or force:
            self.target = wanted_target
        elif self.enabled:
            a = float(np.clip(self.cfg.smoothing, 0.01, 1.0))
            self.target = (1.0 - a) * self.target + a * wanted_target

        if force or not self._initialized:
            self._yaw = rover_yaw_deg + self.cfg.yaw_offset_deg
            self._pitch = self.cfg.pitch_deg
            self._distance = self.cfg.distance
            self._initialized = True
        else:
            # Preserve ONLY native wheel zoom.  Yaw/pitch are owned by the
            # custom orbit controller so Bullet's object-picking mouse mode
            # cannot overwrite them.
            native = self._read_native_camera()
            if native is not None:
                _, _, native_distance = native
                if np.isfinite(native_distance):
                    self._distance = float(native_distance)

        self._process_mouse()
        self._process_keyboard(rover_yaw_deg)

        self._distance = float(np.clip(
            float(self._distance if self._distance is not None else self.cfg.distance),
            self.cfg.min_distance,
            self.cfg.max_distance,
        ))
        self._pitch = float(np.clip(
            float(self._pitch if self._pitch is not None else self.cfg.pitch_deg),
            self.cfg.min_pitch_deg,
            self.cfg.max_pitch_deg,
        ))
        self._yaw = float(
            self._yaw if self._yaw is not None
            else rover_yaw_deg + self.cfg.yaw_offset_deg
        )

        try:
            self.pb.resetDebugVisualizerCamera(
                cameraDistance=self._distance,
                cameraYaw=self._yaw,
                cameraPitch=self._pitch,
                cameraTargetPosition=self.target.tolist(),
            )
        except Exception:
            pass

        # Animate the purely decorative alien sky battle.  The manager is
        # created by configure_clean_gui() and throttles itself internally.
        global _SPACE_COMBAT
        if _SPACE_COMBAT is not None:
            try:
                _SPACE_COMBAT.update()
            except Exception:
                # Visual decoration must never interrupt rover physics.
                pass

def _write_bmp_rgb(path: str, rgb: np.ndarray) -> None:
    """Write a 24-bit BMP using only the standard library + NumPy."""
    import struct

    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("RGB image must have shape HxWx3")

    h, w, _ = arr.shape
    row_bytes = w * 3
    pad = (4 - (row_bytes % 4)) % 4
    pixel_bytes = (row_bytes + pad) * h
    file_size = 54 + pixel_bytes

    header = bytearray()
    header += b"BM"
    header += struct.pack("<IHHI", file_size, 0, 0, 54)
    header += struct.pack("<IIIHHIIIIII", 40, w, h, 1, 24, 0, pixel_bytes, 2835, 2835, 0, 0)

    with open(path, "wb") as fh:
        fh.write(header)
        # BMP is bottom-up and BGR.
        for row in arr[::-1]:
            fh.write(row[:, ::-1].tobytes())
            if pad:
                fh.write(b"\x00" * pad)


def _earth_texture(width: int = 512, height: int = 256) -> np.ndarray:
    """Stylized procedural Earth equirectangular texture."""
    lon = np.linspace(-np.pi, np.pi, width, endpoint=False)[None, :]
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, height)[:, None]

    # Ocean gradient: dark at poles, bright cyan toward illuminated mid-latitudes.
    light = 0.64 + 0.22 * np.cos(lat) + 0.08 * np.cos(lon - 0.8)
    r = 10.0 + 10.0 * light
    g = 58.0 + 55.0 * light
    b = 118.0 + 92.0 * light

    # Pseudo-continent field; intentionally cinematic rather than geographic.
    field = (
        np.sin(1.25 * lon + 0.35 * np.sin(2.0 * lat))
        + 0.72 * np.sin(2.7 * lon - 1.6 * lat + 0.6)
        + 0.45 * np.cos(4.8 * lon + 2.3 * lat - 0.3)
        + 0.28 * np.sin(7.3 * lon - 1.1 * lat)
    )
    continent = (field > 0.78) & (np.abs(lat) < 1.20)
    coast = (field > 0.61) & (field <= 0.78) & (np.abs(lat) < 1.25)

    green = 0.55 + 0.30 * np.cos(2.4 * lat) + 0.12 * np.sin(3.1 * lon)
    r = np.where(continent, 52.0 + 70.0 * green, r)
    g = np.where(continent, 92.0 + 82.0 * green, g)
    b = np.where(continent, 44.0 + 30.0 * green, b)
    r = np.where(coast, 112.0, r)
    g = np.where(coast, 132.0, g)
    b = np.where(coast, 72.0, b)

    # Ice caps.
    ice = np.abs(lat) > 1.20
    ice_strength = np.clip((np.abs(lat) - 1.20) / 0.36, 0.0, 1.0)
    r = np.where(ice, 185.0 + 60.0 * ice_strength, r)
    g = np.where(ice, 210.0 + 42.0 * ice_strength, g)
    b = np.where(ice, 225.0 + 30.0 * ice_strength, b)

    # Wispy clouds.
    clouds = (
        np.sin(5.2 * lon + 1.5 * lat + 0.8)
        + 0.65 * np.cos(8.8 * lon - 2.1 * lat)
        + 0.35 * np.sin(13.0 * lon + 3.6 * lat)
    ) > 1.32
    r = np.where(clouds, 0.70 * r + 0.30 * 245.0, r)
    g = np.where(clouds, 0.70 * g + 0.30 * 248.0, g)
    b = np.where(clouds, 0.70 * b + 0.30 * 252.0, b)

    return np.clip(np.dstack([r, g, b]), 0, 255).astype(np.uint8)


def _sun_texture(width: int = 384, height: int = 192) -> np.ndarray:
    lon = np.linspace(-np.pi, np.pi, width, endpoint=False)[None, :]
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, height)[:, None]
    turbulence = (
        0.55 * np.sin(7.0 * lon + 2.0 * lat)
        + 0.30 * np.sin(15.0 * lon - 5.0 * lat + 0.4)
        + 0.15 * np.cos(29.0 * lon + 8.0 * lat)
    )
    hot = 0.5 + 0.5 * turbulence
    r = 245.0 + 10.0 * hot
    g = 112.0 + 105.0 * hot
    b = 18.0 + 40.0 * hot
    return np.clip(np.dstack([r, g, b]), 0, 255).astype(np.uint8)


def _make_texture(pb, rgb: np.ndarray, prefix: str) -> int | None:
    try:
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=".bmp")
        import os
        os.close(fd)
        _write_bmp_rgb(path, rgb)
        return int(pb.loadTexture(path))
    except Exception:
        return None


def _spawn_visual_sphere(pb, radius: float, rgba, position, texture_id: int | None = None) -> int:
    vis = pb.createVisualShape(pb.GEOM_SPHERE, radius=float(radius), rgbaColor=list(rgba))
    bid = pb.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=list(position),
    )
    if texture_id is not None and texture_id >= 0:
        try:
            pb.changeVisualShape(bid, -1, textureUniqueId=texture_id, rgbaColor=[1.0, 1.0, 1.0, float(rgba[3])])
        except Exception:
            pass
    return int(bid)


def _spawn_space_background(pb, *, seed: int = 9137, star_count: int = 190) -> list[int]:
    """Visual-only cinematic Mars sky scenery.

    The V11 objects were around 60--100 m from the origin and tiny, which made
    them effectively disappear in the normal rover camera.  V12 keeps the sky
    objects inside a ~20--45 m shell and makes the major celestial bodies large
    enough to be visible while orbiting the rover.
    """
    rng = np.random.default_rng(seed)
    ids: list[int] = []

    try:
        # Brighter/larger reusable stars.  All are visual-only.
        star_shapes = [
            pb.createVisualShape(pb.GEOM_SPHERE, radius=0.11, rgbaColor=[0.92, 0.96, 1.00, 1.0]),
            pb.createVisualShape(pb.GEOM_SPHERE, radius=0.14, rgbaColor=[0.60, 0.78, 1.00, 1.0]),
            pb.createVisualShape(pb.GEOM_SPHERE, radius=0.13, rgbaColor=[1.00, 0.84, 0.62, 1.0]),
            pb.createVisualShape(pb.GEOM_SPHERE, radius=0.18, rgbaColor=[1.00, 1.00, 0.94, 1.0]),
        ]

        # Hemisphere of stars around the whole 24 m route.
        for _ in range(int(star_count)):
            az = rng.uniform(-np.pi, np.pi)
            elev = rng.uniform(np.deg2rad(12.0), np.deg2rad(78.0))
            radius = rng.uniform(28.0, 44.0)
            x = 2.0 + radius * np.cos(elev) * np.cos(az)
            y = radius * np.cos(elev) * np.sin(az)
            z = 4.0 + radius * np.sin(elev)
            vis = star_shapes[int(rng.integers(0, len(star_shapes)))]
            ids.append(pb.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=[float(x), float(y), float(z)],
            ))

        # Milky-way-like arc: denser bright band.
        band_vis = pb.createVisualShape(
            pb.GEOM_SPHERE,
            radius=0.095,
            rgbaColor=[0.72, 0.79, 1.0, 0.95],
        )
        for _ in range(120):
            az = rng.uniform(-2.9, 2.9)
            elev = np.deg2rad(36.0 + 12.0 * np.sin(0.72 * az) + rng.normal(0.0, 3.0))
            radius = rng.uniform(31.0, 43.0)
            x = 2.0 + radius * np.cos(elev) * np.cos(az)
            y = radius * np.cos(elev) * np.sin(az)
            z = 5.0 + radius * np.sin(elev)
            ids.append(pb.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=band_vis,
                basePosition=[float(x), float(y), float(z)],
            ))

        # EARTH: deliberately close enough to be obvious in the rover viewer.
        # Position is above and to the side of the route so it appears when the
        # user orbits the camera rather than being hidden behind the terrain.
        earth_tex = _make_texture(pb, _earth_texture(768, 384), 'rover_earth_')
        earth_pos = [8.0, 17.0, 15.5]
        ids.append(_spawn_visual_sphere(pb, 5.2, [0.08, 0.30, 0.92, 1.0], earth_pos, earth_tex))
        ids.append(_spawn_visual_sphere(pb, 5.45, [0.25, 0.62, 1.00, 0.13], earth_pos, None))
        ids.append(_spawn_visual_sphere(pb, 5.75, [0.25, 0.58, 1.00, 0.045], earth_pos, None))

        # Moon placed near Earth for depth/composition.
        ids.append(_spawn_visual_sphere(
            pb, 0.90, [0.68, 0.70, 0.74, 1.0], [18.8, 16.7, 19.5], None
        ))

        # SUN on the opposite side of the route with layered corona.
        sun_tex = _make_texture(pb, _sun_texture(512, 256), 'rover_sun_')
        sun_pos = [-4.0, -18.0, 18.0]
        ids.append(_spawn_visual_sphere(pb, 4.0, [1.00, 0.48, 0.06, 1.0], sun_pos, sun_tex))
        ids.append(_spawn_visual_sphere(pb, 4.35, [1.00, 0.62, 0.10, 0.14], sun_pos, None))
        ids.append(_spawn_visual_sphere(pb, 4.9, [1.00, 0.37, 0.04, 0.055], sun_pos, None))
        ids.append(_spawn_visual_sphere(pb, 5.8, [1.00, 0.28, 0.02, 0.020], sun_pos, None))

        # Small distant planet to make the scene less empty.
        ids.append(_spawn_visual_sphere(
            pb, 1.35, [0.35, 0.46, 0.78, 1.0], [-18.0, 19.0, 14.0], None
        ))

    except Exception:
        # Decoration must never break the research simulator.
        return ids

    return ids



# ============================================================================
# CINEMATIC ALIEN SKY BATTLE (VISUAL ONLY)
# ============================================================================

@dataclass
class _CombatActor:
    kind: str
    team: int
    phase: float
    center: np.ndarray
    radius_xy: float
    altitude: float
    angular_speed: float
    body_ids: list[int]
    local_offsets: list[np.ndarray]
    local_eulers: list[np.ndarray]


@dataclass
class _EnergyBolt:
    body_id: int
    active: bool = False
    t0: float = 0.0
    duration: float = 0.55
    start: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    end: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    team: int = 0


class SkyCombatManager:
    """Animated visual-only alien battle above the rover.

    Nothing created here has a collision shape or mass.  It is intentionally
    separated from the research plant so it cannot change wheel contact,
    vehicle dynamics, controller metrics, or the rover arm sequence.
    """

    def __init__(self, pb, seed: int = 2049):
        self.pb = pb
        self.rng = np.random.default_rng(seed)
        self.ids: list[int] = []
        self.actors: list[_CombatActor] = []
        self.bolts: list[_EnergyBolt] = []
        self.flash_ids: list[int] = []
        self.flash_until: list[float] = []
        self.t_start = time.monotonic()
        self.last_update = 0.0
        self.last_shot = 0.0
        self.shot_period = 0.20
        self._spawn_scene()

    @staticmethod
    def _rot_z(v: np.ndarray, yaw: float) -> np.ndarray:
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], dtype=float)

    def _visual_body(self, shape_type, *, rgba, position=(0, 0, -100), **kwargs) -> int:
        vis = self.pb.createVisualShape(shape_type, rgbaColor=list(rgba), **kwargs)
        bid = self.pb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=list(position),
        )
        self.ids.append(int(bid))
        return int(bid)

    def _spawn_alien_beast(self, team: int, phase: float, center, radius_xy: float, altitude: float, angular_speed: float):
        # Team 0: cyan/teal biomechanical creatures.
        # Team 1: magenta/orange creatures.
        if team == 0:
            body_c = [0.09, 0.72, 0.60, 1.0]
            wing_c = [0.05, 0.28, 0.32, 1.0]
            eye_c = [0.35, 1.00, 0.95, 1.0]
        else:
            body_c = [0.72, 0.15, 0.42, 1.0]
            wing_c = [0.36, 0.06, 0.18, 1.0]
            eye_c = [1.00, 0.58, 0.10, 1.0]

        bodies: list[int] = []
        offsets: list[np.ndarray] = []
        eulers: list[np.ndarray] = []

        # Torso / head.
        bodies.append(self._visual_body(self.pb.GEOM_SPHERE, radius=0.48, rgba=body_c))
        offsets.append(np.array([0.0, 0.0, 0.0]))
        eulers.append(np.zeros(3))

        bodies.append(self._visual_body(self.pb.GEOM_SPHERE, radius=0.32, rgba=body_c))
        offsets.append(np.array([0.55, 0.0, 0.13]))
        eulers.append(np.zeros(3))

        # Glowing eyes.
        for sy in (-1.0, +1.0):
            bodies.append(self._visual_body(self.pb.GEOM_SPHERE, radius=0.085, rgba=eye_c))
            offsets.append(np.array([0.80, sy * 0.15, 0.18]))
            eulers.append(np.zeros(3))

        # Wide insect-like wings.
        for sy in (-1.0, +1.0):
            bodies.append(self._visual_body(
                self.pb.GEOM_BOX,
                halfExtents=[0.44, 0.12, 0.055],
                rgba=wing_c,
            ))
            offsets.append(np.array([-0.05, sy * 0.55, 0.08]))
            eulers.append(np.array([0.0, sy * 0.26, sy * 0.18]))

        # Two trailing tentacles/cannons.
        for sy in (-1.0, +1.0):
            bodies.append(self._visual_body(
                self.pb.GEOM_CAPSULE,
                radius=0.075,
                length=0.68,
                rgba=wing_c,
            ))
            offsets.append(np.array([-0.48, sy * 0.24, -0.18]))
            eulers.append(np.array([0.0, math.pi / 2.0, sy * 0.18]))

        self.actors.append(_CombatActor(
            kind='beast', team=team, phase=float(phase), center=np.asarray(center, dtype=float),
            radius_xy=float(radius_xy), altitude=float(altitude), angular_speed=float(angular_speed),
            body_ids=bodies, local_offsets=offsets, local_eulers=eulers,
        ))

    def _spawn_ufo(self, team: int, phase: float, center, radius_xy: float, altitude: float, angular_speed: float):
        if team == 0:
            hull = [0.16, 0.54, 0.70, 1.0]
            glow = [0.25, 0.95, 1.00, 1.0]
        else:
            hull = [0.52, 0.16, 0.20, 1.0]
            glow = [1.00, 0.35, 0.06, 1.0]

        bodies: list[int] = []
        offsets: list[np.ndarray] = []
        eulers: list[np.ndarray] = []

        bodies.append(self._visual_body(
            self.pb.GEOM_CYLINDER,
            radius=0.62,
            length=0.15,
            rgba=hull,
        ))
        offsets.append(np.zeros(3))
        eulers.append(np.zeros(3))

        bodies.append(self._visual_body(self.pb.GEOM_SPHERE, radius=0.30, rgba=[0.33, 0.63, 0.80, 0.88]))
        offsets.append(np.array([0.0, 0.0, 0.18]))
        eulers.append(np.zeros(3))

        for a in np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False):
            bodies.append(self._visual_body(self.pb.GEOM_SPHERE, radius=0.07, rgba=glow))
            offsets.append(np.array([0.48 * math.cos(a), 0.48 * math.sin(a), -0.05]))
            eulers.append(np.zeros(3))

        self.actors.append(_CombatActor(
            kind='ufo', team=team, phase=float(phase), center=np.asarray(center, dtype=float),
            radius_xy=float(radius_xy), altitude=float(altitude), angular_speed=float(angular_speed),
            body_ids=bodies, local_offsets=offsets, local_eulers=eulers,
        ))

    def _spawn_scene(self) -> None:
        # Place both factions around the 24 m route, close enough to be visible
        # but well above the rover and terrain.
        centers = [
            np.array([-3.0,  7.0, 0.0]),
            np.array([ 4.0, -7.5, 0.0]),
            np.array([ 9.0,  5.5, 0.0]),
            np.array([-9.0, -5.5, 0.0]),
        ]

        for i in range(4):
            self._spawn_alien_beast(
                team=i % 2,
                phase=0.7 + 1.35 * i,
                center=centers[i],
                radius_xy=4.0 + 0.5 * i,
                altitude=8.0 + 1.1 * i,
                angular_speed=(0.19 + 0.025 * i) * (1.0 if i % 2 == 0 else -1.0),
            )

        for i in range(4):
            self._spawn_ufo(
                team=(i + 1) % 2,
                phase=0.2 + 1.55 * i,
                center=np.array([1.5 * (i - 1.5), (-1.0) ** i * 4.0, 0.0]),
                radius_xy=6.0 + 0.35 * i,
                altitude=11.0 + 0.65 * i,
                angular_speed=(-0.16 - 0.018 * i) * (1.0 if i % 2 == 0 else -1.0),
            )

        # Reusable projectile pool.  Half cyan, half orange.
        for i in range(20):
            team = i % 2
            rgba = [0.20, 1.00, 0.95, 1.0] if team == 0 else [1.00, 0.30, 0.06, 1.0]
            bid = self._visual_body(self.pb.GEOM_SPHERE, radius=0.095, rgba=rgba)
            self.bolts.append(_EnergyBolt(body_id=bid, team=team))

        # Impact flash pool.
        for i in range(8):
            rgba = [0.75, 1.00, 1.00, 0.75] if i % 2 == 0 else [1.00, 0.52, 0.08, 0.80]
            bid = self._visual_body(self.pb.GEOM_SPHERE, radius=0.28, rgba=rgba)
            self.flash_ids.append(bid)
            self.flash_until.append(0.0)

    def _actor_pose(self, actor: _CombatActor, t: float) -> tuple[np.ndarray, float, float, float]:
        a = actor.phase + actor.angular_speed * t
        # Elliptical flight path + bobbing so the sky feels alive.
        x = actor.center[0] + actor.radius_xy * math.cos(a)
        y = actor.center[1] + 0.72 * actor.radius_xy * math.sin(a)
        z = actor.altitude + 0.75 * math.sin(1.7 * a + actor.phase)
        dx = -actor.radius_xy * actor.angular_speed * math.sin(a)
        dy = 0.72 * actor.radius_xy * actor.angular_speed * math.cos(a)
        yaw = math.atan2(dy, dx)
        pitch = 0.10 * math.sin(1.9 * a)
        roll = 0.26 * math.sin(1.3 * a + actor.phase)
        return np.array([x, y, z], dtype=float), yaw, pitch, roll

    def _apply_actor_pose(self, actor: _CombatActor, pos: np.ndarray, yaw: float, pitch: float, roll: float, t: float) -> None:
        # Light wing/tentacle flutter for monsters; UFOs remain rigid.
        for j, (bid, off, eul) in enumerate(zip(actor.body_ids, actor.local_offsets, actor.local_eulers)):
            local = np.array(eul, dtype=float)
            if actor.kind == 'beast' and j >= 4:
                local[0] += 0.22 * math.sin(5.0 * t + actor.phase + j)
            world_off = self._rot_z(off, yaw)
            p = pos + world_off
            q = self.pb.getQuaternionFromEuler([
                float(local[0] + roll),
                float(local[1] + pitch),
                float(local[2] + yaw),
            ])
            self.pb.resetBasePositionAndOrientation(bid, p.tolist(), q)

    def _spawn_shot(self, now: float, poses: list[tuple[np.ndarray, float, float, float]]) -> None:
        # Pick a shooter and a target from opposing factions.
        candidates = list(range(len(self.actors)))
        shooter_idx = int(self.rng.choice(candidates))
        shooter_team = self.actors[shooter_idx].team
        targets = [i for i in candidates if self.actors[i].team != shooter_team and i != shooter_idx]
        if not targets:
            return
        target_idx = int(self.rng.choice(targets))

        bolt = next((b for b in self.bolts if not b.active), None)
        if bolt is None:
            # Recycle the oldest active bolt.
            bolt = min(self.bolts, key=lambda b: b.t0)

        start = poses[shooter_idx][0].copy()
        end = poses[target_idx][0].copy()
        # Offset shots slightly above actor center.
        start[2] += 0.1
        end[2] += float(self.rng.uniform(-0.25, 0.25))

        bolt.active = True
        bolt.t0 = now
        bolt.duration = float(self.rng.uniform(0.38, 0.72))
        bolt.start = start
        bolt.end = end
        bolt.team = shooter_team

    def _flash(self, pos: np.ndarray, team: int, now: float) -> None:
        idx = min(range(len(self.flash_until)), key=lambda i: self.flash_until[i])
        self.flash_until[idx] = now + 0.18
        self.pb.resetBasePositionAndOrientation(
            self.flash_ids[idx],
            pos.tolist(),
            [0.0, 0.0, 0.0, 1.0],
        )

    def update(self) -> None:
        now = time.monotonic()
        # 20 Hz is enough for decoration and avoids stealing time from physics.
        if now - self.last_update < 0.05:
            return
        self.last_update = now
        t = now - self.t_start

        poses = [self._actor_pose(actor, t) for actor in self.actors]
        for actor, pose in zip(self.actors, poses):
            self._apply_actor_pose(actor, *pose, t)

        if now - self.last_shot >= self.shot_period:
            self.last_shot = now
            self._spawn_shot(now, poses)

        # Move energy bolts between combatants.
        for bolt in self.bolts:
            if not bolt.active:
                self.pb.resetBasePositionAndOrientation(
                    bolt.body_id, [0.0, 0.0, -100.0], [0.0, 0.0, 0.0, 1.0]
                )
                continue

            u = (now - bolt.t0) / max(bolt.duration, 1e-6)
            if u >= 1.0:
                bolt.active = False
                self._flash(bolt.end, bolt.team, now)
                self.pb.resetBasePositionAndOrientation(
                    bolt.body_id, [0.0, 0.0, -100.0], [0.0, 0.0, 0.0, 1.0]
                )
                continue

            # Curved projectile path with a small vertical arc.
            p = (1.0 - u) * bolt.start + u * bolt.end
            p[2] += 0.55 * math.sin(math.pi * u)
            self.pb.resetBasePositionAndOrientation(
                bolt.body_id, p.tolist(), [0.0, 0.0, 0.0, 1.0]
            )

            # Short luminous tracer line behind the bolt.
            try:
                du = 0.055
                u0 = max(0.0, u - du)
                p0 = (1.0 - u0) * bolt.start + u0 * bolt.end
                p0[2] += 0.55 * math.sin(math.pi * u0)
                color = [0.18, 1.0, 0.92] if bolt.team == 0 else [1.0, 0.24, 0.04]
                self.pb.addUserDebugLine(
                    p0.tolist(), p.tolist(), color,
                    lineWidth=3.0, lifeTime=0.08,
                )
            except Exception:
                pass

        # Hide impact flashes after their short lifetime.
        for i, until in enumerate(self.flash_until):
            if until > 0.0 and now > until:
                self.flash_until[i] = 0.0
                self.pb.resetBasePositionAndOrientation(
                    self.flash_ids[i], [0.0, 0.0, -100.0], [0.0, 0.0, 0.0, 1.0]
                )


def _spawn_space_combat(pb) -> tuple[SkyCombatManager | None, list[int]]:
    try:
        manager = SkyCombatManager(pb)
        return manager, list(manager.ids)
    except Exception:
        return None, []


def configure_clean_gui(pb) -> list[int]:
    """Configure native OpenGL presentation mode and cinematic sky.

    Mouse picking is disabled so a plain left-drag can be reserved for orbiting
    the camera instead of applying forces to the rover.  Bullet's built-in
    keyboard shortcuts are also disabled because J/K/L/O conflict with custom
    camera controls.
    """
    for name in (
        'COV_ENABLE_GUI',
        'COV_ENABLE_RGB_BUFFER_PREVIEW',
        'COV_ENABLE_DEPTH_BUFFER_PREVIEW',
        'COV_ENABLE_SEGMENTATION_MARK_PREVIEW',
        'COV_ENABLE_WIREFRAME',
        'COV_ENABLE_MOUSE_PICKING',
        'COV_ENABLE_KEYBOARD_SHORTCUTS',
    ):
        flag = getattr(pb, name, None)
        if flag is not None:
            try:
                pb.configureDebugVisualizer(flag, 0)
            except Exception:
                pass

    try:
        pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 1)
    except Exception:
        pass

    visual_ids = _spawn_space_background(pb)

    # Spawn animated visual-only alien battle.  The global manager is advanced
    # by CameraManager.update(), so simulation.py does not need to change.
    global _SPACE_COMBAT
    _SPACE_COMBAT, combat_ids = _spawn_space_combat(pb)
    visual_ids.extend(combat_ids)

    return visual_ids

def draw_reference_path(pb, path_xy: np.ndarray, terrain, *, width: float = 2.2) -> list[int]:
    """Legacy Bullet debug path, used only in --viewer bullet mode."""
    ids: list[int] = []
    if len(path_xy) < 2:
        return ids
    for a, b in zip(path_xy[:-1], path_xy[1:]):
        za = terrain.height_at(float(a[0]), float(a[1])) + 0.045
        zb = terrain.height_at(float(b[0]), float(b[1])) + 0.045
        try:
            ids.append(pb.addUserDebugLine(
                [float(a[0]), float(a[1]), za],
                [float(b[0]), float(b[1]), zb],
                [0.15, 0.88, 0.55], width, lifeTime=0.0,
            ))
        except Exception:
            pass
    return ids


def spawn_reference_path_markers(pb, path_xy: np.ndarray, terrain, spacing: int = 3) -> list[int]:
    """Create visual-only path dots that are visible in TinyRenderer."""
    ids: list[int] = []
    if len(path_xy) == 0:
        return ids
    try:
        marker_vis = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.032, rgbaColor=[0.16, 0.95, 0.58, 1.0])
        goal_vis = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.10, rgbaColor=[1.0, 0.72, 0.18, 1.0])
        for pt in path_xy[::max(1, int(spacing))]:
            z = terrain.height_at(float(pt[0]), float(pt[1])) + 0.055
            bid = pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=marker_vis,
                                     basePosition=[float(pt[0]), float(pt[1]), z])
            ids.append(bid)
        goal = path_xy[-1]
        gz = terrain.height_at(float(goal[0]), float(goal[1])) + 0.12
        ids.append(pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=goal_vis,
                                     basePosition=[float(goal[0]), float(goal[1]), gz]))
    except Exception:
        pass
    return ids


class CleanResearchViewer:
    """Standalone presentation viewer using Tkinter + PyBullet TinyRenderer.

    Physics stays in PyBullet DIRECT mode. This avoids the old Bullet ExampleBrowser
    chrome and its RGB/depth/segmentation panes while retaining the same plant.
    """

    CAMERA_MODES = ("CHASE", "WIDE", "TOP")

    def __init__(self, pb, width: int = 640, height: int = 360, title: str = "Mars Rover Research Simulator"):
        try:
            import tkinter as tk
        except Exception as exc:
            raise RuntimeError("Tkinter is unavailable in this Python environment") from exc
        self.tk = tk
        self.pb = pb
        # Prefer Pillow for reliable NumPy -> Tk image transfer on Windows.
        # Tk PhotoImage's raw/base64 PPM handling varies across Tcl/Tk builds.
        try:
            from PIL import Image, ImageTk
            self._PIL_Image = Image
            self._PIL_ImageTk = ImageTk
        except Exception:
            self._PIL_Image = None
            self._PIL_ImageTk = None
        self._fallback_ppm_path = None
        self.width = int(width)
        self.height = int(height)
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="#111316")
        self.root.minsize(900, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.request_quit)

        self._quit = False
        self.paused = False
        self.camera_index = 0
        self._image_ref = None
        self._last_target = None
        self._last_yaw = None
        self._final = False

        top = tk.Frame(self.root, bg="#171a1f", height=58)
        top.pack(fill="x")
        self.title_label = tk.Label(top, text="SIX-WHEEL PLANETARY ROVER", fg="#f1f3f5", bg="#171a1f",
                                    font=("Segoe UI", 15, "bold"), padx=16, pady=10)
        self.title_label.pack(side="left")
        self.case_var = tk.StringVar(value="C3")
        self.case_label = tk.Label(top, textvariable=self.case_var, fg="#66f0ad", bg="#171a1f",
                                   font=("Consolas", 10, "bold"), padx=12)
        self.case_label.pack(side="left")

        controls = tk.Frame(top, bg="#171a1f")
        controls.pack(side="right", padx=12, pady=8)
        self.pause_btn = tk.Button(controls, text="Pause", command=self.toggle_pause, width=9,
                                   bg="#272c33", fg="white", relief="flat", activebackground="#343b44")
        self.pause_btn.pack(side="left", padx=4)
        self.camera_btn = tk.Button(controls, text="Camera", command=self.cycle_camera, width=9,
                                    bg="#272c33", fg="white", relief="flat", activebackground="#343b44")
        self.camera_btn.pack(side="left", padx=4)
        self.quit_btn = tk.Button(controls, text="Quit", command=self.request_quit, width=9,
                                  bg="#5a2428", fg="white", relief="flat", activebackground="#753137")
        self.quit_btn.pack(side="left", padx=4)

        body = tk.Frame(self.root, bg="#111316")
        body.pack(fill="both", expand=True)
        self.image_label = tk.Label(body, bg="#090b0d", bd=0)
        self.image_label.pack(side="left", fill="both", expand=True, padx=(10, 6), pady=10)

        side = tk.Frame(body, bg="#191d22", width=250)
        side.pack(side="right", fill="y", padx=(6, 10), pady=10)
        side.pack_propagate(False)
        tk.Label(side, text="LIVE TELEMETRY", fg="#dfe4ea", bg="#191d22",
                 font=("Segoe UI", 11, "bold"), pady=10).pack(fill="x")
        self.vars = {}
        for key, label in (
            ("scenario", "Scenario"), ("progress", "Path progress"), ("speed", "Speed"),
            ("cte", "Cross-track"), ("slip", "Max |slip|"), ("roll", "Roll"),
            ("pitch", "Pitch"), ("dist", "Goal distance"), ("observer", "FTDO estimate"),
            ("camera", "Camera"), ("status", "Status"),
        ):
            row = tk.Frame(side, bg="#191d22")
            row.pack(fill="x", padx=12, pady=4)
            tk.Label(row, text=label, fg="#8f9aa8", bg="#191d22", font=("Segoe UI", 9)).pack(anchor="w")
            v = tk.StringVar(value="--")
            self.vars[key] = v
            tk.Label(row, textvariable=v, fg="#f3f5f7", bg="#191d22", font=("Consolas", 10, "bold")).pack(anchor="w")

        footer = tk.Frame(self.root, bg="#171a1f")
        footer.pack(fill="x")
        self.footer_var = tk.StringVar(value="P pause/resume   C camera   Q/Esc quit")
        tk.Label(footer, textvariable=self.footer_var, fg="#8f9aa8", bg="#171a1f",
                 font=("Segoe UI", 9), padx=14, pady=6).pack(anchor="w")

        self.root.bind("<Escape>", lambda e: self.request_quit())
        self.root.bind("<KeyPress-q>", lambda e: self.request_quit())
        self.root.bind("<KeyPress-Q>", lambda e: self.request_quit())
        self.root.bind("<KeyPress-p>", lambda e: self.toggle_pause())
        self.root.bind("<KeyPress-P>", lambda e: self.toggle_pause())
        self.root.bind("<KeyPress-c>", lambda e: self.cycle_camera())
        self.root.bind("<KeyPress-C>", lambda e: self.cycle_camera())
        self.root.update_idletasks()
        self.root.update()

    @property
    def closed(self) -> bool:
        return self._quit

    def request_quit(self) -> None:
        self._quit = True

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_btn.configure(text="Resume" if self.paused else "Pause")
        self.vars["status"].set("PAUSED" if self.paused else "RUNNING")

    def cycle_camera(self) -> None:
        self.camera_index = (self.camera_index + 1) % len(self.CAMERA_MODES)
        self.vars["camera"].set(self.CAMERA_MODES[self.camera_index])

    def poll(self) -> None:
        if self._quit:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self._quit = True

    def _camera(self, rover):
        pos, rpy, _ = rover.pose()
        target = np.array([pos[0], pos[1], pos[2] + 0.22], dtype=float)
        yaw = math.degrees(float(rpy[2])) - 38.0
        if self._last_target is None:
            self._last_target = target
            self._last_yaw = yaw
        else:
            a = 0.20
            self._last_target = (1.0 - a) * self._last_target + a * target
            dy = (yaw - float(self._last_yaw) + 180.0) % 360.0 - 180.0
            self._last_yaw = float(self._last_yaw) + a * dy
        mode = self.CAMERA_MODES[self.camera_index]
        if mode == "CHASE":
            dist, pitch, cyaw = 6.1, -21.0, float(self._last_yaw)
        elif mode == "WIDE":
            dist, pitch, cyaw = 10.0, -32.0, float(self._last_yaw) - 14.0
        else:
            dist, pitch, cyaw = 10.5, -82.0, math.degrees(float(rpy[2]))
        return self._last_target.tolist(), dist, cyaw, pitch

    def render(self, rover, telemetry: dict | None = None, force: bool = False) -> None:
        if self._quit:
            return
        target, distance, yaw, pitch = self._camera(rover)
        view = self.pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=0.0,
            upAxisIndex=2,
        )
        proj = self.pb.computeProjectionMatrixFOV(
            fov=54.0,
            aspect=float(self.width) / float(self.height),
            nearVal=0.08,
            farVal=90.0,
        )
        try:
            img = self.pb.getCameraImage(
                self.width,
                self.height,
                viewMatrix=view,
                projectionMatrix=proj,
                renderer=getattr(self.pb, "ER_TINY_RENDERER", 0),
                shadow=0,
                lightDirection=[-0.45, -0.35, -1.0],
                lightColor=[1.0, 0.83, 0.68],
                lightDistance=18.0,
                ambientCoeff=0.42,
                diffuseCoeff=0.68,
                specularCoeff=0.10,
            )
        except TypeError:
            img = self.pb.getCameraImage(self.width, self.height, viewMatrix=view, projectionMatrix=proj,
                                         renderer=getattr(self.pb, "ER_TINY_RENDERER", 0))
        rgba = np.asarray(img[2], dtype=np.uint8).reshape(self.height, self.width, 4)
        rgb = np.ascontiguousarray(rgba[:, :, :3])

        # Robust Windows rendering path.  Pillow avoids the TclError raised by
        # some Anaconda Tcl/Tk builds when binary PPM is supplied through
        # PhotoImage(data=...).
        if self._PIL_Image is not None and self._PIL_ImageTk is not None:
            pil_img = self._PIL_Image.fromarray(rgb, mode="RGB")
            photo = self._PIL_ImageTk.PhotoImage(image=pil_img, master=self.root)
        else:
            # No-Pillow fallback: first try base64 text (not bytes).  If that
            # Tcl/Tk build still rejects it, use a temporary PPM file.
            ppm = f"P6\n{self.width} {self.height}\n255\n".encode("ascii") + rgb.tobytes()
            try:
                encoded = base64.b64encode(ppm).decode("ascii")
                photo = self.tk.PhotoImage(data=encoded, format="PPM", master=self.root)
            except Exception:
                if self._fallback_ppm_path is None:
                    fd, tmp = tempfile.mkstemp(prefix="rover_frame_", suffix=".ppm")
                    try:
                        import os
                        os.close(fd)
                    except Exception:
                        pass
                    self._fallback_ppm_path = tmp
                with open(self._fallback_ppm_path, "wb") as fh:
                    fh.write(ppm)
                photo = self.tk.PhotoImage(file=self._fallback_ppm_path, format="PPM", master=self.root)

        self.image_label.configure(image=photo)
        self._image_ref = photo

        if telemetry:
            self.case_var.set(str(telemetry.get("case", "")))
            self.vars["scenario"].set(str(telemetry.get("scenario", "--")))
            self.vars["progress"].set(f"{100.0*float(telemetry.get('progress', 0.0)):5.1f} %")
            self.vars["speed"].set(f"{float(telemetry.get('speed', 0.0)):+.3f} m/s")
            self.vars["cte"].set(f"{float(telemetry.get('cte', 0.0)):+.3f} m")
            self.vars["slip"].set(f"{float(telemetry.get('slip', 0.0)):.3f}")
            self.vars["roll"].set(f"{float(telemetry.get('roll_deg', 0.0)):+.2f} deg")
            self.vars["pitch"].set(f"{float(telemetry.get('pitch_deg', 0.0)):+.2f} deg")
            self.vars["dist"].set(f"{float(telemetry.get('goal_dist', 0.0)):.2f} m")
            self.vars["observer"].set(f"{float(telemetry.get('dhat', 0.0)):+.3f} m/s²")
            self.vars["camera"].set(self.CAMERA_MODES[self.camera_index])
            if not self._final:
                self.vars["status"].set("PAUSED" if self.paused else "RUNNING")
        self.poll()

    def set_final(self, message: str) -> None:
        self._final = True
        self.vars["status"].set(message)
        self.footer_var.set("Simulation finished. Press Q/Esc or click Quit to close.")
        self.poll()

    def wait_for_close(self) -> None:
        while not self._quit:
            self.poll()
            time.sleep(1.0 / 30.0)
        self.close()

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass
        if self._fallback_ppm_path:
            try:
                import os
                os.remove(self._fallback_ppm_path)
            except Exception:
                pass
            self._fallback_ppm_path = None
        self._quit = True


def wait_for_gui_close(pb, *, message: str = "Simulation complete - press Q/Esc or close the window") -> None:
    """Legacy raw Bullet GUI wait loop."""
    while True:
        try:
            info = pb.getConnectionInfo()
            if not bool(info.get("isConnected", 0)):
                break
            keys = pb.getKeyboardEvents()
            q = keys.get(ord("q"), 0) | keys.get(ord("Q"), 0)
            esc = keys.get(27, 0)
            trigger = getattr(pb, "KEY_WAS_TRIGGERED", 1)
            if (q & trigger) or (esc & trigger):
                break
        except Exception:
            break
        time.sleep(1.0 / 30.0)
