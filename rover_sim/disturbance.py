from __future__ import annotations

import math

from .config import DisturbanceConfig


class LongitudinalDisturbanceProfile:
    def __init__(self, cfg: DisturbanceConfig, enabled: bool):
        self.cfg = cfg
        self.enabled = bool(enabled)

    def acceleration(self, t: float) -> float:
        if not self.enabled:
            return 0.0
        c = self.cfg
        if c.pulse1_start <= t < c.pulse1_end:
            return c.pulse1_accel
        if c.pulse2_start <= t < c.pulse2_end:
            return c.pulse2_accel
        if c.sine_start <= t < c.sine_end:
            return c.sine_amplitude * math.sin(2.0 * math.pi * c.sine_frequency_hz * (t - c.sine_start))
        return 0.0

    def apply(self, pb, body_id: int, mass: float, t: float) -> float:
        a = self.acceleration(t)
        if abs(a) > 1e-12:
            pb.applyExternalForce(
                objectUniqueId=body_id,
                linkIndex=-1,
                forceObj=[mass * a, 0.0, 0.0],
                posObj=[0.0, 0.0, 0.0],
                flags=pb.LINK_FRAME,
            )
        return a
