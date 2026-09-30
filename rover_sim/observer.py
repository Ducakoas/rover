from __future__ import annotations

import numpy as np


def sig_pow(x: float, p: float) -> float:
    return float(np.sign(x) * (abs(x) ** p))


class FixedTimeDisturbanceObserver:
    """Two-state fixed-time disturbance observer for longitudinal speed.

    Nominal model:
        vdot_x = f_nom + d

    Observer:
        z0_dot = f_nom + z1 - a1 sig^alpha(e) - a2 sig^beta(e)
        z1_dot = -b1 sig^(2 alpha - 1)(e) - b2 sig^(2 beta - 1)(e)
        e = z0 - v_x

    z1 is used as the lumped disturbance-acceleration estimate. In the PyBullet
    plant this is an approximate observer model, so terrain/contact effects are
    intentionally treated as lumped uncertainty.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.75,
        beta: float = 1.50,
        a1: float = 3.8,
        a2: float = 0.08,
        b1: float = 6.0,
        b2: float = 0.10,
        clip: float = 2.0,
        warmup_s: float = 0.8,
    ):
        if not (0.5 < alpha < 1.0):
            raise ValueError("alpha must be in (0.5, 1)")
        if beta <= 1.0:
            raise ValueError("beta must exceed 1")
        self.alpha = alpha
        self.beta = beta
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.clip = abs(float(clip))
        self.warmup_s = max(0.0, float(warmup_s))
        self.reset()

    def reset(self) -> None:
        self.z0 = 0.0
        self.z1 = 0.0
        self.initialized = False
        self.elapsed = 0.0
        self.error = 0.0

    @property
    def d_hat(self) -> float:
        return float(self.z1 if self.elapsed >= self.warmup_s else 0.0)

    def update(self, measured_vx: float, f_nom: float, dt: float) -> float:
        dt = max(float(dt), 1e-6)
        vx = float(measured_vx)
        if not self.initialized:
            self.z0 = vx
            self.z1 = 0.0
            self.initialized = True
            self.elapsed = 0.0
            return 0.0

        e = float(self.z0 - vx)
        self.error = e
        dz0 = (
            float(f_nom)
            + self.z1
            - self.a1 * sig_pow(e, self.alpha)
            - self.a2 * sig_pow(e, self.beta)
        )
        dz1 = (
            -self.b1 * sig_pow(e, 2.0 * self.alpha - 1.0)
            - self.b2 * sig_pow(e, 2.0 * self.beta - 1.0)
        )
        self.z0 += dt * dz0
        self.z1 += dt * dz1
        self.z1 = float(np.clip(self.z1, -self.clip, self.clip))
        self.elapsed += dt
        return self.d_hat
