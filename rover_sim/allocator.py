from __future__ import annotations

import numpy as np

from .config import ControlConfig, RoverConfig
from .controllers import BodySpeedPI, _common_longitudinal_accel, _launch_floor


class QPTorqueAllocator:
    """Six-wheel convex torque allocator for C3.

    V9 keeps the V8 Windows-stable projected-gradient implementation (no
    LAPACK/eigenvalue calls), but makes the requested longitudinal torque
    physically consistent on slopes:

        a_req = a_speed_feedback + a_grade + a_roll - d_hat

    Grade/rolling feed-forward and the low-speed launch floor are common plant
    accommodations shared with C0-C2.  The C3-specific part remains the
    FTDO-assisted, slip/load-aware box-constrained six-wheel allocation.
    """

    def __init__(self, rover: RoverConfig, control: ControlConfig, mars_gravity: float):
        self.rover = rover
        self.control = control
        self.mars_gravity = float(mars_gravity)
        self.speed = BodySpeedPI(control)
        self.prev = np.zeros(6, dtype=float)
        self.last_tau_req = 0.0
        self.last_accel_cmd = 0.0
        self.last_caps = np.ones(6, dtype=float) * rover.max_wheel_torque
        self.last_cost = 0.0
        self.last_iterations = 0
        self.last_lipschitz = 0.0

        nominal = rover.nominal_total_mass * self.mars_gravity / 6.0
        self._fz_est = np.ones(6, dtype=float) * nominal
        self._air_time = np.zeros(6, dtype=float)

    def reset(self) -> None:
        self.speed.reset()
        self.prev[:] = 0.0
        self.last_tau_req = 0.0
        self.last_accel_cmd = 0.0
        self.last_cost = 0.0
        self.last_iterations = 0
        nominal = self.rover.nominal_total_mass * self.mars_gravity / 6.0
        self._fz_est[:] = nominal
        self._air_time[:] = 0.0

    def _effective_loads(
        self,
        fz: np.ndarray,
        contacts: np.ndarray | None,
        dt: float,
    ) -> np.ndarray:
        c = self.control
        nominal = self.rover.nominal_total_mass * self.mars_gravity / 6.0
        raw = np.nan_to_num(
            np.asarray(fz, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        raw = np.clip(raw, 0.0, 5.0 * nominal)

        if contacts is None:
            contact = raw > max(1.0, 0.02 * nominal)
        else:
            contact = np.asarray(contacts, dtype=float) > 0.5

        a = float(np.clip(c.normal_load_filter, 0.02, 1.0))
        hold_s = max(float(c.contact_hold_s), 1e-3)

        for i in range(6):
            if bool(contact[i]) and raw[i] > 0.5:
                self._air_time[i] = 0.0
                self._fz_est[i] = (1.0 - a) * self._fz_est[i] + a * raw[i]
            else:
                self._air_time[i] += float(dt)
                decay = float(np.exp(-self._air_time[i] / hold_s))
                self._fz_est[i] = max(0.0, self._fz_est[i] * decay)

        return self._fz_est.copy()

    def _caps(self, mus: np.ndarray, fz_eff: np.ndarray) -> np.ndarray:
        c = self.control
        mu = np.nan_to_num(
            np.asarray(mus, dtype=float), nan=0.2, posinf=1.5, neginf=0.02
        )
        mu = np.clip(mu, 0.02, 1.5)
        load = np.maximum(np.asarray(fz_eff, dtype=float), 0.0)

        cap = (
            float(c.traction_safety)
            * mu
            * load
            * float(self.rover.wheel_radius)
        )

        long_air = self._air_time > float(c.contact_hold_s)
        cap[long_air] = np.minimum(
            cap[long_air], float(c.airborne_torque_limit)
        )

        return np.clip(
            cap,
            0.0,
            float(self.rover.max_wheel_torque),
        )

    @staticmethod
    def _objective(
        tau: np.ndarray,
        tau_req: float,
        tau_sync: np.ndarray,
        tau_prev: np.ndarray,
        slip_diag: np.ndarray,
        wf: float,
        ws: float,
        wd: float,
    ) -> float:
        force_err = float(np.sum(tau) - tau_req)
        j_force = wf * force_err * force_err
        j_slip = float(np.sum(slip_diag * tau * tau))
        d_sync = tau - tau_sync
        d_prev = tau - tau_prev
        j_sync = ws * float(np.dot(d_sync, d_sync))
        j_smooth = wd * float(np.dot(d_prev, d_prev))
        return float(j_force + j_slip + j_sync + j_smooth)

    def update(
        self,
        omega_ref: np.ndarray,
        omega: np.ndarray,
        dt: float,
        *,
        v_ref: float,
        vx: float,
        slips: np.ndarray,
        mus: np.ndarray,
        fz: np.ndarray,
        disturbance_hat: float,
        contacts: np.ndarray | None = None,
        grade_accel: float = 0.0,
    ) -> np.ndarray:
        c = self.control
        dt = max(float(dt), 1e-6)

        omega_ref = np.nan_to_num(
            np.asarray(omega_ref, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        omega = np.nan_to_num(
            np.asarray(omega, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        slips = np.nan_to_num(
            np.asarray(slips, dtype=float), nan=0.0, posinf=2.0, neginf=-2.0
        )
        slips = np.clip(slips, -3.0, 3.0)

        a_feedback = self.speed.acceleration_command(float(v_ref), float(vx), dt)
        a_req = _common_longitudinal_accel(
            c,
            a_feedback,
            float(v_ref),
            float(grade_accel),
            float(disturbance_hat),
        )
        self.last_accel_cmd = a_req

        tau_req = (
            float(self.rover.nominal_total_mass)
            * float(self.rover.wheel_radius)
            * a_req
            / max(float(self.rover.drive_efficiency), 1e-6)
        )
        tau_req = float(np.clip(
            tau_req,
            -6.0 * float(self.rover.max_wheel_torque),
            6.0 * float(self.rover.max_wheel_torque),
        ))
        self.last_tau_req = tau_req

        excess = np.maximum(
            np.abs(slips) - float(c.slip_threshold),
            0.0,
        )
        slip_diag = float(c.qp_slip_weight) * (
            1.0 + float(c.qp_slip_growth) * excess * excess
        )
        slip_diag = np.clip(slip_diag, 0.0, 1e4)

        sync_correction = float(c.sync_gain) * (omega_ref - omega)
        sync_correction = np.clip(
            sync_correction,
            -float(c.max_sync_torque),
            float(c.max_sync_torque),
        )
        tau_sync = np.ones(6, dtype=float) * (tau_req / 6.0) + sync_correction

        fz_eff = self._effective_loads(fz, contacts, dt)
        caps = self._caps(mus, fz_eff)
        self.last_caps = caps.copy()

        if float(np.max(caps)) < 1e-8:
            self.prev[:] = 0.0
            self.last_cost = 0.0
            self.last_iterations = 0
            return self.prev.copy()

        tau_prev = np.clip(self.prev, -caps, caps)
        tau = tau_prev.copy()

        wf = max(float(c.qp_force_weight), 0.0)
        ws = max(float(c.qp_sync_weight), 0.0)
        wd = max(float(c.qp_smooth_weight), 0.0)

        # H = 2[wf*11' + diag(slip_diag) + (ws+wd)I].  The analytic
        # Lipschitz bound avoids all LAPACK calls on Windows.
        L = 2.0 * (
            6.0 * wf
            + float(np.max(slip_diag))
            + ws
            + wd
        )
        L = max(L, 1e-6)
        self.last_lipschitz = L
        step_size = float(c.qp_step_fraction) / L

        n_iter = max(1, int(c.qp_iterations))
        tol = max(float(c.qp_tolerance), 0.0)
        self.last_iterations = n_iter

        for k in range(n_iter):
            sum_err = float(np.sum(tau) - tau_req)
            grad = (
                2.0 * wf * sum_err
                + 2.0 * slip_diag * tau
                + 2.0 * ws * (tau - tau_sync)
                + 2.0 * wd * (tau - tau_prev)
            )
            candidate = np.clip(
                tau - step_size * grad,
                -caps,
                caps,
            )

            if tol > 0.0 and float(np.max(np.abs(candidate - tau))) <= tol:
                tau = candidate
                self.last_iterations = k + 1
                break

            tau = candidate

        # Same bounded launch assistance as the other cases.
        # Preserve slip prioritisation even during launch: a wheel already
        # showing excessive slip receives a smaller launch floor.
        launch_scale = 1.0 / (1.0 + float(c.slip_penalty_gain) * excess)
        launch_scale = np.clip(launch_scale, float(c.min_grip_weight), 1.0)
        tau = _launch_floor(
            tau,
            v_ref=float(v_ref),
            vx=float(vx),
            caps=caps,
            control=c,
            floor_scale=launch_scale,
        )

        # Common motor slew-rate limit.  If a traction cap suddenly decreases,
        # the final clip can still reduce torque faster than this rate; that is a
        # safety constraint rather than a commanded torque step.
        max_delta = max(float(c.max_torque_rate), 0.0) * dt
        tau = self.prev + np.clip(
            tau - self.prev,
            -max_delta,
            max_delta,
        )
        tau = np.clip(tau, -caps, caps)
        tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)

        self.last_cost = self._objective(
            tau,
            tau_req,
            tau_sync,
            tau_prev,
            slip_diag,
            wf,
            ws,
            wd,
        )
        self.prev = tau.copy()
        return tau
