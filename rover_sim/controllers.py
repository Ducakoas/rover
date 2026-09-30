from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import ControlConfig, RoverConfig, WHEEL_ORDER


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _segment_projection(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    ab = b - a
    den = float(ab @ ab)
    if den < 1e-12:
        return a.copy(), 0.0
    t = float(np.clip(((p - a) @ ab) / den, 0.0, 1.0))
    return a + t * ab, t


def _rate_limit(target: np.ndarray, previous: np.ndarray, rate: float, dt: float) -> np.ndarray:
    max_step = max(float(rate), 0.0) * max(float(dt), 1e-6)
    return previous + np.clip(target - previous, -max_step, max_step)


@dataclass
class PathTrackingOutput:
    steer: Dict[str, float]
    omega_ref: Dict[str, float]
    alpha: float
    curvature: float
    turn_radius: float
    target: np.ndarray
    nearest_index: int
    cross_track_error: float
    heading_error: float
    distance_to_goal: float
    progress: float
    speed_ref_safe: float
    lookahead_used: float
    lateral_accel_pred: float
    roll_scale: float


class PurePursuitAckermann:
    """Rollover-aware pure-pursuit + symmetric four-wheel Ackermann steering.

    The geometric pure-pursuit concept remains the same as the baseline, but the
    command path is made physically usable on a rocker-bogie rover by adding:
      1) adaptive look-ahead,
      2) curvature smoothing,
      3) steering-angle rate limiting,
      4) curvature/lateral-acceleration speed scheduling,
      5) roll and roll-rate speed protection,
      6) arc-length path progress.

    These safety limits should be treated as common actuator/safety constraints
    for all controller cases, not as the C3 research contribution.
    """

    def __init__(self, path: np.ndarray, rover: RoverConfig, control: ControlConfig):
        self.path = np.asarray(path, dtype=float)
        if self.path.ndim != 2 or self.path.shape[0] < 2 or self.path.shape[1] < 2:
            raise ValueError("path must be an N x 2 array with N >= 2")
        self.path = self.path[:, :2]
        self.rover = rover
        self.control = control
        self._last_nearest = 0

        ds = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self._s_path = np.concatenate(([0.0], np.cumsum(ds)))
        self._path_length = max(float(self._s_path[-1]), 1e-9)

        self._prev_steer = np.zeros(4, dtype=float)  # LF, LR, RF, RR
        self._curvature_f = 0.0

    def reset(self) -> None:
        self._last_nearest = 0
        self._prev_steer[:] = 0.0
        self._curvature_f = 0.0

    def _nearest_and_cte(self, pos_xy: np.ndarray) -> tuple[int, float, float, float]:
        # Search forward aggressively but allow a few points of backtracking.
        start = max(0, self._last_nearest - 4)
        best_dist = float("inf")
        best_idx = start
        best_signed = 0.0
        best_heading = 0.0
        best_t = 0.0
        for i in range(start, len(self.path) - 1):
            a, b = self.path[i], self.path[i + 1]
            proj, t = _segment_projection(pos_xy, a, b)
            dvec = pos_xy - proj
            dist = float(np.linalg.norm(dvec))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                best_t = t
                tangent = b - a
                best_heading = float(np.arctan2(tangent[1], tangent[0]))
                cross_z = tangent[0] * dvec[1] - tangent[1] * dvec[0]
                best_signed = float(np.sign(cross_z) * dist)
            # Once we are well beyond the current local minimum, stop scanning.
            elif i > best_idx + 60 and dist > best_dist + 3.0:
                break
        self._last_nearest = best_idx
        return best_idx, best_signed, best_heading, best_t

    def _lookahead_point(self, pos_xy: np.ndarray, nearest_idx: int, lookahead: float) -> np.ndarray:
        remaining = max(float(lookahead), 0.15)
        current = pos_xy.copy()
        for i in range(nearest_idx, len(self.path) - 1):
            a = current if i == nearest_idx else self.path[i]
            b = self.path[i + 1]
            seg = b - a
            seg_len = float(np.linalg.norm(seg))
            if seg_len >= remaining and seg_len > 1e-9:
                return a + (remaining / seg_len) * seg
            remaining -= seg_len
            current = b
        return self.path[-1].copy()

    def _roll_scale(self, roll: float, roll_rate: float) -> float:
        c = self.control
        r = abs(float(roll))
        rr = abs(float(roll_rate))
        soft = np.deg2rad(c.roll_soft_deg)
        hard = np.deg2rad(c.roll_hard_deg)

        if r <= soft:
            s_roll = 1.0
        elif r >= hard:
            s_roll = c.roll_hard_speed_scale
        else:
            q = (r - soft) / max(hard - soft, 1e-6)
            s_roll = 1.0 - q * (1.0 - c.roll_hard_speed_scale)

        # Fast roll motion is itself a warning even before the angle is large.
        rr_soft = np.deg2rad(c.roll_rate_soft_deg_s)
        rr_hard = np.deg2rad(c.roll_rate_hard_deg_s)
        if rr <= rr_soft:
            s_rate = 1.0
        elif rr >= rr_hard:
            s_rate = c.roll_rate_hard_speed_scale
        else:
            q = (rr - rr_soft) / max(rr_hard - rr_soft, 1e-6)
            s_rate = 1.0 - q * (1.0 - c.roll_rate_hard_speed_scale)
        return float(np.clip(min(s_roll, s_rate), 0.0, 1.0))

    def update(
        self,
        pos_xy: np.ndarray,
        yaw: float,
        v_ref: float,
        *,
        dt: float,
        vx: float = 0.0,
        roll: float = 0.0,
        roll_rate: float = 0.0,
    ) -> PathTrackingOutput:
        c = self.control
        pos_xy = np.asarray(pos_xy, dtype=float)[:2]
        nearest, cte, path_heading, nearest_t = self._nearest_and_cte(pos_xy)

        # Adaptive lookahead: larger at speed, but bounded to avoid corner cutting.
        lookahead = float(np.clip(
            c.lookahead + c.lookahead_speed_gain * abs(float(vx)),
            c.lookahead_min,
            c.lookahead_max,
        ))
        target = self._lookahead_point(pos_xy, nearest, lookahead)
        to_target = target - pos_xy
        target_heading = float(np.arctan2(to_target[1], to_target[0]))
        alpha = wrap_angle(target_heading - yaw)
        ld = max(float(np.linalg.norm(to_target)), 0.30)

        # Pure-pursuit geometric curvature with low-pass filtering.
        curvature_raw = float(2.0 * np.sin(alpha) / ld)
        filt = float(np.clip(c.curvature_filter, 0.0, 1.0))
        self._curvature_f = (1.0 - filt) * self._curvature_f + filt * curvature_raw
        curvature = float(self._curvature_f)

        d = float(self.rover.half_track)
        L = float(self.rover.half_wheelbase)
        min_r = d + float(c.min_turn_radius_margin)
        if abs(curvature) < 1e-6:
            radius = np.sign(curvature if curvature != 0.0 else 1.0) * 1e6
        else:
            radius = 1.0 / curvature
        if abs(radius) < min_r:
            radius = np.sign(radius if radius != 0.0 else 1.0) * min_r
            curvature = 1.0 / radius

        # Curvature-aware speed: bound predicted lateral acceleration v^2 |kappa|.
        kappa_abs = abs(curvature)
        if kappa_abs > 1e-5:
            v_curve = np.sqrt(max(c.max_lateral_accel, 1e-4) / kappa_abs)
        else:
            v_curve = abs(float(v_ref))

        roll_scale = self._roll_scale(roll, roll_rate)
        v_safe_mag = min(abs(float(v_ref)), float(v_curve)) * roll_scale

        # Additional reduction for large path errors to avoid aggressive recovery turns.
        cte_scale = 1.0 / (1.0 + c.cte_speed_penalty * abs(float(cte)))
        heading_scale = 1.0 / (1.0 + c.heading_speed_penalty * abs(wrap_angle(path_heading - yaw)))
        v_safe_mag *= min(cte_scale, heading_scale)

        # Do not enforce a minimum speed when rollover protection is active.
        if abs(v_ref) > 1e-4 and roll_scale > 0.75:
            v_safe_mag = max(min(v_safe_mag, abs(float(v_ref))), min(c.min_drive_speed, abs(float(v_ref))))
        if roll_scale <= c.roll_stop_scale_threshold:
            v_safe_mag = 0.0
        speed_ref_safe = float(np.sign(v_ref if v_ref != 0.0 else 1.0) * v_safe_mag)
        lateral_accel_pred = float(speed_ref_safe**2 * kappa_abs)

        # Four-wheel Ackermann target steering.
        den_l = radius - d
        den_r = radius + d
        den_l = np.copysign(max(abs(den_l), 1e-6), den_l if den_l != 0.0 else radius)
        den_r = np.copysign(max(abs(den_r), 1e-6), den_r if den_r != 0.0 else radius)
        lf = float(np.arctan(L / den_l))
        rf = float(np.arctan(L / den_r))
        max_angle = min(float(self.rover.max_steer_angle), np.deg2rad(c.max_steer_deg))
        lf = float(np.clip(lf, -max_angle, max_angle))
        rf = float(np.clip(rf, -max_angle, max_angle))
        raw = np.array([lf, -lf, rf, -rf], dtype=float)

        # Common actuator steering-rate limit. This is critical on rough terrain.
        limited = _rate_limit(raw, self._prev_steer, np.deg2rad(c.max_steer_rate_deg_s), dt)
        self._prev_steer = limited.copy()
        steer = {"LF": float(limited[0]), "LR": float(limited[1]), "RF": float(limited[2]), "RR": float(limited[3])}

        # Individual wheel angular-speed references consistent with the ICR.
        if abs(radius) > 1e4:
            omega_ref = {w: float(speed_ref_safe / self.rover.wheel_radius) for w in WHEEL_ORDER}
        else:
            rabs = max(abs(radius), 1e-6)
            wheel_xy = {
                "LF": (+L, +d), "LM": (0.0, +d), "LR": (-L, +d),
                "RF": (+L, -d), "RM": (0.0, -d), "RR": (-L, -d),
            }
            omega_ref: Dict[str, float] = {}
            for w, (xw, yw) in wheel_xy.items():
                rw = float(np.hypot(xw, radius - yw))
                wheel_speed = float(speed_ref_safe * rw / rabs)
                omega_ref[w] = wheel_speed / self.rover.wheel_radius

        # Arc-length progress rather than index progress.
        seg_len = float(np.linalg.norm(self.path[min(nearest + 1, len(self.path) - 1)] - self.path[nearest]))
        s_here = float(self._s_path[nearest] + nearest_t * seg_len)
        progress = float(np.clip(s_here / self._path_length, 0.0, 1.0))

        return PathTrackingOutput(
            steer=steer,
            omega_ref=omega_ref,
            alpha=alpha,
            curvature=curvature,
            turn_radius=float(radius),
            target=target,
            nearest_index=nearest,
            cross_track_error=float(cte),
            heading_error=wrap_angle(path_heading - yaw),
            distance_to_goal=float(np.linalg.norm(self.path[-1] - pos_xy)),
            progress=progress,
            speed_ref_safe=speed_ref_safe,
            lookahead_used=lookahead,
            lateral_accel_pred=lateral_accel_pred,
            roll_scale=roll_scale,
        )


class BaselineWheelPI:
    """C0 wheel-speed PI with common grade/rolling feed-forward.

    V8 could produce only about 5 Nm/wheel at launch.  V9 keeps the C0
    architecture as an individual wheel PI baseline, but adds the same physical
    grade/rolling compensation and launch floor used by every controller case.
    These are common plant/actuator accommodations, not a C3 contribution.
    """

    def __init__(self, rover: RoverConfig, control: ControlConfig):
        self.rover = rover
        self.control = control
        self.int_err = np.zeros(6, dtype=float)
        self.prev_tau = np.zeros(6, dtype=float)

    def reset(self) -> None:
        self.int_err[:] = 0.0
        self.prev_tau[:] = 0.0

    def _feedforward_per_wheel(self, v_ref: float, grade_accel: float) -> float:
        c = self.control
        direction = float(np.sign(v_ref)) if abs(v_ref) > 1e-6 else 0.0
        grade = float(np.clip(
            c.grade_feedforward_gain * float(grade_accel),
            -c.grade_comp_limit_mps2,
            c.grade_comp_limit_mps2,
        ))
        rolling = direction * float(c.rolling_accel_nominal)
        tau_total = (
            self.rover.nominal_total_mass
            * self.rover.wheel_radius
            * (grade + rolling)
            / max(self.rover.drive_efficiency, 1e-6)
        )
        return float(tau_total / 6.0)

    def update(
        self,
        omega_ref: np.ndarray,
        omega: np.ndarray,
        dt: float,
        *,
        v_ref: float | None = None,
        vx: float = 0.0,
        grade_accel: float = 0.0,
    ) -> np.ndarray:
        c = self.control
        omega_ref = np.asarray(omega_ref, dtype=float)
        omega = np.asarray(omega, dtype=float)
        err = omega_ref - omega

        if v_ref is None:
            v_ref = float(np.mean(omega_ref) * self.rover.wheel_radius)

        candidate = np.clip(
            self.int_err + err * dt,
            -c.wheel_int_limit,
            c.wheel_int_limit,
        )

        ff = self._feedforward_per_wheel(float(v_ref), float(grade_accel))
        tau_unsat = c.wheel_kp * err + c.wheel_ki * candidate + ff
        tau_sat = np.clip(tau_unsat, -self.rover.max_wheel_torque, self.rover.max_wheel_torque)

        saturated = np.abs(tau_unsat - tau_sat) > 1e-10
        relieving = np.sign(err) != np.sign(tau_unsat - tau_sat)
        use_candidate = (~saturated) | relieving
        self.int_err = np.where(use_candidate, candidate, self.int_err)

        tau_cmd = c.wheel_kp * err + c.wheel_ki * self.int_err + ff

        # Common launch floor: only while a non-zero forward/reverse request is
        # present and the body is nearly stationary.
        if (
            abs(float(v_ref)) >= c.launch_ref_threshold_mps
            and abs(float(vx)) <= c.launch_speed_threshold_mps
        ):
            direction = float(np.sign(v_ref))
            floor = min(float(c.min_launch_torque_nm), float(self.rover.max_wheel_torque))
            same_direction = direction * tau_cmd >= 0.0
            tau_cmd = np.where(
                same_direction,
                direction * np.maximum(np.abs(tau_cmd), floor),
                tau_cmd,
            )

        tau_cmd = np.clip(tau_cmd, -self.rover.max_wheel_torque, self.rover.max_wheel_torque)
        tau = _rate_limit(tau_cmd, self.prev_tau, c.max_torque_rate, dt)
        self.prev_tau = tau.copy()
        return tau


class BodySpeedPI:
    """Body-speed PI with conditional-integration anti-windup."""

    def __init__(self, control: ControlConfig):
        self.control = control
        self.int_v = 0.0

    def reset(self) -> None:
        self.int_v = 0.0

    def acceleration_command(self, v_ref: float, vx: float, dt: float) -> float:
        ev = float(v_ref - vx)
        candidate = float(np.clip(
            self.int_v + ev * dt,
            -self.control.body_speed_int_limit,
            self.control.body_speed_int_limit,
        ))
        a_unsat = self.control.body_speed_kp * ev + self.control.body_speed_ki * candidate
        a_sat = float(np.clip(
            a_unsat,
            -self.control.max_command_accel,
            self.control.max_command_accel,
        ))

        saturated = abs(a_unsat - a_sat) > 1e-10
        relieving = np.sign(ev) != np.sign(a_unsat - a_sat)
        if (not saturated) or relieving:
            self.int_v = candidate
        return a_sat


def _bounded_observer_compensation(control: ControlConfig, disturbance_hat: float) -> float:
    d = float(np.clip(
        float(disturbance_hat),
        -control.ftdo_comp_limit_mps2,
        control.ftdo_comp_limit_mps2,
    ))
    return float(control.ftdo_comp_gain * d)


def _common_longitudinal_accel(
    control: ControlConfig,
    a_feedback: float,
    v_ref: float,
    grade_accel: float,
    disturbance_hat: float,
) -> float:
    grade = float(np.clip(
        control.grade_feedforward_gain * float(grade_accel),
        -control.grade_comp_limit_mps2,
        control.grade_comp_limit_mps2,
    ))
    rolling = 0.0
    if abs(float(v_ref)) > 1e-6:
        rolling = float(control.rolling_accel_nominal) * float(np.sign(v_ref))
    d_comp = _bounded_observer_compensation(control, disturbance_hat)
    return float(a_feedback + grade + rolling - d_comp)


def _launch_floor(
    tau: np.ndarray,
    *,
    v_ref: float,
    vx: float,
    caps: np.ndarray,
    control: ControlConfig,
    floor_scale: np.ndarray | None = None,
) -> np.ndarray:
    tau = np.asarray(tau, dtype=float).copy()
    caps = np.maximum(np.asarray(caps, dtype=float), 0.0)
    if (
        abs(float(v_ref)) < control.launch_ref_threshold_mps
        or abs(float(vx)) > control.launch_speed_threshold_mps
    ):
        return np.clip(tau, -caps, caps)

    direction = float(np.sign(v_ref))
    if floor_scale is None:
        scale = np.ones_like(caps)
    else:
        scale = np.clip(np.asarray(floor_scale, dtype=float), 0.0, 1.0)
        if scale.shape != caps.shape:
            scale = np.ones_like(caps)
    floor = np.minimum(caps, float(control.min_launch_torque_nm) * scale)
    valid = caps > 0.5
    same_direction = direction * tau >= 0.0
    raised = direction * np.maximum(np.abs(tau), floor)
    tau = np.where(valid & same_direction, raised, tau)
    return np.clip(tau, -caps, caps)


class SlipAwareHeuristicAllocator:
    """C1/C2 slip-weighted allocator with grade compensation and launch aid."""

    def __init__(self, rover: RoverConfig, control: ControlConfig):
        self.rover = rover
        self.control = control
        self.speed = BodySpeedPI(control)
        self.prev_tau = np.zeros(6, dtype=float)
        self.last_tau_req = 0.0
        self.last_accel_cmd = 0.0

    def reset(self) -> None:
        self.speed.reset()
        self.prev_tau[:] = 0.0
        self.last_tau_req = 0.0
        self.last_accel_cmd = 0.0

    def update(
        self,
        omega_ref: np.ndarray,
        omega: np.ndarray,
        dt: float,
        *,
        v_ref: float,
        vx: float,
        slips: np.ndarray,
        disturbance_hat: float = 0.0,
        grade_accel: float = 0.0,
    ) -> np.ndarray:
        c = self.control
        omega_ref = np.asarray(omega_ref, dtype=float)
        omega = np.asarray(omega, dtype=float)
        slips = np.nan_to_num(
            np.asarray(slips, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0
        )

        a_feedback = self.speed.acceleration_command(v_ref, vx, dt)
        a_cmd = _common_longitudinal_accel(
            c, a_feedback, v_ref, grade_accel, disturbance_hat
        )
        self.last_accel_cmd = a_cmd

        tau_total = (
            self.rover.nominal_total_mass
            * self.rover.wheel_radius
            * a_cmd
            / max(self.rover.drive_efficiency, 1e-6)
        )
        tau_total = float(np.clip(
            tau_total,
            -6.0 * self.rover.max_wheel_torque,
            6.0 * self.rover.max_wheel_torque,
        ))
        self.last_tau_req = tau_total

        excess = np.maximum(np.abs(slips) - c.slip_threshold, 0.0)
        grip = 1.0 / (1.0 + c.slip_penalty_gain * excess)
        grip = np.clip(grip, c.min_grip_weight, 1.0)
        weights = grip / max(float(np.sum(grip)), 1e-9)

        sync = c.sync_gain * (omega_ref - omega)
        sync = np.clip(sync, -c.max_sync_torque, c.max_sync_torque)
        tau_cmd = tau_total * weights + sync

        if abs(v_ref) < 1e-3:
            tau_cmd += -c.stop_wheel_damping * omega

        caps = np.ones(6, dtype=float) * float(self.rover.max_wheel_torque)
        tau_cmd = _launch_floor(
            tau_cmd, v_ref=v_ref, vx=vx, caps=caps, control=c, floor_scale=grip
        )
        tau_cmd = np.clip(tau_cmd, -caps, caps)
        tau = _rate_limit(tau_cmd, self.prev_tau, c.max_torque_rate, dt)
        self.prev_tau = tau.copy()
        return tau
