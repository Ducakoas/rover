from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import WHEEL_ORDER


class RunLogger:
    def __init__(self, path_xy: np.ndarray, metadata: dict | None = None):
        self.path_xy = np.asarray(path_xy, dtype=float)
        self.rows: List[Dict[str, float]] = []
        self.energy_j = 0.0
        self.distance_m = 0.0
        self.prev_xy: np.ndarray | None = None
        self.prev_tau: np.ndarray | None = None
        self.metadata = metadata or {}

    def append(self, row: Dict[str, float], dt: float) -> None:
        tau = np.array([row.get(f"tau_{w}", 0.0) for w in WHEEL_ORDER], dtype=float)
        omg = np.array([row.get(f"omega_{w}", 0.0) for w in WHEEL_ORDER], dtype=float)
        pwr = float(np.sum(np.abs(tau * omg)))
        self.energy_j += pwr * dt

        xy = np.array([row.get("x", 0.0), row.get("y", 0.0)], dtype=float)
        if self.prev_xy is not None:
            self.distance_m += float(np.linalg.norm(xy - self.prev_xy))
        self.prev_xy = xy

        out = dict(row)
        out["power_abs_W"] = pwr
        out["energy_abs_J"] = float(self.energy_j)
        out["distance_travelled_m"] = float(self.distance_m)

        if self.prev_tau is None:
            rate = np.zeros(6, dtype=float)
        else:
            rate = (tau - self.prev_tau) / max(dt, 1e-9)
        self.prev_tau = tau.copy()

        for i, w in enumerate(WHEEL_ORDER):
            out[f"tau_rate_{w}"] = float(rate[i])
        self.rows.append(out)

    def _arr(self, key: str) -> np.ndarray:
        return np.array([r.get(key, 0.0) for r in self.rows], dtype=float)

    def metrics(self) -> Dict[str, float]:
        if not self.rows:
            return {}

        A = self._arr
        cte = A("cross_track_error")
        he = A("heading_error")
        speed_err = A("v_ref") - A("vx")
        slip = np.column_stack([A(f"slip_{w}") for w in WHEEL_ORDER])
        tau = np.column_stack([A(f"tau_{w}") for w in WHEEL_ORDER])
        tau_rate = np.column_stack([A(f"tau_rate_{w}") for w in WHEEL_ORDER])
        fz = np.column_stack([A(f"Fz_{w}") for w in WHEEL_ORDER])
        contact = np.column_stack([A(f"contact_{w}") for w in WHEEL_ORDER])
        d_true = A("disturbance_true")
        d_hat = A("disturbance_hat")
        warm = A("observer_active") > 0.5

        if np.any(warm):
            d_rmse = float(np.sqrt(np.mean((d_hat[warm] - d_true[warm]) ** 2)))
        else:
            d_rmse = float("nan")

        t = A("t")
        dt_mean = float(np.mean(np.diff(t))) if len(t) >= 2 else 0.0
        cumulative_abs_slip = float(np.mean(np.sum(np.abs(slip), axis=1)) * len(t) * dt_mean)
        specific_energy = float(self.energy_j / max(self.distance_m, 1e-6))

        return {
            "rmse_cross_track_m": float(np.sqrt(np.mean(cte ** 2))),
            "max_abs_cross_track_m": float(np.max(np.abs(cte))),
            "rmse_heading_deg": float(np.rad2deg(np.sqrt(np.mean(he ** 2)))),
            "max_abs_heading_deg": float(np.max(np.abs(np.rad2deg(he)))),
            "rmse_speed_mps": float(np.sqrt(np.mean(speed_err ** 2))),
            "max_abs_slip": float(np.max(np.abs(slip))),
            "rms_slip": float(np.sqrt(np.mean(slip ** 2))),
            "cumulative_abs_slip_s": cumulative_abs_slip,
            "rms_torque_Nm": float(np.sqrt(np.mean(tau ** 2))),
            "max_abs_torque_Nm": float(np.max(np.abs(tau))),
            "max_abs_torque_rate_Nmps": float(np.max(np.abs(tau_rate))),
            "max_normal_load_N": float(np.max(fz)),
            "normal_load_cv": float(np.std(fz) / max(np.mean(fz), 1e-6)),
            "contact_loss_fraction": float(1.0 - np.mean(contact)),
            "max_abs_roll_deg": float(np.max(np.abs(np.rad2deg(A("roll"))))),
            "max_abs_pitch_deg": float(np.max(np.abs(np.rad2deg(A("pitch"))))),
            "rmse_ftdo_vs_injected_mps2": d_rmse,
            "energy_abs_J": float(self.energy_j),
            "specific_energy_J_per_m": specific_energy,
            "distance_travelled_m": float(self.distance_m),
            "goal_distance_m": float(A("distance_to_goal")[-1]),
            "sim_time_s": float(t[-1]),
            "grasp_success": bool(np.max(A("grasp_success")) > 0.5) if "grasp_success" in self.rows[0] else False,
        }

    def save(self, output_dir: Path, title: str = "Rover simulation") -> Dict[str, float]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            return {}

        keys = list(self.rows[0].keys())
        with (output_dir / "timeseries.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(self.rows)

        metrics = self.metrics()
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, allow_nan=True)
        with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        # V9: plotting is optional post-processing.  If the local Windows image
        # stack is broken, users can comment this one call and still retain all
        # CSV/JSON data.  No tight_layout/bbox_inches='tight' is used.
        self._save_plots(output_dir, title)
        return metrics

    def _save_plots(self, output_dir: Path, title: str) -> None:
        A = self._arr
        t = A("t")

        def finish(fig, name: str) -> None:
            fig.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.93)
            fig.savefig(output_dir / name, dpi=180)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.plot(self.path_xy[:, 0], self.path_xy[:, 1], "--", label="Reference path")
        ax.plot(A("x"), A("y"), label="Rover path")
        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")
        ax.set_title(title)
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.legend()
        finish(fig, "fig01_trajectory.png")

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(t, A("cross_track_error"))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Cross-track error [m]")
        ax.grid(True, alpha=0.25)
        finish(fig, "fig02_cross_track_error.png")

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(t, A("vx"), label="Actual")
        ax.plot(t, A("v_ref"), "--", label="Reference")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Longitudinal speed [m/s]")
        ax.grid(True, alpha=0.25)
        ax.legend()
        finish(fig, "fig03_speed.png")

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for w in WHEEL_ORDER:
            ax.plot(t, A(f"slip_{w}"), label=w)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Longitudinal slip ratio [-]")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        finish(fig, "fig04_wheel_slip.png")

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for w in WHEEL_ORDER:
            ax.plot(t, A(f"tau_{w}"), label=w)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Wheel torque [N m]")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        finish(fig, "fig05_wheel_torque.png")

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(t, np.rad2deg(A("roll")), label="Roll from settled reference")
        ax.plot(t, np.rad2deg(A("pitch")), label="Pitch from settled reference")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Attitude [deg]")
        ax.grid(True, alpha=0.25)
        ax.legend()
        finish(fig, "fig06_attitude.png")

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(t, A("disturbance_true"), "--", label="Injected disturbance")
        ax.plot(t, A("disturbance_hat"), label="FTDO estimate")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Disturbance acceleration [m/s²]")
        ax.grid(True, alpha=0.25)
        ax.legend()
        finish(fig, "fig07_disturbance_observer.png")

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for w in WHEEL_ORDER:
            ax.plot(t, A(f"Fz_{w}"), label=w)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Normal contact force [N]")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        finish(fig, "fig08_normal_loads.png")

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for w in WHEEL_ORDER:
            ax.plot(t, A(f"mu_{w}"), label=w)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Assigned friction coefficient [-]")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        finish(fig, "fig09_friction.png")

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(t, A("energy_abs_J"))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Accumulated |Tω| energy proxy [J]")
        ax.grid(True, alpha=0.25)
        finish(fig, "fig10_energy.png")

        plt.close("all")
