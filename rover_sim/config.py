from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np

WHEEL_ORDER = ("LF", "LM", "LR", "RF", "RM", "RR")
STEER_ORDER = ("LF", "LR", "RF", "RR")
LEFT_WHEELS = ("LF", "LM", "LR")
RIGHT_WHEELS = ("RF", "RM", "RR")

CASE_LABELS: Dict[str, str] = {
    "C0": "Pure Pursuit + Ackermann + wheel PI baseline",
    "C1": "Slip-aware heuristic torque allocation",
    "C2": "C1 + fixed-time disturbance observer",
    "C3": "FTDO-assisted convex QP torque allocation",
}

SCENARIO_LABELS: Dict[str, str] = {
    "flat": "Flat, homogeneous high-friction terrain",
    "nominal": "Mars-like multi-scale uneven rigid terrain",
    "rough": "Broad climb/descent rough terrain",
    "extreme_rough": "Stress-test broad hills, valleys and oblique ridges",
    "slope": "Uneven uphill terrain",
    "low_mu": "Uniform low-friction terrain",
    "split_mu": "Left/right heterogeneous friction",
    "patchy_mu": "Spatially varying friction patches",
    "rock_field": "Uneven terrain with discrete rocks",
    "disturbance": "Uneven terrain with injected longitudinal disturbance",
}

PATH_LABELS: Dict[str, str] = {
    "s_curve": "S-shaped nominal research path",
    "gentle": "Gentle low-curvature path",
    "hill": "Long path for slope/disturbance tests",
    "straight": "Straight reference path",
    "wide_curve": "Wide smooth reference curve",
}

MISSION_LABELS: Dict[str, str] = {
    "path_follow": "Follow the selected fixed reference path",
    "random_sample": "Random sample drop -> terrain-aware A* -> autonomous retrieval",
}


@dataclass
class RoverConfig:
    wheel_radius: float = 0.215
    wheel_width: float = 0.255
    half_track: float = 0.70
    half_wheelbase: float = 0.95
    chassis_mass: float = 130.0
    nominal_total_mass: float = 228.0
    max_wheel_torque: float = 35.0
    max_steer_angle: float = np.deg2rad(24.0)
    max_steer_force: float = 110.0
    steer_position_gain: float = 0.55
    steer_velocity_gain: float = 0.12
    rocker_diff_k: float = 45.0
    rocker_diff_c: float = 7.0
    wheel_inertia: float = 0.10
    drive_efficiency: float = 0.82
    wheel_hold_force: float = 32.0


@dataclass
class ControlConfig:
    desired_speed: float = 0.22
    lookahead: float = 1.10
    lookahead_min: float = 0.90
    lookahead_max: float = 1.75
    lookahead_speed_gain: float = 1.6
    curvature_filter: float = 0.28
    min_turn_radius_margin: float = 0.28

    max_lateral_accel: float = 0.38
    max_steer_deg: float = 22.0
    max_steer_rate_deg_s: float = 28.0
    min_drive_speed: float = 0.06
    cte_speed_penalty: float = 0.55
    heading_speed_penalty: float = 0.70
    roll_soft_deg: float = 6.0
    roll_hard_deg: float = 15.0
    roll_hard_speed_scale: float = 0.12
    roll_stop_scale_threshold: float = 0.13
    roll_rate_soft_deg_s: float = 8.0
    roll_rate_hard_deg_s: float = 28.0
    roll_rate_hard_speed_scale: float = 0.22
    max_torque_rate: float = 45.0
    max_sync_torque: float = 4.0
    stop_wheel_damping: float = 0.65
    stop_radius: float = 0.75
    goal_dwell_s: float = 1.25
    goal_progress_threshold: float = 0.94

    # Tuned common longitudinal authority after V9 rough-terrain tests.
    grade_feedforward_gain: float = 1.15
    grade_comp_limit_mps2: float = 1.80
    rolling_accel_nominal: float = 0.045

    launch_ref_threshold_mps: float = 0.06
    launch_speed_threshold_mps: float = 0.10
    min_launch_torque_nm: float = 9.0

    wheel_kp: float = 9.0
    wheel_ki: float = 1.60
    wheel_int_limit: float = 18.0

    body_speed_kp: float = 1.60
    body_speed_ki: float = 0.42
    body_speed_int_limit: float = 5.0
    max_command_accel: float = 1.35

    sync_gain: float = 0.35
    slip_threshold: float = 0.12
    slip_penalty_gain: float = 9.0
    min_grip_weight: float = 0.06

    ftdo_alpha: float = 0.75
    ftdo_beta: float = 1.50
    ftdo_a1: float = 3.8
    ftdo_a2: float = 0.08
    ftdo_b1: float = 6.0
    ftdo_b2: float = 0.10
    ftdo_clip: float = 2.0
    ftdo_warmup_s: float = 0.8
    ftdo_comp_gain: float = 0.70
    ftdo_comp_limit_mps2: float = 0.85

    qp_force_weight: float = 8.0
    qp_sync_weight: float = 0.35
    qp_smooth_weight: float = 0.60
    qp_slip_weight: float = 0.08
    qp_slip_growth: float = 28.0
    qp_iterations: int = 16
    qp_step_fraction: float = 0.88
    qp_tolerance: float = 1e-4
    traction_safety: float = 0.90
    airborne_torque_limit: float = 4.0
    contact_hold_s: float = 0.22
    normal_load_filter: float = 0.30


@dataclass
class TerrainConfig:
    size_x: float = 52.0
    size_y: float = 38.0
    grid_n: int = 256
    nominal_roughness: float = 0.025
    rough_roughness: float = 0.040
    crater_depth: float = 0.14
    crater_radius: float = 1.7
    base_friction: float = 0.88
    low_friction: float = 0.30
    split_left_friction: float = 0.30
    split_right_friction: float = 0.92
    patch_min_friction: float = 0.28
    patch_max_friction: float = 0.94
    slope_grade: float = 0.055
    rock_count: int = 28
    decorative_rock_count: int = 42
    texture_repeat: float = 9.0
    seed: int = 7


@dataclass
class DisturbanceConfig:
    pulse1_start: float = 10.0
    pulse1_end: float = 16.0
    pulse1_accel: float = 0.32
    pulse2_start: float = 22.0
    pulse2_end: float = 28.0
    pulse2_accel: float = -0.22
    sine_start: float = 32.0
    sine_end: float = 42.0
    sine_amplitude: float = 0.18
    sine_frequency_hz: float = 0.35


@dataclass
class ArmConfig:
    pregrasp_time_s: float = 2.2
    approach_time_s: float = 2.0
    grasp_time_s: float = 1.0
    lift_time_s: float = 2.0
    stow_time_s: float = 3.0
    pregrasp_height_m: float = 0.42
    approach_height_m: float = 0.10
    lift_height_m: float = 0.52
    grasp_distance_m: float = 0.24
    sample_forward_m: float = 1.30
    sample_right_m: float = 0.18


@dataclass
class MissionConfig:
    # Random-sample retrieval mission.
    drop_time_s: float = 3.0
    drop_height_m: float = 3.0
    landing_timeout_s: float = 6.0
    landing_speed_threshold_mps: float = 0.18
    landing_dwell_s: float = 0.45

    sample_x_min: float = -2.0
    sample_x_max: float = 13.0
    sample_y_min: float = -6.5
    sample_y_max: float = 6.5
    min_sample_distance_m: float = 7.0
    max_sample_ground_slope_deg: float = 12.0
    target_attempts: int = 120

    approach_distance_m: float = 1.15
    approach_max_slope_deg: float = 14.0

    planner_resolution_m: float = 0.45
    planner_max_slope_deg: float = 24.0
    planner_slope_weight: float = 7.0
    planner_climb_weight: float = 2.0
    planner_obstacle_clearance_m: float = 0.75
    planned_path_spacing_m: float = 0.30

    # Extra time budget because the rover waits for the drop/landing before driving.
    random_mission_max_time_s: float = 240.0


@dataclass
class SimConfig:
    physics_hz: int = 240
    control_hz: int = 60
    solver_iterations: int = 100
    duration: float = 0.0
    auto_duration_factor: float = 2.15
    auto_duration_margin_s: float = 28.0
    max_auto_duration_s: float = 240.0
    mars_gravity: float = 3.71
    gui: bool = True
    realtime: bool = True
    viewer: str = "native"
    viewer_width: int = 640
    viewer_height: int = 360
    viewer_fps: float = 10.0
    arm_enabled: bool = False
    camera_follow: bool = True
    clean_gui: bool = True
    keep_gui_open: bool = True
    visual_model: str = "nasa"
    output_dir: Path = Path("results/C3_nominal_seed7")
    video_path: str | None = None
    case: str = "C3"
    scenario: str = "nominal"
    path_name: str = "s_curve"
    mission: str = "path_follow"
    seed: int = 7
    settle_time_s: float = 2.2
    draw_path: bool = True
    draw_hud: bool = False
    progress_print_s: float = 5.0

    rover: RoverConfig = field(default_factory=RoverConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)
    mission_cfg: MissionConfig = field(default_factory=MissionConfig)

    @property
    def dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def control_stride(self) -> int:
        return max(1, int(round(self.physics_hz / self.control_hz)))

    @property
    def control_dt(self) -> float:
        return self.control_stride * self.dt
