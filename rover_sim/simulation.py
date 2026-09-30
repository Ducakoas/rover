from __future__ import annotations

import math
import time
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import numpy as np

from .allocator import QPTorqueAllocator
from .config import CASE_LABELS, MISSION_LABELS, SCENARIO_LABELS, SimConfig, WHEEL_ORDER
from .controllers import BaselineWheelPI, PurePursuitAckermann, SlipAwareHeuristicAllocator
from .disturbance import LongitudinalDisturbanceProfile
from .logger import RunLogger
from .observer import FixedTimeDisturbanceObserver
from .paths import get_path
from .planner import choose_approach_point, plan_terrain_astar
from .rover import RoverPlant
from .terrain import create_terrain
from .visualization import (
    CameraManager, CleanResearchViewer, configure_clean_gui, draw_reference_path,
    spawn_reference_path_markers, wait_for_gui_close,
)


def _import_pybullet():
    try:
        import pybullet as p
    except ImportError as exc:
        raise RuntimeError(
            "PyBullet is not installed. Recommended: conda install -n rover311 -c conda-forge pybullet numpy scipy pandas matplotlib"
        ) from exc
    return p


def _make_sample(p, pos):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.065)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.065, rgbaColor=[0.86, 0.16, 0.07, 1.0])
    return p.createMultiBody(baseMass=0.25, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)


def _stow_arm(rover: RoverPlant) -> None:
    rover.command_arm_joint_targets([0.0, -0.85, 1.45, 0.0, -0.45, 0.0])
    rover.command_gripper(0.15)


def _angle_delta(a: float, b: float) -> float:
    return float((float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi)


def _observer_nominal_accel(cfg: SimConfig, previous_torque: np.ndarray, terrain, pos, yaw: float) -> float:
    traction = (
        cfg.rover.drive_efficiency
        * float(np.sum(previous_torque))
        / max(cfg.rover.nominal_total_mass * cfg.rover.wheel_radius, 1e-9)
    )
    slope_angle = terrain.slope_along_heading(float(pos[0]), float(pos[1]), float(yaw))
    grade = cfg.mars_gravity * math.sin(slope_angle)
    rolling = 0.0 if abs(traction) < 1e-6 else cfg.control.rolling_accel_nominal * math.copysign(1.0, traction)
    return float(traction - grade - rolling)


def _path_length(path_xy: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)))


def _time_budget(cfg: SimConfig, path_xy: np.ndarray) -> float:
    if cfg.duration > 0.0:
        return float(cfg.duration)
    nominal = _path_length(path_xy) / max(abs(cfg.control.desired_speed), 0.08)
    return float(min(cfg.max_auto_duration_s, cfg.auto_duration_factor * nominal + cfg.auto_duration_margin_s))


def _key_triggered(p, keys, keycode: int) -> bool:
    return bool(keys.get(keycode, 0) & getattr(p, "KEY_WAS_TRIGGERED", 1))


def _draw_goal_marker(p, terrain, goal: np.ndarray):
    z = terrain.height_at(float(goal[0]), float(goal[1]))
    try:
        p.addUserDebugLine([goal[0], goal[1], z + 0.03], [goal[0], goal[1], z + 0.85], [1.0, 0.72, 0.12], 3.0)
        p.addUserDebugText("GOAL", [goal[0], goal[1], z + 0.92], [1.0, 0.85, 0.25], 1.05)
    except Exception:
        pass


def _rock_positions(p, terrain) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for bid in getattr(terrain, "rock_ids", []):
        try:
            pos, _ = p.getBasePositionAndOrientation(bid)
            out.append((float(pos[0]), float(pos[1])))
        except Exception:
            pass
    return out


def _sample_random_target(p, cfg: SimConfig, terrain, start_xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    c = cfg.mission_cfg
    for _ in range(max(1, int(c.target_attempts))):
        x = float(rng.uniform(c.sample_x_min, c.sample_x_max))
        y = float(rng.uniform(c.sample_y_min, c.sample_y_max))
        q = np.array([x, y], dtype=float)
        if float(np.linalg.norm(q - np.asarray(start_xy, dtype=float))) < c.min_sample_distance_m:
            continue
        gx, gy = terrain.gradient_at(x, y)
        slope_deg = math.degrees(math.atan(math.hypot(gx, gy)))
        if slope_deg > c.max_sample_ground_slope_deg:
            continue
        if any(np.linalg.norm(q - np.asarray(o, dtype=float)) < c.planner_obstacle_clearance_m for o in _rock_positions(p, terrain)):
            continue
        return q
    raise RuntimeError("Could not find a valid random sample-drop location on the terrain")


def run_simulation(cfg: SimConfig, path_xy: np.ndarray | None = None, verbose: bool = True) -> Dict[str, Any]:
    p = _import_pybullet()
    if cfg.case not in CASE_LABELS:
        raise ValueError(f"Unknown case {cfg.case}; choose from {list(CASE_LABELS)}")
    if cfg.scenario not in SCENARIO_LABELS:
        raise ValueError(f"Unknown scenario {cfg.scenario}; choose from {list(SCENARIO_LABELS)}")
    if cfg.mission not in MISSION_LABELS:
        raise ValueError(f"Unknown mission {cfg.mission}; choose from {list(MISSION_LABELS)}")

    base_path = get_path(cfg.path_name) if path_xy is None else np.asarray(path_xy, dtype=float)
    path_xy = np.asarray(base_path, dtype=float)
    # V8: native viewer uses PyBullet's OpenGL window directly.  This removes
    # the expensive CPU TinyRenderer -> NumPy -> Pillow -> Tk copy that made
    # earlier custom-viewer versions lag on Windows.
    use_native_gui = bool(cfg.gui and cfg.viewer == "native")
    use_raw_bullet_gui = bool(cfg.gui and cfg.viewer == "bullet")
    use_pybullet_gui = use_native_gui or use_raw_bullet_gui
    use_clean_viewer = bool(cfg.gui and cfg.viewer == "clean")
    mode = p.GUI if use_pybullet_gui else p.DIRECT
    if use_pybullet_gui:
        # Native OpenGL presentation window.  The dark navy background lets the
        # visual-only star field read as deep space while keeping physics and
        # logging completely unchanged.
        gui_options = (
            "--width=1400 --height=850 "
            "--background_color_red=0.004 "
            "--background_color_green=0.006 "
            "--background_color_blue=0.018"
        )
        try:
            cid = p.connect(mode, options=gui_options)
        except TypeError:
            cid = p.connect(mode)
    else:
        cid = p.connect(mode)
    if cid < 0:
        raise RuntimeError("Could not connect to PyBullet")

    video_log = None
    camera = None
    clean_viewer = None
    termination_reason = "unknown"
    goal_reached = False
    try:
        p.resetSimulation()
        p.setGravity(0, 0, -cfg.mars_gravity)
        p.setTimeStep(cfg.dt)
        p.setPhysicsEngineParameter(
            numSolverIterations=int(cfg.solver_iterations),
            fixedTimeStep=cfg.dt,
            enableConeFriction=1,
            deterministicOverlappingPairs=1,
        )
        if use_native_gui:
            # Always hide the ExampleBrowser panels/previews in fast native mode.
            configure_clean_gui(p)
        elif use_raw_bullet_gui and cfg.clean_gui:
            configure_clean_gui(p)

        terrain = create_terrain(p, cfg.scenario, cfg.terrain, seed=cfg.seed)
        start_xy = np.asarray(base_path[0], dtype=float)
        next_xy = np.asarray(base_path[1], dtype=float)
        start_yaw = float(np.arctan2(next_xy[1] - start_xy[1], next_xy[0] - start_xy[0]))
        start_z = terrain.height_at(float(start_xy[0]), float(start_xy[1])) + 0.66
        urdf_path = Path(__file__).resolve().parents[1] / "rover_nasa.urdf"
        rover = RoverPlant(p, urdf_path, [float(start_xy[0]), float(start_xy[1]), start_z], start_yaw, cfg.rover)
        _stow_arm(rover)

        rng = np.random.default_rng(cfg.seed + 2026)
        planner_result = None
        random_target_xy = None
        sample_id = None
        sample_xy = None
        sample_z = 0.0

        if cfg.mission == "path_follow":
            goal = np.asarray(path_xy[-1], dtype=float)
            prev = np.asarray(path_xy[-2], dtype=float)
            goal_heading = math.atan2(float(goal[1] - prev[1]), float(goal[0] - prev[0]))
            goal_forward = np.array([math.cos(goal_heading), math.sin(goal_heading)], dtype=float)
            goal_right = np.array([math.sin(goal_heading), -math.cos(goal_heading)], dtype=float)
            sample_xy = goal + cfg.arm.sample_forward_m * goal_forward + cfg.arm.sample_right_m * goal_right
            sample_z = terrain.height_at(float(sample_xy[0]), float(sample_xy[1])) + 0.075
            if cfg.arm_enabled:
                sample_id = _make_sample(p, [float(sample_xy[0]), float(sample_xy[1]), sample_z])
                p.changeDynamics(sample_id, -1, lateralFriction=0.8, restitution=0.0)
        else:
            # The rover waits at the initial pose. A physical sample is dropped later,
            # then A* computes a new reference online. Until then this tiny path is
            # only a safe tracker placeholder and is never driven.
            path_xy = np.vstack([start_xy, start_xy + np.array([0.30, 0.0])])
            goal = np.asarray(path_xy[-1], dtype=float)
            random_target_xy = _sample_random_target(p, cfg, terrain, start_xy, rng)

        if use_pybullet_gui:
            if cfg.mission == "path_follow" and cfg.draw_path:
                draw_reference_path(p, path_xy, terrain)
            if cfg.mission == "path_follow":
                _draw_goal_marker(p, terrain, goal)
            if sample_id is not None and sample_xy is not None:
                p.addUserDebugText("SAMPLE", [sample_xy[0], sample_xy[1], sample_z + 0.20], [1, 0.3, 0.1], textSize=1.1)
            camera = CameraManager(p)
            camera.enabled = cfg.camera_follow
            camera.update(rover, force=True)
        elif use_clean_viewer:
            if cfg.mission == "path_follow" and cfg.draw_path:
                spawn_reference_path_markers(p, path_xy, terrain, spacing=3)
            clean_viewer = CleanResearchViewer(
                p, width=cfg.viewer_width, height=cfg.viewer_height,
                title="Mars Rover Research Simulator - V9",
            )
            clean_viewer.render(rover, {
                "case": cfg.case, "scenario": cfg.scenario, "progress": 0.0,
                "speed": 0.0, "cte": 0.0, "slip": 0.0,
                "roll_deg": 0.0, "pitch_deg": 0.0,
                "goal_dist": float(np.linalg.norm(goal - start_xy)), "dhat": 0.0,
            }, force=True)

        if cfg.video_path and use_pybullet_gui:
            Path(cfg.video_path).parent.mkdir(parents=True, exist_ok=True)
            video_log = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, cfg.video_path)

        # Settle suspension before activating control.
        rover.command_steering({"LF": 0.0, "LR": 0.0, "RF": 0.0, "RR": 0.0})
        rover.command_wheel_torques(np.zeros(6))
        for settle_step in range(int(cfg.settle_time_s / cfg.dt)):
            rover.command_passive_differential()
            rover.set_spatial_friction(terrain)
            _stow_arm(rover)
            p.stepSimulation()
            if use_clean_viewer and clean_viewer is not None and settle_step % 24 == 0:
                clean_viewer.poll()
                if clean_viewer.closed:
                    termination_reason = "user_quit_during_settle"
                    break
                clean_viewer.render(rover, {
                    "case": cfg.case, "scenario": cfg.scenario, "progress": 0.0,
                    "speed": 0.0, "cte": 0.0, "slip": 0.0,
                    "roll_deg": 0.0, "pitch_deg": 0.0,
                    "goal_dist": float(np.linalg.norm(goal - start_xy)), "dhat": 0.0,
                })
            if cfg.gui and cfg.realtime:
                time.sleep(cfg.dt)
            if camera is not None and settle_step % 20 == 0:
                camera.update(rover)

        # Settled body attitude is the zero reference used for roll/pitch safety
        # and metrics.  With the V9 neutral suspension geometry this should be
        # close to zero, but subtracting the settled reference avoids measuring
        # tiny URDF/mesh alignment offsets as vehicle attitude.
        _, settled_rpy, _ = rover.pose()
        neutral_roll = float(settled_rpy[0])
        neutral_pitch = float(settled_rpy[1])
        actual_mass = float(rover.actual_total_mass())

        tracker = PurePursuitAckermann(path_xy, cfg.rover, cfg.control)
        baseline = BaselineWheelPI(cfg.rover, cfg.control)
        heuristic = SlipAwareHeuristicAllocator(cfg.rover, cfg.control)
        qp = QPTorqueAllocator(cfg.rover, cfg.control, cfg.mars_gravity)
        observer = FixedTimeDisturbanceObserver(
            alpha=cfg.control.ftdo_alpha,
            beta=cfg.control.ftdo_beta,
            a1=cfg.control.ftdo_a1,
            a2=cfg.control.ftdo_a2,
            b1=cfg.control.ftdo_b1,
            b2=cfg.control.ftdo_b2,
            clip=cfg.control.ftdo_clip,
            warmup_s=cfg.control.ftdo_warmup_s,
        )
        disturbance = LongitudinalDisturbanceProfile(cfg.disturbance, enabled=(cfg.scenario == "disturbance"))

        if cfg.mission == "random_sample":
            max_time = float(cfg.duration) if cfg.duration > 0.0 else float(cfg.mission_cfg.random_mission_max_time_s)
        else:
            max_time = _time_budget(cfg, path_xy)

        metadata = {
            "case": cfg.case,
            "case_label": CASE_LABELS[cfg.case],
            "scenario": cfg.scenario,
            "scenario_label": SCENARIO_LABELS[cfg.scenario],
            "mission": cfg.mission,
            "mission_label": MISSION_LABELS[cfg.mission],
            "path_name": cfg.path_name,
            "path_length_m": _path_length(path_xy),
            "max_sim_time_s": max_time,
            "seed": cfg.seed,
            "mars_gravity_mps2": cfg.mars_gravity,
            "actual_urdf_mass_kg": actual_mass,
            "neutral_roll_deg": float(np.rad2deg(neutral_roll)),
            "neutral_pitch_deg": float(np.rad2deg(neutral_pitch)),
            "control": asdict(cfg.control),
            "arm": asdict(cfg.arm),
            "mission_cfg": asdict(cfg.mission_cfg),
            "rover": asdict(cfg.rover),
            "terrain": asdict(cfg.terrain),
            "note": "Rigid PyBullet terrain/contact model; not a deformable-regolith terramechanics model.",
        }
        logger = RunLogger(path_xy, metadata=metadata)

        phase = "drive" if cfg.mission == "path_follow" else "mission_wait_drop"
        phase_t0 = 0.0
        landing_dwell = 0.0
        sample_drop_time_actual = None
        grasp_constraint = None
        grasp_success = False
        previous_torque = np.zeros(6, dtype=float)
        arm_done_hold = 0.0
        goal_dwell = 0.0
        last_progress_print = -1e9
        aborted = False
        paused = False

        if verbose:
            print("=" * 80)
            print("SIX-WHEEL ROCKER-BOGIE ROVER RESEARCH SIMULATION - V11 AUTONOMOUS SAMPLE RETRIEVAL")
            print(f"Case       : {cfg.case} - {CASE_LABELS[cfg.case]}")
            print(f"Scenario   : {cfg.scenario} - {SCENARIO_LABELS[cfg.scenario]}")
            print(f"Mission    : {cfg.mission} - {MISSION_LABELS[cfg.mission]}")
            if cfg.mission == "path_follow":
                print(f"Path       : {cfg.path_name} ({_path_length(path_xy):.2f} m)")
            else:
                print(f"Drop XY    : ({random_target_xy[0]:+.2f}, {random_target_xy[1]:+.2f}) m")
                print("Planner    : terrain-aware 8-connected A* (online after landing)")
            print(f"Seed       : {cfg.seed}")
            print(f"Mars g     : {cfg.mars_gravity:.2f} m/s^2")
            print(f"Target v   : {cfg.control.desired_speed:.2f} m/s")
            print(f"Time limit : {max_time:.1f} s (auto={cfg.duration <= 0})")
            if cfg.gui:
                print(f"Viewer     : {cfg.viewer}")
                if use_clean_viewer:
                    print("GUI keys   : Q/Esc quit, P pause/resume, C change camera")
                elif use_pybullet_gui:
                    print("Viewer mode: interactive native OpenGL + cinematic space background")
                    print("Camera     : CTRL/ALT + mouse drag = orbit/pan/zoom; wheel/touchpad = zoom")
                    print("Keys       : C follow/free camera, P pause/resume, Q/Esc quit")
            print("=" * 80)
            print("[INIT] Entering physics/control loop...", flush=True)

        n_steps = int(max_time / cfg.dt)
        step = 0
        last_track = None
        while step < n_steps:
            t = step * cfg.dt

            if use_clean_viewer and clean_viewer is not None:
                # Tk event processing does not need to run at the 240 Hz physics
                # rate.  30 Hz keeps controls responsive and cuts GUI overhead.
                poll_stride = max(1, int(round(cfg.physics_hz / 30.0)))
                if step % poll_stride == 0:
                    clean_viewer.poll()
                if clean_viewer.closed:
                    termination_reason = "user_quit"
                    aborted = True
                    break
                paused = bool(clean_viewer.paused)
                if paused:
                    time.sleep(1.0 / 30.0)
                    continue
            elif use_pybullet_gui:
                try:
                    if not bool(p.getConnectionInfo().get("isConnected", 0)):
                        termination_reason = "gui_closed"
                        aborted = True
                        break
                    keys = p.getKeyboardEvents()
                    if _key_triggered(p, keys, ord("q")) or _key_triggered(p, keys, ord("Q")) or _key_triggered(p, keys, 27):
                        termination_reason = "user_quit"
                        aborted = True
                        break
                    if _key_triggered(p, keys, ord("p")) or _key_triggered(p, keys, ord("P")):
                        paused = not paused
                    if _key_triggered(p, keys, ord("c")) or _key_triggered(p, keys, ord("C")):
                        if camera is not None:
                            camera.enabled = not camera.enabled
                except Exception:
                    pass
                if paused:
                    if camera is not None:
                        camera.update(rover)
                    time.sleep(1.0 / 30.0)
                    continue

            rover.command_passive_differential()
            d_true = disturbance.apply(p, rover.body, cfg.rover.nominal_total_mass, t)

            if step % cfg.control_stride == 0:
                cdt = cfg.control_dt
                pos, rpy, _ = rover.pose()
                v_body, w_body = rover.body_velocity()
                vx = float(v_body[0])
                roll_rel = _angle_delta(float(rpy[0]), neutral_roll)
                pitch_rel = _angle_delta(float(rpy[1]), neutral_pitch)
                omegas = rover.wheel_omegas()
                slips = rover.slips()
                mus = rover.set_spatial_friction(terrain)
                fz = rover.normal_forces(terrain.body_id)
                contacts = rover.contact_flags(terrain.body_id)
                rocker_l, rocker_r, bogie_l, bogie_r = rover.suspension_angles()

                slope_angle = terrain.slope_along_heading(
                    float(pos[0]), float(pos[1]), float(rpy[2])
                )
                grade_accel = cfg.mars_gravity * math.sin(slope_angle)

                f_nom = _observer_nominal_accel(
                    cfg, previous_torque, terrain, pos, float(rpy[2])
                )
                d_hat = observer.update(vx, f_nom, cdt) if cfg.case in ("C2", "C3") else 0.0

                if phase in ("mission_wait_drop", "mission_wait_land"):
                    # Rover intentionally waits while the sample falls. No low-level
                    # drive controller is allowed to move the vehicle during this phase.
                    v_ref = 0.0
                    track = tracker.update(
                        pos[:2], float(rpy[2]), 0.0,
                        dt=cdt, vx=vx, roll=roll_rel, roll_rate=float(w_body[0]),
                    )
                    last_track = track
                    omega_ref = np.zeros(6, dtype=float)
                    tau = np.zeros(6, dtype=float)
                    rover.hold_wheels()
                    rover.command_steering({"LF": 0.0, "LR": 0.0, "RF": 0.0, "RR": 0.0})
                    previous_torque[:] = 0.0

                    if phase == "mission_wait_drop" and t >= cfg.mission_cfg.drop_time_s:
                        ground_z = terrain.height_at(float(random_target_xy[0]), float(random_target_xy[1]))
                        spawn_z = ground_z + cfg.mission_cfg.drop_height_m
                        sample_id = _make_sample(
                            p,
                            [float(random_target_xy[0]), float(random_target_xy[1]), float(spawn_z)],
                        )
                        p.changeDynamics(sample_id, -1, lateralFriction=0.85, restitution=0.05)
                        sample_drop_time_actual = t
                        landing_dwell = 0.0
                        phase = "mission_wait_land"
                        phase_t0 = t
                        if use_pybullet_gui:
                            try:
                                p.addUserDebugText(
                                    "RANDOM SAMPLE",
                                    [float(random_target_xy[0]), float(random_target_xy[1]), float(spawn_z + 0.35)],
                                    [1.0, 0.25, 0.08],
                                    textSize=1.15,
                                    lifeTime=4.0,
                                )
                            except Exception:
                                pass
                        if verbose:
                            print(
                                f"[{t:6.1f}s] SAMPLE DROP at "
                                f"({random_target_xy[0]:+.2f}, {random_target_xy[1]:+.2f}) m "
                                f"from {cfg.mission_cfg.drop_height_m:.1f} m above terrain"
                            )

                    elif phase == "mission_wait_land" and sample_id is not None:
                        spos, _ = p.getBasePositionAndOrientation(sample_id)
                        sv, _ = p.getBaseVelocity(sample_id)
                        ground_z = terrain.height_at(float(spos[0]), float(spos[1]))
                        close_to_ground = float(spos[2]) <= ground_z + 0.18
                        slow_vertical = abs(float(sv[2])) <= cfg.mission_cfg.landing_speed_threshold_mps
                        landing_dwell = landing_dwell + cdt if (close_to_ground and slow_vertical) else 0.0
                        timed_out = (
                            sample_drop_time_actual is not None
                            and (t - sample_drop_time_actual) >= cfg.mission_cfg.landing_timeout_s
                        )

                        if landing_dwell >= cfg.mission_cfg.landing_dwell_s or timed_out:
                            sample_xy = np.array([float(spos[0]), float(spos[1])], dtype=float)
                            sample_z = float(spos[2])
                            obstacles = _rock_positions(p, terrain)
                            approach_xy = choose_approach_point(
                                terrain,
                                np.asarray(pos[:2], dtype=float),
                                sample_xy,
                                cfg.mission_cfg.approach_distance_m,
                                cfg.mission_cfg.approach_max_slope_deg,
                                obstacle_xy=obstacles,
                                obstacle_clearance_m=cfg.mission_cfg.planner_obstacle_clearance_m,
                            )
                            planner_result = plan_terrain_astar(
                                terrain,
                                np.asarray(pos[:2], dtype=float),
                                approach_xy,
                                resolution_m=cfg.mission_cfg.planner_resolution_m,
                                max_slope_deg=cfg.mission_cfg.planner_max_slope_deg,
                                slope_weight=cfg.mission_cfg.planner_slope_weight,
                                climb_weight=cfg.mission_cfg.planner_climb_weight,
                                obstacle_xy=obstacles,
                                obstacle_clearance_m=cfg.mission_cfg.planner_obstacle_clearance_m,
                                path_spacing_m=cfg.mission_cfg.planned_path_spacing_m,
                                final_heading_target_xy=sample_xy,
                            )
                            path_xy = planner_result.path_xy
                            goal = np.asarray(path_xy[-1], dtype=float)
                            tracker = PurePursuitAckermann(path_xy, cfg.rover, cfg.control)
                            baseline.reset()
                            heuristic.reset()
                            qp.reset()
                            observer.reset()
                            goal_dwell = 0.0
                            logger.path_xy = np.asarray(path_xy, dtype=float)
                            logger.metadata["path_name"] = "terrain_astar_online"
                            logger.metadata["path_length_m"] = float(planner_result.path_length_m)
                            logger.metadata["sample_xy_m"] = [float(sample_xy[0]), float(sample_xy[1])]
                            logger.metadata["approach_xy_m"] = [float(goal[0]), float(goal[1])]
                            logger.metadata["planner_explored_nodes"] = int(planner_result.explored_nodes)
                            logger.metadata["planner_max_slope_deg_on_path"] = float(planner_result.max_slope_deg_on_path)
                            if use_pybullet_gui:
                                if cfg.draw_path:
                                    draw_reference_path(p, path_xy, terrain)
                                _draw_goal_marker(p, terrain, goal)
                                try:
                                    p.addUserDebugText(
                                        "SAMPLE",
                                        [float(sample_xy[0]), float(sample_xy[1]), sample_z + 0.25],
                                        [1.0, 0.25, 0.08],
                                        textSize=1.1,
                                    )
                                except Exception:
                                    pass
                            # IMPORTANT: the rover has been held at zero wheel
                            # speed during sample drop/landing.  Release those
                            # PyBullet velocity motors explicitly before the
                            # newly planned path starts, otherwise the random
                            # mission can remain pinned at progress=0 even
                            # though C0-C3 are producing drive commands.
                            rover.release_wheels()
                            previous_torque[:] = 0.0
                            phase = "drive"
                            phase_t0 = t
                            if verbose:
                                print(f"[{t:6.1f}s] Sample landed -> terrain-aware A* route generated")
                                print(
                                    f"          sample=({sample_xy[0]:+.2f}, {sample_xy[1]:+.2f}) m  "
                                    f"approach=({goal[0]:+.2f}, {goal[1]:+.2f}) m"
                                )
                                print(
                                    f"          A*: explored={planner_result.explored_nodes}, "
                                    f"path={planner_result.path_length_m:.2f} m, "
                                    f"max slope={planner_result.max_slope_deg_on_path:.1f} deg"
                                )

                elif phase == "drive":
                    # Goal taper is computed before the stateful tracker update so
                    # steering-rate limiting is applied exactly once per control tick.
                    dist_goal_geom = float(np.linalg.norm(path_xy[-1] - pos[:2]))
                    v_request = cfg.control.desired_speed * float(np.clip(dist_goal_geom / 1.8, 0.0, 1.0))
                    if dist_goal_geom > cfg.control.stop_radius:
                        v_request = max(cfg.control.min_drive_speed, v_request)
                    else:
                        v_request = 0.0

                    track = tracker.update(
                        pos[:2], float(rpy[2]), v_request,
                        dt=cdt, vx=vx, roll=roll_rel, roll_rate=float(w_body[0]),
                    )
                    last_track = track
                    dist_goal = float(track.distance_to_goal)
                    v_ref = float(track.speed_ref_safe)
                    rover.command_steering(track.steer)
                    omega_ref = np.array([track.omega_ref[w] for w in WHEEL_ORDER], dtype=float)

                    if cfg.case == "C0":
                        tau = baseline.update(
                            omega_ref, omegas, cdt,
                            v_ref=v_ref, vx=vx, grade_accel=grade_accel,
                        )
                    elif cfg.case == "C1":
                        tau = heuristic.update(
                            omega_ref, omegas, cdt,
                            v_ref=v_ref, vx=vx, slips=slips,
                            disturbance_hat=0.0, grade_accel=grade_accel,
                        )
                    elif cfg.case == "C2":
                        tau = heuristic.update(
                            omega_ref, omegas, cdt,
                            v_ref=v_ref, vx=vx, slips=slips,
                            disturbance_hat=d_hat, grade_accel=grade_accel,
                        )
                    else:
                        tau = qp.update(
                            omega_ref, omegas, cdt,
                            v_ref=v_ref, vx=vx, slips=slips, mus=mus, fz=fz,
                            disturbance_hat=d_hat, contacts=contacts,
                            grade_accel=grade_accel,
                        )
                    rover.command_wheel_torques(tau)
                    previous_torque = tau.copy()

                    goal_condition = (
                        track.progress >= cfg.control.goal_progress_threshold
                        and dist_goal <= cfg.control.stop_radius
                        and abs(vx) < 0.07
                    )
                    goal_dwell = goal_dwell + cdt if goal_condition else 0.0
                    if goal_dwell >= cfg.control.goal_dwell_s:
                        goal_reached = True
                        if cfg.arm_enabled:
                            phase = "arm_pregrasp"
                            phase_t0 = t
                            rover.hold_wheels()
                            rover.command_gripper(1.0)
                            previous_torque[:] = 0.0
                            if verbose:
                                print(f"[{t:6.1f}s] Goal reached -> arm pre-grasp")
                        else:
                            phase = "done"
                            phase_t0 = t
                            termination_reason = "goal_reached"
                else:
                    v_ref = 0.0
                    track = tracker.update(
                        pos[:2], float(rpy[2]), 0.0,
                        dt=cdt, vx=vx, roll=roll_rel, roll_rate=float(w_body[0]),
                    )
                    last_track = track
                    omega_ref = np.zeros(6)
                    tau = np.zeros(6)

                    # Do not merely command zero torque on a slope: actively hold
                    # all six wheels while the arm is moving.
                    rover.hold_wheels()
                    previous_torque[:] = 0.0
                    rover.command_steering({"LF": 0.0, "LR": 0.0, "RF": 0.0, "RR": 0.0})

                    if sample_id is not None:
                        spos_now, _ = p.getBasePositionAndOrientation(sample_id)
                        sample_xy = np.array([float(spos_now[0]), float(spos_now[1])], dtype=float)
                        sample_z = float(spos_now[2])
                    sample_target = np.array([sample_xy[0], sample_xy[1], sample_z], dtype=float)

                    if phase == "arm_pregrasp" and sample_id is not None:
                        rover.command_gripper(1.0)
                        target = sample_target + np.array([0.0, 0.0, cfg.arm.pregrasp_height_m])
                        rover.command_arm_ik(target)
                        if (t - phase_t0) >= cfg.arm.pregrasp_time_s:
                            phase = "arm_approach"
                            phase_t0 = t
                            if verbose:
                                print(f"[{t:6.1f}s] Arm pre-grasp -> approach")

                    elif phase == "arm_approach" and sample_id is not None:
                        rover.command_gripper(1.0)
                        target = sample_target + np.array([0.0, 0.0, cfg.arm.approach_height_m])
                        rover.command_arm_ik(target)
                        ee = rover.ee_position()
                        dist_ee = float(np.linalg.norm(ee - sample_target))
                        if (t - phase_t0) >= cfg.arm.approach_time_s:
                            phase = "arm_grasp"
                            phase_t0 = t
                            if verbose:
                                print(f"[{t:6.1f}s] Arm approach -> grasp (EE distance {dist_ee:.3f} m)")

                    elif phase == "arm_grasp" and sample_id is not None:
                        target = sample_target + np.array([0.0, 0.0, cfg.arm.approach_height_m])
                        rover.command_arm_ik(target)
                        rover.command_gripper(0.0)
                        ee = rover.ee_position()
                        dist_ee = float(np.linalg.norm(ee - sample_target))

                        if (t - phase_t0) >= cfg.arm.grasp_time_s:
                            if dist_ee <= cfg.arm.grasp_distance_m and grasp_constraint is None:
                                grasp_constraint = p.createConstraint(
                                    rover.body, rover.ee_link, sample_id, -1,
                                    p.JOINT_FIXED, [0, 0, 0], [0.05, 0, 0], [0, 0, 0],
                                )
                                grasp_success = True
                                if verbose:
                                    print(f"[{t:6.1f}s] Sample grasped (EE distance {dist_ee:.3f} m)")
                            elif verbose and not grasp_success:
                                print(f"[{t:6.1f}s] Grasp fallback: sample outside nominal threshold ({dist_ee:.3f} m)")

                            phase = "arm_lift"
                            phase_t0 = t

                    elif phase == "arm_lift":
                        rover.command_gripper(0.0)
                        target = sample_target + np.array([0.0, 0.0, cfg.arm.lift_height_m])
                        rover.command_arm_ik(target)
                        if (t - phase_t0) >= cfg.arm.lift_time_s:
                            phase = "arm_stow"
                            phase_t0 = t
                            if verbose:
                                print(f"[{t:6.1f}s] Arm lift -> stow")

                    elif phase == "arm_stow":
                        _stow_arm(rover)
                        if (t - phase_t0) >= cfg.arm.stow_time_s:
                            phase = "done"
                            phase_t0 = t
                            termination_reason = (
                                "goal_reached_arm_complete"
                                if grasp_success
                                else "goal_reached_arm_cycle_complete_no_grasp"
                            )
                            if verbose:
                                print(f"[{t:6.1f}s] Arm stowed -> demo complete (grasp={grasp_success})")

                    elif phase == "done":
                        _stow_arm(rover)
                        rover.hold_wheels()
                        arm_done_hold = t - phase_t0

                row = {
                    "t": float(t),
                    "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                    "roll": float(roll_rel), "pitch": float(pitch_rel), "yaw": float(rpy[2]),
                    "raw_roll": float(rpy[0]), "raw_pitch": float(rpy[1]),
                    "vx": float(vx), "vy": float(v_body[1]), "yaw_rate": float(w_body[2]),
                    "v_ref": float(v_ref),
                    "cross_track_error": float(track.cross_track_error),
                    "heading_error": float(track.heading_error),
                    "distance_to_goal": float(track.distance_to_goal),
                    "path_progress": float(track.progress),
                    "curvature_ref": float(track.curvature),
                    "speed_ref_safe": float(track.speed_ref_safe),
                    "lookahead_used": float(track.lookahead_used),
                    "lateral_accel_pred": float(track.lateral_accel_pred),
                    "roll_speed_scale": float(track.roll_scale),
                    "disturbance_true": float(d_true),
                    "disturbance_hat": float(d_hat),
                    "observer_error": float(observer.error if cfg.case in ("C2", "C3") else 0.0),
                    "observer_active": float(cfg.case in ("C2", "C3") and observer.elapsed >= observer.warmup_s),
                    "nominal_accel": float(f_nom),
                    "terrain_slope_rad": float(slope_angle),
                    "grade_accel_mps2": float(grade_accel),
                    "rocker_left": rocker_l, "rocker_right": rocker_r,
                    "bogie_left": bogie_l, "bogie_right": bogie_r,
                    "phase_code": float({
                        "mission_wait_drop": -2,
                        "mission_wait_land": -1,
                        "drive": 0,
                        "arm_pregrasp": 1,
                        "arm_approach": 2,
                        "arm_grasp": 3,
                        "arm_lift": 4,
                        "arm_stow": 5,
                        "done": 6,
                    }[phase]),
                    "grasp_success": float(grasp_success),
                    "qp_cost": float(qp.last_cost if cfg.case == "C3" else 0.0),
                }
                vlong = rover.wheel_longitudinal_speeds()
                for i, w in enumerate(WHEEL_ORDER):
                    row[f"omega_{w}"] = float(omegas[i])
                    row[f"omega_ref_{w}"] = float(omega_ref[i])
                    row[f"vlong_{w}"] = float(vlong[i])
                    row[f"slip_{w}"] = float(slips[i])
                    row[f"tau_{w}"] = float(tau[i])
                    row[f"mu_{w}"] = float(mus[i])
                    row[f"Fz_{w}"] = float(fz[i])
                    row[f"contact_{w}"] = float(contacts[i])
                    row[f"tau_cap_{w}"] = float(qp.last_caps[i] if cfg.case == "C3" else cfg.rover.max_wheel_torque)
                logger.append(row, cdt)

                if verbose and (t - last_progress_print) >= cfg.progress_print_s:
                    last_progress_print = t
                    print(
                        f"[{t:6.1f}s] progress={100*track.progress:5.1f}%  "
                        f"v={vx:+.3f} m/s  CTE={track.cross_track_error:+.3f} m  "
                        f"goal={track.distance_to_goal:.2f} m  max|slip|={np.max(np.abs(slips)):.2f}",
                        flush=True,
                    )

                if camera is not None:
                    camera_stride = max(cfg.control_stride, int(round(cfg.physics_hz / 30.0)))
                    if step % camera_stride == 0:
                        camera.update(rover)
                if clean_viewer is not None:
                    render_stride = max(1, int(round(cfg.physics_hz / max(cfg.viewer_fps, 1.0))))
                    if step % render_stride == 0:
                        clean_viewer.render(rover, {
                            "case": cfg.case, "scenario": cfg.scenario,
                            "progress": float(track.progress), "speed": float(vx),
                            "cte": float(track.cross_track_error),
                            "slip": float(np.max(np.abs(slips))),
                            "roll_deg": float(np.rad2deg(roll_rel)),
                            "pitch_deg": float(np.rad2deg(pitch_rel)),
                            "goal_dist": float(track.distance_to_goal),
                            "dhat": float(d_hat),
                        })

                if phase == "done" and (not cfg.arm_enabled or arm_done_hold > 1.2):
                    if termination_reason == "unknown":
                        termination_reason = "goal_reached"
                    break

                # Detect a catastrophic fall rather than appearing to stop silently.
                terrain_z = terrain.height_at(float(pos[0]), float(pos[1]))
                if float(pos[2]) < terrain_z - 0.8:
                    termination_reason = "safety_abort_below_terrain"
                    aborted = True
                    if verbose:
                        print(f"[{t:6.1f}s] SAFETY ABORT: rover base fell below terrain reference.")
                    break

            p.stepSimulation()
            step += 1
            if cfg.gui and cfg.realtime:
                time.sleep(cfg.dt)

        if termination_reason == "unknown":
            termination_reason = "max_time_before_goal" if not goal_reached else "goal_reached"

        logger.metadata["termination_reason"] = termination_reason
        logger.metadata["goal_reached"] = bool(goal_reached)
        title = f"{cfg.case} / {cfg.scenario} / seed {cfg.seed}"
        metrics = logger.save(Path(cfg.output_dir), title=title)
        metrics["goal_reached"] = bool(goal_reached)
        metrics["termination_reason"] = termination_reason
        try:
            with (Path(cfg.output_dir) / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, allow_nan=True)
        except Exception:
            pass

        if verbose:
            print("\nRun finished")
            print(f"  termination_reason              : {termination_reason}")
            print(f"  goal_reached                    : {goal_reached}")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k:32s}: {v:.6g}")
            print(f"Saved to: {Path(cfg.output_dir).resolve()}")
            if cfg.gui and cfg.keep_gui_open and not aborted:
                print("3D window is being kept open. Press Q/Esc or click Quit to close it.")

        if cfg.gui and cfg.keep_gui_open and not aborted:
            if clean_viewer is not None:
                clean_viewer.set_final("GOAL REACHED" if goal_reached else termination_reason.upper())
                clean_viewer.render(rover, {
                    "case": cfg.case, "scenario": cfg.scenario,
                    "progress": float(last_track.progress if last_track is not None else 0.0),
                    "speed": 0.0,
                    "cte": float(last_track.cross_track_error if last_track is not None else 0.0),
                    "slip": 0.0,
                    "roll_deg": float(np.rad2deg(_angle_delta(rover.pose()[1][0], neutral_roll))),
                    "pitch_deg": float(np.rad2deg(_angle_delta(rover.pose()[1][1], neutral_pitch))),
                    "goal_dist": float(last_track.distance_to_goal if last_track is not None else np.linalg.norm(goal - rover.pose()[0][:2])),
                    "dhat": float(observer.d_hat if hasattr(observer, "d_hat") else 0.0),
                }, force=True)
                clean_viewer.wait_for_close()
            elif use_pybullet_gui:
                if camera is not None:
                    camera.update(rover, force=True)
                wait_for_gui_close(p)
        return metrics
    finally:
        if clean_viewer is not None and clean_viewer.closed:
            try:
                clean_viewer.close()
            except Exception:
                pass
        if video_log is not None:
            try:
                p.stopStateLogging(video_log)
            except Exception:
                pass
        try:
            if p.isConnected():
                p.disconnect()
        except Exception:
            pass
