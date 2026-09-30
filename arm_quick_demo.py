from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from rover_sim.config import SimConfig
from rover_sim.rover import RoverPlant
from rover_sim.terrain import create_terrain
from rover_sim.visualization import configure_clean_gui, CameraManager


def main() -> None:
    import pybullet as p

    cfg = SimConfig()
    cfg.scenario = "flat"
    cfg.arm_enabled = True

    cid = p.connect(p.GUI)
    if cid < 0:
        raise RuntimeError("Could not open PyBullet GUI")

    try:
        p.resetSimulation()
        p.setGravity(0, 0, -cfg.mars_gravity)
        p.setTimeStep(cfg.dt)
        p.setPhysicsEngineParameter(
            numSolverIterations=cfg.solver_iterations,
            fixedTimeStep=cfg.dt,
            enableConeFriction=1,
            deterministicOverlappingPairs=1,
        )
        configure_clean_gui(p)

        terrain = create_terrain(p, "flat", cfg.terrain, seed=cfg.seed)
        start = np.array([-12.0, 0.0], dtype=float)
        yaw = 0.0
        start_z = terrain.height_at(start[0], start[1]) + 0.66
        urdf = Path(__file__).resolve().parent / "rover_nasa.urdf"
        rover = RoverPlant(p, urdf, [start[0], start[1], start_z], yaw, cfg.rover)

        rover.command_steering({"LF": 0.0, "LR": 0.0, "RF": 0.0, "RR": 0.0})
        rover.command_wheel_torques(np.zeros(6))
        rover.command_arm_joint_targets([0.0, -0.85, 1.45, 0.0, -0.45, 0.0])
        rover.command_gripper(1.0)

        camera = CameraManager(p)
        camera.enabled = True

        for k in range(int(2.2 / cfg.dt)):
            rover.command_passive_differential()
            rover.set_spatial_friction(terrain)
            p.stepSimulation()
            if k % 12 == 0:
                camera.update(rover)
            time.sleep(cfg.dt)

        pos, rpy, _ = rover.pose()
        print(f"Settled rover: roll={math.degrees(rpy[0]):.2f} deg, pitch={math.degrees(rpy[1]):.2f} deg")
        print(f"URDF mass: {rover.actual_total_mass():.2f} kg")

        forward = np.array([math.cos(rpy[2]), math.sin(rpy[2])])
        right = np.array([math.sin(rpy[2]), -math.cos(rpy[2])])
        sample_xy = pos[:2] + 1.25 * forward + 0.10 * right
        sample_z = terrain.height_at(float(sample_xy[0]), float(sample_xy[1])) + 0.075

        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.065)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.065, rgbaColor=[0.86, 0.16, 0.07, 1.0])
        sample = p.createMultiBody(
            baseMass=0.25,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[float(sample_xy[0]), float(sample_xy[1]), sample_z],
        )
        p.changeDynamics(sample, -1, lateralFriction=0.8)

        sample_target = np.array([sample_xy[0], sample_xy[1], sample_z], dtype=float)
        constraint = None

        phases = [
            ("PREGRASP", 2.5),
            ("APPROACH", 2.5),
            ("GRASP", 1.2),
            ("LIFT", 2.5),
            ("STOW", 3.5),
        ]

        for phase, duration in phases:
            print(phase)
            steps = int(duration / cfg.dt)
            for k in range(steps):
                rover.command_passive_differential()
                rover.hold_wheels()

                if phase == "PREGRASP":
                    rover.command_gripper(1.0)
                    rover.command_arm_ik(sample_target + np.array([0.0, 0.0, 0.42]))
                elif phase == "APPROACH":
                    rover.command_gripper(1.0)
                    rover.command_arm_ik(sample_target + np.array([0.0, 0.0, 0.10]))
                elif phase == "GRASP":
                    rover.command_arm_ik(sample_target + np.array([0.0, 0.0, 0.10]))
                    rover.command_gripper(0.0)
                    if k > int(0.65 / cfg.dt) and constraint is None:
                        dist = float(np.linalg.norm(rover.ee_position() - sample_target))
                        if dist < 0.28:
                            constraint = p.createConstraint(
                                rover.body,
                                rover.ee_link,
                                sample,
                                -1,
                                p.JOINT_FIXED,
                                [0, 0, 0],
                                [0.05, 0, 0],
                                [0, 0, 0],
                            )
                            print(f"  sample attached, EE distance={dist:.3f} m")
                elif phase == "LIFT":
                    rover.command_gripper(0.0)
                    rover.command_arm_ik(sample_target + np.array([0.0, 0.0, 0.52]))
                else:
                    rover.command_arm_joint_targets([0.0, -0.85, 1.45, 0.0, -0.45, 0.0])
                    rover.command_gripper(0.15)

                p.stepSimulation()
                if k % 12 == 0:
                    camera.update(rover)
                time.sleep(cfg.dt)

        print("Arm quick demo complete. Close the PyBullet window to exit.")
        while p.isConnected():
            try:
                p.getKeyboardEvents()
                time.sleep(0.05)
            except Exception:
                break

    finally:
        try:
            if p.isConnected():
                p.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
