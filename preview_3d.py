"""Fast native OpenGL rover preview for V8.

This preview intentionally uses PyBullet's native GUI rather than the custom
Tk/TinyRenderer presentation viewer.  The ExampleBrowser chrome and synthetic
RGB/depth/segmentation panes are disabled, so the window is much smoother on
Windows while still showing the same PyBullet plant.
"""
from pathlib import Path
import time

import numpy as np

try:
    import pybullet as p
except ImportError as exc:
    raise SystemExit("PyBullet missing. Use the conda commands in README_VI.md") from exc

from rover_sim.config import SimConfig
from rover_sim.paths import get_path
from rover_sim.rover import RoverPlant
from rover_sim.terrain import create_terrain
from rover_sim.visualization import CameraManager, configure_clean_gui, draw_reference_path

cfg = SimConfig()
cfg.scenario = "rock_field"
cfg.viewer = "native"
cfg.gui = True

cid = p.connect(p.GUI)
if cid < 0:
    raise SystemExit("Cannot connect to PyBullet GUI")

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
    configure_clean_gui(p)

    terrain = create_terrain(p, cfg.scenario, cfg.terrain, seed=cfg.seed)
    path = get_path("s_curve")
    start = path[0]
    yaw = float(np.arctan2(path[1, 1] - path[0, 1], path[1, 0] - path[0, 0]))
    z = terrain.height_at(*start) + 0.62
    rover = RoverPlant(
        p,
        Path(__file__).resolve().parent / "rover_nasa.urdf",
        [start[0], start[1], z],
        yaw,
        cfg.rover,
    )
    rover.command_arm_joint_targets([0.0, -0.85, 1.45, 0.0, -0.45, 0.0])
    draw_reference_path(p, path, terrain)

    camera = CameraManager(p)
    camera.enabled = True
    camera.update(rover, force=True)

    print("V8 fast native preview. Close the PyBullet window to exit.", flush=True)
    step = 0
    camera_stride = max(1, int(round(cfg.physics_hz / 30.0)))
    while bool(p.getConnectionInfo().get("isConnected", 0)):
        rover.command_passive_differential()
        rover.set_spatial_friction(terrain)
        rover.command_wheel_torques(np.zeros(6))
        p.stepSimulation()
        if step % camera_stride == 0:
            camera.update(rover)
        step += 1
        time.sleep(cfg.dt)
finally:
    try:
        if p.isConnected():
            p.disconnect()
    except Exception:
        pass
