import unittest
import numpy as np

from rover_sim.config import SimConfig
from rover_sim.paths import get_path
from rover_sim.terrain import _terrain_heights
from rover_sim.controllers import PurePursuitAckermann
from rover_sim.observer import FixedTimeDisturbanceObserver
from rover_sim.allocator import QPTorqueAllocator


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimConfig()

    def test_path_and_tracker(self):
        path = get_path("s_curve")
        self.assertGreater(len(path), 20)
        yaw = float(np.arctan2(path[1,1]-path[0,1], path[1,0]-path[0,0]))
        out = PurePursuitAckermann(path, self.cfg.rover, self.cfg.control).update(path[0], yaw, 0.3, dt=1/60, vx=0.0, roll=0.0, roll_rate=0.0)
        self.assertAlmostEqual(out.cross_track_error, 0.0, places=6)
        self.assertTrue(all(abs(v) <= self.cfg.rover.max_steer_angle + 1e-9 for v in out.steer.values()))

    def test_terrain_repeatability(self):
        _, _, z1 = _terrain_heights("nominal", self.cfg.terrain, 7)
        _, _, z2 = _terrain_heights("nominal", self.cfg.terrain, 7)
        _, _, z3 = _terrain_heights("nominal", self.cfg.terrain, 11)
        self.assertTrue(np.allclose(z1, z2))
        self.assertFalse(np.allclose(z1, z3))

    def test_ftdo_reconstructs_step(self):
        obs = FixedTimeDisturbanceObserver()
        dt = 1/60
        v = 0.0
        dh = 0.0
        for k in range(900):
            t = k*dt
            d = 0.3 if t > 2 else 0.0
            v += dt*d
            dh = obs.update(v, 0.0, dt)
        self.assertLess(abs(dh-0.3), 0.03)

    def test_qp_penalizes_high_slip_wheel(self):
        q = QPTorqueAllocator(self.cfg.rover, self.cfg.control, self.cfg.mars_gravity)
        slips = np.array([0.05,0.05,0.55,0.05,0.05,0.05])
        tau = None
        for _ in range(20):
            tau = q.update(np.ones(6)*1.4, np.zeros(6), 1/60,
                           v_ref=.3, vx=0.0, slips=slips,
                           mus=np.ones(6)*.9, fz=np.ones(6)*110,
                           disturbance_hat=0.0)
        self.assertLess(tau[2], np.mean(np.delete(tau,2)))
        self.assertLessEqual(np.max(np.abs(tau)), self.cfg.rover.max_wheel_torque + 1e-9)

    def test_qp_transient_contact_loss_does_not_zero_drive(self):
        q = QPTorqueAllocator(self.cfg.rover, self.cfg.control, self.cfg.mars_gravity)
        tau = q.update(np.ones(6)*1.4, np.zeros(6), 1/60,
                       v_ref=.3, vx=0.0, slips=np.zeros(6),
                       mus=np.ones(6)*.88, fz=np.zeros(6),
                       disturbance_hat=0.0, contacts=np.zeros(6))
        self.assertGreater(float(np.sum(np.abs(tau))), 1.0)


if __name__ == "__main__":
    unittest.main()
