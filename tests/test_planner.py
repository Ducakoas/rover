import unittest
import numpy as np

from rover_sim.config import SimConfig
from rover_sim.planner import choose_approach_point, plan_terrain_astar
from rover_sim.terrain import TerrainField, _terrain_heights


class PlannerTests(unittest.TestCase):
    def test_astar_generates_feasible_rough_path(self):
        cfg = SimConfig()
        xs, ys, Z = _terrain_heights("rough", cfg.terrain, 7)
        terrain = TerrainField(-1, "rough", cfg.terrain, xs, ys, Z)
        start = np.array([-12.0, 0.0])
        sample = np.array([8.0, 3.0])
        approach = choose_approach_point(
            terrain,
            start,
            sample,
            cfg.mission_cfg.approach_distance_m,
            cfg.mission_cfg.approach_max_slope_deg,
        )
        result = plan_terrain_astar(
            terrain,
            start,
            approach,
            resolution_m=cfg.mission_cfg.planner_resolution_m,
            max_slope_deg=cfg.mission_cfg.planner_max_slope_deg,
            slope_weight=cfg.mission_cfg.planner_slope_weight,
            climb_weight=cfg.mission_cfg.planner_climb_weight,
            path_spacing_m=cfg.mission_cfg.planned_path_spacing_m,
            final_heading_target_xy=sample,
        )
        self.assertGreater(len(result.path_xy), 5)
        self.assertLess(np.linalg.norm(result.path_xy[0] - start), 1e-6)
        self.assertLess(np.linalg.norm(result.path_xy[-1] - approach), 1e-6)
        self.assertLessEqual(result.max_slope_deg_on_path, cfg.mission_cfg.planner_max_slope_deg + 2.0)


if __name__ == "__main__":
    unittest.main()
