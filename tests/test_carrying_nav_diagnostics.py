import json
import math
import pathlib
import sys
import tempfile
import unittest

import numpy as np


MODULE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

from carrying_nav_diagnostics import (  # noqa: E402
    build_controller_trace,
    build_failure_evidence,
    save_failure_evidence,
)
from supermarket_navigation import (  # noqa: E402
    DELIVERY_TABLE_COSTMAP_BOUNDS,
    LETHAL,
    SupermarketNavigator,
)


class FakeScan:
    def __init__(self):
        self.ranges = [float("inf")] * 360
        self.angle_min = -math.pi
        self.angle_increment = 2.0 * math.pi / len(self.ranges)
        self.range_min = 0.05
        self.range_max = 8.0


class CarryingNavigationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.navigator = SupermarketNavigator()
        self.start = (-1.0, -2.0)
        self.goal = (-1.0, 2.0)

    def disconnect_with_lidar_wall(self):
        costmap = self.navigator.costmap
        costmap._fill_rect(
            costmap.dynamic_raw, -2.5, -0.05, 2.5, 0.05, LETHAL)
        costmap._rebuild_dynamic()
        self.navigator.controller.set_goal(*self.goal, math.pi / 2.0)
        velocity = self.navigator.controller.compute_velocity(
            *self.start, math.pi / 2.0,
            laser_msg=FakeScan(), time_now=4.0)
        self.assertEqual(velocity, (0.0, 0.0, False))
        self.assertIn(
            "full=disconnected", self.navigator.controller.stop_reason)

    def test_capture_reports_components_clearance_and_obstacles(self):
        self.disconnect_with_lidar_wall()
        metadata, layers = build_failure_evidence(
            self.navigator, self.start, self.goal,
            table_bounds=DELIVERY_TABLE_COSTMAP_BOUNDS)

        self.assertFalse(metadata["same_component"])
        self.assertNotEqual(
            metadata["start_component"], metadata["goal_component"])
        self.assertEqual(
            metadata["planner"]["full_failure"], "disconnected")
        self.assertGreater(
            metadata["obstacle_counts"]["lidar_raw_lethal"], 0)
        self.assertGreater(
            metadata["obstacle_counts"]["lidar_inflated_lethal"],
            metadata["obstacle_counts"]["lidar_raw_lethal"])
        self.assertIsNotNone(
            metadata["goal_clearance"]["nearest_master_obstacle_m"])
        self.assertIsNotNone(
            metadata["goal_clearance"]["delivery_table_costmap_m"])
        self.assertEqual(
            set(layers), {
                "static_costmap", "lidar_costmap", "inflated_costmap",
                "master_costmap", "lidar_inflated_costmap",
                "component_labels",
            })

    def test_capture_does_not_mutate_navigation_state(self):
        self.disconnect_with_lidar_wall()
        costmap = self.navigator.costmap
        controller = self.navigator.controller
        planner = self.navigator.planner
        arrays_before = {
            "static": costmap.static.copy(),
            "dynamic_raw": costmap.dynamic_raw.copy(),
            "vision_raw": costmap.vision_raw.copy(),
            "dynamic": costmap.dynamic.copy(),
            "master": costmap.master.copy(),
        }
        path_before = list(controller.path)
        stop_reason_before = controller.stop_reason
        failure_before = planner.failure_reason

        build_failure_evidence(
            self.navigator, self.start, self.goal,
            table_bounds=DELIVERY_TABLE_COSTMAP_BOUNDS)

        for name, expected in arrays_before.items():
            np.testing.assert_array_equal(getattr(costmap, name), expected)
        self.assertEqual(controller.path, path_before)
        self.assertEqual(controller.stop_reason, stop_reason_before)
        self.assertEqual(planner.failure_reason, failure_before)

    def test_blocked_exact_start_reports_nearest_free_displacement(self):
        costmap = self.navigator.costmap
        costmap._fill_rect(
            costmap.dynamic_raw,
            self.start[0] - 0.03, self.start[1] - 0.03,
            self.start[0] + 0.03, self.start[1] + 0.03,
            LETHAL)
        costmap._rebuild_dynamic()
        metadata, _ = build_failure_evidence(
            self.navigator, self.start, self.goal)
        self.assertFalse(metadata["start"]["exact_free"])
        self.assertGreater(
            metadata["start"]["nearest_free_displacement_m"], 0.0)

    def test_save_writes_requested_costmaps_and_metadata(self):
        self.disconnect_with_lidar_wall()
        metadata, layers = build_failure_evidence(
            self.navigator, self.start, self.goal,
            table_bounds=DELIVERY_TABLE_COSTMAP_BOUNDS)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = pathlib.Path(temporary) / "failure_01"
            metadata_path = save_failure_evidence(
                bundle, metadata, layers)
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            for required in (
                    "static_costmap", "lidar_costmap",
                    "inflated_costmap", "master_costmap"):
                artifact = bundle / document["artifacts"][required]
                self.assertTrue(artifact.is_file())
                loaded = np.load(artifact, allow_pickle=False)
                np.testing.assert_array_equal(loaded, layers[required])

    def test_controller_trace_is_read_only_and_complete(self):
        controller = self.navigator.controller
        controller.set_goal(*self.goal, math.pi / 2.0)
        controller.path = [self.start, (-1.0, 0.0), self.goal]
        controller._last_progress_time = 8.0
        path_before = list(controller.path)

        trace = build_controller_trace(
            self.navigator,
            (*self.start, math.pi / 2.0),
            (*self.goal, math.pi / 2.0),
            (0.2, -0.1),
            time_now=10.5,
            stuck_duration_s=4.25,
            payload_state={"carry_expected": True, "selected_arm": "r"})

        self.assertEqual(trace["current_pose"], [-1.0, -2.0, math.pi / 2.0])
        self.assertEqual(trace["goal_pose"], [-1.0, 2.0, math.pi / 2.0])
        self.assertEqual(trace["distance_error_m"], 4.0)
        self.assertEqual(trace["goal_yaw_error_rad"], 0.0)
        self.assertEqual(trace["cmd_vel"], {
            "linear_mps": 0.2, "angular_radps": -0.1})
        self.assertTrue(trace["planned_path_exists"])
        self.assertEqual(trace["path_index"], 0)
        self.assertEqual(trace["stuck_duration_s"], 4.25)
        self.assertEqual(trace["controller_stuck_duration_s"], 2.5)
        self.assertTrue(trace["payload"]["carry_expected"])
        self.assertEqual(controller.path, path_before)


if __name__ == "__main__":
    unittest.main()
