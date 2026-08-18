import math
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.ndimage import maximum_filter


MODULE_DIR = pathlib.Path(__file__).resolve().parents[1] / "examples" / "supermarket_sorting"
sys.path.insert(0, str(MODULE_DIR))

from supermarket_navigation import (  # noqa: E402
    AStarPlanner,
    Costmap2D,
    DELIVERY_APPROACH,
    DELIVERY_TRUNK_ENTRY,
    DELIVERY_TRUNK_EXIT,
    DELIVERY_TABLE_COSTMAP_BOUNDS,
    DELIVERY_TABLE_XML_BOUNDS,
    LETHAL,
    NavigationController,
    SHELF_APPROACH,
    START_POSE,
    SupermarketNavigator,
    WHOLE_BODY_KEEP_OUT_RADIUS,
    depth_image_clearance,
    point_to_rect_clearance,
    wrap_to_pi,
)


class FakeScan:
    def __init__(self, ranges, angle_min=-math.pi, angle_increment=None,
                 range_min=0.05, range_max=8.0):
        self.ranges = list(ranges)
        self.angle_min = angle_min
        self.angle_increment = (
            2.0 * math.pi / len(self.ranges)
            if angle_increment is None else angle_increment)
        self.range_min = range_min
        self.range_max = range_max


class FakeDepthImage:
    def __init__(self, depth_mm):
        array = np.asarray(depth_mm, dtype='<u2')
        self.height, self.width = array.shape
        self.encoding = '16UC1'
        self.is_bigendian = False
        self.step = self.width * 2
        self.data = array.tobytes()


def add_fixed_xml_obstacles(costmap):
    """Add the five default XML boxes in world coordinates."""
    boxes = [
        (-0.338, -1.869, 0.30, 0.20),
        (-0.010, 0.830, 0.20, 0.30),
        (-2.010, -1.114, 0.30, 0.20),
        (-2.122, 1.415, 0.20, 0.30),
        (-1.070, -0.197, 0.30, 0.20),
    ]
    for x, y, half_x, half_y in boxes:
        costmap._fill_rect(
            costmap.dynamic_raw,
            x - half_x, y - half_y, x + half_x, y + half_y,
            LETHAL)
    occupied = (costmap.dynamic_raw == LETHAL).astype(np.int8)
    inflated = maximum_filter(occupied, footprint=costmap._disk)
    costmap.dynamic[inflated > 0] = LETHAL
    costmap.rebuild_master()


class PlannerTests(unittest.TestCase):
    def test_path_preserves_exact_free_goal(self):
        planner = AStarPlanner(Costmap2D())
        goal = SHELF_APPROACH["C"][:2]
        path = planner.plan(*START_POSE[:2], *goal)
        self.assertEqual(path[-1], goal)

    def test_required_static_route_is_reachable(self):
        planner = AStarPlanner(Costmap2D())
        legs = [
            (START_POSE[:2], SHELF_APPROACH["C"][:2]),
            (SHELF_APPROACH["C"][:2], DELIVERY_APPROACH[:2]),
            (DELIVERY_APPROACH[:2], SHELF_APPROACH["C"][:2]),
        ]
        for start, goal in legs:
            with self.subTest(start=start, goal=goal):
                path = planner.plan(*start, *goal)
                self.assertIsNotNone(path)
                self.assertGreaterEqual(len(path), 2)

    def test_required_route_around_default_five_boxes_is_reachable(self):
        costmap = Costmap2D()
        add_fixed_xml_obstacles(costmap)
        planner = AStarPlanner(costmap)
        for shelf, pose in SHELF_APPROACH.items():
            for start, goal in [
                (START_POSE[:2], pose[:2]),
                (pose[:2], DELIVERY_APPROACH[:2]),
                (DELIVERY_APPROACH[:2], pose[:2]),
            ]:
                with self.subTest(shelf=shelf, start=start, goal=goal):
                    self.assertIsNotNone(planner.plan(*start, *goal))

    def test_reusable_delivery_trunk_and_connectors_are_reachable(self):
        costmap = Costmap2D()
        add_fixed_xml_obstacles(costmap)
        planner = AStarPlanner(costmap)
        slot_goals = [
            (-2.20, DELIVERY_APPROACH[1]),
            (-1.94, DELIVERY_APPROACH[1]),
            (-1.68, DELIVERY_APPROACH[1]),
        ]
        legs = [
            (SHELF_APPROACH[shelf][:2], DELIVERY_TRUNK_ENTRY[:2])
            for shelf in SHELF_APPROACH
        ]
        legs += [
            (DELIVERY_TRUNK_ENTRY[:2], DELIVERY_TRUNK_EXIT[:2]),
            (DELIVERY_TRUNK_EXIT[:2], DELIVERY_TRUNK_ENTRY[:2]),
        ]
        legs += [
            (DELIVERY_TRUNK_EXIT[:2], goal) for goal in slot_goals]
        for start, goal in legs:
            with self.subTest(start=start, goal=goal):
                path = planner.plan(*start, *goal)
                self.assertIsNotNone(path)
                self.assertGreaterEqual(len(path), 2)


class CostmapTests(unittest.TestCase):
    def test_table_costmap_bounds_conservatively_contain_xml_geometry(self):
        cxmin, cymin, cxmax, cymax = DELIVERY_TABLE_COSTMAP_BOUNDS
        xmin, ymin, xmax, ymax = DELIVERY_TABLE_XML_BOUNDS
        self.assertLessEqual(cxmin, xmin)
        self.assertLessEqual(cymin, ymin)
        self.assertGreaterEqual(cxmax, xmax)
        self.assertGreaterEqual(cymax, ymax)

    def test_delivery_goal_has_whole_body_table_standoff(self):
        clearance = point_to_rect_clearance(
            *DELIVERY_APPROACH[:2], DELIVERY_TABLE_COSTMAP_BOUNDS)
        # The current close-delivery configuration intentionally carries only
        # a small positive margin; the analytical guard must still remain
        # strictly outside the whole-body exclusion radius.
        self.assertGreater(clearance, WHOLE_BODY_KEEP_OUT_RADIUS)

    def test_depth_image_extracts_central_clearance_in_metres(self):
        depth = np.full((60, 80), 4000, dtype=np.uint16)
        depth[23:38, 28:52] = 850
        clearance = depth_image_clearance(FakeDepthImage(depth))
        self.assertAlmostEqual(clearance, 0.85, places=2)

    def test_depth_obstacle_is_not_cleared_by_low_lidar_ray(self):
        costmap = Costmap2D(vision_ttl_scans=2)
        # Vision mapping now requires two matching frames; local depth braking
        # remains immediate and is covered by the controller test below.
        costmap.update_from_depth_obstacle(0.5, -1.0, 0.0, 0.0)
        costmap.update_from_depth_obstacle(0.5, -1.0, 0.0, 0.0)
        self.assertGreater(np.count_nonzero(costmap.vision_raw), 0)

        clear_scan = FakeScan([float("inf")] * 360)
        costmap.update_from_scan(clear_scan, -1.0, 0.0, 0.0)
        self.assertGreater(np.count_nonzero(costmap.vision_raw), 0)
        for _ in range(2):
            costmap.update_from_scan(clear_scan, -1.0, 0.0, 0.0)
        self.assertEqual(np.count_nonzero(costmap.vision_raw), 0)

    def test_world_below_origin_is_out_of_bounds(self):
        costmap = Costmap2D()
        gx, gy = costmap.world_to_grid(
            costmap.origin_x - 0.001, costmap.origin_y - 0.001)
        self.assertLess(gx, 0)
        self.assertLess(gy, 0)
        self.assertFalse(costmap.in_bounds(gx, gy))

    def test_line_check_covers_grid_cells_touched_near_corner(self):
        costmap = Costmap2D()
        costmap.master[85, 38] = LETHAL
        self.assertFalse(costmap.line_is_free(
            -0.602, 0.551, -0.453, 0.203))

    def test_repeated_scan_does_not_expand_inflation(self):
        costmap = Costmap2D()
        ranges = [float("inf")] * 360
        ranges[180] = 0.5
        scan = FakeScan(ranges)

        costmap.update_from_scan(scan, -1.0, 0.0, 0.0)
        occupied_once = int(np.count_nonzero(costmap.dynamic))
        for _ in range(12):
            costmap.update_from_scan(scan, -1.0, 0.0, 0.0)
        occupied_repeated = int(np.count_nonzero(costmap.dynamic))

        self.assertGreater(occupied_once, 0)
        self.assertEqual(occupied_repeated, occupied_once)

    def test_unobserved_hit_expires(self):
        costmap = Costmap2D(dynamic_ttl_scans=2)
        hit_ranges = [float("inf")] * 360
        hit_ranges[180] = 0.5
        costmap.update_from_scan(FakeScan(hit_ranges), -1.0, 0.0, 0.0)
        self.assertGreater(np.count_nonzero(costmap.dynamic_raw), 0)

        no_observation = FakeScan([float("nan")] * 360)
        for _ in range(3):
            costmap.update_from_scan(no_observation, -1.0, 0.0, 0.0)
        self.assertEqual(np.count_nonzero(costmap.dynamic_raw), 0)
        self.assertEqual(np.count_nonzero(costmap.dynamic), 0)

    def test_single_lidar_miss_does_not_flicker_obstacle(self):
        costmap = Costmap2D()
        hit_ranges = [float("inf")] * 360
        hit_ranges[180] = 0.5
        costmap.update_from_scan(
            FakeScan(hit_ranges), -1.0, 0.0, 0.0)
        self.assertGreater(np.count_nonzero(costmap.dynamic_raw), 0)

        clear_scan = FakeScan([float("inf")] * 360)
        costmap.update_from_scan(clear_scan, -1.0, 0.0, 0.0)
        self.assertGreater(np.count_nonzero(costmap.dynamic_raw), 0)
        costmap.update_from_scan(clear_scan, -1.0, 0.0, 0.0)
        costmap.update_from_scan(clear_scan, -1.0, 0.0, 0.0)
        self.assertEqual(np.count_nonzero(costmap.dynamic_raw), 0)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.costmap = Costmap2D()
        self.controller = NavigationController(
            self.costmap, AStarPlanner(self.costmap))

    def test_forward_cone_uses_angles_not_array_midpoint(self):
        # Scan spans [0, 2pi); forward is index 0, not the array midpoint.
        ranges = [float("inf")] * 360
        ranges[0] = 0.20
        scan = FakeScan(ranges, angle_min=0.0)
        self.assertAlmostEqual(self.controller._front_clearance(scan), 0.20)
        self.assertTrue(self.controller._check_front_blocked(scan))

    def test_lookahead_is_interpolated_inside_long_segment(self):
        self.controller.path = [(0.0, 0.0), (4.0, 0.0)]
        point = self.controller._lookahead_point(0.0, 0.0, 0.40)
        self.assertAlmostEqual(point[0], 0.40)
        self.assertAlmostEqual(point[1], 0.0)

    def test_equal_cost_replan_does_not_flip_current_route(self):
        current = [(-1.0, 0.0), (-1.2, 1.0), (-1.0, 2.0)]
        alternate = [(-1.0, 0.0), (-0.8, 1.0), (-1.0, 2.0)]
        self.controller._install_path(current)
        changed = self.controller._consider_new_path(
            alternate, -1.0, 0.0)
        self.assertFalse(changed)
        self.assertEqual(self.controller.path, current)

    def test_rotation_watchdog_detects_more_than_full_alignment_turn(self):
        triggered = False
        yaw = 0.0
        self.controller._update_rotation_watchdog(-1.0, 0.0, yaw)
        for _ in range(30):
            yaw = wrap_to_pi(yaw + 0.15)
            triggered = self.controller._update_rotation_watchdog(
                -1.0, 0.0, yaw)
            if triggered:
                break
        self.assertTrue(triggered)

        # Translational progress resets accumulated rotation.
        self.assertFalse(self.controller._update_rotation_watchdog(
            -0.85, 0.0, yaw))

    def test_large_heading_error_never_commands_forward_motion(self):
        self.controller.set_goal(-1.0, 2.0, 0.0)
        v, w, reached = self.controller.compute_velocity(
            -1.0, 0.0, -math.pi / 2.0,
            FakeScan([float("inf")] * 360), time_now=1.0)
        self.assertFalse(reached)
        self.assertEqual(v, 0.0)
        self.assertNotEqual(w, 0.0)

    def test_close_obstacle_bypasses_velocity_ramp_and_stops(self):
        self.controller.set_goal(-1.0, 2.0, math.pi / 2.0)
        clear_scan = FakeScan([float("inf")] * 360)
        for tick in range(10):
            v, _, _ = self.controller.compute_velocity(
                -1.0, 0.0, math.pi / 2.0,
                clear_scan, time_now=1.0 + tick * 0.02)
        self.assertGreater(v, 0.0)

        blocked = [float("inf")] * 360
        blocked[180] = 0.20
        v, _, _ = self.controller.compute_velocity(
            -1.0, 0.0, math.pi / 2.0,
            FakeScan(blocked), time_now=1.30)
        self.assertEqual(v, 0.0)

    def test_depth_camera_can_trigger_emergency_stop(self):
        self.controller.set_goal(-1.0, 2.0, math.pi / 2.0)
        clear_scan = FakeScan([float("inf")] * 360)
        for tick in range(10):
            v, _, _ = self.controller.compute_velocity(
                -1.0, 0.0, math.pi / 2.0,
                clear_scan, time_now=2.0 + tick * 0.02)
        self.assertGreater(v, 0.0)

        v, _, _ = self.controller.compute_velocity(
            -1.0, 0.0, math.pi / 2.0,
            clear_scan, depth_clearance=0.20, time_now=2.30)
        self.assertEqual(v, 0.0)

    def test_no_path_early_return_keeps_failure_and_current_sensors(self):
        # A full-width live-lidar wall disconnects start and goal.  This takes
        # the early-return branch which previously logged only bare "no_path"
        # and stale infinity clearances.
        self.costmap._fill_rect(
            self.costmap.dynamic_raw, -2.5, -0.05, 2.5, 0.05, LETHAL)
        self.costmap._rebuild_dynamic()
        self.controller.set_goal(-1.0, 2.0, math.pi / 2.0)
        ranges = [float("inf")] * 360
        ranges[180] = 1.93
        v, w, reached = self.controller.compute_velocity(
            -1.0, -2.0, math.pi / 2.0,
            FakeScan(ranges), depth_clearance=1.69, time_now=4.0)
        self.assertEqual((v, w, reached), (0.0, 0.0, False))
        self.assertIn("full=disconnected", self.controller.stop_reason)
        self.assertIn("obs:lidar=", self.controller.stop_reason)
        self.assertAlmostEqual(self.controller.lidar_clearance, 1.93)
        self.assertAlmostEqual(self.controller.depth_clearance_val, 1.69)

    def test_fallback_records_both_attempts_without_losing_failure(self):
        # Lidar itself disconnects the map; one vision cell forces a fallback
        # attempt as well.  Both failures must survive restoration of vision.
        self.costmap._fill_rect(
            self.costmap.dynamic_raw, -2.5, -0.05, 2.5, 0.05, LETHAL)
        self.costmap.vision_raw[20, 20] = LETHAL
        self.costmap._rebuild_dynamic()
        self.controller.set_goal(-1.0, 2.0, math.pi / 2.0)
        self.controller.compute_velocity(
            -1.0, -2.0, math.pi / 2.0,
            FakeScan([float("inf")] * 360), time_now=5.0)
        reason = self.controller.stop_reason
        self.assertIn("full=disconnected", reason)
        self.assertIn("lidar_only=disconnected", reason)
        self.assertIn("vis=1", reason)
        self.assertEqual(self.controller._last_plan_mode, "failed")

    def test_vision_only_disconnect_uses_lidar_fallback(self):
        # A vision-only wall blocks the full map, but fallback must return a
        # lidar-only path and restore the vision layer afterwards.
        self.costmap._fill_rect(
            self.costmap.vision_raw, -2.5, -0.05, 2.5, 0.05, LETHAL)
        vision_before = self.costmap.vision_raw.copy()
        self.costmap._rebuild_dynamic()
        path = self.controller._try_plan_with_fallback(
            -1.0, -2.0, -1.0, 2.0)
        self.assertIsNotNone(path)
        self.assertEqual(self.controller._last_plan_mode, "lidar_only")
        self.assertEqual(
            self.controller._last_plan_full_failure, "disconnected")
        np.testing.assert_array_equal(
            self.costmap.vision_raw, vision_before)

    def test_table_keepout_blocks_motion_before_contact(self):
        # At y=-2.60 the physical chassis is not touching the table, but a
        # short southward command would violate the whole-body safety margin.
        self.assertFalse(self.controller._motion_is_free(
            -1.60, -2.60, -math.pi / 2.0, 0.20, 0.0,
            horizon=0.45, sample_dt=0.05))

    def test_table_keepout_blocks_unsafe_final_rotation(self):
        self.assertFalse(self.controller._table_rotation_is_free(
            -1.60, -2.67))
        self.assertTrue(self.controller._table_rotation_is_free(
            *DELIVERY_APPROACH[:2]))

    def test_requested_d_b_delivery_sequence_avoids_inflated_boxes(self):
        costmap = Costmap2D()
        add_fixed_xml_obstacles(costmap)
        controller = NavigationController(costmap, AStarPlanner(costmap))
        scan = FakeScan([float("inf")] * 360)
        x, y, yaw = START_POSE
        now = 0.0

        for goal in (
                SHELF_APPROACH["D"],
                SHELF_APPROACH["B"],
                DELIVERY_APPROACH):
            controller.set_goal(*goal)
            reached = False
            leg_x = []
            near_goal_speed = []
            for _ in range(5000):
                now += 0.02
                near_goal = math.hypot(goal[0] - x, goal[1] - y) < 0.60
                v, w, reached = controller.compute_velocity(
                    x, y, yaw, scan, time_now=now)
                yaw = wrap_to_pi(yaw + w * 0.02)
                x += v * math.cos(yaw) * 0.02
                y += v * math.sin(yaw) * 0.02
                leg_x.append(x)
                if near_goal:
                    near_goal_speed.append(v)
                self.assertTrue(costmap.is_free_world(x, y))
                if reached:
                    break
            self.assertTrue(reached, msg=f"failed to reach {goal}")
            if goal == DELIVERY_APPROACH:
                self.assertLessEqual(
                    max(near_goal_speed), controller.near_goal_max_lin + 0.001)
                self.assertGreater(min(leg_x), -1.80)


class NavigatorPathMemoryTests(unittest.TestCase):
    def test_trunk_cache_is_locked_and_reverse_route_hits(self):
        with tempfile.TemporaryDirectory() as tempdir, mock.patch.dict(
                os.environ, {
                    "SUPERMARKET_PATH_MEMORY": "1",
                    "SUPERMARKET_PATH_MEMORY_FILE": str(
                        pathlib.Path(tempdir) / "paths.json"),
                }):
            navigator = SupermarketNavigator()
            path = navigator.planner.plan(
                *DELIVERY_TRUNK_ENTRY[:2], *DELIVERY_TRUNK_EXIT[:2])
            self.assertIsNotNone(path)
            navigator.path_memory.save_path(
                *DELIVERY_TRUNK_ENTRY,
                *DELIVERY_TRUNK_EXIT,
                path,
                source="test_trunk")

            clear_scan = FakeScan([float("inf")] * 360)
            navigator.set_goal(
                *DELIVERY_TRUNK_EXIT,
                cached_start_offset_limit=0.18,
                cached_goal_offset_limit=0.12,
                use_path_memory=True,
                lock_cached_path=True)
            navigator.update(
                *DELIVERY_TRUNK_ENTRY,
                laser_msg=clear_scan, time_now=1.0)
            status = navigator.path_memory_status()
            self.assertTrue(status["cache_hit"])
            self.assertTrue(status["cached_path_active"])
            self.assertTrue(status["cached_path_locked"])

            reverse_available, reverse_lookup = (
                navigator.remembered_path_available(
                    (DELIVERY_TRUNK_EXIT[0], DELIVERY_TRUNK_EXIT[1],
                     math.pi / 4.0),
                    (DELIVERY_TRUNK_ENTRY[0], DELIVERY_TRUNK_ENTRY[1],
                     math.pi / 2.0),
                    start_offset_limit=0.18,
                    goal_offset_limit=0.12))
            self.assertTrue(reverse_available)
            self.assertTrue(reverse_lookup["cache_hit"])

            navigator.set_goal(
                DELIVERY_TRUNK_ENTRY[0], DELIVERY_TRUNK_ENTRY[1],
                math.pi / 2.0,
                cached_start_offset_limit=0.18,
                cached_goal_offset_limit=0.12,
                use_path_memory=True,
                lock_cached_path=True)
            navigator.update(
                DELIVERY_TRUNK_EXIT[0], DELIVERY_TRUNK_EXIT[1],
                math.pi / 4.0,
                laser_msg=clear_scan, time_now=2.0)
            reverse_status = navigator.path_memory_status()
            self.assertTrue(reverse_status["cache_hit"])
            self.assertEqual(
                reverse_status["source"], "test_trunk_reverse")

    def test_connector_goal_does_not_use_or_save_path_memory(self):
        with tempfile.TemporaryDirectory() as tempdir, mock.patch.dict(
                os.environ, {
                    "SUPERMARKET_PATH_MEMORY": "1",
                    "SUPERMARKET_PATH_MEMORY_FILE": str(
                        pathlib.Path(tempdir) / "paths.json"),
                }):
            navigator = SupermarketNavigator()
            navigator.set_goal(
                *DELIVERY_TRUNK_ENTRY, use_path_memory=False)
            navigator.update(
                *SHELF_APPROACH["C"],
                laser_msg=FakeScan([float("inf")] * 360),
                time_now=1.0)
            status = navigator.path_memory_status()
            self.assertFalse(status["goal_uses_path_memory"])
            self.assertFalse(status["cache_hit"])

    def test_cached_recovery_persistently_invalidates_both_directions(self):
        with tempfile.TemporaryDirectory() as tempdir, mock.patch.dict(
                os.environ, {
                    "SUPERMARKET_PATH_MEMORY": "1",
                    "SUPERMARKET_PATH_MEMORY_FILE": str(
                        pathlib.Path(tempdir) / "paths.json"),
                }):
            navigator = SupermarketNavigator()
            path = navigator.planner.plan(
                *DELIVERY_TRUNK_ENTRY[:2], *DELIVERY_TRUNK_EXIT[:2])
            navigator.path_memory.save_path(
                *DELIVERY_TRUNK_ENTRY,
                *DELIVERY_TRUNK_EXIT,
                path,
                source="test_trunk")
            navigator.set_goal(
                *DELIVERY_TRUNK_EXIT,
                use_path_memory=True,
                lock_cached_path=True)
            clear_scan = FakeScan([float("inf")] * 360)
            navigator.update(
                *DELIVERY_TRUNK_ENTRY,
                laser_msg=clear_scan, time_now=1.0)
            self.assertTrue(
                navigator.path_memory_status()["cached_path_active"])

            def recovery(*_args, **_kwargs):
                navigator.controller.stop_reason = "rotation_loop"
                return 0.0, 0.0, False

            with mock.patch.object(
                    navigator.controller, "compute_velocity",
                    side_effect=recovery):
                navigator.update(
                    *DELIVERY_TRUNK_ENTRY,
                    laser_msg=clear_scan, time_now=1.1)
            status = navigator.path_memory_status()
            self.assertFalse(status["cached_path_active"])
            self.assertTrue(status["cached_path_invalidated"])
            self.assertEqual(status["invalidation_reason"], "rotation_loop")
            self.assertEqual(len(status["invalidated_keys"]), 2)
            self.assertEqual(status["cache_size"], 0)

    def test_leg_supervisor_can_persistently_invalidate_active_cache(self):
        with tempfile.TemporaryDirectory() as tempdir, mock.patch.dict(
                os.environ, {
                    "SUPERMARKET_PATH_MEMORY": "1",
                    "SUPERMARKET_PATH_MEMORY_FILE": str(
                        pathlib.Path(tempdir) / "paths.json"),
                }):
            navigator = SupermarketNavigator()
            path = navigator.planner.plan(
                *DELIVERY_TRUNK_ENTRY[:2], *DELIVERY_TRUNK_EXIT[:2])
            navigator.path_memory.save_path(
                *DELIVERY_TRUNK_ENTRY,
                *DELIVERY_TRUNK_EXIT,
                path,
                source="test_trunk")
            navigator.set_goal(
                *DELIVERY_TRUNK_EXIT,
                use_path_memory=True,
                lock_cached_path=True)
            navigator.update(
                *DELIVERY_TRUNK_ENTRY,
                laser_msg=FakeScan([float("inf")] * 360),
                time_now=1.0)

            removed = navigator.invalidate_active_cached_path(
                "route_leg:trunk_forward:hard_timeout", now=2.0)
            status = navigator.path_memory_status()
            self.assertEqual(len(removed), 2)
            self.assertFalse(status["cached_path_active"])
            self.assertTrue(status["cached_path_invalidated"])
            self.assertEqual(status["cache_size"], 0)
            self.assertEqual(navigator.controller._replan_hold_until, 2.0)


if __name__ == "__main__":
    unittest.main()
