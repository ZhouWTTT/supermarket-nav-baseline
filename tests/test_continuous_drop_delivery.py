"""Regression tests for the snapshot-pick delivery/drop sequence."""

import pathlib
import sys
import unittest
from unittest import mock


MODULE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

IMPORT_ERROR = None
try:
    import numpy as np
    import integrated_nav_pick_place as integrated
    import snapshot_pick_client as snapshot
    import yolo_aruco_shelf_pick as pick
except ImportError as exc:  # Host without ROS/discoverse.
    IMPORT_ERROR = exc
    np = None
    integrated = None
    snapshot = None
    pick = None


@unittest.skipIf(
    integrated is None, f"runtime dependencies unavailable: {IMPORT_ERROR}")
class SnapshotPickDeliveryTests(unittest.TestCase):
    @staticmethod
    def _controller(cls, stage):
        controller = cls.__new__(cls)
        controller.flow_phase = "place"
        controller.place_stage = stage
        controller.place_creep_done = False
        controller.cmd_linear = 0.10
        controller.cmd_angular = 0.02
        controller.des_linear = 0.10
        controller.des_angular = 0.02
        controller.cmd_slide = 0.0
        controller.des_slide = 0.0
        controller.cmd_left_arm = np.zeros(6)
        controller.cmd_right_arm = np.zeros(6)
        controller._place_loaded_arm_step_rad = 0.0
        controller._place_arm_target_sent = False
        controller._dual_place_target_sent = False
        controller._dual_descent_sent = False
        controller.use_dual_tissue_grasp = False
        controller.grasp_arm = "l"
        controller.nav = mock.Mock()
        controller.nav.controller.stop_reason = None
        return controller

    def test_refined_place_still_holds_base_after_stage_zero(self):
        controller = self._controller(
            integrated.IntegratedNavPickPlace, stage=1)
        with mock.patch.object(
                pick.ShelfPickController, "smooth_commands"):
            controller.smooth_commands()
        self.assertEqual(controller.cmd_linear, 0.0)
        self.assertEqual(controller.cmd_angular, 0.0)

    def test_snapshot_drop_stage_one_creep_is_not_cancelled(self):
        controller = self._controller(
            snapshot.ContinuousOrderController, stage=1)
        with mock.patch.object(
                pick.ShelfPickController, "smooth_commands"):
            controller.smooth_commands()
        self.assertEqual(controller.cmd_linear, 0.10)
        self.assertEqual(controller.cmd_angular, 0.02)

    def test_snapshot_drop_holds_base_outside_creep_stage(self):
        for stage in (0, 2, 3, 4):
            controller = self._controller(
                snapshot.ContinuousOrderController, stage=stage)
            with mock.patch.object(
                    pick.ShelfPickController, "smooth_commands"):
                controller.smooth_commands()
            self.assertEqual(controller.cmd_linear, 0.0)
            self.assertEqual(controller.cmd_angular, 0.0)

    def test_heweidao_loaded_delivery_limits_only_angular_speed(self):
        controller = integrated.IntegratedNavPickPlace.__new__(
            integrated.IntegratedNavPickPlace)
        controller.target_kind = "heweidao"
        controller.flow_phase = "nav_to_delivery"
        controller._post_grab_slow_turn_until = 110.0
        controller._post_grab_slow_turn_logged = False
        controller.now = mock.Mock(return_value=108.0)
        controller.get_logger = mock.Mock(return_value=mock.Mock())

        controller.set_twist(0.42, -0.91)

        self.assertEqual(controller.des_linear, 0.42)
        self.assertAlmostEqual(
            controller.des_angular,
            -integrated.HEWEIDAO_LOADED_TURN_MAX_RPS)

    def test_heweidao_turn_limit_continues_after_watchdog_grace_expires(self):
        controller = integrated.IntegratedNavPickPlace.__new__(
            integrated.IntegratedNavPickPlace)
        controller.target_kind = "heweidao"
        controller.flow_phase = "nav_to_delivery"
        controller._post_grab_slow_turn_until = 110.0
        controller._post_grab_slow_turn_logged = False
        controller.now = mock.Mock(return_value=110.01)

        controller.set_twist(0.42, -0.91)

        self.assertEqual(controller.des_linear, 0.42)
        self.assertAlmostEqual(
            controller.des_angular,
            -integrated.HEWEIDAO_LOADED_TURN_MAX_RPS)

    def test_post_grab_limit_does_not_touch_other_turns(self):
        controller = integrated.IntegratedNavPickPlace.__new__(
            integrated.IntegratedNavPickPlace)
        controller.target_kind = "heweidao"
        controller.flow_phase = "return_to_shelf"
        controller._post_grab_slow_turn_until = 110.0
        controller._post_grab_slow_turn_logged = False
        controller.now = mock.Mock(return_value=102.0)

        controller.set_twist(0.42, 1.20)

        self.assertEqual(controller.des_linear, 0.42)
        self.assertEqual(controller.des_angular, 1.20)

    def test_post_grab_turn_limit_does_not_touch_other_products(self):
        for kind in ("zhijin", "sanmingzhi", "pingguo"):
            with self.subTest(kind=kind):
                controller = integrated.IntegratedNavPickPlace.__new__(
                    integrated.IntegratedNavPickPlace)
                controller.target_kind = kind
                controller.flow_phase = "nav_to_delivery"
                controller._post_grab_slow_turn_until = 110.0
                controller._post_grab_slow_turn_logged = False
                controller.now = mock.Mock(return_value=102.0)

                controller.set_twist(0.42, -0.91)

                self.assertEqual(controller.des_linear, 0.42)
                self.assertEqual(controller.des_angular, -0.91)

    def test_delivery_watchdog_uses_current_route_leg_goal(self):
        controller = integrated.IntegratedNavPickPlace.__new__(
            integrated.IntegratedNavPickPlace)
        controller.place_world = np.array([-1.8, -3.35, 0.85])
        controller.delivery_nav_stage = "to_trunk_entry"
        controller._route_leg_goal = (-0.7, 1.45, -1.57)

        self.assertEqual(
            controller._delivery_watchdog_goal(),
            (-0.7, 1.45, -1.57))

        controller.delivery_nav_stage = "slot_refine"
        self.assertEqual(
            controller._delivery_watchdog_goal(),
            (-1.8, -2.6, -1.5707963267948966))

    def test_slow_heweidao_turn_does_not_trigger_xy_stall_recovery(self):
        cls = snapshot.ContinuousOrderController
        controller = cls.__new__(cls)
        controller.target_kind = "heweidao"
        controller.flow_phase = "nav_to_delivery"
        controller._post_grab_slow_turn_until = 110.0
        controller._watchdog_phase = "nav_to_delivery"
        controller._watchdog_goal = (-0.7, 1.45, -1.57)
        controller._watchdog_t0 = 90.0
        controller._watchdog_last_xy = np.array([-1.65, 2.34])
        controller._watchdog_resets = 0
        controller._nav_recovery_attempts = 2
        controller.base_xy = np.array([-1.65, 2.34])
        controller.now = mock.Mock(return_value=108.0)
        controller._try_force_nav_recovery = mock.Mock()
        controller._start_nav_recovery = mock.Mock()

        controller._nav_watchdog_check((-0.7, 1.45, -1.57))

        self.assertEqual(controller._watchdog_t0, 108.0)
        self.assertEqual(controller._nav_recovery_attempts, 0)
        controller._try_force_nav_recovery.assert_not_called()
        controller._start_nav_recovery.assert_not_called()

    def test_delivery_leg_change_starts_a_fresh_watchdog_window(self):
        controller = snapshot.ContinuousOrderController.__new__(
            snapshot.ContinuousOrderController)
        controller.target_kind = "chengzi"
        controller.flow_phase = "nav_to_delivery"
        controller._post_grab_slow_turn_until = 0.0
        controller._watchdog_phase = "nav_to_delivery"
        controller._watchdog_goal = (-0.7, 1.45, -1.57)
        controller._watchdog_t0 = 90.0
        controller._watchdog_last_xy = np.array([-0.7, 1.45])
        controller._watchdog_resets = 2
        controller._nav_recovery_attempts = 2
        controller.base_xy = np.array([-0.7, 1.45])
        controller.now = mock.Mock(return_value=108.0)
        controller._try_force_nav_recovery = mock.Mock()
        controller._start_nav_recovery = mock.Mock()

        new_goal = (-1.94, -2.4, -2.356)
        controller._nav_watchdog_check(new_goal)

        self.assertEqual(controller._watchdog_goal, new_goal)
        self.assertEqual(controller._watchdog_t0, 108.0)
        self.assertEqual(controller._watchdog_resets, 0)
        self.assertEqual(controller._nav_recovery_attempts, 0)
        controller._try_force_nav_recovery.assert_not_called()
        controller._start_nav_recovery.assert_not_called()


if __name__ == "__main__":
    unittest.main()
