import math
from pathlib import Path
import sys
import unittest


MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

IMPORT_ERROR = None
try:
    import numpy as np
    import integrated_nav_pick_place as delivery
except ImportError as exc:  # Host may not provide ROS/discoverse dependencies.
    IMPORT_ERROR = exc
    np = None
    delivery = None


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warn(self, message):
        self.messages.append(("warn", message))

    def error(self, message):
        self.messages.append(("error", message))


class FakeNavigator:
    def __init__(self):
        self.calls = []
        self.memory_available = True
        self.invalidations = []

    def set_goal(self, *goal, **kwargs):
        self.calls.append((tuple(goal), dict(kwargs)))

    def remembered_path_available(self, *_args, **_kwargs):
        return self.memory_available, {
            "enabled": True,
            "cache_hit": self.memory_available,
        }

    def invalidate_active_cached_path(self, reason, now=None):
        self.invalidations.append((reason, now))
        return []


@unittest.skipIf(
    delivery is None, f"runtime dependencies unavailable: {IMPORT_ERROR}")
class DeliveryRouteStrategyTests(unittest.TestCase):
    @staticmethod
    def _controller(base_xy=(-1.90, -2.55)):
        controller = delivery.IntegratedNavPickPlace.__new__(
            delivery.IntegratedNavPickPlace)
        controller.base_xy = np.asarray(base_xy, dtype=float)
        controller.base_yaw = -math.pi / 2.0
        controller.place_world = np.asarray([-1.94, -3.43, 0.85])
        controller.place_slot = 1
        controller.flow_phase = "grab"
        controller.des_slide = 0.11
        controller.nav = FakeNavigator()
        controller._logger = FakeLogger()
        controller.get_logger = lambda: controller._logger
        controller.now = lambda: 10.0
        controller._set_flow_phase = (
            lambda phase: setattr(controller, "flow_phase", phase))
        controller._nav_goal = None
        controller._nav_last_log = 0.0
        controller._last_nav_reason = None
        controller._nav_memory_logged = False
        controller._route_leg_name = None
        controller._route_leg_goal = None
        controller._route_leg_started_at = 0.0
        controller._route_leg_last_progress_at = 0.0
        controller._route_leg_best_distance = float("inf")
        controller.delivery_nav_stage = None
        controller.delivery_direct_fallback_used = False
        controller.scan_trunk_route_stage = None
        controller.scan_trunk_route_done = False
        controller.scan_direct_fallback_used = False
        controller.scan_route_final_goal = None
        return controller

    def test_forward_delivery_uses_live_trunk_live_sequence(self):
        controller = self._controller(base_xy=(0.0, 2.30))
        controller._start_delivery_navigation()
        self.assertEqual(controller.delivery_nav_stage, "to_trunk_entry")
        first_goal, first_options = controller.nav.calls[-1]
        self.assertEqual(first_goal, delivery.DELIVERY_TRUNK_ENTRY)
        self.assertFalse(first_options["use_path_memory"])

        controller._route_leg_tick = lambda: (True, None)
        controller._nav_to_delivery_tick()
        self.assertEqual(controller.delivery_nav_stage, "trunk_forward")
        trunk_goal, trunk_options = controller.nav.calls[-1]
        self.assertEqual(trunk_goal, delivery.DELIVERY_TRUNK_EXIT)
        self.assertTrue(trunk_options["use_path_memory"])
        self.assertTrue(trunk_options["lock_cached_path"])

        controller._nav_to_delivery_tick()
        self.assertEqual(controller.delivery_nav_stage, "to_slot")
        slot_goal, slot_options = controller.nav.calls[-1]
        self.assertEqual(slot_goal, controller._delivery_slot_goal())
        self.assertFalse(slot_options["use_path_memory"])

    def test_forward_failure_gets_one_direct_fallback_then_aborts(self):
        controller = self._controller(base_xy=(0.0, 0.5))
        controller._start_delivery_navigation()
        controller._route_leg_tick = lambda: (False, "stalled")
        controller._nav_to_delivery_tick()
        self.assertEqual(controller.delivery_nav_stage, "direct_to_slot")
        self.assertTrue(controller.delivery_direct_fallback_used)
        _, options = controller.nav.calls[-1]
        self.assertFalse(options["use_path_memory"])
        with self.assertRaisesRegex(RuntimeError, "fallback also failed"):
            controller._nav_to_delivery_tick()

    def test_reverse_route_flips_heading_and_reuses_only_the_trunk(self):
        controller = self._controller()
        controller._route_leg_tick = lambda: (True, None)
        target = np.asarray([0.92, delivery.pick.SCAN_Y])

        self.assertFalse(controller._scan_trunk_route_tick(
            target, delivery.pick.YAW_NORTH))
        self.assertEqual(controller.scan_trunk_route_stage, "trunk_reverse")
        exit_goal, exit_options = controller.nav.calls[-2]
        self.assertEqual(exit_goal, delivery.DELIVERY_TRUNK_REVERSE_START)
        self.assertAlmostEqual(exit_goal[2], math.pi / 4.0)
        self.assertAlmostEqual(
            delivery.pick.wrap_to_pi(exit_goal[2] - (-math.pi / 2.0)),
            3.0 * math.pi / 4.0)
        self.assertFalse(exit_options["use_path_memory"])
        trunk_goal, trunk_options = controller.nav.calls[-1]
        self.assertEqual(trunk_goal, delivery.DELIVERY_TRUNK_REVERSE_GOAL)
        self.assertAlmostEqual(trunk_goal[2], math.pi / 2.0)
        self.assertTrue(trunk_options["use_path_memory"])

        self.assertFalse(controller._scan_trunk_route_tick(
            target, delivery.pick.YAW_NORTH))
        self.assertEqual(controller.scan_trunk_route_stage, "to_shelf")
        _, shelf_options = controller.nav.calls[-1]
        self.assertFalse(shelf_options["use_path_memory"])

        self.assertTrue(controller._scan_trunk_route_tick(
            target, delivery.pick.YAW_NORTH))
        self.assertTrue(controller.scan_trunk_route_done)

    def test_first_shelf_trip_without_reverse_memory_avoids_anchor_detour(self):
        controller = self._controller()
        controller.nav.memory_available = False
        controller._route_leg_tick = lambda: (False, None)
        target = np.asarray([-1.735, delivery.pick.SCAN_Y])
        self.assertFalse(controller._scan_trunk_route_tick(
            target, delivery.pick.YAW_NORTH))
        self.assertEqual(
            controller.scan_trunk_route_stage, "direct_to_shelf")
        goal, options = controller.nav.calls[-1]
        self.assertEqual(goal[:2], tuple(target))
        self.assertFalse(options["use_path_memory"])

    def test_drop_signatures_follow_each_grasp_feedback_model(self):
        controller = self._controller()
        controller.target_kind = "pingguo"
        controller.use_dual_tissue_grasp = False
        controller.use_sphere_grasp = True
        controller.sphere_capture_minimum = lambda: 0.50
        controller.selected_gripper_position = lambda: 0.49
        lost, details = controller._transport_drop_signature()
        self.assertTrue(lost)
        self.assertEqual(details["mode"], "sphere")

        controller.target_kind = "maidong"
        controller.use_sphere_grasp = False
        controller.selected_gripper_position = lambda: 0.01
        lost, details = controller._transport_drop_signature()
        self.assertTrue(lost)
        self.assertEqual(details["mode"], "generic")

        controller.target_kind = "zhijin"
        controller.use_dual_tissue_grasp = True
        controller.joints = {
            "left_arm_eef_gripper_joint": 0.06,
            "right_arm_eef_gripper_joint": 0.07,
        }
        lost, details = controller._transport_drop_signature()
        self.assertTrue(lost)
        self.assertEqual(details["mode"], "dual")
        controller.joints["left_arm_eef_gripper_joint"] = 0.02
        self.assertFalse(controller._transport_drop_signature()[0])

    def test_transport_loss_is_debounced_then_requests_retry(self):
        controller = self._controller()
        controller.flow_phase = "nav_to_delivery"
        controller.place_stage = 0
        controller._drop_monitor_armed_at = 0.0
        controller._drop_signature_since = None
        controller._drop_candidate_reference_world = None
        controller._transport_drop_signature = lambda: (
            True, {"mode": "sphere", "measured_grip": 0.1})
        controller._held_product_reference_world = lambda: np.asarray(
            [0.0, 0.0, 1.0])
        events = []
        controller._start_transport_drop_recovery = (
            lambda now, **kwargs: events.append((now, kwargs)))

        controller.set_twist = lambda *_args: None
        controller.cmd_linear = 0.0
        controller.cmd_angular = 0.0
        self.assertTrue(controller._monitor_held_product(10.0))
        self.assertTrue(controller._monitor_held_product(10.2))
        self.assertTrue(controller._monitor_held_product(10.31))
        self.assertFalse(events[0][1]["over_table"])

    def test_loss_above_table_is_classified_as_delivery(self):
        controller = self._controller()
        controller.flow_phase = "place"
        controller.place_stage = 1
        controller._drop_monitor_armed_at = 0.0
        controller._drop_signature_since = 9.0
        controller._drop_candidate_reference_world = None
        controller._transport_drop_signature = lambda: (
            True, {"mode": "generic", "measured_grip": 0.0})
        x_min, y_min, x_max, y_max = delivery.DELIVERY_TABLE_XML_BOUNDS
        controller._held_product_reference_world = lambda: np.asarray([
            0.5 * (x_min + x_max),
            0.5 * (y_min + y_max),
            1.5,
        ])
        controller._drop_candidate_reference_world = (
            controller._held_product_reference_world().copy())
        events = []
        controller._start_transport_drop_recovery = (
            lambda now, **kwargs: events.append((now, kwargs)))
        controller.set_twist = lambda *_args: None
        controller.cmd_linear = 0.0
        controller.cmd_angular = 0.0

        self.assertTrue(controller._monitor_held_product(10.0))
        self.assertTrue(events[0][1]["over_table"])

    def test_intentional_release_stage_is_not_monitored(self):
        controller = self._controller()
        controller.flow_phase = "place"
        controller.place_stage = 3
        controller._drop_monitor_armed_at = 0.0
        controller._drop_signature_since = 9.0
        controller._drop_candidate_reference_world = None
        controller._transport_drop_signature = lambda: (True, {})
        self.assertFalse(controller._monitor_held_product(10.0))
        self.assertIsNone(controller._drop_signature_since)


if __name__ == "__main__":
    unittest.main()
