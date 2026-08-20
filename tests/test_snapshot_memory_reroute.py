"""Regression tests for memory-driven shelf rerouting."""

import pathlib
import sys
import time
import unittest
from unittest import mock

import numpy as np


MODULE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

IMPORT_ERROR = None
try:
    import snapshot_pick_client as client
    import supermarket_navigation as nav
except ImportError as exc:  # Host without ROS/discoverse; Client runs it.
    IMPORT_ERROR = exc
    client = None
    nav = None


@unittest.skipIf(client is None, f"runtime dependencies unavailable: {IMPORT_ERROR}")
class MemoryRerouteTests(unittest.TestCase):
    def test_old_candidate_cannot_trigger_dynamic_reroute(self):
        candidate = {
            "slot_key": "L3|A|1", "shelf": "A", "last_seen": 99.0}
        self.assertFalse(client._memory_candidate_allowed(
            candidate, min_last_seen=100.0))

    def test_fresh_unvisited_candidate_remains_available(self):
        candidate = {
            "slot_key": "L1|C|2", "shelf": "C", "last_seen": 101.0}
        self.assertTrue(client._memory_candidate_allowed(
            candidate, excluded_shelves={"A"}, min_last_seen=100.0))

    def test_a_to_b_marks_a_exhausted_and_blocks_return_to_a(self):
        class _Logger:
            def info(self, _message):
                pass

        class _Controller:
            scan_station_order = [4, 3, 2, 1, 0]
            memory_last_scan_station_x = -1.735
            memory_exhausted_shelves = set()

            def current_scan_station_x(self):
                return -0.850

            def get_logger(self):
                return _Logger()

        controller = _Controller()
        client._update_memory_scan_progress(controller)
        self.assertEqual(controller.memory_exhausted_shelves, {"A"})
        stale_a = {
            "slot_key": "L3|A|1", "shelf": "A", "last_seen": 101.0}
        self.assertFalse(client._memory_candidate_allowed(
            stale_a,
            excluded_shelves=controller.memory_exhausted_shelves,
            min_last_seen=100.0))

    def test_camera_pose_changes_do_not_exhaust_current_shelf(self):
        class _Controller:
            scan_station_order = [4, 3, 2, 1, 0]
            memory_last_scan_station_x = -1.735
            memory_exhausted_shelves = set()

            def current_scan_station_x(self):
                return -1.735

        controller = _Controller()
        client._update_memory_scan_progress(controller)
        self.assertEqual(controller.memory_exhausted_shelves, set())

    def test_grab_complete_consumes_memory_immediately_and_only_once(self):
        calls = []
        controller = client.ContinuousOrderController.__new__(
            client.ContinuousOrderController)
        controller.memory_consumed = False
        controller.memory_consume_callback = lambda ctrl: (
            calls.append(ctrl), setattr(ctrl, "memory_consumed", True))

        with mock.patch.object(
                client.IntegratedNavPickPlace, "_on_grab_complete"):
            controller._on_grab_complete()
            controller._on_grab_complete()

        self.assertEqual(calls, [controller])
        self.assertTrue(controller.memory_consumed)

    def test_immediate_consume_writes_the_grabbed_slot_idempotently(self):
        calls = []

        class _Logger:
            def info(self, _message):
                pass

        class _Controller:
            memory_consumed = False
            target_kind = "kele"
            target_marker_id = None

            def target_slot(self):
                return ("A", "L2", "1")

            def get_logger(self):
                return _Logger()

        class _Tracker:
            def consume_slot(self, *slot, kind=None):
                calls.append((slot, kind))

        controller = _Controller()
        tracker = _Tracker()
        self.assertTrue(client._consume_grabbed_memory(controller, tracker))
        self.assertFalse(client._consume_grabbed_memory(controller, tracker))
        self.assertEqual(calls, [(("A", "L2", "1"), "kele")])

    def test_failed_b_hint_selects_d_without_scanning_c(self):
        """回放 09:20:52 第五单：B 顶层失效后应直接使用 D 记录。"""
        candidates = [
            {
                "slot_key": "L3|B|3", "shelf": "B", "level": "L3",
                "confidence": 0.941, "closest_distance": 0.67,
                "last_seen": 10.0,
            },
            {
                "slot_key": "L2|D|3", "shelf": "D", "level": "L2",
                "confidence": 0.974, "closest_distance": 0.576,
                "last_seen": 9.0,
            },
            {
                "slot_key": "L1|E|2", "shelf": "E", "level": "L1",
                "confidence": 0.971, "closest_distance": 0.584,
                "last_seen": 8.0,
            },
        ]
        initial = client._select_memory_hint(
            candidates, (-1.70, 2.40), 0.90, reliable_only=True)
        self.assertEqual((initial["shelf"], initial["level"]), ("B", "L3"))

        after_b_failure = client._select_memory_hint(
            candidates, (-0.91, 2.45), 0.90,
            exclude_shelf_levels={("B", "L3")},
            reliable_only=True)
        self.assertEqual(
            (after_b_failure["shelf"], after_b_failure["level"]),
            ("D", "L2"))
        self.assertNotEqual(after_b_failure["shelf"], "C")

    def test_reliable_hint_carries_column_for_direct_slot_navigation(self):
        candidates = [
            {
                "slot_key": "L3|B|3", "shelf": "B", "level": "L3",
                "column": "3", "confidence": 0.941,
                "closest_distance": 0.67, "last_seen": 10.0,
            },
        ]
        hint = client._select_memory_hint(
            candidates, (-1.70, 2.40), 0.90, reliable_only=True)
        self.assertEqual(hint["column"], "3")
        self.assertEqual(hint["slot_key"], "L3|B|3")

    def test_memory_direct_hint_allows_all_candidates(self):
        """取消可靠门槛后，任意记忆候选都允许直达（close-recheck 兜底）。"""
        good = {
            "observed_distance": 0.50,
            "world_y": 3.21,
            "sample_count": 6,
            "last_seen": time.time(),
        }
        self.assertTrue(client._memory_direct_hint_ok(good))

        noisy_far = {
            "observed_distance": 2.00,
            "world_y": 2.80,
            "sample_count": 1,
            "last_seen": 0.0,
        }
        self.assertTrue(client._memory_direct_hint_ok(noisy_far))

    def test_require_direct_skips_noisy_candidate_and_selects_good_slot(self):
        now = time.time()
        candidates = [
            {
                "slot_key": "L3|B|3", "shelf": "B", "level": "L3",
                "column": "3", "confidence": 0.969,
                "closest_distance": 0.63, "last_seen": now,
                "world_y": 3.098, "sample_count": 6,
            },
            {
                "slot_key": "L2|B|2", "shelf": "B", "level": "L2",
                "column": "2", "confidence": 0.972,
                "closest_distance": 0.50, "last_seen": now,
                "world_y": 3.22, "sample_count": 6,
            },
        ]
        hint = client._select_memory_hint(
            candidates, (-0.75, 2.45), 0.90,
            reliable_only=True, require_direct=True)
        self.assertEqual(hint["slot_key"], "L2|B|2")

    def test_fallback_hint_prefers_nearest_shelf_over_higher_confidence(self):
        candidates = [
            {
                "slot_key": "L3|E|2", "shelf": "E", "level": "L3",
                "column": "2", "confidence": 0.98,
                "closest_distance": 0.49, "last_seen": time.time(),
            },
            {
                "slot_key": "L1|C|2", "shelf": "C", "level": "L1",
                "column": "2", "confidence": 0.88,
                "closest_distance": 0.55, "last_seen": time.time(),
            },
        ]
        hint = client._select_memory_hint(
            candidates, (-0.75, 2.45), 0.90, reliable_only=False)
        self.assertEqual(hint["slot_key"], "L1|C|2")

    def test_tissue_filters_side_columns_before_nearest_selection(self):
        """回放 09:24:43：B 侧列不得遮住 C 中列纸巾记忆。"""
        candidates = [
            {
                "slot_key": "L2|B|3", "shelf": "B", "level": "L2",
                "column": "3", "confidence": 0.928,
                "closest_distance": 0.94, "last_seen": time.time(),
            },
            {
                "slot_key": "L1|C|2", "shelf": "C", "level": "L1",
                "column": "2", "confidence": 0.935,
                "closest_distance": 1.46, "last_seen": time.time(),
            },
            {
                "slot_key": "L2|E|2", "shelf": "E", "level": "L2",
                "column": "2", "confidence": 0.983,
                "closest_distance": 0.65, "last_seen": time.time(),
            },
        ]

        eligible = client.grasp_eligible_candidates("zhijin", candidates)
        hint = client._select_memory_hint(
            eligible, (-1.69, 2.41), 0.90, reliable_only=False)

        self.assertEqual(
            [item["slot_key"] for item in eligible],
            ["L1|C|2", "L2|E|2"])
        self.assertEqual(hint["slot_key"], "L1|C|2")

    def test_non_tissue_candidates_keep_all_columns(self):
        candidates = [
            {"slot_key": "L1|B|1", "column": "1"},
            {"slot_key": "L2|C|2", "column": "2"},
            {"slot_key": "L3|D|3", "column": "3"},
        ]
        self.assertEqual(
            client.grasp_eligible_candidates("heweidao", candidates),
            candidates)

    def test_nav_watchdog_can_force_reverse_recovery(self):
        controller = client.ContinuousOrderController.__new__(
            client.ContinuousOrderController)
        controller.flow_phase = "nav_to_delivery"
        controller.laser_msg = object()
        controller.base_xy = np.array([0.53, 2.23])
        controller.base_yaw = -3.05

        inner = mock.Mock()
        inner._maybe_start_reverse_recovery.return_value = True
        controller.nav = mock.Mock()
        controller.nav.controller = inner

        self.assertTrue(controller._try_force_nav_recovery(1.0))
        inner._maybe_start_reverse_recovery.assert_called_once()

    def test_reverse_recovery_allows_safe_inflation_halo_escape(self):
        class FakeCostmap:
            resolution = 0.05

            @staticmethod
            def is_static_free_world(x, y):
                return True

            @staticmethod
            def is_free_world(x, y):
                return False

            @staticmethod
            def raw_dynamic_clearance_world(x, y):
                return 0.10

        controller = nav.NavigationController.__new__(nav.NavigationController)
        controller.cm = FakeCostmap()
        # 默认仍拒绝进入动态膨胀区。
        self.assertFalse(controller._straight_translation_is_free(
            0.0, 0.0, 0.0, -0.10))
        # 倒车恢复允许在“未踩到原始障碍、静态安全、后向雷达安全”的前提下
        # 穿过动态膨胀区，避免局部死锁。
        self.assertTrue(controller._straight_translation_is_free(
            0.0, 0.0, 0.0, -0.10, allow_dynamic_inflation=True))

    def test_flow_level_nav_recovery_starts_when_rear_is_safe(self):
        controller = client.ContinuousOrderController.__new__(
            client.ContinuousOrderController)
        controller.flow_phase = "nav_to_delivery"
        controller._nav_recovery_phase = None
        controller._nav_recovery_attempts = 0
        controller.base_xy = np.array([0.60, 2.24])
        controller.base_yaw = 2.72
        controller.laser_msg = object()
        controller.now = mock.Mock(return_value=100.0)
        controller.set_twist = mock.Mock()

        class _Logger:
            @staticmethod
            def info(message):
                pass

            @staticmethod
            def warn(message):
                pass

        controller.get_logger = lambda: _Logger()

        class FakeCostmap:
            @staticmethod
            def is_static_free_world(x, y):
                return True

        inner = mock.Mock()
        inner.cm = FakeCostmap()
        inner._rear_clearance.return_value = 1.0
        controller.nav = mock.Mock()
        controller.nav.controller = inner

        self.assertTrue(controller._start_nav_recovery((-1.8, -2.6)))
        self.assertEqual(controller._nav_recovery_phase, "backup")
        self.assertEqual(controller._nav_recovery_attempts, 1)

    def test_snapshot_client_has_flow_level_nav_recovery(self):
        controller = client.ContinuousOrderController.__new__(
            client.ContinuousOrderController)
        controller.flow_phase = "nav_to_delivery"
        controller._nav_recovery_phase = None
        controller._nav_recovery_attempts = 0
        controller.base_xy = np.array([0.58, 2.23])
        controller.base_yaw = 2.69
        controller.laser_msg = object()
        controller.now = mock.Mock(return_value=100.0)
        controller.set_twist = mock.Mock()

        class _Logger:
            @staticmethod
            def info(message):
                pass

            @staticmethod
            def warn(message):
                pass

        controller.get_logger = lambda: _Logger()
        inner = mock.Mock()
        inner._rear_clearance.return_value = 1.0
        controller.nav = mock.Mock()
        controller.nav.controller = inner

        self.assertTrue(controller._start_nav_recovery((-1.8, -2.6)))
        self.assertEqual(controller._nav_recovery_phase, "backup")

    def test_flow_level_nav_recovery_refuses_when_rear_is_close(self):
        controller = client.ContinuousOrderController.__new__(
            client.ContinuousOrderController)
        controller.flow_phase = "nav_to_delivery"
        controller._nav_recovery_phase = None
        controller._nav_recovery_attempts = 0
        controller.base_xy = np.array([0.60, 2.24])
        controller.base_yaw = 2.72
        controller.laser_msg = object()
        controller.now = mock.Mock(return_value=100.0)
        controller.set_twist = mock.Mock()

        class _Logger:
            @staticmethod
            def info(message):
                pass

            @staticmethod
            def warn(message):
                pass

        controller.get_logger = lambda: _Logger()
        inner = mock.Mock()
        inner._rear_clearance.return_value = 0.30
        controller.nav = mock.Mock()
        controller.nav.controller = inner

        self.assertFalse(controller._start_nav_recovery((-1.8, -2.6)))
        self.assertIsNone(controller._nav_recovery_phase)

    def test_hint_view_exhaustion_emits_failover_event(self):
        controller = client.ContinuousOrderController.__new__(
            client.ContinuousOrderController)
        controller.inventory_scan_hint_active = True
        controller.memory_active_hint = ("B", "L3")
        controller.memory_failed_hint = None

        with mock.patch.object(
                client.IntegratedNavPickPlace,
                "_restore_full_scan_after_inventory_hint",
                autospec=True) as restore:
            restore.side_effect = lambda ctrl: setattr(
                ctrl, "inventory_scan_hint_active", False)
            controller._restore_full_scan_after_inventory_hint()

        self.assertEqual(controller.memory_failed_hint, ("B", "L3"))
        self.assertIsNone(controller.memory_active_hint)

    def test_master_memory_hint_can_enter_direct_slot_flow(self):
        controller = client.IntegratedNavPickPlace.__new__(
            client.IntegratedNavPickPlace)
        controller.close_recheck = True
        controller.configure_direct_slot_target = mock.Mock(return_value=True)
        controller.configure_inventory_scan_hint = mock.Mock()
        controller.scan_preferred_x = None
        controller.memory_active_hint = None
        controller.memory_last_scan_station_x = None
        controller.get_logger = mock.Mock(return_value=mock.Mock())
        hint = {
            "x": 0.035,
            "z": 0.852,
            "shelf": "C",
            "level": "L2",
            "column": "3",
            "confidence": 0.91,
            "world_y": 3.18,
            "world_z": 0.90,
        }

        controller._apply_memory_hint(hint, "test")

        controller.configure_direct_slot_target.assert_called_once_with(
            "C", "L2", "3", product_y=3.18, product_z=0.90)
        controller.configure_inventory_scan_hint.assert_not_called()
        self.assertEqual(controller.memory_active_hint, ("C", "L2"))
        self.assertEqual(controller.scan_preferred_x, 0.035)


if __name__ == "__main__":
    unittest.main()
