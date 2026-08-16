"""Regression tests for memory-driven shelf rerouting."""

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
    import snapshot_pick_client as client
except ImportError as exc:  # Host without ROS/discoverse; Client runs it.
    IMPORT_ERROR = exc
    client = None


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


if __name__ == "__main__":
    unittest.main()
