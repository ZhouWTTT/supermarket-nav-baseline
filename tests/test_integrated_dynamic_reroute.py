"""Regression tests for direct-only in-flight memory rerouting."""

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
    import yolo_aruco_shelf_pick as pick
except ImportError as exc:  # Host without ROS/discoverse; Client runs it.
    IMPORT_ERROR = exc
    np = None
    integrated = None
    pick = None


@unittest.skipIf(
    integrated is None,
    f"runtime dependencies unavailable: {IMPORT_ERROR}",
)
class DynamicRerouteControllerTests(unittest.TestCase):
    @staticmethod
    def _controller(direct_accepted):
        controller = integrated.IntegratedNavPickPlace.__new__(
            integrated.IntegratedNavPickPlace)
        controller.memory_file = pathlib.Path("/tmp/not-read-in-mock.json")
        controller.state = pick.STATE_GO_SCAN
        controller.target_world = None
        controller.memory_failed_hint = None
        controller.memory_rerouted = False
        controller._memory_last_reroute_check = 0.0
        controller.memory_reroute_not_before = 1.0
        controller.dynamic_direct_enabled = True
        controller.scan_station_order = None
        controller.scan_preferred_x = 1.8
        controller.base_xy = np.array([1.51, -1.62])
        controller.memory_exhausted_shelves = set()
        controller._update_memory_scan_progress = mock.Mock()
        controller._select_live_memory_hint = mock.Mock(return_value={
            "slot_key": "L1|D|1",
            "shelf": "D",
            "level": "L1",
            "column": "1",
            "x": 0.92,
            "z": 0.50,
            "confidence": 0.95,
            "travel": 4.05,
            "observed_distance": 1.0,
        })
        controller._try_apply_direct_memory_hint = mock.Mock(
            return_value=direct_accepted)
        controller._apply_memory_hint = mock.Mock()
        return controller

    def test_failed_direct_admission_keeps_current_goal(self):
        controller = self._controller(direct_accepted=False)

        controller._memory_route_tick()

        controller._select_live_memory_hint.assert_called_once_with(
            reliable_only=True,
            min_last_seen=1.0,
            require_direct=True)
        controller._apply_memory_hint.assert_not_called()
        self.assertFalse(controller.memory_rerouted)

    def test_only_accepted_concrete_slot_consumes_reroute(self):
        controller = self._controller(direct_accepted=True)

        controller._memory_route_tick()

        controller._apply_memory_hint.assert_not_called()
        self.assertTrue(controller.memory_rerouted)

    def test_live_memory_selects_nearest_pending_kind(self):
        controller = integrated.IntegratedNavPickPlace.__new__(
            integrated.IntegratedNavPickPlace)
        controller.memory_file = pathlib.Path("/tmp/matrix.json")
        controller.target_kind = "kouxiangtang"
        controller.opportunistic_target_kinds = (
            "kouxiangtang", "kele", "chengzi")
        controller.base_xy = np.array([0.0, 0.0])
        controller.memory_confidence_threshold = 0.90
        controller.excluded_slot_keys = set()
        controller.excluded_slot_keys_by_kind = {}
        controller.memory_exhausted_shelves = set()
        controller.memory_exhausted_shelves_by_kind = {}
        controller.memory_failed_hint_levels = set()
        controller.memory_failed_hint_levels_by_kind = {}

        routes = {
            "kouxiangtang": {"travel": 3.0, "confidence": 0.97},
            "kele": {"travel": 1.2, "confidence": 0.93},
            "chengzi": {"travel": 2.0, "confidence": 0.98},
        }

        def select(kind, *_args, **_kwargs):
            return dict(routes[kind])

        with mock.patch.object(
                integrated, "read_memory_document", return_value={}), \
                mock.patch.object(
                    integrated, "primary_candidates_from_document",
                    return_value=[]), \
                mock.patch.object(
                    integrated, "candidates_from_document", return_value=[]), \
                mock.patch.object(
                    integrated, "select_memory_route_hint",
                    side_effect=select):
            hint = controller._select_live_memory_hint(reliable_only=True)

        self.assertEqual(hint["target_kind"], "kele")
        self.assertEqual(hint["travel"], 1.2)

    def test_matrix_hint_switches_kind_before_scan_route(self):
        controller = integrated.IntegratedNavPickPlace.__new__(
            integrated.IntegratedNavPickPlace)
        controller.target_kind = "kouxiangtang"
        controller._set_pregrasp_target_kind = mock.Mock(
            side_effect=lambda kind: setattr(controller, "target_kind", kind))
        controller.get_logger = mock.Mock(return_value=mock.Mock())

        previous = controller._activate_memory_target_kind({
            "target_kind": "kele",
            "travel": 0.8,
        }, "test")

        self.assertEqual(previous, "kouxiangtang")
        self.assertEqual(controller.target_kind, "kele")
        controller._set_pregrasp_target_kind.assert_called_once_with("kele")


if __name__ == "__main__":
    unittest.main()
