"""Regression tests for optional-ArUco close recheck."""

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
    import yolo_aruco_shelf_pick as pick
except ImportError as exc:  # Host without ROS/discoverse; Client runs it.
    IMPORT_ERROR = exc
    np = None
    pick = None


@unittest.skipIf(pick is None, f"runtime dependencies unavailable: {IMPORT_ERROR}")
class CloseRecheckFallbackTests(unittest.TestCase):
    @staticmethod
    def _controller(marker_id=None):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.target_marker_id = marker_id
        controller.target_world = np.array([-1.08, 3.292, 0.549])
        return controller

    @staticmethod
    def _near_detection():
        return {"world": [-1.07, 3.212, 0.653], "conf": 0.97}

    def test_unidentified_target_ignores_adjacent_marker_and_uses_depth(self):
        controller = self._controller(marker_id=None)
        with mock.patch.object(
                pick, "marker_below_yolo", return_value={"id": 13}):
            matched, source = controller._recheck_detection_matches(
                self._near_detection(), [{"id": 13}])
        self.assertTrue(matched)
        self.assertEqual(source, "depth(ignore-aruco=13)")

    def test_wrong_marker_does_not_veto_valid_depth(self):
        controller = self._controller(marker_id=9)
        with mock.patch.object(
                pick, "marker_below_yolo", return_value={"id": 13}):
            matched, source = controller._recheck_detection_matches(
                self._near_detection(), [{"id": 13}])
        self.assertTrue(matched)
        self.assertEqual(source, "depth(ignore-aruco=13,expected=9)")

    def test_wrong_marker_and_wrong_depth_still_fail(self):
        controller = self._controller(marker_id=9)
        detection = {"world": [-0.60, 3.60, 0.90], "conf": 0.99}
        with mock.patch.object(
                pick, "marker_below_yolo", return_value={"id": 13}):
            matched, source = controller._recheck_detection_matches(
                detection, [{"id": 13}])
        self.assertFalse(matched)
        self.assertEqual(source, "depth(ignore-aruco=13,expected=9)")

    def test_matching_real_marker_can_confirm_directly(self):
        controller = self._controller(marker_id=9)
        detection = {"world": [-0.60, 3.60, 0.90], "conf": 0.99}
        with mock.patch.object(
                pick, "marker_below_yolo", return_value={"id": 9}):
            matched, source = controller._recheck_detection_matches(
                detection, [{"id": 9}])
        self.assertTrue(matched)
        self.assertEqual(source, "aruco")


if __name__ == "__main__":
    unittest.main()
