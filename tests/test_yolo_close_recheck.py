"""Regression tests for optional-ArUco close recheck and YOLO fallback."""

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
except ImportError as exc:  # Host without ROS/discoverse; Client container runs it.
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
        # Same values seen in the failed fifth-order shupian close recheck.
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

    def test_position_fallback_does_not_create_a_fake_decoded_marker(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.revisit_box_world = np.array([-1.064, 3.217, 0.708])
        controller.revisit_box_conf = 0.97
        controller.revisit_box_confirmations = 4
        controller.target_kind = "shupian"
        controller.excluded_marker_ids = set()
        controller.excluded_slot_keys = set()
        controller.recheck_marker_skips = set()
        controller.skipped_tissue_markers = set()
        controller.target_marker_id = None
        controller.target_physical_marker_id = None
        controller._recheck_passed = False

        class _Logger:
            def warn(self, _message):
                pass

        controller.get_logger = lambda: _Logger()
        committed = {}

        def commit(target_world, marker_id, source, extra="", **_kwargs):
            committed.update(
                target_world=np.asarray(target_world), marker_id=marker_id,
                source=source, extra=extra)

        controller._commit_localised_target = commit
        self.assertTrue(controller._try_position_fallback())
        self.assertIsNone(controller.target_marker_id)
        self.assertIsNone(committed["marker_id"])
        self.assertEqual(committed["source"], "position_fallback")
        self.assertIn("inferred_slot_marker=9", committed["extra"])

    def test_position_fallback_respects_yolo_only_slot_exclusion(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.revisit_box_world = np.array([-1.064, 3.217, 0.708])
        controller.revisit_box_conf = 0.97
        controller.revisit_box_confirmations = 4
        controller.target_kind = "shupian"
        controller.excluded_marker_ids = set()
        controller.excluded_slot_keys = {"L1|B|1"}
        controller.recheck_marker_skips = set()
        controller.skipped_tissue_markers = set()

        class _Logger:
            def warn(self, _message):
                pass

        controller.get_logger = lambda: _Logger()
        controller._commit_localised_target = mock.Mock()
        self.assertFalse(controller._try_position_fallback())
        controller._commit_localised_target.assert_not_called()


if __name__ == "__main__":
    unittest.main()
