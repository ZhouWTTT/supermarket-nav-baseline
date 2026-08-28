"""Regression tests for optional-ArUco close recheck."""

import pathlib
import sys
import unittest
from collections import deque
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
    def _controller(marker_id=9, source="position_fallback"):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.target_kind = "zhijin"
        controller.target_marker_id = marker_id
        controller.target_world = np.array([-1.08, 3.292, 0.549])
        controller.target_localisation_source = source
        controller.committed_slot = ("A", "L1", "3")
        controller.excluded_slot_keys = set()
        controller.excluded_slot_keys_by_kind = {}
        controller.shelf_level = "lower"
        controller.recheck_target_seen = True
        controller.marker_positions = deque()
        controller.depth_target_samples = deque()
        controller.last_association_pair = None
        controller.association_candidate_id = None
        controller.association_confirmation_count = 0
        controller.direct_slot_target_active = False
        controller.get_logger = mock.Mock(return_value=mock.Mock())
        return controller

    @staticmethod
    def _near_detection():
        return {
            "class": "zhijin",
            "world": [-1.07, 3.212, 0.653],
            "conf": 0.97,
        }

    def test_no_marker_uses_yolo_depth(self):
        controller = self._controller(marker_id=9)
        with mock.patch.object(pick, "marker_below_yolo", return_value=None):
            matched, source = controller._recheck_detection_matches(
                self._near_detection(), [])
        self.assertTrue(matched)
        self.assertEqual(source, "depth(no-associated-aruco)")

    def test_wrong_marker_does_not_veto_valid_yolo_depth(self):
        controller = self._controller(marker_id=9)
        with mock.patch.object(
                pick, "marker_below_yolo", return_value={"id": 13}):
            matched, source = controller._recheck_detection_matches(
                self._near_detection(), [{"id": 13}])
        self.assertTrue(matched)
        self.assertEqual(source, "depth(ignore-aruco=13,expected=9)")

    def test_matching_marker_cannot_override_wrong_depth(self):
        controller = self._controller(marker_id=9)
        detection = {
            "class": "zhijin",
            "world": [-0.60, 3.60, 0.90],
            "conf": 0.99,
        }
        with mock.patch.object(
                pick, "marker_below_yolo", return_value={"id": 9}):
            matched, source = controller._recheck_detection_matches(
                detection, [{"id": 9}])
        self.assertFalse(matched)
        self.assertEqual(source, "aruco-depth-conflict")

    def test_expected_marker_x_only_match_yields_aruco_x(self):
        controller = self._controller(marker_id=9)
        # Expected marker decoded with the right world X but a PnP Z that is
        # one shelf level off: the close-range relaxation still accepts the
        # x-only association (depth+aruco-x) instead of dropping the code.
        detection = self._near_detection()
        with mock.patch.object(
                pick, "marker_below_yolo", return_value=None):
            markers = [{
                "id": 9,
                "position_world": [-1.075, 3.22, 1.150],
            }]
            matched, source = controller._recheck_detection_matches(
                detection, markers)
        self.assertTrue(matched)
        self.assertEqual(source, "depth+aruco-x")

    def test_position_fallback_survives_inconclusive_optional_recheck(self):
        controller = self._controller(source="position_fallback")
        self.assertTrue(controller._depth_recheck_fallback_available())

    def test_inconclusive_optional_recheck_proceeds_with_depth_target(self):
        controller = self._controller(source="position_fallback")
        controller.recheck_poses = (("lower", 0.3, 0.0, -0.45),)
        controller.recheck_confirmation_times = deque()
        controller.recheck_last_yolo_stamp = None
        controller.recheck_last_confirmation_source = None
        controller.recheck_conflict_marker_id = None
        controller.recheck_conflict_count = 0
        controller.recheck_conflict_confirmed = False
        controller.recheck_fresh_yolo_frames = 3
        controller.recheck_target_seen = True
        controller.scan_camera_ready_since = 1.0
        controller._recheck_passed = False
        controller._start_grasp_settle = mock.Mock()
        logger = mock.Mock()
        controller.get_logger = mock.Mock(return_value=logger)

        controller._recheck_fail()

        self.assertTrue(controller._recheck_passed)
        controller._start_grasp_settle.assert_called_once_with()
        self.assertIsNotNone(controller.target_world)
        self.assertEqual(controller.committed_slot, ("A", "L1", "3"))

    def test_inconclusive_recheck_without_target_view_denies_fallback(self):
        """无码超时时若复核从未见过目标品类，禁止 memory 兜底抓取。"""
        controller = self._controller(source="position_fallback")
        controller.recheck_target_seen = False
        self.assertFalse(controller._depth_recheck_fallback_available())

        controller.recheck_target_seen = True
        self.assertTrue(controller._depth_recheck_fallback_available())

    def test_aruco_only_target_has_no_depth_fallback(self):
        controller = self._controller(source="aruco(no-depth)")
        self.assertFalse(controller._depth_recheck_fallback_available())


@unittest.skipIf(pick is None, f"runtime dependencies unavailable: {IMPORT_ERROR}")
class RecheckDepthCorrectionTests(unittest.TestCase):
    """ArUco 成功识别后的深度相机矫正（仅限识别成功分支，非阻塞）。"""

    @staticmethod
    def _sphere_controller(source="aruco(no-depth)"):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.target_kind = "pingguo"
        controller.target_marker_id = 5
        controller.target_world = np.array([-1.515, 3.243, 0.886])
        controller.target_localisation_source = source
        controller.committed_slot = ("A", "L2", "3")
        controller.excluded_slot_keys = set()
        controller.excluded_slot_keys_by_kind = {}
        controller.shelf_level = "middle"
        controller.is_top_shelf = False
        controller.recheck_last_confirmation_source = None
        controller.recheck_last_confirmation_detection = None
        logger = mock.Mock()
        controller.get_logger = mock.Mock(return_value=logger)
        return controller

    @staticmethod
    def _near_detection(x=-1.525, y=3.210, z=0.860, delta_ms=20.0):
        return {
            "class": "pingguo",
            "world": [x, y, z],
            "conf": 0.97,
            "depth_delta_ms": delta_ms,
        }

    def test_eligible_sources(self):
        controller = self._sphere_controller()
        self.assertTrue(controller._recheck_depth_correction_eligible(
            "depth+aruco"))
        self.assertTrue(controller._recheck_depth_correction_eligible(
            "depth+aruco-x"))
        # 纯 ArUco 直达锁定 + 复核通过（含深度通过但无码/错码）也允许矫正。
        controller.target_localisation_source = "aruco(no-depth)"
        self.assertTrue(controller._recheck_depth_correction_eligible(
            "depth(no-associated-aruco)"))
        self.assertTrue(controller._recheck_depth_correction_eligible(
            "depth(ignore-aruco=13,expected=5)"))

    def test_depth_only_source_without_direct_aruco_not_eligible(self):
        controller = self._sphere_controller(source="aruco+depth")
        self.assertFalse(controller._recheck_depth_correction_eligible(
            "depth(no-associated-aruco)"))
        self.assertFalse(controller._recheck_depth_correction_eligible(
            "depth(ignore-aruco=13,expected=5)"))
        self.assertFalse(controller._recheck_depth_correction_eligible(
            "class-mismatch"))

    def test_depth_aruco_source_corrects_sphere_target(self):
        controller = self._sphere_controller()
        controller.recheck_last_confirmation_source = "depth+aruco"
        controller.recheck_last_confirmation_detection = (
            self._near_detection())
        delta = controller._apply_recheck_depth_correction()
        self.assertIsNotNone(delta)
        self.assertAlmostEqual(float(delta[0]), -0.010, places=6)
        self.assertAlmostEqual(float(delta[1]), 0.002, places=6)
        np.testing.assert_allclose(
            controller.target_world,
            [-1.525, 3.210 + pick.PRODUCT_HALF_DEPTH_M["pingguo"], 0.886])

    def test_aruco_x_source_corrects_target(self):
        controller = self._sphere_controller()
        controller.recheck_last_confirmation_source = "depth+aruco-x"
        controller.recheck_last_confirmation_detection = (
            self._near_detection())
        self.assertIsNotNone(
            controller._apply_recheck_depth_correction())

    def test_direct_aruco_target_corrected_with_depth_only_source(self):
        # ARUCO 直达锁定（扫描期无深度）的目标，复核深度通过后同样矫正。
        controller = self._sphere_controller(source="aruco(no-depth)")
        controller.recheck_last_confirmation_source = (
            "depth(no-associated-aruco)")
        controller.recheck_last_confirmation_detection = (
            self._near_detection())
        self.assertIsNotNone(
            controller._apply_recheck_depth_correction())

    def test_depth_only_target_not_corrected(self):
        # 已深度融合目标 + 复核无码通过：不属于允许矫正的识别成功分支。
        controller = self._sphere_controller(source="aruco+depth")
        controller.recheck_last_confirmation_source = (
            "depth(no-associated-aruco)")
        controller.recheck_last_confirmation_detection = (
            self._near_detection())
        before = controller.target_world.copy()
        self.assertIsNone(controller._apply_recheck_depth_correction())
        np.testing.assert_allclose(controller.target_world, before)

    def test_large_delta_skipped(self):
        controller = self._sphere_controller()
        controller.recheck_last_confirmation_source = "depth+aruco"
        controller.recheck_last_confirmation_detection = (
            self._near_detection(x=-1.300, y=3.210, z=0.860))
        before = controller.target_world.copy()
        self.assertIsNone(controller._apply_recheck_depth_correction())
        np.testing.assert_allclose(controller.target_world, before)

    def test_stale_depth_skipped(self):
        controller = self._sphere_controller()
        controller.recheck_last_confirmation_source = "depth+aruco"
        controller.recheck_last_confirmation_detection = (
            self._near_detection(delta_ms=500.0))
        before = controller.target_world.copy()
        self.assertIsNone(controller._apply_recheck_depth_correction())
        np.testing.assert_allclose(controller.target_world, before)


if __name__ == "__main__":
    unittest.main()
