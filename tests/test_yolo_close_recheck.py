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
    def test_heweidao_post_extension_moves_forward_and_down(self):
        nominal = np.array([0.5, 3.25, 0.56])

        lowered = pick.generic_post_extend_world(nominal, "heweidao")
        ordinary = pick.generic_post_extend_world(nominal, "maidong")

        self.assertAlmostEqual(
            lowered[1] - nominal[1],
            pick.GENERIC_POST_CONTACT_EXTENSION_M)
        self.assertAlmostEqual(
            nominal[2] - lowered[2],
            pick.GENERIC_POST_EXTEND_Z_DROP_M_BY_KIND["heweidao"])
        self.assertAlmostEqual(ordinary[2], nominal[2])

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

    def test_direct_slot_target_uses_fixed_geometry_and_sets_align(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.target_kind = "maidong"
        controller.excluded_slot_keys = set()

        class _Logger:
            def warn(self, _message):
                pass

        controller.get_logger = lambda: _Logger()
        committed = {}

        def commit(target_world, marker_id, source, extra="", shelf=None):
            committed.update(
                target_world=np.asarray(target_world),
                marker_id=marker_id,
                source=source,
                extra=extra,
                shelf=shelf)

        controller._commit_localised_target = commit
        self.assertTrue(controller.configure_direct_slot_target(
            "C", "L2", "3", product_y=3.1626))
        self.assertEqual(committed["source"], "memory_direct")
        self.assertIsNone(committed["marker_id"])
        self.assertEqual(committed["shelf"], "C")
        self.assertAlmostEqual(committed["target_world"][0], 0.035 + 0.22)
        # 记忆深度 y 可能偏近/偏远，固定货架平面 y 才是可靠抓取纵向位置。
        self.assertAlmostEqual(
            committed["target_world"][1], pick.SHELF_PRODUCT_CENTER_Y_M)
        self.assertAlmostEqual(
            committed["target_world"][2],
            pick.SHELF_SURFACE_Z_M["middle"]
            + pick.PRODUCT_HALF_HEIGHT_M["maidong"])

    def test_detection_world_falls_back_to_front_world(self):
        detection = {"front_world": [1.0, 2.0, 3.0]}
        world = pick.ShelfPickController._detection_world(detection)
        self.assertIsNotNone(world)
        np.testing.assert_allclose(world, [1.0, 2.0, 3.0])

    def test_recheck_uses_front_world_when_center_depth_missing(self):
        controller = self._controller(marker_id=None)
        detection = {
            "front_world": [-1.07, 3.212, 0.653],
            "conf": 0.97,
        }
        with mock.patch.object(
                pick, "marker_below_yolo", return_value=None):
            matched, source = controller._recheck_detection_matches(
                detection, [])
        self.assertTrue(matched)
        self.assertEqual(source, "depth(no-aruco)")

    def test_grasp_settle_waits_for_stationary_base(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.base_xy = np.array([1.0, 2.0])
        controller.base_yaw = 0.0
        controller.des_linear = 0.0
        controller.des_angular = 0.0
        controller.now = mock.Mock(return_value=100.0)
        controller._proceed_to_deploy = mock.Mock()
        controller.set_state = mock.Mock()

        controller._start_grasp_settle()
        controller.set_state.assert_called_once_with(
            pick.STATE_GRASP_SETTLE)
        controller.now.return_value = 100.25
        controller._grasp_settle_tick()
        controller._proceed_to_deploy.assert_called_once_with()

    def test_top_pregrasp_can_switch_arm_once(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.grasp_arm = "r"
        controller.target_world = np.array([0.92, 3.243, 1.2365])
        controller.align_base_x = 0.81
        controller.align_base_y = 2.553
        controller.is_top_shelf = True
        controller.ik_retry_forward_m = 0.02
        controller.deploy_retry_count = 2
        controller._recheck_passed = False
        controller._grasp_arm_switch_count = 0
        controller.set_state = mock.Mock()

        class _Logger:
            @staticmethod
            def warn(message):
                pass

        controller.get_logger = lambda: _Logger()

        self.assertTrue(controller._switch_grasp_arm_retry("test"))
        self.assertEqual(controller.grasp_arm, "l")
        self.assertAlmostEqual(controller.target_world[0], 0.92)
        self.assertAlmostEqual(controller.align_base_x, 1.02)
        self.assertAlmostEqual(controller.align_base_y, 2.553)
        self.assertTrue(controller._recheck_passed)
        self.assertEqual(controller._grasp_arm_switch_count, 1)
        controller.set_state.assert_called_once_with(pick.STATE_ALIGN)

        self.assertFalse(controller._switch_grasp_arm_retry("test"))


if __name__ == "__main__":
    unittest.main()
