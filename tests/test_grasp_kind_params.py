"""Kind-switch grasp parameter consistency tests.

Cross-kind switching is disabled at the runner level, but the worker must
still be safe if it ever runs with multiple candidates: every kind-dependent
grasp parameter that is initialised from the kind must be refreshed by
``_set_pregrasp_target_kind``.
"""

import pathlib
import sys
import unittest
from collections import deque


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
class GraspKindParameterTests(unittest.TestCase):
    @staticmethod
    def _controller():
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.state = pick.STATE_GO_SCAN
        controller.target_marker_id = None
        controller.target_world = None
        controller.inventory_scan_hint_active = False
        controller.skipped_tissue_markers = set()
        controller.skipped_tissue_slots = set()
        controller.scan_unlocked_markers = {}
        controller.scan_unlocked_boxes = {}
        controller.yolo_frames = deque()
        controller.marker_positions = deque()
        controller.depth_target_samples = deque()
        controller.association_candidate_id = None
        controller.association_confirmation_count = 0
        controller.last_association_pair = None
        controller.default_scan_poses = pick.SCAN_CAMERA_POSES
        controller.scan_poses = pick.SCAN_CAMERA_POSES
        return controller

    def test_kind_switch_refreshes_all_kind_dependent_params(self):
        controller = self._controller()
        controller.target_kind = "pingguo"
        controller.product_height = pick.PRODUCT_CENTER_ABOVE_MARKER_M["pingguo"]
        controller.product_grasp_width = pick.PRODUCT_GRASP_WIDTH_M["pingguo"]
        controller.grip_preshape_command = 0.9
        controller.use_dual_tissue_grasp = False

        controller._set_pregrasp_target_kind("zhijin")

        self.assertEqual(controller.target_kind, "zhijin")
        self.assertTrue(controller.use_dual_tissue_grasp)
        self.assertEqual(
            controller.product_height,
            pick.PRODUCT_CENTER_ABOVE_MARKER_M["zhijin"])
        self.assertEqual(
            controller.product_grasp_width,
            pick.PRODUCT_GRASP_WIDTH_M["zhijin"])
        self.assertAlmostEqual(
            controller.grip_preshape_command,
            float(np.clip(
                (pick.PRODUCT_GRASP_WIDTH_M["zhijin"]
                 + pick.GRIP_PRESHAPE_CLEARANCE_M)
                / pick.GRIPPER_MAX_OPENING_M,
                pick.GRIP_CLOSE + pick.GENERIC_EMPTY_GRIP_MARGIN,
                pick.GRIP_OPEN)))

    def test_switch_away_from_tissue_clears_dual(self):
        controller = self._controller()
        controller.target_kind = "zhijin"
        controller.product_height = pick.PRODUCT_CENTER_ABOVE_MARKER_M["zhijin"]
        controller.product_grasp_width = pick.PRODUCT_GRASP_WIDTH_M["zhijin"]
        controller.grip_preshape_command = 0.5
        controller.use_dual_tissue_grasp = True

        controller._set_pregrasp_target_kind("heweidao")

        self.assertEqual(controller.target_kind, "heweidao")
        self.assertFalse(controller.use_dual_tissue_grasp)
        self.assertEqual(
            controller.product_height,
            pick.PRODUCT_CENTER_ABOVE_MARKER_M["heweidao"])


if __name__ == "__main__":
    unittest.main()
