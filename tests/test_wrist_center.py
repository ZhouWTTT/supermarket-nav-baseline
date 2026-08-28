"""Unit tests for wrist-camera grasp centering (default-off feature).

The feature consumes wrist YOLO records of the active arm during DEPLOY and
applies a single, clamped lateral correction to the contact target.  Tests
cover the projection round-trip and the centering gates.
"""

import pathlib
import sys
import threading
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


def _camera_record(position, rotation, bbox_xyxy, conf=0.95):
    return {
        "class": "pingguo",
        "conf": conf,
        "camera": "right",
        "camera_world_position": list(map(float, position)),
        "camera_world_rotation": np.asarray(rotation).reshape(-1).tolist(),
        "bbox_xyxy": list(map(float, bbox_xyxy)),
        "stamp_ns": 1_000_000_000,
    }


def _wrist_controller():
    controller = pick.ShelfPickController.__new__(
        pick.ShelfPickController)
    controller.lock = threading.Lock()
    controller.target_kind = "pingguo"
    controller.grasp_arm = "r"
    controller.use_dual_tissue_grasp = False
    controller.use_sphere_grasp = False
    controller.wrist_center_enabled = True
    controller.wrist_yolo_frames = deque(maxlen=30)
    controller._wrist_center_applied = False
    controller.forward_contact_world = np.array([0.0, 3.243, 0.9])
    controller.sphere_contact_world = None
    controller.post_extend_nominal_world = None
    controller.post_extend_target_world = None
    controller.post_extend_arm_joints = None
    controller.approach_arm_joints = []
    controller.state = pick.STATE_DEPLOY
    logger = mock.Mock()
    controller.get_logger = mock.Mock(return_value=logger)
    return controller


def _aimed_camera_pose(contact):
    """Camera pose whose optical axis points at ``contact`` (world)."""
    position = np.array([0.0, 3.0, 0.95])
    direction = np.asarray(contact, dtype=float) - position
    z_cam = direction / np.linalg.norm(direction)
    x_cam = np.cross(z_cam, np.array([0.0, 0.0, 1.0]))
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    return position, np.column_stack([x_cam, y_cam, z_cam])


@unittest.skipIf(pick is None, f"runtime dependencies unavailable: {IMPORT_ERROR}")
class WristProjectionTests(unittest.TestCase):
    def test_projection_roundtrip_recovers_world_x(self):
        contact = np.array([0.0, 3.243, 0.9])
        position, rotation = _aimed_camera_pose(contact)
        record = _camera_record(position, rotation, [0, 0, 10, 10])

        px, _ = pick.ShelfPickController._project_to_wrist_pixels(
            contact, record)
        self.assertIsNotNone(px)
        self.assertAlmostEqual(float(px[0]), 320.0, delta=0.5)
        self.assertAlmostEqual(float(px[1]), 240.0, delta=0.5)

        product = contact.copy()
        product[0] = 0.02
        px_product, _ = pick.ShelfPickController._project_to_wrist_pixels(
            product, record)
        world_x = pick.ShelfPickController._wrist_pixel_to_world_x(
            float(px_product[0]), float(px_product[1]),
            record, pick.SHELF_PRODUCT_CENTER_Y_M)
        self.assertIsNotNone(world_x)
        self.assertAlmostEqual(world_x, 0.02, delta=0.003)


@unittest.skipIf(pick is None, f"runtime dependencies unavailable: {IMPORT_ERROR}")
class WristCenterTests(unittest.TestCase):
    def _detection(self, product_x, conf=0.95):
        contact = np.array([0.0, 3.243, 0.9])
        position, rotation = _aimed_camera_pose(contact)
        product = contact.copy()
        product[0] = product_x
        px, _ = pick.ShelfPickController._project_to_wrist_pixels(
            product, _camera_record(position, rotation, [0, 0, 10, 10]))
        u, v = float(px[0]), float(px[1])
        return _camera_record(
            position, rotation, [u - 20, v - 20, u + 20, v + 20],
            conf=conf)

    def test_disabled_is_noop(self):
        controller = _wrist_controller()
        controller.wrist_center_enabled = False
        controller.wrist_yolo_frames.append((1, [self._detection(0.02)]))
        controller._maybe_wrist_center_contact()
        self.assertAlmostEqual(controller.forward_contact_world[0], 0.0)
        self.assertFalse(controller._wrist_center_applied)

    def test_no_detection_is_noop(self):
        controller = _wrist_controller()
        controller._maybe_wrist_center_contact()
        self.assertAlmostEqual(controller.forward_contact_world[0], 0.0)
        self.assertTrue(controller._wrist_center_applied)

    def test_low_confidence_is_noop(self):
        controller = _wrist_controller()
        controller.wrist_yolo_frames.append(
            (1, [self._detection(0.02, conf=0.60)]))
        controller._maybe_wrist_center_contact()
        self.assertAlmostEqual(controller.forward_contact_world[0], 0.0)

    def test_corrects_within_clamp(self):
        controller = _wrist_controller()
        controller.wrist_yolo_frames.append((1, [self._detection(0.02)]))
        controller._maybe_wrist_center_contact()
        self.assertAlmostEqual(
            controller.forward_contact_world[0], 0.02, delta=0.004)
        self.assertTrue(controller._wrist_center_applied)

    def test_exceeding_clamp_is_noop(self):
        controller = _wrist_controller()
        controller.wrist_yolo_frames.append((1, [self._detection(0.05)]))
        controller._maybe_wrist_center_contact()
        self.assertAlmostEqual(controller.forward_contact_world[0], 0.0)

    def test_single_shot(self):
        controller = _wrist_controller()
        controller.wrist_yolo_frames.append((1, [self._detection(0.02)]))
        controller._maybe_wrist_center_contact()
        first_x = controller.forward_contact_world[0]
        controller.wrist_yolo_frames.append((2, [self._detection(-0.02)]))
        controller._maybe_wrist_center_contact()
        self.assertAlmostEqual(controller.forward_contact_world[0], first_x)


if __name__ == "__main__":
    unittest.main()
