"""Regression tests for effort-limit standstill during horizontal refinement.

The velocity topic keeps reporting ±0.1 rad/s on a loaded joint that is
physically static at its effort limit.  Standstill must therefore be judged
from measured joint positions, otherwise the horizontal refine stage dead-waits
its full 12 s timeout before the recovery path descends in place.
"""

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
except ImportError as exc:  # Host without ROS/discoverse; Client runs it.
    IMPORT_ERROR = exc
    np = None
    integrated = None


@unittest.skipIf(
    integrated is None,
    f"runtime dependencies unavailable: {IMPORT_ERROR}",
)
class PlaceRefineStandstillTests(unittest.TestCase):
    @staticmethod
    def _controller():
        controller = integrated.IntegratedNavPickPlace.__new__(
            integrated.IntegratedNavPickPlace)
        controller.grasp_arm = "r"
        controller.joints = {
            "right_arm_joint1": -0.9820,
            "right_arm_joint2": -1.8926,
            "right_arm_joint3": 1.1722,
            "right_arm_joint4": -2.2280,
            "right_arm_joint5": -1.6130,
            "right_arm_joint6": -2.4504,
            "left_arm_joint1": 0.0,
            "left_arm_joint2": -0.166,
            "left_arm_joint3": 0.032,
            "left_arm_joint4": 0.0,
            "left_arm_joint5": 1.571,
            "left_arm_joint6": 2.223,
            "slide_joint": 0.1746,
        }
        # j6 reports 0.12 rad/s while its position is static: this is exactly
        # the noise observed on the fifth delivery's effort-limited joint.
        controller.joint_velocities = {
            "right_arm_joint1": -0.002,
            "right_arm_joint2": -0.001,
            "right_arm_joint3": 0.000,
            "right_arm_joint4": 0.005,
            "right_arm_joint5": -0.005,
            "right_arm_joint6": 0.120,
            "slide_joint": 0.00003,
        }
        controller.des_left_arm = np.array(
            [0.0, -0.166, 0.032, 0.0, 1.571, 2.223])
        controller.des_right_arm = np.array(
            [-0.9901, -1.8946, 1.1720, -2.2151, -1.7484, -2.4508])
        controller.des_slide = 0.1692
        controller.commands_ready_since = None
        controller._place_refine_target_sent_at = 100.0
        controller._place_refine_motion_stable_since = None
        controller._place_refine_motion_anchor = None
        logger = mock.Mock()
        controller.get_logger = mock.Mock(return_value=logger)
        return controller

    def test_noisy_velocity_static_position_is_standstill(self):
        controller = self._controller()

        # First sample after the command min-age sets the drift anchor.
        settled, residual = controller._place_refine_command_settled(
            now=100.8, dual=False)
        self.assertFalse(settled)
        self.assertFalse(residual)
        self.assertIsNotNone(controller._place_refine_motion_anchor)

        # Same static pose (positions unchanged) after the settle window is
        # accepted as standstill despite j6 velocity noise.
        settled, residual = controller._place_refine_command_settled(
            now=101.2, dual=False)
        self.assertTrue(settled)
        self.assertTrue(residual)

    def test_moving_arm_never_accepted(self):
        controller = self._controller()
        controller._place_refine_command_settled(now=100.8, dual=False)

        # A genuinely moving arm drifts more than the position tolerance on
        # every sample and keeps re-anchoring instead of accumulating the
        # 0.3 s stable window.
        controller.joints["right_arm_joint5"] = -1.5930
        settled, residual = controller._place_refine_command_settled(
            now=101.1, dual=False)
        self.assertFalse(settled)
        controller.joints["right_arm_joint5"] = -1.5730
        settled, residual = controller._place_refine_command_settled(
            now=101.4, dual=False)
        self.assertFalse(settled)
        controller.joints["right_arm_joint5"] = -1.5530
        settled, residual = controller._place_refine_command_settled(
            now=101.7, dual=False)
        self.assertFalse(settled)

    def test_command_converged_fast_path(self):
        controller = self._controller()
        # Joints exactly at the desired pose: commands_ready settles quickly.
        controller.joints["right_arm_joint1"] = -0.9901
        controller.joints["right_arm_joint2"] = -1.8946
        controller.joints["right_arm_joint3"] = 1.1720
        controller.joints["right_arm_joint4"] = -2.2151
        controller.joints["right_arm_joint5"] = -1.7484
        controller.joints["right_arm_joint6"] = -2.4508
        controller.joints["slide_joint"] = 0.1692
        controller.joint_velocities["slide_joint"] = 0.0
        now_sequence = iter([100.0, 100.1])
        controller.now = mock.Mock(side_effect=lambda: next(now_sequence))

        settled, residual = controller._place_refine_command_settled(
            now=100.1, dual=False)
        self.assertFalse(settled)
        settled, residual = controller._place_refine_command_settled(
            now=100.2, dual=False)
        self.assertTrue(settled)
        self.assertFalse(residual)


if __name__ == "__main__":
    unittest.main()
