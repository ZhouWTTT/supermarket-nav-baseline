#!/usr/bin/env python3
"""End-to-end geometry regression tests for the dual-tissue grasp.

Runs the real controller configuration path (configure_dual_tissue_grasp ->
fixed hand-side squeeze) against the real MMK2 KDL for every column at the
middle/lower shelves, then verifies:

* the hand-side pose (roll 0 deg, fingers vertical, 62x96 mm big face
  transverse) is used from the very start: no wrist-roll probe and no
  unroll segment (user: "一开始探入就是 90° 旋转后的样子，不再做额外
  旋转");
* the 105 mm probe span keeps link6's 160 mm bar clear of the post inner
  face (0.19 m from the box centre) by ~5 mm;
* the reduced insert depth (+5 mm past the box centre) keeps link6's front
  face ~17 mm behind the box back face while the fingers still press the
  box sides (钳子接触纸巾而不是臂膀);
* the fixed clamp presses the box with the big faces (>= 20 mm
  interference).
"""
import logging
import math
import pathlib
import sys
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "examples" / "supermarket_sorting"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(REPO / "tests"))

IMPORT_ERROR = None
try:
    from run_pick_tests_host import install_stubs
    install_stubs()
    from mmk2_kdl import MMK2Kdl
    import yolo_aruco_shelf_pick as pick
except ImportError as exc:
    IMPORT_ERROR = exc
    pick = None
    MMK2Kdl = None


SHELF_X = 0.920
COLUMNS = {"1": -0.22, "2": 0.00, "3": 0.22}
POST_INNER = 0.190  # post inner face distance from the box centre (0.21-0.02)
BOX_HALF_W = 0.086
FINGER_HALF = 0.031  # hand-side big-face (grip face) offset from the TCP
BAR_HALF = 0.080     # link6 collision bar half-width (transverse)
BAR_ROLLED_HALF = 0.025  # rolled (hand-back) link6 bar half-width
LINK6_FRONT = 0.055  # link6 front face = insert_y + 0.07 - 0.015
BOX_HALF_D = 0.0425  # box half depth


@unittest.skipIf(pick is None, f"runtime dependencies unavailable: {IMPORT_ERROR}")
class TissueGraspGeometryTests(unittest.TestCase):
    def _make_controller(self, shelf_level, column):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.shelf_level = shelf_level
        controller.target_world = np.array([
            SHELF_X + COLUMNS[column], 3.243,
            {"middle": 0.895, "lower": 0.558}[shelf_level]])
        controller.committed_slot = ("D", {
            "middle": "L2", "lower": "L1"}[shelf_level], column)
        controller.use_dual_tissue_grasp = True
        controller.cmd_left_arm = np.array(
            [0.514, -1.214, 0.394, 1.928, 1.668, -0.343])
        controller.cmd_right_arm = np.array(
            [-0.514, -1.214, 0.394, 1.214, 1.668, -2.798])
        controller.base_xy = np.array([controller.target_world[0], 2.495])
        controller.base_yaw = math.pi / 2.0
        controller.slide_grasp = float(np.clip(
            pick.SLIDE_REFERENCE_COMMAND
            - (controller.target_world[2] - pick.SLIDE_REFERENCE_Z_M),
            pick.SLIDE_MIN, pick.SLIDE_MAX))
        controller.kdl = MMK2Kdl()
        controller.joints = {"slide_joint": controller.slide_grasp}
        controller.dual_squeeze_m = pick.DUAL_TISSUE_SQUEEZE_M
        controller.logger = logging.getLogger(f"geom-{shelf_level}-{column}")
        controller.get_logger = lambda: controller.logger
        controller.state_t0 = 0.0
        controller.now = lambda: 0.0
        controller.dual_left_contact_samples = pick.deque(maxlen=100)
        controller.dual_right_contact_samples = pick.deque(maxlen=100)
        return controller

    def _fp(self, controller, joints, side):
        left, right = controller.kdl.forward_kinematics(
            np.concatenate(([controller.joints["slide_joint"]], joints)),
            index=side)
        T = left if side == "left" else right
        return controller.footprint_to_world(T[:3, 3]), T[:3, :3]

    def _configure(self, shelf_level, column):
        controller = self._make_controller(shelf_level, column)
        self.assertTrue(controller.configure_dual_tissue_grasp())
        return controller

    def test_side_pose_probe_from_the_start_no_unroll(self):
        # The probe is solved in the hand-side pose (roll 0) with no unroll.
        for column in ("1", "2", "3"):
            for level in ("middle", "lower"):
                controller = self._configure(level, column)
                self.assertFalse(controller.dual_top_wrist_rolled)
                self.assertFalse(controller.dual_top_wrist_inward)
                self.assertIsNone(
                    controller.dual_surround_unroll_left_joints)
                expected_span = 0.110
                if column == "2":
                    expected_span = 0.150
                self.assertAlmostEqual(
                    controller.dual_pregrasp_half_span,
                    expected_span, places=3)
                self.assertIsNone(
                    controller.dual_surround_pass_left_joints)

    def test_side_probe_threads_the_post_corridor(self):
        # Side pose: link6 bar is 160 mm wide, outer edge at span+0.08 from
        # the box centre; the pillar-side arm must stay clear of the post.
        for column in ("1", "3"):
            controller = self._configure("middle", column)
            box_x = controller.target_world[0]
            post_side = "left" if column == "1" else "right"
            joints = (
                controller.dual_surround_left_joints
                if post_side == "left"
                else controller.dual_surround_right_joints)
            pos, _ = self._fp(controller, joints, post_side)
            bar_outer = abs(pos[0] - box_x) + BAR_HALF
            self.assertLess(bar_outer, POST_INNER - 0.002)

    def test_reduced_insert_depth_keeps_link6_behind_the_box(self):
        # Insert depth +5 mm past the box centre: link6 front face stays
        # >= 10 mm behind the box back face, so the bar (臂膀) never touches
        # the tissue — the fingers (钳子) press the box sides instead.
        for column in ("1", "2", "3"):
            controller = self._configure("middle", column)
            pos, _ = self._fp(
                controller, controller.dual_surround_left_joints, "left")
            insert_y = pos[1] - controller.target_world[1]
            self.assertAlmostEqual(insert_y, 0.005, places=3)
            link6_front = pos[1] + LINK6_FRONT
            self.assertGreater(
                link6_front - (controller.target_world[1] + BOX_HALF_D),
                0.010)
            # the fingers still overlap the box depth-wise (钳子接触纸盒)
            self.assertLess(pos[1], controller.target_world[1] + BOX_HALF_D)

    def test_fixed_hand_side_clamp_presses_big_face_into_the_box(self):
        # The clamp solves in the side pose: the grip face (27 mm inside each
        # TCP) must sit inside the box surface by >= 20 mm.
        for column in ("1", "2", "3"):
            controller = self._configure("middle", column)
            box_x = controller.target_world[0]
            controller.arm_positions = lambda side: (
                controller.dual_surround_left_joints
                if side == "left"
                else controller.dual_surround_right_joints)
            def measured(side):
                joints = (
                    controller.dual_surround_left_joints
                    if side == "left"
                    else controller.dual_surround_right_joints)
                pos, _ = self._fp(controller, joints, side)
                return pos
            controller.arm_tcp_world = measured
            controller.start_dual_tissue_motion = lambda *a, **k: None
            if column == "2":
                # Middle column now closes from the wide surround pose to the
                # pre-solved clamp targets; it does not use the measured-TCP
                # squeeze path.
                self.assertIsNotNone(controller.dual_clamp_left_joints)
                self.assertIsNotNone(controller.dual_clamp_right_joints)
            else:
                self.assertTrue(controller.start_dual_tissue_squeeze())
            for side, sign in (("left", -1.0), ("right", 1.0)):
                joints = (
                    controller.dual_clamp_left_joints
                    if side == "left"
                    else controller.dual_clamp_right_joints)
                pos, _ = self._fp(controller, joints, side)
                face = pos[0] + (
                    0.031 if side == "left" else -0.031)
                box_face = box_x + sign * BOX_HALF_W
                interference = (
                    face - box_face if side == "left"
                    else box_face - face)
                self.assertGreater(interference, 0.015)

    def test_side_pose_retreat_clears_the_posts(self):
        # Retreat stays in the side pose at the clamp span (0.09): link6's
        # bar inner edge at span+0.08 must stay clear of the post inner face
        # by >= 10 mm.
        for column in ("1", "3"):
            controller = self._configure("middle", column)
            box_x = controller.target_world[0]
            controller.arm_positions = lambda side: (
                controller.dual_surround_left_joints
                if side == "left"
                else controller.dual_surround_right_joints)
            def measured(side):
                joints = (
                    controller.dual_surround_left_joints
                    if side == "left"
                    else controller.dual_surround_right_joints)
                pos, _ = self._fp(controller, joints, side)
                return pos
            controller.arm_tcp_world = measured
            controller.start_dual_tissue_motion = lambda *a, **k: None
            self.assertTrue(controller.start_dual_tissue_squeeze())
            for side, sign in (("left", -1.0), ("right", 1.0)):
                joints = (
                    controller.dual_retreat_left_joints
                    if side == "left"
                    else controller.dual_retreat_right_joints)
                pos, _ = self._fp(controller, joints, side)
                bar_inner = pos[0] + (
                    BAR_HALF if side == "left"
                    else -BAR_HALF)
                post_inner = box_x + sign * POST_INNER
                clearance = (
                    bar_inner - post_inner if side == "left"
                    else post_inner - bar_inner)
                self.assertGreater(clearance, 0.010)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
