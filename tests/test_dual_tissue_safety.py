"""Regression tests for dual-tissue shelf-clearance and endpoint safety."""

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
except ImportError as exc:  # Host without ROS/discoverse; client container runs it.
    IMPORT_ERROR = exc
    np = None
    pick = None


@unittest.skipIf(pick is None, f"runtime dependencies unavailable: {IMPORT_ERROR}")
class DualTissueSafetyTests(unittest.TestCase):
    @staticmethod
    def _logger():
        return mock.Mock()

    def test_top_shelf_tcp_keeps_validated_support_height(self):
        # The top TCP raise is calibrated so the closed finger tips (70 mm
        # below the TCP) end up ~14 mm above the box bottom: the box slides
        # that much and sits on the finger tips ("clamp + support").
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.shelf_level = "top"
        controller.target_world = np.array([1.58, 3.2411, 1.248])
        controller.committed_slot = ("E", "L3", "2")
        controller.cmd_left_arm = np.zeros(6)
        controller.cmd_right_arm = np.zeros(6)
        controller.slide_grasp = 0.0
        controller.logger = self._logger()
        controller.get_logger = lambda: controller.logger
        solved_targets = []
        def solve(left, right, *_args, **_kwargs):
            solved_targets.append((left.copy(), right.copy()))
            return np.zeros(6), np.zeros(6)
        controller.solve_kdl_both_world = solve

        self.assertTrue(controller.configure_dual_tissue_grasp())
        self.assertAlmostEqual(controller.dual_contact_tcp_z, 1.283)
        self.assertAlmostEqual(
            controller.dual_contact_tcp_z - controller.target_world[2],
            0.035)
        # finger tips (tcp - 0.070) sit ~24 mm above the 1.189 m board top:
        # the box slides that much and sits on the finger tips
        self.assertAlmostEqual(
            controller.dual_contact_tcp_z - 0.070,
            pick.SHELF_SURFACE_Z_M["top"] + 0.024, places=3)
        self.assertIsNotNone(controller.dual_pregrasp_left_joints)
        self.assertIsNotNone(controller.dual_surround_left_joints)
        self.assertAlmostEqual(controller.dual_pregrasp_half_span, 0.150)
        self.assertAlmostEqual(solved_targets[0][0][0], 1.430)
        self.assertAlmostEqual(solved_targets[1][0][0], 1.430)
        self.assertAlmostEqual(
            controller.dual_surround_half_span, 0.150)
        self.assertIsNotNone(controller.dual_surround_close_left_joints)
        self.assertAlmostEqual(solved_targets[1][0][2], 1.283)
        self.assertFalse(controller.dual_top_wrist_rolled)
        self.assertFalse(controller.dual_top_wrist_inward)
        self.assertIsNone(controller.dual_surround_pass_left_joints)
        self.assertIsNone(controller.dual_surround_forward_left_joints)
        self.assertIsNone(controller.dual_surround_return_left_joints)
        self.assertIsNone(controller.dual_surround_unroll_left_joints)
        self.assertIsNone(controller.dual_surround_unroll_right_joints)
        self.assertIsNotNone(controller.dual_clamp_left_joints)
        self.assertIsNotNone(controller.dual_retreat_left_joints)

    def test_top_dual_ik_uses_mirrored_narrow_wrist_orientation(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.shelf_level = "top"
        controller.dual_top_wrist_rolled = True
        controller.dual_top_wrist_inward = False
        controller.slide_grasp = 0.0
        controller.world_to_footprint = lambda point: np.asarray(point)
        controller.kdl = mock.Mock()
        controller.kdl.inverse_kinematics.return_value = [np.arange(13)]

        controller.solve_kdl_both_world(
            np.array([0.7, 0.1, 1.28]),
            np.array([0.7, -0.1, 1.28]),
            np.zeros(6), np.zeros(6))

        call = controller.kdl.inverse_kinematics.call_args.kwargs
        expected_left = pick.Rotation.from_euler(
            "x", pick.DUAL_TISSUE_TOP_WRIST_ROLL_RAD).as_matrix()
        expected_right = pick.Rotation.from_euler(
            "x", -pick.DUAL_TISSUE_TOP_WRIST_ROLL_RAD).as_matrix()
        np.testing.assert_allclose(call["T_left"][:3, :3], expected_left)
        np.testing.assert_allclose(call["T_right"][:3, :3], expected_right)

        controller.solve_kdl_both_world(
            np.array([0.7, 0.18, 1.42]),
            np.array([0.7, -0.18, 1.42]),
            np.zeros(6), np.zeros(6),
            top_wrist_rolled=True, top_wrist_inward=True)
        unroll_call = controller.kdl.inverse_kinematics.call_args.kwargs
        expected_left_inward = pick.Rotation.from_euler(
            "x", -pick.DUAL_TISSUE_TOP_WRIST_ROLL_RAD).as_matrix()
        expected_right_inward = pick.Rotation.from_euler(
            "x", pick.DUAL_TISSUE_TOP_WRIST_ROLL_RAD).as_matrix()
        np.testing.assert_allclose(
            unroll_call["T_left"][:3, :3], expected_left_inward)
        np.testing.assert_allclose(
            unroll_call["T_right"][:3, :3], expected_right_inward)

    def test_middle_dual_ik_uses_mirrored_narrow_wrist_orientation(self):
        # The rolled-wrist pose is no longer top-shelf only: the middle/lower
        # direct probe must thread the 104 mm corridor between the front post
        # and the box, which the unrolled 160 mm link6 box cannot do.
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.shelf_level = "middle"
        controller.dual_top_wrist_rolled = True
        controller.dual_top_wrist_inward = True
        controller.slide_grasp = 0.138
        controller.world_to_footprint = lambda point: np.asarray(point)
        controller.kdl = mock.Mock()
        controller.kdl.inverse_kinematics.return_value = [np.arange(13)]

        controller.solve_kdl_both_world(
            np.array([0.56, 0.6, 0.866]),
            np.array([0.84, 0.6, 0.866]),
            np.zeros(6), np.zeros(6))

        call = controller.kdl.inverse_kinematics.call_args.kwargs
        expected_left = pick.Rotation.from_euler(
            "x", -pick.DUAL_TISSUE_TOP_WRIST_ROLL_RAD).as_matrix()
        expected_right = pick.Rotation.from_euler(
            "x", pick.DUAL_TISSUE_TOP_WRIST_ROLL_RAD).as_matrix()
        np.testing.assert_allclose(call["T_left"][:3, :3], expected_left)
        np.testing.assert_allclose(call["T_right"][:3, :3], expected_right)

    def test_contact_timeout_aborts_without_unconfirmed_squeeze(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.state_t0 = 0.0
        controller.dual_contact_duration_s = 1.0
        controller.now = lambda: 2.0
        controller.dual_contact_start_left_joints = np.zeros(6)
        controller.dual_contact_start_right_joints = np.zeros(6)
        controller.dual_contact_target_left_joints = np.ones(6)
        controller.dual_contact_target_right_joints = np.ones(6)
        controller.dual_contact_start_left_world = np.array([1.48, 3.27, 1.28])
        controller.dual_contact_start_right_world = np.array([1.68, 3.27, 1.28])
        controller.dual_contact_goal_left_world = np.array([1.53, 3.27, 1.28])
        controller.dual_contact_goal_right_world = np.array([1.63, 3.27, 1.28])
        controller.dual_left_contacted = False
        controller.dual_right_contacted = False
        controller.dual_left_contact_samples = pick.deque(maxlen=100)
        controller.dual_right_contact_samples = pick.deque(maxlen=100)
        controller.arm_tcp_world = lambda side: (
            np.array([1.481, 3.27, 1.28]) if side == "left"
            else np.array([1.679, 3.27, 1.28]))
        controller.logger = self._logger()
        controller.get_logger = lambda: controller.logger

        self.assertEqual(
            controller.advance_dual_tissue_contact_search(), "failed")
        controller.logger.error.assert_called_once()

    def test_surround_starts_direct_insertion_for_all_levels(self):
        # The top shelf uses the same direct insertion as middle/lower (no
        # overhead rise/widen/descend): the overhead path forced the right
        # wrist joint6 to jump branches (~326 deg) and look like it was
        # rotating constantly.
        for level in ("top", "middle", "lower"):
            controller = pick.ShelfPickController.__new__(
                pick.ShelfPickController)
            controller.shelf_level = level
            controller.base_xy = np.zeros(2)
            controller.dual_insert_forward_m = 0.005
            controller.dual_pregrasp_half_span = 0.105
            controller.dual_overhead_route = False
            controller.dual_surround_pass_left_joints = None
            controller.dual_surround_pass_right_joints = None
            controller.dual_surround_unroll_left_joints = None
            controller.dual_surround_unroll_right_joints = None
            controller.dual_surround_left_joints = np.ones(6)
            controller.dual_surround_right_joints = np.ones(6) * 2
            controller.start_dual_tissue_motion = mock.Mock()

            controller.start_dual_tissue_surround()

            self.assertEqual(controller.dual_surround_stage, 0)
            args = controller.start_dual_tissue_motion.call_args.args
            self.assertEqual(args[0], "surround")
            np.testing.assert_array_equal(args[1], np.ones(6))
            self.assertTrue(
                controller.start_dual_tissue_motion.call_args.kwargs[
                    "require_convergence"])

    def test_surround_advances_directly_into_squeeze_for_all_levels(self):
        # No widen/unroll/descend stages: the side-pose insertion is followed
        # directly by the fixed hand-side deep clamp.
        for level in ("top", "middle"):
            controller = pick.ShelfPickController.__new__(
                pick.ShelfPickController)
            controller.shelf_level = level
            controller.dual_surround_stage = 0
            controller.dual_overhead_route = False
            controller.dual_surround_unroll_left_joints = None
            controller.dual_surround_unroll_right_joints = None
            controller.start_dual_tissue_squeeze = mock.Mock(
                return_value=True)
            controller.set_state = mock.Mock()

            controller.advance_dual_tissue_surround_sequence()

            self.assertEqual(controller.dual_surround_stage, 1)
            controller.start_dual_tissue_squeeze.assert_called_once()

    def test_middle_surround_advances_into_lateral_close_stage(self):
        # The middle-column tissue grasp first reaches in at a wide span,
        # then closes to the clamp span before holding and lifting.
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.dual_surround_stage = 0
        controller.dual_middle_extend_close = True
        controller.dual_pregrasp_half_span = 0.150
        controller.dual_clamp_half_span = 0.090
        controller.dual_surround_close_left_joints = np.ones(6)
        controller.dual_surround_close_right_joints = np.ones(6) * 2
        controller.start_dual_tissue_motion = mock.Mock()
        controller.start_dual_tissue_squeeze = mock.Mock()
        controller.set_state = mock.Mock()
        controller.get_logger = self._logger

        controller.advance_dual_tissue_surround_sequence()

        self.assertEqual(controller.dual_surround_stage, 1)
        controller.start_dual_tissue_squeeze.assert_not_called()
        args = controller.start_dual_tissue_motion.call_args.args
        self.assertEqual(args[0], "surround_close")
        np.testing.assert_array_equal(args[1], np.ones(6))
        np.testing.assert_array_equal(args[2], np.ones(6) * 2)
        self.assertAlmostEqual(args[3], 0.060)
        self.assertEqual(args[5], pick.STATE_DUAL_SQUEEZE)
        self.assertTrue(
            controller.start_dual_tissue_motion.call_args.kwargs[
                "require_convergence"])

    def test_top_arm_lift_uses_slow_synchronized_duration(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.arm_positions = lambda _side: np.zeros(6)
        controller.base_xy = np.zeros(2)
        controller.now = lambda: 1.0
        controller.get_logger = self._logger
        controller.set_state = mock.Mock()

        controller.start_dual_tissue_motion(
            "arm_lift", np.ones(6), np.ones(6),
            pick.DUAL_TISSUE_LIFT_M,
            pick.DUAL_TISSUE_ARM_LIFT_SPEED_MPS,
            pick.STATE_LIFT)

        self.assertAlmostEqual(controller.dual_motion_duration_s, 7.5)
        controller.set_state.assert_called_once_with(pick.STATE_LIFT)

    def test_top_fork_places_unrolled_right_wrist_below_overhang(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.target_world = np.array([1.58, 3.2608, 1.248])
        controller.slide_grasp = pick.SLIDE_MIN
        controller.arm_tcp_world = lambda side: (
            np.array([1.46, 3.141, 1.283]) if side == "left"
            else np.array([1.70, 3.141, 1.283]))
        controller.arm_positions = lambda _side: np.zeros(6)
        controller.logger = self._logger()
        controller.get_logger = lambda: controller.logger
        solved = []

        def solve(left, right, left_ref, right_ref, **kwargs):
            solved.append((left.copy(), right.copy(), kwargs))
            return left_ref + 0.01, right_ref + 0.01

        controller.solve_kdl_both_world = solve

        self.assertTrue(controller.configure_dual_tissue_top_fork())
        # Stage 14 keeps the support pose while the chassis carries the box
        # past the shelf edge (added with the fork's final retreat).
        self.assertEqual(set(controller.dual_top_fork_targets), set(range(2, 15)))
        np.testing.assert_allclose(
            controller.dual_top_fork_targets[8][1],
            controller.dual_top_fork_targets[7][1])
        for stage in (9, 10, 11):
            np.testing.assert_allclose(
                controller.dual_top_fork_targets[stage][1],
                controller.dual_top_fork_targets[8][1])
        np.testing.assert_allclose(
            controller.dual_top_fork_targets[12][1],
            controller.dual_top_fork_targets[11][1])
        # Stage 4 is still rolled and below the top board surface.
        self.assertAlmostEqual(
            solved[2][1][2],
            pick.SHELF_SURFACE_Z_M["top"]
            - pick.DUAL_TISSUE_TOP_FORK_TCP_BELOW_SURFACE_M)
        self.assertIsNone(solved[2][2]["right_rotation"])
        # From stage 5 onward the right wrist is yawed in the footprint frame
        # so link6 points along shelf depth; its TCP is then moved to the
        # measured bilateral midpoint so the bar sits under the right half.
        expected_fork_rotation = pick.Rotation.from_euler(
            "z", pick.math.pi / 2.0).as_matrix()
        np.testing.assert_allclose(
            solved[3][2]["right_rotation"], expected_fork_rotation)
        self.assertAlmostEqual(
            solved[4][1][0],
            0.5 * (1.46 + 1.70)
            - pick.DUAL_TISSUE_TOP_FORK_BAR_LATERAL_OFFSET_M)
        self.assertAlmostEqual(
            solved[4][1][1],
            controller.target_world[1]
            - pick.DUAL_TISSUE_TOP_FORK_FRONT_BACKOFF_M)
        # Stage 9 routes the left wrist over the tissue and behind its back
        # face (raised, not yet lowered); stage 11 lowers it before the push.
        self.assertAlmostEqual(
            solved[6][0][0],
            1.46 - pick.DUAL_TISSUE_TOP_FORK_PUSHER_RELEASE_M)
        self.assertAlmostEqual(
            solved[6][0][2],
            1.283 + pick.DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M
            + pick.DUAL_TISSUE_TOP_FORK_PUSHER_OVERHEAD_M)
        self.assertAlmostEqual(solved[6][1][1], solved[4][1][1])
        self.assertAlmostEqual(
            solved[6][2]["target_height"],
            pick.DUAL_TISSUE_TOP_FORK_SLIDE_M
            - pick.DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M)
        # Stage 10 (index 7) moves the pusher to the tissue centre line.
        self.assertAlmostEqual(
            solved[7][0][0],
            controller.target_world[0]
            + pick.DUAL_TISSUE_TOP_FORK_PUSHER_X_LEAD_M)
        self.assertAlmostEqual(solved[7][1][1], solved[6][1][1])
        # Stage 11 (index 8) lowers the pusher behind the box; stage 12
        # (index 9) pushes it forward over the support bar.
        self.assertAlmostEqual(solved[8][0][1], solved[7][0][1])
        self.assertAlmostEqual(
            solved[8][0][2],
            1.283 + pick.DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M
            - pick.DUAL_TISSUE_TOP_FORK_PUSHER_LOWER_M)
        self.assertAlmostEqual(solved[8][1][1], solved[7][1][1])
        self.assertAlmostEqual(
            solved[9][0][1],
            solved[8][0][1] - pick.DUAL_TISSUE_TOP_FORK_PUSH_M)
        self.assertAlmostEqual(solved[9][1][1], solved[8][1][1])

    def test_top_fork_stage_eight_uses_slide_with_fixed_arm_joints(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.dual_top_fork_targets = {
            8: (np.ones(6), np.ones(6) * 2)}
        controller.start_dual_tissue_motion = mock.Mock()
        controller.dual_lift_use_arm = True
        controller.set_state = mock.Mock()
        controller.get_logger = self._logger

        controller.start_dual_tissue_top_fork_stage(8)

        self.assertFalse(controller.dual_lift_use_arm)
        self.assertEqual(controller.dual_top_extract_stage, 8)
        controller.start_dual_tissue_motion.assert_not_called()
        controller.set_state.assert_called_once_with(pick.STATE_LIFT)
        np.testing.assert_allclose(controller.des_left_arm, np.ones(6))
        np.testing.assert_allclose(controller.des_right_arm, np.ones(6) * 2)
        self.assertAlmostEqual(
            controller.des_slide,
            pick.DUAL_TISSUE_TOP_FORK_SLIDE_M
            - pick.DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M)

    def test_top_fork_stage_twelve_arms_pusher_motion(self):
        # Stage 12 (push over the support bar) is a gated motion, not a slide
        # lift: the pusher moves while the support joints stay fixed.
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.dual_top_fork_targets = {
            12: (np.ones(6), np.ones(6) * 2)}
        controller.start_dual_tissue_motion = mock.Mock()
        controller.dual_lift_use_arm = True
        controller.set_state = mock.Mock()
        controller.get_logger = self._logger

        controller.start_dual_tissue_top_fork_stage(12)

        self.assertEqual(controller.dual_top_extract_stage, 12)
        controller.start_dual_tissue_motion.assert_called_once()
        args = controller.start_dual_tissue_motion.call_args.args
        self.assertEqual(args[0], "top_fork_push_over_support")
        self.assertEqual(args[5], pick.STATE_RETREAT)
        self.assertTrue(
            controller.start_dual_tissue_motion.call_args.kwargs[
                "require_convergence"])
        self.assertAlmostEqual(
            controller.des_slide,
            pick.DUAL_TISSUE_TOP_FORK_SLIDE_M
            - pick.DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M)

    def test_top_fork_keeps_initial_pull_short_and_allows_rolled_descent(self):
        # The shelf front is 150 mm in front of its 3.323 m centre.  The
        # initial pull must expose only a lip, not move the box completely
        # past that edge as the former 120 mm setting did.
        self.assertLessEqual(pick.DUAL_TISSUE_TOP_EDGE_BACKOFF_M, 0.055)
        # The long link6 box reaches about 46 mm above its TCP while the wrist
        # rotates.  The full sweep must remain below the board underside.
        board_underside = pick.SHELF_SURFACE_Z_M["top"] - 0.020
        unroll_top = (
            pick.SHELF_SURFACE_Z_M["top"]
            - pick.DUAL_TISSUE_TOP_FORK_TCP_BELOW_SURFACE_M
            + 0.046)
        self.assertLess(unroll_top, board_underside)

        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.dual_top_fork_targets = {
            3: (np.ones(6), np.ones(6) * 2),
            4: (np.ones(6) * 3, np.ones(6) * 4),
        }
        controller.start_dual_tissue_motion = mock.Mock()

        controller.start_dual_tissue_top_fork_stage(3)
        stage_three = controller.start_dual_tissue_motion.call_args
        self.assertAlmostEqual(
            stage_three.args[3],
            pick.DUAL_TISSUE_TOP_FORK_FRONT_BACKOFF_M
            - pick.DUAL_TISSUE_TOP_EDGE_BACKOFF_M)

        controller.start_dual_tissue_top_fork_stage(4)
        stage_four = controller.start_dual_tissue_motion.call_args
        self.assertFalse(stage_four.kwargs["require_convergence"])

    def test_top_tissue_alignment_keeps_overhead_elbow_clearance(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.target_kind = "zhijin"
        controller.use_dual_tissue_grasp = True
        controller.state = pick.STATE_SCAN
        controller.state_t0 = 0.0
        controller.now = lambda: 1.0
        controller.get_logger = self._logger
        controller.target_marker_id = None
        controller.target_physical_marker_id = None
        controller.nav_target = None
        controller.committed_slot = None
        controller._commit_localised_target(
            np.array([1.58, 3.240, 1.248]), None, "test")

        expected = (
            3.240 - pick.TOP_GRASP_CENTER_DISTANCE_M
            + pick.DUAL_TISSUE_ALIGN_FORWARD_M)
        self.assertAlmostEqual(controller.align_base_y, expected)
        self.assertLess(
            controller.align_base_y,
            3.240 - pick.TOP_GRASP_CENTER_DISTANCE_M)

    def test_tissue_rotation_plans_three_stages_and_starts_contact(self):
        saved_enabled = pick.TISSUE_ROTATE_ENABLED
        pick.TISSUE_ROTATE_ENABLED = True
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.shelf_level = "top"
        controller.target_world = np.array([1.58, 3.2409, 1.248])
        controller.cmd_left_arm = np.zeros(6)
        controller.cmd_right_arm = np.zeros(6)
        controller.slide_grasp = 0.08
        controller.logger = self._logger()
        controller.get_logger = lambda: controller.logger
        controller.solve_kdl_both_world = (
            lambda *args, **kwargs: (np.zeros(6), np.zeros(6)))

        self.assertTrue(controller.configure_tissue_90_rotation())
        self.assertEqual(set(controller.tissue_rotate_targets), {0, 1, 2})

        controller.use_dual_tissue_grasp = True
        controller.tissue_rotated_90 = False
        controller.start_dual_tissue_motion = mock.Mock()
        controller.begin_manip_base_hold = mock.Mock()
        self.assertTrue(controller._prepare_tissue_rotation_if_needed())
        self.assertEqual(controller.tissue_rotate_stage, 0)
        controller.start_dual_tissue_motion.assert_called_once()
        self.assertEqual(
            controller.start_dual_tissue_motion.call_args.args[0],
            "rotate_anchor_pre")
        pick.TISSUE_ROTATE_ENABLED = saved_enabled

    def test_middle_column_skips_tissue_prerotation(self):
        saved_enabled = pick.TISSUE_ROTATE_ENABLED
        pick.TISSUE_ROTATE_ENABLED = True
        for level in ("middle", "lower", "top"):
            controller = pick.ShelfPickController.__new__(
                pick.ShelfPickController)
            controller.shelf_level = level
            controller.committed_slot = (
                "D", {"middle": "L2", "lower": "L1", "top": "L3"}[level], "2")
            controller.use_dual_tissue_grasp = True
            controller.tissue_rotated_90 = False
            controller.get_logger = self._logger
            controller.configure_tissue_90_rotation = mock.Mock(
                return_value=True)
            controller.start_tissue_rotate_stage = mock.Mock()

            self.assertFalse(
                controller._prepare_tissue_rotation_if_needed())
            controller.configure_tissue_90_rotation.assert_not_called()
            controller.start_tissue_rotate_stage.assert_not_called()
        pick.TISSUE_ROTATE_ENABLED = saved_enabled

    def test_tissue_direct_slot_rejects_side_columns(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.target_kind = "zhijin"
        controller.get_logger = self._logger

        self.assertFalse(controller.configure_direct_slot_target(
            "D", "L2", "1"))
        self.assertFalse(controller.configure_direct_slot_target(
            "D", "L2", "3"))

    def test_scan_exhaustion_marks_no_middle_tissue_for_tissue(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.target_kind = "zhijin"
        controller.max_scan_cycles = 1
        controller.scan_cycles = 0
        controller.scan_poses = pick.SCAN_CAMERA_POSES
        controller.scan_pose_index = len(controller.scan_poses) - 1
        controller.scan_index = len(pick.SCAN_X) - 1
        controller.scan_station_order = list(range(len(pick.SCAN_X)))
        controller.inventory_scan_hint_active = False
        controller.scan_camera_ready_since = None
        controller.base_xy = np.zeros(2)
        controller.scan_preferred_x = None
        controller.scan_prefer_west_start = False
        controller.get_logger = self._logger
        controller.set_state = mock.Mock()

        self.assertFalse(controller._advance_scan_pose())
        self.assertTrue(controller.no_middle_tissue)
        controller.set_state.assert_any_call(pick.STATE_ABORT)

    def test_tissue_association_ignores_side_column_marker(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.state = pick.STATE_SCAN
        controller.state_t0 = 0.0
        controller.now = lambda: 1.0
        controller.target_kind = "zhijin"
        controller.target_marker_id = None
        controller.excluded_marker_ids = set()
        controller.recheck_marker_skips = set()
        controller.skipped_tissue_markers = set()
        controller.skipped_tissue_slots = set()
        controller.last_association_pair = None
        controller.get_logger = self._logger
        detection = {
            "class": "zhijin",
            "conf": 0.95,
            "world": [0.70, 3.24, 0.85],
        }
        marker = {
            "id": 12,
            "position_world": [0.70, 3.24, 0.85],
        }
        controller.yolo_frames = pick.deque(
            [(100, [detection])], maxlen=24)
        controller.aruco_frames = pick.deque(
            [(100, [marker])], maxlen=24)

        with mock.patch.object(
                pick, "marker_below_yolo", return_value=marker):
            controller.try_association_locked()

        self.assertIsNone(controller.target_marker_id)
        self.assertIn(12, controller.skipped_tissue_markers)

    def test_rotated_tissue_grasp_uses_narrow_spans(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.shelf_level = "middle"
        controller.tissue_rotated_90 = True
        controller.target_world = np.array([0.70, 3.243, 0.895])
        controller.committed_slot = ("D", "L2", "2")
        controller.cmd_left_arm = np.zeros(6)
        controller.cmd_right_arm = np.zeros(6)
        controller.slide_grasp = 0.138
        controller.logger = self._logger()
        controller.get_logger = lambda: controller.logger
        solved = []
        def solve(left, right, *_args, **_kwargs):
            solved.append((left.copy(), right.copy()))
            return np.zeros(6), np.zeros(6)
        controller.solve_kdl_both_world = solve

        self.assertTrue(controller.configure_dual_tissue_grasp())
        self.assertAlmostEqual(controller.dual_pregrasp_half_span, 0.065)
        self.assertAlmostEqual(controller.dual_clamp_half_span, 0.050)
        self.assertAlmostEqual(solved[0][0][0], 0.635)
        self.assertAlmostEqual(solved[0][1][0], 0.765)

    def test_center_column_direct_probe_uses_hand_back_pose(self):
        # Side columns keep the narrow direct probe.  The middle column now
        # uses a synchronized wide reach-and-clamp into the tissue.
        for level in ("middle", "lower"):
            level_label = {"middle": "L2", "lower": "L1"}[level]
            for column in ("1", "2", "3"):
                controller = pick.ShelfPickController.__new__(
                    pick.ShelfPickController)
                controller.shelf_level = level
                controller.target_world = np.array([0.70, 3.243, 0.895])
                controller.committed_slot = ("D", level_label, column)
                controller.cmd_left_arm = np.zeros(6)
                controller.cmd_right_arm = np.zeros(6)
                controller.slide_grasp = 0.138
                controller.logger = self._logger()
                controller.get_logger = lambda: controller.logger
                solved_targets = []

                def solve(left, right, *_args, **_kwargs):
                    solved_targets.append((left.copy(), right.copy()))
                    return np.zeros(6), np.zeros(6)

                controller.solve_kdl_both_world = solve

                self.assertTrue(controller.configure_dual_tissue_grasp())
                # side-pose direct probe from the start
                self.assertFalse(controller.dual_top_wrist_rolled)
                self.assertFalse(controller.dual_top_wrist_inward)
                if column == "1":
                    self.assertAlmostEqual(
                        controller.dual_pregrasp_half_span, 0.110)
                    self.assertAlmostEqual(solved_targets[0][0][0], 0.600)
                    self.assertAlmostEqual(solved_targets[0][1][0], 0.810)
                    self.assertAlmostEqual(solved_targets[1][0][0], 0.600)
                    self.assertAlmostEqual(solved_targets[1][1][0], 0.810)
                elif column == "3":
                    self.assertAlmostEqual(
                        controller.dual_pregrasp_half_span, 0.110)
                    self.assertAlmostEqual(solved_targets[0][0][0], 0.590)
                    self.assertAlmostEqual(solved_targets[0][1][0], 0.800)
                    self.assertAlmostEqual(solved_targets[1][0][0], 0.590)
                    self.assertAlmostEqual(solved_targets[1][1][0], 0.800)
                else:
                    self.assertAlmostEqual(
                        controller.dual_pregrasp_half_span, 0.150)
                    self.assertAlmostEqual(solved_targets[0][0][0], 0.550)
                    self.assertAlmostEqual(solved_targets[0][1][0], 0.850)
                    self.assertAlmostEqual(solved_targets[1][0][0], 0.550)
                    self.assertAlmostEqual(solved_targets[1][1][0], 0.850)
                    self.assertAlmostEqual(solved_targets[2][0][0], 0.610)
                    self.assertAlmostEqual(solved_targets[2][1][0], 0.790)
                    self.assertAlmostEqual(solved_targets[3][0][0], 0.610)
                    self.assertAlmostEqual(solved_targets[3][1][0], 0.790)
                    self.assertAlmostEqual(
                        controller.dual_surround_half_span, 0.150)
                    self.assertIsNotNone(
                        controller.dual_surround_close_left_joints)
                self.assertFalse(controller.dual_overhead_route)
                self.assertIsNone(controller.dual_surround_pass_left_joints)
                self.assertIsNone(controller.dual_surround_unroll_left_joints)
                # direct probing keeps clamp/retreat unset until the squeeze
                if column == "2":
                    self.assertFalse(controller.dual_direct_probe)
                    self.assertIsNotNone(controller.dual_clamp_left_joints)
                    self.assertIsNotNone(controller.dual_retreat_left_joints)
                    self.assertIsNotNone(controller.dual_surround_left_joints)
                else:
                    self.assertTrue(controller.dual_direct_probe)
                    self.assertIsNone(controller.dual_clamp_left_joints)
                    self.assertIsNone(controller.dual_retreat_left_joints)
                self.assertIsNone(
                    controller.dual_surround_unroll_left_joints)
                self.assertIsNone(
                    controller.dual_surround_unroll_right_joints)

    def test_middle_squeeze_uses_measured_anchor_preload(self):
        # Hand-side (roll 0 deg, fingers vertical) clamp anchored at the
        # measured probe TCPs: each arm moves inward by only the 10 mm
        # preload, so a blind long push cannot slam the right arm into the
        # box before closing (observed 20260818 right TCP stopped at 1.680
        # against a 1.670 nominal target).
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.shelf_level = "middle"
        controller.dual_squeeze_m = pick.DUAL_TISSUE_SQUEEZE_M
        controller.dual_clamp_half_span = pick.DUAL_TISSUE_CLAMP_HALF_SPAN_M
        controller.dual_pregrasp_half_span = 0.14
        controller.dual_insert_forward_m = pick.DUAL_TISSUE_INSERT_FORWARD_M
        controller.dual_contact_tcp_z = 0.851 + pick.DUAL_TISSUE_TCP_CLEARANCE_M
        controller.target_world = np.array([0.70, 3.243, 0.895])
        controller.arm_positions = lambda _side: np.zeros(6)
        controller.arm_tcp_world = lambda side: np.array(
            [0.60, 3.08, 0.936] if side == "left"
            else [0.80, 3.08, 0.936])
        controller.logger = self._logger()
        controller.get_logger = lambda: controller.logger
        controller.start_dual_tissue_motion = mock.Mock()
        solved = []

        def solve(left, right, left_ref, right_ref, **_kwargs):
            solved.append((left.copy(), right.copy()))
            return left_ref + 0.01, right_ref + 0.01

        controller.solve_kdl_both_world = solve

        self.assertTrue(controller.start_dual_tissue_squeeze())
        squeeze_left, squeeze_right = solved[0]
        self.assertAlmostEqual(
            squeeze_left[0],
            0.60 + pick.DUAL_TISSUE_SQUEEZE_M)
        self.assertAlmostEqual(
            squeeze_right[0],
            0.80 - pick.DUAL_TISSUE_SQUEEZE_M)
        self.assertAlmostEqual(
            squeeze_left[1],
            3.08)
        # retreat stays at the pregrasp backoff for the middle shelf
        retreat_left, retreat_right = solved[2]
        self.assertAlmostEqual(
            retreat_left[1],
            3.243 - pick.DUAL_TISSUE_PREGRASP_BACKOFF_M)
        controller.start_dual_tissue_motion.assert_called_once()

    def test_top_clamp_arms_arm_lift_directly_without_edge_extract(self):
        # The top-shelf clamp must go straight into the synchronized arm
        # lift.  The former edge-extract (pull the box onto the board lip)
        # left the box rear on the board, so the lift tilted it around the
        # lip and the box never came off (observed 20260818).  The direct
        # lift from the board centre is the flow that completed grasp,
        # navigation and placement in the 20260817 runs.
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.use_dual_tissue_grasp = True
        controller.state = pick.STATE_CLOSE
        controller.shelf_level = "top"
        controller.dual_top_extract_stage = 0
        controller.dual_lift_use_arm = False
        controller.state_t0 = 0.0
        controller.now = lambda: 4.5
        controller.base_xy = np.zeros(2)
        controller.base_yaw = 0.0
        controller.joints = {"slide_joint": pick.SLIDE_MIN}
        controller.initialized = True
        controller.cmd_linear = 0.0
        controller.cmd_angular = 0.0
        controller.last_status_log = -1.0
        controller.use_sphere_grasp = False
        controller.target_marker_id = None
        controller.tcp_diagnostic_ground_truth = False
        controller.manip_base_hold_xy = None
        controller.manip_base_hold_yaw = None
        controller.target_world = np.array([1.58, 3.243, 1.248])
        controller.slide_grasp = pick.SLIDE_MIN
        controller.dual_clamp_left_joints = np.ones(6)
        controller.dual_clamp_right_joints = np.ones(6) * 2
        controller.dual_lift_settled_since = None
        controller.arm_positions = lambda _side: np.ones(6)
        measured_tcp = np.array([1.46, 3.274, 1.286])
        controller.arm_tcp_world = lambda _side: measured_tcp.copy()
        controller.arm_target_tcp_world = (
            lambda _side, _joints: measured_tcp.copy())
        controller.get_logger = lambda: self._logger()
        controller.set_state = mock.Mock()
        controller.start_dual_tissue_motion = mock.Mock()
        controller.smooth_commands = mock.Mock()
        controller.publish_commands = mock.Mock()

        def solve(left, right, left_ref, right_ref, **_kwargs):
            return left_ref + 0.01, right_ref + 0.01

        controller.solve_kdl_both_world = solve

        controller.tick()

        self.assertTrue(controller.dual_lift_use_arm)
        self.assertEqual(controller.dual_top_extract_stage, 0)
        args = controller.start_dual_tissue_motion.call_args.args
        self.assertEqual(args[0], "arm_lift")
        self.assertEqual(args[5], pick.STATE_LIFT)
        self.assertAlmostEqual(args[3], pick.DUAL_TISSUE_LIFT_M)
        self.assertTrue(
            controller.start_dual_tissue_motion.call_args.kwargs[
                "require_convergence"])

    def test_middle_clamp_starts_slide_lift_for_all_non_top_levels(self):
        for level in ("middle", "lower"):
            controller = pick.ShelfPickController.__new__(
                pick.ShelfPickController)
            controller.use_dual_tissue_grasp = True
            controller.state = pick.STATE_CLOSE
            controller.shelf_level = level
            controller.dual_top_extract_stage = 0
            controller.dual_lift_use_arm = False
            controller.state_t0 = 0.0
            controller.now = lambda: 4.5
            controller.base_xy = np.zeros(2)
            controller.base_yaw = 0.0
            controller.joints = {"slide_joint": 0.1}
            controller.initialized = True
            controller.cmd_linear = 0.0
            controller.cmd_angular = 0.0
            controller.last_status_log = -1.0
            controller.use_sphere_grasp = False
            controller.target_marker_id = None
            controller.tcp_diagnostic_ground_truth = False
            controller.manip_base_hold_xy = None
            controller.manip_base_hold_yaw = None
            controller.target_world = np.array([0.70, 3.243, 0.895])
            controller.slide_grasp = 0.138
            controller.dual_clamp_left_joints = np.ones(6)
            controller.dual_clamp_right_joints = np.ones(6) * 2
            controller.dual_lift_settled_since = None
            controller.get_logger = lambda: self._logger()
            controller.set_state = mock.Mock()
            controller.start_dual_tissue_slide_lift = mock.Mock()
            controller.smooth_commands = mock.Mock()
            controller.publish_commands = mock.Mock()

            controller.tick()

            self.assertFalse(controller.dual_lift_use_arm)
            controller.start_dual_tissue_slide_lift.assert_called_once()

    @staticmethod
    def _motion_controller(measured_left, measured_right, gated=True):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.state_t0 = 0.0
        controller.dual_motion_duration_s = 2.0
        controller.dual_motion_start_left = np.zeros(6)
        controller.dual_motion_start_right = np.zeros(6)
        controller.dual_motion_target_left = np.ones(6)
        controller.dual_motion_target_right = np.ones(6)
        controller.dual_motion_label = "surround"
        controller.dual_motion_require_convergence = gated
        controller.dual_motion_endpoint_ready_since = None
        controller.now = lambda: 2.8
        controller.arm_positions = lambda side: (
            measured_left if side == "left" else measured_right)
        controller.arm_tcp_world = lambda _side: np.zeros(3)
        controller.logger = DualTissueSafetyTests._logger()
        controller.get_logger = lambda: controller.logger
        controller.set_state = mock.Mock()
        return controller

    def test_gated_surround_aborts_when_endpoint_does_not_converge(self):
        controller = self._motion_controller(
            np.zeros(6), np.ones(6), gated=True)

        self.assertEqual(controller.advance_dual_tissue_motion(), "failed")
        controller.set_state.assert_called_once_with(pick.STATE_ABORT)
        controller.logger.error.assert_called_once()

    def test_gated_surround_reaches_after_stable_convergence(self):
        controller = self._motion_controller(
            np.ones(6), np.ones(6), gated=True)
        controller.dual_motion_endpoint_ready_since = 2.5

        self.assertEqual(controller.advance_dual_tissue_motion(), "reached")
        controller.set_state.assert_not_called()

    def test_gated_motion_accepts_reached_tcp_with_redundant_wrist_error(self):
        controller = self._motion_controller(
            np.array([1, 1, 1, 1, 0.95, 1], dtype=float),
            np.array([1, 1, 1, 1, 1.05, 1], dtype=float),
            gated=True)
        controller.arm_target_tcp_world = lambda _side, _joints: np.zeros(3)
        controller.dual_motion_endpoint_ready_since = 2.0
        controller.now = lambda: 2.8

        self.assertEqual(controller.advance_dual_tissue_motion(), "reached")
        controller.set_state.assert_not_called()

    def test_deploy_waits_for_measured_dual_arm_pregrasp(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.dual_surround_left_joints = np.ones(6)
        controller.dual_surround_right_joints = np.ones(6)
        controller.dual_commands_ready = mock.Mock(return_value=False)
        controller.start_dual_tissue_surround = mock.Mock()
        controller.set_state = mock.Mock()
        controller.get_logger = lambda: self._logger()

        controller.advance_dual_tissue_deploy(
            pick.DUAL_TISSUE_DEPLOY_DWELL_S + 0.5)

        controller.start_dual_tissue_surround.assert_not_called()
        controller.set_state.assert_not_called()

    def test_deploy_starts_as_soon_as_measured_pregrasp_is_stable(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.dual_surround_left_joints = np.ones(6)
        controller.dual_surround_right_joints = np.ones(6)
        controller.dual_commands_ready = mock.Mock(return_value=True)
        controller.dual_arm_error = mock.Mock(return_value=0.01)
        controller.start_dual_tissue_surround = mock.Mock()
        controller.set_state = mock.Mock()
        controller.get_logger = lambda: self._logger()

        controller.advance_dual_tissue_deploy(
            pick.DUAL_TISSUE_DEPLOY_DWELL_S)

        controller.start_dual_tissue_surround.assert_called_once()
        controller.set_state.assert_not_called()

    def test_deploy_aborts_before_insertion_after_feedback_timeout(self):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.dual_surround_left_joints = np.ones(6)
        controller.dual_surround_right_joints = np.ones(6)
        controller.dual_commands_ready = mock.Mock(return_value=False)
        controller.dual_arm_error = mock.Mock(return_value=0.12)
        controller.joints = {"slide_joint": 0.1}
        controller.des_slide = 0.1
        controller.start_dual_tissue_surround = mock.Mock()
        controller.set_state = mock.Mock()
        controller.get_logger = lambda: self._logger()

        controller.advance_dual_tissue_deploy(
            pick.DUAL_TISSUE_DEPLOY_TIMEOUT_S)

        controller.start_dual_tissue_surround.assert_not_called()
        controller.set_state.assert_called_once_with(pick.STATE_ABORT)

    def test_contact_squeeze_keeps_non_gated_compliance(self):
        controller = self._motion_controller(
            np.zeros(6), np.zeros(6), gated=False)

        self.assertEqual(controller.advance_dual_tissue_motion(), "reached")
        controller.set_state.assert_not_called()

    @staticmethod
    def _slide_lift_controller(now, measured_slide):
        controller = pick.ShelfPickController.__new__(
            pick.ShelfPickController)
        controller.slide_grasp = 0.11
        controller.des_slide = controller.slide_grasp
        controller.state_t0 = 0.0
        controller.dual_lift_settled_since = None
        controller.shelf_level = "middle"
        controller.dual_top_extract_stage = 0
        controller.joints = {"slide_joint": measured_slide}
        controller.now = lambda: now
        controller.logger = DualTissueSafetyTests._logger()
        controller.get_logger = lambda: controller.logger
        controller.set_state = mock.Mock()
        return controller

    def test_middle_tissue_slide_lift_waits_for_stable_height(self):
        controller = self._slide_lift_controller(1.0, 0.05)

        self.assertEqual(
            controller.advance_dual_tissue_slide_lift(), "moving")
        self.assertAlmostEqual(controller.dual_lift_settled_since, 1.0)
        controller.now = lambda: 1.3
        self.assertEqual(
            controller.advance_dual_tissue_slide_lift(), "reached")
        self.assertAlmostEqual(controller.des_slide, 0.05)

    def test_middle_tissue_slide_lift_aborts_before_unraised_retreat(self):
        controller = self._slide_lift_controller(5.6, 0.10)

        self.assertEqual(
            controller.advance_dual_tissue_slide_lift(), "failed")
        controller.logger.error.assert_called_once()

    def test_middle_tissue_slide_lift_enters_lift_state(self):
        controller = self._slide_lift_controller(0.0, 0.11)

        controller.start_dual_tissue_slide_lift()

        controller.set_state.assert_called_once_with(pick.STATE_LIFT)
        self.assertAlmostEqual(controller.des_slide, 0.05)


if __name__ == "__main__":
    unittest.main()
