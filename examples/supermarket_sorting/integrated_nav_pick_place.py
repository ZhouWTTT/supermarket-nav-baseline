#!/usr/bin/env python3
"""Integrate the baseline SupermarketNavigator with the current shelf-pick pipeline.

This is a NEW orchestrator script only — it does not modify any existing file.
It subclasses ``ShelfPickController`` (yolo_aruco_shelf_pick.py) and reuses the
baseline navigation module (supermarket_navigation.py) purely by import.

Attempted end-to-end flow:

    start
      -> navigator drives to the shelf scan stations (GO_SCAN transit)
      -> YOLO + ArUco visual localisation (unchanged parent states)
      -> grasp the requested goods (unchanged parent states)
      -> navigator drives through the obstacle corridor to the delivery table
      -> arm extends, lowers the held product near the table, releases, retreats

Usage (inside the client container, mirroring yolo_aruco_shelf_pick.py)::

    python3 examples/supermarket_sorting/integrated_nav_pick_place.py \
        --target-kind kele --max-scan-cycles 2
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# sys.path: current pick pipeline first (its own module-level sys.path setup
# then wins for mmk2_kdl / perception imports), baseline dir appended after.
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import yolo_aruco_shelf_pick as pick  # noqa: E402  (parent pipeline, unmodified)

from sensor_msgs.msg import LaserScan  # noqa: E402
from supermarket_navigation import (  # noqa: E402  (baseline nav, unmodified)
    DELIVERY_APPROACH,
    DELIVERY_TABLE_COSTMAP_BOUNDS,
    DELIVERY_TABLE_XML_BOUNDS,
    SupermarketNavigator,
    WHOLE_BODY_KEEP_OUT_RADIUS,
    point_to_rect_clearance,
)

# ---------------------------------------------------------------------------
# MuJoCo compatibility shim.
#
# The official client image bundles mujoco 3.2.7, whose XML schema has no mesh
# ``inertia`` attribute and rejects the flat aruco marker quad used by the
# scene ("mesh volume is too small").  The FK model is only used to compute
# camera/site poses, so we replace the flat 3 cm quad with an equivalent
# 2 mm-thick box at runtime — valid on every mujoco version and identical for
# forward kinematics.  This only monkey-patches the class in this process; no
# repository file is modified.
# ---------------------------------------------------------------------------
import re as _re  # noqa: E402

_ARUCO_MESH_RE = _re.compile(
    r'<mesh name="aruco_marker_3cm_mesh".*?/>', _re.S)
_ARUCO_MESH_BOX = (
    '<mesh name="aruco_marker_3cm_mesh"\n'
    '          vertex="-0.015 -0.015 0  0.015 -0.015 0  0.015 0.015 0  '
    '-0.015 0.015 0  -0.015 -0.015 0.002  0.015 -0.015 0.002  '
    '0.015 0.015 0.002  -0.015 0.015 0.002"\n'
    '          texcoord="0 0  1 0  1 1  0 1  0 0  1 0  1 1  0 1"\n'
    '          face="0 1 2  0 2 3  4 5 6  4 6 7  0 4 5  0 5 1  '
    '1 5 6  1 6 2  2 6 7  2 7 3  3 7 4  3 4 0"/>')


def _mujoco_compat_xml(text: str) -> str:
    """Sanitise the scene XML so old client mujoco versions can load it."""
    return _ARUCO_MESH_RE.sub(_ARUCO_MESH_BOX, text)


try:  # noqa: E402
    import discoverse.robots.mmk2.mmk2_fk as _mmk2_fk_mod
except ImportError:
    _mmk2_fk_mod = None

if _mmk2_fk_mod is not None:
    _orig_mmk2fk_init = _mmk2_fk_mod.MMK2FK.__init__

    def _mmk2fk_compat_init(self, mjcf_path=None):
        if mjcf_path is None:
            task_dir = (
                pathlib.Path(_mmk2_fk_mod.DISCOVERSE_ROOT_DIR)
                / "examples" / "supermarket_sorting")
            src = task_dir / "mjcf" / "retail_competition.xml"
            runtime = pathlib.Path("/tmp/retail_competition_fk.xml")
            runtime.write_text(
                _mujoco_compat_xml(
                    src.read_text().replace(
                        "__REPO_ROOT__", str(task_dir))))
            mjcf_path = str(runtime)
        _orig_mmk2fk_init(self, mjcf_path)

    _mmk2_fk_mod.MMK2FK.__init__ = _mmk2fk_compat_init


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DELIVERY_TABLE_PLACE_WORLD = (-1.80, -3.35, 0.85)  # x, y, minimum approach z
DELIVERY_TABLE_TOP_Z_M = 0.767

# Product centre heights above their supporting surface.  These are the
# physical half-heights of the collision geometry in the competition scene.
# The placement controller targets the product centre at table top + this
# value + a small clearance, rather than opening the gripper at one fixed TCP
# height for every product.
PRODUCT_HALF_HEIGHT_M = {
    "sanmingzhi": 0.0494,
    "heweidao": 0.0525,
    "shupian": 0.1050,
    "zhijin": 0.0440,
    "maidong": 0.1050,
    "kele": 0.0725,
    "kouxiangtang": 0.0400,
    "pingguo": 0.0350,
    "chengzi": 0.0370,
}
PLACE_PRODUCT_BOTTOM_CLEARANCE_M = 0.015
PLACE_APPROACH_CLEARANCE_M = 0.060
PLACE_RELEASE_HEIGHT_LOWER_TOLERANCE_M = 0.010
PLACE_RELEASE_HEIGHT_UPPER_TOLERANCE_M = 0.035
PLACE_DESCENT_SLIDE_STEP_M = 0.0015
PLACE_CLEAR_TABLE_MARGIN_M = 0.060
PLACE_CLEAR_TABLE_SPEED_MPS = 0.10
PLACE_CLEAR_TABLE_TIMEOUT_S = 15.0
NAV_TRANSIT_GATE_M = 0.35          # beyond this distance, use the navigator
NAV_LASER_STALE_S = 0.50           # fail safe if the 12 Hz scan stops
NAV_STATE_STALE_S = 0.50           # odom/joints must also remain live
NAV_PROGRESS_LOG_S = 3.0

# Keep the held product clear of the shelf before delivery navigation starts
# turning the base.  The arms and product still protrude toward the shelf at
# the end of the parent grasp state machine.
BACKUP_SPEED_MPS = 0.10
BACKUP_TIMEOUT_S = 8.0

# A* stops outside the table's inflated costmap.  From that safe pose, make a
# short, slow, yaw-controlled final approach before extending the arm.  The
# physical chassis front remains clear of the table at the nominal endpoint.
PLACE_CREEP_DISTANCE_M = 0.20
PLACE_CREEP_SPEED_MPS = 0.04
PLACE_CREEP_FRONT_STOP_M = 0.30
PLACE_CREEP_YAW_GAIN = 2.0
PLACE_CREEP_MAX_ANGULAR_RPS = 0.30
PLACE_RELEASE_TABLE_MARGIN_M = 0.04
# 放桌手臂到位超时与备选姿态重试：第 0 步发出放桌手臂目标后，若长时间不
# 收敛（肩关节被货物/机体卡住），自动换下一组 d/z/slide 候选重新解 IK，
# 多次仍不行才抛错让上层重试/跳过，避免在放桌点永久冻结。
PLACE_ARM_SETTLE_TIMEOUT_S = 8.0
PLACE_ARM_RETRY_MAX = 3

# Compact post-place travel pose (mirrors INIT_ARM in supermarket_sorting_client)
PLACE_RETREAT_ARM_L = [0.0, -0.166, 0.032, 0.0, 1.571, 2.223]
PLACE_RETREAT_ARM_R = [0.0, -0.166, 0.032, 0.0, -1.571, -2.223]


class IntegratedNavPickPlace(pick.ShelfPickController):
    """Shelf-pick controller whose driving is done by the baseline navigator.

    Flow phases (``flow_phase``):
      "grab"            — parent state machine (GO_SCAN/SCAN/ALIGN/.../DONE);
                          its drive_to() is overridden to use the navigator for
                          long-range transit while keeping the precise final
                          alignment untouched.
      "backup"          — reverse with yaw hold to clear the shelf.
      "nav_to_delivery" — navigator to DELIVERY_APPROACH with goods held.
      "place"           — extend, descend near the table, release, retreat.
      "done"            — flow finished.
    """

    def __init__(
            self, target_kind: str, max_scan_cycles: int,
            tcp_diagnostic_ground_truth: bool, scan_skip_lower: bool,
            place_x: float = DELIVERY_TABLE_PLACE_WORLD[0],
            place_y: float = DELIVERY_TABLE_PLACE_WORLD[1],
            place_z: float = DELIVERY_TABLE_PLACE_WORLD[2],
            place_release_dwell_s: float = 2.0,
            place_retreat_dwell_s: float = 1.0,
            nav_during_scan: bool = True,
            backup_after_grab_m: float = 0.20,
            place_creep_m: float = PLACE_CREEP_DISTANCE_M,
            close_recheck: bool = True):
        super().__init__(
            target_kind, max_scan_cycles,
            tcp_diagnostic_ground_truth, scan_skip_lower,
            close_recheck=close_recheck)

        self.nav_during_scan = nav_during_scan
        self.backup_after_grab_m = float(backup_after_grab_m)
        self.place_creep_m = float(place_creep_m)
        self.place_world = np.array(
            [place_x, place_y, place_z], dtype=float)
        self.place_min_approach_z = float(place_z)
        self.place_release_dwell_s = place_release_dwell_s
        self.place_retreat_dwell_s = place_retreat_dwell_s

        # ── laser for the navigator ──
        self.laser_msg = None
        self.last_scan_time = None
        self.create_subscription(
            LaserScan, "/slamware_ros_sdk_server_node/scan",
            self._scan_cb, 10)

        # ── baseline navigator (same interface as the demo) ──
        self.nav = SupermarketNavigator()

        # ── our flow state ──
        self.flow_phase = "grab"
        self._nav_goal = None
        self._nav_last_log = 0.0
        self._last_nav_reason = None
        self.place_stage = 0
        self.place_t0 = 0.0
        self.place_arm_joints = None
        self.place_slide_cmd = None
        self.place_release_world = None
        self.place_release_slide_cmd = None
        self._place_ik_attempted = None
        self._place_arm_target_sent = False
        self._place_arm_sent_t0 = None
        self._place_candidate_skip = 0
        self._place_descent_sent = False
        self._place_retreat_sent = False
        self._dual_descent_sent = False
        self.dual_release_slide_cmd = None
        self.place_creep_start_y = None
        self.place_creep_done = False
        self._backup_start_xy = None
        self._backup_start_yaw = 0.0
        self._backup_t0 = 0.0
        self._backup_logged = False
        self._flow_done_logged = False
        self._table_escape_logged = False
        self._laser_warn_log = 0.0
        self._state_warn_log = 0.0

        self.get_logger().info(
            "integrated nav+pick+place ready; "
            f"nav_during_scan={nav_during_scan} "
            f"close_recheck={int(close_recheck)} "
            f"place_world={np.round(self.place_world, 3)} "
            f"backup_after_grab={self.backup_after_grab_m:.2f}m "
            f"place_creep={self.place_creep_m:.2f}m "
            f"release_dwell={place_release_dwell_s}s "
            f"retreat_dwell={place_retreat_dwell_s}s")

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _scan_cb(self, msg) -> None:
        self.laser_msg = msg
        self.last_scan_time = self.now()

    def _laser_stale(self, now: float) -> bool:
        return (
            self.last_scan_time is None
            or now - self.last_scan_time > NAV_LASER_STALE_S)

    # ------------------------------------------------------------------
    # drive_to override — navigator for transit, parent logic for the last
    # few centimetres where 0.10 m navigator tolerance is not precise enough.
    # ------------------------------------------------------------------
    def drive_to(self, target_xy, final_yaw: float,
                 position_tolerance: float = 0.055) -> bool:
        target = np.asarray(target_xy, dtype=float)
        distance = float(np.linalg.norm(target - self.base_xy))

        # A previous delivery may leave the chassis inside the conservative
        # whole-body table keep-out.  The normal navigator correctly forbids
        # turning there, but that also means it cannot align toward the next
        # shelf.  First retrace the final south-facing approach in reverse;
        # this moves directly away from known static geometry without an
        # unsafe in-place arm sweep.
        table_clearance = point_to_rect_clearance(
            float(self.base_xy[0]), float(self.base_xy[1]),
            DELIVERY_TABLE_COSTMAP_BOUNDS)
        safe_clearance = (
            WHOLE_BODY_KEEP_OUT_RADIUS + PLACE_CLEAR_TABLE_MARGIN_M)
        if table_clearance < safe_clearance:
            yaw_error = pick.wrap_to_pi(-math.pi / 2.0 - self.base_yaw)
            if abs(yaw_error) <= 0.20:
                self.set_twist(
                    -PLACE_CLEAR_TABLE_SPEED_MPS,
                    float(np.clip(2.0 * yaw_error, -0.25, 0.25)))
            else:
                # Do not rotate a deployed whole body beside the table.  This
                # should only occur after an external pose disturbance.
                self.set_twist(0.0, 0.0)
            if not self._table_escape_logged:
                self._table_escape_logged = True
                self.get_logger().warn(
                    "starting inside delivery-table keep-out; "
                    f"clearance={table_clearance:.3f}m "
                    f"required={safe_clearance:.3f}m; reversing north "
                    "before normal navigation")
            return False
        if self._table_escape_logged:
            self._table_escape_logged = False
            self.get_logger().info(
                f"delivery-table startup escape complete; "
                f"clearance={table_clearance:.3f}m")

        if (self.nav_during_scan
                and distance > max(NAV_TRANSIT_GATE_M,
                                   position_tolerance + 0.15)):
            now = self.now()
            goal = (float(target[0]), float(target[1]), float(final_yaw))
            if self._nav_goal != goal:
                self._nav_goal = goal
                self.nav.set_goal(*goal)
                self._nav_last_log = 0.0

            if self._laser_stale(now):
                self.set_twist(0.0, 0.0)
                if now - self._laser_warn_log > 1.0:
                    self.get_logger().warn(
                        "waiting for fresh laser scan during transit "
                        f"(last={self.last_scan_time})")
                    self._laser_warn_log = now
                return False

            v, w, reached = self.nav.update(
                self.base_xy[0], self.base_xy[1], self.base_yaw,
                laser_msg=self.laser_msg, time_now=now)
            self.set_twist(v, w)

            ctrl = self.nav.controller
            if (ctrl.stop_reason is not None
                    and ctrl.stop_reason != self._last_nav_reason):
                self._last_nav_reason = ctrl.stop_reason
                self.get_logger().info(
                    f"[nav] stop_reason={ctrl.stop_reason} "
                    f"lidar={ctrl.lidar_clearance:.2f}m "
                    f"v={v:.2f} w={w:.2f}")

            if now - self._nav_last_log >= NAV_PROGRESS_LOG_S:
                self._nav_last_log = now
                self.get_logger().info(
                    f"[nav] to=({goal[0]:.2f},{goal[1]:.2f},"
                    f"{math.degrees(goal[2]):.0f}°) "
                    f"pos=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
                    f"yaw={math.degrees(self.base_yaw):.0f}° "
                    f"dist={distance:.2f}m v={v:.2f} w={w:.2f} "
                    f"reached={reached}")

            if not reached:
                return False
            # Navigator coarse arrival → fall through to precise alignment.

        return super().drive_to(target_xy, final_yaw, position_tolerance)

    # ------------------------------------------------------------------
    # flow hooks
    # ------------------------------------------------------------------
    def _on_grab_complete(self) -> None:
        self.get_logger().info(
            f"[flow] goods grabbed (marker={self.target_marker_id}, "
            f"kind={self.target_kind}, state={self.state}); "
            "preparing delivery transit")
        if self.backup_after_grab_m > 1e-4:
            self.flow_phase = "backup"
            self._backup_start_xy = self.base_xy.copy()
            self._backup_start_yaw = float(self.base_yaw)
            self._backup_t0 = self.now()
            self._backup_logged = False
            self.get_logger().info(
                f"[flow] backing up {self.backup_after_grab_m:.2f}m "
                "before delivery rotation")
            return
        self._start_delivery_navigation()

    def _start_delivery_navigation(self) -> None:
        self.flow_phase = "nav_to_delivery"
        self._nav_goal = None
        self._nav_last_log = 0.0
        self.nav.set_goal(*DELIVERY_APPROACH)

    def _backup_tick(self) -> None:
        """Reverse along the grasp heading while holding the current yaw."""
        now = self.now()
        if self._backup_start_xy is None:
            self._backup_start_xy = self.base_xy.copy()
            self._backup_start_yaw = float(self.base_yaw)
            self._backup_t0 = now

        heading = np.array([
            math.cos(self._backup_start_yaw),
            math.sin(self._backup_start_yaw),
        ])
        moved_back = float(np.dot(
            self._backup_start_xy - self.base_xy, heading))
        yaw_err = pick.wrap_to_pi(self._backup_start_yaw - self.base_yaw)
        elapsed = now - self._backup_t0

        reached = moved_back >= self.backup_after_grab_m
        timed_out = elapsed > BACKUP_TIMEOUT_S
        if reached or timed_out:
            self.set_twist(0.0, 0.0)
            self._start_delivery_navigation()
            message = (
                f"[flow] backup finished (moved={moved_back:.3f}m, "
                f"elapsed={elapsed:.1f}s); starting delivery navigation")
            if timed_out and not reached:
                self.get_logger().warn(message + " after timeout")
            else:
                self.get_logger().info(message)
            return

        angular = float(np.clip(2.0 * yaw_err, -0.6, 0.6))
        self.set_twist(-BACKUP_SPEED_MPS, angular)
        if not self._backup_logged and elapsed >= 1.0:
            self._backup_logged = True
            self.get_logger().info(
                f"[backup] dist={moved_back:.3f}/"
                f"{self.backup_after_grab_m:.2f}m "
                f"yaw_err={math.degrees(yaw_err):.1f}°")

    def _nav_to_delivery_tick(self) -> None:
        now = self.now()
        if self._laser_stale(now):
            self.set_twist(0.0, 0.0)
            if now - self._laser_warn_log > 1.0:
                self.get_logger().warn(
                    "waiting for fresh laser scan on the way to delivery")
                self._laser_warn_log = now
            return

        v, w, reached = self.nav.update(
            self.base_xy[0], self.base_xy[1], self.base_yaw,
            laser_msg=self.laser_msg, time_now=now)
        self.set_twist(v, w)

        ctrl = self.nav.controller
        if (ctrl.stop_reason is not None
                and ctrl.stop_reason != self._last_nav_reason):
            self._last_nav_reason = ctrl.stop_reason
            self.get_logger().info(
                f"[nav→delivery] stop_reason={ctrl.stop_reason} "
                f"lidar={ctrl.lidar_clearance:.2f}m")

        if now - self._nav_last_log >= NAV_PROGRESS_LOG_S:
            self._nav_last_log = now
            self.get_logger().info(
                f"[nav→delivery] pos=({self.base_xy[0]:.2f},"
                f"{self.base_xy[1]:.2f}) "
                f"yaw={math.degrees(self.base_yaw):.0f}° "
                f"v={v:.2f} w={w:.2f} reached={reached}")

        if reached:
            # Navigator yaw tolerance is 0.15 rad; refine to face south.
            yaw_err = pick.wrap_to_pi(-math.pi / 2.0 - self.base_yaw)
            if abs(yaw_err) > 0.03:
                self.set_twist(0.0, 2.0 * yaw_err)
                return
            self.set_twist(0.0, 0.0)
            self.flow_phase = "place"
            self.place_stage = 0
            self.place_t0 = now
            self.get_logger().info(
                f"[flow] arrived at delivery approach "
                f"pos=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
                f"yaw={math.degrees(self.base_yaw):.0f}°; placing")

    def _set_selected_grip(self, value: float) -> None:
        if self.grasp_arm == "r":
            self.des_right_grip = float(value)
        else:
            self.des_left_grip = float(value)

    def _product_release_z(self) -> float:
        """Return TCP height that leaves the product just above the table.

        The TCP does not always pass through the product centre.  Short top
        goods, for example, deliberately use a higher wrist pose to clear the
        shelf.  Preserve that grasp-time vertical offset so the product bottom
        -- rather than the wrist -- receives the configured table clearance.
        """
        half_height = PRODUCT_HALF_HEIGHT_M[self.target_kind]
        tcp_above_product_center = 0.0
        if self.target_world is not None:
            if (self.use_dual_tissue_grasp
                    and self.dual_contact_tcp_z is not None):
                tcp_above_product_center = (
                    float(self.dual_contact_tcp_z)
                    - float(self.target_world[2]))
            elif self.forward_contact_world is not None:
                tcp_above_product_center = (
                    float(self.forward_contact_world[2])
                    - float(self.target_world[2]))
        return (
            DELIVERY_TABLE_TOP_Z_M
            + half_height
            + PLACE_PRODUCT_BOTTOM_CLEARANCE_M
            + tcp_above_product_center)

    def _compute_place_arm_joints(
            self, candidate_skip: int = 0) -> np.ndarray | None:
        """Solve an approach pose with enough slide travel for a low release.

        The numeric IK depends heavily on the reference joints.  At the
        delivery pose the shelf pregrasp joints are far from any solution, so
        we also try the compact INIT pose and the measured joints.  Once an
        approach pose is found, the final vertical descent keeps the arm joints
        fixed and increases the downward-facing slide joint.  The result
        (including failure) is cached per ``candidate_skip`` level to avoid
        per-tick recomputation.  When the arm does not converge (jammed by
        the held goods or the robot body), ``candidate_skip`` skips already
        tried approach candidates and returns the next distinct solution.
        """
        if (self._place_ik_attempted is not None
                and self._place_ik_attempted == candidate_skip):
            return self.place_arm_joints
        self._place_ik_attempted = candidate_skip

        measured = self.selected_arm_positions()
        compact = np.asarray(
            PLACE_RETREAT_ARM_R if self.grasp_arm == "r"
            else PLACE_RETREAT_ARM_L, dtype=float)
        refs = [compact, measured]
        if self.pregrasp_arm_joints is not None:
            refs.append(np.asarray(self.pregrasp_arm_joints, dtype=float))

        bx, by = float(self.base_xy[0]), float(self.base_xy[1])
        # Table top is ~0.767 m and the centre is at y=-3.41.  After the final
        # base creep, 0.55--0.65 m reaches the interior without forcing the arm
        # to its old 0.70--0.80 m reach limit.
        release_z = self._product_release_z()
        minimum_approach_z = max(
            self.place_min_approach_z,
            release_z + PLACE_APPROACH_CLEARANCE_M)
        z_candidates = tuple(
            minimum_approach_z + offset for offset in (0.0, 0.02, 0.04))
        d_candidates = (0.65, 0.60, 0.55)
        # Top-shelf grasps pin the slide at SLIDE_MIN, which leaves the arm too
        # high to reach the table; raising the slide lowers the whole arm into
        # reach.  Middle/lower grasps keep their grasp slide.
        slide_candidates = []
        for slide in (self.slide_grasp, 0.20, 0.30, 0.35, 0.40, 0.45):
            slide = float(np.clip(slide, pick.SLIDE_MIN, pick.SLIDE_MAX))
            if not any(abs(slide - item) < 1e-6
                       for item in slide_candidates):
                slide_candidates.append(slide)

        solved_count = 0
        for d in d_candidates:
            for z in z_candidates:
                descent = z - release_z
                for slide in slide_candidates:
                    release_slide = slide + descent
                    if release_slide > pick.SLIDE_MAX + 1e-6:
                        continue
                    world = np.array([bx, by - d, z], dtype=float)
                    for ref in refs:
                        joints = self._solve_place_world(world, ref, slide)
                        if joints is None:
                            continue
                        if solved_count < candidate_skip:
                            solved_count += 1
                            continue
                        self.place_world = world
                        self.place_arm_joints = joints
                        self.place_slide_cmd = slide
                        self.place_release_world = np.array(
                            [world[0], world[1], release_z], dtype=float)
                        self.place_release_slide_cmd = release_slide
                        self.get_logger().info(
                            f"[place] approach IK={np.round(world, 3)} "
                            f"release={np.round(self.place_release_world, 3)} "
                            f"slide={slide:.3f}->{release_slide:.3f} "
                            f"descent={descent:.3f}m d={d:.2f}m "
                            f"refs_tried={len(refs)}")
                        return joints

        self.get_logger().error(
            "[place] no approach IK with enough downward slide travel; "
            "keeping gripper closed")
        return None

    def _solve_place_world(
            self, world: np.ndarray, reference: np.ndarray,
            slide: float) -> np.ndarray | None:
        """Solve the selected arm to ``world`` at a given slide height."""
        target = np.eye(4)
        target[:3, 3] = self.world_to_footprint(world)
        reference = np.asarray(reference, dtype=float)
        ref_with_slide = np.concatenate(([slide], reference))
        try:
            if self.grasp_arm == "r":
                solutions = self.kdl.inverse_kinematics(
                    T_right=target, target_height=slide,
                    ref_pos=ref_with_slide)
            else:
                solutions = self.kdl.inverse_kinematics(
                    T_left=target, target_height=slide,
                    ref_pos=ref_with_slide)
        except Exception:  # noqa: BLE001 - try next candidate
            return None
        if solutions is None or len(solutions) == 0:
            return None
        candidates = [np.asarray(item[1:], dtype=float) for item in solutions]
        return min(
            candidates,
            key=lambda item: float(np.max(np.abs(item - reference))))

    def _laser_front_range(self) -> float | None:
        msg = self.laser_msg
        if msg is None or not msg.ranges:
            return None
        n = len(msg.ranges)
        half = max(1, int(round(n * 0.10)))
        centre = n // 2
        window = msg.ranges[max(0, centre - half):centre + half]
        valid = [float(r) for r in window
                 if r is not None and math.isfinite(float(r))
                 and r > 0.05 and r < 8.0]
        return min(valid) if valid else None

    def _advance_place_creep(self) -> bool:
        """Move slowly from the navigation endpoint to the arm-place pose."""
        if self.place_creep_done:
            return True
        if self.place_creep_start_y is None:
            self.place_creep_start_y = float(self.base_xy[1])

        front = self._laser_front_range()
        crept = float(self.place_creep_start_y - self.base_xy[1])
        distance_reached = crept >= self.place_creep_m
        front_reached = (
            front is not None and front <= PLACE_CREEP_FRONT_STOP_M)
        if self.place_creep_m <= 1e-4 or distance_reached or front_reached:
            self.set_twist(0.0, 0.0)
            self.place_creep_done = True
            reason = (
                "distance" if distance_reached else
                "lidar" if front_reached else "disabled")
            self.get_logger().info(
                f"[place] final approach finished (reason={reason} "
                f"crept={crept:.3f}m front={front} "
                f"base=({self.base_xy[0]:.3f},{self.base_xy[1]:.3f}))")
            return True

        yaw_err = pick.wrap_to_pi(-math.pi / 2.0 - self.base_yaw)
        angular = float(np.clip(
            PLACE_CREEP_YAW_GAIN * yaw_err,
            -PLACE_CREEP_MAX_ANGULAR_RPS,
            PLACE_CREEP_MAX_ANGULAR_RPS))
        # Pause translation if odometry shows an unexpectedly large yaw error;
        # correct it first so a nominally southward creep cannot cut sideways.
        linear = PLACE_CREEP_SPEED_MPS if abs(yaw_err) <= 0.10 else 0.0
        self.set_twist(linear, angular)
        return False

    @staticmethod
    def _tcp_over_delivery_table(tcp: np.ndarray | None) -> bool:
        """Require the measured release point to lie inside the tabletop."""
        if tcp is None or np.asarray(tcp).shape != (3,):
            return False
        x_min, y_min, x_max, y_max = DELIVERY_TABLE_XML_BOUNDS
        margin = PLACE_RELEASE_TABLE_MARGIN_M
        return (
            np.all(np.isfinite(tcp))
            and x_min + margin <= float(tcp[0]) <= x_max - margin
            and y_min + margin <= float(tcp[1]) <= y_max - margin)

    @staticmethod
    def _tcp_at_release_height(
            tcp: np.ndarray | None, target_z: float) -> bool:
        """Require measured TCP height to be close to the low release pose."""
        return (
            tcp is not None
            and np.asarray(tcp).shape == (3,)
            and np.all(np.isfinite(tcp))
            and target_z - PLACE_RELEASE_HEIGHT_LOWER_TOLERANCE_M
            <= float(tcp[2])
            <= target_z + PLACE_RELEASE_HEIGHT_UPPER_TOLERANCE_M)

    def _dual_release_world(self) -> np.ndarray | None:
        """Approximate the held tissue centre by the two measured TCPs."""
        left = self.arm_tcp_world("left")
        right = self.arm_tcp_world("right")
        if left is None or right is None:
            return None
        return 0.5 * (np.asarray(left) + np.asarray(right))

    def _place_tick(self) -> None:
        now = self.now()
        if self.place_stage == 4:
            self._clear_delivery_table_tick(now)
            return
        if self.use_dual_tissue_grasp:
            self._place_tick_dual(now)
            return

        if self.place_stage == 0:
            # 1) perform a guarded final base approach; 2) solve and reach a
            # high approach pose over the table.
            if not self._advance_place_creep():
                return
            if self.place_arm_joints is None:
                self.place_arm_joints = self._compute_place_arm_joints(
                    candidate_skip=self._place_candidate_skip)
                if self.place_arm_joints is not None:
                    # Send once — set_selected_arm_target resets
                    # commands_ready_since, so calling it every tick would
                    # prevent the settling gate from ever passing.
                    self.set_selected_arm_target(self.place_arm_joints)
                    if self.place_slide_cmd is not None:
                        self.des_slide = self.place_slide_cmd
                    self._place_arm_target_sent = True
                    self._place_arm_sent_t0 = now
                else:
                    raise RuntimeError(
                        "place IK failed; refusing to release goods off-table")
            if self.commands_ready(arm_tolerance=0.05, slide_tolerance=0.05):
                tcp = self.selected_tcp_world()
                if not self._tcp_over_delivery_table(tcp):
                    raise RuntimeError(
                        "measured place TCP is outside delivery tabletop: "
                        f"{None if tcp is None else np.round(tcp, 3)}")
                self.get_logger().info(
                    f"[place] arm at approach pose; tcp="
                    f"{None if tcp is None else np.round(tcp, 3)}; "
                    "starting vertical descent with gripper closed")
                self.place_stage = 1
                self.place_t0 = now
                if self.place_release_slide_cmd is None:
                    raise RuntimeError(
                        "place release slide target was not computed")
                self.des_slide = self.place_release_slide_cmd
                self.commands_ready_since = None
                self._place_descent_sent = True
            elif (self._place_arm_sent_t0 is not None
                    and now - self._place_arm_sent_t0
                    >= PLACE_ARM_SETTLE_TIMEOUT_S):
                # 手臂长时间不到位（可能被货物/机体卡住）：换下一组备选姿态
                # 重新解 IK；多次仍不行才抛错交给上层重试/跳过。
                self._place_candidate_skip += 1
                if self._place_candidate_skip > PLACE_ARM_RETRY_MAX:
                    raise RuntimeError(
                        f"place arm did not converge after "
                        f"{PLACE_ARM_RETRY_MAX} retries "
                        f"(arm_err={self.selected_arm_error():.3f}rad)")
                self.get_logger().warn(
                    f"[place] arm not converged within "
                    f"{PLACE_ARM_SETTLE_TIMEOUT_S:.0f}s "
                    f"(arm_err={self.selected_arm_error():.3f}rad); "
                    f"retrying candidate "
                    f"{self._place_candidate_skip}/{PLACE_ARM_RETRY_MAX}")
                self.place_arm_joints = None
                self._place_arm_sent_t0 = None
                self._place_arm_target_sent = False
                self.commands_ready_since = None
        elif self.place_stage == 1:
            # Keep the product clamped while the slide lowers the complete arm
            # vertically.  Opening is forbidden until measured XY and Z both
            # confirm a near-table release pose.
            if not self.commands_ready(
                    arm_tolerance=0.05, slide_tolerance=0.025):
                return
            tcp = self.selected_tcp_world()
            if not self._tcp_over_delivery_table(tcp):
                raise RuntimeError(
                    "measured lowered TCP is outside delivery tabletop: "
                    f"{None if tcp is None else np.round(tcp, 3)}")
            target_z = float(self.place_release_world[2])
            if not self._tcp_at_release_height(tcp, target_z):
                raise RuntimeError(
                    "measured TCP did not reach safe release height: "
                    f"tcp={None if tcp is None else np.round(tcp, 3)} "
                    f"target_z={target_z:.3f}")
            self.get_logger().info(
                f"[place] low pose verified; tcp={np.round(tcp, 3)} "
                f"product_bottom_clearance="
                f"{PLACE_PRODUCT_BOTTOM_CLEARANCE_M:.3f}m; releasing")
            self.place_stage = 2
            self.place_t0 = now
        elif self.place_stage == 2:
            self._set_selected_grip(pick.GRIP_OPEN)
            if now - self.place_t0 >= self.place_release_dwell_s:
                self.get_logger().info(
                    "[place] gripper released; retreating arm")
                self.place_stage = 3
                self.place_t0 = now
                self._place_retreat_sent = False
        elif self.place_stage == 3:
            self._set_selected_grip(pick.GRIP_OPEN)
            if not self._place_retreat_sent:
                self._place_retreat_sent = True
                joints = (
                    PLACE_RETREAT_ARM_R if self.grasp_arm == "r"
                    else PLACE_RETREAT_ARM_L)
                self.set_selected_arm_target(np.asarray(joints, dtype=float))
                self.des_slide = pick.SLIDE_REFERENCE_COMMAND
            if (now - self.place_t0 >= self.place_retreat_dwell_s
                    and self.commands_ready(
                        arm_tolerance=0.08, slide_tolerance=0.05)):
                self.place_stage = 4
                self.place_t0 = now
                self.get_logger().info(
                    "[place] arm safely retracted; backing out of the "
                    "delivery-table keep-out")

    def _place_tick_dual(self, now: float) -> None:
        """Dual-arm tissue place with measured vertical slide descent."""
        if self.place_stage == 0:
            self.des_left_grip = pick.DUAL_TISSUE_GRIP_COMMAND
            self.des_right_grip = pick.DUAL_TISSUE_GRIP_COMMAND
            if not self._advance_place_creep():
                return
            release_world = self._dual_release_world()
            if not self._tcp_over_delivery_table(release_world):
                raise RuntimeError(
                    "dual-arm release point is outside delivery tabletop: "
                    f"{None if release_world is None else np.round(release_world, 3)}")
            if not self._dual_descent_sent:
                measured_slide = self.joints.get("slide_joint")
                if measured_slide is None:
                    return
                target_z = self._product_release_z()
                # slide axis is world -Z, so adding the measured centre-height
                # error moves the clamped tissue centre to target_z.
                target_slide = float(measured_slide) + (
                    float(release_world[2]) - target_z)
                if not pick.SLIDE_MIN <= target_slide <= pick.SLIDE_MAX:
                    raise RuntimeError(
                        "dual-arm safe release is outside slide range: "
                        f"current={float(measured_slide):.3f} "
                        f"target={target_slide:.3f} "
                        f"tcp_z={float(release_world[2]):.3f} "
                        f"release_z={target_z:.3f}")
                self.dual_release_slide_cmd = target_slide
                self.des_slide = target_slide
                self.commands_ready_since = None
                self._dual_descent_sent = True
                self.get_logger().info(
                    f"[place-dual] descending with tissue clamped; "
                    f"centre={np.round(release_world, 3)} "
                    f"target_z={target_z:.3f} "
                    f"slide={float(measured_slide):.3f}->{target_slide:.3f}")
                return
            if not self.dual_commands_ready(
                    arm_tolerance=0.05, slide_tolerance=0.025):
                return
            release_world = self._dual_release_world()
            target_z = self._product_release_z()
            if (not self._tcp_over_delivery_table(release_world)
                    or not self._tcp_at_release_height(
                        release_world, target_z)):
                raise RuntimeError(
                    "dual-arm measured release pose is unsafe: "
                    f"centre={None if release_world is None else np.round(release_world, 3)} "
                    f"target_z={target_z:.3f}")
            self.place_stage = 1
            self.place_t0 = now
            self.get_logger().info(
                f"[place-dual] low pose verified; "
                f"centre={np.round(release_world, 3)}; releasing")
        elif self.place_stage == 1:
            self.des_left_grip = pick.GRIP_OPEN
            self.des_right_grip = pick.GRIP_OPEN
            if now - self.place_t0 >= self.place_release_dwell_s:
                self.place_stage = 2
                self.place_t0 = now
        elif self.place_stage == 2:
            self.des_left_arm = np.asarray(PLACE_RETREAT_ARM_L, dtype=float)
            self.des_right_arm = np.asarray(PLACE_RETREAT_ARM_R, dtype=float)
            self.des_left_grip = pick.GRIP_OPEN
            self.des_right_grip = pick.GRIP_OPEN
            self.des_slide = pick.SLIDE_REFERENCE_COMMAND
            if (now - self.place_t0 >= self.place_retreat_dwell_s
                    and self.dual_commands_ready(
                        arm_tolerance=0.08, slide_tolerance=0.05)):
                self.place_stage = 4
                self.place_t0 = now
                self.get_logger().info(
                    "[place-dual] arms safely retracted; backing out of "
                    "the delivery-table keep-out")

    def _clear_delivery_table_tick(self, now: float) -> None:
        """Back away after release so the next order can safely turn."""
        if self.use_dual_tissue_grasp:
            self.des_left_grip = pick.GRIP_OPEN
            self.des_right_grip = pick.GRIP_OPEN
        else:
            self._set_selected_grip(pick.GRIP_OPEN)

        clearance = point_to_rect_clearance(
            float(self.base_xy[0]), float(self.base_xy[1]),
            DELIVERY_TABLE_COSTMAP_BOUNDS)
        required = WHOLE_BODY_KEEP_OUT_RADIUS + PLACE_CLEAR_TABLE_MARGIN_M
        if clearance >= required:
            self.set_twist(0.0, 0.0)
            self.flow_phase = "done"
            self.place_t0 = now
            self.get_logger().info(
                f"[flow] PLACE COMPLETE — {self.target_kind} delivered; "
                f"table cleared (clearance={clearance:.3f}m); "
                f"base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f})")
            return

        elapsed = now - self.place_t0
        if elapsed >= PLACE_CLEAR_TABLE_TIMEOUT_S:
            self.set_twist(0.0, 0.0)
            raise RuntimeError(
                "could not back out of delivery-table keep-out after place: "
                f"clearance={clearance:.3f}m required={required:.3f}m")

        yaw_error = pick.wrap_to_pi(-math.pi / 2.0 - self.base_yaw)
        if abs(yaw_error) > 0.20:
            self.set_twist(0.0, 0.0)
            raise RuntimeError(
                "unsafe yaw while backing away from delivery table: "
                f"error={math.degrees(yaw_error):.1f}deg")
        self.set_twist(
            -PLACE_CLEAR_TABLE_SPEED_MPS,
            float(np.clip(2.0 * yaw_error, -0.25, 0.25)))

    def smooth_commands(self) -> None:
        """Use a slower slide rate during the final loaded descent."""
        previous_slide = self.cmd_slide
        super().smooth_commands()
        single_descent = (
            not self.use_dual_tissue_grasp and self.place_stage == 1)
        dual_descent = (
            self.use_dual_tissue_grasp
            and self.place_stage == 0
            and self._dual_descent_sent)
        if self.flow_phase == "place" and (single_descent or dual_descent):
            self.cmd_slide = float(self.slew(
                previous_slide,
                self.des_slide,
                PLACE_DESCENT_SLIDE_STEP_M))

    # ------------------------------------------------------------------
    # main control loop
    # ------------------------------------------------------------------
    def tick(self) -> None:
        if self.base_xy is None or not self.joints:
            return
        now = self.now()
        odom_stale = (
            self.last_odom_time is None
            or now - self.last_odom_time > NAV_STATE_STALE_S)
        joints_stale = (
            self.last_joint_time is None
            or now - self.last_joint_time > NAV_STATE_STALE_S)
        laser_stale = self._laser_stale(now)
        if odom_stale or joints_stale or laser_stale:
            self.cmd_vel_pub.publish(pick.Twist())
            if now - self._state_warn_log > 1.0:
                self.get_logger().warn(
                    "stopping for stale robot feedback "
                    f"(odom_stale={odom_stale}, "
                    f"joints_stale={joints_stale}, "
                    f"laser_stale={laser_stale})")
                self._state_warn_log = now
            return
        if not self.initialized:
            self.initialize_commands()

        if self.flow_phase == "grab":
            prev_state = self.state
            super().tick()
            if (prev_state != pick.STATE_DONE
                    and self.state == pick.STATE_DONE):
                self._on_grab_complete()
            return

        # Post-grab phases: same command pipeline tail as the parent tick.
        self.set_twist(0.0, 0.0)
        if self.flow_phase == "backup":
            self._backup_tick()
        elif self.flow_phase == "nav_to_delivery":
            self._nav_to_delivery_tick()
        elif self.flow_phase == "place":
            self._place_tick()
        elif not self._flow_done_logged:
            self._flow_done_logged = True
            self.get_logger().info(
                f"[flow] done — final base=({self.base_xy[0]:.2f},"
                f"{self.base_xy[1]:.2f})")
        if (self.flow_phase == "done"
                and self.now() - self.place_t0 > 3.0):
            self.get_logger().info("[flow] flow finished; shutting down")
            import rclpy
            rclpy.shutdown()
            return

        self.apply_manip_base_hold()
        self.smooth_commands()
        self.publish_commands()

        if self.now() - self.last_status_log > 1.0:
            self.get_logger().info(
                f"[flow] phase={self.flow_phase} "
                f"state={self.state} place_stage={self.place_stage} "
                f"base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
                f"yaw={math.degrees(self.base_yaw):.0f}°")
            self.last_status_log = self.now()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="integrated nav + YOLO/ArUco pick + place client")
    parser.add_argument(
        "--target-kind", required=True,
        choices=sorted(pick.PRODUCT_CENTER_ABOVE_MARKER_M),
        help="exact goods class to remove from the shelf")
    parser.add_argument(
        "--order-id", default="manual",
        help="anonymous competition order id recorded in the worker result")
    parser.add_argument(
        "--result-file",
        help="write a machine-readable worker result for competition_runner")
    parser.add_argument(
        "--exclude-marker-id", action="append", type=int, default=[],
        help="ignore a marker already delivered or failed in this match")
    parser.add_argument(
        "--formal-mode", action="store_true",
        help="disable all fixed-layout diagnostic shortcuts")
    parser.add_argument(
        "--weights", default=str(REPO_ROOT / "examples" / "supermarket_sorting" / "perception" / "checkpoints" / "best.pt"),
        help="multi-class Ultralytics checkpoint (default: repository best.pt)")
    parser.add_argument(
        "--confidence", type=float, default=0.45)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--show", action="store_true", help="show the YOLO result window")
    parser.add_argument(
        "--max-scan-cycles", type=int, default=3)
    parser.add_argument(
        "--tcp-diagnostic-ground-truth", action="store_true")
    parser.add_argument(
        "--scan-skip-lower", action="store_true")
    parser.add_argument(
        "--no-close-recheck", action="store_true",
        help="disable close-range class verification before grasping")
    parser.add_argument(
        "--place-x", type=float, default=DELIVERY_TABLE_PLACE_WORLD[0])
    parser.add_argument(
        "--place-y", type=float, default=DELIVERY_TABLE_PLACE_WORLD[1])
    parser.add_argument(
        "--place-z", type=float, default=DELIVERY_TABLE_PLACE_WORLD[2],
        help="minimum TCP height for the pre-release approach pose")
    parser.add_argument(
        "--place-release-dwell", type=float, default=2.0,
        help="seconds the gripper stays open before retreating")
    parser.add_argument(
        "--place-retreat-dwell", type=float, default=1.0)
    parser.add_argument(
        "--backup-after-grab", type=float, default=0.20,
        help="base backup distance in metres after grasp and before delivery "
             "navigation (0 disables)")
    parser.add_argument(
        "--place-creep-distance", type=float,
        default=PLACE_CREEP_DISTANCE_M,
        help="guarded final approach distance toward the delivery table in "
             "metres (0 disables)")
    parser.add_argument(
        "--no-nav-during-scan", action="store_true",
        help="use the parent straight-line drive_to between scan stations")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if args.max_scan_cycles < 1:
        parser.error("--max-scan-cycles must be >= 1")
    if args.backup_after_grab < 0.0:
        parser.error("--backup-after-grab must be >= 0")
    if args.place_creep_distance < 0.0:
        parser.error("--place-creep-distance must be >= 0")
    if args.formal_mode and (
            args.tcp_diagnostic_ground_truth or args.scan_skip_lower):
        parser.error(
            "formal mode forbids fixed-layout ground truth and scan shortcuts")
    invalid_markers = [value for value in args.exclude_marker_id
                       if value < 0 or value > 44]
    if invalid_markers:
        parser.error(f"invalid ArUco marker ids: {invalid_markers}")
    return args


def _cv_gui_available() -> bool:
    """True if this OpenCV build supports HighGUI windows (GTK etc.)."""
    try:
        import cv2
        cv2.namedWindow("__cv_gui_probe__")
        cv2.destroyWindow("__cv_gui_probe__")
        return True
    except cv2.error:
        return False


def _write_result(path: str | None, document: dict) -> None:
    if not path:
        return
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()
    weights = str(pathlib.Path(args.weights).expanduser().resolve())
    if not pathlib.Path(weights).is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    import rclpy
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

    rclpy.init()
    nodes = []
    spin_thread = None
    controller = None
    caught_error = None
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        yolo_node = pick.KeleDetectNode(
            backend="yolo", pub_res_img=True, device=args.device,
            # Formal scans publish every detected class so the parent runner
            # can retain a cross-order kind-to-ArUco inventory.  The motion
            # controller still filters its own requested target kind.
            weights=weights,
            target_kind=None if args.formal_mode else args.target_kind,
            confidence=args.confidence, show=False,
            camera_names=("head",))
        aruco_node = pick.ArucoDetectNode(
            "head", marker_size=pick.MARKER_SIZE_M, publish_tf=False,
            publish_result_image=True)
        controller = IntegratedNavPickPlace(
            args.target_kind, args.max_scan_cycles,
            args.tcp_diagnostic_ground_truth, args.scan_skip_lower,
            place_x=args.place_x, place_y=args.place_y, place_z=args.place_z,
            place_release_dwell_s=args.place_release_dwell,
            place_retreat_dwell_s=args.place_retreat_dwell,
            nav_during_scan=not args.no_nav_during_scan,
            backup_after_grab_m=args.backup_after_grab,
            place_creep_m=args.place_creep_distance,
            close_recheck=not args.no_close_recheck)
        controller.excluded_marker_ids = set(args.exclude_marker_id)
        if controller.excluded_marker_ids:
            controller.get_logger().info(
                "excluding markers from earlier attempts: "
                f"{sorted(controller.excluded_marker_ids)}")
        nodes = [yolo_node, aruco_node, controller]
        viewer = None
        if args.show:
            if _cv_gui_available():
                viewer = pick.MainThreadResultViewer(controller)
                nodes.append(viewer)
            else:
                # Official client image ships an OpenCV without GTK/HighGUI;
                # the simulation window (server side) is still available.
                controller.get_logger().warn(
                    "OpenCV has no GUI support; skipping the YOLO window "
                    "(the server-side simulation window still shows motion)")
        for node in nodes:
            executor.add_node(node)

        if viewer is None:
            executor.spin()
        else:
            def spin_in_background():
                try:
                    executor.spin()
                except ExternalShutdownException:
                    pass

            spin_thread = threading.Thread(
                target=spin_in_background,
                name="ros2_executor", daemon=True)
            spin_thread.start()
            while rclpy.ok():
                key = viewer.show()
                if key in (ord("q"), 27):
                    controller.get_logger().info(
                        "q/Esc pressed in result window; stopping")
                    rclpy.shutdown()
                    break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:  # noqa: BLE001 - worker must report the failure
        caught_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        delivered = bool(
            controller is not None
            and controller.flow_phase == "done")
        state = None if controller is None else controller.state
        phase = None if controller is None else controller.flow_phase
        marker_id = (
            None if controller is None else controller.target_marker_id)
        error = None if delivered else (
            caught_error or f"worker stopped in phase={phase} state={state}")
        _write_result(args.result_file, {
            "schema_version": 1,
            "order_id": args.order_id,
            "kind": args.target_kind,
            "status": "delivered" if delivered else "failed",
            "marker_id": marker_id,
            "phase": phase,
            "state": state,
            "error": error,
            "elapsed_s": round(time.monotonic() - started_at, 3),
            "formal_mode": bool(args.formal_mode),
        })
        if rclpy.ok():
            rclpy.shutdown()
        try:
            executor.shutdown()
        except Exception:  # noqa: BLE001 - result is already persisted
            pass
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        for node in nodes:
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001 - best-effort worker cleanup
                pass
        if args.show:
            import cv2
            cv2.destroyAllWindows()
    return 0 if delivered else 2


if __name__ == "__main__":
    raise SystemExit(main())
