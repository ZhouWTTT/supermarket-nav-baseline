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
      -> arm extends over the table, gripper releases, arm retreats

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
    SupermarketNavigator,
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
DELIVERY_TABLE_PLACE_WORLD = (-1.80, -3.35, 0.85)  # x, y, z target above table
NAV_TRANSIT_GATE_M = 0.35          # beyond this distance, use the navigator
NAV_LASER_STALE_S = 0.50           # fail safe if the 12 Hz scan stops
NAV_STATE_STALE_S = 0.50           # odom/joints must also remain live
NAV_PROGRESS_LOG_S = 3.0

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
      "nav_to_delivery" — navigator to DELIVERY_APPROACH with goods held.
      "place"           — extend arm over the table, release, retreat.
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
            nav_during_scan: bool = True):
        super().__init__(
            target_kind, max_scan_cycles,
            tcp_diagnostic_ground_truth, scan_skip_lower)

        self.nav_during_scan = nav_during_scan
        self.place_world = np.array(
            [place_x, place_y, place_z], dtype=float)
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
        self._place_ik_attempted = False
        self._place_arm_target_sent = False
        self._place_retreat_sent = False
        self.place_creep_start_y = None
        self.place_creep_done = False
        self._flow_done_logged = False
        self._laser_warn_log = 0.0
        self._state_warn_log = 0.0

        self.get_logger().info(
            "integrated nav+pick+place ready; "
            f"nav_during_scan={nav_during_scan} "
            f"place_world={np.round(self.place_world, 3)} "
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
            "navigating to delivery table")
        self.flow_phase = "nav_to_delivery"
        self._nav_goal = None
        self._nav_last_log = 0.0
        self.nav.set_goal(*DELIVERY_APPROACH)

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

    def _compute_place_arm_joints(self) -> np.ndarray | None:
        """Solve the selected arm over the table; try several refs/targets once.

        The numeric IK depends heavily on the reference joints.  At the
        delivery pose the shelf pregrasp joints are far from any solution, so
        we also try the compact INIT pose and the measured joints.  The result
        (including a failure) is cached to avoid per-tick recomputation.
        """
        if self._place_ik_attempted:
            return self.place_arm_joints
        self._place_ik_attempted = True

        measured = self.selected_arm_positions()
        compact = np.asarray(
            PLACE_RETREAT_ARM_R if self.grasp_arm == "r"
            else PLACE_RETREAT_ARM_L, dtype=float)
        refs = [compact, measured]
        if self.pregrasp_arm_joints is not None:
            refs.append(np.asarray(self.pregrasp_arm_joints, dtype=float))

        tcp = self.selected_tcp_world()
        held_z = float(tcp[2]) if tcp is not None else 0.95
        bx, by = float(self.base_xy[0]), float(self.base_xy[1])
        # Table top is ~0.767 m and the centre is at y=-3.41.  Release targets
        # keep the wrist a little above the bottle height so the product drops
        # only a few centimetres, and prefer deeper d to centre over the table.
        z_candidates = (0.90, 0.92, 0.95)
        d_candidates = (0.80, 0.75, 0.70)
        # Top-shelf grasps pin the slide at SLIDE_MIN, which leaves the arm too
        # high to reach the table; raising the slide lowers the whole arm into
        # reach.  Middle/lower grasps keep their grasp slide.
        slide_candidates = [self.slide_grasp]
        if self.slide_grasp <= pick.SLIDE_MIN + 0.05:
            slide_candidates += [0.30, 0.35, 0.40]

        for slide in slide_candidates:
            for d in d_candidates:
                for z in z_candidates:
                    world = np.array([bx, by - d, z], dtype=float)
                    for ref in refs:
                        joints = self._solve_place_world(world, ref, slide)
                        if joints is None:
                            continue
                        self.place_world = world
                        self.place_arm_joints = joints
                        self.place_slide_cmd = slide
                        self.get_logger().info(
                            f"[place] IK target world={np.round(world, 3)} "
                            f"slide={slide:.2f} d={d:.2f}m z={z:.2f}m "
                            f"refs_tried={len(refs)}")
                        return joints

        # Last resort: keep the current slide and release at the held height
        # (the product may drop from higher than ideal, but the flow completes).
        for d in d_candidates:
            world = np.array([bx, by - d, held_z], dtype=float)
            for ref in refs:
                joints = self._solve_place_world(
                    world, ref, self.slide_grasp)
                if joints is None:
                    continue
                self.place_world = world
                self.place_arm_joints = joints
                self.place_slide_cmd = self.slide_grasp
                self.get_logger().warn(
                    f"[place] fallback IK at held height "
                    f"world={np.round(world, 3)} slide={self.slide_grasp:.2f}")
                return joints

        self.get_logger().warn(
            "[place] no IK solution over the table; will creep forward and "
            "release in the current pose")
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

    def _place_tick(self) -> None:
        now = self.now()
        if self.use_dual_tissue_grasp:
            self._place_tick_dual(now)
            return

        if self.place_stage == 0:
            # 1) creep the base a few cm toward the table so the release lands
            # near the table centre (the nav arrival pose sits close to the
            # table's north edge); 2) solve the arm over the table (once);
            # 3) as a last resort release in the held pose.
            if not self.place_creep_done:
                if self.place_creep_start_y is None:
                    self.place_creep_start_y = float(self.base_xy[1])
                front = self._laser_front_range()
                crept = float(self.place_creep_start_y - self.base_xy[1])
                if crept >= 0.07 or (front is not None and front < 0.38):
                    self.set_twist(0.0, 0.0)
                    self.place_creep_done = True
                    self.get_logger().info(
                        f"[place] creep finished (crept={crept:.3f}m "
                        f"front={front}); solving arm over the table")
                else:
                    self.set_twist(0.06, 0.0)
                    return
            if self.place_arm_joints is None:
                self.place_arm_joints = self._compute_place_arm_joints()
                if self.place_arm_joints is not None:
                    # Send once — set_selected_arm_target resets
                    # commands_ready_since, so calling it every tick would
                    # prevent the settling gate from ever passing.
                    self.set_selected_arm_target(self.place_arm_joints)
                    if self.place_slide_cmd is not None:
                        self.des_slide = self.place_slide_cmd
                    self._place_arm_target_sent = True
                elif not getattr(self, "_place_fallback_logged", False):
                    self._place_fallback_logged = True
                    self.get_logger().warn(
                        "[place] no IK solution even after creep; releasing "
                        "in the current held pose")
            if self.commands_ready(arm_tolerance=0.05, slide_tolerance=0.05):
                tcp = self.selected_tcp_world()
                self.get_logger().info(
                    f"[place] arm extended over table; tcp="
                    f"{None if tcp is None else np.round(tcp, 3)}")
                self.place_stage = 1
                self.place_t0 = now
        elif self.place_stage == 1:
            self._set_selected_grip(pick.GRIP_OPEN)
            if now - self.place_t0 >= self.place_release_dwell_s:
                self.get_logger().info(
                    "[place] gripper released; retreating arm")
                self.place_stage = 2
                self.place_t0 = now
                self._place_retreat_sent = False
        elif self.place_stage == 2:
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
                self.flow_phase = "done"
                self.get_logger().info(
                    f"[flow] PLACE COMPLETE — {self.target_kind} delivered "
                    f"to the table; base=({self.base_xy[0]:.2f},"
                    f"{self.base_xy[1]:.2f})")

    def _place_tick_dual(self, now: float) -> None:
        """Best-effort dual-arm (zhijin) place: keep the clamp, release, retreat."""
        if self.place_stage == 0:
            self.des_left_grip = pick.DUAL_TISSUE_GRIP_COMMAND
            self.des_right_grip = pick.DUAL_TISSUE_GRIP_COMMAND
            self.place_stage = 1
            self.place_t0 = now
            self.get_logger().info(
                "[place-dual] holding clamp pose; starting release sequence")
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
                self.flow_phase = "done"
                self.get_logger().info(
                    f"[flow] PLACE COMPLETE — {self.target_kind} delivered "
                    f"to the table; base=({self.base_xy[0]:.2f},"
                    f"{self.base_xy[1]:.2f})")

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
        if self.flow_phase == "nav_to_delivery":
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
        "--preferred-marker-id", type=int,
        help="prefer a confirmed inventory marker for this order")
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
        "--place-x", type=float, default=DELIVERY_TABLE_PLACE_WORLD[0])
    parser.add_argument(
        "--place-y", type=float, default=DELIVERY_TABLE_PLACE_WORLD[1])
    parser.add_argument(
        "--place-z", type=float, default=DELIVERY_TABLE_PLACE_WORLD[2])
    parser.add_argument(
        "--place-release-dwell", type=float, default=2.0,
        help="seconds the gripper stays open before retreating")
    parser.add_argument(
        "--place-retreat-dwell", type=float, default=1.0)
    parser.add_argument(
        "--no-nav-during-scan", action="store_true",
        help="use the parent straight-line drive_to between scan stations")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if args.max_scan_cycles < 1:
        parser.error("--max-scan-cycles must be >= 1")
    if args.formal_mode and (
            args.tcp_diagnostic_ground_truth or args.scan_skip_lower):
        parser.error(
            "formal mode forbids fixed-layout ground truth and scan shortcuts")
    invalid_markers = [value for value in args.exclude_marker_id
                       if value < 0 or value > 44]
    if (args.preferred_marker_id is not None
            and not 0 <= args.preferred_marker_id <= 44):
        invalid_markers.append(args.preferred_marker_id)
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
            nav_during_scan=not args.no_nav_during_scan)
        controller.excluded_marker_ids = set(args.exclude_marker_id)
        controller.preferred_marker_id = args.preferred_marker_id
        if controller.excluded_marker_ids:
            controller.get_logger().info(
                "excluding markers from earlier attempts: "
                f"{sorted(controller.excluded_marker_ids)}")
        if controller.preferred_marker_id is not None:
            controller.get_logger().info(
                f"using confirmed inventory marker: "
                f"{controller.preferred_marker_id}")
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
