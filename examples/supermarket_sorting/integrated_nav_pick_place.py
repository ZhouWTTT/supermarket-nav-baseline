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
from std_msgs.msg import Bool  # noqa: E402
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

# Five deterministic delivery slots.  The robot approaches from +Y and faces
# south, so more-negative Y is deeper on the table.  Three staggered inner
# slots are filled first, followed by two outer slots.  A literal five-item
# depth-only row would leave less than 90 mm between centres on this 440 mm
# deep table and would overlap the larger products.
DELIVERY_PLACE_SLOTS_XY = (
    (-2.20, -3.45),  # 1: deepest, inner-left
    (-1.94, -3.43),  # 2: inner-centre
    (-1.68, -3.41),  # 3: inner-right
    (-2.07, -3.29),  # 4: outer-left
    (-1.81, -3.27),  # 5: outer-right / nearest
)
PLACE_SLOT_IK_NUDGE_M = 0.020
PLACE_SLOT_XY_TOLERANCE_M = 0.060

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
PLACE_DESCENT_SLIDE_STEP_M = 0.0030
PLACE_CLEAR_TABLE_MARGIN_M = 0.060
PLACE_CLEAR_TABLE_SPEED_MPS = 0.30
PLACE_CLEAR_TABLE_TIMEOUT_S = 15.0
# Keep the fast, obstacle-aware navigator active until its 0.10 m coarse
# tolerance.  The parent controller is retained only for the final few
# centimetres needed by perception/manipulation alignment.  The old 0.35 m
# hand-off made every shelf station spend roughly 15--20 s in low-speed trim.
NAV_TRANSIT_GATE_M = 0.10
NAV_PRECISE_HANDOFF_MARGIN_M = 0.02
NAV_LASER_STALE_S = 0.50           # fail safe if the 12 Hz scan stops
NAV_STATE_STALE_S = 0.50           # odom/joints must also remain live
NAV_PROGRESS_LOG_S = 3.0

# Keep the held product clear of the shelf before delivery navigation starts
# turning the base.  The arms and product still protrude toward the shelf at
# the end of the parent grasp state machine.
BACKUP_SPEED_MPS = 0.30
BACKUP_TIMEOUT_S = 8.0
TRANSIT_SLIDE_TARGET_M = 0.006
TRANSIT_SLIDE_TOLERANCE_M = 0.010
TRANSIT_SLIDE_TIMEOUT_S = 8.0
# Gripper commands use 1.0=open and 0.0=fully closed.  Add holding preload
# only after the arm has withdrawn from the shelf, so capture stability/empty
# grasp checks remain unchanged.  This moves sandwich 0.16 -> 0.12; generic
# and dual grasps are already at the 0.0 limit.  Spheres use the gentler
# explicit 0.06 target below to avoid excessive squeeze.
TRANSPORT_GRIP_PRELOAD_COMMAND = 0.04
SPHERE_TRANSPORT_GRIP_COMMAND = 0.06

# A* stops outside the table's inflated costmap.  From that safe pose, make a
# short, slow, yaw-controlled final approach before extending the arm.  The
# physical chassis front remains clear of the table at the nominal endpoint.
PLACE_CREEP_DISTANCE_M = 0.20
PLACE_CREEP_SPEED_MPS = 0.12
PLACE_CREEP_FRONT_STOP_M = 0.30
# Preserve the successful longitudinal arm reach measured on the deepest
# slot, but do not drive the same 0.20 m for outer slots that are substantially
# closer to the aisle.  The normal configured creep remains a hard upper cap.
PLACE_BASE_TO_SLOT_LONGITUDINAL_M = 0.67
PLACE_CREEP_GOAL_TOLERANCE_M = 0.01
PLACE_CREEP_YAW_GAIN = 2.0
PLACE_CREEP_MAX_ANGULAR_RPS = 0.30
PLACE_RELEASE_TABLE_MARGIN_M = 0.04
PLACE_APPROACH_SOFT_DWELL_S = 2.0
PLACE_APPROACH_SOFT_ARM_TOLERANCE_RAD = 0.08
PLACE_APPROACH_SOFT_SLIDE_TOLERANCE_M = 0.05
PLACE_APPROACH_HARD_TIMEOUT_S = 15.0
PLACE_APPROACH_PROGRESS_LOG_S = 2.0
# The table-clear state already verifies the arms and chassis clearance.  One
# short zero-command interval is sufficient before worker shutdown; the old
# unconditional 3 s dwell accumulated once per order without adding safety.
FLOW_DONE_SETTLE_S = 0.25

# Fixed initial arm posture from ``supermarket_sorting_client.INIT_ARM``.
# Both arms return here after every release; restoring only the selected arm
# can preserve a stale pose inherited from an earlier failed order.
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
      "restore_height"  — restore the lift to its startup height.
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
            place_slot: int | None = None,
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

        # Wall-clock phase telemetry is intentionally observational only.  It
        # makes the next formal run actionable without changing any motion,
        # perception, or safety decision.
        self._pick_state_started_at = time.monotonic()
        self._pick_state_elapsed_s: dict[str, float] = {}
        self._flow_phase_started_at = time.monotonic()
        self._flow_phase_elapsed_s: dict[str, float] = {}
        self._flow_phase_distance_m: dict[str, float] = {}
        self._telemetry_last_base_xy = None

        self.nav_during_scan = nav_during_scan
        self.backup_after_grab_m = float(backup_after_grab_m)
        self.place_creep_m = float(place_creep_m)
        self.place_slot = place_slot
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
        self.perception_enable_pub = self.create_publisher(
            Bool, "/supermarket_sorting/perception_enable", 10)
        self.manage_external_perception = False
        self.local_perception_nodes = ()
        self._perception_requested = None
        self._perception_request_last_at = float("-inf")

        # ── baseline navigator (same interface as the demo) ──
        self.nav = SupermarketNavigator()
        self.get_logger().info(
            "path_memory="
            + json.dumps(self.nav.path_memory_status(), ensure_ascii=False)
        )

        # ── our flow state ──
        self.flow_phase = "grab"
        self._nav_goal = None
        self._nav_last_log = 0.0
        self._last_nav_reason = None
        self._nav_memory_logged = False
        self.place_stage = 0
        self.place_t0 = 0.0
        self.place_arm_joints = None
        self.place_slide_cmd = None
        self.place_release_world = None
        self.place_release_slide_cmd = None
        self._place_ik_attempted = False
        self._place_arm_target_sent = False
        self._place_descent_sent = False
        self._place_retreat_sent = False
        self._dual_descent_sent = False
        self._dual_place_target_sent = False
        self.dual_release_slide_cmd = None
        self.place_creep_start_y = None
        self.place_creep_done = False
        self._place_stage0_wait_log = 0.0
        self._backup_start_xy = None
        self._backup_start_yaw = 0.0
        self._backup_t0 = 0.0
        self._backup_logged = False
        self._height_restore_t0 = 0.0
        self._height_restore_timeout_logged = False
        self._transport_grip_command = None
        self._flow_done_logged = False
        self._table_escape_logged = False
        self._laser_warn_log = 0.0
        self._state_warn_log = 0.0

        self.get_logger().info(
            "integrated nav+pick+place ready; "
            f"nav_during_scan={nav_during_scan} "
            f"close_recheck={int(close_recheck)} "
            f"place_slot={None if place_slot is None else place_slot + 1} "
            f"place_world={np.round(self.place_world, 3)} "
            f"backup_after_grab={self.backup_after_grab_m:.2f}m "
            f"place_creep={self.place_creep_m:.2f}m "
            f"release_dwell={place_release_dwell_s}s "
            f"retreat_dwell={place_retreat_dwell_s}s")

    def set_state(self, new_state: str) -> None:
        """Accumulate parent pick-state wall time without altering its FSM."""
        previous = getattr(self, "state", None)
        started = getattr(self, "_pick_state_started_at", None)
        now = time.monotonic()
        if previous is not None and started is not None and previous != new_state:
            self._pick_state_elapsed_s[previous] = (
                self._pick_state_elapsed_s.get(previous, 0.0)
                + max(0.0, now - started))
        super().set_state(new_state)
        if previous != self.state and started is not None:
            self._pick_state_started_at = now

    def _set_flow_phase(self, new_phase: str) -> None:
        """Change the outer phase and account for elapsed wall-clock time."""
        if new_phase == self.flow_phase:
            return
        now = time.monotonic()
        previous = self.flow_phase
        self._flow_phase_elapsed_s[previous] = (
            self._flow_phase_elapsed_s.get(previous, 0.0)
            + max(0.0, now - self._flow_phase_started_at))
        self.flow_phase = new_phase
        self._flow_phase_started_at = now

    def timing_snapshot(self) -> dict[str, dict[str, float]]:
        """Return accumulated timings including the currently active states."""
        now = time.monotonic()
        pick_states = dict(self._pick_state_elapsed_s)
        pick_states[self.state] = (
            pick_states.get(self.state, 0.0)
            + max(0.0, now - self._pick_state_started_at))
        flow_phases = dict(self._flow_phase_elapsed_s)
        flow_phases[self.flow_phase] = (
            flow_phases.get(self.flow_phase, 0.0)
            + max(0.0, now - self._flow_phase_started_at))
        return {
            "pick_state_elapsed_s": {
                key: round(value, 3)
                for key, value in sorted(pick_states.items())
            },
            "flow_phase_elapsed_s": {
                key: round(value, 3)
                for key, value in sorted(flow_phases.items())
            },
            "flow_phase_distance_m": {
                key: round(value, 3)
                for key, value in sorted(
                    self._flow_phase_distance_m.items())
            },
        }

    def _record_motion_telemetry(self) -> None:
        """Accumulate measured travel per phase without affecting control."""
        current = np.asarray(self.base_xy, dtype=float).copy()
        previous = self._telemetry_last_base_xy
        self._telemetry_last_base_xy = current
        if previous is None:
            return
        distance = float(np.linalg.norm(current - previous))
        # Ignore an odometry reset/teleport across server restarts; normal
        # 50 Hz motion is orders of magnitude below this threshold.
        if not math.isfinite(distance) or distance > 0.50:
            return
        self._flow_phase_distance_m[self.flow_phase] = (
            self._flow_phase_distance_m.get(self.flow_phase, 0.0)
            + distance)

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

    def configure_external_perception(self, enabled: bool) -> None:
        """Let this worker gate the shared detector around scan states."""
        self.manage_external_perception = bool(enabled)
        self._perception_requested = None
        self._perception_request_last_at = float("-inf")
        self._publish_perception_request(False, force=True)

    def configure_local_perception(self, *nodes) -> None:
        """Apply the same duty cycle when persistent perception is absent."""
        self.local_perception_nodes = tuple(nodes)
        self._perception_requested = None
        self._perception_request_last_at = float("-inf")
        self._publish_perception_request(False, force=True)

    def _publish_perception_request(
            self, enabled: bool, force: bool = False) -> None:
        if (not self.manage_external_perception
                and not self.local_perception_nodes):
            return
        enabled = bool(enabled)
        now = self.now()
        if (not force
                and enabled == self._perception_requested
                and now - self._perception_request_last_at < 0.5):
            return
        if self.manage_external_perception:
            self.perception_enable_pub.publish(Bool(data=enabled))
        for node in self.local_perception_nodes:
            node.set_enabled(enabled)
        self._perception_requested = enabled
        self._perception_request_last_at = now

    def initialize_commands(self) -> None:
        """Initialize commands while keeping a fixed post-grasp transit height."""
        super().initialize_commands()
        measured_slide = self.joints.get("slide_joint")
        self.get_logger().info(
            f"[flow] configured post-grasp transit slide="
            f"{TRANSIT_SLIDE_TARGET_M:.3f} "
            f"(initial measured={measured_slide})")

    @staticmethod
    def _transit_slide_target() -> float:
        return float(TRANSIT_SLIDE_TARGET_M)

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
                                   position_tolerance
                                   + NAV_PRECISE_HANDOFF_MARGIN_M)):
            now = self.now()
            goal = (float(target[0]), float(target[1]), float(final_yaw))
            if self._nav_goal != goal:
                self._nav_goal = goal
                self.nav.set_goal(*goal)
                self._nav_last_log = 0.0
                self._nav_memory_logged = False
                self.get_logger().info(
                    "[nav] new_goal="
                    + json.dumps(
                        {
                            "goal": [round(goal[0], 3), round(goal[1], 3), round(goal[2], 3)],
                            "path_memory": self.nav.path_memory_status(),
                        },
                        ensure_ascii=False,
                    )
                )

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
            if not self._nav_memory_logged:
                self._nav_memory_logged = True
                self.get_logger().info(
                    "[nav] path_memory_runtime="
                    + json.dumps(self.nav.path_memory_status(), ensure_ascii=False)
                )

            ctrl = self.nav.controller
            if (ctrl.stop_reason is not None
                    and ctrl.stop_reason != self._last_nav_reason):
                self._last_nav_reason = ctrl.stop_reason
                self.get_logger().info(
                    f"[nav] stop_reason={ctrl.stop_reason} "
                    f"lidar={ctrl.lidar_clearance:.2f}m "
                    f"rear={ctrl.rear_clearance:.2f}m "
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
            # Discard the transit command before the centimetre-scale parent
            # controller takes over; otherwise its second velocity ramp can
            # carry momentum through the hand-off and create an avoidable
            # overshoot/reverse correction cycle.
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0

        arrived = super().drive_to(
            target_xy, final_yaw, position_tolerance)
        if arrived:
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0
        return arrived

    # ------------------------------------------------------------------
    # flow hooks
    # ------------------------------------------------------------------
    def _on_grab_complete(self) -> None:
        # The parent reaches DONE only after horizontal withdrawal is complete.
        # Increase holding preload now, then restore during straight backup;
        # rotation waits for verified height feedback.
        self._capture_transport_grip_command()
        self.des_slide = self._transit_slide_target()
        self.get_logger().info(
            f"[flow] goods grabbed (marker={self.target_marker_id}, "
            f"kind={self.target_kind}, state={self.state}); "
            "preparing delivery transit")
        if self.backup_after_grab_m > 1e-4:
            self._set_flow_phase("backup")
            self._backup_start_xy = self.base_xy.copy()
            self._backup_start_yaw = float(self.base_yaw)
            self._backup_t0 = self.now()
            self._backup_logged = False
            self.get_logger().info(
                f"[flow] backing up {self.backup_after_grab_m:.2f}m "
                "before delivery rotation")
            return
        self._start_height_restore()

    def _start_height_restore(self) -> None:
        self._set_flow_phase("restore_height")
        self._height_restore_t0 = self.now()
        self._height_restore_timeout_logged = False
        self.des_slide = self._transit_slide_target()
        self.set_twist(0.0, 0.0)
        self.get_logger().info(
            f"[flow] restoring post-grasp transit slide: measured="
            f"{self.joints.get('slide_joint')} "
            f"target={self.des_slide:.3f}")

    def _restore_height_tick(self) -> None:
        now = self.now()
        target_slide = self._transit_slide_target()
        self.set_twist(0.0, 0.0)
        self.des_slide = target_slide
        measured_slide = self.joints.get("slide_joint")
        if measured_slide is not None and math.isfinite(float(measured_slide)):
            error = abs(float(measured_slide) - target_slide)
            if error <= TRANSIT_SLIDE_TOLERANCE_M:
                self.get_logger().info(
                    f"[flow] post-grasp transit slide restored: measured="
                    f"{float(measured_slide):.3f} error={error:.3f}m; "
                    "starting delivery navigation")
                self._start_delivery_navigation()
                return
        if (not self._height_restore_timeout_logged
                and now - self._height_restore_t0
                >= TRANSIT_SLIDE_TIMEOUT_S):
            self._height_restore_timeout_logged = True
            self.get_logger().warn(
                f"[flow] post-grasp transit slide restore timed out after "
                f"{TRANSIT_SLIDE_TIMEOUT_S:.1f}s "
                f"(measured={measured_slide}, target={target_slide:.3f}); "
                "remaining stopped and holding the target")

    def _start_delivery_navigation(self) -> None:
        self._set_flow_phase("nav_to_delivery")
        self.des_slide = self._transit_slide_target()
        self._nav_goal = None
        self._nav_last_log = 0.0
        self._nav_memory_logged = False
        # Align the chassis with the assigned table slot before the guarded
        # southward creep.  This keeps the arm motion predominantly forward
        # even for the left/right slots and makes the fixed world target much
        # more likely to have a nearby IK solution.
        delivery_goal = (
            float(self.place_world[0]),
            DELIVERY_APPROACH[1],
            DELIVERY_APPROACH[2],
        )
        # Each table slot has a distinct approach X.  Reuse an exact cached
        # route for the same slot, but reject a nearby route whose old goal is
        # laterally displaced enough to steer into a now-blocked corridor.
        self.nav.set_goal(
            *delivery_goal, cached_goal_offset_limit=0.08)
        self.get_logger().info(
            f"[nav→delivery] assigned slot="
            f"{None if self.place_slot is None else self.place_slot + 1} "
            f"target={np.round(self.place_world[:2], 3)} "
            f"approach={np.round(delivery_goal[:2], 3)}")

    def _backup_tick(self) -> None:
        """Reverse along the grasp heading while holding the current yaw."""
        now = self.now()
        self.des_slide = self._transit_slide_target()
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
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0
            message = (
                f"[flow] backup finished (moved={moved_back:.3f}m, "
                f"elapsed={elapsed:.1f}s); verifying transit height")
            if timed_out and not reached:
                self.get_logger().warn(message + " after timeout")
            else:
                self.get_logger().info(message)
            self._start_height_restore()
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
        self.des_slide = self._transit_slide_target()
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
        if not self._nav_memory_logged:
            self._nav_memory_logged = True
            self.get_logger().info(
                "[nav→delivery] path_memory_runtime="
                + json.dumps(self.nav.path_memory_status(), ensure_ascii=False)
            )

        ctrl = self.nav.controller
        if (ctrl.stop_reason is not None
                and ctrl.stop_reason != self._last_nav_reason):
            self._last_nav_reason = ctrl.stop_reason
            self.get_logger().info(
                f"[nav→delivery] stop_reason={ctrl.stop_reason} "
                f"lidar={ctrl.lidar_clearance:.2f}m "
                f"rear={ctrl.rear_clearance:.2f}m")

        if now - self._nav_last_log >= NAV_PROGRESS_LOG_S:
            self._nav_last_log = now
            self.get_logger().info(
                f"[nav→delivery] pos=({self.base_xy[0]:.2f},"
                f"{self.base_xy[1]:.2f}) "
                f"yaw={math.degrees(self.base_yaw):.0f}° "
                f"v={v:.2f} w={w:.2f} reached={reached}")

        if reached:
            # The navigator has completed its own acceleration profile.  Do
            # not let the parent command filter carry residual translation
            # into the table-facing yaw refinement or placement phase.
            self.cmd_linear = 0.0
            # Navigator yaw tolerance is 0.15 rad; refine to face south.
            yaw_err = pick.wrap_to_pi(-math.pi / 2.0 - self.base_yaw)
            if abs(yaw_err) > 0.03:
                self.set_twist(0.0, 2.0 * yaw_err)
                return
            self.set_twist(0.0, 0.0)
            self.cmd_angular = 0.0
            self._set_flow_phase("place")
            self.place_stage = 0
            self.place_t0 = now
            self.get_logger().info(
                f"[flow] arrived at delivery approach "
                f"pos=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
                f"yaw={math.degrees(self.base_yaw):.0f}°; placing with "
                f"grip_command={self._transport_grip_command} "
                f"measured_grip={self.selected_gripper_position()}")

    def _set_selected_grip(self, value: float) -> None:
        if self.grasp_arm == "r":
            self.des_right_grip = float(value)
        else:
            self.des_left_grip = float(value)

    def _capture_transport_grip_command(self) -> None:
        """Strengthen and remember the closed command after shelf retreat."""
        if self.use_dual_tissue_grasp:
            self._transport_grip_command = float(
                pick.DUAL_TISSUE_GRIP_COMMAND)
            self.des_left_grip = self._transport_grip_command
            self.des_right_grip = self._transport_grip_command
            self.get_logger().info(
                f"[grip-hold] dual transport command="
                f"{self._transport_grip_command:.3f} (position limit)")
            return
        grasp_command = float(
            self.des_right_grip
            if self.grasp_arm == "r" else self.des_left_grip)
        if self.use_sphere_grasp:
            self._transport_grip_command = SPHERE_TRANSPORT_GRIP_COMMAND
        else:
            self._transport_grip_command = float(np.clip(
                grasp_command - TRANSPORT_GRIP_PRELOAD_COMMAND,
                0.0, pick.GRIP_OPEN))
        self._set_selected_grip(self._transport_grip_command)
        self.get_logger().info(
            f"[grip-hold] arm={self.grasp_arm} "
            f"capture_command={grasp_command:.3f} -> "
            f"transport_command={self._transport_grip_command:.3f} "
            f"measured={self.selected_gripper_position()}")

    def _hold_grasp_during_transport(self) -> None:
        """Reassert the closed command until the verified release stage."""
        if self._transport_grip_command is None:
            return
        if self.use_dual_tissue_grasp:
            self.des_left_grip = self._transport_grip_command
            self.des_right_grip = self._transport_grip_command
        else:
            self._set_selected_grip(self._transport_grip_command)

    def _command_initial_arm_posture(self) -> None:
        """Open both grippers and return both arms to the fixed initial pose."""
        self.des_left_arm = np.asarray(PLACE_RETREAT_ARM_L, dtype=float)
        self.des_right_arm = np.asarray(PLACE_RETREAT_ARM_R, dtype=float)
        self.des_left_grip = pick.GRIP_OPEN
        self.des_right_grip = pick.GRIP_OPEN
        self.des_slide = pick.SLIDE_REFERENCE_COMMAND

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

    def _compute_place_arm_joints(self) -> np.ndarray | None:
        """Solve an approach pose with enough slide travel for a low release.

        The numeric IK depends heavily on the reference joints.  At the
        delivery pose the shelf pregrasp joints are far from any solution, so
        we also try the compact INIT pose and the measured joints.  Once an
        approach pose is found, the final vertical descent keeps the arm joints
        fixed and increases the downward-facing slide joint.  The result
        (including failure) is cached to avoid per-tick recomputation.
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

        # The formal runner assigns an absolute table slot.  Manual runs use
        # --place-x/--place-y in exactly the same way.  Small bounded nudges
        # are IK fallbacks only; they cannot move a product into another slot.
        target_x = float(self.place_world[0])
        target_y = float(self.place_world[1])
        xy_candidates = (
            (target_x, target_y),
            (target_x, target_y + PLACE_SLOT_IK_NUDGE_M),
            (target_x, target_y - PLACE_SLOT_IK_NUDGE_M),
            (target_x + PLACE_SLOT_IK_NUDGE_M, target_y),
            (target_x - PLACE_SLOT_IK_NUDGE_M, target_y),
        )
        release_z = self._product_release_z()
        minimum_approach_z = max(
            self.place_min_approach_z,
            release_z + PLACE_APPROACH_CLEARANCE_M)
        z_candidates = tuple(
            minimum_approach_z + offset for offset in (0.0, 0.02, 0.04))
        # Top-shelf grasps pin the slide at SLIDE_MIN, which leaves the arm too
        # high to reach the table; raising the slide lowers the whole arm into
        # reach.  Middle/lower grasps keep their grasp slide.
        slide_candidates = []
        for slide in (self.slide_grasp, 0.20, 0.30, 0.35, 0.40, 0.45):
            slide = float(np.clip(slide, pick.SLIDE_MIN, pick.SLIDE_MAX))
            if not any(abs(slide - item) < 1e-6
                       for item in slide_candidates):
                slide_candidates.append(slide)

        for x, y in xy_candidates:
            for z in z_candidates:
                descent = z - release_z
                for slide in slide_candidates:
                    release_slide = slide + descent
                    if release_slide > pick.SLIDE_MAX + 1e-6:
                        continue
                    world = np.array([x, y, z], dtype=float)
                    for ref in refs:
                        joints = self._solve_place_world(world, ref, slide)
                        if joints is None:
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
                            f"descent={descent:.3f}m "
                            f"slot={None if self.place_slot is None else self.place_slot + 1} "
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
        slot_base_goal_y = float(
            self.place_world[1] + PLACE_BASE_TO_SLOT_LONGITUDINAL_M)
        slot_reached = (
            float(self.base_xy[1])
            <= slot_base_goal_y + PLACE_CREEP_GOAL_TOLERANCE_M)
        distance_reached = crept >= self.place_creep_m
        front_reached = (
            front is not None and front <= PLACE_CREEP_FRONT_STOP_M)
        if (self.place_creep_m <= 1e-4 or slot_reached
                or distance_reached or front_reached):
            self.set_twist(0.0, 0.0)
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0
            self.place_creep_done = True
            reason = (
                "slot_depth" if slot_reached else
                "distance_cap" if distance_reached else
                "lidar" if front_reached else "disabled")
            self.get_logger().info(
                f"[place] final approach finished (reason={reason} "
                f"crept={crept:.3f}m front={front} "
                f"slot_base_goal_y={slot_base_goal_y:.3f} "
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

    def _tcp_at_assigned_slot(self, tcp: np.ndarray | None) -> bool:
        """Require the measured release XY to remain near its own slot."""
        if tcp is None or np.asarray(tcp).shape != (3,):
            return False
        error = np.asarray(tcp[:2], dtype=float) - self.place_world[:2]
        return bool(
            np.all(np.isfinite(error))
            and np.linalg.norm(error) <= PLACE_SLOT_XY_TOLERANCE_M)

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
                self.place_arm_joints = self._compute_place_arm_joints()
                if self.place_arm_joints is not None:
                    # Send once — set_selected_arm_target resets
                    # commands_ready_since, so calling it every tick would
                    # prevent the settling gate from ever passing.
                    self.set_selected_arm_target(self.place_arm_joints)
                    if self.place_slide_cmd is not None:
                        self.des_slide = self.place_slide_cmd
                    self.place_t0 = now
                    self._place_stage0_wait_log = 0.0
                    self._place_arm_target_sent = True
                else:
                    raise RuntimeError(
                        "place IK failed; refusing to release goods off-table")
            arm_error = self.selected_arm_error()
            measured_slide = self.joints.get("slide_joint")
            slide_error = (
                float("inf") if measured_slide is None
                else abs(float(measured_slide) - self.des_slide))
            approach_elapsed = (
                0.0 if not self._place_arm_target_sent
                else now - self.place_t0)
            converged = self.commands_ready(
                arm_tolerance=0.05, slide_tolerance=0.05)
            soft_ready = (
                self._place_arm_target_sent
                and approach_elapsed >= PLACE_APPROACH_SOFT_DWELL_S
                and arm_error <= PLACE_APPROACH_SOFT_ARM_TOLERANCE_RAD
                and slide_error <= PLACE_APPROACH_SOFT_SLIDE_TOLERANCE_M)
            if converged or soft_ready:
                gate = "converged" if converged else "soft"
                tcp = self.selected_tcp_world()
                if not self._tcp_over_delivery_table(tcp):
                    raise RuntimeError(
                        "measured place TCP is outside delivery tabletop: "
                        f"{None if tcp is None else np.round(tcp, 3)}")
                if not self._tcp_at_assigned_slot(tcp):
                    raise RuntimeError(
                        "measured place TCP missed assigned slot: "
                        f"tcp={None if tcp is None else np.round(tcp, 3)} "
                        f"slot={np.round(self.place_world[:2], 3)}")
                self.get_logger().info(
                    f"[place] arm at approach pose gate={gate} "
                    f"elapsed={approach_elapsed:.2f}s "
                    f"arm_error={arm_error:.4f}rad "
                    f"slide_error={slide_error:.4f}m tcp="
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
            elif (self._place_arm_target_sent
                    and now - self._place_stage0_wait_log
                    >= PLACE_APPROACH_PROGRESS_LOG_S):
                self._place_stage0_wait_log = now
                self.get_logger().info(
                    f"[place] waiting for approach pose "
                    f"elapsed={approach_elapsed:.2f}s "
                    f"arm_error={arm_error:.4f}rad "
                    f"slide_error={slide_error:.4f}m")
            if (self._place_arm_target_sent
                    and approach_elapsed >= PLACE_APPROACH_HARD_TIMEOUT_S
                    and not (converged or soft_ready)):
                raise RuntimeError(
                    "[place] approach pose did not settle within "
                    f"{PLACE_APPROACH_HARD_TIMEOUT_S:.0f}s "
                    f"(arm_error={arm_error:.4f}rad "
                    f"slide_error={slide_error:.4f}m)")
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
            if not self._tcp_at_assigned_slot(tcp):
                raise RuntimeError(
                    "measured lowered TCP missed assigned slot: "
                    f"tcp={None if tcp is None else np.round(tcp, 3)} "
                    f"slot={np.round(self.place_world[:2], 3)}")
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
                    "[place] gripper released; restoring both arms while "
                    "backing away from the table")
                self._command_initial_arm_posture()
                self.place_stage = 4
                self.place_t0 = now
                self._place_retreat_sent = False

    def _configure_dual_place_target(self) -> bool:
        """Translate the clamped tissue centre to the assigned slot."""
        left_tcp = self.arm_tcp_world("left")
        right_tcp = self.arm_tcp_world("right")
        left_reference = self.arm_positions("left")
        right_reference = self.arm_positions("right")
        measured_slide = self.joints.get("slide_joint")
        if (left_tcp is None or right_tcp is None
                or left_reference is None or right_reference is None
                or measured_slide is None):
            return False

        centre = 0.5 * (np.asarray(left_tcp) + np.asarray(right_tcp))
        offset_xy = self.place_world[:2] - centre[:2]
        left_goal = np.asarray(left_tcp, dtype=float).copy()
        right_goal = np.asarray(right_tcp, dtype=float).copy()
        left_goal[:2] += offset_xy
        right_goal[:2] += offset_xy

        left_target = np.eye(4)
        right_target = np.eye(4)
        left_target[:3, 3] = self.world_to_footprint(left_goal)
        right_target[:3, 3] = self.world_to_footprint(right_goal)
        slide = float(measured_slide)
        reference = np.concatenate((
            [slide],
            np.asarray(left_reference, dtype=float),
            np.asarray(right_reference, dtype=float),
        ))
        try:
            solutions = self.kdl.inverse_kinematics(
                T_left=left_target,
                T_right=right_target,
                target_height=slide,
                ref_pos=reference)
        except Exception as exc:  # noqa: BLE001 - report a safe place failure
            self.get_logger().error(
                f"[place-dual] assigned-slot IK raised: {exc}")
            return False
        if solutions is None or len(solutions) == 0:
            self.get_logger().error(
                "[place-dual] no IK solution for assigned slot "
                f"{np.round(self.place_world[:2], 3)}")
            return False

        candidates = [
            np.asarray(item[1:], dtype=float) for item in solutions]
        arms_reference = reference[1:]
        best = min(
            candidates,
            key=lambda item: float(np.max(
                np.abs(item - arms_reference))))
        self.des_left_arm = best[:6].copy()
        self.des_right_arm = best[6:].copy()
        self.commands_ready_since = None
        self._dual_place_target_sent = True
        self.place_t0 = self.now()
        self.get_logger().info(
            f"[place-dual] moving clamped tissue to slot="
            f"{None if self.place_slot is None else self.place_slot + 1} "
            f"centre={np.round(centre, 3)} "
            f"target={np.round(self.place_world[:2], 3)} "
            f"offset={np.round(offset_xy, 3)}")
        return True

    def _place_tick_dual(self, now: float) -> None:
        """Dual-arm tissue place at its slot, then descend vertically."""
        if self.place_stage == 0:
            self.des_left_grip = pick.DUAL_TISSUE_GRIP_COMMAND
            self.des_right_grip = pick.DUAL_TISSUE_GRIP_COMMAND
            if not self._advance_place_creep():
                return
            if not self._dual_place_target_sent:
                if not self._configure_dual_place_target():
                    raise RuntimeError(
                        "dual-arm IK failed for assigned delivery slot")
                return
            if not self._dual_descent_sent:
                if not self.dual_commands_ready(
                        arm_tolerance=0.05, slide_tolerance=0.025):
                    if now - self.place_t0 >= PLACE_APPROACH_HARD_TIMEOUT_S:
                        raise RuntimeError(
                            "dual-arm assigned-slot approach did not settle "
                            f"within {PLACE_APPROACH_HARD_TIMEOUT_S:.0f}s")
                    return
                release_world = self._dual_release_world()
                if (not self._tcp_over_delivery_table(release_world)
                        or not self._tcp_at_assigned_slot(release_world)):
                    raise RuntimeError(
                        "dual-arm centre missed assigned delivery slot: "
                        f"centre={None if release_world is None else np.round(release_world, 3)} "
                        f"slot={np.round(self.place_world[:2], 3)}")
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
                    or not self._tcp_at_assigned_slot(release_world)
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
                self._command_initial_arm_posture()
                self.place_stage = 4
                self.place_t0 = now

    def _clear_delivery_table_tick(self, now: float) -> None:
        """Back away after release so the next order can safely turn."""
        # Keep both arms in the initial posture throughout the base retreat,
        # independent of which arm performed the grasp.
        self._command_initial_arm_posture()

        clearance = point_to_rect_clearance(
            float(self.base_xy[0]), float(self.base_xy[1]),
            DELIVERY_TABLE_COSTMAP_BOUNDS)
        required = WHOLE_BODY_KEEP_OUT_RADIUS + PLACE_CLEAR_TABLE_MARGIN_M
        if clearance >= required:
            self.set_twist(0.0, 0.0)
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0
            if (now - self.place_t0 < self.place_retreat_dwell_s
                    or not self.dual_commands_ready(
                        arm_tolerance=0.08,
                        slide_tolerance=0.05)):
                return
            self._set_flow_phase("done")
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

        # NavigationController already applies acceleration ramps during
        # normal motion.  Do not apply the parent's second ramp in the unsafe
        # direction when the navigator has explicitly requested a stop: at
        # 0.90 m/s, a 0.03-per-tick decay could otherwise preserve forward
        # motion for roughly 0.6 s after a lidar/trajectory stop.  Angular
        # motion is cancelled only for reasons that require the complete base
        # to hold; obstacle stops may still rotate in place to find a route.
        nav_reason = self.nav.controller.stop_reason
        if (self.flow_phase in {"grab", "nav_to_delivery"}
                and abs(self.des_linear) <= 1e-9
                and nav_reason is not None):
            self.cmd_linear = 0.0
        full_hold = (
            nav_reason == "table_keepout"
            or nav_reason == "rotation_loop"
            or nav_reason in {
                "reverse_recovery_start", "lateral_escape_replan"
            }
            or (isinstance(nav_reason, str)
                and (nav_reason.startswith("no_path")
                     or nav_reason.startswith("stuck_no_path"))))
        if (self.flow_phase in {"grab", "nav_to_delivery"}
                and abs(self.des_angular) <= 1e-9
                and full_hold):
            self.cmd_angular = 0.0

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
            self._publish_perception_request(False)
            return
        self._record_motion_telemetry()
        now = self.now()
        odom_stale = (
            self.last_odom_time is None
            or now - self.last_odom_time > NAV_STATE_STALE_S)
        joints_stale = (
            self.last_joint_time is None
            or now - self.last_joint_time > NAV_STATE_STALE_S)
        laser_stale = self._laser_stale(now)
        stable_perception_state = (
            self.state == pick.STATE_SCAN
            or (self.state in {pick.STATE_REVISIT, pick.STATE_RECHECK}
                and self.scan_camera_ready_since is not None))
        perception_needed = (
            not (odom_stale or joints_stale or laser_stale)
            and self.flow_phase == "grab"
            and stable_perception_state)
        self._publish_perception_request(perception_needed)
        if odom_stale or joints_stale or laser_stale:
            # The direct zero command below bypasses normal smoothing.  Keep
            # the internal command state consistent as well so feedback
            # recovery cannot resume from a stale pre-stop velocity.
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0
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
        # A release command is legal only after the corresponding placement
        # controller has verified the assigned slot and low release pose.
        # Reassert the captured holding command through base motion and arm
        # positioning so no stale/default command can loosen the gripper while
        # the loaded robot turns or reaches over the table.
        single_place_hold = (
            self.flow_phase == "place"
            and not self.use_dual_tissue_grasp
            and self.place_stage in {0, 1})
        dual_place_hold = (
            self.flow_phase == "place"
            and self.use_dual_tissue_grasp
            and self.place_stage == 0)
        if (self.flow_phase in {
                "backup", "restore_height", "nav_to_delivery"}
                or single_place_hold or dual_place_hold):
            self._hold_grasp_during_transport()
        if self.flow_phase == "backup":
            self._backup_tick()
        elif self.flow_phase == "restore_height":
            self._restore_height_tick()
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
                and self.now() - self.place_t0 > FLOW_DONE_SETTLE_S):
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
        "--candidate-kind", action="append", default=[],
        choices=sorted(pick.PRODUCT_CENTER_ABOVE_MARKER_M),
        help="another pending order class that may become this trip's sole "
             "target after repeated scan detections")
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
        "--external-perception", action="store_true",
        help="consume the persistent runner-owned YOLO/ArUco topics instead "
             "of loading another detector model in this worker")
    parser.add_argument(
        "--weights", default=str(REPO_ROOT / "examples" / "supermarket_sorting" / "perception" / "checkpoints" / "best.pt"),
        help="multi-class Ultralytics checkpoint (default: repository best.pt)")
    parser.add_argument(
        "--confidence", type=float, default=0.45)
    parser.add_argument(
        "--max-inference-hz", type=float, default=12.0,
        help="maximum YOLO source-frame rate during active scan states")
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
        "--place-slot", type=int,
        choices=range(len(DELIVERY_PLACE_SLOTS_XY)),
        help="zero-based deterministic delivery slot; overrides "
             "--place-x/--place-y")
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
    parser.add_argument(
        "--scan-start-west", action="store_true",
        help="scan from the westmost shelf (A) first; used for orders after "
             "the first in a match")
    parser.add_argument(
        "--scan-start-x", type=float,
        help="measured product world X from cross-order inventory; chooses "
             "the nearest first scan station without bypassing perception")
    parser.add_argument(
        "--scan-marker-z", type=float,
        help="measured shelf-marker Z paired with --scan-start-x; prioritises "
             "that camera level with automatic full-scan fallback")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if not 0.0 < args.max_inference_hz < float("inf"):
        parser.error("--max-inference-hz must be finite and positive")
    if args.max_scan_cycles < 1:
        parser.error("--max-scan-cycles must be >= 1")
    if args.backup_after_grab < 0.0:
        parser.error("--backup-after-grab must be >= 0")
    if args.place_creep_distance < 0.0:
        parser.error("--place-creep-distance must be >= 0")
    if args.scan_start_x is not None and not math.isfinite(args.scan_start_x):
        parser.error("--scan-start-x must be finite")
    if args.scan_marker_z is not None and not math.isfinite(args.scan_marker_z):
        parser.error("--scan-marker-z must be finite")
    if args.scan_marker_z is not None and args.scan_start_x is None:
        parser.error("--scan-marker-z requires --scan-start-x")
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
    from run_log import start_run_log
    start_run_log("worker")
    args = parse_args()
    started_at = time.monotonic()
    place_x, place_y = args.place_x, args.place_y
    if args.place_slot is not None:
        place_x, place_y = DELIVERY_PLACE_SLOTS_XY[args.place_slot]
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
        controller = IntegratedNavPickPlace(
            args.target_kind, args.max_scan_cycles,
            args.tcp_diagnostic_ground_truth, args.scan_skip_lower,
            place_x=place_x, place_y=place_y, place_z=args.place_z,
            place_slot=args.place_slot,
            place_release_dwell_s=args.place_release_dwell,
            place_retreat_dwell_s=args.place_retreat_dwell,
            nav_during_scan=not args.no_nav_during_scan,
            backup_after_grab_m=args.backup_after_grab,
            place_creep_m=args.place_creep_distance,
            close_recheck=not args.no_close_recheck)
        controller.configure_external_perception(args.external_perception)
        controller.configure_opportunistic_targets(args.candidate_kind)
        controller.scan_prefer_west_start = args.scan_start_west
        if args.scan_start_x is not None:
            controller.configure_inventory_scan_hint(
                args.scan_start_x, args.scan_marker_z)
        controller.excluded_marker_ids = set(args.exclude_marker_id)
        if controller.excluded_marker_ids:
            controller.get_logger().info(
                "excluding markers from earlier attempts: "
                f"{sorted(controller.excluded_marker_ids)}")
        nodes = [controller]
        if args.external_perception:
            controller.get_logger().info(
                "using runner-owned persistent YOLO/ArUco perception")
        else:
            yolo_node = pick.KeleDetectNode(
                backend="yolo", pub_res_img=args.show, device=args.device,
                # Multi-order scans publish every detected class so the
                # controller can select one visible pending order.
                weights=weights,
                target_kind=(
                    None if args.formal_mode or args.candidate_kind
                    else args.target_kind),
                confidence=args.confidence, show=False,
                camera_names=("head",),
                max_inference_hz=args.max_inference_hz)
            aruco_node = pick.ArucoDetectNode(
                "head", marker_size=pick.MARKER_SIZE_M, publish_tf=False,
                publish_result_image=args.show)
            controller.configure_local_perception(yolo_node, aruco_node)
            nodes[0:0] = [yolo_node, aruco_node]
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
        result_document = {
            "schema_version": 1,
            "order_id": args.order_id,
            "kind": (
                args.target_kind if controller is None
                else controller.target_kind),
            "requested_kind": args.target_kind,
            "status": "delivered" if delivered else "failed",
            "marker_id": marker_id,
            "phase": phase,
            "state": state,
            "error": error,
            "elapsed_s": round(time.monotonic() - started_at, 3),
            "formal_mode": bool(args.formal_mode),
            "place_slot": args.place_slot,
            "place_xy": [place_x, place_y],
        }
        if controller is not None:
            result_document.update(controller.timing_snapshot())
        _write_result(args.result_file, result_document)
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
