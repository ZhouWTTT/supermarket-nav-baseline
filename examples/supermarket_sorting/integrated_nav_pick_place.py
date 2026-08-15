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
from collections import deque
from collections import Counter
import json
import math
import os
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

from candidate_observation_context import normalize_kind  # noqa: E402
from run_scan_coverage import shelf_band_for_pose  # noqa: E402

from place_retry_manager import (  # noqa: E402
    PlaceFailureReason,
    PlaceIKCandidate,
    PlaceRetryDecision,
    PlaceRetryManager,
    RecoverablePlaceFailure,
    ordered_unique_candidates,
)

from sensor_msgs.msg import Image, LaserScan  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from carrying_nav_diagnostics import (  # noqa: E402
    build_controller_trace,
    build_failure_evidence,
    save_failure_evidence,
)
from replay_observation_controller import (  # noqa: E402
    ADVANCE as REPLAY_ADVANCE,
    ReplayObservationController,
    ReplayObservationSnapshot,
)
from replay_outcome_memory import (  # noqa: E402
    NO_FRESH_RGB,
    RGB_NOT_PROCESSED_BY_YOLO,
    classify_fresh_frame_outcome,
    fresh_frame_gate_should_hold,
    minimum_processed_frames_from_success_evidence,
    processed_frame_wait_budget_s_from_success_evidence,
)
from replay_viewpoint_convergence import (  # noqa: E402
    ReplayViewpointConvergenceController,
    select_replay_head_poses,
)
from score_telemetry import EventLog, IdleGapObserver  # noqa: E402
from supermarket_navigation import (  # noqa: E402  (baseline nav, unmodified)
    DELIVERY_APPROACH,
    DELIVERY_TABLE_COSTMAP_BOUNDS,
    DELIVERY_TABLE_XML_BOUNDS,
    SupermarketNavigator,
    WHOLE_BODY_KEEP_OUT_RADIUS,
    point_to_rect_clearance,
)


def preferred_local_scan_exhausted(
        preferred_marker_id: int | None,
        previous_state: str,
        previous_pose_index: int,
        pose_count: int,
        current_pose_index: int,
        current_state: str,
        target_world) -> bool:
    """Return whether a provisional candidate's local replay was exhausted."""
    return (
        preferred_marker_id is not None
        and previous_state == pick.STATE_SCAN
        and previous_pose_index == pose_count - 1
        and current_pose_index == 0
        and current_state == pick.STATE_GO_SCAN
        and target_world is None)


def candidate_replay_poses(hint: dict) -> tuple:
    """Return one derived/observed view and one nearby backup view."""
    return select_replay_head_poses(
        hint, pick.SCAN_CAMERA_POSES,
        top_shelf_z_m=pick.TOP_SHELF_Z_M,
        middle_shelf_z_min_m=pick.MIDDLE_SHELF_Z_MIN_M)


def candidate_failure_stage_reason(
        diagnostics: dict, *, timed_out: bool = False) -> tuple[str, str]:
    """Classify the first strict gate that a bounded replay did not pass."""
    if ("raw_fresh_rgb_frame_count" in diagnostics
            and "yolo_processed_fresh_frame_count" in diagnostics):
        fresh_class = classify_fresh_frame_outcome(
            raw_fresh_rgb_count=diagnostics["raw_fresh_rgb_frame_count"],
            yolo_processed_count=diagnostics[
                "yolo_processed_fresh_frame_count"],
            target_detection_count=diagnostics.get(
                "target_kind_detection_count", 0))
        if fresh_class == NO_FRESH_RGB:
            return "fresh_rgb", "no_fresh_rgb"
        if fresh_class == RGB_NOT_PROCESSED_BY_YOLO:
            return "yolo_processing", "rgb_not_processed_by_yolo"
    if diagnostics.get("target_kind_detection_count", 0) <= 0:
        return "target_kind_detection", "no_target_kind"
    if diagnostics.get("aruco_detection_count", 0) <= 0:
        return "aruco_detection", "no_aruco"
    if diagnostics.get("association_candidate_count", 0) <= 0:
        return "association", "no_association"
    if diagnostics.get("max_association_confirmations", 0) < pick.ASSOCIATION_CONFIRMATIONS_REQUIRED:
        return "association", "insufficient_confirmations"
    if diagnostics.get("max_marker_samples", 0) < pick.MARKER_SAMPLES_REQUIRED:
        if timed_out:
            return "sample_collection", "scan_timeout_before_required_samples"
        return "sample_collection", "insufficient_marker_samples_before_pose_advance"
    if diagnostics.get("max_marker_spread_m", 0.0) > pick.MARKER_SAMPLE_SPREAD_MAX_M:
        return "sample_spread", "spread_reject"
    return "strict_localization", "scan_timeout" if timed_out else "bounds_reject"

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
PLACE_ARM_SETTLE_TIMEOUT_S = 8.0
PLACE_ARM_RETRY_MAX = 3
PLACE_CLEAR_TABLE_MARGIN_M = 0.060
PLACE_CLEAR_TABLE_SPEED_MPS = 0.10
PLACE_CLEAR_TABLE_TIMEOUT_S = 15.0
NAV_TRANSIT_GATE_M = 0.35          # beyond this distance, use the navigator
NAV_LASER_STALE_S = 0.50           # fail safe if the 12 Hz scan stops
NAV_STATE_STALE_S = 0.50           # odom/joints must also remain live
NAV_PROGRESS_LOG_S = 3.0
CANDIDATE_SCAN_DWELL_S = 2.5

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
            close_recheck: bool = True,
            telemetry: EventLog | None = None,
            order_id: str = "manual",
            run_prefix: str = "manual",
            memory_mode: str = "off",
            candidate_attempt_budget_s: float = 45.0,
            attempt_id: str = "manual-attempt",
            scan_coverage: dict | None = None,
            carrying_diagnostic_dir: str | pathlib.Path | None = None):
        self._telemetry_ready = False
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
        self.telemetry = telemetry if telemetry is not None else EventLog(None)
        # Trace telemetry is explicit.  The shelf controller never falls back
        # to ordinary telemetry when its sink is unset.
        self.strict_trace_sink = self.telemetry.emit
        self.order_id = order_id
        self.run_prefix = run_prefix
        self.memory_mode = memory_mode
        self.candidate_attempt_budget_s = float(candidate_attempt_budget_s)
        self.attempt_id = str(attempt_id)
        self.scan_coverage_snapshot = scan_coverage or {}
        self._covered_scan_keys = {
            (int(record["station_id"]), str(record["pose_name"]),
             str(record["shelf_band"]))
            for record in self.scan_coverage_snapshot.get("records", [])
            if record.get("run_prefix") == run_prefix
            and record.get("state") == "COVERED_VALID"
        }
        self._discovery_resumed_from_prior_worker = bool(
            self._covered_scan_keys)
        self._scan_pose_rgb_stamps = set()
        self._scan_pose_aruco_stamps = set()
        self._scan_pose_kinds = set()
        self._scan_pose_yolo_counts = Counter()
        self._scan_pose_aruco_detection_count = 0
        self._scan_pose_aruco_ids = set()
        self._scan_pose_pair_keys = set()
        self._scan_pose_pair_desync = 0
        self._global_scan_pose_active = None
        self._reported_covered_scan_keys = set()
        self.carrying_diagnostic_dir = (
            None if carrying_diagnostic_dir is None
            else pathlib.Path(carrying_diagnostic_dir))
        self.idle_observer = IdleGapObserver(15.0)
        self._grasp_event_open = False
        self._revalidation_event_open = False
        self._telemetry_ready = True
        self.candidate_hint = None
        self.candidate_attempt_started_at = None
        self.candidate_attempt_active = False
        self.candidate_strict_started_at = None
        self.candidate_timed_out = False
        self._candidate_budget_extension_emitted = False
        self.candidate_diagnostics = {}
        self.replay_observation_controller = ReplayObservationController(
            required_samples=pick.MARKER_SAMPLES_REQUIRED)
        self.replay_viewpoint_controller = (
            ReplayViewpointConvergenceController(
                base_position_tolerance_m=0.055,
                base_yaw_tolerance_rad=pick.NAV_YAW_DEADBAND_RAD,
                slide_tolerance_m=pick.SCAN_CAMERA_REACHED_SLIDE_M,
                head_tolerance_rad=pick.SCAN_CAMERA_REACHED_HEAD_RAD,
                stable_window_s=pick.SCAN_CAMERA_STABLE_S))
        self.replay_observation_policy = os.environ.get(
            "SUPERMARKET_REPLAY_OBSERVATION_POLICY", "adaptive"
        ).strip().lower()
        if self.replay_observation_policy not in {"adaptive", "fixed"}:
            raise ValueError(
                "SUPERMARKET_REPLAY_OBSERVATION_POLICY must be "
                "'adaptive' or 'fixed'")
        self._candidate_pose_last_progress_signature = None
        self._candidate_pose_end_emitted = False
        self._candidate_seen_yolo_stamps = set()
        self._candidate_seen_aruco_stamps = set()
        self._candidate_seen_association_pairs = set()
        self._candidate_raw_rgb_stamps = set()
        self._candidate_fresh_rgb_stamps = set()
        self._candidate_fresh_aruco_stamps = set()
        self._candidate_all_detected_kinds = set()
        self._candidate_first_fresh_rgb_at = None
        self._candidate_first_target_kind_at = None
        self._candidate_reacq_last_emit = float("-inf")
        self._candidate_reacq_emit_count = 0
        self._fresh_frame_gate_last_signature = None
        self._replay_pose_ids_attempted = set()
        self.fresh_frame_gate_mode = os.environ.get(
            "SUPERMARKET_R15_FRESH_FRAME_GATE", "control").strip().lower()
        if self.fresh_frame_gate_mode not in {"off", "shadow", "control"}:
            raise ValueError(
                "SUPERMARKET_R15_FRESH_FRAME_GATE must be off, shadow, or control")
        self.validated_marker_id = None
        self.validated_target_world = None
        self.validated_station_context = None
        self.validation_elapsed_s = None
        self.candidate_first_failure_stage = None
        self.candidate_first_failure_reason = None
        # Read-only execution classification.  It does not alter the deploy
        # gate, timeout, arm command, or transition chosen by the base class.
        self.execution_failure_stage = None
        self.execution_failure_reason = None

        # ── laser for the navigator ──
        self.laser_msg = None
        self.last_scan_time = None
        self.create_subscription(
            LaserScan, "/slamware_ros_sdk_server_node/scan",
            self._scan_cb, 10)
        # KeleDetectNode publishes this image only after completing YOLO for
        # the corresponding source RGB header.  Unlike the JSON detection
        # topic it also exists when the detector returns an empty list, so it
        # is read-only proof that a fresh zero-detection frame was processed.
        self.create_subscription(
            Image, "/kele/result_image", self._yolo_result_image_cb, 10)
        self.create_subscription(
            Image, "/head_camera/color/image_raw", self._raw_rgb_image_cb, 10)
        if memory_mode == "run_inventory":
            self.create_subscription(
                String, "/supermarket_sorting/memory_hint",
                self._memory_hint_cb, 10)
        self.scan_progress_publisher = self.create_publisher(
            String, "/supermarket_sorting/scan_progress", 50)
        self.worker_progress_publisher = self.create_publisher(
            String, "/supermarket_sorting/worker_progress", 20)

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
        self._place_retry_manager = None
        self._place_arm_sent_t0 = None
        self._place_grip_hold_command = None
        self._place_arm_target_sent = False
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
        self._backup_event_open = False
        self._backup_summary = None
        self._carrying_chain_id = None
        self._carrying_failure_sequence = 0
        self._carrying_failure_saved = False
        self._carrying_delivery_started_at = None
        self._carrying_delivery_event_open = False
        self._carrying_plan_observed = False
        self._carrying_trace_last_emit = float("-inf")
        self._carrying_trace_sequence = 0
        self._carrying_trace = deque(maxlen=1200)
        self._carrying_trace_best_distance = float("inf")
        self._carrying_trace_last_progress_time = None
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

    def yolo_cb(self, message: String) -> None:
        raw_records = [record for record in pick.decode_list(message)
                       if record.get("camera", "head") == "head"]
        if self._global_scan_pose_active is not None and raw_records:
            stamp = pick.stamp_from_record(raw_records[0])
            if stamp is not None:
                self._scan_pose_rgb_stamps.add(stamp)
                self._scan_pose_kinds.update(
                    normalize_kind(record.get("class"))
                    for record in raw_records if record.get("class"))
                self._scan_pose_yolo_counts.update(
                    normalize_kind(record.get("class"))
                    for record in raw_records if record.get("class"))
        if self.candidate_attempt_active:
            records = raw_records
            stamp = pick.stamp_from_record(records[0]) if records else None
            source = (self.candidate_hint or {}).get("source_yolo_stamp_ns")
            try:
                fresh = stamp is not None and (source is None
                                               or stamp > int(source))
            except (TypeError, ValueError):
                fresh = stamp is not None
            viewpoint = self.replay_viewpoint_controller.snapshot()
            fresh = (fresh and viewpoint.strict_scan_allowed
                     and stamp is not None
                     and self.replay_viewpoint_controller.frame_is_fresh(
                         stamp))
            if fresh and stamp not in self._candidate_fresh_rgb_stamps:
                self._candidate_fresh_rgb_stamps.add(stamp)
                now = self.now()
                if self._candidate_first_fresh_rgb_at is None:
                    self._candidate_first_fresh_rgb_at = now
                kinds = {normalize_kind(record.get("class"))
                         for record in records if record.get("class")}
                self._candidate_all_detected_kinds.update(kinds)
                if (normalize_kind(self.target_kind) in kinds
                        and self._candidate_first_target_kind_at is None):
                    self._candidate_first_target_kind_at = now
        if (self.candidate_attempt_active
                and self.state == pick.STATE_SCAN
                and raw_records):
            stamp = pick.stamp_from_record(raw_records[0])
            if (stamp is None
                    or not self.replay_viewpoint_controller.frame_is_fresh(
                        stamp)):
                return
        super().yolo_cb(message)
        self._observe_candidate_perception()

    def _yolo_result_image_cb(self, message: Image) -> None:
        if not self.candidate_attempt_active:
            return
        stamp = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec)
        source = (self.candidate_hint or {}).get("source_yolo_stamp_ns")
        try:
            fresh = source is None or stamp > int(source)
        except (TypeError, ValueError):
            fresh = True
        viewpoint = self.replay_viewpoint_controller.snapshot()
        fresh = (fresh and viewpoint.strict_scan_allowed
                 and self.replay_viewpoint_controller.frame_is_fresh(stamp))
        if fresh and stamp not in self._candidate_fresh_rgb_stamps:
            self._candidate_fresh_rgb_stamps.add(stamp)
            if self._candidate_first_fresh_rgb_at is None:
                self._candidate_first_fresh_rgb_at = self.now()

    def _raw_rgb_image_cb(self, message: Image) -> None:
        if not self.candidate_attempt_active:
            return
        stamp = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec)
        source = (self.candidate_hint or {}).get("source_yolo_stamp_ns")
        try:
            fresh = source is None or stamp > int(source)
        except (TypeError, ValueError):
            fresh = True
        viewpoint = self.replay_viewpoint_controller.snapshot()
        if (fresh and viewpoint.strict_scan_allowed
                and self.replay_viewpoint_controller.frame_is_fresh(stamp)):
            self._candidate_raw_rgb_stamps.add(stamp)

    def aruco_cb(self, message: String) -> None:
        raw_records = [record for record in pick.decode_list(message)
                       if record.get("camera", "head") == "head"]
        if self._global_scan_pose_active is not None and raw_records:
            stamp = pick.stamp_from_record(raw_records[0])
            if stamp is not None:
                self._scan_pose_aruco_stamps.add(stamp)
                self._scan_pose_aruco_detection_count += len(raw_records)
                for record in raw_records:
                    try:
                        self._scan_pose_aruco_ids.add(int(record["id"]))
                    except (KeyError, TypeError, ValueError):
                        pass
                if self._scan_pose_rgb_stamps:
                    nearest = min(
                        self._scan_pose_rgb_stamps,
                        key=lambda value: abs(value - stamp))
                    if abs(nearest - stamp) <= pick.ARUCO_SYNC_TOLERANCE_NS:
                        self._scan_pose_pair_keys.add((nearest, stamp))
                    else:
                        self._scan_pose_pair_desync += 1
        if self.candidate_attempt_active:
            records = raw_records
            stamp = pick.stamp_from_record(records[0]) if records else None
            source = (self.candidate_hint or {}).get("source_aruco_stamp_ns")
            try:
                fresh = stamp is not None and (source is None
                                               or stamp > int(source))
            except (TypeError, ValueError):
                fresh = stamp is not None
            viewpoint = self.replay_viewpoint_controller.snapshot()
            fresh = (fresh and viewpoint.strict_scan_allowed
                     and stamp is not None
                     and self.replay_viewpoint_controller.frame_is_fresh(
                         stamp))
            if fresh:
                self._candidate_fresh_aruco_stamps.add(stamp)
        if (self.candidate_attempt_active
                and self.state == pick.STATE_SCAN
                and raw_records):
            stamp = pick.stamp_from_record(raw_records[0])
            if (stamp is None
                    or not self.replay_viewpoint_controller.frame_is_fresh(
                        stamp)):
                return
        super().aruco_cb(message)
        self._observe_candidate_perception()

    def _publish_worker_phase(self) -> None:
        if self.flow_phase in {"backup"}:
            phase = "GRASPED"
        elif self.flow_phase == "nav_to_delivery":
            phase = "NAV_TO_DELIVERY"
        elif self.flow_phase == "place":
            phase = "PLACING"
        elif self.flow_phase == "done":
            phase = "DONE"
        elif self.state in {pick.STATE_DEPLOY, pick.STATE_CLOSE,
                           pick.STATE_LIFT, pick.STATE_RETREAT}:
            phase = "GRASP_ATTEMPT"
        elif self.candidate_attempt_active:
            phase = "CANDIDATE_REPLAY"
        else:
            phase = "DISCOVERY"
        self.worker_progress_publisher.publish(String(data=json.dumps({
            "attempt_id": self.attempt_id, "order_id": self.order_id,
            "run_prefix": self.run_prefix, "phase": phase,
        }, separators=(",", ":"))))

    def _global_coverage_key(self) -> tuple | None:
        if self.candidate_attempt_active or self.scan_station_order is None:
            return None
        if not (0 <= self.scan_index < len(self.scan_station_order)
                and 0 <= self.scan_pose_index < len(self.scan_poses)):
            return None
        station = int(self.scan_station_order[self.scan_index])
        pose_name = str(self.scan_poses[self.scan_pose_index][0])
        return (station, pose_name, shelf_band_for_pose(pose_name))

    def _publish_scan_progress(self, phase: str, **extra) -> None:
        key = self._global_scan_pose_active
        if key is None:
            return
        station, pose_name, band = key
        document = {
            "run_prefix": self.run_prefix, "attempt_id": self.attempt_id,
            "order_id": self.order_id, "phase": phase,
            "station_id": station, "pose_name": pose_name,
            "shelf_band": band, "monotonic_s": self.now(),
            **extra,
        }
        self.scan_progress_publisher.publish(String(data=json.dumps(
            document, separators=(",", ":"))))

    def _candidate_station_order(self, hint: dict) -> list[int]:
        ordered = list(super()._nearest_scan_stations())
        station = hint.get("scan_station_hint")
        index = station.get("index") if isinstance(station, dict) else station
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = None
        if index is not None and 0 <= index < len(pick.SCAN_X):
            ordered.remove(index)
            ordered.insert(0, index)
        return ordered

    def _nearest_scan_stations(self) -> list[int]:
        ordered = list(super()._nearest_scan_stations())
        if not isinstance(self.candidate_hint, dict):
            return ordered
        station = self.candidate_hint.get("scan_station_hint")
        index = station.get("index") if isinstance(station, dict) else station
        try:
            index = int(index)
        except (TypeError, ValueError):
            return ordered
        if 0 <= index < len(pick.SCAN_X):
            ordered.remove(index)
            ordered.insert(0, index)
        return ordered

    def _new_candidate_diagnostics(self) -> dict:
        return {
            "target_kind_detection_count": 0,
            "aruco_detection_count": 0,
            "aruco_seen_ids": [],
            "association_candidate_count": 0,
            "max_association_confirmations": 0,
            "max_marker_samples": 0,
            "max_marker_spread_m": 0.0,
            "no_target_kind": 0,
            "no_aruco": 0,
            "no_association": 0,
            "preferred_marker_mismatch": 0,
            "pair_desync": 0,
            "freshness_reject": 0,
            "insufficient_confirmations": 0,
            "sample_reject": 0,
            "spread_reject": 0,
            "bounds_reject": 0,
            "scan_timeout": 0,
        }

    def configure_candidate_hint(self, hint: dict) -> bool:
        """Start a bounded local replay without granting marker authority."""
        try:
            marker_id = int(hint.get(
                "provisional_marker_id", hint.get("marker_id")))
        except (TypeError, ValueError):
            return False
        if (not 0 <= marker_id <= 44
                or hint.get("kind", self.target_kind) != self.target_kind):
            return False
        normalized = dict(hint)
        normalized.setdefault("schema_version", 2)
        normalized.setdefault("candidate_id", f"candidate-{marker_id}")
        normalized.setdefault("kind", self.target_kind)
        normalized["provisional_marker_id"] = marker_id
        normalized.setdefault(
            "provisional_marker_world", hint.get("position_world"))
        normalized.setdefault("hint_source", "DERIVED_VIEW_HINT")
        normalized["head_pose_hint"] = list(candidate_replay_poses(normalized)[0])

        with self.lock:
            self.candidate_hint = normalized
            self.candidate_attempt_started_at = self.now()
            self.candidate_attempt_active = True
            self.candidate_strict_started_at = None
            self.candidate_timed_out = False
            self._candidate_budget_extension_emitted = False
            self.candidate_diagnostics = self._new_candidate_diagnostics()
            self._candidate_seen_yolo_stamps.clear()
            self._candidate_seen_aruco_stamps.clear()
            self._candidate_seen_association_pairs.clear()
            self._candidate_raw_rgb_stamps.clear()
            self._candidate_fresh_rgb_stamps.clear()
            self._candidate_fresh_aruco_stamps.clear()
            self._candidate_all_detected_kinds.clear()
            self._candidate_first_fresh_rgb_at = None
            self._candidate_first_target_kind_at = None
            self._candidate_reacq_last_emit = float("-inf")
            self._candidate_reacq_emit_count = 0
            self._fresh_frame_gate_last_signature = None
            self._replay_pose_ids_attempted.clear()
            self.validated_marker_id = None
            self.validated_target_world = None
            self.validated_station_context = None
            self.validation_elapsed_s = None
            self.candidate_first_failure_stage = None
            self.candidate_first_failure_reason = None
            # This marker and world position choose only the revisit view.
            self.preferred_marker_id = marker_id
            self.required_exact_marker_id = None
            self.scan_index = 0
            self.scan_pose_index = 0
            self.scan_cycles = 0
            self.scan_poses = candidate_replay_poses(normalized)
            self.scan_station_order = self._candidate_station_order(normalized)
            self.scan_dwell_s = CANDIDATE_SCAN_DWELL_S
            self.scan_camera_ready_since = None
            self.target_marker_id = None
            self.target_physical_marker_id = None
            self.target_world = None
            self.association_candidate_id = None
            self.association_confirmation_count = 0
            self.marker_positions.clear()
            self.depth_target_samples.clear()
            self.yolo_frames.clear()
            self.aruco_frames.clear()
            self._begin_replay_viewpoint_epoch()
        common = self._candidate_event_context()
        self.telemetry.emit(
            "candidate_attempt_start", **common,
            candidate_attempt_budget_s=self.candidate_attempt_budget_s,
            replay_observation_policy=self.replay_observation_policy)
        self.get_logger().info(
            f"[memory] candidate replay started id="
            f"{normalized['candidate_id']} provisional_marker={marker_id} "
            f"station={normalized.get('scan_station_hint')} "
            f"poses={[pose[0] for pose in self.scan_poses]} "
            f"budget={self.candidate_attempt_budget_s:.1f}s; marker is a "
            "view hint, not an exact-identity lock")
        self.set_state(pick.STATE_GO_SCAN)
        return True

    def _replay_viewpoint_pose_id(self) -> str:
        pose = self.scan_poses[self.scan_pose_index]
        return f"{self.scan_index}:{self.scan_pose_index}:{pose[0]}"

    def _begin_replay_viewpoint_epoch(self) -> None:
        """Isolate every replay pose before any strict evidence is admitted."""
        epoch = self.replay_viewpoint_controller.start_pose(
            pose_id=self._replay_viewpoint_pose_id(), now_s=self.now())
        self.marker_positions.clear()
        self.depth_target_samples.clear()
        self.association_candidate_id = None
        self.association_confirmation_count = 0
        self.last_association_pair = None
        self.target_marker_id = None
        self.target_physical_marker_id = None
        self.yolo_frames.clear()
        self.aruco_frames.clear()
        if self._telemetry_ready:
            self.telemetry.emit(
                "replay_viewpoint_epoch_started",
                **self._candidate_event_context(),
                scan_epoch_id=epoch,
                pose_id=self.replay_viewpoint_controller.pose_id,
                strict_sample_accumulator_cleared=True)

    def _observe_replay_viewpoint_convergence(self) -> None:
        if (not self.candidate_attempt_active
                or self.state != pick.STATE_GO_SCAN
                or self.base_xy is None):
            return
        observed = (self.candidate_hint or {}).get("observed_base_pose")
        if (self.scan_index == 0 and isinstance(observed, (list, tuple))
                and len(observed) == 3):
            base_target = [float(observed[0]), float(observed[1])]
            yaw_target = float(observed[2])
        else:
            base_target = [self.current_scan_station_x(), pick.SCAN_Y]
            yaw_target = pick.YAW_NORTH
        pose = self.current_scan_camera_pose()
        actual_head = (
            self.joints.get("slide_joint"),
            self.joints.get("head_yaw_joint"),
            self.joints.get("head_pitch_joint"))
        if any(value is None for value in actual_head):
            return
        position_error = float(np.linalg.norm(
            np.asarray(base_target, dtype=float) - self.base_xy))
        yaw_error = abs(pick.wrap_to_pi(yaw_target - self.base_yaw))
        head_error = tuple(
            float(actual_head[index]) - float(pose[index + 1])
            for index in range(3))
        before = self.replay_viewpoint_controller.snapshot()
        after = self.replay_viewpoint_controller.observe(
            now_s=self.now(), base_position_error_m=position_error,
            base_yaw_error_rad=yaw_error, head_error=head_error)
        if (after.strict_scan_allowed and not before.strict_scan_allowed):
            source_stamps = []
            if self.yolo_frames:
                source_stamps.append(self.yolo_frames[-1][0])
            if self.aruco_frames:
                source_stamps.append(self.aruco_frames[-1][0])
            self.replay_viewpoint_controller.set_source_stamp_boundary(
                *source_stamps)
            self.telemetry.emit(
                "replay_viewpoint_converged",
                **self._candidate_event_context(),
                scan_epoch_id=after.scan_epoch_id, pose_id=after.pose_id,
                base_target_reached=after.base_target_reached,
                head_target_reached=after.head_target_reached,
                camera_settle_start_monotonic_s=(
                    after.stable_since_monotonic_s),
                camera_settle_end_monotonic_s=(
                    after.convergence_monotonic_s),
                strict_scan_started_after_convergence=True)

    def _candidate_event_context(self) -> dict:
        hint = self.candidate_hint or {}
        return {
            "order_id": self.order_id,
            "candidate_id": hint.get("candidate_id"),
            "kind": self.target_kind,
            "normalized_kind": normalize_kind(self.target_kind),
            "provisional_marker_id": hint.get("provisional_marker_id"),
            "provisional_marker_world": hint.get("provisional_marker_world"),
            "source_yolo_stamp_ns": hint.get("source_yolo_stamp_ns"),
            "source_aruco_stamp_ns": hint.get("source_aruco_stamp_ns"),
            "confidence": hint.get("confidence"),
            "confirmations": hint.get("confirmations"),
            "observation_base_pose_hint": hint.get("observation_base_pose_hint"),
            "head_pose_hint": hint.get("head_pose_hint"),
            "scan_station_hint": hint.get("scan_station_hint"),
            "scan_pitch_hint": hint.get("scan_pitch_hint"),
            "hint_source": hint.get("hint_source"),
            "context_source": hint.get("context_source", "DERIVED"),
            "context_type": hint.get("context_type", "DERIVED_CONTEXT"),
            "context_quality": hint.get("context_quality"),
            "creation_base_pose": hint.get("observed_base_pose"),
            "creation_head_pose": hint.get("observed_head_pose"),
            "creation_scan_station": hint.get("observed_scan_station"),
            "creation_pose_name": hint.get("observed_pose_name"),
            "creation_source_stamps": hint.get("observed_source_stamps"),
            "creation_target_bbox": hint.get("target_bbox_summary"),
            "creation_marker_pixel": hint.get("marker_pixel_summary"),
        }

    def _candidate_reacquisition_payload(self) -> dict:
        hint = self.candidate_hint or {}
        target_base = hint.get("observed_base_pose")
        actual_base = (None if self.base_xy is None else [
            float(self.base_xy[0]), float(self.base_xy[1]),
            float(self.base_yaw)])
        target_head = None
        if self.scan_poses and 0 <= self.scan_pose_index < len(self.scan_poses):
            target_head = list(self.scan_poses[self.scan_pose_index])
        actual_head = None
        head_names = ("slide_joint", "head_yaw_joint", "head_pitch_joint")
        if all(name in self.joints for name in head_names):
            actual_head = [self.joints[name] for name in head_names]
        position_error = yaw_error = None
        if (isinstance(target_base, (list, tuple)) and len(target_base) == 3
                and actual_base is not None):
            position_error = math.hypot(
                actual_base[0] - float(target_base[0]),
                actual_base[1] - float(target_base[1]))
            yaw_error = abs(pick.wrap_to_pi(
                actual_base[2] - float(target_base[2])))
        head_error = None
        if (isinstance(target_head, (list, tuple)) and len(target_head) == 4
                and actual_head is not None):
            head_error = [actual_head[index] - float(target_head[index + 1])
                          for index in range(3)]
        started = self.candidate_attempt_started_at
        first_rgb = (None if started is None
                     or self._candidate_first_fresh_rgb_at is None
                     else self._candidate_first_fresh_rgb_at - started)
        first_target = (None if started is None
                        or self._candidate_first_target_kind_at is None
                        else self._candidate_first_target_kind_at - started)
        viewpoint = self.replay_viewpoint_controller.snapshot()
        return {
            "replay_base_pose_target": target_base,
            "replay_base_pose_actual": actual_base,
            "replay_base_position_error_m": (None if position_error is None
                                             else round(position_error, 6)),
            "replay_base_yaw_error_rad": (None if yaw_error is None
                                         else round(yaw_error, 6)),
            "replay_head_pose_target": target_head,
            "replay_head_pose_actual": actual_head,
            "replay_head_error": (None if head_error is None else
                                  [round(value, 6) for value in head_error]),
            "fresh_rgb_frame_count": len(self._candidate_raw_rgb_stamps),
            "yolo_processed_fresh_frame_count": len(
                self._candidate_fresh_rgb_stamps),
            "fresh_aruco_frame_count": len(self._candidate_fresh_aruco_stamps),
            "target_kind_detection_count": self.candidate_diagnostics.get(
                "target_kind_detection_count", 0),
            "all_detected_kinds": sorted(self._candidate_all_detected_kinds),
            "time_to_first_fresh_rgb_s": (None if first_rgb is None
                                          else round(first_rgb, 3)),
            "time_to_first_target_kind_s": (None if first_target is None
                                             else round(first_target, 3)),
            "scan_epoch_id": viewpoint.scan_epoch_id,
            "pose_id": viewpoint.pose_id,
            "base_target_reached": viewpoint.base_target_reached,
            "head_target_reached": viewpoint.head_target_reached,
            "camera_settled_after_target_reached": (
                viewpoint.camera_settled_after_target_reached),
            "strict_scan_started_after_convergence": (
                viewpoint.strict_scan_allowed),
        }

    def _observe_candidate_perception(self) -> None:
        if not self.candidate_attempt_active:
            return
        with self.lock:
            diagnostics = self.candidate_diagnostics
            if self.yolo_frames:
                stamp, detections = self.yolo_frames[-1]
                if stamp not in self._candidate_seen_yolo_stamps:
                    self._candidate_seen_yolo_stamps.add(stamp)
                    exact = sum(
                        item.get("class") == self.target_kind
                        for item in detections)
                    diagnostics["target_kind_detection_count"] += exact
                    diagnostics["no_target_kind"] += int(exact == 0)
            if self.aruco_frames:
                stamp, markers = self.aruco_frames[-1]
                if stamp not in self._candidate_seen_aruco_stamps:
                    self._candidate_seen_aruco_stamps.add(stamp)
                    diagnostics["aruco_detection_count"] += len(markers)
                    diagnostics["no_aruco"] += int(not markers)
                    seen = set(diagnostics["aruco_seen_ids"])
                    for marker in markers:
                        try:
                            seen.add(int(marker["id"]))
                        except (KeyError, TypeError, ValueError):
                            pass
                    diagnostics["aruco_seen_ids"] = sorted(seen)[:45]
            if self.yolo_frames and self.aruco_frames:
                yolo_stamp = self.yolo_frames[-1][0]
                aruco_stamp = min(
                    self.aruco_frames,
                    key=lambda frame: abs(frame[0] - yolo_stamp))[0]
                diagnostics["pair_desync"] += int(
                    abs(aruco_stamp - yolo_stamp) > pick.ARUCO_SYNC_TOLERANCE_NS)
            pair = self.last_association_pair
            if (pair is not None
                    and pair not in self._candidate_seen_association_pairs
                    and self.association_candidate_id is not None):
                self._candidate_seen_association_pairs.add(pair)
                diagnostics["association_candidate_count"] += 1
            diagnostics["max_association_confirmations"] = max(
                diagnostics["max_association_confirmations"],
                int(self.association_confirmation_count))
            diagnostics["max_marker_samples"] = max(
                diagnostics["max_marker_samples"], len(self.marker_positions))
            if len(self.marker_positions) >= 2:
                spread = float(np.max(np.ptp(
                    np.asarray(self.marker_positions, dtype=float), axis=0)))
                diagnostics["max_marker_spread_m"] = max(
                    diagnostics["max_marker_spread_m"], spread)
            observed = self.target_marker_id
            provisional = (self.candidate_hint or {}).get(
                "provisional_marker_id")
            if (observed is not None and provisional is not None
                    and int(observed) != int(provisional)):
                diagnostics["preferred_marker_mismatch"] = 1

    def _memory_hint_cb(self, message: String) -> None:
        """Apply a same-run public-perception hint, then re-localise normally."""
        try:
            hint = json.loads(message.data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if (hint.get("schema_version") not in {1, 2}
                or hint.get("run_prefix") != self.run_prefix
                or hint.get("order_id") != self.order_id
                or hint.get("kind") != self.target_kind
                or self.flow_phase != "grab"
                or self.state not in {pick.STATE_GO_SCAN, pick.STATE_SCAN}
                or self.target_world is not None
                or self.candidate_attempt_active):
            return
        hint.setdefault(
            "provisional_marker_id", hint.get("marker_id"))
        hint.setdefault(
            "provisional_marker_world", hint.get("position_world"))
        self.configure_candidate_hint(hint)

    def _candidate_station_context(self) -> dict:
        pose = None
        if self.scan_poses and 0 <= self.scan_pose_index < len(self.scan_poses):
            pose = list(self.scan_poses[self.scan_pose_index])
        station_index = None
        if (self.scan_station_order is not None
                and 0 <= self.scan_index < len(self.scan_station_order)):
            station_index = int(self.scan_station_order[self.scan_index])
        return {
            "scan_station_index": station_index,
            "scan_station_x": (
                None if station_index is None
                else float(pick.SCAN_X[station_index])),
            "scan_pose": pose,
            "base_pose": (
                None if self.base_xy is None
                else [float(self.base_xy[0]), float(self.base_xy[1]),
                      float(self.base_yaw)]),
        }

    def _replay_observation_snapshot(self) -> ReplayObservationSnapshot:
        raw = super().replay_observation_snapshot()
        samples = int(raw["accepted_sample_count"])
        confirmations = int(raw["association_confirmation_count"])
        association_candidates = 0
        if raw["association_candidate_present"]:
            if self.target_marker_id is None:
                association_candidates = confirmations
            else:
                association_candidates = max(
                    confirmations,
                    pick.ASSOCIATION_CONFIRMATIONS_REQUIRED - 1 + samples)
        synchronized = int(raw["fresh_synchronized_pair_count"])
        association_success_rate = (
            min(1.0, confirmations / synchronized)
            if synchronized > 0 else 0.0)
        return ReplayObservationSnapshot(
            target_kind_detection_count=int(
                raw["target_kind_detection_count"]),
            aruco_detection_count=int(raw["aruco_detection_count"]),
            fresh_synchronized_pair_count=int(
                raw["fresh_synchronized_pair_count"]),
            duplicate_count=int(raw["duplicate_count"]),
            freshness_rejection_count=int(
                raw["freshness_rejection_count"]),
            association_candidate_count=association_candidates,
            association_confirmation_count=confirmations,
            association_success_rate=association_success_rate,
            accepted_sample_count=samples,
            localized=bool(raw["localized"]),
        )

    def _terminal_replay_observation_snapshot(
            self, *, localized: bool) -> ReplayObservationSnapshot:
        """Capture final counters without reacquiring the perception lock.

        Strict localization changes state from inside its locked callback, so
        the public read-only snapshot cannot be called re-entrantly here.
        """
        previous = self.replay_observation_controller.last_snapshot
        samples = len(self.marker_positions)
        confirmations = int(self.association_confirmation_count)
        association_candidates = previous.association_candidate_count
        if self.association_candidate_id is not None:
            association_candidates = max(
                association_candidates,
                confirmations if self.target_marker_id is None else
                pick.ASSOCIATION_CONFIRMATIONS_REQUIRED - 1 + samples)
        return ReplayObservationSnapshot(
            target_kind_detection_count=(
                previous.target_kind_detection_count),
            aruco_detection_count=previous.aruco_detection_count,
            fresh_synchronized_pair_count=(
                previous.fresh_synchronized_pair_count),
            duplicate_count=previous.duplicate_count,
            freshness_rejection_count=(
                previous.freshness_rejection_count),
            association_candidate_count=association_candidates,
            association_confirmation_count=confirmations,
            association_success_rate=previous.association_success_rate,
            accepted_sample_count=samples,
            localized=localized)

    @staticmethod
    def _replay_snapshot_payload(
            snapshot: ReplayObservationSnapshot) -> dict:
        return {
            "accepted_sample_count": snapshot.accepted_sample_count,
            "association_confirmation_count": (
                snapshot.association_confirmation_count),
            "association_candidate_count": (
                snapshot.association_candidate_count),
            "fresh_synchronized_pair_count": (
                snapshot.fresh_synchronized_pair_count),
            "duplicate_count": snapshot.duplicate_count,
            "freshness_rejection_count": (
                snapshot.freshness_rejection_count),
            "target_kind_detection_count": (
                snapshot.target_kind_detection_count),
            "aruco_detection_count": snapshot.aruco_detection_count,
            "association_success_rate": round(
                snapshot.association_success_rate, 6),
        }

    def _start_replay_pose_observation(self, now: float) -> None:
        snapshot = self._replay_observation_snapshot()
        self.replay_observation_controller.start_pose(now, snapshot)
        self._candidate_pose_last_progress_signature = None
        self._candidate_pose_end_emitted = False
        self._replay_pose_ids_attempted.add(
            self.replay_viewpoint_controller.pose_id)
        self.scan_dwell_s = (
            CANDIDATE_SCAN_DWELL_S
            if self.replay_observation_policy == "fixed"
            else float("inf"))
        self.telemetry.emit(
            "replay_pose_start",
            **self._candidate_event_context(),
            station_context=self._candidate_station_context(),
            replay_observation_policy=self.replay_observation_policy,
            **self._replay_snapshot_payload(snapshot))
        self.telemetry.emit(
            "candidate_observation_context_replayed",
            **self._candidate_event_context(),
            **self._candidate_reacquisition_payload())

    def _emit_replay_pose_end(
            self, reason: str, now: float,
            snapshot: ReplayObservationSnapshot | None = None) -> None:
        if self._candidate_pose_end_emitted:
            return
        snapshot = snapshot or self._replay_observation_snapshot()
        started = self.replay_observation_controller.pose_started_at
        elapsed = 0.0 if started is None else max(0.0, now - started)
        self.telemetry.emit(
            "replay_pose_end",
            **self._candidate_event_context(),
            station_context=self._candidate_station_context(),
            pose_elapsed_s=round(elapsed, 3),
            pose_advance_reason=reason,
            **self._replay_snapshot_payload(snapshot))
        self._candidate_pose_end_emitted = True

    def _apply_replay_observation_policy(self, now: float) -> None:
        if (not self.candidate_attempt_active
                or self.state != pick.STATE_SCAN):
            return
        snapshot = self._replay_observation_snapshot()
        signature = (
            snapshot.accepted_sample_count,
            snapshot.association_confirmation_count,
            snapshot.association_candidate_count,
            snapshot.fresh_synchronized_pair_count,
            snapshot.duplicate_count,
            snapshot.freshness_rejection_count,
        )
        if signature != self._candidate_pose_last_progress_signature:
            self._candidate_pose_last_progress_signature = signature
            started = self.replay_observation_controller.pose_started_at
            elapsed = 0.0 if started is None else max(0.0, now - started)
            self.telemetry.emit(
                "replay_observation_progress",
                **self._candidate_event_context(),
                station_context=self._candidate_station_context(),
                pose_elapsed_s=round(elapsed, 3),
                **self._replay_snapshot_payload(snapshot))
            if (now - self._candidate_reacq_last_emit >= 0.5
                    and self._candidate_reacq_emit_count < 120):
                self._candidate_reacq_last_emit = now
                self._candidate_reacq_emit_count += 1
                self.telemetry.emit(
                    "candidate_target_reacquisition_progress",
                    **self._candidate_event_context(),
                    **self._candidate_reacquisition_payload())
        if self.replay_observation_policy == "fixed":
            started = self.replay_observation_controller.pose_started_at
            if (started is not None
                    and now - started >= CANDIDATE_SCAN_DWELL_S):
                self._emit_replay_pose_end(
                    "fixed_dwell_elapsed", now, snapshot)
            self.scan_dwell_s = CANDIDATE_SCAN_DWELL_S
            return
        started = self.replay_observation_controller.pose_started_at
        pose_elapsed = 0.0 if started is None else max(0.0, now - started)
        would_hold = fresh_frame_gate_should_hold(
            target_detection_count=snapshot.target_kind_detection_count,
            yolo_processed_count=len(self._candidate_fresh_rgb_stamps),
            pose_elapsed_s=pose_elapsed)
        gate_signature = (
            would_hold, len(self._candidate_raw_rgb_stamps),
            len(self._candidate_fresh_rgb_stamps),
            snapshot.target_kind_detection_count)
        if gate_signature != self._fresh_frame_gate_last_signature:
            self._fresh_frame_gate_last_signature = gate_signature
            self.telemetry.emit(
                "fresh_frame_gate_decision",
                **self._candidate_event_context(),
                gate_mode=self.fresh_frame_gate_mode,
                would_hold=would_hold,
                raw_fresh_rgb_frame_count=len(self._candidate_raw_rgb_stamps),
                yolo_processed_fresh_frame_count=len(
                    self._candidate_fresh_rgb_stamps),
                target_kind_detection_count=(
                    snapshot.target_kind_detection_count),
                minimum_processed_frame_count=(
                    minimum_processed_frames_from_success_evidence()),
                wait_budget_s=(
                    processed_frame_wait_budget_s_from_success_evidence()),
                pose_elapsed_s=round(pose_elapsed, 3))
        if self.fresh_frame_gate_mode == "control" and would_hold:
            self.scan_dwell_s = float("inf")
            return
        decision = self.replay_observation_controller.observe(now, snapshot)
        if decision.action == REPLAY_ADVANCE:
            self._emit_replay_pose_end(decision.reason, now, snapshot)
            # The unchanged parent owns pose indexing and buffer reset.  A
            # zero dwell asks it to perform that normal transition now.
            self.scan_dwell_s = 0.0
        else:
            self.scan_dwell_s = float("inf")

    def _adaptive_observation_budget_available(self, now: float) -> bool:
        return (
            self.replay_observation_policy == "adaptive"
            and self.state == pick.STATE_SCAN
            and self.replay_observation_controller.observation_budget_available(
                now))

    def _finalize_candidate_attempt(self, *, promoted: bool) -> None:
        if not self.candidate_attempt_active:
            return
        now = self.now()
        diagnostics = self.candidate_diagnostics
        diagnostics["raw_fresh_rgb_frame_count"] = len(
            self._candidate_raw_rgb_stamps)
        diagnostics["yolo_processed_fresh_frame_count"] = len(
            self._candidate_fresh_rgb_stamps)
        diagnostics["max_association_confirmations"] = max(
            diagnostics.get("max_association_confirmations", 0),
            int(self.association_confirmation_count))
        diagnostics["max_marker_samples"] = max(
            diagnostics.get("max_marker_samples", 0),
            len(self.marker_positions))
        if len(self.marker_positions) >= 2:
            diagnostics["max_marker_spread_m"] = max(
                diagnostics.get("max_marker_spread_m", 0.0),
                float(np.max(np.ptp(
                    np.asarray(self.marker_positions, dtype=float), axis=0))))
        context = self._candidate_event_context()
        station_context = self._candidate_station_context()
        elapsed = max(
            0.0, now - float(self.candidate_attempt_started_at or now))
        strict_elapsed = None
        if self.candidate_strict_started_at is not None:
            strict_elapsed = max(0.0, now - self.candidate_strict_started_at)
            self.telemetry.emit(
                "strict_localization_end", **context,
                elapsed_s=round(strict_elapsed, 3), passed=promoted,
                strictly_observed_marker_id=self.target_marker_id)

        first_stage = None
        first_reason = None
        corrected = False
        if promoted:
            self.validated_marker_id = self.target_marker_id
            self.validated_target_world = (
                None if self.target_world is None
                else [float(value) for value in self.target_world])
            self.validated_station_context = station_context
            self.validation_elapsed_s = round(elapsed, 3)
            self.required_exact_marker_id = self.target_marker_id
            provisional = context.get("provisional_marker_id")
            corrected = (
                provisional is not None and self.target_marker_id is not None
                and int(provisional) != int(self.target_marker_id))
            self.telemetry.emit(
                "localization_validated", **context,
                strictly_observed_marker_id=self.target_marker_id,
                validated_marker_id=self.validated_marker_id,
                validated_target_world=self.validated_target_world,
                validated_station_context=station_context,
                validation_elapsed_s=self.validation_elapsed_s)
            self.telemetry.emit(
                "candidate_promoted", **context,
                validated_marker_id=self.validated_marker_id,
                validation_elapsed_s=self.validation_elapsed_s)
            if corrected:
                self.telemetry.emit(
                    "candidate_identity_corrected", **context,
                    strictly_observed_marker_id=self.target_marker_id,
                    validated_marker_id=self.validated_marker_id)
        else:
            first_stage, first_reason = candidate_failure_stage_reason(
                diagnostics, timed_out=self.candidate_timed_out)
            self.candidate_first_failure_stage = first_stage
            self.candidate_first_failure_reason = first_reason
            diagnostics["no_association"] = int(
                diagnostics.get("association_candidate_count", 0) <= 0)
            diagnostics["insufficient_confirmations"] = int(
                diagnostics.get("association_candidate_count", 0) > 0
                and diagnostics.get("max_association_confirmations", 0)
                < pick.ASSOCIATION_CONFIRMATIONS_REQUIRED)
            diagnostics["sample_reject"] = int(
                diagnostics.get("max_association_confirmations", 0)
                >= pick.ASSOCIATION_CONFIRMATIONS_REQUIRED
                and diagnostics.get("max_marker_samples", 0)
                < pick.MARKER_SAMPLES_REQUIRED)
            diagnostics["spread_reject"] = int(
                diagnostics.get("max_marker_spread_m", 0.0)
                > pick.MARKER_SAMPLE_SPREAD_MAX_M)
            diagnostics["scan_timeout"] = int(self.candidate_timed_out)
            self.telemetry.emit(
                "candidate_invalidated", **context,
                strictly_observed_marker_id=self.target_marker_id,
                first_failure_stage=first_stage,
                first_failure_reason=first_reason)

        self.telemetry.emit(
            "candidate_attempt_end", **context,
            candidate_revisit_s=round(elapsed, 3),
            strict_localization_s=(
                None if strict_elapsed is None else round(strict_elapsed, 3)),
            first_failure_stage=first_stage,
            first_failure_reason=first_reason,
            candidate_promoted=promoted,
            candidate_identity_corrected=corrected,
            candidate_invalidated=not promoted,
            strictly_observed_marker_id=self.target_marker_id,
            validated_marker_id=self.validated_marker_id,
            **self.strict_trace_summary(),
            **diagnostics)
        self.strict_trace_emit_summary(
            first_failure_stage=first_stage,
            final_outcome=(
                "LOCALIZATION_VALIDATED" if promoted else
                (first_reason or "OTHER_TERMINAL")))
        self.telemetry.emit(
            "candidate_target_reacquisition_end", **context,
            **self._candidate_reacquisition_payload(),
            exit_reason=(
                "localization_validated" if promoted
                else (first_reason or "candidate_attempt_ended")))
        self.candidate_attempt_active = False
        self.scan_dwell_s = pick.SCAN_DWELL_S

    def set_state(self, new_state: str) -> None:
        previous = getattr(self, "state", None)
        if (getattr(self, "_telemetry_ready", False)
                and new_state == pick.STATE_ABORT
                and previous == pick.STATE_DEPLOY
                and self.validated_marker_id is not None
                and not self.use_dual_tissue_grasp
                and not self.use_sphere_grasp
                and self.now() - self.state_t0
                >= pick.GENERIC_DEPLOY_HARD_TIMEOUT_S):
            self.execution_failure_stage = "pregrasp"
            self.execution_failure_reason = (
                "pregrasp_arm_convergence_timeout")
        if (new_state == pick.STATE_SCAN
                and self.candidate_attempt_active
                and not self.replay_viewpoint_controller.snapshot(
                ).strict_scan_allowed):
            return
        super().set_state(new_state)
        if (not getattr(self, "_telemetry_ready", False)
                or previous == new_state):
            return
        if (new_state == pick.STATE_GO_SCAN
                and self.candidate_attempt_active):
            self._begin_replay_viewpoint_epoch()
        common = {"order_id": self.order_id, "kind": self.target_kind}
        if (new_state == pick.STATE_SCAN
                and not self.candidate_attempt_active):
            self._global_scan_pose_active = self._global_coverage_key()
            self._scan_pose_rgb_stamps.clear()
            self._scan_pose_aruco_stamps.clear()
            self._scan_pose_kinds.clear()
            self._scan_pose_yolo_counts.clear()
            self._scan_pose_aruco_detection_count = 0
            self._scan_pose_aruco_ids.clear()
            self._scan_pose_pair_keys.clear()
            self._scan_pose_pair_desync = 0
            self._publish_scan_progress(
                "started", resumed=self._discovery_resumed_from_prior_worker,
                camera_settled=True)
        if (previous == pick.STATE_SCAN
                and new_state != pick.STATE_SCAN
                and self._global_scan_pose_active is not None):
            key = self._global_scan_pose_active
            self._publish_scan_progress(
                "completed", camera_settled=True,
                pose_completed=True, interrupted=False,
                fresh_rgb_frame_count=len(self._scan_pose_rgb_stamps),
                fresh_aruco_frame_count=len(self._scan_pose_aruco_stamps),
                observed_kinds=sorted(self._scan_pose_kinds),
                yolo_detection_count_by_kind=dict(
                    sorted(self._scan_pose_yolo_counts.items())),
                aruco_detection_count=self._scan_pose_aruco_detection_count,
                aruco_seen_ids=sorted(self._scan_pose_aruco_ids),
                synchronized_frame_pair_count=len(self._scan_pose_pair_keys),
                pair_desync_reject_count=self._scan_pose_pair_desync,
                association_attempt_count=0,
                association_success_count=0,
                association_reject_reason_counts={},
                candidate_ids_created=[],
                completion_reason=("target_localized"
                                   if new_state == pick.STATE_ALIGN
                                   else "pose_observation_elapsed"))
            self._covered_scan_keys.add(key)
            self._global_scan_pose_active = None
        if (previous == pick.STATE_SCAN
                and self.candidate_attempt_active
                and new_state != pick.STATE_SCAN
                and self.replay_observation_controller.pose_started_at
                is not None
                and not self._candidate_pose_end_emitted):
            reason = (
                "authoritative_localization"
                if new_state == pick.STATE_ALIGN
                else f"state_transition:{new_state}")
            terminal_snapshot = self._terminal_replay_observation_snapshot(
                localized=new_state == pick.STATE_ALIGN)
            self._emit_replay_pose_end(
                reason, self.now(), terminal_snapshot)
        if (new_state == pick.STATE_SCAN
                and self.candidate_attempt_active):
            self._start_replay_pose_observation(self.now())
            if self.candidate_strict_started_at is None:
                self.candidate_strict_started_at = self.now()
                self.telemetry.emit(
                    "strict_localization_start",
                    **self._candidate_event_context(),
                    station_context=self._candidate_station_context())
        if new_state == pick.STATE_ALIGN:
            self.telemetry.emit(
                "search_end", **common, marker_id=self.target_marker_id,
                outcome="localized")
            self.telemetry.emit(
                "full_scan_end", **common, marker_id=self.target_marker_id,
                reason="target_localized")
            self.telemetry.emit(
                "navigation_to_shelf_end", **common,
                marker_id=self.target_marker_id)
            self._finalize_candidate_attempt(promoted=True)
        if new_state == pick.STATE_RECHECK:
            self._revalidation_event_open = True
            self.telemetry.emit(
                "local_revalidation_start", **common,
                marker_id=self.target_marker_id)
        if (previous == pick.STATE_RECHECK
                and self._revalidation_event_open
                and new_state != pick.STATE_RECHECK):
            self._revalidation_event_open = False
            self.telemetry.emit(
                "local_revalidation_end", **common,
                marker_id=self.target_marker_id,
                passed=new_state == pick.STATE_DEPLOY)
        if new_state == pick.STATE_DEPLOY:
            self._grasp_event_open = True
            self.telemetry.emit(
                "grasp_start", **common, marker_id=self.target_marker_id)
        if new_state == pick.STATE_DONE and self._grasp_event_open:
            self._grasp_event_open = False
            self.telemetry.emit(
                "grasp_end", **common, marker_id=self.target_marker_id,
                success=True)
        if new_state == pick.STATE_ABORT:
            if self._grasp_event_open:
                self._grasp_event_open = False
                self.telemetry.emit(
                    "grasp_end", **common, marker_id=self.target_marker_id,
                    success=False)
            failure = (
                "localization_failure" if self.target_world is None
                else "grasp_failure")
            self.telemetry.emit(
                failure, **common, marker_id=self.target_marker_id,
                previous_state=previous)
            self._finalize_candidate_attempt(promoted=False)

    def _observe_flow_transition(self, previous: str) -> None:
        if previous == self.flow_phase:
            return
        common = {"order_id": self.order_id, "kind": self.target_kind,
                  "marker_id": self.target_marker_id}
        if self.flow_phase == "nav_to_delivery":
            self.telemetry.emit("navigation_to_delivery_start", **common)
        elif self.flow_phase == "place":
            self.telemetry.emit("navigation_to_delivery_end", **common)
            self.telemetry.emit("place_start", **common)
        elif self.flow_phase == "done":
            self.telemetry.emit("place_end", **common, success=True)

    def _observe_idle_progress(self, now: float) -> None:
        if self.base_xy is None:
            return
        joint_values = tuple(
            round(float(value), 1)
            for _, value in sorted(self.joints.items()))
        signature = (
            self.flow_phase, self.state, self.place_stage,
            round(float(self.base_xy[0]), 1),
            round(float(self.base_xy[1]), 1),
            round(float(self.base_yaw), 1),
            joint_values,
        )
        duration = self.idle_observer.update(now, signature)
        if duration is not None and duration > 15.0:
            self.telemetry.emit(
                "idle_gap", order_id=self.order_id,
                kind=self.target_kind, duration_s=round(duration, 3),
                phase=self.flow_phase, state=self.state)

    def flush_idle_telemetry(self) -> None:
        duration = self.idle_observer.flush(self.now())
        if duration is not None and duration > 15.0:
            self.telemetry.emit(
                "idle_gap", order_id=self.order_id,
                kind=self.target_kind, duration_s=round(duration, 3),
                phase=self.flow_phase, state=self.state)

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
        observed = ((self.candidate_hint or {}).get("observed_base_pose")
                    if self.candidate_attempt_active else None)
        if (self.flow_phase == "grab"
                and self.state in {pick.STATE_GO_SCAN, pick.STATE_SCAN}
                and self.scan_index == 0
                and isinstance(observed, (list, tuple))
                and len(observed) == 3):
            try:
                target_xy = [float(observed[0]), float(observed[1])]
                final_yaw = float(observed[2])
            except (TypeError, ValueError):
                pass
            position_tolerance = min(
                float(position_tolerance),
                self.replay_viewpoint_controller.base_position_tolerance_m)
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
                if ctrl.stop_reason.startswith("no_path"):
                    self.telemetry.emit(
                        "no_path", order_id=self.order_id,
                        kind=self.target_kind, phase="navigation_to_shelf",
                        reason=ctrl.stop_reason)
                elif "replan" in ctrl.stop_reason:
                    self.telemetry.emit(
                        "replan", order_id=self.order_id,
                        kind=self.target_kind, phase="navigation_to_shelf",
                        reason=ctrl.stop_reason)

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

        return super().drive_to(target, final_yaw, position_tolerance)

    # ------------------------------------------------------------------
    # flow hooks
    # ------------------------------------------------------------------
    @staticmethod
    def _diagnostic_vector(value) -> list[float] | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            return None
        return [round(float(item), 6) for item in array]

    def _carrying_event_context(self) -> dict:
        return {
            "order_id": self.order_id,
            "kind": self.target_kind,
            "marker_id": self.target_marker_id,
            "carrying_chain_id": self._carrying_chain_id,
        }

    def _payload_diagnostic_state(self) -> dict:
        measured_gripper = self.selected_gripper_position()
        if self.grasp_arm == "r":
            commanded_gripper = self.cmd_right_grip
        else:
            commanded_gripper = self.cmd_left_grip
        return {
            "carry_expected": (
                self._carrying_chain_id is not None
                and self.state == pick.STATE_DONE),
            "parent_state": self.state,
            "marker_id": self.target_marker_id,
            "selected_arm": self.grasp_arm,
            "dual_arm": bool(self.use_dual_tissue_grasp),
            "measured_gripper": (
                None if measured_gripper is None
                else round(float(measured_gripper), 6)),
            "commanded_gripper": round(float(commanded_gripper), 6),
            "selected_tcp_world": self._diagnostic_vector(
                self.selected_tcp_world()),
        }

    def _on_grab_complete(self) -> None:
        now = self.now()
        self._carrying_chain_id = (
            f"{self.order_id}:marker-{self.target_marker_id}:"
            f"{int(now * 1000)}")
        self._carrying_failure_saved = False
        self._backup_summary = None
        self._carrying_plan_observed = False
        self._carrying_trace.clear()
        self._carrying_trace_last_emit = float("-inf")
        self._carrying_trace_sequence = 0
        self._carrying_trace_best_distance = float("inf")
        self._carrying_trace_last_progress_time = None
        self.telemetry.emit(
            "carrying_chain_start",
            **self._carrying_event_context(),
            parent_state=self.state,
            base_pose=[
                round(float(self.base_xy[0]), 6),
                round(float(self.base_xy[1]), 6),
                round(float(self.base_yaw), 6),
            ],
            selected_arm=self.grasp_arm,
            dual_arm=bool(self.use_dual_tissue_grasp),
            selected_tcp_world=self._diagnostic_vector(
                self.selected_tcp_world()),
            target_world=self._diagnostic_vector(self.target_world),
            payload=self._payload_diagnostic_state(),
            backup_commanded_distance_m=self.backup_after_grab_m)
        self.get_logger().info(
            f"[flow] goods grabbed (marker={self.target_marker_id}, "
            f"kind={self.target_kind}, state={self.state}); "
            "preparing delivery transit")
        if self.backup_after_grab_m > 1e-4:
            self.flow_phase = "backup"
            self._backup_start_xy = self.base_xy.copy()
            self._backup_start_yaw = float(self.base_yaw)
            self._backup_t0 = now
            self._backup_logged = False
            self._backup_event_open = True
            self.telemetry.emit(
                "carrying_backup_start",
                **self._carrying_event_context(),
                commanded_distance_m=self.backup_after_grab_m,
                start_pose=[
                    round(float(self._backup_start_xy[0]), 6),
                    round(float(self._backup_start_xy[1]), 6),
                    round(self._backup_start_yaw, 6),
                ],
                backup_speed_mps=BACKUP_SPEED_MPS,
                timeout_s=BACKUP_TIMEOUT_S)
            self.get_logger().info(
                f"[flow] backing up {self.backup_after_grab_m:.2f}m "
                "before delivery rotation")
            return
        self._backup_summary = {
            "commanded_distance_m": self.backup_after_grab_m,
            "actual_projected_distance_m": 0.0,
            "actual_euclidean_distance_m": 0.0,
            "elapsed_s": 0.0,
            "reached": True,
            "timed_out": False,
            "skipped": True,
        }
        self.telemetry.emit(
            "carrying_backup_end",
            **self._carrying_event_context(),
            **self._backup_summary,
            end_pose=[
                round(float(self.base_xy[0]), 6),
                round(float(self.base_xy[1]), 6),
                round(float(self.base_yaw), 6),
            ])
        self._start_delivery_navigation()

    def _start_delivery_navigation(self) -> None:
        now = self.now()
        self.flow_phase = "nav_to_delivery"
        self._nav_goal = None
        self._nav_last_log = 0.0
        self._last_nav_reason = None
        # Opt in only for the verified payload leg.  All shelf/search transit
        # continues to use the unchanged empty-navigation controller profile.
        self.nav.set_carrying(True)
        self.nav.set_goal(*DELIVERY_APPROACH)
        self._carrying_delivery_started_at = now
        self._carrying_delivery_event_open = True
        initial_distance = math.hypot(
            DELIVERY_APPROACH[0] - float(self.base_xy[0]),
            DELIVERY_APPROACH[1] - float(self.base_xy[1]))
        self._carrying_trace_best_distance = initial_distance
        self._carrying_trace_last_progress_time = now
        self.telemetry.emit(
            "carrying_delivery_request",
            **self._carrying_event_context(),
            start_pose=[
                round(float(self.base_xy[0]), 6),
                round(float(self.base_xy[1]), 6),
                round(float(self.base_yaw), 6),
            ],
            goal_pose=[round(float(value), 6)
                       for value in DELIVERY_APPROACH],
            goal_table_clearance_m=round(point_to_rect_clearance(
                *DELIVERY_APPROACH[:2], DELIVERY_TABLE_COSTMAP_BOUNDS), 6),
            payload=self._payload_diagnostic_state(),
            backup=self._backup_summary)

    def _observe_delivery_plan(self, now: float) -> None:
        if self._carrying_plan_observed:
            return
        self._carrying_plan_observed = True
        try:
            metadata, _ = build_failure_evidence(
                self.nav,
                (float(self.base_xy[0]), float(self.base_xy[1])),
                DELIVERY_APPROACH[:2],
                table_bounds=DELIVERY_TABLE_COSTMAP_BOUNDS)
        except Exception as exc:
            self.telemetry.emit(
                "carrying_delivery_plan_observation_error",
                **self._carrying_event_context(),
                reason=f"{type(exc).__name__}: {exc}")
            return
        controller = self.nav.controller
        self.telemetry.emit(
            "carrying_delivery_plan_observation",
            **self._carrying_event_context(),
            observed_at_s=round(now, 6),
            planned_path_exists=bool(controller.path),
            path_point_count=len(controller.path),
            start=metadata["start"],
            goal=metadata["goal"],
            start_component=metadata["start_component"],
            goal_component=metadata["goal_component"],
            same_component=metadata["same_component"],
            goal_clearance=metadata["goal_clearance"],
            obstacle_counts=metadata["obstacle_counts"],
            planner=metadata["planner"],
            payload=self._payload_diagnostic_state())

    def _record_carrying_controller_trace(
            self, now: float, v: float, w: float, *, force=False) -> None:
        distance = math.hypot(
            DELIVERY_APPROACH[0] - float(self.base_xy[0]),
            DELIVERY_APPROACH[1] - float(self.base_xy[1]))
        if distance < self._carrying_trace_best_distance - 0.06:
            self._carrying_trace_best_distance = distance
            self._carrying_trace_last_progress_time = now
        if self._carrying_trace_last_progress_time is None:
            self._carrying_trace_last_progress_time = now
        stuck_duration = now - self._carrying_trace_last_progress_time
        if not force and now - self._carrying_trace_last_emit < 0.25:
            return
        self._carrying_trace_last_emit = now
        self._carrying_trace_sequence += 1
        trace = build_controller_trace(
            self.nav,
            (float(self.base_xy[0]), float(self.base_xy[1]),
             float(self.base_yaw)),
            DELIVERY_APPROACH,
            (v, w),
            time_now=now,
            stuck_duration_s=stuck_duration,
            payload_state=self._payload_diagnostic_state())
        trace["sequence"] = self._carrying_trace_sequence
        self._carrying_trace.append(trace)
        self.telemetry.emit(
            "carrying_local_controller_trace",
            **self._carrying_event_context(),
            **trace)

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
            euclidean_distance = float(np.linalg.norm(
                self.base_xy - self._backup_start_xy))
            self._backup_summary = {
                "commanded_distance_m": round(
                    self.backup_after_grab_m, 6),
                "actual_projected_distance_m": round(moved_back, 6),
                "actual_euclidean_distance_m": round(
                    euclidean_distance, 6),
                "elapsed_s": round(elapsed, 6),
                "reached": bool(reached),
                "timed_out": bool(timed_out),
                "skipped": False,
            }
            if self._backup_event_open:
                self._backup_event_open = False
                self.telemetry.emit(
                    "carrying_backup_end",
                    **self._carrying_event_context(),
                    **self._backup_summary,
                    start_pose=[
                        round(float(self._backup_start_xy[0]), 6),
                        round(float(self._backup_start_xy[1]), 6),
                        round(float(self._backup_start_yaw), 6),
                    ],
                    end_pose=[
                        round(float(self.base_xy[0]), 6),
                        round(float(self.base_xy[1]), 6),
                        round(float(self.base_yaw), 6),
                    ])
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

    def _capture_carrying_navigation_failure(self, now: float) -> None:
        """Persist one read-only evidence bundle for this carrying chain."""
        if self._carrying_failure_saved:
            return
        if self.carrying_diagnostic_dir is None:
            self.telemetry.emit(
                "carrying_navigation_failure_snapshot_error",
                **self._carrying_event_context(),
                reason="diagnostic_directory_unavailable")
            self._carrying_failure_saved = True
            return

        self._carrying_failure_sequence += 1
        safe_order = "".join(
            character if character.isalnum() or character in "_.-" else "_"
            for character in str(self.order_id))[:80] or "order"
        directory = self.carrying_diagnostic_dir / (
            f"failure_{self._carrying_failure_sequence:02d}_"
            f"{safe_order}_{int(now * 1000)}")
        try:
            metadata, layers = build_failure_evidence(
                self.nav,
                (float(self.base_xy[0]), float(self.base_xy[1])),
                DELIVERY_APPROACH[:2],
                table_bounds=DELIVERY_TABLE_COSTMAP_BOUNDS)
            metadata.update({
                "monotonic_s": round(now, 6),
                "run_prefix": self.run_prefix,
                "order_id": self.order_id,
                "kind": self.target_kind,
                "marker_id": self.target_marker_id,
                "flow_phase": self.flow_phase,
                "parent_state": self.state,
                "carrying_chain_id": self._carrying_chain_id,
                "base_pose": [
                    round(float(self.base_xy[0]), 6),
                    round(float(self.base_xy[1]), 6),
                    round(float(self.base_yaw), 6),
                ],
                "backup": self._backup_summary,
                "payload": self._payload_diagnostic_state(),
                "local_controller_trace": list(self._carrying_trace),
            })
            metadata_path = save_failure_evidence(
                directory, metadata, layers)
        except Exception as exc:  # diagnostics must never change navigation
            self.telemetry.emit(
                "carrying_navigation_failure_snapshot_error",
                **self._carrying_event_context(),
                reason=f"{type(exc).__name__}: {exc}")
            self.get_logger().error(
                "[carrying-diag] failed to save no-path evidence: "
                f"{type(exc).__name__}: {exc}")
            return

        self._carrying_failure_saved = True
        self.telemetry.emit(
            "carrying_navigation_failure_snapshot",
            **self._carrying_event_context(),
            evidence_file=str(metadata_path),
            stop_reason=metadata["planner"]["stop_reason"],
            start_component=metadata["start_component"],
            goal_component=metadata["goal_component"],
            same_component=metadata["same_component"],
            start_nearest_free_displacement_m=metadata["start"].get(
                "nearest_free_displacement_m"),
            goal_nearest_free_displacement_m=metadata["goal"].get(
                "nearest_free_displacement_m"),
            goal_clearance=metadata["goal_clearance"],
            obstacle_counts=metadata["obstacle_counts"])
        self.get_logger().warn(
            "[carrying-diag] saved no-path evidence to "
            f"{metadata_path}")

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
        self._observe_delivery_plan(now)
        failure_stop = (
            ctrl.stop_reason is not None
            and ctrl.stop_reason.startswith(("no_path", "stuck_no_path"))
            and not self._carrying_failure_saved)
        self._record_carrying_controller_trace(
            now, v, w, force=failure_stop)
        if (ctrl.stop_reason is not None
                and ctrl.stop_reason.startswith(
                    ("no_path", "stuck_no_path"))):
            self._capture_carrying_navigation_failure(now)
        if (ctrl.stop_reason is not None
                and ctrl.stop_reason != self._last_nav_reason):
            self._last_nav_reason = ctrl.stop_reason
            self.get_logger().info(
                f"[nav→delivery] stop_reason={ctrl.stop_reason} "
                f"lidar={ctrl.lidar_clearance:.2f}m")
            if ctrl.stop_reason.startswith("no_path"):
                self.telemetry.emit(
                    "no_path", order_id=self.order_id,
                    kind=self.target_kind, phase="navigation_to_delivery",
                    reason=ctrl.stop_reason)
            elif "replan" in ctrl.stop_reason:
                self.telemetry.emit(
                    "replan", order_id=self.order_id,
                    kind=self.target_kind, phase="navigation_to_delivery",
                    reason=ctrl.stop_reason)

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
            if self._carrying_delivery_event_open:
                self._carrying_delivery_event_open = False
                self.telemetry.emit(
                    "carrying_delivery_end",
                    **self._carrying_event_context(),
                    reached=True,
                    elapsed_s=round(
                        0.0 if self._carrying_delivery_started_at is None
                        else now - self._carrying_delivery_started_at, 6),
                    planned_path_exists=bool(ctrl.path),
                    payload=self._payload_diagnostic_state(),
                    end_pose=[
                        round(float(self.base_xy[0]), 6),
                        round(float(self.base_xy[1]), 6),
                        round(float(self.base_yaw), 6),
                    ])
            self.flow_phase = "place"
            self.place_stage = 0
            self.place_t0 = now
            self._place_retry_manager = None
            self._place_arm_sent_t0 = None
            self._place_grip_hold_command = self._selected_grip_command()
            if (not math.isfinite(self._place_grip_hold_command)
                    or self._place_grip_hold_command
                    >= pick.GRIP_OPEN - 1e-6):
                raise RuntimeError(
                    "place start has no closed gripper command; refusing "
                    "to continue with an unsecured payload")
            self.get_logger().info(
                f"[flow] arrived at delivery approach "
                f"pos=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
                f"yaw={math.degrees(self.base_yaw):.0f}°; placing")

    def _set_selected_grip(self, value: float) -> None:
        if self.grasp_arm == "r":
            self.des_right_grip = float(value)
        else:
            self.des_left_grip = float(value)

    def _selected_grip_command(self) -> float:
        return float(
            self.des_right_grip if self.grasp_arm == "r"
            else self.des_left_grip)

    def _hold_place_gripper_closed(self) -> None:
        """Reassert the captured grasp command without changing grip policy."""
        if self._place_grip_hold_command is None:
            self._place_grip_hold_command = self._selected_grip_command()
        command = float(self._place_grip_hold_command)
        if not math.isfinite(command) or command >= pick.GRIP_OPEN - 1e-6:
            raise RuntimeError(
                "place retry lost the closed-gripper command; refusing "
                "to continue")
        self._set_selected_grip(command)

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

    def _compute_place_ik_candidates(self) -> tuple[PlaceIKCandidate, ...]:
        """Solve ordered approaches with enough slide travel for low release.

        The numeric IK depends heavily on the reference joints.  At the
        delivery pose the shelf pregrasp joints are far from any solution, so
        we also try the compact INIT pose and the measured joints.  Once an
        approach pose is found, the final vertical descent keeps the arm joints
        fixed and increases the downward-facing slide joint.  Planner order is
        preserved; the retry manager removes numerically equivalent solutions.
        """
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

        candidates = []
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
                        candidate = PlaceIKCandidate(
                            approach_world_pose=world,
                            arm_joints=joints,
                            slide_target=slide,
                            release_world_pose=(
                                world[0], world[1], release_z),
                            release_slide=release_slide,
                        )
                        if not self._place_candidate_is_safe(candidate):
                            continue
                        candidates.append(candidate)
                        unique = ordered_unique_candidates(candidates)
                        if len(unique) >= PLACE_ARM_RETRY_MAX + 1:
                            self.get_logger().info(
                                f"[place] planned {len(unique)} distinct "
                                "safe IK candidates for fail-closed retry")
                            return unique

        if not candidates:
            self.get_logger().error(
                "[place] no approach IK with enough downward slide travel; "
                "keeping gripper closed")
        unique = ordered_unique_candidates(candidates)
        self.get_logger().info(
            f"[place] planned {len(unique)} distinct safe IK candidates")
        return unique

    def _place_candidate_is_safe(self, candidate: PlaceIKCandidate) -> bool:
        """Validate a candidate against the unchanged release envelope."""
        approach = np.asarray(candidate.approach_world_pose, dtype=float)
        joints = np.asarray(candidate.arm_joints, dtype=float)
        release = np.asarray(candidate.release_world_pose, dtype=float)
        if (not candidate.finite
                or approach.shape != (3,)
                or joints.shape != (6,)
                or release.shape != (3,)):
            return False
        if (not pick.SLIDE_MIN <= candidate.slide_target <= pick.SLIDE_MAX
                or not pick.SLIDE_MIN
                <= candidate.release_slide <= pick.SLIDE_MAX):
            return False
        expected_release_z = self._product_release_z()
        if abs(float(release[2]) - expected_release_z) > 1e-6:
            return False
        if (float(approach[2])
                < float(release[2]) + PLACE_APPROACH_CLEARANCE_M - 1e-6):
            return False
        if candidate.release_slide + 1e-6 < candidate.slide_target:
            return False
        return (
            self._tcp_over_delivery_table(approach)
            and self._tcp_over_delivery_table(release))

    def _activate_place_candidate(
            self, decision: PlaceRetryDecision, now: float) -> None:
        """Stop, hold the payload, and send one freshly validated candidate."""
        candidate = decision.candidate
        if not decision.should_activate or candidate is None:
            raise RuntimeError("invalid place retry activation decision")
        if not self._place_candidate_is_safe(candidate):
            raise RuntimeError(
                "place candidate changed after validation; refusing command")
        self.set_twist(0.0, 0.0)
        self._hold_place_gripper_closed()
        self.place_world = np.asarray(
            candidate.approach_world_pose, dtype=float)
        self.place_arm_joints = np.asarray(candidate.arm_joints, dtype=float)
        self.place_slide_cmd = float(candidate.slide_target)
        self.place_release_world = np.asarray(
            candidate.release_world_pose, dtype=float)
        self.place_release_slide_cmd = float(candidate.release_slide)
        self.set_selected_arm_target(self.place_arm_joints)
        self.des_slide = self.place_slide_cmd
        self.commands_ready_since = None
        self._place_arm_target_sent = True
        self._place_arm_sent_t0 = float(now)
        self.get_logger().info(
            f"[place-retry] activating candidate={decision.candidate_index} "
            f"retry={self._place_retry_manager.retry_count} "
            f"approach={np.round(self.place_world, 3)} "
            f"release={np.round(self.place_release_world, 3)} "
            f"slide={self.place_slide_cmd:.3f}->"
            f"{self.place_release_slide_cmd:.3f}")

    def _start_place_retry(self, now: float) -> None:
        self.set_twist(0.0, 0.0)
        self._hold_place_gripper_closed()
        self._place_retry_manager = PlaceRetryManager(
            self._compute_place_ik_candidates(),
            self._place_candidate_is_safe,
            max_retries=PLACE_ARM_RETRY_MAX)
        decision = self._place_retry_manager.start()
        if not decision.should_activate:
            raise RecoverablePlaceFailure(
                "BLOCKED_AT_PLACE_IK", decision.detail)
        self._activate_place_candidate(decision, now)

    def _retry_place_settle(self, now: float) -> None:
        self.set_twist(0.0, 0.0)
        self._hold_place_gripper_closed()
        decision = self._place_retry_manager.retry(
            PlaceFailureReason.ARM_SETTLE_TIMEOUT)
        if not decision.should_activate:
            raise RecoverablePlaceFailure(
                "BLOCKED_AT_PLACE_SETTLE", decision.detail)
        self.get_logger().warn(
            f"[place-retry] arm failed to settle within "
            f"{PLACE_ARM_SETTLE_TIMEOUT_S:.1f}s; switching to candidate="
            f"{decision.candidate_index} while keeping gripper closed")
        self._activate_place_candidate(decision, now)

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
            self._hold_place_gripper_closed()
            if not self._advance_place_creep():
                return
            if self._place_retry_manager is None:
                self._start_place_retry(now)
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
                self._retry_place_settle(now)
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
        self._publish_worker_phase()
        previous_flow = self.flow_phase
        self._observe_idle_progress(now)
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
            if (not self.candidate_attempt_active
                    and self.state == pick.STATE_GO_SCAN):
                # Skip only runner-authoritative COVERED_VALID observations.
                # This advances the existing route indices; it does not create
                # target_world, authorize motion, or bypass strict localization.
                for _ in range(max(1, len(pick.SCAN_X) * len(self.scan_poses))):
                    key = self._global_coverage_key()
                    if key not in self._covered_scan_keys:
                        break
                    self.scan_pose_index += 1
                    if self.scan_pose_index >= len(self.scan_poses):
                        self.scan_pose_index = 0
                        self.scan_index += 1
                        if self.scan_index >= len(pick.SCAN_X):
                            self.scan_index = 0
                            self.scan_cycles += 1
                    if key not in self._reported_covered_scan_keys:
                        self._reported_covered_scan_keys.add(key)
                        self.telemetry.emit(
                            "covered_pose_reused", order_id=self.order_id,
                            kind=self.target_kind, station_id=key[0],
                            pose_name=key[1], shelf_band=key[2])
                if self.scan_cycles >= self.max_scan_cycles:
                    self.get_logger().error(
                        "all runner-authoritative covered poses exhausted "
                        "without target localization")
                    self.set_state(pick.STATE_ABORT)
                    self._observe_flow_transition(previous_flow)
                    return
            candidate_budget_exhausted = (
                self.candidate_attempt_active
                    and self.candidate_attempt_started_at is not None
                    and now - self.candidate_attempt_started_at
                    >= self.candidate_attempt_budget_s
                    and self.target_world is None)
            if (candidate_budget_exhausted
                    and self._adaptive_observation_budget_available(now)):
                if not self._candidate_budget_extension_emitted:
                    self._candidate_budget_extension_emitted = True
                    started = self.replay_observation_controller.pose_started_at
                    self.telemetry.emit(
                        "candidate_attempt_budget_extension",
                        **self._candidate_event_context(),
                        reason="adaptive_observation_in_progress",
                        pose_elapsed_s=round(
                            0.0 if started is None else now - started, 3),
                        max_observation_budget_s=(
                            self.replay_observation_controller.max_wait_s))
            elif candidate_budget_exhausted:
                self.candidate_timed_out = True
                self.get_logger().warn(
                    "[memory] candidate attempt budget exhausted before "
                    "authoritative localisation")
                self.set_state(pick.STATE_ABORT)
                self._observe_flow_transition(previous_flow)
                return
            prev_state = self.state
            prev_scan_pose_index = self.scan_pose_index
            self._observe_replay_viewpoint_convergence()
            self._apply_replay_observation_policy(now)
            super().tick()
            # A preferred marker authorizes a local re-observation only. If
            # every camera pose at that marker's station has been exhausted
            # without producing the parent's authoritative target_world, the
            # remaining shelf stations cannot succeed because association
            # still accepts only preferred_marker_id. Return control to the
            # runner so a separately observed backup candidate can be tried.
            local_scan_failed = preferred_local_scan_exhausted(
                None if self.candidate_hint is None else
                self.candidate_hint.get("provisional_marker_id"),
                prev_state,
                prev_scan_pose_index,
                len(self.scan_poses),
                self.scan_pose_index,
                self.state,
                self.target_world)
            if local_scan_failed:
                self.get_logger().warn(
                    "[memory] preferred marker local scan exhausted; "
                    f"marker={self.preferred_marker_id} "
                    "failed authoritative localisation, returning to the "
                    "runner for backup-candidate selection")
                self.set_state(pick.STATE_ABORT)
            if (prev_state != pick.STATE_DONE
                    and self.state == pick.STATE_DONE):
                self._on_grab_complete()
            self._observe_flow_transition(previous_flow)
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
        self._observe_flow_transition(previous_flow)
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
    parser.add_argument("--telemetry-file")
    parser.add_argument(
        "--carrying-diagnostic-dir",
        help="directory for read-only carrying no-path costmap evidence; "
             "defaults beside --telemetry-file or --result-file")
    parser.add_argument("--run-prefix", default="manual")
    parser.add_argument("--attempt-id", default="manual-attempt")
    parser.add_argument(
        "--scan-coverage-json",
        help="read-only runner-authoritative run coverage snapshot")
    parser.add_argument(
        "--memory-mode", choices=("off", "run_inventory"), default="off")
    parser.add_argument(
        "--exclude-marker-id", action="append", type=int, default=[],
        help="ignore a marker already delivered or failed in this match")
    parser.add_argument(
        "--preferred-marker-id", type=int,
        help="legacy provisional marker viewpoint hint (never a hard lock)")
    parser.add_argument(
        "--candidate-hint-json",
        help="two-tier provisional candidate and viewpoint replay context")
    parser.add_argument(
        "--candidate-attempt-budget", type=float, default=45.0,
        help="maximum seconds from candidate revisit start to validation")
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
    if args.candidate_attempt_budget <= 0.0:
        parser.error("--candidate-attempt-budget must be positive")
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
    if args.candidate_hint_json:
        try:
            candidate_hint = json.loads(args.candidate_hint_json)
        except json.JSONDecodeError as exc:
            parser.error(f"invalid --candidate-hint-json: {exc}")
        if not isinstance(candidate_hint, dict):
            parser.error("--candidate-hint-json must decode to an object")
    if args.scan_coverage_json:
        try:
            scan_coverage = json.loads(args.scan_coverage_json)
        except json.JSONDecodeError as exc:
            parser.error(f"invalid --scan-coverage-json: {exc}")
        if (not isinstance(scan_coverage, dict)
                or scan_coverage.get("run_prefix") != args.run_prefix):
            parser.error("scan coverage must belong to --run-prefix")
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
    telemetry = EventLog(
        args.telemetry_file,
        run_prefix=args.run_prefix,
        mode=args.memory_mode,
    )
    carrying_diagnostic_dir = args.carrying_diagnostic_dir
    if carrying_diagnostic_dir is None:
        diagnostic_anchor = args.telemetry_file or args.result_file
        if diagnostic_anchor:
            carrying_diagnostic_dir = str(
                pathlib.Path(diagnostic_anchor).parent
                / "carrying_navigation_failures")
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
            close_recheck=not args.no_close_recheck,
            telemetry=telemetry,
            order_id=args.order_id,
            run_prefix=args.run_prefix,
            memory_mode=args.memory_mode,
            candidate_attempt_budget_s=args.candidate_attempt_budget,
            attempt_id=args.attempt_id,
            scan_coverage=(None if not args.scan_coverage_json
                           else json.loads(args.scan_coverage_json)),
            carrying_diagnostic_dir=carrying_diagnostic_dir)
        controller.excluded_marker_ids = set(args.exclude_marker_id)
        if controller.excluded_marker_ids:
            controller.get_logger().info(
                "excluding markers from earlier attempts: "
                f"{sorted(controller.excluded_marker_ids)}")
        candidate_hint = None
        if args.candidate_hint_json:
            candidate_hint = json.loads(args.candidate_hint_json)
        elif args.preferred_marker_id is not None:
            candidate_hint = {
                "schema_version": 2,
                "candidate_id": f"legacy-{args.preferred_marker_id}",
                "kind": args.target_kind,
                "provisional_marker_id": args.preferred_marker_id,
                "hint_source": "DERIVED_VIEW_HINT",
                "scan_station_hint": {
                    "index": len(pick.SCAN_X) - 1
                    - args.preferred_marker_id // 9,
                },
            }
        if candidate_hint is not None:
            if not controller.configure_candidate_hint(candidate_hint):
                raise ValueError("invalid provisional candidate hint")
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
        if controller is not None:
            controller.flush_idle_telemetry()
        delivered = bool(
            controller is not None
            and controller.flow_phase == "done")
        state = None if controller is None else controller.state
        phase = None if controller is None else controller.flow_phase
        marker_id = (
            None if controller is None else controller.target_marker_id)
        candidate_hint = (
            None if controller is None else controller.candidate_hint)
        error = None if delivered else (
            caught_error or f"worker stopped in phase={phase} state={state}")
        _write_result(args.result_file, {
            "schema_version": 1,
            "order_id": args.order_id,
            "kind": args.target_kind,
            "status": "delivered" if delivered else "failed",
            "marker_id": marker_id,
            "candidate_id": (
                None if candidate_hint is None
                else candidate_hint.get("candidate_id")),
            "candidate_state": (
                None if controller is None or candidate_hint is None
                else ("LOCALIZATION_VALIDATED"
                      if controller.validated_marker_id is not None
                      else "INVALIDATED")),
            "provisional_marker_id": (
                None if candidate_hint is None
                else candidate_hint.get("provisional_marker_id")),
            "validated_marker_id": (
                None if controller is None
                else controller.validated_marker_id),
            "validated_target_world": (
                None if controller is None
                else controller.validated_target_world),
            "validated_station_context": (
                None if controller is None
                else controller.validated_station_context),
            "validation_elapsed_s": (
                None if controller is None
                else controller.validation_elapsed_s),
            "candidate_first_failure_stage": (
                None if controller is None
                else controller.candidate_first_failure_stage),
            "candidate_first_failure_reason": (
                None if controller is None
                else controller.candidate_first_failure_reason),
            "execution_failure_stage": (
                None if controller is None
                else controller.execution_failure_stage),
            "execution_failure_reason": (
                None if controller is None
                else controller.execution_failure_reason),
            "raw_fresh_rgb_frame_count": (
                0 if controller is None else len(
                    controller._candidate_raw_rgb_stamps)),
            "yolo_processed_fresh_frame_count": (
                0 if controller is None else len(
                    controller._candidate_fresh_rgb_stamps)),
            "target_kind_detection_count": (
                0 if controller is None else int(
                    controller.candidate_diagnostics.get(
                        "target_kind_detection_count", 0))),
            "replay_pose_ids_attempted": (
                [] if controller is None else sorted(
                    controller._replay_pose_ids_attempted)),
            "fresh_frame_gate_mode": (
                None if controller is None else controller.fresh_frame_gate_mode),
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
