#!/usr/bin/env python3
"""Formal multi-order entry point for the supermarket competition.

The proven single-item controller remains an isolated worker process.  This
node owns the match lifecycle: it receives the transient task, validates it,
selects orders, supervises workers, records results, and continues after an
individual item fails.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import JointState

from candidate_observation_context import (
    CONTEXT_COMPLETE,
    make_observed_context,
    normalize_kind,
)
from candidate_admission_trace import (
    INVENTORY_SYNC_TOLERANCE_NS,
    CandidateAdmissionTrace,
    first_loss_stage,
    nearest_synchronized_frame,
)
from candidate_attempt_memory import (
    CandidateAttemptMemory,
    CandidateAttemptOutcome,
    completion_estimate,
    make_fingerprint,
    reactivation_requirements,
)
from anytime_discovery_policy import (
    AnytimeDiscoveryPolicy,
    DiscoverySegment,
    START_DISCOVERY_SEGMENT,
    stable_segment_id,
)
from run_scan_coverage import (
    CoverageKey,
    RunScanCoverage,
    shelf_band_for_pose,
    stable_attempt_id,
)
from replay_outcome_memory import (
    NO_FRESH_RGB,
    RGB_NOT_PROCESSED_BY_YOLO,
    SUCCEEDED as REPLAY_SUCCEEDED,
    ReplayOutcome,
    ReplayOutcomeMemory,
    classify_fresh_frame_outcome,
    make_equivalence_key,
)
from strict_replay_outcome_memory import (
    BACKUP_AVAILABLE as STRICT_BACKUP_AVAILABLE,
    EXHAUSTED_UNTIL_MATERIAL_CHANGE as STRICT_EXHAUSTED,
    LOCALIZATION_VALIDATED as STRICT_LOCALIZATION_VALIDATED,
    NO_ASSOCIATION as STRICT_NO_ASSOCIATION,
    PREGRASP_ARM_CONVERGENCE_TIMEOUT,
    PROCESS_FAILURE as STRICT_PROCESS_FAILURE,
    SENSOR_INVALID as STRICT_SENSOR_INVALID,
    SPREAD_RECOVERY_AVAILABLE as STRICT_SPREAD_RECOVERY_AVAILABLE,
    SPREAD_REJECT as STRICT_SPREAD_REJECT,
    StrictReplayOutcomeMemory,
    make_equivalence_key as make_strict_equivalence_key,
)

from competition_task import (
    CompetitionTask,
    GRASP_COST,
    TaskMessageError,
    associate_detection_marker,
    marker_arguments,
)
from score_telemetry import (
    EventLog,
    build_summary,
    read_events,
    write_summary_csv,
)
from score_first_order_policy import (
    OrderOption,
    candidate_state as score_first_candidate_state,
    select_order as score_first_select_order,
    stronger_ready_order,
)
from source_state_history import (
    BoundedSourceStateHistory,
    build_candidate_source_state_evidence,
    classify_candidate_outcome,
)
from yolo_aruco_shelf_pick import (
    MIDDLE_SHELF_Z_MIN_M,
    SCAN_CAMERA_POSES,
    SCAN_X,
    SCAN_Y,
    TOP_SHELF_Z_M,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_WORKER = HERE / "integrated_nav_pick_place.py"
DEFAULT_WEIGHTS = HERE / "perception" / "checkpoints" / "best.pt"
PROVISIONAL_VIEW_HINT = "PROVISIONAL_VIEW_HINT"
LOCALIZATION_VALIDATED = "LOCALIZATION_VALIDATED"
INVALIDATED = "INVALIDATED"
DELIVERED = "DELIVERED"
SOFT_DEADLINE_S = 570.0
INFLIGHT_COMPLETION_PHASES = frozenset({
    "GRASPED", "BACKUP", "NAV_TO_DELIVERY", "DELIVERING",
    "PLACE_APPROACH", "PLACING", "PLACE_RETRY", "DONE",
    "DONE_AWAITING_RESULT",
})


def runner_deadline_action(
        elapsed_s: float, phase: str | None, *,
        hard_deadline_s: float) -> str:
    """Apply soft gating only to new/pre-grasp work in the runner."""
    if elapsed_s >= hard_deadline_s:
        return "HARD_STOP"
    if elapsed_s < min(SOFT_DEADLINE_S, hard_deadline_s):
        return "ALLOW_NEW"
    if phase in INFLIGHT_COMPLETION_PHASES:
        return "ALLOW_INFLIGHT"
    return "BLOCK_NEW"


def atomic_write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:96] or "run"


def _record_ros_logger_exception(event_log, exc: Exception, context: str) -> None:
    """Report a ROS logger failure without using ROS logging recursively."""
    try:
        event_log.emit(
            "ros_logger_exception",
            ros_logger_exception_type=type(exc).__name__,
            ros_logger_exception_message=str(exc),
            ros_logger_exception_context=context,
            ros_logger_fallback_used=True,
        )
    except Exception:
        pass
    try:
        sys.stderr.write(
            "ROS logger failure contained "
            f"context={context} type={type(exc).__name__}: {exc}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _safe_log_info(logger, message: str, *, event_log, context: str) -> bool:
    """Best-effort INFO at a fixed-severity rclpy callsite."""
    try:
        logger.info(message)
        return True
    except Exception as exc:
        _record_ros_logger_exception(event_log, exc, context)
        return False


def _safe_log_error(logger, message: str, *, event_log, context: str) -> bool:
    """Best-effort ERROR at a fixed-severity rclpy callsite."""
    try:
        logger.error(message)
        return True
    except Exception as exc:
        _record_ros_logger_exception(event_log, exc, context)
        return False


def derive_candidate_view_hint(position_world) -> dict:
    """Derive a replay station/pose only from public marker perception."""
    try:
        x, _, z = map(float, position_world[:3])
    except (TypeError, ValueError, IndexError):
        x, z = 0.0, MIDDLE_SHELF_Z_MIN_M
    station_index = min(
        range(len(SCAN_X)), key=lambda index: abs(SCAN_X[index] - x))
    if z >= TOP_SHELF_Z_M:
        pose_index = 0
    elif z >= MIDDLE_SHELF_Z_MIN_M:
        pose_index = 2
    else:
        pose_index = 3
    pose = SCAN_CAMERA_POSES[pose_index]
    return {
        "observation_base_pose_hint": None,
        "head_pose_hint": list(pose),
        "scan_station_hint": {
            "index": station_index,
            "x": float(SCAN_X[station_index]),
            "y": float(SCAN_Y),
        },
        "scan_pitch_hint": float(pose[3]),
        "hint_source": "DERIVED_VIEW_HINT",
    }


class CompetitionRunner(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("supermarket_competition_runner")
        self.args = args
        self.task: CompetitionTask | None = None
        self.task_started_at: float | None = None
        self.worker: subprocess.Popen | None = None
        self.worker_started_at: float | None = None
        self.worker_result_path: Path | None = None
        self.current_order = None
        self.worker_stop_reason: str | None = None
        self.worker_terminate_at: float | None = None
        self.preferred_marker_id: int | None = None
        self.current_candidate: dict | None = None
        self.current_attempt_fingerprint = None
        self.candidate_attempt_memory = CandidateAttemptMemory()
        self.replay_outcome_memory: ReplayOutcomeMemory | None = None
        self.strict_failure_outcome_memory: StrictReplayOutcomeMemory | None = None
        self.fallback_rescan_open = False
        self.finished = False
        self.latest_yolo: tuple[int, list[dict]] | None = None
        self.latest_aruco: tuple[int, list[dict]] | None = None
        self.inventory_yolo_frames = deque(maxlen=30)
        self.inventory_aruco_frames = deque(maxlen=120)
        self.inventory_processed_yolo_stamps: set[int] = set()
        self.latest_base_pose: tuple[float, float, float] | None = None
        self.latest_head_pose: tuple[float, float, float] | None = None
        self.source_state_history = BoundedSourceStateHistory()
        self.candidate_source_evidence: dict[str, object] = {}
        self.last_inventory_pair: tuple[int, int] | None = None
        self.last_inventory_yolo_stamp: int | None = None
        self.inventory: dict[int, dict] = {}
        self.scan_coverage: set[int] = set()
        self.run_dir: Path | None = None
        self.event_file: Path | None = None
        self.event_log = EventLog(None)
        self.task_document: dict | None = None
        self.current_order_sequence = 0
        self.last_selection: dict = {}
        self.hint_sent_for_order: str | None = None
        self.deferred_orders: dict[str, dict] = {}
        self.orders_deferred_once: set[str] = set()
        self.run_coverage: RunScanCoverage | None = None
        self.attempt_started_ids: set[str] = set()
        self.attempt_terminal_ids: set[str] = set()
        self.worker_started_count = 0
        self.current_attempt_id: str | None = None
        self.worker_runtime_phase: str | None = None
        self.task_generation = 0
        self.current_worker_binding: dict | None = None
        self.terminal_result_fingerprints: dict[str, str] = {}
        self.terminal_outcome_accepted_ids: set[str] = set()
        self.terminal_outcome_rejections: set[tuple[str, str, str]] = set()
        self.soft_deadline_emitted = False
        self.inflight_allowed_emitted = False
        self.hard_deadline_emitted = False
        self.new_attempt_blocked_emitted = False
        self.retry_suppression_events: set[str] = set()
        self.candidate_reactivation_events: set[str] = set()
        self.strict_suppression_events: set[str] = set()
        self.strict_recovery_attempt_events: set[str] = set()
        self.strict_failure_memory_mode = os.environ.get(
            "SUPERMARKET_STRICT_FAILURE_MEMORY_MODE", "shadow"
        ).strip().lower()
        if self.strict_failure_memory_mode not in {"off", "shadow", "control"}:
            raise ValueError(
                "SUPERMARKET_STRICT_FAILURE_MEMORY_MODE must be off, shadow, "
                "or control")
        self.anytime_discovery_policy = AnytimeDiscoveryPolicy()
        self.active_discovery_segment: dict | None = None
        self.discovery_segment_candidate_ids_before: set[str] = set()
        self.coverage_mode = os.environ.get(
            "SUPERMARKET_SCAN_COVERAGE_MODE", "shadow").strip().lower()
        if self.coverage_mode == "authoritative":
            # Backward-compatible spelling; R10 exposes only resume_only.
            self.coverage_mode = "resume_only"
        if self.coverage_mode not in {"off", "shadow", "resume_only"}:
            raise ValueError(
                "SUPERMARKET_SCAN_COVERAGE_MODE must be off, shadow, or "
                "resume_only")
        self.candidate_funnel: CandidateAdmissionTrace | None = None

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, "/supermarket_sorting/task", self._task_cb, qos)
        self.create_subscription(
            String, "/goods/yolo_detections", self._yolo_cb, 10)
        self.create_subscription(
            String, "/aruco/head/detections", self._aruco_cb, 10)
        self.create_subscription(
            Odometry, "/slamware_ros_sdk_server_node/odom",
            self._odom_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(
            String, "/supermarket_sorting/scan_progress",
            self._scan_progress_cb, 50)
        self.create_subscription(
            String, "/supermarket_sorting/worker_progress",
            self._worker_progress_cb, 20)
        self.stop_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.memory_hint_publisher = self.create_publisher(
            String, "/supermarket_sorting/memory_hint", 10)
        self.create_timer(0.20, self._tick)
        self.get_logger().info(
            "competition runner ready; waiting for transient task on "
            "/supermarket_sorting/task")

    def _task_cb(self, message: String) -> None:
        try:
            incoming = CompetitionTask.from_json(message.data)
        except TaskMessageError as exc:
            self.get_logger().error(f"rejecting invalid task: {exc}")
            self._publish_stop()
            return

        if self.task is not None and incoming.run_prefix == self.task.run_prefix:
            return
        if self.worker is not None:
            self.get_logger().warn(
                "new run_prefix received while a worker is active; stopping "
                "the old match before accepting the new task")
            self.current_order = None
            self._request_worker_stop("server_restart")

        self.task = incoming
        self.task_started_at = time.monotonic()
        self.finished = False
        self.inventory.clear()
        self.scan_coverage.clear()
        self.latest_yolo = None
        self.latest_aruco = None
        self.inventory_yolo_frames.clear()
        self.inventory_aruco_frames.clear()
        self.inventory_processed_yolo_stamps.clear()
        self.latest_base_pose = None
        self.latest_head_pose = None
        self.source_state_history.clear()
        self.candidate_source_evidence.clear()
        self.last_inventory_pair = None
        self.last_inventory_yolo_stamp = None
        self.current_order_sequence = 0
        self.hint_sent_for_order = None
        self.current_candidate = None
        self.current_attempt_fingerprint = None
        self.candidate_attempt_memory = CandidateAttemptMemory()
        self.replay_outcome_memory = ReplayOutcomeMemory(incoming.run_prefix)
        self.strict_failure_outcome_memory = StrictReplayOutcomeMemory(
            incoming.run_prefix)
        self.fallback_rescan_open = False
        self.deferred_orders.clear()
        self.orders_deferred_once.clear()
        route = [
            (station, str(pose[0]), shelf_band_for_pose(str(pose[0])))
            for station in range(len(SCAN_X))
            for pose in SCAN_CAMERA_POSES
        ]
        self.run_coverage = RunScanCoverage(incoming.run_prefix, route)
        self.attempt_started_ids.clear()
        self.attempt_terminal_ids.clear()
        self.worker_started_count = 0
        self.current_attempt_id = None
        self.worker_runtime_phase = None
        self.task_generation += 1
        self.current_worker_binding = None
        self.terminal_result_fingerprints.clear()
        self.terminal_outcome_accepted_ids.clear()
        self.terminal_outcome_rejections.clear()
        self.soft_deadline_emitted = False
        self.inflight_allowed_emitted = False
        self.hard_deadline_emitted = False
        self.new_attempt_blocked_emitted = False
        self.retry_suppression_events.clear()
        self.candidate_reactivation_events.clear()
        self.strict_suppression_events.clear()
        self.strict_recovery_attempt_events.clear()
        self.active_discovery_segment = None
        getattr(self, "discovery_segment_candidate_ids_before", set()).clear()
        self.candidate_funnel = CandidateAdmissionTrace(
            incoming.run_prefix, max_poses=8)
        self.task_document = {
            "schema_version": incoming.schema_version,
            "run_prefix": incoming.run_prefix,
            "count": len(incoming.orders),
            "targets": [
                {"id": order.id, "kind": order.kind}
                for order in incoming.orders
            ],
        }
        self.run_dir = (
            Path(self.args.runtime_dir) / safe_component(incoming.run_prefix))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.event_file = self.run_dir / "events.jsonl"
        if self.event_file.exists():
            self.event_file.unlink()
        self.event_log = EventLog(
            self.event_file,
            run_prefix=incoming.run_prefix,
            mode=self.args.memory_mode,
        )
        atomic_write_json(self.run_dir / "task.json", self.task_document)
        self.event_log.emit(
            "match_start",
            product_seed=self.args.product_seed,
            obstacle_seed=self.args.obstacle_seed,
            task_kinds=[order.kind for order in incoming.orders],
            strict_failure_memory_mode=self.strict_failure_memory_mode,
        )
        self.event_log.emit("task_received", task=self.task_document)
        self.get_logger().info(
            f"accepted task run={incoming.run_prefix} "
            f"count={len(incoming.orders)} kinds="
            f"{[order.kind for order in incoming.orders]}")
        self._write_summary("accepted")

    def _tick(self) -> None:
        if self.finished:
            self._publish_stop()
            return

        now = time.monotonic()
        elapsed = (0.0 if self.task_started_at is None
                   else now - self.task_started_at)
        deadline = runner_deadline_action(
            elapsed, self.worker_runtime_phase,
            hard_deadline_s=self.args.match_timeout)
        match_expired = deadline == "HARD_STOP"
        if self.worker is not None:
            # Terminal observation is deliberately first in the tick.  Soft
            # deadline state may gate subsequent work, never this outcome.
            return_code = self.worker.poll()
            result = self._read_worker_result()
            inspected = (
                {"valid": False, "reason": "worker_deferred_nonterminal"}
                if self.worker_stop_reason == "score_first_defer" else
                self._inspect_terminal_result(result))
            if inspected.get("valid"):
                # The worker writes its result atomically before its final ROS
                # teardown.  Preserve the active identity while that teardown
                # completes and never reclassify it as pre-grasp work.
                self.worker_runtime_phase = "DONE_AWAITING_RESULT"
                deadline = runner_deadline_action(
                    elapsed, self.worker_runtime_phase,
                    hard_deadline_s=self.args.match_timeout)
                match_expired = deadline == "HARD_STOP"
            if elapsed >= min(SOFT_DEADLINE_S, self.args.match_timeout):
                if not self.soft_deadline_emitted:
                    self.soft_deadline_emitted = True
                    self.event_log.emit(
                        "soft_deadline_reached", elapsed_s=round(elapsed, 3),
                        worker_phase=self.worker_runtime_phase)
            if return_code is not None:
                self._finish_worker(
                    return_code, result=result, inspected=inspected)
            elif (match_expired and inspected.get("valid")
                  and inspected.get("completion_s") is not None
                  and inspected["completion_s"] <= self.args.match_timeout):
                # One final nonblocking result poll wins over hard-stop.  Only
                # a result authoritatively completed by the boundary can score.
                self._finish_worker(
                    0 if result.get("status") == "delivered" else 1,
                    result=result, inspected=inspected)
            elif match_expired:
                if not self.hard_deadline_emitted:
                    self.hard_deadline_emitted = True
                    self.event_log.emit(
                        "hard_deadline_reached", elapsed_s=round(elapsed, 3),
                        final_safe_state="safe_stop_requested")
                self._request_worker_stop("match_timeout")
            elif deadline == "ALLOW_INFLIGHT":
                if not self.inflight_allowed_emitted:
                    self.inflight_allowed_emitted = True
                    self.event_log.emit(
                        "inflight_completion_allowed",
                        attempt_id=self.current_attempt_id,
                        order_id=(None if self.current_order is None
                                  else self.current_order.id),
                        phase=self.worker_runtime_phase)
            elif deadline == "BLOCK_NEW":
                if not self.new_attempt_blocked_emitted:
                    self.new_attempt_blocked_emitted = True
                    self.event_log.emit(
                        "new_attempt_blocked",
                        attempt_id=self.current_attempt_id,
                        order_id=(None if self.current_order is None
                                  else self.current_order.id),
                        reason="soft_deadline_pre_grasp")
                self._request_worker_stop("soft_deadline_new_work_blocked")
            elif (self.worker_terminate_at is not None
                  and now - self.worker_terminate_at >= 3.0):
                self.get_logger().error("worker ignored SIGTERM; sending SIGKILL")
                self.worker.kill()
            elif (self.args.order_timeout > 0.0
                  and self.worker_started_at is not None
                  and now - self.worker_started_at >= self.args.order_timeout):
                self._request_worker_stop("order_timeout")
            return

        if elapsed >= min(SOFT_DEADLINE_S, self.args.match_timeout):
            if not self.soft_deadline_emitted:
                self.soft_deadline_emitted = True
                self.event_log.emit(
                    "soft_deadline_reached", elapsed_s=round(elapsed, 3),
                    worker_phase=self.worker_runtime_phase)
        if self.task is None:
            self._publish_stop()
            return
        if match_expired:
            self.get_logger().error("match hard deadline reached; stopping safely")
            self._finish_match("match_timeout")
            return
        if deadline == "BLOCK_NEW":
            if not self.new_attempt_blocked_emitted:
                self.new_attempt_blocked_emitted = True
                self.event_log.emit(
                    "new_attempt_blocked",
                    reason="soft_deadline_no_active_worker")
            self._finish_match("soft_deadline_no_new_attempt")
            return

        order, preferred_marker = self._select_order()
        if order is None:
            reason = self.last_selection.get("no_selection_reason", "orders_terminal")
            self._finish_match(reason)
            return
        self._start_worker(order, preferred_marker)

    def _start_worker(self, order, preferred_marker: int | None) -> None:
        assert self.task is not None
        run_dir = (
            Path(self.args.runtime_dir)
            / safe_component(self.task.run_prefix))
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / (
            f"order_{order.source_index + 1:02d}_attempt_"
            f"{order.attempts + 1}.json")
        if result_path.exists():
            result_path.unlink()

        command = [
            sys.executable,
            str(Path(self.args.worker).resolve()),
            "--target-kind", order.kind,
            "--order-id", order.id,
            "--weights", str(Path(self.args.weights).resolve()),
            "--confidence", str(self.args.confidence),
            "--max-scan-cycles", str(self.args.max_scan_cycles),
            "--result-file", str(result_path),
            "--formal-mode",
            "--memory-mode", self.args.memory_mode,
            "--run-prefix", self.task.run_prefix,
            "--candidate-attempt-budget",
            str(self.args.candidate_attempt_budget),
        ]
        attempt_ordinal = order.attempts + 1
        if preferred_marker is None and self.coverage_mode == "resume_only":
            attempt_ordinal = self.worker_started_count + 1
        attempt_id = stable_attempt_id(
            self.task.run_prefix, order.id, attempt_ordinal)
        command.extend(["--attempt-id", attempt_id])
        if (preferred_marker is None and self.run_coverage is not None
                and self.coverage_mode == "resume_only"):
            snapshot = self.run_coverage.snapshot()
            command.extend([
                "--scan-coverage-json",
                json.dumps(snapshot, separators=(",", ":")),
            ])
        if self.event_file is not None:
            command.extend(["--telemetry-file", str(self.event_file)])
        command.extend(marker_arguments(self.task.excluded_markers(order.kind)))
        if preferred_marker is not None:
            candidate = self.inventory.get(preferred_marker)
            if candidate is not None:
                command.extend([
                    "--candidate-hint-json",
                    json.dumps(candidate, separators=(",", ":")),
                ])
        if self.args.show:
            command.append("--show")

        self.current_order = order
        self.current_order_sequence += 1
        sequence = self.current_order_sequence
        self.preferred_marker_id = preferred_marker
        self.current_candidate = (
            None if preferred_marker is None
            else self.inventory.get(preferred_marker))
        if preferred_marker is None:
            self.active_discovery_segment = self.last_selection.get(
                "selected_discovery_segment")
            self.discovery_segment_candidate_ids_before = {
                str(entry.get("candidate_id"))
                for entry in self.inventory.values()
                if entry.get("candidate_id") is not None}
        else:
            self.active_discovery_segment = None
            getattr(
                self, "discovery_segment_candidate_ids_before", set()).clear()
        self.current_attempt_fingerprint = make_fingerprint(
            run_prefix=self.task.run_prefix, order_id=order.id,
            product_kind=order.kind, candidate=self.current_candidate,
            coverage_revision=len(self.scan_coverage))
        self.worker_result_path = result_path
        self.worker_started_at = time.monotonic()
        self.current_attempt_id = attempt_id
        self.worker_stop_reason = None
        self.worker_terminate_at = None
        self.hint_sent_for_order = order.id if preferred_marker is not None else None
        deferred = self.deferred_orders.pop(order.id, None)
        if deferred is not None:
            self.event_log.emit(
                "order_reactivated",
                order_id=order.id,
                kind=order.kind,
                reason="new_or_best_available_evidence",
                previous_defer=deferred,
            )
        selection = dict(self.last_selection)
        self.event_log.emit(
            "order_selected",
            order_id=order.id,
            kind=order.kind,
            order_sequence=sequence,
            preferred_marker_id=preferred_marker,
            selection=selection,
        )
        memory_event = (
            "target_memory_hit" if preferred_marker is not None
            else "target_memory_miss")
        self.event_log.emit(
            memory_event,
            order_id=order.id,
            kind=order.kind,
            order_sequence=sequence,
            marker_id=preferred_marker,
            source="confirmed_run_inventory" if preferred_marker is not None
            else "worker_full_scan",
        )
        self.event_log.emit(
            "search_start", order_id=order.id, kind=order.kind,
            order_sequence=sequence)
        self.event_log.emit(
            "navigation_to_shelf_start", order_id=order.id,
            kind=order.kind, order_sequence=sequence)
        if preferred_marker is None:
            self.event_log.emit(
                "full_scan_start", order_id=order.id, kind=order.kind,
                order_sequence=sequence)
            self.fallback_rescan_open = (
                self.args.memory_mode == "run_inventory"
                and any(
                    entry.get("kind") == order.kind
                    and entry.get("state") == INVALIDATED
                    for entry in self.inventory.values()))
            if self.fallback_rescan_open:
                self.event_log.emit(
                    "fallback_rescan_start", order_id=order.id,
                    kind=order.kind, order_sequence=sequence)
        elif preferred_marker in self.inventory:
            self.inventory[preferred_marker]["reserved_order_id"] = order.id
        self.get_logger().info(
            f"starting order id={order.id} kind={order.kind} "
            f"attempt={order.attempts + 1}/{self.args.max_attempts} "
            f"preferred_marker={preferred_marker} "
            f"excluded_markers={self.task.excluded_markers(order.kind)}")
        try:
            self.worker = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=os.environ.copy(),
                start_new_session=False,
            )
            self.current_worker_binding = {
                "run_prefix": self.task.run_prefix,
                "generation": self.task_generation,
                "order_id": order.id,
                "attempt_id": attempt_id,
                "worker_pid": self.worker.pid,
                "result_path": str(result_path.resolve()),
            }
            self.worker_started_count += 1
            if attempt_id not in self.attempt_started_ids:
                self.attempt_started_ids.add(attempt_id)
                self.event_log.emit(
                    "attempt_started", attempt_id=attempt_id,
                    order_id=order.id, kind=order.kind,
                    order_sequence=sequence,
                    worker_pid=self.worker.pid)
            retry_decision = self.candidate_attempt_memory.decision(
                self.current_attempt_fingerprint)
            strict_key, strict_decision = self._strict_key_and_decision(
                order, self.current_candidate)
            if strict_key is not None and strict_decision is not None:
                self.event_log.emit(
                    "strict_candidate_selected", attempt_id=attempt_id,
                    order_id=order.id, kind=order.kind,
                    candidate_id=self.current_candidate.get("candidate_id"),
                    strict_failure_equivalence_key=strict_key.as_dict(),
                    strict_failure_equivalence_digest=strict_key.digest,
                    strict_failure_material_revision=(
                        strict_decision.material_revision),
                    strict_failure_outcome=(
                        self.strict_failure_outcome_memory.last_outcome(
                            strict_key)),
                    strict_failure_memory_state=strict_decision.state,
                    strict_recovery_slot=strict_decision.recovery_slot,
                    strict_retry_allowed=strict_decision.allowed,
                    strict_retry_suppressed=False,
                    strict_reactivation_reason=(
                        strict_decision.reactivation_reason),
                    strict_candidate_selection_reason=strict_decision.reason,
                    strict_failure_memory_mode=(
                        self.strict_failure_memory_mode),
                    authoritative=(
                        self.strict_failure_memory_mode == "control"))
                if strict_decision.state == STRICT_SPREAD_RECOVERY_AVAILABLE:
                    if attempt_id not in self.strict_recovery_attempt_events:
                        self.strict_recovery_attempt_events.add(attempt_id)
                        self.event_log.emit(
                            "strict_spread_recovery_attempt",
                            attempt_id=attempt_id, order_id=order.id,
                            kind=order.kind,
                            candidate_id=self.current_candidate.get(
                                "candidate_id"),
                            strict_failure_equivalence_digest=(
                                strict_key.digest),
                            strict_recovery_slot=(
                                strict_decision.recovery_slot))
                if (self.strict_failure_memory_mode == "shadow"
                        and not strict_decision.allowed):
                    self.event_log.emit(
                        "strict_equivalent_retry_started",
                        attempt_id=attempt_id, order_id=order.id,
                        kind=order.kind,
                        candidate_id=self.current_candidate.get("candidate_id"),
                        strict_failure_equivalence_digest=strict_key.digest,
                        strict_failure_outcome=(
                            self.strict_failure_outcome_memory.last_outcome(
                                strict_key)),
                        strict_failure_memory_state=strict_decision.state,
                        authoritative=False)
            deadline_feasible = bool(
                self.last_selection.get("deadline_feasible", True))
            self.event_log.emit(
                "candidate_attempt_fingerprint", attempt_id=attempt_id,
                order_id=order.id, kind=order.kind,
                fingerprint_digest=self.current_attempt_fingerprint.digest,
                fingerprint=self.current_attempt_fingerprint.as_dict(),
                fingerprint_status=(
                    "NEW_EVIDENCE" if retry_decision.new_evidence
                    else "UNTRIED"),
                deadline_feasible=deadline_feasible,
                new_evidence_since_previous_attempt=(
                    retry_decision.new_evidence),
                reactivation_reason=retry_decision.reason)
            if retry_decision.new_evidence:
                self.event_log.emit(
                    "candidate_new_evidence", attempt_id=attempt_id,
                    order_id=order.id, kind=order.kind,
                    candidate_id=self.current_attempt_fingerprint.candidate_id,
                    fingerprint_digest=self.current_attempt_fingerprint.digest,
                    reason=retry_decision.reason)
                self.event_log.emit(
                    "candidate_reactivated", attempt_id=attempt_id,
                    order_id=order.id, kind=order.kind,
                    candidate_id=self.current_attempt_fingerprint.candidate_id,
                    reactivation_reason=retry_decision.reason)
            self.event_log.emit(
                "candidate_deadline_feasibility", attempt_id=attempt_id,
                order_id=order.id, kind=order.kind,
                candidate_id=self.current_attempt_fingerprint.candidate_id,
                fingerprint_digest=self.current_attempt_fingerprint.digest,
                deadline_feasible=deadline_feasible,
                estimated_completion_s=self.last_selection.get(
                    "estimated_completion_s"),
                remaining_hard_s=self.last_selection.get(
                    "remaining_match_time_s"),
                deadline_slack_s=self.last_selection.get("deadline_slack_s"),
                estimate_sample_count=completion_estimate(
                    self.current_candidate is not None).estimate_sample_count,
                estimate_source=completion_estimate(
                    self.current_candidate is not None).estimate_source)
            self.event_log.emit(
                "worker_started", attempt_id=attempt_id,
                order_id=order.id, kind=order.kind,
                worker_pid=self.worker.pid)
        except OSError as exc:
            self.get_logger().error(f"cannot start order worker: {exc}")
            self.task.finish_attempt(
                order,
                delivered=False,
                error=f"worker_start: {exc}",
                max_attempts=self.args.max_attempts,
            )
            self.current_order = None
            self.preferred_marker_id = None
            self.current_candidate = None
            self.current_attempt_fingerprint = None
            self.worker_started_at = None
            self.worker_result_path = None
            self._write_summary("worker_start_failed")
            self._publish_stop()

    def _request_worker_stop(self, reason: str) -> None:
        if self.worker is None or self.worker.poll() is not None:
            return
        if self.worker_terminate_at is None:
            self.worker_stop_reason = reason
            self.worker_terminate_at = time.monotonic()
            self.get_logger().error(f"stopping worker: {reason}")
            self._publish_stop()
            self.worker.terminate()

    def _read_worker_result(self) -> dict:
        if self.worker_result_path is None or not self.worker_result_path.exists():
            return {}
        try:
            value = json.loads(self.worker_result_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"cannot read worker result: {exc}")
            return {}

    def _terminal_completion_s(self, result: dict) -> float | None:
        for key in ("terminal_result_completion_s", "completion_s"):
            try:
                value = float(result[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        try:
            absolute = float(result["completion_monotonic_s"])
        except (KeyError, TypeError, ValueError):
            absolute = float("nan")
        if math.isfinite(absolute) and self.task_started_at is not None:
            return max(0.0, absolute - self.task_started_at)
        if (result.get("status") == "delivered" and self.event_file is not None
                and self.task_started_at is not None
                and self.current_order is not None):
            candidates = [
                event for event in read_events(self.event_file)
                if event.get("event") == "place_end"
                and event.get("success") is True
                and event.get("order_id") == self.current_order.id
                and (self.worker_started_at is None or float(
                    event.get("monotonic_s", 0.0)) >= self.worker_started_at)
            ]
            if candidates:
                return max(0.0, float(candidates[-1]["monotonic_s"])
                           - self.task_started_at)
        return None

    def _emit_terminal_rejection(
            self, attempt_id: str | None, reason: str,
            fingerprint: str = "") -> None:
        key = (str(attempt_id), reason, fingerprint)
        rejections = getattr(self, "terminal_outcome_rejections", None)
        if rejections is None:
            self.terminal_outcome_rejections = set()
            rejections = self.terminal_outcome_rejections
        if key in rejections:
            return
        rejections.add(key)
        self.event_log.emit(
            "terminal_outcome_rejected", attempt_id=attempt_id,
            order_id=(None if self.current_order is None
                      else self.current_order.id),
            terminal_outcome_rejected_reason=reason)

    def _inspect_terminal_result(self, result: dict) -> dict:
        if not result:
            return {"valid": False, "reason": "terminal_result_absent"}
        attempt_id = getattr(self, "current_attempt_id", None)
        fingerprint = json.dumps(
            result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        observed = getattr(self, "terminal_result_fingerprints", None)
        if observed is None:
            self.terminal_result_fingerprints = {}
            observed = self.terminal_result_fingerprints
        previous = observed.get(str(attempt_id))
        if previous is not None and previous != fingerprint:
            reason = "same_attempt_conflicting_result"
            self._emit_terminal_rejection(attempt_id, reason, fingerprint)
            return {"valid": False, "reason": reason,
                    "fingerprint": fingerprint}
        first_observation = previous is None
        observed[str(attempt_id)] = fingerprint

        binding = getattr(self, "current_worker_binding", None)
        order = self.current_order
        task = self.task
        reason = None
        if binding is None or attempt_id is None or order is None or task is None:
            reason = "no_active_attempt_binding"
        elif binding.get("generation") != getattr(self, "task_generation", 0):
            reason = "prior_generation_result"
        elif (binding.get("run_prefix") != task.run_prefix
              or result.get("run_prefix", task.run_prefix) != task.run_prefix):
            reason = "stale_run_result"
        elif (binding.get("attempt_id") != attempt_id
              or result.get("attempt_id", attempt_id) != attempt_id):
            reason = "stale_attempt_result"
        elif (binding.get("order_id") != order.id
              or result.get("order_id") != order.id):
            reason = "stale_order_result"
        elif result.get("generation", binding["generation"]) != binding["generation"]:
            reason = "prior_generation_result"
        elif (result.get("worker_pid", binding["worker_pid"])
              != binding["worker_pid"]):
            reason = "stale_worker_result"
        elif result.get("status") not in {"delivered", "failed"}:
            reason = "invalid_terminal_status"

        completion_s = self._terminal_completion_s(result)
        if first_observation:
            self.event_log.emit(
                "terminal_result_observed", attempt_id=attempt_id,
                order_id=None if order is None else order.id,
                terminal_result_status=result.get("status"),
                terminal_result_completion_s=completion_s)
        if reason is not None:
            self._emit_terminal_rejection(attempt_id, reason, fingerprint)
            return {"valid": False, "reason": reason,
                    "fingerprint": fingerprint,
                    "completion_s": completion_s}
        return {"valid": True, "fingerprint": fingerprint,
                "completion_s": completion_s}

    def _strict_key_and_decision(self, order, candidate: dict | None):
        memory = getattr(self, "strict_failure_outcome_memory", None)
        if memory is None or candidate is None or self.task is None:
            return None, None
        key = make_strict_equivalence_key(
            run_prefix=self.task.run_prefix, kind=order.kind,
            marker_id=int(candidate["marker_id"]), candidate=candidate)
        return key, memory.decision(key)

    def _emit_strict_suppression(
            self, *, order_id: str, kind: str, candidate: dict,
            key, decision) -> None:
        event_key = f"{key.digest}:{decision.state}"
        if event_key in self.strict_suppression_events:
            return
        self.strict_suppression_events.add(event_key)
        self.event_log.emit(
            "strict_retry_suppressed",
            order_id=order_id, kind=kind,
            candidate_id=candidate.get("candidate_id"),
            strict_failure_equivalence_key=key.as_dict(),
            strict_failure_equivalence_digest=key.digest,
            strict_failure_material_revision=decision.material_revision,
            strict_failure_outcome=(
                self.strict_failure_outcome_memory.last_outcome(key)),
            strict_failure_memory_state=decision.state,
            strict_recovery_slot=decision.recovery_slot,
            strict_retry_allowed=False,
            strict_retry_suppressed=True,
            strict_reactivation_reason=decision.reactivation_reason,
            strict_candidate_selection_reason=decision.reason,
            actual_suppressed_elapsed_s=0.0,
            offline_counterfactual_estimate_s=(
                completion_estimate(True).estimated_completion_s),
            authoritative=True)
        if decision.state == STRICT_EXHAUSTED:
            self.event_log.emit(
                "strict_exhausted_context_skipped",
                order_id=order_id, kind=kind,
                candidate_id=candidate.get("candidate_id"),
                strict_failure_equivalence_digest=key.digest,
                strict_failure_material_revision=decision.material_revision)

    def _record_strict_outcome(
            self, *, result: dict, order, attempt_id: str,
            delivered: bool, process_failure: bool, sensor_invalid: bool,
            failure_stage: str, failure_reason: str,
            validated_marker_id: int | None) -> None:
        candidate = self.current_candidate
        key, before = self._strict_key_and_decision(order, candidate)
        memory = getattr(self, "strict_failure_outcome_memory", None)
        if key is None or before is None or memory is None:
            return
        execution_reason = str(result.get("execution_failure_reason") or "")
        if process_failure:
            outcome = STRICT_PROCESS_FAILURE
        elif sensor_invalid:
            outcome = STRICT_SENSOR_INVALID
        elif execution_reason == "pregrasp_arm_convergence_timeout":
            outcome = PREGRASP_ARM_CONVERGENCE_TIMEOUT
        elif validated_marker_id is not None:
            outcome = STRICT_LOCALIZATION_VALIDATED
        elif failure_reason == "no_association":
            outcome = STRICT_NO_ASSOCIATION
        elif failure_reason == "spread_reject":
            outcome = STRICT_SPREAD_REJECT
        else:
            # Grasp/navigation/place outcomes are outside strict localization
            # memory and must not be relabelled as association failures.
            return
        after = memory.record(
            key, attempt_id=str(attempt_id), outcome=outcome,
            pose_ids=result.get("replay_pose_ids_attempted") or ())
        if (before.state == STRICT_SPREAD_RECOVERY_AVAILABLE
                and outcome == STRICT_LOCALIZATION_VALIDATED):
            self.event_log.emit(
                "strict_spread_recovery_success", attempt_id=attempt_id,
                order_id=order.id, kind=order.kind,
                candidate_id=candidate.get("candidate_id"),
                strict_failure_equivalence_digest=key.digest)
        self.event_log.emit(
            "strict_failure_outcome", attempt_id=attempt_id,
            order_id=order.id, kind=order.kind,
            candidate_id=candidate.get("candidate_id"),
            strict_failure_equivalence_key=key.as_dict(),
            strict_failure_equivalence_digest=key.digest,
            strict_failure_material_revision=after.material_revision,
            strict_failure_outcome=outcome,
            strict_failure_memory_state=after.state,
            strict_recovery_slot=after.recovery_slot,
            strict_retry_allowed=after.allowed,
            strict_retry_suppressed=(
                self.strict_failure_memory_mode == "control"
                and not after.allowed),
            strict_reactivation_reason=after.reactivation_reason,
            strict_candidate_selection_reason=after.reason,
            strict_failure_memory_mode=self.strict_failure_memory_mode,
            authoritative=self.strict_failure_memory_mode == "control",
            delivered=delivered,
            failure_stage=failure_stage,
            failure_reason=failure_reason)

    def _finish_worker(
            self, return_code: int, *, result: dict | None = None,
            inspected: dict | None = None) -> None:
        result = self._read_worker_result() if result is None else result
        inspected = (self._inspect_terminal_result(result)
                     if inspected is None else inspected)
        order = self.current_order
        terminal_attempt_id = getattr(self, "current_attempt_id", None)
        if (self.worker_stop_reason == "discovery_segment_complete"
                and self.task is not None and order is not None):
            elapsed = (
                None if self.worker_started_at is None else
                round(max(0.0, time.monotonic() - self.worker_started_at), 3))
            self.event_log.emit(
                "discovery_worker_yielded", attempt_id=terminal_attempt_id,
                order_id=order.id, kind=order.kind, elapsed_s=elapsed,
                attempts_consumed=0,
                segment=self.active_discovery_segment)
            self._write_summary("discovery_segment_completed")
            self.worker = None
            self.worker_started_at = None
            self.worker_result_path = None
            self.current_order = None
            self.preferred_marker_id = None
            self.current_candidate = None
            self.current_attempt_id = None
            self.current_attempt_fingerprint = None
            self.current_worker_binding = None
            self.worker_runtime_phase = None
            self.worker_stop_reason = None
            self.worker_terminate_at = None
            self.hint_sent_for_order = None
            self.fallback_rescan_open = False
            self.active_discovery_segment = None
            getattr(
                self, "discovery_segment_candidate_ids_before", set()).clear()
            self._publish_stop()
            return
        if (self.worker_stop_reason == "score_first_defer"
                and self.task is not None and order is not None):
            elapsed = (
                None if self.worker_started_at is None else
                round(max(0.0, time.monotonic() - self.worker_started_at), 3))
            defer_record = {
                "reason": "stronger_current_run_evidence_available",
                "elapsed_s": elapsed,
                "attempts_consumed": 0,
            }
            self.deferred_orders[order.id] = defer_record
            self.orders_deferred_once.add(order.id)
            self.event_log.emit(
                "order_deferred",
                order_id=order.id,
                kind=order.kind,
                order_sequence=self.current_order_sequence,
                **defer_record,
            )
            self._write_summary("worker_deferred")
            self.worker = None
            self.worker_started_at = None
            self.worker_result_path = None
            self.current_order = None
            self.preferred_marker_id = None
            self.current_candidate = None
            self.fallback_rescan_open = False
            self.worker_stop_reason = None
            self.worker_terminate_at = None
            self.hint_sent_for_order = None
            if terminal_attempt_id is not None:
                self.terminal_result_fingerprints.pop(
                    str(terminal_attempt_id), None)
            self.current_attempt_id = None
            self.current_attempt_fingerprint = None
            self.current_worker_binding = None
            self.worker_runtime_phase = None
            self._publish_stop()
            return
        accepted_ids = getattr(self, "terminal_outcome_accepted_ids", set())
        duplicate = terminal_attempt_id in accepted_ids
        completion_s = inspected.get("completion_s")
        within_hard_deadline = (
            completion_s is not None
            and completion_s <= self.args.match_timeout)
        delivered = bool(
            inspected.get("valid")
            and not duplicate
            and result.get("status") == "delivered"
            and within_hard_deadline)
        rejection_reason = inspected.get("reason")
        if duplicate:
            rejection_reason = "duplicate_terminal_result"
        elif (inspected.get("valid") and result.get("status") == "delivered"
              and completion_s is None):
            rejection_reason = "terminal_completion_timestamp_unavailable"
        elif (inspected.get("valid") and result.get("status") == "delivered"
              and not within_hard_deadline):
            rejection_reason = "completion_after_hard_deadline"
        if rejection_reason is not None:
            self._emit_terminal_rejection(
                terminal_attempt_id, rejection_reason,
                inspected.get("fingerprint", ""))
        marker_id = result.get("marker_id")
        if not isinstance(marker_id, int):
            marker_id = None
        validated_marker_id = result.get("validated_marker_id")
        if not isinstance(validated_marker_id, int):
            validated_marker_id = None
        attempted_marker_id = (
            validated_marker_id if validated_marker_id is not None
            else marker_id if self.current_candidate is None else None)
        error = (
            rejection_reason
            or self.worker_stop_reason
            or result.get("error")
            or result.get("status")
            or f"worker_exit_{return_code}")
        process_failure = not result or rejection_reason in {
            "terminal_result_absent", "invalid_terminal_status"}
        failure_stage = str(
            "process" if process_failure else
            result.get("candidate_first_failure_stage")
            or result.get("execution_failure_stage")
            or result.get("phase") or "worker")
        failure_reason = str(
            "process_error" if process_failure else
            result.get("candidate_first_failure_reason")
            or result.get("execution_failure_reason")
            or ("delivered" if delivered else error))
        sensor_invalid = failure_reason in {
            "no_fresh_rgb", "rgb_not_processed_by_yolo"}

        if self.task is not None and order is not None and not duplicate:
            self.task.finish_attempt(
                order,
                delivered=delivered,
                marker_id=attempted_marker_id,
                error=None if delivered else str(error),
                # Attempt count is telemetry, not a semantic hard-failure
                # condition.  Fingerprint/deadline admission controls retries.
                max_attempts=sys.maxsize,
            )
            if inspected.get("valid") and rejection_reason is None:
                if not hasattr(self, "terminal_outcome_accepted_ids"):
                    self.terminal_outcome_accepted_ids = set()
                self.terminal_outcome_accepted_ids.add(terminal_attempt_id)
                after_soft = bool(
                    completion_s is not None
                    and completion_s >= min(SOFT_DEADLINE_S,
                                            self.args.match_timeout))
                delivered_count = self.task.summary().get("delivered", 0)
                self.event_log.emit(
                    "terminal_outcome_accepted",
                    attempt_id=terminal_attempt_id, order_id=order.id,
                    terminal_result_status=result.get("status"),
                    terminal_result_completion_s=completion_s,
                    inflight_completed_after_soft_deadline=after_soft,
                    delivered_count_after_acceptance=delivered_count)
                if delivered and after_soft:
                    self.event_log.emit(
                        "inflight_completed_after_soft_deadline",
                        attempt_id=terminal_attempt_id, order_id=order.id,
                        terminal_result_completion_s=completion_s,
                        delivered_count_after_acceptance=delivered_count)
            if self.current_candidate is not None:
                provisional_entry = self.current_candidate
                provisional_marker_id = int(
                    provisional_entry["provisional_marker_id"])
                provisional_entry.pop("reserved_order_id", None)
                source_evidence = getattr(
                    self, "candidate_source_evidence", {}).get(
                    str(provisional_entry.get("candidate_id")))
                if source_evidence is not None:
                    self.event_log.emit(
                        "candidate_source_state_outcome_join",
                        attempt_id=terminal_attempt_id,
                        order_id=order.id,
                        candidate_id=provisional_entry.get("candidate_id"),
                        kind=order.kind,
                        candidate_creation_context=source_evidence.as_dict(),
                        attempt_replay_context={
                            "observed_base_pose": provisional_entry.get(
                                "observed_base_pose"),
                            "observed_head_pose": provisional_entry.get(
                                "observed_head_pose"),
                            "observed_source_stamps": provisional_entry.get(
                                "observed_source_stamps"),
                        },
                        first_failure_stage=failure_stage,
                        first_failure_reason=failure_reason,
                        terminal_result=classify_candidate_outcome(result))
                if process_failure:
                    provisional_entry["state"] = PROVISIONAL_VIEW_HINT
                elif validated_marker_id is None:
                    provisional_entry["state"] = (
                        PROVISIONAL_VIEW_HINT if sensor_invalid else INVALIDATED)
                    provisional_entry["first_failure_stage"] = result.get(
                        "candidate_first_failure_stage")
                    provisional_entry["first_failure_reason"] = result.get(
                        "candidate_first_failure_reason")
                    if sensor_invalid:
                        self.event_log.emit(
                            "candidate_runtime_invalid",
                            attempt_id=terminal_attempt_id,
                            order_id=order.id,
                            candidate_id=provisional_entry.get("candidate_id"),
                            kind=order.kind,
                            failure_reason=failure_reason,
                            semantic_candidate_penalized=False)
                else:
                    corrected = validated_marker_id != provisional_marker_id
                    if corrected:
                        provisional_entry["state"] = INVALIDATED
                        provisional_entry["corrected_to_marker_id"] = (
                            validated_marker_id)
                        validated_entry = self.inventory.get(
                            validated_marker_id, dict(provisional_entry))
                        validated_entry["candidate_id"] = (
                            f"validated-{validated_marker_id}")
                        validated_entry["provisional_marker_id"] = (
                            provisional_marker_id)
                        validated_entry["marker_id"] = validated_marker_id
                        self.inventory[validated_marker_id] = validated_entry
                    else:
                        validated_entry = provisional_entry
                    validated_entry.update({
                        "validation_state": LOCALIZATION_VALIDATED,
                        "validated_marker_id": validated_marker_id,
                        "validated_target_world": result.get(
                            "validated_target_world"),
                        "validated_station_context": result.get(
                            "validated_station_context"),
                        "validation_elapsed_s": result.get(
                            "validation_elapsed_s"),
                        "state": (
                            DELIVERED if delivered else
                            PROVISIONAL_VIEW_HINT
                            if failure_stage in {
                                "navigation", "delivery_navigation",
                                "nav_to_delivery"} else INVALIDATED),
                    })
            fingerprint = getattr(self, "current_attempt_fingerprint", None)
            if fingerprint is not None:
                outcome = CandidateAttemptOutcome(
                    fingerprint=fingerprint,
                    failure_stage=failure_stage,
                    failure_reason=failure_reason,
                    terminal_s=float(completion_s or max(
                        0.0, time.monotonic()
                        - (self.task_started_at or time.monotonic()))),
                    evidence_revision=(
                        fingerprint.candidate_evidence_revision),
                    candidate_state_after_failure=(
                        DELIVERED if delivered else INVALIDATED),
                    reactivation_requirements=reactivation_requirements(
                        failure_stage, failure_reason),
                )
                if sensor_invalid:
                    self.event_log.emit(
                        "candidate_semantic_penalty_skipped",
                        attempt_id=terminal_attempt_id,
                        order_id=order.id, kind=order.kind,
                        candidate_id=fingerprint.candidate_id,
                        failure_reason=failure_reason,
                        reactivation_requirement="sensor_recovered")
                elif self.candidate_attempt_memory.record(outcome):
                    self.event_log.emit(
                        "candidate_attempt_outcome", attempt_id=terminal_attempt_id,
                        order_id=order.id, kind=order.kind,
                        delivered=delivered, **outcome.as_dict())
                    retry = self.candidate_attempt_memory.decision(fingerprint)
                    if not delivered and not retry.allowed:
                        self.retry_suppression_events.add(fingerprint.digest)
                        self.event_log.emit(
                            "candidate_retry_suppressed",
                            attempt_id=terminal_attempt_id,
                            order_id=order.id, kind=order.kind,
                            candidate_id=fingerprint.candidate_id,
                            fingerprint_digest=fingerprint.digest,
                            reason=retry.reason,
                            model_estimated_avoided_time_s=(
                                completion_estimate(
                                    fingerprint.marker_id is not None)
                                .estimated_completion_s))
            if terminal_attempt_id is not None:
                self._record_strict_outcome(
                    result=result, order=order,
                    attempt_id=str(terminal_attempt_id), delivered=delivered,
                    process_failure=process_failure,
                    sensor_invalid=sensor_invalid,
                    failure_stage=failure_stage,
                    failure_reason=failure_reason,
                    validated_marker_id=validated_marker_id)
            replay_memory = getattr(self, "replay_outcome_memory", None)
            if (replay_memory is not None and self.current_candidate is not None
                    and terminal_attempt_id is not None):
                try:
                    replay_key = make_equivalence_key(
                        run_prefix=self.task.run_prefix, kind=order.kind,
                        marker_id=int(self.current_candidate["marker_id"]),
                        candidate=self.current_candidate)
                    fresh_class = classify_fresh_frame_outcome(
                        raw_fresh_rgb_count=int(result.get(
                            "raw_fresh_rgb_frame_count", 0)),
                        yolo_processed_count=int(result.get(
                            "yolo_processed_fresh_frame_count", 0)),
                        target_detection_count=int(result.get(
                            "target_kind_detection_count", 0)))
                    replay_success = validated_marker_id is not None
                    pose_ids = result.get("replay_pose_ids_attempted") or [
                        replay_key.primary_replay_pose_id]
                    replay_state = None
                    for pose_id in pose_ids:
                        replay_state = replay_memory.record(ReplayOutcome(
                            equivalence_key=replay_key,
                            candidate_id=str(self.current_candidate.get(
                                "candidate_id")),
                            attempt_id=str(terminal_attempt_id),
                            pose_id=str(pose_id).split(":")[-1],
                            outcome=(REPLAY_SUCCEEDED if replay_success
                                     else "FAILED"),
                            failure_class=(None if replay_success
                                           else fresh_class),
                            fresh_rgb_count=int(result.get(
                                "raw_fresh_rgb_frame_count", 0)),
                            yolo_processed_count=int(result.get(
                                "yolo_processed_fresh_frame_count", 0)),
                            target_detection_count=int(result.get(
                                "target_kind_detection_count", 0)),
                            attempt_start_s=round(max(
                                0.0, (self.worker_started_at or time.monotonic())
                                - (self.task_started_at or time.monotonic())), 3),
                            attempt_end_s=float(completion_s or max(
                                0.0, time.monotonic()
                                - (self.task_started_at or time.monotonic()))),
                            material_context_revision=replay_key.digest,
                            reactivation_requirements=(
                                ("sensor_recovered",) if sensor_invalid else
                                ("material_context_changed",))))
                    allowed, replay_reason = replay_memory.decision(replay_key)
                    self.event_log.emit(
                        "replay_outcome_shadow",
                        attempt_id=terminal_attempt_id,
                        order_id=order.id, kind=order.kind,
                        candidate_id=self.current_candidate.get("candidate_id"),
                        equivalence_key=replay_key.as_dict(),
                        equivalence_digest=replay_key.digest,
                        replay_state=replay_state,
                        shadow_allowed=allowed,
                        shadow_reason=replay_reason,
                        authoritative=False,
                        failure_class=(None if replay_success else fresh_class),
                        raw_fresh_rgb_frame_count=int(result.get(
                            "raw_fresh_rgb_frame_count", 0)),
                        yolo_processed_fresh_frame_count=int(result.get(
                            "yolo_processed_fresh_frame_count", 0)),
                        target_kind_detection_count=int(result.get(
                            "target_kind_detection_count", 0)))
                except (KeyError, TypeError, ValueError) as exc:
                    self.event_log.emit(
                        "replay_outcome_shadow_unavailable",
                        attempt_id=terminal_attempt_id,
                        order_id=order.id, kind=order.kind,
                        reason=f"{type(exc).__name__}: {exc}")
            if self.fallback_rescan_open:
                self.event_log.emit(
                    "fallback_rescan_end", order_id=order.id,
                    kind=order.kind,
                    order_sequence=self.current_order_sequence,
                    outcome=(
                        "localized" if marker_id is not None else "failed"))
            self.event_log.emit(
                "order_completed" if delivered else "order_failed",
                order_id=order.id,
                kind=order.kind,
                order_sequence=self.current_order_sequence,
                marker_id=attempted_marker_id,
                return_code=return_code,
                reason=None if delivered else str(error),
            )
            if (terminal_attempt_id is not None
                    and terminal_attempt_id not in getattr(
                        self, "attempt_terminal_ids", set())):
                if not hasattr(self, "attempt_terminal_ids"):
                    self.attempt_terminal_ids = set()
                self.attempt_terminal_ids.add(terminal_attempt_id)
                self.event_log.emit(
                    "attempt_terminal", attempt_id=terminal_attempt_id,
                    order_id=order.id, kind=order.kind,
                    delivered=delivered, return_code=return_code,
                    reason=None if delivered else str(error))
            self._write_summary("worker_finished")

            message = (
                f"order id={order.id} kind={order.kind} "
                f"status={order.status} marker={attempted_marker_id} "
                f"attempts={order.attempts}")
            if delivered:
                logged = _safe_log_info(
                    self.get_logger(), message, event_log=self.event_log,
                    context="finish_worker_delivered")
            else:
                logged = _safe_log_error(
                    self.get_logger(), message, event_log=self.event_log,
                    context="finish_worker_failed")
            if not logged:
                # The authoritative snapshot above already committed.  This
                # best-effort refresh only makes the contained logger fault
                # visible in the summary; it can never undo the outcome.
                try:
                    self._write_summary("worker_finished")
                except Exception as exc:
                    try:
                        sys.stderr.write(
                            "Logger-fault summary refresh failed: "
                            f"{type(exc).__name__}: {exc}\n")
                        sys.stderr.flush()
                    except Exception:
                        pass

        self.worker = None
        self.worker_started_at = None
        self.worker_result_path = None
        self.current_order = None
        self.preferred_marker_id = None
        self.current_candidate = None
        self.active_discovery_segment = None
        getattr(self, "discovery_segment_candidate_ids_before", set()).clear()
        self.fallback_rescan_open = False
        self.worker_stop_reason = None
        self.worker_terminate_at = None
        self.current_attempt_id = None
        self.current_attempt_fingerprint = None
        self.current_worker_binding = None
        self.worker_runtime_phase = None
        self._publish_stop()

    def _write_summary(self, reason: str) -> None:
        if self.task is None:
            return
        document = self.task.summary()
        document["reason"] = reason
        inventory = [dict(entry) for _, entry in sorted(self.inventory.items())]
        if self.task_started_at is not None:
            document["elapsed_s"] = round(
                time.monotonic() - self.task_started_at, 3)
        summary_path = (
            Path(self.args.summary_file)
            if self.args.summary_file else
            Path(self.args.runtime_dir)
            / safe_component(self.task.run_prefix)
            / "summary.json")
        score_summary = build_summary(
            read_events(self.event_file),
            document,
            mode=self.args.memory_mode,
            product_seed=self.args.product_seed,
            obstacle_seed=self.args.obstacle_seed,
            task_kinds=[order.kind for order in self.task.orders],
            inventory=inventory,
            scan_coverage=sorted(self.scan_coverage),
        )
        coverage_snapshot = (
            None if self.run_coverage is None
            else self.run_coverage.snapshot())
        coverage_metrics = (
            {} if coverage_snapshot is None
            else coverage_snapshot.get("metrics", {}))
        selected_ids = {
            event.get("order_id") for event in read_events(self.event_file)
            if event.get("event") == "order_selected"
            and isinstance(event.get("order_id"), str)
        }
        started_order_ids = {
            attempt_id.split(":attempt-")[0].split(":", 1)[-1]
            for attempt_id in self.attempt_started_ids
        }
        terminal_order_ids = {
            attempt_id.split(":attempt-")[0].split(":", 1)[-1]
            for attempt_id in self.attempt_terminal_ids
        }
        delivered_ids = {
            order.id for order in self.task.orders
            if order.status == "delivered"
        }
        attempted_ids = {
            order.id for order in self.task.orders
            if order.attempts > 0 or order.id in self.orders_deferred_once
        }
        score_summary.update({
            "order_runtime_states": {
                order.id: (
                    "DELIVERED" if order.status == "delivered" else
                    "FAILED" if order.status == "failed" else
                    "DEFERRED" if order.id in self.deferred_orders else
                    "IN_PROGRESS" if self.current_order is order else
                    "READY_PROVISIONAL"
                    if self._inventory_for_order(order) is not None else
                    "PENDING")
                for order in self.task.orders
            },
            "deferred_count": len(self.deferred_orders),
            "attempted_order_count": len(attempted_ids),
            "unattempted_order_count": len(self.task.orders) - len(attempted_ids),
            "orders_deferred_once": sorted(self.orders_deferred_once),
            "order_selected_count": sum(
                event.get("event") == "order_selected"
                for event in read_events(self.event_file)),
            "attempt_started_count": len(self.attempt_started_ids),
            "worker_started_count": self.worker_started_count,
            "attempt_terminal_count": len(self.attempt_terminal_ids),
            "order_delivered_count": len(delivered_ids),
            "unique_orders_selected": len(selected_ids),
            "unique_orders_attempt_started": len(started_order_ids),
            "unique_orders_terminal": len(terminal_order_ids),
            "unique_orders_delivered": len(delivered_ids),
            "run_scan_coverage": coverage_snapshot,
            **coverage_metrics,
        })
        atomic_write_json(summary_path, score_summary)
        write_summary_csv(summary_path.with_suffix(".csv"), score_summary)

    def _finish_match(self, reason: str) -> None:
        self.finished = True
        self._publish_stop()
        self.event_log.emit(
            "match_end",
            reason=reason,
            delivered_count=(self.task.summary().get("delivered", 0)
                             if self.task is not None else 0),
        )
        self._write_summary(reason)
        summary = self.task.summary() if self.task is not None else {}
        self.get_logger().info(
            f"match finished reason={reason} "
            f"delivered={summary.get('delivered', 0)}/"
            f"{summary.get('count', 0)} failed={summary.get('failed', 0)}")
        rclpy.shutdown()

    def _publish_stop(self) -> None:
        try:
            self.stop_publisher.publish(Twist())
        except Exception:  # noqa: BLE001 - ROS context may already be closed
            pass

    @staticmethod
    def _decode_records(message: String) -> list[dict]:
        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return []
        return [item for item in value if isinstance(item, dict)] \
            if isinstance(value, list) else []

    def _yolo_cb(self, message: String) -> None:
        records = [item for item in self._decode_records(message)
                   if item.get("camera", "head") == "head"]
        if not records:
            return
        try:
            stamp = int(records[0]["stamp_ns"])
        except (KeyError, TypeError, ValueError):
            return
        self.latest_yolo = (stamp, records)
        self.inventory_yolo_frames.append((stamp, records))
        self._update_inventory()

    def _aruco_cb(self, message: String) -> None:
        records = [item for item in self._decode_records(message)
                   if item.get("camera", "head") == "head"]
        if not records:
            return
        try:
            stamp = int(records[0]["stamp_ns"])
        except (KeyError, TypeError, ValueError):
            return
        self.latest_aruco = (stamp, records)
        self.inventory_aruco_frames.append((stamp, records))
        self._update_inventory()

    def _odom_cb(self, message: Odometry) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z
                   + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y
                         + orientation.z * orientation.z),
        )
        values = (float(position.x), float(position.y), float(yaw))
        if all(math.isfinite(value) for value in values):
            self.latest_base_pose = values
            try:
                stamp_ns = (int(message.header.stamp.sec) * 1_000_000_000
                            + int(message.header.stamp.nanosec))
                self.source_state_history.append_odom(
                    source_stamp_ns=stamp_ns,
                    callback_receipt_monotonic_ns=time.monotonic_ns(),
                    x=values[0], y=values[1], yaw=values[2])
            except (AttributeError, TypeError, ValueError):
                pass

    def _joint_cb(self, message: JointState) -> None:
        joints = {
            name: float(message.position[index])
            for index, name in enumerate(message.name)
            if index < len(message.position)
        }
        names = ("slide_joint", "head_yaw_joint", "head_pitch_joint")
        if all(name in joints and math.isfinite(joints[name]) for name in names):
            self.latest_head_pose = tuple(joints[name] for name in names)
            try:
                stamp_ns = (int(message.header.stamp.sec) * 1_000_000_000
                            + int(message.header.stamp.nanosec))
                self.source_state_history.append_joint(
                    source_stamp_ns=stamp_ns,
                    callback_receipt_monotonic_ns=time.monotonic_ns(),
                    slide=joints["slide_joint"],
                    head_yaw=joints["head_yaw_joint"],
                    head_pitch=joints["head_pitch_joint"])
            except (AttributeError, TypeError, ValueError):
                pass

    def _scan_progress_cb(self, message: String) -> None:
        if self.task is None or self.run_coverage is None:
            return
        try:
            event = json.loads(message.data)
            key = CoverageKey(
                str(event["run_prefix"]), int(event["station_id"]),
                str(event["pose_name"]), str(event["shelf_band"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if key.run_prefix != self.task.run_prefix:
            return
        funnel = (None if self.candidate_funnel is None else
                  self.candidate_funnel.current(event.get("attempt_id")))
        if event.get("phase") == "started":
            if self.coverage_mode == "off":
                accepted = False
            else:
                accepted = self.run_coverage.start(
                    key, stamp=float(event.get("monotonic_s", time.monotonic())),
                    resumed=bool(event.get("resumed", False)),
                    estimated_duration_s=getattr(
                        self.args, "discovery_segment_estimate", 2.0))
            pending_kinds = [
                order.kind for order in self.task.orders
                if order.status == "pending"]
            if self.candidate_funnel is not None:
                self.candidate_funnel.start_pose(
                    attempt_id=event.get("attempt_id", "unknown"),
                    station_id=key.station_id, pose_name=key.pose_name,
                    shelf_band=key.shelf_band,
                    pending_kinds=pending_kinds)
            self.event_log.emit(
                "scan_pose_started", coverage_key=event,
                accepted=accepted, attempt_id=event.get("attempt_id"))
            if (accepted and not getattr(self, "current_candidate", None)
                    and self.coverage_mode == "resume_only"):
                segment = stable_segment_id(
                    key.run_prefix, key.station_id, key.pose_name,
                    key.shelf_band)
                self.active_discovery_segment = {
                    "segment_id": segment, "coverage_key": {
                        "run_prefix": key.run_prefix,
                        "station_id": key.station_id,
                        "pose_name": key.pose_name,
                        "shelf_band": key.shelf_band},
                    "estimated_cost_s": getattr(
                        self.args, "discovery_segment_estimate", 2.0)}
                self.event_log.emit(
                    "discovery_segment_started", segment_id=segment,
                    coverage_key=self.active_discovery_segment["coverage_key"],
                    estimated_cost_s=getattr(
                        self.args, "discovery_segment_estimate", 2.0),
                    attempt_id=event.get("attempt_id"))
        elif event.get("phase") == "completed":
            valid = False
            if self.coverage_mode != "off":
                valid = self.run_coverage.complete(
                    key, stamp=float(event.get("monotonic_s", time.monotonic())),
                    fresh_rgb_frame_count=event.get("fresh_rgb_frame_count", 0),
                    fresh_aruco_frame_count=event.get("fresh_aruco_frame_count", 0),
                    observed_kinds=event.get("observed_kinds", []),
                    candidate_ids_created=event.get("candidate_ids_created", []),
                    completion_reason=event.get("completion_reason", "pose_elapsed"),
                    camera_settled=bool(event.get("camera_settled", False)),
                    pose_completed=bool(event.get("pose_completed", False)),
                    interrupted=bool(event.get("interrupted", False)))
            self.event_log.emit(
                "scan_pose_completed", coverage_key=event,
                covered_valid=valid, attempt_id=event.get("attempt_id"))
            if (self.coverage_mode == "resume_only"
                    and self.active_discovery_segment is not None
                    and event.get("attempt_id") == self.current_attempt_id):
                candidate_ids_after = {
                    str(entry.get("candidate_id"))
                    for entry in self.inventory.values()
                    if entry.get("candidate_id") is not None}
                created = sorted(
                    candidate_ids_after
                    - self.discovery_segment_candidate_ids_before)
                record = self.run_coverage.records.get(key)
                self.event_log.emit(
                    "discovery_segment_completed",
                    **self.active_discovery_segment,
                    actual_cost_s=(None if record is None else
                                   record.actual_duration_s),
                    covered_valid=valid,
                    observed_kinds=event.get("observed_kinds", []),
                    candidate_ids_created=created,
                    candidate_created_count=len(created),
                    fresh_rgb_frame_count=event.get(
                        "fresh_rgb_frame_count", 0),
                    fresh_aruco_frame_count=event.get(
                        "fresh_aruco_frame_count", 0))
                if self.worker is not None and self.worker_stop_reason is None:
                    self._request_worker_stop("discovery_segment_complete")
            summary = (None if self.candidate_funnel is None else
                       self.candidate_funnel.end_pose(event.get("attempt_id")))
            if summary is not None:
                summary.update({
                    "fresh_rgb_frame_count": event.get(
                        "fresh_rgb_frame_count", 0),
                    "fresh_aruco_frame_count": event.get(
                        "fresh_aruco_frame_count", 0),
                    "yolo_detection_count_by_kind": event.get(
                        "yolo_detection_count_by_kind", {}),
                    "target_task_kind_detection_count": sum(
                        int(event.get("yolo_detection_count_by_kind", {}).get(
                            kind, 0)) for kind in summary["pending_task_kinds"]),
                    "aruco_detection_count": event.get(
                        "aruco_detection_count", 0),
                    "aruco_seen_ids": event.get("aruco_seen_ids", []),
                    "synchronized_frame_pair_count": event.get(
                        "synchronized_frame_pair_count", 0),
                    "pair_desync_reject_count": event.get(
                        "pair_desync_reject_count", 0),
                    "association_attempt_count": event.get(
                        "association_attempt_count", 0),
                    "association_success_count": event.get(
                        "association_success_count", 0),
                    "association_reject_reason_counts": event.get(
                        "association_reject_reason_counts", {}),
                })
                summary["first_loss_stage"] = first_loss_stage(summary)
                self.event_log.emit(
                    "candidate_admission_funnel_pose", **summary)

    def _worker_progress_cb(self, message: String) -> None:
        try:
            event = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if (not isinstance(event, dict)
                or event.get("attempt_id") != self.current_attempt_id):
            return
        phase = event.get("phase")
        if isinstance(phase, str):
            self.worker_runtime_phase = phase

    def _candidate_order_id(self, kind: str) -> str | None:
        if self.task is None:
            return None
        normalized = normalize_kind(kind)
        return next((
            order.id for order in self.task.orders
            if order.status == "pending"
            and normalize_kind(order.kind) == normalized
        ), None)

    def _update_inventory(self) -> None:
        if self.args.memory_mode != "run_inventory":
            return
        if (self.task is None or not self.inventory_yolo_frames
                or not self.inventory_aruco_frames):
            return
        funnel = (None if self.candidate_funnel is None else
                  self.candidate_funnel.current(self.current_attempt_id))
        matched = None
        for yolo_frame in self.inventory_yolo_frames:
            if yolo_frame[0] in self.inventory_processed_yolo_stamps:
                continue
            aruco_frame = nearest_synchronized_frame(
                yolo_frame[0], self.inventory_aruco_frames,
                tolerance_ns=INVENTORY_SYNC_TOLERANCE_NS)
            if aruco_frame is not None:
                matched = (yolo_frame, aruco_frame)
                break
        if matched is None:
            if funnel is not None:
                funnel.pair_desync_reject_count += 1
            return
        (yolo_stamp, detections), (aruco_stamp, markers) = matched
        self.inventory_processed_yolo_stamps.add(yolo_stamp)
        retained_stamps = {frame[0] for frame in self.inventory_yolo_frames}
        self.inventory_processed_yolo_stamps.intersection_update(retained_stamps)
        pair = (yolo_stamp, aruco_stamp)
        if (pair == self.last_inventory_pair
                or yolo_stamp == self.last_inventory_yolo_stamp):
            return
        self.last_inventory_pair = pair
        self.last_inventory_yolo_stamp = yolo_stamp
        wanted = {order.kind for order in self.task.orders
                  if order.status == "pending"}
        if funnel is not None:
            pair_key = (yolo_stamp, aruco_stamp)
            if pair_key in funnel.seen_pair_keys:
                funnel.duplicate_pair_count += 1
            else:
                funnel.seen_pair_keys.add(pair_key)
                funnel.synchronized_frame_pair_count += 1
        for detection in detections:
            kind = detection.get("class")
            if kind not in wanted:
                continue
            marker = associate_detection_marker(detection, markers)
            if marker is None:
                if funnel is not None:
                    funnel.association_attempt_count += 1
                    funnel.association_reject_reason_counts["no_legal_marker"] += 1
                continue
            if funnel is not None:
                funnel.association_attempt_count += 1
                funnel.association_success_count += 1
            marker_id = int(marker["id"])
            self.scan_coverage.add(marker_id)
            previous = self.inventory.get(marker_id)
            confirmations = (
                int(previous["confirmations"]) + 1
                if previous is not None and previous.get("kind") == kind
                else 1)
            if (funnel is not None and previous is not None
                    and previous.get("kind") != kind):
                funnel.confirmation_reset_count += 1
                funnel.confirmation_reset_reasons["marker_key_kind_changed"] += 1
            try:
                confidence = float(detection.get("conf", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            position_world = marker.get("position_world")
            view_hint = derive_candidate_view_hint(position_world)
            station_index = None
            if self.latest_base_pose is not None:
                station_index = min(
                    range(len(SCAN_X)),
                    key=lambda index: abs(
                        SCAN_X[index] - self.latest_base_pose[0]))
            context = make_observed_context(
                base_pose=self.latest_base_pose,
                head_pose=self.latest_head_pose,
                camera_poses=SCAN_CAMERA_POSES,
                station_index=station_index,
                station_x=(None if station_index is None
                           else SCAN_X[station_index]),
                station_y=(None if station_index is None else SCAN_Y),
                yolo_stamp=yolo_stamp,
                aruco_stamp=aruco_stamp,
                detection=detection,
                marker=marker,
                controller_state="inventory_observation",
            ).as_dict()
            if context["context_quality"] == CONTEXT_COMPLETE:
                view_hint = {
                    "observation_base_pose_hint": list(
                        context["observed_base_pose"]),
                    "head_pose_hint": list(context["observed_head_pose"]),
                    "scan_station_hint": {
                        "index": station_index,
                        "x": float(SCAN_X[station_index]),
                        "y": float(SCAN_Y),
                    },
                    "scan_pitch_hint": float(
                        context["observed_head_pose"][3]),
                    "hint_source": "OBSERVED_CONTEXT",
                }
            entry = {
                "candidate_id": f"candidate-{marker_id}",
                "marker_id": marker_id,
                "provisional_marker_id": marker_id,
                "provisional_marker_world": position_world,
                "kind": kind,
                "normalized_kind": normalize_kind(kind),
                "position_world": position_world,
                "source_yolo_stamp_ns": yolo_stamp,
                "source_aruco_stamp_ns": aruco_stamp,
                "confidence": round(max(
                    confidence,
                    float(previous.get("confidence", 0.0))
                    if previous is not None else 0.0), 4),
                "confirmations": min(confirmations, 1000),
                "stamp_ns": yolo_stamp,
                "candidate_created_monotonic_s": (
                    previous.get("candidate_created_monotonic_s")
                    if previous is not None else time.monotonic()),
                "state": (
                    previous.get("state", PROVISIONAL_VIEW_HINT)
                    if previous is not None else PROVISIONAL_VIEW_HINT),
                **view_hint,
                **context,
            }
            # Creation context is an immutable fact.  Later confirmations may
            # update confidence/stamps, but never silently move the replay view.
            if (previous is not None
                    and previous.get("confirmations", 0)
                    >= self.args.inventory_confirmations):
                for key in (
                        "observation_base_pose_hint", "head_pose_hint",
                        "scan_station_hint", "scan_pitch_hint", "hint_source",
                        "context_type", "context_source", "context_quality",
                        "observed_base_pose", "observed_head_pose",
                        "observed_scan_station", "observed_pose_name",
                        "observed_source_stamps", "target_bbox_summary",
                        "marker_pixel_summary", "association_summary",
                        "controller_state", "scan_index", "pitch_index",
                        "camera_settled"):
                    entry[key] = previous.get(key)
            if previous is not None and "reserved_order_id" in previous:
                entry["reserved_order_id"] = previous["reserved_order_id"]
            if (previous is not None
                    and previous.get("state") == INVALIDATED
                    and self.task is not None):
                order_id = self._candidate_order_id(kind)
                fingerprint = make_fingerprint(
                    run_prefix=self.task.run_prefix,
                    order_id=order_id,
                    product_kind=kind,
                    candidate=entry,
                    coverage_revision=len(self.scan_coverage))
                retry = self.candidate_attempt_memory.decision(fingerprint)
                if retry.allowed and retry.new_evidence:
                    strict_key, strict_decision = self._strict_key_and_decision(
                        next(item for item in self.task.orders
                             if item.id == order_id), entry)
                    strict_control_blocked = bool(
                        self.strict_failure_memory_mode == "control"
                        and strict_decision is not None
                        and not strict_decision.allowed)
                    if strict_control_blocked:
                        self._emit_strict_suppression(
                            order_id=order_id, kind=kind, candidate=entry,
                            key=strict_key, decision=strict_decision)
                    else:
                        entry["state"] = PROVISIONAL_VIEW_HINT
                    if (not strict_control_blocked
                            and fingerprint.digest
                            not in self.candidate_reactivation_events):
                        self.candidate_reactivation_events.add(fingerprint.digest)
                        self.event_log.emit(
                            "candidate_new_evidence", order_id=order_id,
                            kind=kind, candidate_id=entry["candidate_id"],
                            fingerprint_digest=fingerprint.digest,
                            reason=retry.reason)
                        self.event_log.emit(
                            "candidate_reactivated", order_id=order_id,
                            kind=kind, candidate_id=entry["candidate_id"],
                            fingerprint_digest=fingerprint.digest,
                            reactivation_reason=retry.reason,
                            strict_failure_memory_state=(
                                None if strict_decision is None
                                else strict_decision.state),
                            strict_retry_allowed=(
                                True if strict_decision is None
                                else strict_decision.allowed),
                            strict_failure_memory_mode=(
                                self.strict_failure_memory_mode))
            self.inventory[marker_id] = entry
            if confirmations == self.args.inventory_confirmations:
                if funnel is not None:
                    funnel.threshold_reached_count += 1
                    funnel.candidate_constructed_count += 1
                    funnel.candidate_received_by_runner_count += 1
                    funnel.candidate_inserted_into_inventory_count += 1
                self.event_log.emit(
                    "candidate_admission",
                    candidate_key=[self.task.run_prefix, kind, marker_id],
                    unique_pair_key=[yolo_stamp, aruco_stamp],
                    confirmation_count_before=confirmations - 1,
                    confirmation_count_after=confirmations,
                    threshold_required=self.args.inventory_confirmations,
                    threshold_reached=True,
                    candidate_constructed=True,
                    candidate_context_quality=entry["context_quality"],
                    candidate_emitted_to_runner=True,
                    candidate_received_by_runner=True,
                    candidate_inserted_into_inventory=True,
                    candidate_rejected_reason=None,
                )
                self.event_log.emit(
                    "candidate_observation_context_created",
                    candidate_id=entry["candidate_id"],
                    order_id=self._candidate_order_id(kind),
                    kind=kind,
                    normalized_kind=entry["normalized_kind"],
                    context_source=entry["context_source"],
                    creation_base_pose=entry["observed_base_pose"],
                    creation_head_pose=entry["observed_head_pose"],
                    creation_scan_station=entry["observed_scan_station"],
                    creation_pose_name=entry["observed_pose_name"],
                    creation_source_stamps=entry["observed_source_stamps"],
                    creation_target_bbox=entry["target_bbox_summary"],
                    creation_marker_pixel=entry["marker_pixel_summary"],
                    association_summary=entry["association_summary"],
                    context_quality=entry["context_quality"],
                    context_type=entry["context_type"],
                )
                source_evidence = build_candidate_source_state_evidence(
                    history=self.source_state_history,
                    run_prefix=self.task.run_prefix,
                    candidate_id=entry["candidate_id"],
                    kind=kind,
                    marker_id=marker_id,
                    confirmation_count=confirmations,
                    yolo_source_stamp_ns=yolo_stamp,
                    aruco_source_stamp_ns=aruco_stamp,
                    callback_latest_base_pose=self.latest_base_pose,
                    callback_latest_head_pose=self.latest_head_pose,
                )
                self.candidate_source_evidence[entry["candidate_id"]] = (
                    source_evidence)
                self.event_log.emit(
                    "candidate_source_state_evidence",
                    **source_evidence.as_dict())
                self.event_log.emit(
                    "candidate_created", **entry)
                self.get_logger().info(
                    f"provisional candidate created marker={marker_id} "
                    f"kind={kind} confidence={entry['confidence']:.3f} "
                    f"station={entry['scan_station_hint']} "
                    f"pose={entry['head_pose_hint'][0]}")
        self._maybe_publish_memory_hint()

    def _inventory_for_order(self, order):
        if self.args.memory_mode != "run_inventory":
            return None
        excluded = set(self.task.excluded_markers(order.kind))
        entries = [
            entry for marker_id, entry in self.inventory.items()
            if marker_id not in excluded
            and entry.get("kind") == order.kind
            and entry.get("confirmations", 0) >= self.args.inventory_confirmations
            and entry.get("state", PROVISIONAL_VIEW_HINT)
            == PROVISIONAL_VIEW_HINT
        ]
        if not entries:
            return None
        memory = getattr(self, "candidate_attempt_memory", None)
        if memory is not None and self.task is not None:
            entries = [
                entry for entry in entries
                if memory.decision(make_fingerprint(
                    run_prefix=self.task.run_prefix,
                    order_id=order.id,
                    product_kind=order.kind,
                    candidate=entry,
                    coverage_revision=len(getattr(
                        self, "scan_coverage", ())))).allowed
            ]
        if not entries:
            return None

        def delivery_distance(entry):
            try:
                x, y = map(float, entry["position_world"][:2])
                return ((x + 1.94) ** 2 + (y + 3.41) ** 2) ** 0.5
            except (KeyError, TypeError, ValueError):
                return float("inf")

        return min(entries, key=lambda item: (
            delivery_distance(item), -float(item.get("confidence", 0.0))))

    def _maybe_publish_memory_hint(self) -> None:
        if (self.args.memory_mode != "run_inventory"
                or self.task is None or self.current_order is None
                or self.worker is None
                or self.hint_sent_for_order == self.current_order.id):
            return
        entry = self._inventory_for_order(self.current_order)
        if entry is None:
            options = self._score_first_options()
            stronger = stronger_ready_order(
                self.current_order.id,
                self.current_candidate is not None,
                options,
            )
            if (stronger is not None and self.worker_stop_reason is None
                    and self.current_order.id not in self.deferred_orders):
                self.event_log.emit(
                    "score_first_preemption",
                    order_id=self.current_order.id,
                    kind=self.current_order.kind,
                    stronger_order_id=stronger.order_id,
                    stronger_candidate_state=stronger.candidate_state,
                    stronger_marker_id=stronger.marker_id,
                    stronger_estimated_completion_s=(
                        stronger.estimated_completion_s),
                )
                self._request_worker_stop("score_first_defer")
            return
        marker_id = int(entry["marker_id"])
        entry["reserved_order_id"] = self.current_order.id
        self.preferred_marker_id = marker_id
        self.current_candidate = entry
        self.current_attempt_fingerprint = make_fingerprint(
            run_prefix=self.task.run_prefix,
            order_id=self.current_order.id,
            product_kind=self.current_order.kind,
            candidate=entry,
            coverage_revision=len(self.scan_coverage))
        self.hint_sent_for_order = self.current_order.id
        hint = {
            **entry,
            "schema_version": 2,
            "run_prefix": self.task.run_prefix,
            "order_id": self.current_order.id,
        }
        self.memory_hint_publisher.publish(
            String(data=json.dumps(hint, separators=(",", ":"))))
        self.event_log.emit(
            "target_memory_hit",
            order_id=self.current_order.id,
            kind=self.current_order.kind,
            order_sequence=self.current_order_sequence,
            marker_id=marker_id,
            source="live_run_inventory_coverage_complete",
        )
        retry = self.candidate_attempt_memory.decision(
            self.current_attempt_fingerprint)
        self.event_log.emit(
            "candidate_attempt_fingerprint",
            attempt_id=self.current_attempt_id,
            order_id=self.current_order.id,
            kind=self.current_order.kind,
            fingerprint_digest=self.current_attempt_fingerprint.digest,
            fingerprint=self.current_attempt_fingerprint.as_dict(),
            fingerprint_status=(
                "NEW_EVIDENCE" if retry.new_evidence else "UNTRIED"),
            new_evidence_since_previous_attempt=retry.new_evidence,
            reactivation_reason="live_candidate_acquired")
        self.event_log.emit(
            "full_scan_end",
            order_id=self.current_order.id,
            kind=self.current_order.kind,
            order_sequence=self.current_order_sequence,
            reason="current_order_candidate_ready",
        )
        self.get_logger().info(
            f"memory hint order={self.current_order.id} "
            f"kind={self.current_order.kind} provisional_marker={marker_id}; "
            "current order has a confirmed current-run candidate")

    def _score_first_options(self) -> list[OrderOption]:
        assert self.task is not None
        deferred_orders = getattr(self, "deferred_orders", {})
        options: list[OrderOption] = []
        task_started_at = getattr(self, "task_started_at", None)
        elapsed = (0.0 if task_started_at is None else max(
            0.0, time.monotonic() - task_started_at))
        remaining_hard_s = max(
            0.0, float(getattr(self.args, "match_timeout", 600.0)) - elapsed)
        match_start = task_started_at or time.monotonic()
        for order in self.task.orders:
            if order.status != "pending":
                continue
            entries = [
                entry for entry in self.inventory.values()
                if entry.get("kind") == order.kind
                and entry.get("confirmations", 0)
                >= self.args.inventory_confirmations
                and entry.get("state", PROVISIONAL_VIEW_HINT)
                == PROVISIONAL_VIEW_HINT
            ]
            candidate_entries = entries if entries else [None]
            for entry in candidate_entries:
                fingerprint = make_fingerprint(
                    run_prefix=self.task.run_prefix, order_id=order.id,
                    product_kind=order.kind, candidate=entry,
                    coverage_revision=len(getattr(self, "scan_coverage", ())))
                retry = getattr(
                    self, "candidate_attempt_memory",
                    CandidateAttemptMemory()).decision(fingerprint)
                strict_key = None
                strict_decision = None
                if (entry is not None
                        and self.strict_failure_memory_mode != "off"):
                    strict_key, strict_decision = (
                        self._strict_key_and_decision(order, entry))
                strict_control_blocked = bool(
                    self.strict_failure_memory_mode == "control"
                    and strict_decision is not None
                    and not strict_decision.allowed)
                if strict_control_blocked:
                    self._emit_strict_suppression(
                        order_id=order.id, kind=order.kind, candidate=entry,
                        key=strict_key, decision=strict_decision)
                estimate = completion_estimate(entry is not None)
                feasibility = estimate.feasibility(remaining_hard_s)
                costs = {
                    "estimated_to_reacquire_localize_grasp_s": (
                        estimate.estimated_to_reacquire_localize_grasp_s),
                    "estimated_delivery_s": estimate.estimated_delivery_s,
                    "estimated_place_s": estimate.estimated_place_s,
                    "safety_margin_s": estimate.safety_margin_s,
                }
                state = score_first_candidate_state(
                    validated=bool(
                        entry is not None and entry.get(
                            "validation_state") == LOCALIZATION_VALIDATED),
                    provisional=entry is not None,
                    deferred=order.id in deferred_orders,
                )
                created = match_start
                if entry is not None:
                    try:
                        created = float(entry.get(
                            "candidate_created_monotonic_s", match_start))
                    except (TypeError, ValueError):
                        created = match_start
                status = (
                    "SUPPRESSED" if not retry.allowed or strict_control_blocked else
                    "NEW_EVIDENCE" if retry.new_evidence else "UNTRIED")
                options.append(OrderOption(
                    order_id=order.id,
                    source_index=order.source_index,
                    attempts=order.attempts,
                    candidate_state=state,
                    candidate_count=len(entries),
                    marker_id=(None if entry is None else int(entry["marker_id"])),
                    cost_components=costs,
                    estimated_completion_s=estimate.estimated_completion_s,
                    candidate_id=fingerprint.candidate_id,
                    fingerprint_digest=fingerprint.digest,
                    fingerprint_status=status,
                    new_evidence=retry.new_evidence,
                    reactivation_reason=retry.reason,
                    context_complete=bool(
                        entry is not None and entry.get("context_quality")
                        == CONTEXT_COMPLETE),
                    deadline_feasible=bool(feasibility["deadline_feasible"]),
                    remaining_hard_s=remaining_hard_s,
                    deadline_slack_s=float(feasibility["deadline_slack_s"]),
                    evidence_created_s=max(0.0, created - match_start),
                    association_pair_count=int(
                        (entry or {}).get("confirmations", 0)),
                    sync_delta_ns=(
                        2**63 - 1 if entry is None else abs(int(
                            entry.get("source_yolo_stamp_ns", 0)) - int(
                            entry.get("source_aruco_stamp_ns", 0)))),
                    confirmation_count=int(
                        (entry or {}).get("confirmations", 0)),
                    strict_memory_state=(
                        "UNTRIED" if strict_decision is None
                        else strict_decision.state),
                    strict_retry_allowed=(
                        True if strict_decision is None
                        else strict_decision.allowed),
                    strict_control_active=(
                        self.strict_failure_memory_mode == "control"),
                    strict_candidate_selection_reason=(
                        "strict_memory_not_applicable"
                        if strict_decision is None else strict_decision.reason),
                    strict_failure_equivalence_digest=(
                        None if strict_key is None else strict_key.digest),
                    strict_failure_material_revision=(
                        None if strict_decision is None
                        else strict_decision.material_revision),
                ))
        return options

    def _next_discovery_segments(self) -> list[DiscoverySegment]:
        if self.run_coverage is None:
            return []
        pending_kinds = {
            normalize_kind(order.kind) for order in self.task.orders
            if order.status == "pending"}
        current_station = None
        if self.latest_base_pose is not None:
            current_station = min(
                range(len(SCAN_X)),
                key=lambda index: abs(SCAN_X[index] - self.latest_base_pose[0]))
        segments = []
        uncovered = self.run_coverage.uncovered_keys_from_cursor()
        # The established worker consumes the current route cursor.  Until a
        # separately validated route-jump primitive exists, never claim that
        # a later value-ranked segment is the one actually executed.
        for route_index, key in enumerate(
                uncovered[:1], start=self.run_coverage.cursor_index):
            record = self.run_coverage.records[key]
            observed = {normalize_kind(kind) for kind in record.observed_kinds}
            travel = (0.0 if current_station is None else
                      abs(key.station_id - current_station)
                      * getattr(
                          self.args, "discovery_station_travel_estimate",
                          12.0))
            segments.append(DiscoverySegment(
                segment_id=stable_segment_id(
                    key.run_prefix, key.station_id, key.pose_name,
                    key.shelf_band),
                coverage_key=key,
                estimated_cost_s=(
                    getattr(self.args, "discovery_segment_estimate", 2.0)
                    + travel),
                can_observe_missing_pending_kind=bool(
                    pending_kinds.intersection(observed)),
                historical_context_complete_yield=0,
                current_run_candidate_yield=len(record.candidate_ids_created),
                travel_cost_s=travel,
                route_index=route_index,
            ))
        return segments

    def _select_order(self):
        assert self.task is not None
        candidates = [
            order for order in self.task.orders
            if order.status == "pending"
        ]
        if not candidates:
            return None, None
        if self.args.memory_mode == "off":
            order = min(candidates, key=lambda item: (
                item.attempts, GRASP_COST.get(item.kind, 10.0),
                item.source_index))
            entry = None
            costs = {
                "estimated_candidate_revisit_s": 180.0,
                "estimated_strict_localization_s": 0.0,
                "estimated_grasp_s": float(GRASP_COST.get(order.kind, 10.0)),
                "estimated_delivery_s": 90.0,
                "estimated_place_s": 0.0,
                "explicit_failure_penalty_s": float(order.attempts * 30.0),
            }
            reason = "baseline_grasp_cost"
            options = []
        else:
            options = self._score_first_options()
            if getattr(self, "coverage_mode", "shadow") != "resume_only":
                selected = score_first_select_order(options)
                if selected is not None:
                    order = next(item for item in candidates
                                 if item.id == selected.order_id)
                    entry = (
                        None if selected.marker_id is None else
                        self.inventory.get(selected.marker_id))
                    costs = dict(selected.cost_components)
                    reason = "minimum_estimated_completion_time"
                    # Fall through to the unchanged shared selection record.
                    self.last_selection = {
                        "strategy": reason,
                        "list_order_id": candidates[0].id,
                        "selected_order_id": order.id,
                        "candidate_state": selected.candidate_state,
                        "costs": costs,
                        "estimated_completion_s": round(
                            sum(costs.values()), 3),
                        "selected_candidate_id": selected.candidate_id,
                        "selected_fingerprint_digest": (
                            selected.fingerprint_digest),
                        "selected_fingerprint_status": (
                            selected.fingerprint_status),
                        "new_evidence": selected.new_evidence,
                        "reactivation_reason": selected.reactivation_reason,
                        "strict_failure_memory_state": (
                            selected.strict_memory_state),
                        "strict_retry_allowed": selected.strict_retry_allowed,
                        "strict_candidate_selection_reason": (
                            selected.strict_candidate_selection_reason),
                        "strict_failure_equivalence_digest": (
                            selected.strict_failure_equivalence_digest),
                        "strict_failure_material_revision": (
                            selected.strict_failure_material_revision),
                        "strict_failure_memory_mode": (
                            self.strict_failure_memory_mode),
                        "deadline_feasible": selected.deadline_feasible,
                        "deadline_slack_s": round(
                            selected.deadline_slack_s, 3),
                        "orders_deferred": sorted(
                            getattr(self, "deferred_orders", {})),
                        "remaining_match_time_s": round(max(
                            0.0, getattr(self.args, "match_timeout", 600.0)
                            - (time.monotonic()
                               - (getattr(self, "task_started_at", None)
                                  or time.monotonic()))),
                            3),
                        "candidates": [item.as_dict() for item in options],
                    }
                    return order, (None if entry is None else
                                   int(entry["marker_id"]))
            candidate_options = [
                item for item in options if item.marker_id is not None]
            selected = score_first_select_order(candidate_options)
            if selected is None:
                blocked_deadline = [
                    item for item in candidate_options
                    if not item.deadline_feasible]
                suppressed = [
                    item for item in candidate_options
                    if item.fingerprint_status == "SUPPRESSED"]
                elapsed = max(
                    0.0, time.monotonic()
                    - (self.task_started_at or time.monotonic()))
                segments = self._next_discovery_segments()
                decision = self.anytime_discovery_policy.decide(
                    elapsed_s=elapsed,
                    hard_deadline_s=self.args.match_timeout,
                    pending_orders=candidates,
                    current_run_coverage=(
                        None if self.run_coverage is None else
                        self.run_coverage.snapshot()),
                    discovery_cursor=(
                        None if self.run_coverage is None else
                        self.run_coverage.cursor_index),
                    current_candidates=candidate_options,
                    candidate_completion_estimate_s=(
                        completion_estimate(True).estimated_completion_s),
                    next_scan_segments=segments,
                    safety_margin_s=getattr(
                        self.args, "discovery_safety_margin", 6.0),
                    candidate_attempt_available=bool(candidate_options),
                    candidate_attempt_feasible=False)
                self.event_log.emit(
                    "anytime_discovery_decision", **decision.as_dict())
                if (decision.action == START_DISCOVERY_SEGMENT
                        and decision.selected_segment is not None):
                    order = min(candidates, key=lambda item: item.source_index)
                    entry = None
                    segment = decision.selected_segment
                    costs = {
                        "estimated_scan_segment_s": (
                            segment.estimated_cost_s),
                        "completion_reserve_s": (
                            decision.completion_reserve_s),
                        "discovery_safety_margin_s": (
                            decision.safety_margin_s),
                    }
                    reason = "anytime_discovery_segment"
                    selected = next(
                        item for item in options
                        if item.order_id == order.id and item.marker_id is None)
                    self.last_selection = {
                        "strategy": reason,
                        "selected_discovery_segment": segment.as_dict(),
                        "available_discovery_budget_s": (
                            decision.available_discovery_budget_s),
                        "completion_reserve_s": decision.completion_reserve_s,
                        "deadline_feasible": True,
                        "candidates": [item.as_dict() for item in options],
                    }
                    return order, None
                for item in blocked_deadline:
                    self.event_log.emit(
                        "candidate_deadline_feasibility",
                        order_id=item.order_id,
                        candidate_id=item.candidate_id,
                        fingerprint_digest=item.fingerprint_digest,
                        deadline_feasible=False,
                        estimated_completion_s=item.estimated_completion_s,
                        remaining_hard_s=item.remaining_hard_s,
                        deadline_slack_s=item.deadline_slack_s,
                        estimate_sample_count=completion_estimate(
                            item.marker_id is not None).estimate_sample_count,
                        estimate_source=completion_estimate(
                            item.marker_id is not None).estimate_source)
                self.last_selection = {
                    "no_selection_reason": (
                        decision.reason),
                    "anytime_discovery_decision": decision.as_dict(),
                    "candidates": [item.as_dict() for item in options],
                    "suppressed_repeat_count": len(suppressed),
                    "deadline_infeasible_count": len(blocked_deadline),
                }
                return None, None
            order = next(item for item in candidates
                          if item.id == selected.order_id)
            entry = (
                None if selected.marker_id is None
                else self.inventory.get(selected.marker_id))
            costs = dict(selected.cost_components)
            # Keep the established telemetry label for backwards-compatible
            # consumers; candidate_state records the new evidence tier.
            reason = "minimum_estimated_completion_time"
        self.last_selection = {
            "strategy": reason,
            "list_order_id": candidates[0].id,
            "selected_order_id": order.id,
            "candidate_state": (
                None if not options else
                next(item.candidate_state for item in options
                     if item.order_id == order.id)),
            "costs": costs,
            "estimated_completion_s": round(sum(costs.values()), 3),
            "selected_candidate_id": (
                None if not options else selected.candidate_id),
            "selected_fingerprint_digest": (
                None if not options else selected.fingerprint_digest),
            "selected_fingerprint_status": (
                None if not options else selected.fingerprint_status),
            "new_evidence": (
                False if not options else selected.new_evidence),
            "reactivation_reason": (
                None if not options else selected.reactivation_reason),
            "strict_failure_memory_state": (
                None if not options else selected.strict_memory_state),
            "strict_retry_allowed": (
                True if not options else selected.strict_retry_allowed),
            "strict_candidate_selection_reason": (
                None if not options
                else selected.strict_candidate_selection_reason),
            "strict_failure_equivalence_digest": (
                None if not options
                else selected.strict_failure_equivalence_digest),
            "strict_failure_material_revision": (
                None if not options
                else selected.strict_failure_material_revision),
            "strict_failure_memory_mode": self.strict_failure_memory_mode,
            "deadline_feasible": (
                True if not options else selected.deadline_feasible),
            "deadline_slack_s": (
                None if not options else round(selected.deadline_slack_s, 3)),
            "orders_deferred": sorted(getattr(self, "deferred_orders", {})),
            "remaining_match_time_s": (
                None if getattr(self, "task_started_at", None) is None else round(max(
                    0.0, self.args.match_timeout
                    - (time.monotonic() - self.task_started_at)), 3)),
            "candidates": [item.as_dict() for item in options],
        }
        marker_id = None if entry is None else int(entry["marker_id"])
        return order, marker_id

    def stop(self) -> None:
        self._publish_stop()
        if self.worker is not None and self.worker.poll() is None:
            self.worker.terminate()
            try:
                self.worker.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait(timeout=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="formal supermarket multi-order task runner")
    parser.add_argument("--worker", default=str(DEFAULT_WORKER))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--max-scan-cycles", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--inventory-confirmations", type=int, default=3)
    parser.add_argument(
        "--candidate-attempt-budget", type=float, default=45.0,
        help="maximum seconds from provisional revisit to validation")
    parser.add_argument(
        "--order-timeout", type=float, default=0.0,
        help="per-order timeout in seconds; 0 disables it")
    parser.add_argument("--match-timeout", type=float, default=600.0)
    parser.add_argument(
        "--discovery-segment-estimate", type=float, default=2.0,
        help="R9 observed pose maximum 1.46s rounded up for admission")
    parser.add_argument(
        "--discovery-station-travel-estimate", type=float, default=12.0)
    parser.add_argument(
        "--discovery-safety-margin", type=float, default=6.0)
    parser.add_argument("--runtime-dir", default="/tmp/supermarket_competition")
    parser.add_argument("--summary-file")
    parser.add_argument(
        "--memory-mode", choices=("off", "run_inventory"),
        default=os.getenv("SUPERMARKET_MEMORY_MODE", "off"))
    parser.add_argument(
        "--product-seed", default=os.getenv("SUPERMARKET_SEED"))
    parser.add_argument(
        "--obstacle-seed", default=os.getenv("SUPERMARKET_OBSTACLE_SEED"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not Path(args.worker).is_file():
        parser.error(f"worker not found: {args.worker}")
    if not Path(args.weights).is_file():
        parser.error(f"weights not found: {args.weights}")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if (args.max_scan_cycles < 1 or args.max_attempts < 1
            or args.inventory_confirmations < 1):
        parser.error("scan cycles, attempts, and confirmations must be >= 1")
    if args.order_timeout < 0.0:
        parser.error("--order-timeout must be >= 0")
    if args.candidate_attempt_budget <= 0.0:
        parser.error("--candidate-attempt-budget must be positive")
    if args.match_timeout <= 0.0:
        parser.error("--match-timeout must be positive")
    if (args.discovery_segment_estimate <= 0.0
            or args.discovery_station_travel_estimate < 0.0
            or args.discovery_safety_margin < 0.0):
        parser.error("discovery estimates must be non-negative and segment positive")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = CompetitionRunner(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
