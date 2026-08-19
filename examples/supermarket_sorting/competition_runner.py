#!/usr/bin/env python3
"""Formal multi-order entry point for the supermarket competition.

The proven single-item controller remains an isolated worker process.  This
node owns the match lifecycle: it receives the transient task, validates it,
selects orders, supervises workers, records results, and continues after an
individual item fails.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from competition_task import (
    CompetitionTask,
    GRASP_COST,
    TaskMessageError,
    marker_arguments,
)
from memory_matrix import MemoryMatrixTracker, select_memory_hint


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_WORKER = HERE / "integrated_nav_pick_place.py"
DEFAULT_PERCEPTION_WORKER = HERE / "persistent_perception.py"
DEFAULT_WEIGHTS = HERE / "perception" / "checkpoints" / "best.pt"
DELIVERY_PLACE_SLOT_COUNT = 5


def atomic_write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:96] or "run"


def detector_process_env(device: str) -> dict[str, str]:
    """Build a detector environment consistent with the requested device.

    The Ultralytics release in the official client calls
    ``torch.cuda.synchronize()`` from its profiler whenever CUDA is visible,
    even when inference was explicitly placed on CPU.  The simulator nearly
    fills the 4 GiB GPU, so that unrelated profiler call can OOM a CPU-only
    detector.  Hide CUDA at process start for CPU mode; auto/cuda retain the
    caller's normal device visibility.
    """
    env = os.environ.copy()
    if str(device).lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def clear_path_memory_file() -> None:
    """Remove the shared path-memory file (best effort).

    The arena places five random corridor boxes per run, so a route saved
    under one layout can guide the next match into a stale obstacle and stall
    a leg (observed: 216 s detour + 150 s trunk timeout on run_abac6b642eca).
    The runner therefore clears the file when the client process exits and
    again at startup, so a crash-killed process cannot leak stale routes into
    the next match.  Path resolution mirrors supermarket_navigation.py.
    """
    path_text = os.environ.get(
        "SUPERMARKET_PATH_MEMORY_FILE",
        "/root/.cache/supermarket_path_memory.json")
    path = Path(path_text)
    try:
        if path.exists():
            path.unlink()
            print(
                f"[runner] cleared path memory {path} for a fresh match")
    except OSError as exc:
        # Best effort: a stale cache only costs a replan, never safety.
        print(f"[runner] cannot clear path memory {path}: {exc}")


class CompetitionRunner(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("supermarket_competition_runner")
        self.args = args
        self.task: CompetitionTask | None = None
        self.task_started_at: float | None = None
        self.worker: subprocess.Popen | None = None
        self.worker_uses_external_perception = False
        self.perception_worker: subprocess.Popen | None = None
        self.perception_restart_after = 0.0
        self.perception_started_at: float | None = None
        self.perception_ready_path = Path(
            f"/tmp/supermarket_perception_{os.getpid()}.ready")
        self.worker_started_at: float | None = None
        self.worker_started_wall_at: str | None = None
        self.worker_result_path: Path | None = None
        self.current_order = None
        self.worker_candidate_kinds: set[str] = set()
        self.worker_stop_reason: str | None = None
        self.worker_terminate_at: float | None = None
        self.spawned_workers = 0
        # Match wxj snapshot semantics: retries of the first logical order
        # still use the normal E-side start.  Only after a different order is
        # dispatched does no-hint full scanning switch permanently to A->E.
        self.first_dispatched_order_id: str | None = None
        self.after_first_order_started = False
        self.finished = False
        self.selected_memory_hint: dict | None = None
        self.immediate_retry_order_id: str | None = None
        self.failed_memory_slots: dict[str, set[str]] = {}
        self.memory_path = Path(self.args.runtime_dir) / (
            f"memory_matrix_waiting_{os.getpid()}.json")
        self.order_first_started: dict[str, tuple[float, str]] = {}
        self.order_active_elapsed_s: dict[str, float] = {}
        self.order_timings: dict[str, dict] = {}
        self.attempt_timings: list[dict] = []

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, "/supermarket_sorting/task", self._task_cb, qos)
        self.stop_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.perception_control_publisher = self.create_publisher(
            Bool, "/supermarket_sorting/perception_enable", 10)
        self.create_timer(0.20, self._tick)
        self.memory_tracker = MemoryMatrixTracker(
            confirmations=self.args.memory_confirmations,
            output_path=self.memory_path)
        self.get_logger().info(
            "competition runner ready; waiting for transient task on "
            "/supermarket_sorting/task")
        self._publish_perception_enabled(False)
        self._ensure_persistent_perception()

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
        self.spawned_workers = 0
        self.first_dispatched_order_id = None
        self.after_first_order_started = False
        self.finished = False
        run_dir = (
            Path(self.args.runtime_dir)
            / safe_component(incoming.run_prefix))
        self.memory_path = run_dir / "memory_matrix.json"
        self.memory_tracker.start_run(self.memory_path)
        self.selected_memory_hint = None
        self.immediate_retry_order_id = None
        self.failed_memory_slots.clear()
        self.worker_candidate_kinds.clear()
        self.order_first_started.clear()
        self.order_active_elapsed_s.clear()
        self.order_timings.clear()
        self.attempt_timings.clear()
        self.get_logger().info(
            f"accepted task run={incoming.run_prefix} "
            f"count={len(incoming.orders)} kinds="
            f"{[order.kind for order in incoming.orders]}")
        self._write_summary("accepted")

    def _tick(self) -> None:
        # Never start a second detector while a local-perception worker is
        # active.  If an external daemon dies, however, restart it immediately
        # so the current controller can continue consuming the same topics.
        perception_ready = False
        if self.worker is None or self.worker_uses_external_perception:
            perception_ready = self._ensure_persistent_perception()
        if self.finished:
            self._publish_stop()
            return

        now = time.monotonic()
        match_expired = (
            self.task_started_at is not None
            and now - self.task_started_at >= self.args.match_timeout)
        if self.worker is not None:
            return_code = self.worker.poll()
            if return_code is not None:
                self._finish_worker(return_code)
            elif match_expired:
                self._request_worker_stop("match_timeout")
            elif (self.worker_terminate_at is not None
                  and now - self.worker_terminate_at >= 3.0):
                self.get_logger().error("worker ignored SIGTERM; sending SIGKILL")
                self.worker.kill()
            elif (self.args.order_timeout > 0.0
                  and self.worker_started_at is not None
                  and now - self.worker_started_at >= self.args.order_timeout):
                self._request_worker_stop("order_timeout")
            return

        if self.task is None:
            self._publish_stop()
            return
        if (not self.args.no_persistent_perception
                and self.perception_worker is not None
                and not perception_ready):
            # The model is loading in parallel with task reception.  Do not
            # start a duplicate local detector; the exact readiness sentinel
            # normally appears within a few seconds.
            self._publish_stop()
            return
        if match_expired:
            self.get_logger().error("match soft deadline reached; stopping safely")
            self._finish_match("match_timeout")
            return

        order = self._select_order()
        if order is None:
            self._finish_match("orders_terminal")
            return
        self._start_worker(order)

    def _start_worker(self, order) -> None:
        assert self.task is not None
        self._publish_perception_enabled(False)
        # Slots are consumed only by successful deliveries.  A failed attempt
        # therefore retries the same slot instead of leaving an empty gap on
        # the table.
        place_slot = sum(
            item.status == "delivered" for item in self.task.orders)
        if place_slot >= DELIVERY_PLACE_SLOT_COUNT:
            self.get_logger().error(
                "delivery placement slots exhausted; refusing to stack "
                f"another order on slot {DELIVERY_PLACE_SLOT_COUNT - 1}")
            self._finish_match("placement_slots_exhausted")
            return
        run_dir = (
            Path(self.args.runtime_dir)
            / safe_component(self.task.run_prefix))
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / (
            f"worker_{self.spawned_workers + 1:02d}_dispatch_"
            f"{order.source_index + 1:02d}_attempt_"
            f"{order.attempts + 1}.json")
        if result_path.exists():
            result_path.unlink()

        same_order_drop_retry = (
            self.immediate_retry_order_id == order.id)
        candidate_kinds = (
            [order.kind] if same_order_drop_retry
            else self._candidate_kinds_for(order))
        self.immediate_retry_order_id = None
        command = [
            sys.executable,
            str(Path(self.args.worker).resolve()),
            "--target-kind", order.kind,
            "--order-id", order.id,
            "--weights", str(Path(self.args.weights).resolve()),
            "--confidence", str(self.args.confidence),
            "--max-inference-hz", str(self.args.inference_hz),
            "--device", self.args.device,
            "--max-scan-cycles", str(self.args.max_scan_cycles),
            "--result-file", str(result_path),
            "--formal-mode",
            "--place-slot", str(place_slot),
        ]
        external_perception = self._ensure_persistent_perception()
        if external_perception:
            command.append("--external-perception")
        for kind in candidate_kinds[1:]:
            command.extend(["--candidate-kind", kind])
        excluded_markers = sorted({
            marker_id
            for kind in candidate_kinds
            for marker_id in self.task.excluded_markers(kind)
        })
        command.extend(marker_arguments(excluded_markers))
        excluded_slots = sorted({
            slot_key
            for kind in candidate_kinds
            for slot_key in self.failed_memory_slots.get(kind, set())
        })
        for slot_key in excluded_slots:
            command.extend(["--exclude-slot-key", slot_key])
        command.extend([
            "--memory-file", str(self.memory_path),
            "--memory-confidence-threshold",
            str(self.args.memory_confidence_threshold),
        ])
        # The first physically delivered item always occupies slot zero.
        # Keep that worker alive after placement so it returns to shelf A and
        # records a stationary inventory sweep before the next order starts.
        return_west_after_place = place_slot == 0
        if return_west_after_place:
            command.append("--return-west-after-place")
        scan_hint_x = None
        scan_marker_z = None
        if self.selected_memory_hint is not None:
            try:
                scan_hint_x = float(self.selected_memory_hint["x"])
                scan_marker_z = float(self.selected_memory_hint["z"])
            except (KeyError, TypeError, ValueError):
                scan_hint_x = None
                scan_marker_z = None
        if scan_hint_x is not None:
            command.extend(["--scan-start-x", str(scan_hint_x)])
            if scan_marker_z is not None:
                command.extend(["--scan-marker-z", str(scan_marker_z)])
        # wxj snapshot semantics: the first logical order starts from E,
        # including all of its retries.  Once a different order starts, every
        # later no-hint full scan starts from the westmost shelf A.  A memory
        # hint still takes precedence and routes directly to its shelf/level.
        if self.first_dispatched_order_id is None:
            self.first_dispatched_order_id = order.id
        elif order.id != self.first_dispatched_order_id:
            self.after_first_order_started = True
        scan_start_west = (
            self.after_first_order_started and scan_hint_x is None)
        if scan_start_west:
            command.append("--scan-start-west")
        if self.args.show:
            command.append("--show")

        self.current_order = order
        self.worker_uses_external_perception = external_perception
        self.worker_candidate_kinds = set(candidate_kinds)
        self.spawned_workers += 1
        self.worker_result_path = result_path
        self.worker_started_at = time.monotonic()
        self.worker_started_wall_at = self._wall_time_now()
        self.worker_stop_reason = None
        self.worker_terminate_at = None
        self.get_logger().info(
            f"starting order id={order.id} kind={order.kind} "
            f"place_slot={place_slot + 1}/{DELIVERY_PLACE_SLOT_COUNT} "
            f"attempt={order.attempts + 1}/{self.args.max_attempts} "
            f"start={self.worker_started_wall_at} "
            f"scan_hint_x={scan_hint_x} "
            f"scan_marker_z={scan_marker_z} "
            f"scan_start_west={int(scan_start_west)} "
            f"return_west_after_place="
            f"{int(return_west_after_place)} "
            f"same_order_drop_retry={int(same_order_drop_retry)} "
            f"memory_hint={self.selected_memory_hint} "
            f"persistent_perception={int(external_perception)} "
            f"single_item_candidates={candidate_kinds} "
            f"excluded_markers={excluded_markers} "
            f"excluded_slots={excluded_slots}")
        try:
            self.worker = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=detector_process_env(self.args.device),
                start_new_session=False,
            )
        except OSError as exc:
            self.get_logger().error(f"cannot start order worker: {exc}")
            self.task.finish_attempt(
                order,
                delivered=False,
                error=f"worker_start: {exc}",
                max_attempts=self.args.max_attempts,
            )
            self._ensure_order_timing_started(order)
            self._record_order_timing(order)
            self.current_order = None
            self.worker_uses_external_perception = False
            self.worker_candidate_kinds.clear()
            self.worker_started_at = None
            self.worker_started_wall_at = None
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
            self._publish_perception_enabled(False)
            self._publish_stop()
            self.worker.terminate()

    def _ensure_persistent_perception(self) -> bool:
        """Keep one all-class detector alive across sequential item trips."""
        if self.args.no_persistent_perception:
            return False
        now = time.monotonic()
        if self.perception_worker is not None:
            return_code = self.perception_worker.poll()
            if return_code is None:
                if self.perception_ready_path.exists():
                    return True
                if (self.perception_started_at is not None
                        and now - self.perception_started_at > 30.0):
                    self.get_logger().error(
                        "persistent perception did not become ready within "
                        "30s; terminating it and using local perception for "
                        "the next worker")
                    timed_out_worker = self.perception_worker
                    timed_out_worker.terminate()
                    try:
                        timed_out_worker.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        timed_out_worker.kill()
                        timed_out_worker.wait(timeout=2.0)
                    self.perception_worker = None
                    self.perception_started_at = None
                    self.perception_ready_path.unlink(missing_ok=True)
                    self.perception_restart_after = now + 5.0
                return False
            self.get_logger().error(
                "persistent perception exited unexpectedly with code="
                f"{return_code}; scheduling restart")
            self.perception_worker = None
            self.perception_started_at = None
            self.perception_ready_path.unlink(missing_ok=True)
            self.perception_restart_after = now + 1.0
        if now < self.perception_restart_after:
            return False
        command = [
            sys.executable,
            str(Path(self.args.perception_worker).resolve()),
            "--weights", str(Path(self.args.weights).resolve()),
            "--confidence", str(self.args.confidence),
            "--device", self.args.device,
            "--max-inference-hz", str(self.args.inference_hz),
            "--ready-file", str(self.perception_ready_path),
        ]
        if self.args.show:
            command.append("--publish-result-images")
        self.perception_ready_path.unlink(missing_ok=True)
        try:
            self.perception_worker = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=detector_process_env(self.args.device),
                start_new_session=False,
            )
        except OSError as exc:
            self.get_logger().error(
                f"cannot start persistent perception: {exc}; workers will "
                "fall back to local perception")
            self.perception_worker = None
            self.perception_started_at = None
            self.perception_restart_after = now + 5.0
            return False
        self.perception_started_at = now
        self.get_logger().info(
            "started persistent all-class YOLO/ArUco perception; waiting for "
            "the model-ready handshake; detector_device="
            f"{self.args.device} cuda_visible="
            f"{'hidden' if self.args.device == 'cpu' else 'inherited'}")
        return False

    def _read_worker_result(self) -> dict:
        if self.worker_result_path is None or not self.worker_result_path.exists():
            return {}
        try:
            value = json.loads(self.worker_result_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"cannot read worker result: {exc}")
            return {}

    @staticmethod
    def _wall_time_now() -> str:
        """Return a local, timezone-aware timestamp for human-facing logs."""
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _ensure_order_timing_started(self, order) -> None:
        """Attach the active worker interval to the order it actually chose."""
        started_monotonic = self.worker_started_at or time.monotonic()
        started_wall = self.worker_started_wall_at or self._wall_time_now()
        self.order_first_started.setdefault(
            order.id, (started_monotonic, started_wall))

    def _record_order_timing(
            self, order, worker_result: dict | None = None) -> None:
        """Persist one attempt and report a terminal order's total timing."""
        ended_monotonic = time.monotonic()
        ended_wall = self._wall_time_now()
        started_monotonic = (
            ended_monotonic
            if self.worker_started_at is None
            else self.worker_started_at)
        started_wall = self.worker_started_wall_at or ended_wall
        attempt_elapsed = max(0.0, ended_monotonic - started_monotonic)
        total_active = (
            self.order_active_elapsed_s.get(order.id, 0.0)
            + attempt_elapsed)
        self.order_active_elapsed_s[order.id] = total_active

        attempt_record = {
            "order_id": order.id,
            "kind": order.kind,
            "attempt": order.attempts,
            "started_at": started_wall,
            "ended_at": ended_wall,
            "elapsed_s": round(attempt_elapsed, 3),
            "status_after_attempt": order.status,
        }
        if isinstance(worker_result, dict):
            for key in (
                    "pick_state_elapsed_s",
                    "flow_phase_elapsed_s",
                    "flow_phase_distance_m"):
                value = worker_result.get(key)
                if isinstance(value, dict):
                    attempt_record[key] = dict(value)
            drop_event = worker_result.get("drop_event")
            if isinstance(drop_event, dict):
                attempt_record["drop_event"] = dict(drop_event)
                attempt_record["delivery_completed_by_drop"] = bool(
                    worker_result.get("delivery_completed_by_drop"))
        self.attempt_timings.append(attempt_record)

        first_monotonic, first_wall = self.order_first_started.get(
            order.id, (started_monotonic, started_wall))
        wall_elapsed = max(0.0, ended_monotonic - first_monotonic)
        if order.status in {"delivered", "failed"}:
            timing = {
                "order_id": order.id,
                "kind": order.kind,
                "started_at": first_wall,
                "ended_at": ended_wall,
                "elapsed_s": round(wall_elapsed, 3),
                "active_elapsed_s": round(total_active, 3),
                "last_attempt_elapsed_s": round(attempt_elapsed, 3),
                "attempts": order.attempts,
                "status": order.status,
            }
            self.order_timings[order.id] = timing
            timing_message = (
                f"[order-timing] COMPLETE id={order.id} kind={order.kind} "
                f"status={order.status} start={first_wall} end={ended_wall} "
                f"elapsed={wall_elapsed:.3f}s "
                f"active={total_active:.3f}s attempts={order.attempts}")
            # Keep severity call sites separate: rclpy rejects changing one
            # cached call site's severity between invocations.
            if order.status == "delivered":
                self.get_logger().info(timing_message)
            else:
                self.get_logger().error(timing_message)
        else:
            self.get_logger().warn(
                f"[order-timing] ATTEMPT id={order.id} kind={order.kind} "
                f"attempt={order.attempts} start={started_wall} "
                f"end={ended_wall} elapsed={attempt_elapsed:.3f}s; "
                "order remains pending")

    def _finish_worker(self, return_code: int) -> None:
        self._publish_perception_enabled(False)
        result = self._read_worker_result()
        drop_event = result.get("drop_event")
        if isinstance(drop_event, dict):
            self.get_logger().info(
                "[drop-monitor] worker result="
                + json.dumps(
                    drop_event, ensure_ascii=False, separators=(",", ":")))
        dispatch_order = self.current_order
        order, reported_kind_valid = self._resolve_worker_order(result)
        reported_delivered = (
            return_code == 0
            and result.get("status") == "delivered")
        delivered = bool(
            reported_delivered and reported_kind_valid and order is not None)
        marker_id = result.get("marker_id")
        if not isinstance(marker_id, int):
            marker_id = None
        result_slot = result.get("slot")
        if (not isinstance(result_slot, list)
                or len(result_slot) != 3
                or not all(isinstance(value, str) for value in result_slot)):
            result_slot = None
        if not reported_kind_valid:
            # A malformed or out-of-scope result must not blacklist a marker
            # under the dispatch order's (possibly different) product class.
            marker_id = None
        error = (
            self.worker_stop_reason
            or result.get("error")
            or result.get("status")
            or f"worker_exit_{return_code}")
        if reported_delivered and not reported_kind_valid:
            error = (
                "worker reported a delivered kind that is not an eligible "
                f"pending order: {result.get('kind')!r}")

        if self.task is not None and order is not None:
            if dispatch_order is not None and order.id != dispatch_order.id:
                self.get_logger().info(
                    f"single-item worker selected visible pending order "
                    f"id={order.id} kind={order.kind} instead of dispatch "
                    f"id={dispatch_order.id} kind={dispatch_order.kind}")
            self.task.finish_attempt(
                order,
                delivered=delivered,
                marker_id=marker_id,
                error=None if delivered else str(error),
                max_attempts=self.args.max_attempts,
            )
            if (not delivered
                    and isinstance(drop_event, dict)
                    and drop_event.get("outcome") == "retry"
                    and order.status == "pending"
                    and order.attempts < self.args.max_attempts):
                self.immediate_retry_order_id = order.id
                self.get_logger().info(
                    "[drop-monitor] scheduling immediate same-order worker "
                    f"restart id={order.id} kind={order.kind} "
                    f"next_attempt={order.attempts + 1}")
            if result_slot is not None:
                shelf, level, column = result_slot
                if delivered:
                    # The worker normally publishes this immediately after
                    # shelf exit.  Repeat it here as an idempotent fallback.
                    self.memory_tracker.consume_slot(
                        shelf, level, column, kind=order.kind)
                else:
                    self.failed_memory_slots.setdefault(
                        order.kind, set()).add(
                            f"{level}|{shelf}|{column}")
            self._ensure_order_timing_started(order)
            self._record_order_timing(order, result)
            summary = (
                f"order id={order.id} kind={order.kind} "
                f"status={order.status} marker={marker_id} slot={result_slot} "
                f"attempts={order.attempts}")
            # 注意: rclpy 按"调用点"缓存日志严重级别, 同一行不能在不同调用间
            # 切换 info/error, 否则抛 "Logger severity cannot be changed"。
            # 拆成两个调用点即可各自独立缓存。
            if delivered:
                self.get_logger().info(summary)
            else:
                self.get_logger().error(summary)
            match_elapsed = (
                0.0 if self.task_started_at is None
                else max(0.0, time.monotonic() - self.task_started_at))
            pending_count = sum(
                item.status == "pending" for item in self.task.orders)
            remaining_budget = self.args.target_time - match_elapsed
            allowance = (
                0.0 if pending_count == 0
                else remaining_budget / pending_count)
            self.get_logger().info(
                f"[time-budget] elapsed={match_elapsed:.3f}s "
                f"target={self.args.target_time:.1f}s "
                f"remaining={remaining_budget:.3f}s "
                f"pending={pending_count} "
                f"allowance_per_pending={allowance:.3f}s")
            self._write_summary("worker_finished")

        self.worker = None
        self.worker_started_at = None
        self.worker_started_wall_at = None
        self.worker_result_path = None
        self.current_order = None
        self.worker_uses_external_perception = False
        self.worker_candidate_kinds.clear()
        self.worker_stop_reason = None
        self.worker_terminate_at = None
        self._publish_stop()

    def _resolve_worker_order(self, result: dict):
        """Map a worker's committed class to one eligible pending order."""
        dispatch_order = self.current_order
        if self.task is None or dispatch_order is None:
            return dispatch_order, False
        kind = result.get("kind")
        if (not isinstance(kind, str)
                or kind not in self.worker_candidate_kinds):
            return dispatch_order, False
        candidates = [
            order for order in self.task.orders
            if order.status == "pending"
            and order.attempts < self.args.max_attempts
            and order.kind == kind
        ]
        if not candidates:
            return dispatch_order, False
        if dispatch_order in candidates:
            return dispatch_order, True
        return min(
            candidates,
            key=lambda order: (order.attempts, order.source_index)), True

    def _candidate_kinds_for(self, dispatch_order) -> list[str]:
        """Return pending classes in deterministic single-trip priority order."""
        assert self.task is not None
        eligible = [
            order for order in self.task.orders
            if order.status == "pending"
            and order.attempts < self.args.max_attempts
        ]
        eligible.sort(key=lambda order: (
            order.id != dispatch_order.id,
            order.attempts,
            GRASP_COST.get(order.kind, 10.0),
            order.source_index,
        ))
        kinds = []
        for order in eligible:
            if order.kind not in kinds:
                kinds.append(order.kind)
        return kinds

    def _write_summary(self, reason: str) -> None:
        if self.task is None:
            return
        document = self.task.summary()
        document["reason"] = reason
        document["memory_matrix"] = self.memory_tracker.matrix.to_json()
        document["memory_matrix_file"] = str(self.memory_path)
        document["order_timings"] = [
            dict(record) for record in self.order_timings.values()
        ]
        document["attempt_timings"] = [
            dict(record) for record in self.attempt_timings
        ]
        if self.task_started_at is not None:
            elapsed = max(0.0, time.monotonic() - self.task_started_at)
            document["elapsed_s"] = round(elapsed, 3)
            document["target_time_s"] = self.args.target_time
            document["remaining_to_target_s"] = round(
                self.args.target_time - elapsed, 3)
        summary_path = (
            Path(self.args.summary_file)
            if self.args.summary_file else
            Path(self.args.runtime_dir)
            / safe_component(self.task.run_prefix)
            / "summary.json")
        atomic_write_json(summary_path, document)

    def _finish_match(self, reason: str) -> None:
        self.finished = True
        self._publish_perception_enabled(False)
        self._publish_stop()
        self._write_summary(reason)
        summary = self.task.summary() if self.task is not None else {}
        elapsed = (
            0.0 if self.task_started_at is None
            else max(0.0, time.monotonic() - self.task_started_at))
        self.get_logger().info(
            f"match finished reason={reason} "
            f"delivered={summary.get('delivered', 0)}/"
            f"{summary.get('count', 0)} failed={summary.get('failed', 0)} "
            f"elapsed={elapsed:.3f}s target={self.args.target_time:.1f}s "
            f"within_target={int(elapsed <= self.args.target_time)}")

    def _publish_stop(self) -> None:
        try:
            self.stop_publisher.publish(Twist())
        except Exception:  # noqa: BLE001 - ROS context may already be closed
            pass

    def _publish_perception_enabled(self, enabled: bool) -> None:
        try:
            self.perception_control_publisher.publish(
                Bool(data=bool(enabled)))
        except Exception:  # noqa: BLE001 - ROS context may already be closed
            pass

    def _select_order(self):
        assert self.task is not None
        self.selected_memory_hint = None
        candidates = [
            order for order in self.task.orders
            if order.status == "pending"
            and order.attempts < self.args.max_attempts
        ]
        if not candidates:
            self.immediate_retry_order_id = None
            return None

        if self.immediate_retry_order_id is not None:
            retry = next(
                (order for order in candidates
                 if order.id == self.immediate_retry_order_id),
                None)
            if retry is not None:
                candidates = [retry]
            else:
                self.immediate_retry_order_id = None

        ranked = []
        for order in candidates:
            memory_candidates, observer_xy = (
                self.memory_tracker.routing_snapshot(order.kind))
            hint = select_memory_hint(
                memory_candidates,
                observer_xy,
                self.args.memory_confidence_threshold,
                exclude_slots=self.failed_memory_slots.get(
                    order.kind, set()),
                # 严格模式保留在这里，若 snapshot 兜底实跑效果不好，
                # 注释下一行并恢复这一行即可：
                # reliable_only=True)
                reliable_only=False)
            travel = (
                float("inf") if hint is None
                else float(hint.get("travel", float("inf"))))
            ranked.append((
                (order.attempts, hint is None, travel,
                 GRASP_COST.get(order.kind, 10.0), order.source_index),
                order, hint))
        _, order, hint = min(ranked, key=lambda item: item[0])
        self.selected_memory_hint = (
            None if hint is None else dict(hint))
        return order

    def stop(self) -> None:
        self._publish_perception_enabled(False)
        self._publish_stop()
        if self.worker is not None and self.worker.poll() is None:
            self.worker.terminate()
            try:
                self.worker.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait(timeout=2.0)
        if (self.perception_worker is not None
                and self.perception_worker.poll() is None):
            self.perception_worker.terminate()
            try:
                self.perception_worker.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.perception_worker.kill()
                self.perception_worker.wait(timeout=2.0)
        self.perception_ready_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="formal supermarket multi-order task runner")
    parser.add_argument("--worker", default=str(DEFAULT_WORKER))
    parser.add_argument(
        "--perception-worker", default=str(DEFAULT_PERCEPTION_WORKER))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--no-persistent-perception", action="store_true",
        help="load YOLO/ArUco inside every order worker instead of reusing one "
             "runner-owned detector process")
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument(
        "--inference-hz", type=float, default=12.0,
        help="maximum YOLO source-frame rate during active scan states")
    parser.add_argument("--max-scan-cycles", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--memory-confirmations", "--inventory-confirmations",
        dest="memory_confirmations", type=int, default=3,
        help="YOLO frames required by the memory matrix; the inventory name "
             "is retained as a compatibility alias")
    parser.add_argument(
        "--memory-confidence-threshold", type=float, default=0.90,
        help="minimum confidence for normal memory-directed shelf routing")
    parser.add_argument(
        "--order-timeout", type=float, default=0.0,
        help="per-order timeout in seconds; 0 disables it")
    parser.add_argument("--match-timeout", type=float, default=570.0)
    parser.add_argument(
        "--target-time", type=float, default=400.0,
        help="performance target reported in summaries; does not stop motion")
    parser.add_argument("--runtime-dir", default="/tmp/supermarket_competition")
    parser.add_argument("--summary-file")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not Path(args.worker).is_file():
        parser.error(f"worker not found: {args.worker}")
    if (not args.no_persistent_perception
            and not Path(args.perception_worker).is_file()):
        parser.error(
            f"perception worker not found: {args.perception_worker}")
    if not Path(args.weights).is_file():
        parser.error(f"weights not found: {args.weights}")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if not 0.0 <= args.memory_confidence_threshold <= 1.0:
        parser.error("--memory-confidence-threshold must be in [0, 1]")
    if not 0.0 < args.inference_hz < float("inf"):
        parser.error("--inference-hz must be finite and positive")
    if (args.max_scan_cycles < 1 or args.max_attempts < 1
            or args.memory_confirmations < 1):
        parser.error("scan cycles, attempts, and confirmations must be >= 1")
    if args.order_timeout < 0.0:
        parser.error("--order-timeout must be >= 0")
    if args.match_timeout <= 0.0:
        parser.error("--match-timeout must be positive")
    if args.target_time <= 0.0:
        parser.error("--target-time must be positive")
    return args


def main() -> None:
    from run_log import start_run_log
    start_run_log("competition_runner")
    args = parse_args()
    # A previous client process may have been killed before its cleanup ran.
    # Start every match with an empty path memory so stale obstacle layouts
    # from an earlier run cannot misroute this one.
    clear_path_memory_file()
    rclpy.init()
    node = CompetitionRunner(args)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.add_node(node.memory_tracker)
    try:
        # Do not call rclpy.shutdown() from a timer callback.  In the official
        # Humble image that can wait on the callback currently executing and
        # leave both the runner and its persistent perception child alive
        # after all orders are terminal.  Let the callback set ``finished``;
        # the main thread then leaves the executor and performs orderly cleanup.
        # The runner and matrix tracker retain their default mutually-exclusive
        # callback groups, so each node stays internally serial while the two
        # independent nodes can service callbacks concurrently.
        while rclpy.ok() and not node.finished:
            executor.spin_once(timeout_sec=0.05)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.stop()
        node.memory_tracker.tick_write()
        node.memory_tracker.destroy_node()
        node.destroy_node()
        # Each client process owns one match; drop the shared path memory so
        # the next match replans against its own obstacle layout.
        clear_path_memory_file()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
