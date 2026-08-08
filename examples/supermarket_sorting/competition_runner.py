#!/usr/bin/env python3
"""Formal five-order entry point for the supermarket competition.

The proven single-item controller remains an isolated worker process.  This
node owns the match lifecycle: it receives the transient task, validates it,
selects orders, supervises workers, records results, and continues after an
individual item fails.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from competition_task import (
    CompetitionTask,
    GRASP_COST,
    TaskMessageError,
    associate_detection_marker,
    marker_arguments,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_WORKER = HERE / "integrated_nav_pick_place.py"
DEFAULT_WEIGHTS = HERE / "perception" / "checkpoints" / "best.pt"


def atomic_write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:96] or "run"


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
        self.finished = False
        self.latest_yolo: tuple[int, list[dict]] | None = None
        self.latest_aruco: tuple[int, list[dict]] | None = None
        self.last_inventory_pair: tuple[int, int] | None = None
        self.last_inventory_yolo_stamp: int | None = None
        self.inventory: dict[int, dict] = {}

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
        self.stop_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
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
        self.latest_yolo = None
        self.latest_aruco = None
        self.last_inventory_pair = None
        self.last_inventory_yolo_stamp = None
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
            elif (self.worker_started_at is not None
                  and now - self.worker_started_at >= self.args.order_timeout):
                self._request_worker_stop("order_timeout")
            return

        if self.task is None:
            self._publish_stop()
            return
        if match_expired:
            self.get_logger().error("match soft deadline reached; stopping safely")
            self._finish_match("match_timeout")
            return

        order, preferred_marker = self._select_order()
        if order is None:
            self._finish_match("orders_terminal")
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
        ]
        command.extend(marker_arguments(self.task.excluded_markers(order.kind)))
        if preferred_marker is not None:
            command.extend(["--preferred-marker-id", str(preferred_marker)])
        if self.args.show:
            command.append("--show")

        self.current_order = order
        self.preferred_marker_id = preferred_marker
        self.worker_result_path = result_path
        self.worker_started_at = time.monotonic()
        self.worker_stop_reason = None
        self.worker_terminate_at = None
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

    def _finish_worker(self, return_code: int) -> None:
        result = self._read_worker_result()
        order = self.current_order
        delivered = (
            return_code == 0
            and result.get("status") == "delivered")
        marker_id = result.get("marker_id")
        if not isinstance(marker_id, int):
            marker_id = None
        error = (
            self.worker_stop_reason
            or result.get("error")
            or result.get("status")
            or f"worker_exit_{return_code}")

        if self.task is not None and order is not None:
            self.task.finish_attempt(
                order,
                delivered=delivered,
                marker_id=marker_id,
                error=None if delivered else str(error),
                max_attempts=self.args.max_attempts,
            )
            level = self.get_logger().info if delivered else self.get_logger().error
            level(
                f"order id={order.id} kind={order.kind} "
                f"status={order.status} marker={marker_id} "
                f"attempts={order.attempts}")
            self._write_summary("worker_finished")

        self.worker = None
        self.worker_started_at = None
        self.worker_result_path = None
        self.current_order = None
        self.preferred_marker_id = None
        self.worker_stop_reason = None
        self.worker_terminate_at = None
        self._publish_stop()

    def _write_summary(self, reason: str) -> None:
        if self.task is None:
            return
        document = self.task.summary()
        document["reason"] = reason
        document["inventory"] = [
            dict(entry) for _, entry in sorted(self.inventory.items())
        ]
        if self.task_started_at is not None:
            document["elapsed_s"] = round(
                time.monotonic() - self.task_started_at, 3)
        summary_path = (
            Path(self.args.summary_file)
            if self.args.summary_file else
            Path(self.args.runtime_dir)
            / safe_component(self.task.run_prefix)
            / "summary.json")
        atomic_write_json(summary_path, document)

    def _finish_match(self, reason: str) -> None:
        self.finished = True
        self._publish_stop()
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
        self._update_inventory()

    def _update_inventory(self) -> None:
        if self.task is None or self.latest_yolo is None or self.latest_aruco is None:
            return
        yolo_stamp, detections = self.latest_yolo
        aruco_stamp, markers = self.latest_aruco
        if abs(yolo_stamp - aruco_stamp) > 200_000_000:
            return
        pair = (yolo_stamp, aruco_stamp)
        if (pair == self.last_inventory_pair
                or yolo_stamp == self.last_inventory_yolo_stamp):
            return
        self.last_inventory_pair = pair
        self.last_inventory_yolo_stamp = yolo_stamp
        wanted = {order.kind for order in self.task.orders
                  if order.status == "pending"}
        for detection in detections:
            kind = detection.get("class")
            if kind not in wanted:
                continue
            marker = associate_detection_marker(detection, markers)
            if marker is None:
                continue
            marker_id = int(marker["id"])
            previous = self.inventory.get(marker_id)
            confirmations = (
                int(previous["confirmations"]) + 1
                if previous is not None and previous.get("kind") == kind
                else 1)
            try:
                confidence = float(detection.get("conf", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            entry = {
                "marker_id": marker_id,
                "kind": kind,
                "position_world": marker.get("position_world"),
                "confidence": round(max(
                    confidence,
                    float(previous.get("confidence", 0.0))
                    if previous is not None else 0.0), 4),
                "confirmations": min(confirmations, 1000),
                "stamp_ns": yolo_stamp,
            }
            self.inventory[marker_id] = entry
            if confirmations == self.args.inventory_confirmations:
                self.get_logger().info(
                    f"inventory confirmed marker={marker_id} "
                    f"kind={kind} confidence={entry['confidence']:.3f}")

    def _select_order(self):
        assert self.task is not None
        candidates = [
            order for order in self.task.orders
            if order.status == "pending"
            and order.attempts < self.args.max_attempts
        ]
        if not candidates:
            return None, None

        def inventory_for(order):
            excluded = set(self.task.excluded_markers(order.kind))
            entries = [
                entry for marker_id, entry in self.inventory.items()
                if marker_id not in excluded
                and entry.get("kind") == order.kind
                and entry.get("confirmations", 0)
                >= self.args.inventory_confirmations
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

        ranked = []
        for order in candidates:
            entry = inventory_for(order)
            distance = float("inf")
            if entry is not None:
                try:
                    x, y = map(float, entry["position_world"][:2])
                    distance = ((x + 1.94) ** 2 + (y + 3.41) ** 2) ** 0.5
                except (KeyError, TypeError, ValueError):
                    pass
            ranked.append((
                (order.attempts, entry is None, distance,
                 GRASP_COST.get(order.kind, 10.0), order.source_index),
                order, entry))
        _, order, entry = min(ranked, key=lambda item: item[0])
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
        description="formal supermarket five-order task runner")
    parser.add_argument("--worker", default=str(DEFAULT_WORKER))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--max-scan-cycles", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--inventory-confirmations", type=int, default=3)
    parser.add_argument("--order-timeout", type=float, default=150.0)
    parser.add_argument("--match-timeout", type=float, default=570.0)
    parser.add_argument("--runtime-dir", default="/tmp/supermarket_competition")
    parser.add_argument("--summary-file")
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
    if args.order_timeout <= 0.0 or args.match_timeout <= 0.0:
        parser.error("timeouts must be positive")
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
