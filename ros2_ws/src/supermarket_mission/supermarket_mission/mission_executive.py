"""Long-lived owner of order selection and PickItem -> PlaceItem sequencing."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from supermarket_interfaces.action import PickItem, PlaceItem
from .mission_state import FirstOrderScanPolicy, MissionPhase, MissionState


repo_root = Path(os.environ.get("SUPERMARKET_REPO_ROOT", "/workspace/baseline"))
examples_dir = repo_root / "examples" / "supermarket_sorting"
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from competition_task import CompetitionTask, TaskMessageError  # noqa: E402
from memory_matrix import (  # noqa: E402
    MemoryMatrixTracker,
    grasp_eligible_candidates,
    select_memory_hint,
)


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:96] or "run"


class MissionExecutive(Node):
    def __init__(self, memory_tracker: MemoryMatrixTracker) -> None:
        super().__init__("supermarket_mission_executive")
        self.declare_parameter("max_attempts", 2)
        self.declare_parameter("memory_confidence", 0.90)
        self.declare_parameter("enable_zhijin_middle_column", False)
        self.declare_parameter("runtime_dir", "/tmp/supermarket_competition")
        self.declare_parameter("match_timeout_s", 570.0)
        self.memory_tracker = memory_tracker
        self.task: CompetitionTask | None = None
        self.state = MissionState()
        self.current_order = None
        self.current_hint: dict | None = None
        self.failed_slots: dict[str, set[str]] = {}
        self.memory_file = Path("/tmp/memory_matrix_waiting.json")
        self.run_started_at = 0.0
        self.delivered_marker_id = -1
        self.action_kind = ""
        self.send_future = None
        self.goal_handle = None
        self.result_future = None
        self.action_run_prefix = ""
        self._pending_goal = None
        self.action_started_at = 0.0
        self.next_action_send_at = 0.0
        self.action_ever_sent = False
        self.scan_policy = FirstOrderScanPolicy()
        group = ReentrantCallbackGroup()
        self.pick_client = ActionClient(
            self, PickItem, "/pick_item", callback_group=group
        )
        self.place_client = ActionClient(
            self, PlaceItem, "/place_item", callback_group=group
        )
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, "/supermarket_sorting/task", self._task_cb, qos,
            callback_group=group,
        )
        self.perception_pub = self.create_publisher(
            Bool, "/supermarket_sorting/perception_enable", 10
        )
        self.event_pub = self.create_publisher(String, "/mission/events", 50)
        self.create_timer(0.10, self._tick, callback_group=group)

    @property
    def max_attempts(self) -> int:
        return int(self.get_parameter("max_attempts").value)

    def _task_cb(self, message: String) -> None:
        try:
            incoming = CompetitionTask.from_json(message.data)
        except TaskMessageError as exc:
            self.get_logger().error(f"rejecting invalid task: {exc}")
            return
        if self.task is not None and incoming.run_prefix == self.task.run_prefix:
            return
        self._cancel_active_action()
        self.task = incoming
        self.current_order = None
        self.current_hint = None
        self.failed_slots.clear()
        self.scan_policy.reset()
        self.run_started_at = time.monotonic()
        runtime = Path(str(self.get_parameter("runtime_dir").value))
        run_dir = runtime / safe_component(incoming.run_prefix)
        self.memory_file = run_dir / "memory_matrix.json"
        self.memory_tracker.start_run(self.memory_file)
        self.state.new_run(incoming.run_prefix)
        self.perception_pub.publish(Bool(data=True))
        self._event("task_accepted", count=len(incoming.orders))
        self._write_summary("accepted")

    def _tick(self) -> None:
        if self.task is None:
            self.perception_pub.publish(Bool(data=False))
            return
        if (
            time.monotonic() - self.run_started_at
            >= float(self.get_parameter("match_timeout_s").value)
        ):
            self._cancel_active_action()
            self.perception_pub.publish(Bool(data=False))
            self._write_summary("match_timeout")
            self.task = None
            self.state = MissionState()
            return
        self.perception_pub.publish(Bool(data=True))
        if self.action_kind:
            self._poll_action()
            return
        if self.state.phase is not MissionPhase.SELECT_ORDER:
            return
        order, hint = self._select_order()
        if order is None:
            self._write_summary("orders_terminal")
            self.perception_pub.publish(Bool(data=False))
            self.state.transition(MissionPhase.WAIT_TASK)
            return
        self.current_order = order
        self.current_hint = hint
        self.state.select(order.id)
        self._send_pick(order, hint)

    def _select_order(self):
        candidates = [
            order for order in self.task.orders
            if order.status == "pending" and order.attempts < self.max_attempts
        ]
        if not candidates:
            return None, None
        evaluated = []
        threshold = float(self.get_parameter("memory_confidence").value)
        for order in candidates:
            memory_candidates, observer = self.memory_tracker.routing_snapshot(
                order.kind
            )
            memory_candidates = grasp_eligible_candidates(
                order.kind,
                memory_candidates,
                enable_zhijin_middle_column=bool(
                    self.get_parameter("enable_zhijin_middle_column").value
                ),
            )
            hint = select_memory_hint(
                memory_candidates,
                observer,
                threshold,
                exclude_slots=self.failed_slots.get(order.kind, set()),
                reliable_only=False,
            )
            travel = math.inf if hint is None else float(hint["travel"])
            evaluated.append((order, hint, travel))
        order, hint, _travel = min(
            evaluated,
            key=lambda item: (
                item[1] is None,
                item[2],
                item[0].source_index,
            ),
        )
        self._event(
            "order_selected", order_id=order.id, kind=order.kind,
            travel=None if hint is None else _travel,
        )
        return order, None if hint is None else dict(hint)

    def _send_pick(self, order, hint) -> None:
        goal = PickItem.Goal()
        goal.order_id = order.id
        goal.kind = order.kind
        goal.memory_file = str(self.memory_file)
        goal.memory_confidence_threshold = float(
            self.get_parameter("memory_confidence").value
        )
        goal.dynamic_direct = True
        # Preserve wxj semantics exactly: every retry of the first logical
        # order starts from E.  Only after a different order is dispatched do
        # no-hint scans start from A.  A measured memory hint still wins.
        goal.scan_start_west = self.scan_policy.prefer_west(
            order.id, has_memory_hint=hint is not None
        )
        goal.enable_zhijin_middle_column = bool(
            self.get_parameter("enable_zhijin_middle_column").value
        )
        goal.excluded_marker_ids = self.task.excluded_markers(order.kind)
        goal.excluded_slot_keys = sorted(self.failed_slots.get(order.kind, set()))
        goal.world_x = math.nan
        goal.world_y = math.nan
        goal.world_z = math.nan
        if hint is not None:
            goal.slot_key = str(hint.get("slot_key", ""))
            goal.shelf = str(hint.get("shelf", ""))
            goal.level = str(hint.get("level", ""))
            goal.column = str(hint.get("column", ""))
            try:
                scan_x = float(hint.get("x"))
                if math.isfinite(scan_x):
                    goal.world_x = scan_x
            except (TypeError, ValueError):
                pass
            for field in ("world_x", "world_y", "world_z"):
                if field == "world_x":
                    continue
                try:
                    value = float(hint.get(field))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    setattr(goal, field, value)
        self._start_action("pick", self.pick_client, goal, self._pick_feedback)

    def _send_place(self) -> None:
        goal = PlaceItem.Goal()
        goal.order_id = self.current_order.id
        goal.kind = self.current_order.kind
        place_slot = sum(
            order.status == "delivered" for order in self.task.orders
        )
        if place_slot >= 5:
            self._finish_failure("place: all five delivery slots are occupied")
            return
        goal.place_slot = place_slot
        self._start_action("place", self.place_client, goal, self._place_feedback)

    def _start_action(self, kind, client, goal, feedback_callback) -> None:
        self.action_kind = kind
        self.action_run_prefix = self.task.run_prefix
        self.goal_handle = None
        self.result_future = None
        self._pending_goal = (client, goal, feedback_callback)
        self.action_started_at = time.monotonic()
        self.next_action_send_at = self.action_started_at
        self.action_ever_sent = False
        if not client.server_is_ready():
            self.send_future = None
            return
        self.send_future = client.send_goal_async(
            goal, feedback_callback=feedback_callback
        )
        self.action_ever_sent = True

    def _poll_action(self) -> None:
        if self.task is None or self.action_run_prefix != self.task.run_prefix:
            return
        if self.send_future is None:
            client, goal, callback = self._pending_goal
            now = time.monotonic()
            wait_limit = 10.0 if self.action_ever_sent else 5.0
            if now - self.action_started_at >= wait_limit:
                self._finish_failure(
                    f"{self.action_kind} action unavailable after "
                    f"{wait_limit:.0f} seconds"
                )
                return
            if now >= self.next_action_send_at and client.server_is_ready():
                self.send_future = client.send_goal_async(
                    goal, feedback_callback=callback
                )
                self.action_ever_sent = True
            return
        if self.goal_handle is None:
            if not self.send_future.done():
                return
            try:
                self.goal_handle = self.send_future.result()
            except Exception as exc:
                self._finish_failure(
                    f"{self.action_kind} goal transport failed: {exc}"
                )
                return
            if self.goal_handle is None or not self.goal_handle.accepted:
                # A previous canceled controller may still be completing its
                # bounded posture recovery.  Retry the same mission action
                # briefly instead of consuming an order attempt immediately.
                if time.monotonic() - self.action_started_at < 10.0:
                    self.goal_handle = None
                    self.send_future = None
                    self.next_action_send_at = time.monotonic() + 0.25
                    return
                self._finish_failure(f"{self.action_kind} action rejected")
                return
            self.result_future = self.goal_handle.get_result_async()
        if self.result_future is None or not self.result_future.done():
            return
        try:
            wrapped = self.result_future.result()
        except Exception as exc:
            self._finish_failure(
                f"{self.action_kind} result transport failed: {exc}"
            )
            return
        if self.action_kind == "pick":
            self._finish_pick(wrapped)
        else:
            self._finish_place(wrapped)

    def _finish_pick(self, wrapped) -> None:
        result = wrapped.result
        success = (
            wrapped.status == GoalStatus.STATUS_SUCCEEDED
            and result.code == PickItem.Result.SUCCEEDED
            and result.object_held
        )
        self._clear_action()
        if not success:
            if result.resolved_slot_key:
                self.failed_slots.setdefault(self.current_order.kind, set()).add(
                    result.resolved_slot_key
                )
            self._finish_failure(f"pick: {result.detail}")
            return
        self.delivered_marker_id = int(result.marker_id)
        if self.state.phase is MissionPhase.NAV_SHELF:
            self.state.transition(MissionPhase.PICK)
        elif self.state.phase is MissionPhase.VERIFY_SLOT:
            self.state.transition(MissionPhase.PICK)
        self.state.transition(MissionPhase.NAV_DELIVERY)
        self._event("pick_succeeded", marker_id=self.delivered_marker_id)
        self._send_place()

    def _finish_place(self, wrapped) -> None:
        result = wrapped.result
        success = (
            wrapped.status == GoalStatus.STATUS_SUCCEEDED
            and result.code == PlaceItem.Result.SUCCEEDED
            and result.object_released
        )
        self._clear_action()
        if not success:
            self._finish_failure(f"place: {result.detail}")
            return
        if self.state.phase is MissionPhase.NAV_DELIVERY:
            self.state.transition(MissionPhase.PLACE)
        self.state.transition(MissionPhase.UPDATE_MEMORY)
        self.task.finish_attempt(
            self.current_order,
            delivered=True,
            marker_id=(
                None if self.delivered_marker_id < 0 else self.delivered_marker_id
            ),
            max_attempts=self.max_attempts,
        )
        self._event("order_delivered", order_id=self.current_order.id)
        self.current_order = None
        self.current_hint = None
        self.state.order_id = ""
        self.state.transition(MissionPhase.SELECT_ORDER)
        self._write_summary("order_delivered")

    def _finish_failure(self, detail: str) -> None:
        self._clear_action()
        if self.current_order is not None and self.task is not None:
            self.task.finish_attempt(
                self.current_order,
                delivered=False,
                error=detail,
                max_attempts=self.max_attempts,
            )
            self._event(
                "order_failed_attempt",
                order_id=self.current_order.id,
                detail=detail,
            )
        if self.state.phase in {
            MissionPhase.NAV_SHELF, MissionPhase.VERIFY_SLOT, MissionPhase.PICK,
            MissionPhase.NAV_DELIVERY, MissionPhase.PLACE,
        }:
            self.state.release_order()
        self.current_order = None
        self.current_hint = None
        self._write_summary("order_failed_attempt")

    def _pick_feedback(self, message) -> None:
        state = str(message.feedback.state)
        if self.state.phase is MissionPhase.NAV_SHELF and state in {
            pick_state for pick_state in ("align", "recheck")
        }:
            self.state.transition(MissionPhase.VERIFY_SLOT)
        if self.state.phase in {
            MissionPhase.NAV_SHELF, MissionPhase.VERIFY_SLOT
        } and state in {
            "deploy", "arm_forward", "post_extend", "dual_contact",
            "dual_squeeze", "close", "trial_lift", "lift", "retreat", "done",
        }:
            self.state.transition(MissionPhase.PICK)

    def _place_feedback(self, message) -> None:
        phase = str(message.feedback.state)
        if self.state.phase is MissionPhase.NAV_DELIVERY and phase == "place":
            self.state.transition(MissionPhase.PLACE)

    def _cancel_active_action(self) -> None:
        if self.goal_handle is not None:
            if self.result_future is None or not self.result_future.done():
                self.goal_handle.cancel_goal_async()
        elif self.send_future is not None:
            pending = self.send_future
            if pending.done():
                self._cancel_goal_from_send_future(pending)
            else:
                pending.add_done_callback(self._cancel_goal_from_send_future)
        self._clear_action()

    def _cancel_goal_from_send_future(self, future) -> None:
        """Cancel a goal accepted after its run was already superseded."""

        try:
            handle = future.result()
        except Exception:
            return
        if handle is not None and handle.accepted:
            handle.cancel_goal_async()
            self.get_logger().warn(
                "canceled a late-accepted action goal from the previous run"
            )

    def _clear_action(self) -> None:
        self.action_kind = ""
        self.send_future = None
        self.goal_handle = None
        self.result_future = None
        self.action_run_prefix = ""
        self._pending_goal = None
        self.action_started_at = 0.0
        self.next_action_send_at = 0.0
        self.action_ever_sent = False

    def _event(self, event: str, **extra) -> None:
        payload = {
            "time": time.time(),
            "run_prefix": self.state.run_prefix,
            "phase": self.state.phase.name,
            "order_id": self.state.order_id,
            "event": event,
            **extra,
        }
        self.event_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))

    def _write_summary(self, reason: str) -> None:
        if self.task is None:
            return
        document = self.task.summary()
        document.update(
            reason=reason,
            mission_phase=self.state.phase.name,
            memory_matrix_file=str(self.memory_file),
            elapsed_s=round(time.monotonic() - self.run_started_at, 3),
        )
        path = self.memory_file.parent / "mission_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)


def main(args=None) -> None:
    rclpy.init(args=args)
    memory = MemoryMatrixTracker(
        confirmations=3,
        output_path=Path("/tmp/memory_matrix_waiting.json"),
        record_everywhere=True,
    )
    executive = MissionExecutive(memory)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(memory)
    executor.add_node(executive)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executive._cancel_active_action()
        memory.tick_write()
        executor.shutdown(timeout_sec=2.0)
        executive.destroy_node()
        memory.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
