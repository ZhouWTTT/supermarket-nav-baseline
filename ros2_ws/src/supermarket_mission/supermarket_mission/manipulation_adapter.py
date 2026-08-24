"""Expose the verified integrated controller as one PickItem -> PlaceItem pair."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future

from supermarket_interfaces.action import PickItem, PlaceItem


repo_root = Path(os.environ.get("SUPERMARKET_REPO_ROOT", "/workspace/baseline"))
examples_dir = repo_root / "examples" / "supermarket_sorting"
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

import yolo_aruco_shelf_pick as pick  # noqa: E402
from integrated_nav_pick_place import (  # noqa: E402
    DELIVERY_PLACE_SLOTS_XY,
    IntegratedNavPickPlace,
)


class ManipulationActionAdapter(Node):
    def __init__(self) -> None:
        super().__init__("supermarket_manipulation_adapter")
        self.declare_parameter("max_scan_cycles", 2)
        self.declare_parameter("pick_timeout_s", 300.0)
        self.declare_parameter("place_timeout_s", 240.0)
        self.executor: MultiThreadedExecutor | None = None
        self.controller: IntegratedNavPickPlace | None = None
        self.active_order_id = ""
        self.pick_reserved = False
        self.place_active = False
        group = ReentrantCallbackGroup()
        self.pick_server = ActionServer(
            self,
            PickItem,
            "/pick_item",
            execute_callback=self._execute_pick,
            goal_callback=self._pick_goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=group,
        )
        self.place_server = ActionServer(
            self,
            PlaceItem,
            "/place_item",
            execute_callback=self._execute_place,
            goal_callback=self._place_goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=group,
        )
        self.create_timer(0.20, self._maintenance_tick, callback_group=group)

    def _pick_goal(self, request) -> GoalResponse:
        if self.controller is not None or self.pick_reserved or self.place_active:
            return GoalResponse.REJECT
        if not request.order_id or not request.kind:
            return GoalResponse.REJECT
        self.pick_reserved = True
        return GoalResponse.ACCEPT

    def _place_goal(self, request) -> GoalResponse:
        controller = self.controller
        if (
            controller is None
            or self.place_active
            or request.order_id != self.active_order_id
            or controller.flow_phase != "pick_complete_hold"
            or request.place_slot >= len(DELIVERY_PLACE_SLOTS_XY)
        ):
            return GoalResponse.REJECT
        self.place_active = True
        return GoalResponse.ACCEPT

    def _build_controller(self, request) -> IntegratedNavPickPlace:
        if self.executor is None:
            raise RuntimeError("adapter executor is not attached")
        controller = IntegratedNavPickPlace(
            request.kind,
            int(self.get_parameter("max_scan_cycles").value),
            False,
            False,
            place_slot=None,
            nav_during_scan=True,
            close_recheck=True,
            return_west_after_place=False,
            managed_lifecycle=True,
            pause_after_grab=True,
        )
        controller.perception_always_on = True
        controller.dynamic_direct_enabled = bool(request.dynamic_direct)
        controller.enable_zhijin_middle_column = bool(
            request.enable_zhijin_middle_column
        )
        controller.configure_external_perception(True)
        controller.excluded_marker_ids = {
            int(value) for value in request.excluded_marker_ids if 0 <= int(value) <= 44
        }
        controller.excluded_slot_keys = set(request.excluded_slot_keys)
        if request.memory_file:
            controller.configure_memory_routing(
                request.memory_file,
                float(request.memory_confidence_threshold or 0.90),
                initial_x=(request.world_x if math.isfinite(request.world_x) else None),
                initial_z=(request.world_z if math.isfinite(request.world_z) else None),
            )
        if request.shelf and request.level and request.column:
            controller.configure_direct_slot_target(
                request.shelf,
                request.level,
                request.column,
                product_y=(request.world_y if math.isfinite(request.world_y) else None),
                product_z=(request.world_z if math.isfinite(request.world_z) else None),
            )
        controller.scan_prefer_west_start = bool(request.scan_start_west)
        return controller

    async def _execute_pick(self, goal_handle):
        result = PickItem.Result()
        request = goal_handle.request
        try:
            self.controller = self._build_controller(request)
            self.active_order_id = request.order_id
            self.executor.add_node(self.controller)
            deadline = time.monotonic() + float(
                self.get_parameter("pick_timeout_s").value
            )
            while rclpy.ok():
                controller = self.controller
                if controller is None:
                    raise RuntimeError("pick controller disappeared")
                feedback = PickItem.Feedback()
                feedback.state = str(controller.state)
                feedback.progress = self._pick_progress(controller.state)
                feedback.detail = str(controller.abort_reason or "")
                goal_handle.publish_feedback(feedback)
                if controller.flow_phase == "pick_complete_hold":
                    goal_handle.succeed()
                    result.code = PickItem.Result.SUCCEEDED
                    result.detail = "verified grasp held for matching PlaceItem"
                    result.object_held = True
                    result.marker_id = int(
                        -1 if controller.target_marker_id is None
                        else controller.target_marker_id
                    )
                    result.resolved_kind = str(controller.target_kind)
                    result.resolved_slot_key = str(controller.target_slot_key() or "")
                    return result
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.code = PickItem.Result.CANCELED
                    result.detail = "pick canceled after bounded posture recovery"
                    await self._recover_and_dispose("pick canceled")
                    return result
                if controller.managed_terminal:
                    goal_handle.abort()
                    result.code = PickItem.Result.GRASP_FAILED
                    result.detail = (
                        controller.managed_terminal_reason
                        or controller.abort_reason
                        or "pick controller terminated"
                    )
                    self._dispose_controller()
                    return result
                if time.monotonic() >= deadline:
                    goal_handle.abort()
                    result.code = PickItem.Result.GRASP_FAILED
                    result.detail = "pick action exceeded its bounded timeout"
                    await self._recover_and_dispose("pick timeout")
                    return result
                await self._sleep(0.20)
        except Exception as exc:
            self.get_logger().error(f"PickItem adapter failure: {exc}")
            if goal_handle.is_active:
                goal_handle.abort()
            result.code = PickItem.Result.INTERNAL_ERROR
            result.detail = str(exc)
            await self._recover_and_dispose("pick adapter internal error")
            return result
        finally:
            self.pick_reserved = False

    async def _execute_place(self, goal_handle):
        result = PlaceItem.Result()
        request = goal_handle.request
        try:
            controller = self.controller
            if controller is None:
                raise RuntimeError("PlaceItem has no held PickItem controller")
            controller.resume_managed_place(int(request.place_slot))
            deadline = time.monotonic() + float(
                self.get_parameter("place_timeout_s").value
            )
            while rclpy.ok():
                controller = self.controller
                if controller is None:
                    raise RuntimeError("place controller disappeared")
                feedback = PlaceItem.Feedback()
                feedback.state = str(controller.flow_phase)
                feedback.progress = self._place_progress(controller)
                feedback.detail = str(controller.terminal_error or "")
                goal_handle.publish_feedback(feedback)
                if controller.placement_completed and controller.flow_phase == "done":
                    goal_handle.succeed()
                    result.code = PlaceItem.Result.SUCCEEDED
                    result.detail = "placement and table-clear retreat completed"
                    result.object_released = True
                    self._dispose_controller()
                    return result
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.code = PlaceItem.Result.CANCELED
                    result.detail = "place canceled after bounded posture recovery"
                    await self._recover_and_dispose("place canceled")
                    return result
                if controller.managed_terminal:
                    if controller.placement_completed:
                        goal_handle.succeed()
                        result.code = PlaceItem.Result.SUCCEEDED
                        result.detail = "placement completed before managed stop"
                        result.object_released = True
                    else:
                        goal_handle.abort()
                        result.code = PlaceItem.Result.PLACE_FAILED
                        result.detail = (
                            controller.managed_terminal_reason
                            or controller.terminal_error
                            or "place controller terminated"
                        )
                    self._dispose_controller()
                    return result
                if time.monotonic() >= deadline:
                    goal_handle.abort()
                    result.code = PlaceItem.Result.PLACE_FAILED
                    result.detail = "place action exceeded its bounded timeout"
                    await self._recover_and_dispose("place timeout")
                    return result
                await self._sleep(0.20)
        except Exception as exc:
            self.get_logger().error(f"PlaceItem adapter failure: {exc}")
            if goal_handle.is_active:
                goal_handle.abort()
            result.code = PlaceItem.Result.INTERNAL_ERROR
            result.detail = str(exc)
            await self._recover_and_dispose("place adapter internal error")
            return result
        finally:
            self.place_active = False

    def _cancel_controller(self, reason: str) -> None:
        controller = self.controller
        if controller is not None:
            if controller.semantic_nav is not None:
                controller.semantic_nav.cancel()
            controller._request_lifecycle_stop(reason)
        self._dispose_controller()

    async def _recover_and_dispose(self, reason: str) -> None:
        """Stop navigation and let the existing safe recovery finish first."""

        controller = self.controller
        if controller is None:
            return
        if controller.semantic_nav is not None:
            controller.semantic_nav.cancel()
        try:
            controller._enter_fatal_recovery(RuntimeError(reason))
        except Exception as exc:
            self.get_logger().error(f"could not start managed recovery: {exc}")
            controller._request_lifecycle_stop(reason)
        deadline = time.monotonic() + 8.0
        while (
            self.controller is controller
            and not controller.managed_terminal
            and time.monotonic() < deadline
            and rclpy.ok()
        ):
            await self._sleep(0.10)
        if self.controller is controller:
            controller.set_twist(0.0, 0.0)
            self._dispose_controller()

    def _maintenance_tick(self) -> None:
        """Dispose terminal orphan controllers left after PickItem returned."""

        controller = self.controller
        if (
            controller is not None
            and not self.pick_reserved
            and not self.place_active
            and controller.managed_terminal
        ):
            self.get_logger().warn(
                "disposing held controller after autonomous terminal state: "
                f"{controller.managed_terminal_reason}"
            )
            self._dispose_controller()

    def _dispose_controller(self) -> None:
        controller, self.controller = self.controller, None
        self.active_order_id = ""
        if controller is None:
            return
        try:
            controller.set_twist(0.0, 0.0)
            controller.cmd_vel_pub.publish(pick.Twist())
        except Exception:
            pass
        if self.executor is not None:
            self.executor.remove_node(controller)
        controller.destroy_node()

    async def _sleep(self, duration_s: float) -> None:
        gate = Future()

        def wake():
            if not gate.done():
                gate.set_result(None)

        timer = self.create_timer(
            duration_s, wake, callback_group=ReentrantCallbackGroup()
        )
        try:
            await gate
        finally:
            timer.cancel()
            self.destroy_timer(timer)

    @staticmethod
    def _pick_progress(state: str) -> float:
        states = [
            pick.STATE_GO_SCAN, pick.STATE_SCAN, pick.STATE_ALIGN,
            pick.STATE_RECHECK, pick.STATE_DEPLOY, pick.STATE_ARM_FORWARD,
            pick.STATE_CLOSE, pick.STATE_LIFT, pick.STATE_RETREAT, pick.STATE_DONE,
        ]
        try:
            return float(states.index(state)) / float(len(states) - 1)
        except ValueError:
            return 0.0

    @staticmethod
    def _place_progress(controller) -> float:
        if controller.flow_phase in {"backup", "restore_height"}:
            return 0.15
        if controller.flow_phase == "nav_to_delivery":
            return 0.35
        if controller.flow_phase == "place":
            return min(0.95, 0.55 + 0.10 * int(controller.place_stage))
        if controller.flow_phase == "done":
            return 1.0
        return 0.0


def main(args=None) -> None:
    rclpy.init(args=args)
    adapter = ManipulationActionAdapter()
    executor = MultiThreadedExecutor(num_threads=8)
    adapter.executor = executor
    executor.add_node(adapter)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        adapter._cancel_controller("adapter shutdown")
        executor.shutdown(timeout_sec=2.0)
        adapter.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
