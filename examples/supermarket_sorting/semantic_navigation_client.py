"""Non-blocking client used by the candidate business worker.

The helper contains no recovery policy.  It submits one semantic session and
reports its terminal result; NavigationSession remains the sole owner of
planning, replanning, backup and alternate-route budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

from geometry_msgs.msg import PoseStamped, Quaternion
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from supermarket_interfaces.action import NavigateSemantic


@dataclass(frozen=True)
class NavigationSnapshot:
    state: str
    done: bool
    succeeded: bool
    code: int | None = None
    detail: str = ""
    remaining_path_m: float = math.inf
    recovery_count: int = 0
    selected_route: str = ""
    block_reason: str = ""


def quaternion_from_yaw(yaw: float) -> Quaternion:
    message = Quaternion()
    message.z = math.sin(0.5 * float(yaw))
    message.w = math.cos(0.5 * float(yaw))
    return message


class SemanticNavigationClient:
    def __init__(self, node) -> None:
        self.node = node
        self.client = ActionClient(
            node,
            NavigateSemantic,
            "/navigate_semantic",
            callback_group=ReentrantCallbackGroup(),
        )
        self._send_future = None
        self._goal_handle = None
        self._result_future = None
        self._feedback = NavigationSnapshot("idle", False, False)
        self._terminal = None
        self._server_deadline = 0.0
        self._pending_goal = None

    @property
    def active(self) -> bool:
        return self._pending_goal is not None and self._terminal is None

    def start(
        self,
        *,
        session_id: str,
        goal: tuple[float, float, float],
        route_profile: str,
        footprint_profile: str,
        no_progress_timeout_s: float = 5.0,
        total_timeout_s: float = 120.0,
    ) -> None:
        self.cancel()
        request = NavigateSemantic.Goal()
        request.session_id = str(session_id)
        request.goal = PoseStamped()
        request.goal.header.frame_id = "odom"
        request.goal.header.stamp = self.node.get_clock().now().to_msg()
        request.goal.pose.position.x = float(goal[0])
        request.goal.pose.position.y = float(goal[1])
        request.goal.pose.orientation = quaternion_from_yaw(goal[2])
        request.route_profile = str(route_profile)
        request.footprint_profile = str(footprint_profile)
        request.exact_goal_required = True
        request.no_progress_timeout_s = float(no_progress_timeout_s)
        request.total_timeout_s = float(total_timeout_s)
        self._pending_goal = request
        self._send_future = None
        self._goal_handle = None
        self._result_future = None
        self._terminal = None
        self._server_deadline = time.monotonic() + 2.0
        self._feedback = NavigationSnapshot("waiting_server", False, False)
        self._try_send()

    def _try_send(self) -> None:
        if self._pending_goal is None or self._send_future is not None:
            return
        if not self.client.server_is_ready():
            return
        self._send_future = self.client.send_goal_async(
            self._pending_goal, feedback_callback=self._feedback_callback
        )
        self._feedback = NavigationSnapshot("sending", False, False)

    def _feedback_callback(self, message) -> None:
        feedback = message.feedback
        self._feedback = NavigationSnapshot(
            state=f"feedback:{int(feedback.state)}",
            done=False,
            succeeded=False,
            remaining_path_m=float(feedback.remaining_path_m),
            recovery_count=int(feedback.recovery_count),
            selected_route=str(feedback.selected_route),
            block_reason=str(feedback.block_reason),
        )

    def poll(self) -> NavigationSnapshot:
        if self._terminal is not None:
            return self._terminal
        if self._pending_goal is None:
            return NavigationSnapshot("idle", False, False)
        self._try_send()
        if self._send_future is None:
            if time.monotonic() >= self._server_deadline:
                self._terminal = NavigationSnapshot(
                    "done", True, False, detail="navigation action server unavailable"
                )
                self._pending_goal = None
                return self._terminal
            return self._feedback
        if self._goal_handle is None:
            if not self._send_future.done():
                return self._feedback
            self._goal_handle = self._send_future.result()
            if self._goal_handle is None or not self._goal_handle.accepted:
                self._terminal = NavigationSnapshot(
                    "done", True, False, detail="navigation session rejected"
                )
                self._pending_goal = None
                return self._terminal
            self._result_future = self._goal_handle.get_result_async()
            self._feedback = NavigationSnapshot("running", False, False)
        if self._result_future is None or not self._result_future.done():
            return self._feedback

        wrapped = self._result_future.result()
        result = wrapped.result
        code = int(result.code)
        action_succeeded = wrapped.status == GoalStatus.STATUS_SUCCEEDED
        exact_succeeded = bool(result.exact_goal_reached)
        self._terminal = NavigationSnapshot(
            state="done",
            done=True,
            succeeded=(
                action_succeeded
                and code == NavigateSemantic.Result.SUCCEEDED
                and exact_succeeded
            ),
            code=code,
            detail=(
                str(result.detail)
                if action_succeeded
                else f"action status={wrapped.status}; {result.detail}"
            ),
            recovery_count=int(result.recovery_count),
            selected_route=str(result.selected_route),
        )
        self._pending_goal = None
        return self._terminal

    def cancel(self) -> None:
        if self._goal_handle is not None and self._result_future is not None:
            if not self._result_future.done():
                self._goal_handle.cancel_goal_async()
        self._pending_goal = None
        self._send_future = None
        self._goal_handle = None
        self._result_future = None
        self._terminal = None
        self._feedback = NavigationSnapshot("idle", False, False)
