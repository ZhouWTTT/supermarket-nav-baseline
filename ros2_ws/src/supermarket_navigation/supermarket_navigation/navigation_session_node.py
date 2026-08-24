"""Semantic navigation action server with one owner for every recovery."""

from __future__ import annotations

import json
import math
import struct
import time
import traceback
from typing import Any

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped
from nav2_msgs.action import BackUp, ComputePathToPose, FollowPath
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.task import Future
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String

from supermarket_interfaces.action import NavigateSemantic
from supermarket_interfaces.msg import MotionMode
from supermarket_interfaces.srv import SetFootprintProfile, SetMotionMode

from .session_state import NavigationSessionState, SessionEvent, SessionPhase
from .topology import ArenaTopology, Point2


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def quaternion_from_yaw(yaw: float):
    from geometry_msgs.msg import Quaternion

    q = Quaternion()
    q.z = math.sin(0.5 * yaw)
    q.w = math.cos(0.5 * yaw)
    return q


class NavigationSessionNode(Node):
    def __init__(self) -> None:
        super().__init__("supermarket_navigation_session")
        self.group = ReentrantCallbackGroup()
        self.declare_parameter("planner_id", "GridBased")
        self.declare_parameter("controller_id", "FollowPath")
        self.declare_parameter("goal_checker_id", "general_goal_checker")
        self.declare_parameter("sensor_stale_s", 0.50)
        self.declare_parameter("planning_timeout_s", 2.0)
        self.declare_parameter("backup_distance_m", 0.15)
        self.declare_parameter("backup_speed_mps", 0.08)
        self.declare_parameter("position_tolerance_m", 0.10)
        self.declare_parameter("yaw_tolerance_rad", 0.15)
        self.sensor_stale_s = float(self.get_parameter("sensor_stale_s").value)
        self.planning_timeout_s = float(
            self.get_parameter("planning_timeout_s").value
        )
        self.backup_distance_m = float(
            self.get_parameter("backup_distance_m").value
        )
        self.backup_speed_mps = float(self.get_parameter("backup_speed_mps").value)
        self.position_tolerance_m = float(
            self.get_parameter("position_tolerance_m").value
        )
        self.yaw_tolerance_rad = float(
            self.get_parameter("yaw_tolerance_rad").value
        )
        self.topology = ArenaTopology.competition_default()
        self.pose: tuple[float, float, float] | None = None
        self.last_odom_at = float("-inf")
        self.last_scan_at = float("-inf")
        self.active_session_id: str | None = None
        self.run_prefix = ""
        self.confirmed_obstacles: tuple[Point2, ...] = ()

        self.compute_client = ActionClient(
            self,
            ComputePathToPose,
            "/compute_path_to_pose",
            callback_group=self.group,
        )
        self.follow_client = ActionClient(
            self, FollowPath, "/follow_path", callback_group=self.group
        )
        self.backup_client = ActionClient(
            self, BackUp, "/backup", callback_group=self.group
        )
        self.motion_mode_client = self.create_client(
            SetMotionMode, "/motion/set_mode", callback_group=self.group
        )
        self.footprint_client = self.create_client(
            SetFootprintProfile,
            "/motion/set_footprint_profile",
            callback_group=self.group,
        )
        self.event_pub = self.create_publisher(String, "/nav/events", 50)
        self.speed_limit_pub = self.create_publisher(
            SpeedLimit, "/speed_limit", 10
        )
        self.create_subscription(
            Odometry, "/nav/odom", self._odom_cb, 20, callback_group=self.group
        )
        self.create_subscription(
            LaserScan,
            "/nav/scan",
            self._scan_cb,
            qos_profile_sensor_data,
            callback_group=self.group,
        )
        self.create_subscription(
            PointCloud2,
            "/nav/confirmed_obstacles",
            self._confirmed_obstacles_cb,
            10,
            callback_group=self.group,
        )
        task_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String,
            "/supermarket_sorting/task",
            self._task_cb,
            task_qos,
            callback_group=self.group,
        )
        self.server = ActionServer(
            self,
            NavigateSemantic,
            "/navigate_semantic",
            execute_callback=self._execute,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.group,
        )

    def _odom_cb(self, message: Odometry) -> None:
        p = message.pose.pose.position
        self.pose = (float(p.x), float(p.y), yaw_from_quaternion(message.pose.pose.orientation))
        self.last_odom_at = time.monotonic()

    def _scan_cb(self, _message: LaserScan) -> None:
        self.last_scan_at = time.monotonic()

    def _confirmed_obstacles_cb(self, message: PointCloud2) -> None:
        offsets = {field.name: field.offset for field in message.fields}
        if "x" not in offsets or "y" not in offsets or message.point_step <= 0:
            self.confirmed_obstacles = ()
            return
        byte_order = ">" if message.is_bigendian else "<"
        data = bytes(message.data)
        points: list[Point2] = []
        for row in range(int(message.height)):
            row_start = row * int(message.row_step)
            for column in range(int(message.width)):
                start = row_start + column * int(message.point_step)
                try:
                    x = struct.unpack_from(byte_order + "f", data, start + offsets["x"])[0]
                    y = struct.unpack_from(byte_order + "f", data, start + offsets["y"])[0]
                except struct.error:
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    points.append(Point2(float(x), float(y)))
        self.confirmed_obstacles = tuple(points)

    def _task_cb(self, message: String) -> None:
        try:
            run_prefix = str(json.loads(message.data)["run_prefix"])
        except (ValueError, KeyError, TypeError):
            return
        if run_prefix != self.run_prefix:
            self.run_prefix = run_prefix
            self.confirmed_obstacles = ()
            self.topology.reset_failures()

    def _goal_callback(self, goal_request) -> GoalResponse:
        if not goal_request.session_id:
            return GoalResponse.REJECT
        if self.active_session_id is not None:
            self.get_logger().warn(
                f"rejecting session={goal_request.session_id}; "
                f"active={self.active_session_id}"
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _sensor_stale(self) -> bool:
        now = time.monotonic()
        return (
            self.pose is None
            or now - self.last_odom_at > self.sensor_stale_s
            or now - self.last_scan_at > self.sensor_stale_s
        )

    async def _execute(self, goal_handle):
        request = goal_handle.request
        self.active_session_id = request.session_id
        result = NavigateSemantic.Result()
        try:
            if self._sensor_stale():
                return await self._finish(
                    goal_handle,
                    result,
                    NavigateSemantic.Result.SENSOR_STALE,
                    "fresh odom and scan are required before navigation",
                    abort=True,
                )

            start = Point2(self.pose[0], self.pose[1])
            target = Point2(
                request.goal.pose.position.x, request.goal.pose.position.y
            )
            footprint_width = self._profile_width(request.footprint_profile)
            obstacle_penalties = self.topology.obstacle_edge_penalties(
                self.confirmed_obstacles, footprint_width
            )
            route_data = self.topology.candidate_routes(
                start,
                target,
                footprint_width_m=footprint_width,
                blocked_edges=obstacle_penalties,
                limit=3,
            )
            if not route_data:
                return await self._finish(
                    goal_handle,
                    result,
                    NavigateSemantic.Result.NO_PATH,
                    "semantic graph has no route compatible with the footprint",
                    abort=True,
                )
            names = tuple(item[0] for item in route_data)
            state = NavigationSessionState(
                session_id=request.session_id,
                route_candidates=names,
                no_progress_timeout_s=(
                    request.no_progress_timeout_s
                    if request.no_progress_timeout_s > 0.0
                    else 5.0
                ),
                total_timeout_s=(
                    request.total_timeout_s
                    if request.total_timeout_s > 0.0
                    else self._default_total_timeout(request.route_profile)
                ),
            )
            state.transition(SessionEvent.START)
            await self._set_motion_mode(MotionMode.STOP, request.session_id)
            footprint_ok = await self._set_footprint(request.footprint_profile)
            if not footprint_ok:
                return await self._finish(
                    goal_handle,
                    result,
                    NavigateSemantic.Result.INTERNAL_ERROR,
                    "failed to apply requested footprint while stopped",
                    abort=True,
                )
            if not await self._set_motion_mode(
                MotionMode.NAVIGATION, request.session_id
            ):
                return await self._finish(
                    goal_handle,
                    result,
                    NavigateSemantic.Result.SAFETY_STOP,
                    "motion arbiter rejected navigation ownership",
                    abort=True,
                )
            self._publish_speed_limit(request.route_profile)

            route_lookup = {name: nodes for name, nodes, _cost in route_data}
            terminal_failures: set[int] = set()
            while not state.terminal:
                if goal_handle.is_cancel_requested:
                    state.transition(SessionEvent.CANCEL)
                    goal_handle.canceled()
                    return await self._finish(
                        goal_handle,
                        result,
                        NavigateSemantic.Result.CANCELED,
                        "navigation canceled",
                    )
                if state.timed_out():
                    state.transition(SessionEvent.EXHAUSTED)
                    return await self._finish(
                        goal_handle,
                        result,
                        NavigateSemantic.Result.TIMEOUT,
                        "navigation session total timeout",
                        abort=True,
                        state=state,
                    )
                selected = state.select_next_route()
                if selected is None:
                    state.transition(SessionEvent.EXHAUSTED)
                    code = self._failure_code_after_exhaustion(terminal_failures)
                    return await self._finish(
                        goal_handle,
                        result,
                        code,
                        "all bounded semantic route candidates failed",
                        abort=True,
                        state=state,
                    )
                state.transition(SessionEvent.ROUTE_SELECTED)
                self._publish_feedback(goal_handle, state, math.inf)
                self._event(state, "route_selected")

                same_route_retry = True
                while same_route_retry:
                    same_route_retry = False
                    path, detail = await self._compute_route_path(
                        route_lookup[selected], request.goal, goal_handle
                    )
                    if path is None:
                        self._event(state, "plan_failed", detail=detail)
                        terminal_failures.add(
                            NavigateSemantic.Result.GOAL_BLOCKED
                            if detail.startswith("goal_blocked:")
                            else NavigateSemantic.Result.NO_PATH
                        )
                        if self.pose is not None:
                            self.topology.mark_route_failed_near(
                                route_lookup[selected], Point2(self.pose[0], self.pose[1])
                            )
                        state.transition(SessionEvent.ALTERNATE_AVAILABLE)
                        state.transition(SessionEvent.ALTERNATE_AVAILABLE)
                        break
                    state.transition(SessionEvent.PLAN_READY)
                    state.update_progress(self._path_length(path))
                    outcome, detail = await self._follow_path(
                        path, state, goal_handle
                    )
                    if outcome == "succeeded":
                        state.transition(SessionEvent.APPROACH_STARTED)
                        exact = self._exact_goal_reached(request.goal)
                        if exact:
                            state.transition(SessionEvent.EXACT_GOAL_REACHED)
                            goal_handle.succeed()
                            return await self._finish(
                                goal_handle,
                                result,
                                NavigateSemantic.Result.SUCCEEDED,
                                "exact requested goal reached",
                                state=state,
                                exact=True,
                            )
                        state.transition(SessionEvent.EXHAUSTED)
                        return await self._finish(
                            goal_handle,
                            result,
                            NavigateSemantic.Result.GOAL_BLOCKED,
                            "path follower ended without reaching the requested pose",
                            abort=True,
                            state=state,
                            exact=False,
                        )
                    if outcome == "canceled":
                        state.transition(SessionEvent.CANCEL)
                        goal_handle.canceled()
                        return await self._finish(
                            goal_handle,
                            result,
                            NavigateSemantic.Result.CANCELED,
                            detail,
                            state=state,
                        )
                    if outcome == "sensor_stale":
                        state.transition(SessionEvent.EXHAUSTED)
                        return await self._finish(
                            goal_handle,
                            result,
                            NavigateSemantic.Result.SENSOR_STALE,
                            detail,
                            abort=True,
                            state=state,
                        )
                    if outcome == "path_invalid" and state.request_same_route_replan():
                        state.transition(SessionEvent.PATH_INVALID)
                        same_route_retry = True
                        self._event(state, "same_route_replan_preserving_context")
                        continue

                    terminal_failures.add(
                        NavigateSemantic.Result.NO_PROGRESS
                        if outcome == "no_progress"
                        else NavigateSemantic.Result.NO_PATH
                    )
                    state.transition(SessionEvent.NO_PROGRESS)
                    anchor_index = state.record_block(
                        self.pose[0], self.pose[1], detail
                    )
                    self.topology.mark_route_failed_near(
                        route_lookup[selected], Point2(self.pose[0], self.pose[1])
                    )
                    if state.may_backup_at(anchor_index):
                        state.transition(SessionEvent.BACKUP_ALLOWED)
                        state.spend_backup(anchor_index)
                        backed_up = await self._backup_once(goal_handle)
                        self._event(
                            state,
                            "backup_complete" if backed_up else "backup_rejected",
                            anchor_index=anchor_index,
                        )
                        state.transition(SessionEvent.BACKUP_DONE)
                    else:
                        state.transition(SessionEvent.BACKUP_SKIPPED)
                    state.transition(SessionEvent.ALTERNATE_AVAILABLE)
                    break

            raise RuntimeError("navigation loop reached an impossible terminal state")
        except Exception as exc:  # keep the safety chain fail-closed
            self.get_logger().error(
                f"navigation session failed: {exc}\n{traceback.format_exc()}"
            )
            if goal_handle.is_active:
                goal_handle.abort()
            return await self._finish(
                goal_handle,
                result,
                NavigateSemantic.Result.INTERNAL_ERROR,
                f"internal error: {exc}",
            )
        finally:
            await self._set_motion_mode(MotionMode.STOP, request.session_id)
            self.active_session_id = None

    async def _compute_route_path(self, nodes, requested_goal, goal_handle):
        if not self.compute_client.wait_for_server(timeout_sec=0.5):
            return None, "planner action server unavailable"
        current = self._pose_stamped_from_current()
        targets: list[PoseStamped] = []
        route_points = [node.point for node in nodes[1:]]
        for index, point in enumerate(route_points):
            target = PoseStamped()
            target.header.frame_id = "odom"
            target.header.stamp = self.get_clock().now().to_msg()
            target.pose.position.x = point.x
            target.pose.position.y = point.y
            if index + 1 < len(route_points):
                following = route_points[index + 1]
                yaw = math.atan2(following.y - point.y, following.x - point.x)
            else:
                yaw = yaw_from_quaternion(requested_goal.pose.orientation)
            target.pose.orientation = quaternion_from_yaw(yaw)
            targets.append(target)
        if not targets or self._pose_distance(targets[-1], requested_goal) > 0.03:
            targets.append(requested_goal)
        else:
            targets[-1] = requested_goal

        combined = Path()
        combined.header.frame_id = "odom"
        combined.header.stamp = self.get_clock().now().to_msg()
        for target in targets:
            if goal_handle.is_cancel_requested:
                return None, "canceled during planning"
            goal = ComputePathToPose.Goal()
            goal.start = current
            goal.goal = target
            goal.planner_id = str(self.get_parameter("planner_id").value)
            goal.use_start = True
            send_future = self.compute_client.send_goal_async(goal)
            if not await self._wait_future(send_future, self.planning_timeout_s):
                return None, "planner goal acceptance timeout"
            planner_handle = send_future.result()
            if planner_handle is None or not planner_handle.accepted:
                return None, "planner rejected route segment"
            result_future = planner_handle.get_result_async()
            if not await self._wait_future(result_future, self.planning_timeout_s):
                await planner_handle.cancel_goal_async()
                return None, "planner execution exceeded 2 seconds"
            wrapped = result_future.result()
            if wrapped.status != GoalStatus.STATUS_SUCCEEDED or not wrapped.result.path.poses:
                return None, f"planner failed with status={wrapped.status}"
            poses = list(wrapped.result.path.poses)
            if (
                target is targets[-1]
                and self._pose_distance(poses[-1], requested_goal)
                > 0.5 * self.position_tolerance_m
            ):
                return None, "goal_blocked: planner returned a relocated endpoint"
            if combined.poses and poses:
                poses = poses[1:]
            combined.poses.extend(poses)
            current = target
        return combined, "planned"

    async def _follow_path(self, path, state, goal_handle):
        if not self.follow_client.wait_for_server(timeout_sec=0.5):
            return "path_invalid", "controller action server unavailable"
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = str(self.get_parameter("controller_id").value)
        goal.goal_checker_id = str(self.get_parameter("goal_checker_id").value)

        def feedback_callback(feedback_message):
            remaining = float(feedback_message.feedback.distance_to_goal)
            state.update_progress(remaining)
            self._publish_feedback(goal_handle, state, remaining)

        send_future = self.follow_client.send_goal_async(
            goal, feedback_callback=feedback_callback
        )
        if not await self._wait_future(send_future, 1.0):
            return "path_invalid", "controller goal acceptance timeout"
        follow_handle = send_future.result()
        if follow_handle is None or not follow_handle.accepted:
            return "path_invalid", "controller rejected path"
        result_future = follow_handle.get_result_async()
        if self.pose is not None:
            state.update_displacement(self.pose[0], self.pose[1])
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                await follow_handle.cancel_goal_async()
                return "canceled", "canceled during path following"
            if self._sensor_stale():
                await follow_handle.cancel_goal_async()
                return "sensor_stale", "odom or scan stale during path following"
            if state.timed_out():
                await follow_handle.cancel_goal_async()
                return "path_invalid", "session total timeout while following"
            if self.pose is not None:
                state.update_displacement(self.pose[0], self.pose[1])
            if state.no_progress():
                await follow_handle.cancel_goal_async()
                return "no_progress", "remaining path did not improve for bounded interval"
            await self._sleep(0.05)
        wrapped = result_future.result()
        if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
            return "succeeded", "path followed"
        if wrapped.status == GoalStatus.STATUS_CANCELED:
            return "canceled", "path action canceled"
        if state.no_progress():
            return "no_progress", "controller aborted after measured progress stopped"
        return "path_invalid", f"path follower failed with status={wrapped.status}"

    async def _backup_once(self, goal_handle) -> bool:
        if not self.backup_client.wait_for_server(timeout_sec=0.5):
            return False
        goal = BackUp.Goal()
        goal.target = Point(x=-abs(self.backup_distance_m), y=0.0, z=0.0)
        goal.speed = abs(self.backup_speed_mps)
        goal.time_allowance = Duration(sec=5, nanosec=0)
        send_future = self.backup_client.send_goal_async(goal)
        if not await self._wait_future(send_future, 1.0):
            return False
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return False
        result_future = handle.get_result_async()
        while not result_future.done():
            if goal_handle.is_cancel_requested or self._sensor_stale():
                await handle.cancel_goal_async()
                return False
            await self._sleep(0.05)
        return result_future.result().status == GoalStatus.STATUS_SUCCEEDED

    async def _set_motion_mode(self, mode: int, owner: str) -> bool:
        if not self.motion_mode_client.wait_for_service(timeout_sec=0.5):
            return False
        request = SetMotionMode.Request()
        request.mode = int(mode)
        request.owner = owner
        request.reset_emergency_stop = False
        future = self.motion_mode_client.call_async(request)
        if not await self._wait_future(future, 1.0):
            return False
        response = future.result()
        return bool(response and response.accepted)

    async def _set_footprint(self, profile: str) -> bool:
        if not self.footprint_client.wait_for_service(timeout_sec=0.5):
            return False
        # STOP status and this service response travel through independent ROS
        # callbacks.  Retry the stopped-state check briefly so scheduling
        # order cannot turn a valid atomic switch into an intermittent fault.
        for _attempt in range(5):
            request = SetFootprintProfile.Request()
            request.profile = profile or "COMPACT_TRANSIT"
            future = self.footprint_client.call_async(request)
            if not await self._wait_future(future, 1.0):
                return False
            response = future.result()
            if response and response.applied:
                return True
            if response is None or "motion mode STOP" not in response.detail:
                return False
            await self._sleep(0.05)
        return False

    async def _wait_future(self, future, timeout_s: float) -> bool:
        """Await a ROS future with a ROS timer, not an asyncio event loop.

        Humble's executor advances action callback coroutines directly; there
        is no ambient asyncio loop, so asyncio.sleep()/wait_for() are invalid.
        A rclpy Future yields control back to the executor correctly.
        """
        if future.done():
            return True
        gate = Future()

        def completed(_future) -> None:
            if not gate.done():
                gate.set_result(True)

        def timed_out() -> None:
            if not gate.done():
                gate.set_result(False)

        future.add_done_callback(completed)
        timer = self.create_timer(
            max(0.001, float(timeout_s)), timed_out, callback_group=self.group
        )
        try:
            return bool(await gate)
        finally:
            timer.cancel()
            self.destroy_timer(timer)

    async def _sleep(self, duration_s: float) -> None:
        gate = Future()

        def wake() -> None:
            if not gate.done():
                gate.set_result(None)

        timer = self.create_timer(
            max(0.001, float(duration_s)), wake, callback_group=self.group
        )
        try:
            await gate
        finally:
            timer.cancel()
            self.destroy_timer(timer)

    def _pose_stamped_from_current(self) -> PoseStamped:
        message = PoseStamped()
        message.header.frame_id = "odom"
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.position.x = self.pose[0]
        message.pose.position.y = self.pose[1]
        message.pose.orientation = quaternion_from_yaw(self.pose[2])
        return message

    def _exact_goal_reached(self, goal: PoseStamped) -> bool:
        if self.pose is None:
            return False
        position_error = math.hypot(
            self.pose[0] - goal.pose.position.x,
            self.pose[1] - goal.pose.position.y,
        )
        goal_yaw = yaw_from_quaternion(goal.pose.orientation)
        yaw_error = abs((self.pose[2] - goal_yaw + math.pi) % (2.0 * math.pi) - math.pi)
        return (
            position_error <= self.position_tolerance_m
            and yaw_error <= self.yaw_tolerance_rad
        )

    @staticmethod
    def _pose_distance(a: PoseStamped, b: PoseStamped) -> float:
        return math.hypot(
            a.pose.position.x - b.pose.position.x,
            a.pose.position.y - b.pose.position.y,
        )

    @staticmethod
    def _path_length(path: Path) -> float:
        return sum(
            math.hypot(
                b.pose.position.x - a.pose.position.x,
                b.pose.position.y - a.pose.position.y,
            )
            for a, b in zip(path.poses, path.poses[1:])
        )

    @staticmethod
    def _profile_width(profile: str) -> float:
        return {
            "COMPACT_TRANSIT": 0.48,
            "LOADED_TRANSIT": 0.54,
            "SHELF_APPROACH": 0.56,
            "DELIVERY_APPROACH": 0.58,
            "MANIPULATION_EXTENDED": 1.10,
        }.get(profile, 0.48)

    @staticmethod
    def _default_total_timeout(route_profile: str) -> float:
        if route_profile == "SHELF_SCAN":
            return 90.0
        if route_profile in {"SHELF_APPROACH", "DELIVERY"}:
            return 120.0
        return 30.0

    @staticmethod
    def _failure_code_after_exhaustion(failures: set[int]) -> int:
        if NavigateSemantic.Result.GOAL_BLOCKED in failures:
            return NavigateSemantic.Result.GOAL_BLOCKED
        if NavigateSemantic.Result.NO_PROGRESS in failures:
            return NavigateSemantic.Result.NO_PROGRESS
        return NavigateSemantic.Result.NO_PATH

    def _publish_speed_limit(self, route_profile: str) -> None:
        speed = {
            "SHELF_SCAN": 0.35,
            "COMPACT_TRANSIT": 0.35,
            "LOADED_TRANSIT": 0.25,
            "SHELF_APPROACH": 0.12,
            "DELIVERY_APPROACH": 0.12,
            "DELIVERY": 0.25,
        }.get(str(route_profile), 0.25)
        message = SpeedLimit()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "odom"
        message.percentage = False
        message.speed_limit = float(speed)
        self.speed_limit_pub.publish(message)

    def _publish_feedback(self, goal_handle, state, remaining: float) -> None:
        feedback = NavigateSemantic.Feedback()
        phase_mapping = {
            SessionPhase.NEW: NavigateSemantic.Feedback.NEW,
            SessionPhase.SELECT_ROUTE: NavigateSemantic.Feedback.SELECT_ROUTE,
            SessionPhase.PLAN: NavigateSemantic.Feedback.PLAN,
            SessionPhase.TRACK: NavigateSemantic.Feedback.TRACK,
            SessionPhase.APPROACH: NavigateSemantic.Feedback.APPROACH,
            SessionPhase.BLOCKED: NavigateSemantic.Feedback.BLOCKED,
            SessionPhase.BACKUP_ONCE: NavigateSemantic.Feedback.BACKUP_ONCE,
            SessionPhase.ALTERNATE_ROUTE: NavigateSemantic.Feedback.ALTERNATE_ROUTE,
        }
        feedback.state = phase_mapping.get(state.phase, NavigateSemantic.Feedback.BLOCKED)
        feedback.remaining_path_m = float(remaining)
        feedback.recovery_count = state.recovery_count
        feedback.selected_route = state.selected_route
        feedback.block_reason = state.block_reason
        goal_handle.publish_feedback(feedback)

    def _event(self, state, event: str, **extra: Any) -> None:
        payload = {
            "time": time.time(),
            "session_id": state.session_id,
            "phase": state.phase.name,
            "event": event,
            "selected_route": state.selected_route,
            "recovery_count": state.recovery_count,
            "block_reason": state.block_reason,
            **extra,
        }
        self.event_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))

    async def _finish(
        self,
        goal_handle,
        result,
        code: int,
        detail: str,
        *,
        abort: bool = False,
        state: NavigationSessionState | None = None,
        exact: bool = False,
    ):
        await self._set_motion_mode(MotionMode.STOP, self.active_session_id or "finish")
        if abort and goal_handle.is_active:
            goal_handle.abort()
        result.code = int(code)
        result.detail = detail
        result.exact_goal_reached = bool(exact)
        if self.pose is not None:
            result.final_pose = self._pose_stamped_from_current()
        if state is not None:
            result.recovery_count = state.recovery_count
            result.selected_route = state.selected_route
            self._event(state, "finished", code=int(code), detail=detail)
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavigationSessionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
