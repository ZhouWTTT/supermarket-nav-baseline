"""Fail-closed arbitration for all base velocity producers."""

from __future__ import annotations

import json
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener

from supermarket_interfaces.msg import MotionMode
from supermarket_interfaces.srv import SetMotionMode


class MotionArbiter(Node):
    def __init__(self) -> None:
        super().__init__("supermarket_motion_arbiter")
        self.declare_parameter("command_timeout_s", 0.20)
        # The official server nominally advertises 12 Hz laser data, but the
        # measured rate under renderer load can fall to about 3.1 Hz.  The
        # approved stale boundary is 0.5 s; command freshness remains 0.2 s.
        self.declare_parameter("sensor_timeout_s", 0.50)
        self.declare_parameter("publish_hz", 50.0)
        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)
        self.sensor_timeout_s = float(self.get_parameter("sensor_timeout_s").value)
        publish_hz = float(self.get_parameter("publish_hz").value)
        self.active_mode = MotionMode.STOP
        self.active_owner = "startup"
        self.emergency_stop = False
        self.emergency_reason = ""
        self.run_prefix = ""
        self.new_task_reset_pending = False
        self._last_commands: dict[int, tuple[float, Twist]] = {}
        self._last_odom_at = float("-inf")
        self._last_scan_at = float("-inf")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.output_pub = self.create_publisher(
            Twist, "/motion/selected_cmd_vel", 10
        )
        self.status_pub = self.create_publisher(MotionMode, "/motion/mode", 10)
        self.create_subscription(Twist, "/motion/nav_cmd_vel", self._nav_cb, 10)
        self.create_subscription(
            Twist, "/motion/manip_cmd_vel", self._manip_cb, 10
        )
        self.create_subscription(
            Bool, "/motion/emergency_stop", self._emergency_cb, 10
        )
        task_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, "/supermarket_sorting/task", self._task_cb, task_qos
        )
        self.create_subscription(Odometry, "/nav/odom", self._odom_cb, 20)
        self.create_subscription(
            LaserScan, "/nav/scan", self._scan_cb, qos_profile_sensor_data
        )
        self.create_service(
            SetMotionMode, "/motion/set_mode", self._set_mode_cb
        )
        self.create_timer(1.0 / max(1.0, publish_hz), self._publish_selected)
        self.get_logger().info(
            f"motion arbiter ready; fail-closed timeout={self.command_timeout_s:.2f}s"
        )

    def _nav_cb(self, message: Twist) -> None:
        self._last_commands[MotionMode.NAVIGATION] = (time.monotonic(), message)

    def _manip_cb(self, message: Twist) -> None:
        self._last_commands[MotionMode.MANIPULATION] = (time.monotonic(), message)

    def _emergency_cb(self, message: Bool) -> None:
        if message.data:
            self._latch_emergency("external emergency stop")

    def _task_cb(self, message: String) -> None:
        try:
            run_prefix = str(json.loads(message.data)["run_prefix"])
        except (ValueError, KeyError, TypeError):
            return
        if not run_prefix or run_prefix == self.run_prefix:
            return
        self.run_prefix = run_prefix
        self.new_task_reset_pending = True
        self.active_mode = MotionMode.STOP
        self.active_owner = f"new_task:{run_prefix}"
        self.output_pub.publish(Twist())
        self.get_logger().info(
            f"new task initialized in STOP; safety reset pending fresh feedback "
            f"run_prefix={run_prefix}"
        )

    def _odom_cb(self, _message: Odometry) -> None:
        self._last_odom_at = time.monotonic()

    def _scan_cb(self, _message: LaserScan) -> None:
        self._last_scan_at = time.monotonic()

    def _critical_feedback_fresh(self) -> bool:
        now = time.monotonic()
        sensors_fresh = (
            now - self._last_odom_at <= self.sensor_timeout_s
            and now - self._last_scan_at <= self.sensor_timeout_s
        )
        if not sensors_fresh:
            return False
        return self.tf_buffer.can_transform("odom", "base_link", Time())

    def _latch_emergency(self, reason: str) -> None:
        newly_latched = not self.emergency_stop
        self.emergency_stop = True
        self.emergency_reason = str(reason)
        self.active_mode = MotionMode.STOP
        self.active_owner = "emergency_stop"
        self.output_pub.publish(Twist())
        if newly_latched:
            self.get_logger().error(f"motion emergency stop latched: {reason}")

    def _complete_new_task_reset_if_ready(self) -> bool:
        if not self.new_task_reset_pending:
            return True
        if not self._critical_feedback_fresh():
            return False
        self.emergency_stop = False
        self.emergency_reason = ""
        self.new_task_reset_pending = False
        self.active_mode = MotionMode.STOP
        self.active_owner = f"new_task_ready:{self.run_prefix}"
        self.get_logger().info(
            f"new task safety reset completed with fresh feedback "
            f"run_prefix={self.run_prefix}"
        )
        return True

    def _set_mode_cb(self, request, response):
        valid_modes = {
            MotionMode.STOP,
            MotionMode.NAVIGATION,
            MotionMode.MANIPULATION,
        }
        if request.mode not in valid_modes:
            response.accepted = False
            response.detail = f"invalid motion mode {request.mode}"
        elif request.mode == MotionMode.STOP:
            self.active_mode = MotionMode.STOP
            self.active_owner = request.owner or "anonymous"
            self.output_pub.publish(Twist())
            if request.reset_emergency_stop:
                if self._critical_feedback_fresh():
                    self.emergency_stop = False
                    self.emergency_reason = ""
                    response.accepted = True
                    response.detail = "stopped and emergency latch reset"
                else:
                    response.accepted = False
                    response.detail = (
                        "stopped but cannot reset emergency latch without "
                        "fresh odom, scan and TF")
            else:
                response.accepted = True
                response.detail = "motion stopped"
        elif (
            request.mode != MotionMode.STOP
            and self.new_task_reset_pending
            and not self._complete_new_task_reset_if_ready()
        ):
            response.accepted = False
            response.detail = (
                "new task remains stopped until odom, scan and TF are fresh"
            )
        elif request.reset_emergency_stop and not self._critical_feedback_fresh():
            response.accepted = False
            response.detail = "cannot reset emergency stop without fresh odom, scan and TF"
        elif self.emergency_stop and not request.reset_emergency_stop:
            response.accepted = False
            response.detail = f"emergency stop is latched: {self.emergency_reason}"
        elif request.mode != MotionMode.STOP and not self._critical_feedback_fresh():
            self._latch_emergency("critical odom, scan or TF feedback is stale")
            response.accepted = False
            response.detail = self.emergency_reason
        else:
            if request.reset_emergency_stop:
                self.emergency_stop = False
                self.emergency_reason = ""
            self.active_mode = int(request.mode)
            self.active_owner = request.owner or "anonymous"
            self._last_commands.pop(self.active_mode, None)
            self.output_pub.publish(Twist())
            response.accepted = True
            response.detail = "motion source changed; awaiting a fresh command"
        response.active_mode = self.active_mode
        response.active_owner = self.active_owner
        self._publish_status()
        return response

    def _publish_status(self) -> None:
        status = MotionMode()
        status.mode = self.active_mode
        status.owner = self.active_owner
        status.stamp = self.get_clock().now().to_msg()
        self.status_pub.publish(status)

    def _publish_selected(self) -> None:
        output = Twist()
        if self.new_task_reset_pending:
            self._complete_new_task_reset_if_ready()
        if (
            not self.emergency_stop
            and self.active_mode != MotionMode.STOP
            and not self._critical_feedback_fresh()
        ):
            self._latch_emergency("critical odom, scan or TF feedback became stale")
        if not self.emergency_stop and self.active_mode != MotionMode.STOP:
            command = self._last_commands.get(self.active_mode)
            if command is not None:
                received_at, message = command
                if time.monotonic() - received_at <= self.command_timeout_s:
                    output = message
        self.output_pub.publish(output)
        self._publish_status()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MotionArbiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.output_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
