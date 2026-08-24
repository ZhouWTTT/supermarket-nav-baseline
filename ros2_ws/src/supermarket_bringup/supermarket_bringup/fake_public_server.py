"""Deterministic public-topic fixture for candidate navigation fault tests.

This is deliberately not a competition backend: it exposes only the same
odom, LaserScan, TF, task and velocity interfaces available to a Client.  It
contains no product coordinates, anonymous-ID mapping or server workspace
data and is never included by the formal/candidate mission launches.
"""

from __future__ import annotations

import json
import math
import time

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def quaternion_from_yaw(yaw: float):
    from geometry_msgs.msg import Quaternion

    value = Quaternion()
    value.z = math.sin(0.5 * yaw)
    value.w = math.cos(0.5 * yaw)
    return value


class FakePublicServer(Node):
    def __init__(self) -> None:
        super().__init__("supermarket_fake_public_server")
        self.declare_parameter("initial_x", 1.92)
        self.declare_parameter("initial_y", -3.17)
        self.declare_parameter("initial_yaw", math.pi / 2.0)
        self.declare_parameter("run_prefix", "nav_fixture_run")
        self.x = float(self.get_parameter("initial_x").value)
        self.y = float(self.get_parameter("initial_y").value)
        self.yaw = float(self.get_parameter("initial_yaw").value)
        self.command = Twist()
        self.command_at = float("-inf")
        self.last_tick = time.monotonic()
        self.scan_divider = 0
        self.drop_odom = False
        self.drop_scan = False
        self.freeze_motion = False
        self.front_blocked = False
        self.rear_blocked = False

        self.odom_pub = self.create_publisher(
            Odometry, "/slamware_ros_sdk_server_node/odom", 20
        )
        scan_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.scan_pub = self.create_publisher(
            LaserScan, "/slamware_ros_sdk_server_node/scan", scan_qos
        )
        task_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.task_pub = self.create_publisher(
            String, "/supermarket_sorting/task", task_qos
        )
        self.tf_pub = TransformBroadcaster(self)
        self.static_tf_pub = StaticTransformBroadcaster(self)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 20)
        for name, attribute in (
            ("drop_odom", "drop_odom"),
            ("drop_scan", "drop_scan"),
            ("freeze_motion", "freeze_motion"),
            ("front_blocked", "front_blocked"),
            ("rear_blocked", "rear_blocked"),
        ):
            self.create_service(
                SetBool,
                f"/test/{name}",
                lambda request, response, key=attribute: self._set_flag(
                    key, request, response
                ),
            )
        self.create_timer(0.02, self._tick)
        self._publish_static_tf()
        self._publish_task()

    def _cmd_cb(self, message: Twist) -> None:
        self.command = message
        self.command_at = time.monotonic()

    def _set_flag(self, attribute: str, request, response):
        setattr(self, attribute, bool(request.data))
        response.success = True
        response.message = f"{attribute}={int(bool(request.data))}"
        return response

    def _publish_task(self) -> None:
        run_prefix = str(self.get_parameter("run_prefix").value)
        task = {
            "schema_version": 1,
            "run_prefix": run_prefix,
            "count": 1,
            "targets": [{"id": f"item_{run_prefix}_fixture", "kind": "kele"}],
        }
        self.task_pub.publish(
            String(data=json.dumps(task, separators=(",", ":")))
        )

    def _publish_static_tf(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = "laser"
        transform.transform.translation.x = 0.09
        transform.transform.rotation.w = 1.0
        self.static_tf_pub.sendTransform(transform)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(0.05, max(0.0, now - self.last_tick))
        self.last_tick = now
        command_fresh = now - self.command_at <= 0.25
        if command_fresh and not self.freeze_motion:
            linear = float(self.command.linear.x)
            angular = float(self.command.angular.z)
            self.x += linear * math.cos(self.yaw) * dt
            self.y += linear * math.sin(self.yaw) * dt
            self.yaw = (self.yaw + angular * dt + math.pi) % (2.0 * math.pi) - math.pi
        else:
            linear = 0.0
            angular = 0.0
        stamp = self.get_clock().now().to_msg()
        if not self.drop_odom:
            self._publish_odom(stamp, linear, angular)
            self._publish_dynamic_tf(stamp)
        self.scan_divider = (self.scan_divider + 1) % 4
        if self.scan_divider == 0 and not self.drop_scan:
            self._publish_scan(stamp)

    def _publish_odom(self, stamp, linear: float, angular: float) -> None:
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = "odom"
        message.child_frame_id = "base_link"
        message.pose.pose.position.x = self.x
        message.pose.pose.position.y = self.y
        message.pose.pose.orientation = quaternion_from_yaw(self.yaw)
        message.twist.twist.linear.x = linear
        message.twist.twist.angular.z = angular
        self.odom_pub.publish(message)

    def _publish_dynamic_tf(self, stamp) -> None:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = self.x
        transform.transform.translation.y = self.y
        transform.transform.rotation = quaternion_from_yaw(self.yaw)
        self.tf_pub.sendTransform(transform)

    def _publish_scan(self, stamp) -> None:
        count = 360
        message = LaserScan()
        message.header.stamp = stamp
        message.header.frame_id = "laser"
        message.angle_min = -math.pi
        message.angle_max = math.pi - 2.0 * math.pi / count
        message.angle_increment = 2.0 * math.pi / count
        message.time_increment = 0.0
        message.scan_time = 0.08
        message.range_min = 0.05
        message.range_max = 8.0
        ranges = [math.inf] * count
        if self.front_blocked:
            for index in list(range(0, 9)) + list(range(count - 8, count)):
                ranges[index] = 0.22
        if self.rear_blocked:
            center = count // 2
            for index in range(center - 8, center + 9):
                ranges[index] = 0.12
        message.ranges = ranges
        message.intensities = []
        self.scan_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakePublicServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
