"""Normalize the public competition odometry and laser topics for Nav2."""

from __future__ import annotations

import copy

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class SensorAdapter(Node):
    def __init__(self) -> None:
        super().__init__("supermarket_sensor_adapter")
        self.declare_parameter(
            "source_odom_topic", "/slamware_ros_sdk_server_node/odom"
        )
        self.declare_parameter(
            "source_scan_topic", "/slamware_ros_sdk_server_node/scan"
        )
        self.declare_parameter("output_odom_topic", "/nav/odom")
        self.declare_parameter("output_scan_topic", "/nav/scan")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("laser_frame", "laser")

        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.laser_frame = self.get_parameter("laser_frame").value
        self.odom_pub = self.create_publisher(
            Odometry, self.get_parameter("output_odom_topic").value, 20
        )
        # Publish the normalized stream reliably.  Nav2's obstacle layer in
        # Humble requests reliable LaserScan data, while the official sensor
        # source itself may remain best-effort.
        normalized_scan_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE
        )
        self.scan_pub = self.create_publisher(
            LaserScan,
            self.get_parameter("output_scan_topic").value,
            normalized_scan_qos,
        )
        self.create_subscription(
            Odometry,
            self.get_parameter("source_odom_topic").value,
            self._odom_cb,
            20,
        )
        self.create_subscription(
            LaserScan,
            self.get_parameter("source_scan_topic").value,
            self._scan_cb,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "normalizing public odom/scan topics; no server-private data is used"
        )

    def _odom_cb(self, message: Odometry) -> None:
        output = copy.deepcopy(message)
        output.header.frame_id = self.odom_frame
        output.child_frame_id = self.base_frame
        self.odom_pub.publish(output)

    def _scan_cb(self, message: LaserScan) -> None:
        output = copy.deepcopy(message)
        output.header.frame_id = self.laser_frame
        self.scan_pub.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SensorAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
