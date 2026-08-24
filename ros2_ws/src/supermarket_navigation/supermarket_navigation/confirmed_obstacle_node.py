"""Build a per-run confirmed obstacle cloud from public laser and odometry."""

from __future__ import annotations

import json
import math
import struct
import time

import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import String

from .confirmed_obstacles import ConfirmedObstacleGrid, StaticOccupancyMask


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class ConfirmedObstacleTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("supermarket_confirmed_obstacle_tracker")
        self.declare_parameter("resolution_m", 0.10)
        self.declare_parameter("hits_required", 3)
        self.declare_parameter("hit_window_s", 1.0)
        self.declare_parameter("free_rays_to_clear", 5)
        self.declare_parameter("laser_offset_x_m", 0.09)
        self.declare_parameter("min_obstacle_range_m", 0.08)
        self.declare_parameter("max_obstacle_range_m", 5.0)
        self.declare_parameter("static_filter_radius_m", 0.08)
        self.declare_parameter("minimum_cluster_cells", 2)
        self.grid = ConfirmedObstacleGrid(
            resolution_m=self.get_parameter("resolution_m").value,
            hits_required=self.get_parameter("hits_required").value,
            hit_window_s=self.get_parameter("hit_window_s").value,
            free_rays_to_clear=self.get_parameter("free_rays_to_clear").value,
        )
        self.laser_offset_x = float(self.get_parameter("laser_offset_x_m").value)
        self.min_range = float(self.get_parameter("min_obstacle_range_m").value)
        self.max_range = float(self.get_parameter("max_obstacle_range_m").value)
        self.static_filter_radius = float(
            self.get_parameter("static_filter_radius_m").value
        )
        self.minimum_cluster_cells = int(
            self.get_parameter("minimum_cluster_cells").value
        )
        self.pose: tuple[float, float, float] | None = None
        self.last_odom_at = float("-inf")
        self.static_mask: StaticOccupancyMask | None = None
        self.scan_sequence = 0

        task_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/nav/confirmed_obstacles", 10
        )
        self.create_subscription(Odometry, "/nav/odom", self._odom_cb, 20)
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, map_qos)
        self.create_subscription(
            LaserScan, "/nav/scan", self._scan_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            String, "/supermarket_sorting/task", self._task_cb, task_qos
        )
        self.create_timer(0.5, self._publish_cloud)

    def _task_cb(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            run_prefix = str(payload["run_prefix"])
        except (ValueError, KeyError, TypeError) as exc:
            self.get_logger().error(f"invalid task message for obstacle reset: {exc}")
            return
        if self.grid.reset_for_run(run_prefix):
            self.get_logger().info(
                f"confirmed obstacle evidence reset for run_prefix={run_prefix}"
            )
            self._publish_cloud()

    def _odom_cb(self, message: Odometry) -> None:
        position = message.pose.pose.position
        self.pose = (
            float(position.x),
            float(position.y),
            yaw_from_quaternion(message.pose.pose.orientation),
        )
        self.last_odom_at = time.monotonic()

    def _map_cb(self, message: OccupancyGrid) -> None:
        info = message.info
        self.static_mask = StaticOccupancyMask(
            width=int(info.width),
            height=int(info.height),
            resolution_m=float(info.resolution),
            origin_x=float(info.origin.position.x),
            origin_y=float(info.origin.position.y),
            origin_yaw=yaw_from_quaternion(info.origin.orientation),
            data=tuple(message.data),
        )

    def _scan_cb(self, message: LaserScan) -> None:
        if self.pose is None or time.monotonic() - self.last_odom_at > 0.5:
            return
        bx, by, yaw = self.pose
        lx = bx + self.laser_offset_x * math.cos(yaw)
        ly = by + self.laser_offset_x * math.sin(yaw)
        now_s = time.monotonic()
        self.scan_sequence += 1
        observation_id = self.scan_sequence
        ranges = message.ranges
        valid_hits: list[tuple[float, float]] = []
        for index, measured in enumerate(ranges):
            distance = float(measured)
            if not math.isfinite(distance):
                continue
            if distance < max(self.min_range, float(message.range_min)):
                continue
            if distance > min(self.max_range, float(message.range_max)):
                continue
            angle = yaw + float(message.angle_min) + index * float(message.angle_increment)
            wx = lx + distance * math.cos(angle)
            wy = ly + distance * math.sin(angle)
            if (
                self.static_mask is not None
                and self.static_mask.is_occupied(
                    wx, wy, radius_m=self.static_filter_radius
                )
            ):
                continue
            valid_hits.append((wx, wy))
            self.grid.observe_hit(wx, wy, now_s, observation_id=observation_id)

        # Confirmed obstacles are static in one run.  Clear one only after the
        # corresponding beam repeatedly observes free space beyond it.
        for wx, wy in list(self.grid.confirmed_points()):
            dx, dy = wx - lx, wy - ly
            expected_range = math.hypot(dx, dy)
            relative_angle = math.atan2(dy, dx) - yaw
            while relative_angle < float(message.angle_min):
                relative_angle += 2.0 * math.pi
            while relative_angle > float(message.angle_max):
                relative_angle -= 2.0 * math.pi
            index = round(
                (relative_angle - float(message.angle_min))
                / float(message.angle_increment)
            )
            if not 0 <= index < len(ranges):
                continue
            measured = float(ranges[index])
            if (not math.isfinite(measured) or measured > expected_range + 0.15):
                # Do not clear a cell that was itself hit in this scan.
                if all(math.hypot(wx - hx, wy - hy) > self.grid.resolution_m
                       for hx, hy in valid_hits):
                    self.grid.observe_free(wx, wy)
        self._publish_cloud()

    def _publish_cloud(self) -> None:
        points = self.grid.clustered_confirmed_points(self.minimum_cluster_cells)
        message = PointCloud2()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "odom"
        message.height = 1
        message.width = len(points)
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = 12
        message.row_step = 12 * len(points)
        message.is_dense = True
        message.data = b"".join(struct.pack("<fff", x, y, 0.10) for x, y in points)
        self.cloud_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConfirmedObstacleTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
