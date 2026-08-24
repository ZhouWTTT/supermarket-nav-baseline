"""Atomic, stopped-state footprint profile changes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point32, Polygon, PolygonStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from supermarket_interfaces.msg import MotionMode
from supermarket_interfaces.srv import SetFootprintProfile


@dataclass(frozen=True)
class FootprintProfile:
    points: tuple[tuple[float, float], ...]


def _profile_document_path() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = Path(get_package_share_directory("supermarket_navigation"))
        candidate = installed / "config" / "footprint_profiles.json"
        if candidate.exists():
            return candidate
    except (ImportError, LookupError):
        pass
    return Path(__file__).parents[1] / "config" / "footprint_profiles.json"


def _load_profiles() -> tuple[dict[str, FootprintProfile], dict]:
    document = json.loads(_profile_document_path().read_text(encoding="utf-8"))
    profiles: dict[str, FootprintProfile] = {}
    for name, raw_points in document["profiles"].items():
        points = tuple((float(point[0]), float(point[1])) for point in raw_points)
        if len(points) < 3 or not all(
            math.isfinite(value) for point in points for value in point
        ):
            raise ValueError(f"invalid generated footprint profile {name}")
        profiles[str(name)] = FootprintProfile(points)
    required = {
        "COMPACT_TRANSIT", "LOADED_TRANSIT", "SHELF_APPROACH",
        "DELIVERY_APPROACH", "MANIPULATION_EXTENDED",
    }
    if set(profiles) != required:
        raise ValueError("generated footprint profile set is incomplete")
    return profiles, document


PROFILES, PROFILE_DOCUMENT = _load_profiles()


class FootprintManager(Node):
    def __init__(self) -> None:
        super().__init__("supermarket_footprint_manager")
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.active_motion_mode = MotionMode.STOP
        self.active_profile = "COMPACT_TRANSIT"
        self.global_pub = self.create_publisher(
            Polygon, "/global_costmap/footprint", qos
        )
        self.local_pub = self.create_publisher(
            Polygon, "/local_costmap/footprint", qos
        )
        self.safety_pub = self.create_publisher(
            PolygonStamped, "/motion/safety_footprint", qos
        )
        self.create_subscription(MotionMode, "/motion/mode", self._mode_cb, 10)
        self.create_service(
            SetFootprintProfile,
            "/motion/set_footprint_profile",
            self._set_profile_cb,
        )
        self.create_timer(1.0, self._publish_active)
        self._publish_active()

    def _mode_cb(self, message: MotionMode) -> None:
        self.active_motion_mode = int(message.mode)

    def _set_profile_cb(self, request, response):
        if request.profile not in PROFILES:
            response.applied = False
            response.active_profile = self.active_profile
            response.detail = f"unknown footprint profile {request.profile!r}"
            return response
        if self.active_motion_mode != MotionMode.STOP:
            response.applied = False
            response.active_profile = self.active_profile
            response.detail = "footprint changes require motion mode STOP"
            return response
        self.active_profile = request.profile
        self._publish_active()
        response.applied = True
        response.active_profile = self.active_profile
        response.detail = "profile published to both costmaps and safety monitor"
        return response

    def _publish_active(self) -> None:
        polygon = Polygon()
        for x, y in PROFILES[self.active_profile].points:
            point = Point32()
            point.x = float(x)
            point.y = float(y)
            polygon.points.append(point)
        self.global_pub.publish(polygon)
        self.local_pub.publish(polygon)
        safety_polygon = PolygonStamped()
        safety_polygon.header.frame_id = "base_link"
        safety_polygon.header.stamp = self.get_clock().now().to_msg()
        safety_polygon.polygon = polygon
        self.safety_pub.publish(safety_polygon)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FootprintManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
