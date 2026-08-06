#!/usr/bin/env python3
"""
Navigation demo for the Supermarket Sorting task.

Drives the MMK2 robot autonomously between task target poses
using the SupermarketNavigator (costmap + A* + pure pursuit + laser
obstacle avoidance).  Arms are tucked during transit to minimise the
robot footprint and prevent wall collisions.

Sequence:
    Start → Delivery → Shelf D → Delivery → Shelf B → Delivery →
    Shelf C → Delivery → Stop

Set ``SUPERMARKET_NAV_SHELVES`` to a comma-separated list to override the
shelves visited (default: "D,B,C").
"""

import math
import os
import random
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan, JointState
from scipy.spatial.transform import Rotation

from supermarket_navigation import (
    SupermarketNavigator, DELIVERY_APPROACH, DELIVERY_TABLE_COSTMAP_BOUNDS,
    SHELF_APPROACH, depth_image_clearance, point_to_rect_clearance,
)

# ---------------------------------------------------------------------------
# Arm parking configuration used by the supplied baseline.
#
# Joint limits (Airbot Play arm):
#   j1:[-3.15, 2.08]  j2:[-2.96, 0.18]  j3:[-0.09, 3.16]
#   j4:[-3.01, 3.01]  j5:[-1.86, 1.86]  j6:[-3.02, 3.02]
#
# The previous "tucked" pose [-0.5,-2.0,2.5,...] put both end effectors about
# 0.744 m in front of base_link.  These verified baseline poses keep them about
# 0.249 m forward and 0.218 m to each side, close to the chassis envelope.
# ---------------------------------------------------------------------------
PARK_ARM_L = [0.0, -0.166, 0.032, 0.0, 1.571, 2.223, 0.0]
PARK_ARM_R = [0.0, -0.166, 0.032, 0.0, -1.571, -2.223, 0.0]
NAV_HEAD = [0.0, -0.50]

# ---------------------------------------------------------------------------
# Navigation-only task sequence
# ---------------------------------------------------------------------------
def _build_sequence():
    # Number of pick-and-deliver rounds (default 5, matching competition).
    num_rounds = int(os.environ.get("SUPERMARKET_NAV_ROUNDS", "5"))
    # Optional fixed shelf list via env var; random selection when empty.
    raw = os.environ.get("SUPERMARKET_NAV_SHELVES", "")
    fixed = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if fixed:
        invalid = [s for s in fixed if s not in SHELF_APPROACH]
        if invalid:
            raise ValueError(
                "SUPERMARKET_NAV_SHELVES only accepts A-E; invalid: "
                + ",".join(invalid))
        shelves = fixed[:num_rounds]
        while len(shelves) < num_rounds:
            shelves.append(random.choice(list(SHELF_APPROACH)))
    else:
        all_shelves = list(SHELF_APPROACH)
        shelves = [random.choice(all_shelves) for _ in range(num_rounds)]

    # Start → Delivery → (Random-Shelf → Delivery) × N
    sequence = [("delivery", DELIVERY_APPROACH)]
    for shelf in shelves:
        sequence.append((f"shelf_{shelf}", SHELF_APPROACH[shelf]))
        sequence.append(("delivery", DELIVERY_APPROACH))
    return sequence


DEMO_SEQUENCE = _build_sequence()


class NavigationDemoNode(Node):
    """Standalone navigation demo — drives MMK2 through a sequence of poses.

    Arms are folded in a compact travel configuration during transit so the
    overall robot footprint stays within the costmap inflation radius.
    """

    def __init__(self):
        super().__init__("navigation_demo")

        self.navigator = SupermarketNavigator()

        # Robot state
        self.base_x = self.base_y = self.base_yaw = None
        self.last_odom_time = None
        self.laser_msg = None
        self.last_scan_time = None
        self.depth_clearance = None
        self.last_depth_time = None
        self.depth_seq = 0
        self.jpos = {}
        self._arms_tucked = False
        self._last_sensor_warn = float('-inf')
        self._last_safety_warn = float('-inf')

        # Demo state machine
        self.seq_index = 0
        self.arrive_timer = 0.0          # seconds since arrival at current task goal
        self.arrive_dwell = 2.0          # pause at each task goal
        self.state = "TUCKING"           # TUCKING → NAVIGATING | DWELLING | DONE

        # Set first goal
        self._next_goal()

        # ---- subscribers ----
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom",
                                 self._odom_cb, 10)
        self.create_subscription(LaserScan, "/slamware_ros_sdk_server_node/scan",
                                 self._scan_cb, 10)
        self.create_subscription(
            Image, "/head_camera/aligned_depth_to_color/image_raw",
            self._depth_cb, 5)
        self.create_subscription(JointState, "/joint_states",
                                 self._js_cb, 10)

        # ---- publishers ----
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.larm_pub = self.create_publisher(
            Float64MultiArray, "/left_arm_forward_position_controller/commands", 5)
        self.rarm_pub = self.create_publisher(
            Float64MultiArray, "/right_arm_forward_position_controller/commands", 5)
        self.head_pub = self.create_publisher(
            Float64MultiArray, "/head_forward_position_controller/commands", 5)

        # Control loop at 50 Hz
        self.dt = 0.02
        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            "Navigation demo up — waiting for odom + laser ...")

    # ---- subscriptions ----
    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_x = p.x
        self.base_y = p.y
        self.base_yaw = Rotation.from_quat(
            [q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        self.last_odom_time = self._now()

    def _scan_cb(self, msg):
        self.laser_msg = msg
        self.last_scan_time = self._now()

    def _depth_cb(self, msg):
        self.depth_clearance = depth_image_clearance(msg)
        if self.depth_clearance is not None:
            # Depth is measured along the pitched optical axis; convert it to
            # approximate horizontal clearance used by the 2-D costmap.
            self.depth_clearance *= math.cos(abs(NAV_HEAD[1]))
        self.last_depth_time = self._now()
        self.depth_seq += 1

    def _js_cb(self, msg):
        self.jpos = {n: msg.position[i]
                     for i, n in enumerate(msg.name)
                     if i < len(msg.position)}

    # ---- arm helpers ----
    def _publish_arms(self, l_arm, r_arm):
        """Publish joint targets for both arms (6 joints + gripper each)."""
        self.larm_pub.publish(Float64MultiArray(
            data=[float(x) for x in l_arm]))
        self.rarm_pub.publish(Float64MultiArray(
            data=[float(x) for x in r_arm]))
        self.head_pub.publish(Float64MultiArray(
            data=[float(x) for x in NAV_HEAD]))

    def _arms_at_travel(self):
        """Check whether arms and navigation camera are in the safe posture."""
        if not self.jpos:
            return False
        for i in range(6):
            key = f"left_arm_joint{i+1}"
            target = PARK_ARM_L[i]
            if key not in self.jpos or abs(self.jpos[key] - target) > 0.08:
                return False
        for i in range(6):
            key = f"right_arm_joint{i+1}"
            target = PARK_ARM_R[i]
            if key not in self.jpos or abs(self.jpos[key] - target) > 0.08:
                return False
        for key, target in zip(
                ("head_yaw_joint", "head_pitch_joint"), NAV_HEAD):
            if key not in self.jpos or abs(self.jpos[key] - target) > 0.08:
                return False
        return True

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- state machine ----
    def _next_goal(self):
        """Advance to the next task goal; A* chooses the route online."""
        if self.seq_index >= len(DEMO_SEQUENCE):
            self.state = "DONE"
            self.get_logger().info("=== DEMO COMPLETE ===")
            return

        name, (x, y, yaw) = DEMO_SEQUENCE[self.seq_index]
        self.get_logger().info(
            f"--- [{self.seq_index+1}/{len(DEMO_SEQUENCE)}] "
            f"Navigating to {name}: ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f}°) ---")
        self.navigator.set_goal(x, y, yaw)
        self.state = "TUCKING"
        self._arms_tucked = False

    # ---- main loop ----
    def _tick(self):
        try:
            self._tick_impl()
        except Exception as e:
            self.get_logger().error(f"tick crashed: {type(e).__name__}: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _tick_impl(self):
        if self.base_x is None:
            return

        # Always keep arms in travel position (publish every tick)
        if not self._arms_tucked:
            self._publish_arms(PARK_ARM_L, PARK_ARM_R)

        if self.state == "DONE":
            self._publish_cmd(0.0, 0.0)
            return

        if self.state == "TUCKING":
            # Wait for arms to fold before starting to move
            self._publish_cmd(0.0, 0.0)
            if self._arms_at_travel():
                self._arms_tucked = True
                self.state = "NAVIGATING"
                self.get_logger().info("arms tucked — starting navigation")
            return

        if self.state == "DWELLING":
            self.arrive_timer += self.dt
            self._publish_cmd(0.0, 0.0)
            if self.arrive_timer >= self.arrive_dwell:
                self.seq_index += 1
                self._next_goal()
            return

        # NAVIGATING — keep arms tucked throughout
        self._publish_arms(PARK_ARM_L, PARK_ARM_R)

        # Autonomous obstacle avoidance must fail safe if lidar has not arrived
        # or the Server/Client connection stops updating it.
        now = self._now()
        scan_stale = (
            self.last_scan_time is None or now - self.last_scan_time > 0.50)
        odom_stale = (
            self.last_odom_time is None or now - self.last_odom_time > 0.50)
        if scan_stale or odom_stale:
            self._publish_cmd(0.0, 0.0)
            if now - self._last_sensor_warn > 1.0:
                self.get_logger().warn(
                    "waiting for fresh scan/odom data "
                    f"(scan_stale={scan_stale}, odom_stale={odom_stale})")
                self._last_sensor_warn = now
            return

        try:
            depth_fresh = (
                self.last_depth_time is not None and
                now - self.last_depth_time <= 0.50)
            v, w, reached = self.navigator.update(
                self.base_x, self.base_y, self.base_yaw,
                laser_msg=self.laser_msg,
                depth_clearance=(self.depth_clearance
                                 if depth_fresh else None),
                depth_token=(self.depth_seq if depth_fresh else None),
                time_now=now)
        except Exception as e:
            self.get_logger().error(f"navigator.update crashed: {e}")
            self._publish_cmd(0.0, 0.0)
            return

        self._publish_cmd(v, w)

        # ── pause depth writes during pure rotation (prevents obstacle arcs) ──
        self.navigator.costmap._pause_depth = (abs(v) < 1e-6 and abs(w) > 0.05)

        # ── periodic diagnostic log ──
        ctrl = self.navigator.controller
        reason = ctrl.stop_reason
        if reason != ctrl._last_logged_reason:
            ctrl._last_logged_reason = reason
            if reason is not None:
                self.get_logger().info(
                    f"stop_reason={reason} "
                    f"lidar={ctrl.lidar_clearance:.2f}m "
                    f"depth={ctrl.depth_clearance_val:.2f}m "
                    f"v={v:.2f} w={w:.2f}")

        if (self.navigator.controller.last_safety_stop ==
                "delivery_table_keepout" and
                now - self._last_safety_warn > 1.0):
            clearance = point_to_rect_clearance(
                self.base_x, self.base_y, DELIVERY_TABLE_COSTMAP_BOUNDS)
            self.get_logger().warn(
                "delivery-table keep-out active: "
                f"base=({self.base_x:.2f},{self.base_y:.2f}), "
                f"table_clearance={clearance:.2f} m")
            self._last_safety_warn = now

        if reached:
            name, _ = DEMO_SEQUENCE[self.seq_index]
            clearance_text = ""
            if name == "delivery":
                clearance = point_to_rect_clearance(
                    self.base_x, self.base_y,
                    DELIVERY_TABLE_COSTMAP_BOUNDS)
                clearance_text = f" table_clearance={clearance:.2f}m"
            self.get_logger().info(
                f"✓ arrived at {name} "
                f"pos=({self.base_x:.2f},{self.base_y:.2f}) "
                f"yaw={math.degrees(self.base_yaw):.0f}°"
                f"{clearance_text}")
            self.state = "DWELLING"
            self.arrive_timer = 0.0

    def _publish_cmd(self, v, w):
        tw = Twist()
        tw.linear.x = float(v)
        tw.angular.z = float(w)
        self.cmd_pub.publish(tw)


def main():
    rclpy.init()
    node = NavigationDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
