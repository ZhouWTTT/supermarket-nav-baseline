#!/usr/bin/env python3
"""Detect the 45 shelf ArUco markers from all MMK2 RGB cameras."""

import argparse
import json
import math

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Int32MultiArray, String
from scipy.spatial.transform import Rotation

from discoverse.robots.mmk2.mmk2_fk import MMK2FK


MARKER_SIZE_M = 0.03
VALID_IDS = set(range(45))
# RGB and aligned depth are rendered sequentially in this simulator.  Their
# observed stamp separation can reach about 120 ms even while the camera is
# stationary, so 75 ms unnecessarily forced head ArUco back to scale-sensitive
# PnP.  Refinement begins only after a settle dwell; 150 ms accepts the paired
# RGB-D frames without mixing moving-camera poses.
DEPTH_SYNC_MAX_NS = 150_000_000
CAMERAS = {
    "head": ("/head_camera/color/image_raw", "/head_camera/color/camera_info", "head_camera"),
    "left": ("/left_camera/color/image_raw", "/left_camera/color/camera_info", "left_camera"),
    "right": ("/right_camera/color/image_raw", "/right_camera/color/camera_info", "right_camera"),
}


def rotation_matrix_to_quaternion(matrix):
    """Return an xyzw quaternion without requiring an extra geometry package."""
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array([
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        ])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                         (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    if i == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                         (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                     0.25 * s, (m[1, 0] - m[0, 1]) / s])


def solve_marker_pose(corners, camera_matrix, distortion, marker_size):
    half = marker_size * 0.5
    object_points = np.array([
        [-half, half, 0.0], [half, half, 0.0],
        [half, -half, 0.0], [-half, -half, 0.0],
    ], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, np.asarray(corners, dtype=np.float32).reshape(4, 2),
        camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return rvec.reshape(3), tvec.reshape(3)


class ArucoDetectNode(Node):
    def __init__(self, camera_name, marker_size=MARKER_SIZE_M, publish_tf=True,
                 publish_result_image=True):
        super().__init__(f"aruco_detect_{camera_name}")
        image_topic, info_topic, default_frame = CAMERAS[camera_name]
        self.camera_name = camera_name
        self.default_frame = default_frame
        self.marker_size = marker_size
        self.publish_tf = publish_tf
        self.publish_result_image = publish_result_image
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion = np.zeros(5, dtype=float)
        self.depth_image = None
        self.depth_stamp_ns = None
        self.fk = MMK2FK()
        self.base_pos = None
        self.base_quat = None
        self.slide = 0.0
        self.head = [0.0, 0.0]
        self.left_arm = [0.0] * 6
        self.right_arm = [0.0] * 6
        self.decode_orientation = None

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "DetectorParameters"):
            parameters = cv2.aruco.DetectorParameters()
        else:
            parameters = cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            self._detect = lambda gray: detector.detectMarkers(gray)[:2]
        else:
            self._detect = lambda gray: cv2.aruco.detectMarkers(
                gray, dictionary, parameters=parameters)[:2]

        prefix = f"/aruco/{camera_name}"
        self.ids_pub = self.create_publisher(Int32MultiArray, f"{prefix}/ids", 10)
        self.poses_pub = self.create_publisher(PoseArray, f"{prefix}/poses", 10)
        self.detections_pub = self.create_publisher(String, f"{prefix}/detections", 10)
        self.result_pub = self.create_publisher(Image, f"{prefix}/result_image", 5)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) if publish_tf else None
        self.create_subscription(CameraInfo, info_topic, self.camera_info_cb, 10)
        self.create_subscription(Image, image_topic, self.image_cb, 10)
        if camera_name == "head":
            self.create_subscription(
                Image, "/head_camera/aligned_depth_to_color/image_raw",
                self.depth_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 10)
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom",
                                 self.odom_cb, 10)
        self.get_logger().info(
            f"listening on {image_topic}; dictionary=DICT_4X4_50, marker_size={marker_size:.3f}m")

    def camera_info_cb(self, msg):
        self.camera_matrix = np.asarray(msg.k, dtype=float).reshape(3, 3)
        if msg.d:
            self.distortion = np.asarray(msg.d, dtype=float)

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough")
        self.depth_stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec))

    def marker_center_xyz(self, corners, image_stamp_ns=None):
        """Deproject the ArUco centre using aligned rendered depth.

        The marker supplies the exact pixels and identity; depth supplies range.
        This is substantially less biased than estimating distance from a tiny
        30 mm square with solvePnP.  PnP remains the fallback and orientation
        source when aligned depth is unavailable.
        """
        if self.depth_image is None or self.camera_matrix is None:
            return None
        if (image_stamp_ns is None or self.depth_stamp_ns is None
                or abs(image_stamp_ns - self.depth_stamp_ns)
                    > DEPTH_SYNC_MAX_NS):
            return None
        uv = np.asarray(corners, dtype=float).reshape(4, 2).mean(axis=0)
        u, v = int(round(uv[0])), int(round(uv[1]))
        h, w = self.depth_image.shape[:2]
        radius = 3
        x0, x1 = max(0, u - radius), min(w, u + radius + 1)
        y0, y1 = max(0, v - radius), min(h, v + radius + 1)
        patch = np.asarray(self.depth_image[y0:y1, x0:x1], dtype=float)
        values = patch[np.isfinite(patch) & (patch > 0)]
        if values.size < 4:
            return None
        depth = float(np.median(values))
        if self.depth_image.dtype == np.uint16 or depth > 20.0:
            depth *= 0.001
        if not 0.10 < depth < 3.0:
            return None
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        return np.array([(uv[0] - cx) * depth / fx,
                         (uv[1] - cy) * depth / fy, depth])

    def joint_cb(self, msg):
        joints = {name: msg.position[i] for i, name in enumerate(msg.name)
                  if i < len(msg.position)}
        self.slide = joints.get("slide_joint", self.slide)
        self.head = [joints.get("head_yaw_joint", self.head[0]),
                     joints.get("head_pitch_joint", self.head[1])]
        self.left_arm = [
            joints.get(f"left_arm_joint{i + 1}", self.left_arm[i])
            for i in range(6)]
        self.right_arm = [
            joints.get(f"right_arm_joint{i + 1}", self.right_arm[i])
            for i in range(6)]

    def odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.base_pos = [p.x, p.y, p.z]
        self.base_quat = [q.w, q.x, q.y, q.z]

    def camera_world_tmat(self):
        """Measured optical-frame pose; no shelf/layout coordinates."""
        if self.base_pos is None:
            return None
        self.fk.set_base_pose(self.base_pos, self.base_quat)
        self.fk.set_slide_joint(float(self.slide))
        self.fk.set_head_joints([float(self.head[0]), float(self.head[1])])
        self.fk.set_left_arm_joints(self.left_arm)
        self.fk.set_right_arm_joints(self.right_arm)
        pose_getter = {
            "head": self.fk.get_head_camera_pose,
            "left": self.fk.get_left_camera_pose,
            "right": self.fk.get_right_camera_pose,
        }[self.camera_name]
        position, quat_wxyz = pose_getter()
        transform = np.eye(4)
        transform[:3, 3] = position
        transform[:3, :3] = Rotation.from_quat(
            quat_wxyz[[1, 2, 3, 0]]).as_matrix()
        if self.camera_name != "head":
            # The wrist camera itself is pitched -30 degrees relative to the
            # left_cam/right_cam optical-convention site in the MJCF.
            transform[:3, :3] = transform[:3, :3] @ Rotation.from_euler(
                "x", -0.5236).as_matrix()
        return transform

    def collect_valid(self, corners, ids, image_stamp_ns=None):
        valid = []
        if ids is None:
            return valid
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in VALID_IDS:
                continue
            pose = solve_marker_pose(marker_corners, self.camera_matrix,
                                     self.distortion, self.marker_size)
            if pose is None:
                continue
            rvec, pnp_tvec = pose
            depth_tvec = self.marker_center_xyz(
                marker_corners, image_stamp_ns=image_stamp_ns)
            source = "aruco_rgbd" if depth_tvec is not None else "aruco_pnp"
            valid.append((marker_id, marker_corners, rvec,
                          depth_tvec if depth_tvec is not None else pnp_tvec,
                          source))
        return valid

    def image_cb(self, msg):
        if self.camera_matrix is None:
            return
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect(gray)
        image_stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec))
        valid = self.collect_valid(
            corners, ids, image_stamp_ns=image_stamp_ns)
        orientation = "normal"
        if valid and self.decode_orientation is None:
            self.decode_orientation = orientation
            self.get_logger().info(
                "ArUco decode orientation=normal; no layout lookup used")
        valid.sort(key=lambda item: item[0])
        self.publish_detections(msg, valid, orientation)
        if self.publish_result_image:
            self.publish_visualization(msg, image, valid)

    def publish_detections(self, image_msg, detections, orientation="normal"):
        frame_id = image_msg.header.frame_id or self.default_frame
        ids_msg = Int32MultiArray(data=[item[0] for item in detections])
        poses_msg = PoseArray()
        poses_msg.header = image_msg.header
        poses_msg.header.frame_id = frame_id
        records = []
        transforms = []
        camera_world = self.camera_world_tmat()
        stamp_ns = (int(image_msg.header.stamp.sec) * 1_000_000_000
                    + int(image_msg.header.stamp.nanosec))
        for marker_id, marker_corners, rvec, tvec, source in detections:
            rotation, _ = cv2.Rodrigues(rvec)
            quat = rotation_matrix_to_quaternion(rotation)
            corner_pixels = np.asarray(marker_corners, dtype=float).reshape(4, 2)
            pixel_center = corner_pixels.mean(axis=0)
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, tvec)
            pose.orientation.x, pose.orientation.y = float(quat[0]), float(quat[1])
            pose.orientation.z, pose.orientation.w = float(quat[2]), float(quat[3])
            poses_msg.poses.append(pose)
            record = {"id": marker_id, "position": tvec.tolist(),
                      "position_camera": tvec.tolist(),
                      "camera": self.camera_name,
                      "camera_matrix": self.camera_matrix.reshape(-1).tolist(),
                      "pixel_center": pixel_center.tolist(),
                      "corners": corner_pixels.tolist(),
                      "quaternion_xyzw": quat.tolist(),
                      "position_source": source,
                      "stamp_ns": stamp_ns,
                      "decode_orientation": orientation}
            if camera_world is not None:
                world = (camera_world @ np.r_[tvec, 1.0])[:3]
                record["position_world"] = world.tolist()
                record["camera_world_rotation"] = (
                    camera_world[:3, :3].reshape(-1).tolist())
            records.append(record)
            if self.tf_broadcaster is not None:
                transform = TransformStamped()
                transform.header = poses_msg.header
                transform.child_frame_id = f"aruco_{self.camera_name}_{marker_id:02d}"
                transform.transform.translation.x = pose.position.x
                transform.transform.translation.y = pose.position.y
                transform.transform.translation.z = pose.position.z
                transform.transform.rotation = pose.orientation
                transforms.append(transform)
        self.ids_pub.publish(ids_msg)
        self.poses_pub.publish(poses_msg)
        self.detections_pub.publish(String(data=json.dumps(records, separators=(",", ":"))))
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

    def publish_visualization(self, image_msg, image, detections):
        for marker_id, corners, rvec, tvec, _ in detections:
            cv2.aruco.drawDetectedMarkers(image, [corners], np.array([[marker_id]]))
            cv2.drawFrameAxes(image, self.camera_matrix, self.distortion,
                              rvec, tvec, self.marker_size * 0.5)
        result = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        result.header = image_msg.header
        self.result_pub.publish(result)


def main():
    parser = argparse.ArgumentParser(description="Detect supermarket shelf ArUco markers")
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS),
                        help="camera nodes to start (default: head left right)")
    parser.add_argument("--marker-size", type=float, default=MARKER_SIZE_M,
                        help="marker side length in metres (default: 0.03)")
    parser.add_argument("--no-tf", action="store_true", help="disable marker TF publishing")
    parser.add_argument("--no-result-image", action="store_true",
                        help="disable annotated image publishing")
    args = parser.parse_args()

    rclpy.init()
    nodes = [ArucoDetectNode(camera, args.marker_size, not args.no_tf,
                             not args.no_result_image) for camera in args.cameras]
    executor = MultiThreadedExecutor(num_threads=max(2, len(nodes)))
    for node in nodes:
        executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        for node in nodes:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
