#!/usr/bin/env python3
"""Multi-class goods perception node for the Supermarket Sorting task.

Runs the selected detector for any class stored in the supplied checkpoint and
publishes both world-frame RGB-D points and 2-D boxes.  The 2-D boxes allow the
pick client to associate a product with the ArUco marker directly below it.

Pipeline
--------
  /{head,left,right}_camera/color/image_raw (RGB, bgr8 / rgb8)
  /head_camera/aligned_depth_to_color/...   (head depth, mono16 in mm)
  /{head,left,right}_camera/color/camera_info (K)
  /joint_states + /odom                   (drive MMK2FK -> camera-in-world)
        |
        v  2-D detector backend (Blob / GT / YOLO)  -> bbox centre (u,v)
        v  pixel2cam: deproject (u,v,depth) with K  -> camera-frame point
        v  T_cam_world @ p_cam (MMK2FK headeye site) -> WORLD point
        |
        v  publish /kele/detections (legacy Detection3DArray, world frame)
           publish /goods/yolo_detections (JSON with bbox + foreground depth)
           publish /kele/result_image (debug overlay)

The head camera publishes RGB-D world points.  The wrist cameras are RGB-only
and publish boxes for same-view ArUco association by the pick client.  All
three views share one detector instance, so multi-view search does not load
three copies of the YOLO checkpoint.
"""

import argparse
import json
import os
import threading
from collections import deque
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose

from discoverse.robots.mmk2.mmk2_fk import MMK2FK

from backends import GtProjectionBackend, BlobBackend, YoloBackend

LAYOUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "retail_competition_layout.json")
DEFAULT_GOODS_CKPT = str(Path(__file__).resolve().parents[2] / "best.pt")
CAMERAS = {
    "head": {
        "rgb": "/head_camera/color/image_raw",
        "info": "/head_camera/color/camera_info",
        "depth": "/head_camera/aligned_depth_to_color/image_raw",
    },
    "left": {
        "rgb": "/left_camera/color/image_raw",
        "info": "/left_camera/color/camera_info",
        "depth": None,
    },
    "right": {
        "rgb": "/right_camera/color/image_raw",
        "info": "/right_camera/color/camera_info",
        "depth": None,
    },
}

# RGB and aligned depth arrive on separate ROS topics.  A previous-frame depth
# image is acceptable at the simulator's 24 Hz rate, but a frame from an old
# head pose is not: it was the source of metre-scale YOLO centre-point errors
# while the head was stepping between scan pitches.
# RGB and depth are rendered/published sequentially; at the observed simulator
# rate their stamps can differ by about 120 ms.  Scanning is accepted only
# after a 400 ms head-settle dwell, so a 150 ms bound is still safely within a
# stationary camera pose while accommodating that pipeline skew.
DEPTH_SYNC_MAX_NS = 150_000_000

# Foreground depth is measured only inside the useful part of a YOLO box.  The
# lower part is deliberately removed because supermarket shelf labels/rails
# frequently overlap the bottom of a detector box.
FOREGROUND_INSET_X = 0.12
FOREGROUND_INSET_TOP = 0.08
FOREGROUND_INSET_BOTTOM = 0.28
FOREGROUND_PERCENTILE = 30.0
FOREGROUND_BAND_M = 0.025
FOREGROUND_DEPTH_MIN_M = 0.15
FOREGROUND_DEPTH_MAX_M = 2.50


def stamp_ns(msg):
    return (int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec))


def closest_stamped_frame(frames, target_stamp_ns):
    """Return ``(stamp, frame)`` whose source stamp is nearest the RGB stamp."""
    if not frames:
        return None, None
    return min(frames, key=lambda frame: abs(target_stamp_ns - frame[0]))


def foreground_depth_estimate(depth_img, bbox_xyxy, K, T_cam_world):
    """Estimate a YOLO object's visible front surface from an aligned depth ROI.

    Edges and the lower shelf-label band are removed first.  A near, but not
    minimum, percentile rejects isolated close-depth noise; pixels in a narrow
    band around that percentile are then deprojected individually and combined
    with a spatial median.  The function is deliberately independent of class,
    shelf level and ArUco ID.
    """
    if depth_img is None or np.asarray(depth_img).ndim < 2:
        return None
    image_h, image_w = np.asarray(depth_img).shape[:2]
    try:
        x0, y0, x1, y1 = map(int, bbox_xyxy)
    except (TypeError, ValueError):
        return None
    x0, x1 = max(0, x0), min(image_w, x1 + 1)
    y0, y1 = max(0, y0), min(image_h, y1 + 1)
    box_w, box_h = x1 - x0, y1 - y0
    if box_w < 6 or box_h < 6:
        return None

    roi_x0 = x0 + max(1, int(round(box_w * FOREGROUND_INSET_X)))
    roi_x1 = x1 - max(1, int(round(box_w * FOREGROUND_INSET_X)))
    roi_y0 = y0 + max(1, int(round(box_h * FOREGROUND_INSET_TOP)))
    roi_y1 = y1 - max(1, int(round(box_h * FOREGROUND_INSET_BOTTOM)))
    if roi_x1 - roi_x0 < 3 or roi_y1 - roi_y0 < 3:
        return None

    roi_raw = np.asarray(depth_img)[roi_y0:roi_y1, roi_x0:roi_x1]
    roi_m = roi_raw.astype(np.float32)
    # Simulation/RealSense-style mono16 depth is millimetres.  Retain support
    # for an external float-depth publisher that already uses metres.
    positive = roi_m[np.isfinite(roi_m) & (roi_m > 0)]
    if not len(positive):
        return None
    if np.issubdtype(roi_raw.dtype, np.integer) or np.median(positive) > 10.0:
        roi_m *= 1e-3

    valid_mask = (
        np.isfinite(roi_m)
        & (roi_m >= FOREGROUND_DEPTH_MIN_M)
        & (roi_m <= FOREGROUND_DEPTH_MAX_M)
    )
    valid_depths = roi_m[valid_mask]
    min_valid = max(16, int(round(roi_m.size * 0.08)))
    if len(valid_depths) < min_valid:
        return None

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        return None
    rotation_world = np.asarray(T_cam_world, dtype=float)[:3, :3]
    translation_world = np.asarray(T_cam_world, dtype=float)[:3, 3]

    candidates = []
    # Multiple quantiles let the client use the same-frame ArUco pose to reject
    # a close occluder.  A single "nearest" percentile cannot distinguish the
    # target from a robot link that happens to overlap the YOLO rectangle.
    for percentile in (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0):
        depth_q = float(np.percentile(valid_depths, percentile))
        candidate_mask = (
            valid_mask
            & (roi_m >= depth_q - FOREGROUND_BAND_M)
            & (roi_m <= depth_q + FOREGROUND_BAND_M)
        )
        local_v, local_u = np.nonzero(candidate_mask)
        if len(local_u) < max(8, min_valid // 3):
            continue
        depths_m = roi_m[local_v, local_u].astype(float)
        depth_m = float(np.median(depths_m))
        # Adjacent percentiles often describe the same physical surface.
        if any(abs(depth_m - item["depth_m"]) < 0.015
               for item in candidates):
            continue
        pixels_u = local_u.astype(float) + roi_x0
        pixels_v = local_v.astype(float) + roi_y0
        points_cam = np.column_stack((
            (pixels_u - cx) * depths_m / fx,
            (pixels_v - cy) * depths_m / fy,
            depths_m,
        ))
        point_camera = np.median(points_cam, axis=0)
        point_world = rotation_world @ point_camera + translation_world
        depth_mad = float(np.median(np.abs(depths_m - depth_m)))
        candidates.append({
            "depth_m": depth_m,
            "depth_mad_m": depth_mad,
            "pixel": [
                int(round(float(np.median(pixels_u)))),
                int(round(float(np.median(pixels_v)))),
            ],
            "camera": point_camera.tolist(),
            "world": point_world.tolist(),
            "valid_count": int(len(depths_m)),
            "percentile": percentile,
        })
    if not candidates:
        return None
    primary = min(
        candidates,
        key=lambda item: abs(item["percentile"] - FOREGROUND_PERCENTILE))
    return {
        "front_depth_m": primary["depth_m"],
        "front_depth_mad_m": primary["depth_mad_m"],
        "front_pixel": primary["pixel"],
        "front_camera": primary["camera"],
        "front_world": primary["world"],
        "front_valid_count": primary["valid_count"],
        "front_roi_xyxy": [roi_x0, roi_y0, roi_x1 - 1, roi_y1 - 1],
        "front_percentile": primary["percentile"],
        "front_candidates": candidates,
    }


class KeleDetectNode(Node):
    def __init__(self, backend="blob", pub_res_img=True, device="auto",
                 weights=None, target_kind=None, confidence=0.45, show=False,
                 camera_names=("head",)):
        super().__init__("kele_detect")
        self.bridge = CvBridge()
        self.pub_res_img = pub_res_img
        self.show = bool(show)
        self.camera_names = tuple(dict.fromkeys(camera_names))
        unknown = set(self.camera_names) - set(CAMERAS)
        if unknown or not self.camera_names:
            raise ValueError(f"invalid camera names: {sorted(unknown)}")

        # Intrinsics/depth are maintained independently for every RGB stream.
        # Only the head camera has an aligned depth publisher in this server.
        self.K = {name: None for name in self.camera_names}
        # YOLO may take longer than one camera period.  Retain a short history
        # so the RGB callback can choose the closest depth by source stamp
        # after inference instead of blindly taking whichever frame is newest.
        self._depth_frames = {
            name: deque(maxlen=12) for name in self.camera_names}
        self._depth_lock = threading.Lock()
        # The pick controller switches inference from the head view to the
        # selected wrist view only after reaching pregrasp.  Keeping inactive
        # subscriptions connected avoids ROS graph churn while preventing
        # three camera streams from competing for one YOLO model.
        self._active_camera_lock = threading.Lock()
        self._active_cameras = set(self.camera_names)
        # Three YOLO RGB callbacks intentionally remain in the node's default
        # mutually-exclusive group because they share one model and one FK
        # object.  Fast state/depth callbacks use separate groups so continuous
        # inference cannot leave odom at the first scan station or starve the
        # aligned-depth stream.
        self.state_callback_group = MutuallyExclusiveCallbackGroup()
        self.depth_callback_group = MutuallyExclusiveCallbackGroup()

        # live robot state for the camera->world transform
        self.fk = MMK2FK()
        self.base_pos = None        # [x, y, z]
        self.base_quat = None       # [w, x, y, z]
        self.slide = 0.0
        self.head = [0.0, 0.0]
        self.left_arm = [0.0] * 6
        self.right_arm = [0.0] * 6

        # detector backend (pluggable)
        self.backend_name = backend
        if backend == "gt":
            self.detector = GtProjectionBackend(LAYOUT_JSON)
        elif backend == "yolo":
            checkpoint = weights or DEFAULT_GOODS_CKPT
            self.detector = YoloBackend(
                checkpoint, conf_thresh=confidence, device=device,
                target_class=target_kind)
            if self.detector.model is None:
                raise RuntimeError(
                    f"YOLO checkpoint could not be loaded: {checkpoint}. "
                    "Check that the .pt file was copied completely.")
        else:
            self.detector = BlobBackend()
        self.get_logger().info(
            f"goods detector up; backend={backend} target={target_kind or 'all'} "
            f"weights={weights or DEFAULT_GOODS_CKPT} "
            f"cameras={','.join(self.camera_names)}")

        # subscriptions
        for camera_name in self.camera_names:
            config = CAMERAS[camera_name]
            self.create_subscription(
                CameraInfo, config["info"],
                lambda msg, name=camera_name: self.camera_info_cb(name, msg),
                10, callback_group=self.depth_callback_group)
            if config["depth"] is not None:
                self.create_subscription(
                    Image, config["depth"],
                    lambda msg, name=camera_name: self.depth_cb(name, msg), 10,
                    callback_group=self.depth_callback_group)
            self.create_subscription(
                Image, config["rgb"],
                lambda msg, name=camera_name: self.rgb_cb(name, msg), 10)
        self.create_subscription(
            JointState, "/joint_states", self.js_cb, 10,
            callback_group=self.state_callback_group)
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom",
                                 self.odom_cb, 10,
                                 callback_group=self.state_callback_group)

        # publishers
        self.det_pub = self.create_publisher(Detection3DArray, "/kele/detections", 10)
        self.det_2d_pub = self.create_publisher(
            String, "/goods/yolo_detections", 10)
        self.img_pub = self.create_publisher(Image, "/kele/result_image", 5)

    def set_active_cameras(self, camera_names):
        requested = set(camera_names)
        unknown = requested - set(self.camera_names)
        if unknown or not requested:
            raise ValueError(f"invalid active camera names: {sorted(unknown)}")
        with self._active_camera_lock:
            changed = requested != self._active_cameras
            self._active_cameras = requested
        if changed:
            self.get_logger().info(
                f"active YOLO cameras={','.join(sorted(requested))}")

    # ---- state callbacks ----
    def camera_info_cb(self, camera_name: str, msg: CameraInfo):
        self.K[camera_name] = np.array(msg.k, dtype=float).reshape(3, 3)

    def depth_cb(self, camera_name: str, msg: Image):
        msg_stamp_ns = stamp_ns(msg)
        with self._depth_lock:
            self._depth_frames[camera_name].append((msg_stamp_ns, msg))

    def js_cb(self, msg: JointState):
        jp = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}
        self.slide = jp.get("slide_joint", self.slide)
        self.head = [jp.get("head_yaw_joint", self.head[0]),
                     jp.get("head_pitch_joint", self.head[1])]
        self.left_arm = [
            jp.get(f"left_arm_joint{i + 1}", self.left_arm[i])
            for i in range(6)]
        self.right_arm = [
            jp.get(f"right_arm_joint{i + 1}", self.right_arm[i])
            for i in range(6)]

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_pos = [p.x, p.y, p.z]
        self.base_quat = [q.w, q.x, q.y, q.z]

    # ---- camera->world transform from live state ----
    def camera_world_tmat(self, camera_name):
        """4x4 camera(optical)->world built from odom + slide/head via MMK2FK."""
        if self.base_pos is None or self.base_quat is None:
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
        }[camera_name]
        pos, quat = pose_getter()   # quat wxyz, world
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        if camera_name != "head":
            # left_cam/right_cam sites supply the OpenGL->OpenCV flip but are
            # siblings of the MuJoCo cameras.  Include the cameras' -30 degree
            # mounting pitch from arm_left.xml/arm_right.xml.
            T[:3, :3] = T[:3, :3] @ Rotation.from_euler(
                "x", -0.5236).as_matrix()
        return T

    @staticmethod
    def pixel_to_cam(K, u, v, depth_m):
        """Deproject a pixel + metric depth to a camera-optical-frame point."""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        return np.array([x, y, depth_m])

    @staticmethod
    def patch_depth_m(depth_img, u, v, r=4):
        """Median depth (m) over a patch, ignoring zero (invalid) pixels."""
        h, w = depth_img.shape[:2]
        y0, y1 = max(0, v - r), min(h, v + r + 1)
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        patch = depth_img[y0:y1, x0:x1].astype(np.float32)
        valid = patch[patch > 0]
        return float(np.median(valid)) * 1e-3 if len(valid) else 0.0

    # ---- main RGB callback ----
    def rgb_cb(self, camera_name: str, msg: Image):
        with self._active_camera_lock:
            if camera_name not in self._active_cameras:
                return
        K = self.K[camera_name]
        if K is None:
            return
        T_cam_world = self.camera_world_tmat(camera_name)
        if T_cam_world is None:
            return

        rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        rgb_stamp_ns = stamp_ns(msg)
        if self.backend_name == "yolo":
            # YOLO itself is RGB-only.  Run inference first so the separately
            # scheduled depth callback has time to receive the matching/future
            # aligned frame, then take a timestamp-bounded snapshot for the
            # foreground calculation.
            dets = self.detector.detect(rgb, None, K, T_cam_world)
        else:
            dets = None

        with self._depth_lock:
            depth_frames = list(self._depth_frames[camera_name])
        depth_stamp, depth_msg = closest_stamped_frame(
            depth_frames, rgb_stamp_ns)
        depth_delta_ns = (abs(rgb_stamp_ns - depth_stamp)
                          if depth_stamp is not None else None)
        depth_synced = (
            depth_msg is not None
            and depth_delta_ns is not None
            and depth_delta_ns <= DEPTH_SYNC_MAX_NS)
        depth = (self.bridge.imgmsg_to_cv2(depth_msg)
                 if depth_synced else None)
        if self.backend_name != "yolo":
            if depth is None:
                return
            dets = self.detector.detect(rgb, depth, K, T_cam_world)

        out = []
        vis = rgb.copy() if self.pub_res_img else rgb
        for d in dets:
            u, v = int(d["x"]), int(d["y"])
            depth_m = (self.patch_depth_m(depth, u, v)
                       if depth is not None else 0.0)

            w, h = int(d["w"]), int(d["h"])
            x0, y0 = max(0, u - w // 2), max(0, v - h // 2)
            x1 = min(rgb.shape[1] - 1, u + w // 2)
            y1 = min(rgb.shape[0] - 1, v + h // 2)
            rec = {
                "class": str(d["class"]),
                "conf": float(d.get("conf", 0.0)),
                "pixel_center": [u, v],
                "bbox_xyxy": [x0, y0, x1, y1],
                "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
                "stamp_ns": rgb_stamp_ns,
                "camera": camera_name,
                "camera_world_position": T_cam_world[:3, 3].tolist(),
                "camera_world_rotation": T_cam_world[:3, :3].reshape(-1).tolist(),
            }
            if depth_delta_ns is not None:
                rec["depth_delta_ms"] = depth_delta_ns * 1e-6
            foreground = foreground_depth_estimate(
                depth, [x0, y0, x1, y1], K, T_cam_world)
            if foreground is not None:
                rec.update(foreground)
            p_world = None
            if depth_m > 0.0:
                p_cam = self.pixel_to_cam(K, u, v, depth_m)
                p_world = (T_cam_world @ np.r_[p_cam, 1.0])[:3]
                rec["world"] = p_world.tolist()
            # coord-bridge validation logging (GT backend only)
            if "gt_world_pos" in d and p_world is not None:
                err = np.linalg.norm(p_world - d["gt_world_pos"]) * 1e3
                rec["gt_err_mm"] = err
                self.get_logger().info(
                    f"[{d.get('body','?')}] world={np.round(p_world,3)} "
                    f"gt={np.round(d['gt_world_pos'],3)} err={err:.2f}mm")
            out.append(rec)

            if self.pub_res_img:
                color = (0, 255, 0)
                cv2.rectangle(vis, (x0, y0), (x1, y1), color, 3)
                label = (f"{d['class']} {float(d.get('conf', 0.0)):.2f} "
                         f"cam={camera_name}"
                         + (f" d={depth_m:.2f}m" if depth_m > 0.0 else ""))
                text_y = max(20, y0 - 8)
                cv2.putText(vis, label, (x0, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2,
                            cv2.LINE_AA)
                if p_world is not None:
                    world_label = (f"xyz=({p_world[0]:.2f},"
                                   f"{p_world[1]:.2f},{p_world[2]:.2f})")
                    cv2.putText(
                        vis, world_label,
                        (x0, min(vis.shape[0] - 6, text_y + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1,
                        cv2.LINE_AA)
                if foreground is not None:
                    rx0, ry0, rx1, ry1 = foreground["front_roi_xyxy"]
                    front_u, front_v = foreground["front_pixel"]
                    cv2.rectangle(
                        vis, (rx0, ry0), (rx1, ry1), (0, 180, 255), 1)
                    cv2.drawMarker(
                        vis, (front_u, front_v), (0, 180, 255),
                        cv2.MARKER_CROSS, 12, 2)
                    front_label = (
                        f"front={foreground['front_depth_m']:.3f}m "
                        f"n={foreground['front_valid_count']}")
                    cv2.putText(
                        vis, front_label,
                        (x0, min(vis.shape[0] - 6, text_y + 38)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 180, 255), 1,
                        cv2.LINE_AA)
                cv2.circle(vis, (u, v), 4, (0, 0, 255), -1)

        self.publish_detections(out, msg.header.stamp)
        self.det_2d_pub.publish(String(
            data=json.dumps(out, separators=(",", ":"))))
        if self.pub_res_img:
            image_msg = self.bridge.cv2_to_imgmsg(vis, "bgr8")
            image_msg.header = msg.header
            self.img_pub.publish(image_msg)
            if self.show:
                cv2.imshow(f"YOLO product detections: {camera_name}", vis)
                cv2.waitKey(1)

    def publish_detections(self, recs, stamp):
        msg = Detection3DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        for r in recs:
            if "world" not in r:
                continue
            det = Detection3D()
            det.header = msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(r["class"])
            hyp.hypothesis.score = float(r["conf"])
            hyp.pose.pose.position.x = float(r["world"][0])
            hyp.pose.pose.position.y = float(r["world"][1])
            hyp.pose.pose.position.z = float(r["world"][2])
            det.results.append(hyp)
            msg.detections.append(det)
        self.det_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description="multi-class goods perception node")
    parser.add_argument("--backend", default="blob",
                        choices=["blob", "gt", "yolo"],
                        help="2-D detector backend (default: blob)")
    parser.add_argument("--no-result-image", action="store_true",
                        help="disable /kele/result_image publishing")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="YOLO inference device (default: auto)")
    parser.add_argument("--weights", default=DEFAULT_GOODS_CKPT,
                        help="Ultralytics checkpoint (default: repository best.pt)")
    parser.add_argument("--target-kind",
                        help="publish only this checkpoint class name")
    parser.add_argument("--confidence", type=float, default=0.45,
                        help="minimum YOLO confidence (default: 0.45)")
    parser.add_argument("--show", action="store_true",
                        help="show a live window containing YOLO boxes")
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS,
                        default=["head"],
                        help="RGB cameras used by the shared YOLO model")
    args = parser.parse_args()

    rclpy.init()
    node = KeleDetectNode(backend=args.backend, pub_res_img=not args.no_result_image,
                          device=args.device, weights=args.weights,
                          target_kind=args.target_kind,
                          confidence=args.confidence, show=args.show,
                          camera_names=args.cameras)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if args.show:
            cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
