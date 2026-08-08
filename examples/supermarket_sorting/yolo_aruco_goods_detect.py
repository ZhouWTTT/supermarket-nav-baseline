#!/usr/bin/env python3
"""Show YOLO goods boxes and shelf ArUco markers in the same image.

The simulator/server must already be publishing the MMK2 camera topics.  From
the repository root, run::

    source /opt/ros/humble/setup.bash
    python3 examples/supermarket_sorting/yolo_aruco_goods_detect.py

The default input is the head camera and the default checkpoint is ``best.pt``
in the repository root.  Press ``q`` or ``Esc`` in an OpenCV window to stop.

Besides the live windows, each selected camera publishes:

* ``/goods/yolo_aruco/<camera>/result_image`` (annotated sensor_msgs/Image)
* ``/goods/yolo_aruco/<camera>/detections`` (JSON std_msgs/String)
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from perception.backends import YoloBackend


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPOSITORY_ROOT / "best.pt"
VALID_ARUCO_IDS = set(range(45))
CAMERAS = {
    "head": "/head_camera/color/image_raw",
    "left": "/left_camera/color/image_raw",
    "right": "/right_camera/color/image_raw",
}

YOLO_COLOR = (0, 255, 0)       # BGR: green
ARUCO_COLOR = (255, 0, 255)    # BGR: magenta


def message_stamp_ns(msg: Image) -> int:
    """Convert a ROS Image timestamp to integer nanoseconds."""
    return (int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec))


class YoloArucoGoodsDetector(Node):
    """Run the project's YOLO backend and ArUco detector on camera frames."""

    def __init__(self, weights: str, confidence: float, device: str,
                 camera_names: list[str], show: bool = True,
                 publish_result_image: bool = True):
        super().__init__("yolo_aruco_goods_detector")
        self.bridge = CvBridge()
        self.show = show
        self.publish_result_image = publish_result_image

        self.yolo = YoloBackend(
            weights, conf_thresh=confidence, device=device,
            target_class=None)
        if self.yolo.model is None:
            raise RuntimeError(
                f"YOLO weights could not be loaded: {weights}. "
                "Check the path and that ultralytics is installed.")

        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "This OpenCV build has no aruco module; install an OpenCV "
                "build with ArUco support (the project Docker image includes it).")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "DetectorParameters"):
            parameters = cv2.aruco.DetectorParameters()
        else:
            parameters = cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            self.detect_aruco = lambda gray: detector.detectMarkers(gray)[:2]
        else:
            self.detect_aruco = lambda gray: cv2.aruco.detectMarkers(
                gray, dictionary, parameters=parameters)[:2]

        self.result_publishers = {}
        self.detection_publishers = {}
        for camera_name in camera_names:
            prefix = f"/goods/yolo_aruco/{camera_name}"
            self.result_publishers[camera_name] = self.create_publisher(
                Image, f"{prefix}/result_image", 5)
            self.detection_publishers[camera_name] = self.create_publisher(
                String, f"{prefix}/detections", 10)
            self.create_subscription(
                Image, CAMERAS[camera_name],
                lambda msg, name=camera_name: self.image_callback(name, msg),
                10)

        classes = ", ".join(self.yolo.class_names.values())
        self.get_logger().info(
            f"YOLO + ArUco detector ready; cameras={','.join(camera_names)}; "
            f"classes=[{classes}]")

    @staticmethod
    def collect_aruco(corners, ids) -> list[dict]:
        """Return only the shelf marker IDs used by this project (0--44)."""
        detections = []
        if ids is None:
            return detections
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in VALID_ARUCO_IDS:
                continue
            pixels = np.asarray(marker_corners, dtype=float).reshape(4, 2)
            detections.append({
                "id": marker_id,
                "pixel_center": pixels.mean(axis=0).tolist(),
                "corners": pixels.tolist(),
            })
        detections.sort(key=lambda item: item["id"])
        return detections

    @staticmethod
    def draw_yolo(image: np.ndarray, detections: list[dict]) -> list[dict]:
        """Draw YOLO rectangles and return JSON-serialisable records."""
        image_h, image_w = image.shape[:2]
        records = []
        for detection in detections:
            center_x, center_y = int(detection["x"]), int(detection["y"])
            box_w, box_h = int(detection["w"]), int(detection["h"])
            x0 = max(0, center_x - box_w // 2)
            y0 = max(0, center_y - box_h // 2)
            x1 = min(image_w - 1, center_x + box_w // 2)
            y1 = min(image_h - 1, center_y + box_h // 2)
            class_name = str(detection["class"])
            confidence = float(detection.get("conf", 0.0))

            cv2.rectangle(image, (x0, y0), (x1, y1), YOLO_COLOR, 2)
            label = f"{class_name} {confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            label_top = max(0, y0 - text_h - baseline - 4)
            cv2.rectangle(
                image, (x0, label_top),
                (min(image_w - 1, x0 + text_w + 6), y0), YOLO_COLOR, -1)
            cv2.putText(
                image, label, (x0 + 3, max(text_h, y0 - baseline - 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
                cv2.LINE_AA)
            records.append({
                "class": class_name,
                "confidence": confidence,
                "pixel_center": [center_x, center_y],
                "bbox_xyxy": [x0, y0, x1, y1],
            })
        return records

    @staticmethod
    def draw_aruco(image: np.ndarray, detections: list[dict]) -> None:
        """Draw marker quadrilaterals, centres, and IDs."""
        for detection in detections:
            corners = np.rint(detection["corners"]).astype(np.int32)
            center = tuple(np.rint(detection["pixel_center"]).astype(int))
            cv2.polylines(
                image, [corners.reshape(-1, 1, 2)], True,
                ARUCO_COLOR, 3, cv2.LINE_AA)
            cv2.drawMarker(
                image, center, ARUCO_COLOR, cv2.MARKER_CROSS, 12, 2)
            label_x = int(corners[:, 0].min())
            label_y = max(18, int(corners[:, 1].min()) - 7)
            cv2.putText(
                image, f"ArUco ID={detection['id']}", (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, ARUCO_COLOR, 2,
                cv2.LINE_AA)

    @staticmethod
    def draw_summary(image: np.ndarray, camera_name: str,
                     yolo_count: int, aruco_count: int,
                     elapsed_ms: float) -> None:
        summary = (f"{camera_name} | YOLO goods: {yolo_count} | "
                   f"ArUco: {aruco_count} | {elapsed_ms:.1f} ms")
        (text_w, _), _ = cv2.getTextSize(
            summary, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(image, (0, 0), (min(image.shape[1] - 1, text_w + 14), 30),
                      (20, 20, 20), -1)
        cv2.putText(image, summary, (7, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 2, cv2.LINE_AA)

    def image_callback(self, camera_name: str, msg: Image) -> None:
        started = time.perf_counter()
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # YoloBackend is RGB-only despite its generic RGB-D backend signature;
        # K and T are unused by this backend.
        yolo_raw = self.yolo.detect(image, None, np.eye(3), None)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_corners, aruco_ids = self.detect_aruco(gray)
        aruco_records = self.collect_aruco(aruco_corners, aruco_ids)

        annotated = image.copy()
        yolo_records = self.draw_yolo(annotated, yolo_raw)
        self.draw_aruco(annotated, aruco_records)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.draw_summary(
            annotated, camera_name, len(yolo_records), len(aruco_records),
            elapsed_ms)

        payload = {
            "camera": camera_name,
            "stamp_ns": message_stamp_ns(msg),
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "yolo": yolo_records,
            "aruco": aruco_records,
        }
        self.detection_publishers[camera_name].publish(
            String(data=json.dumps(payload, separators=(",", ":"))))

        if self.publish_result_image:
            result_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            result_msg.header = msg.header
            self.result_publishers[camera_name].publish(result_msg)

        if self.show:
            cv2.imshow(f"YOLO + ArUco goods detection [{camera_name}]", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.get_logger().info("q/Esc pressed; stopping")
                rclpy.try_shutdown()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display YOLO goods boxes and shelf ArUco markers")
    parser.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS),
        help="Ultralytics YOLO checkpoint (default: repository best.pt)")
    parser.add_argument(
        "--confidence", type=float, default=0.45,
        help="minimum YOLO confidence in [0, 1] (default: 0.45)")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="YOLO inference device (default: auto)")
    parser.add_argument(
        "--cameras", nargs="+", choices=CAMERAS, default=["head"],
        help="camera streams to detect (default: head)")
    parser.add_argument(
        "--no-show", action="store_true",
        help="do not open OpenCV windows; ROS result images are still published")
    parser.add_argument(
        "--no-result-image", action="store_true",
        help="do not publish annotated result images")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = None
    try:
        node = YoloArucoGoodsDetector(
            weights=args.weights,
            confidence=args.confidence,
            device=args.device,
            camera_names=args.cameras,
            show=not args.no_show,
            publish_result_image=not args.no_result_image)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
