#!/usr/bin/env python3
"""即插即用的运行录像脚本（独立于抓取流程，不影响镜像）。

订阅一个 ROS 相机话题，把画面写成 MP4。无论流程正常结束、被 Ctrl-C /
docker stop 停掉、还是长时间收不到帧（server 挂了），都会把视频落盘。

用法（在 client 容器内，与仿真同一 ROS_DOMAIN_ID）::

    python3 record_run.py \
        --topic /head_camera/color/image_raw \
        --output /workspace/supermarket_sorting_task/logs/run_xxx.mp4

退出条件：
  * SIGTERM/SIGINT（docker stop 等）→ 立即收尾保存；
  * 超过 --no-frame-timeout 秒没有新帧（server 停止）→ 收尾保存；
  * 启动后 --start-timeout 秒内一帧都没有 → 写一帧黑帧占位并保存，
    保证“每次运行都有一条视频”。
"""

from __future__ import annotations

import argparse
import os
import signal
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


class RunRecorder(Node):
    def __init__(self, topic: str, output: str, fps: float,
                 no_frame_timeout_s: float, start_timeout_s: float,
                 width: int = 0, height: int = 0):
        super().__init__("run_recorder")
        self.bridge = CvBridge()
        self.output = output
        self.fps = float(fps)
        self.no_frame_timeout_s = float(no_frame_timeout_s)
        self.start_timeout_s = float(start_timeout_s)
        self.target_width = int(width)
        self.target_height = int(height)
        self.writer = None
        self.size = (640, 480)
        self.frame_count = 0
        self.last_frame_t = time.monotonic()
        self.start_t = time.monotonic()
        self.done = False
        self.create_subscription(Image, topic, self._image_cb, 5)
        self.create_timer(0.5, self._tick)
        self.get_logger().info(
            f"run recorder ready; topic={topic} output={output} "
            f"fps={self.fps}")

    def _open_writer(self) -> None:
        if self.writer is not None:
            return
        out_dir = os.path.dirname(self.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        try:
            self.writer = cv2.VideoWriter(
                self.output, cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps, self.size)
            if not self.writer.isOpened():
                raise RuntimeError("mp4v codec unavailable")
        except Exception:
            self.output = os.path.splitext(self.output)[0] + ".avi"
            self.writer = cv2.VideoWriter(
                self.output, cv2.VideoWriter_fourcc(*"MJPG"),
                self.fps, self.size)
        self.get_logger().info(f"video writer opened: {self.output}")

    def _image_cb(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:  # noqa: BLE001 - skip malformed frames
            return
        if img is None or img.size == 0:
            return
        h, w = img.shape[:2]
        if self.target_width > 0 and self.target_height > 0:
            self.size = (self.target_width, self.target_height)
            img = cv2.resize(
                img, self.size, interpolation=cv2.INTER_AREA)
        else:
            self.size = (w, h)
        self._open_writer()
        self.writer.write(img)
        self.frame_count += 1
        self.last_frame_t = time.monotonic()

    def _tick(self) -> None:
        now = time.monotonic()
        if self.writer is None:
            if now - self.start_t >= self.start_timeout_s:
                self.get_logger().warn(
                    f"no camera frame within {self.start_timeout_s:.0f}s; "
                    "writing a placeholder frame so a video still exists")
                self._open_writer()
                placeholder = np.zeros(
                    (self.size[1], self.size[0], 3), dtype=np.uint8)
                cv2.putText(
                    placeholder, "no camera feed", (40, self.size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2,
                    cv2.LINE_AA)
                self.writer.write(placeholder)
                self.frame_count = 1
                self._finalize()
                if rclpy.ok():
                    rclpy.shutdown()
            return
        if now - self.last_frame_t >= self.no_frame_timeout_s:
            self.get_logger().warn(
                f"no camera frame for {self.no_frame_timeout_s:.0f}s "
                "(server may have stopped); finalizing video")
            self._finalize()
            if rclpy.ok():
                rclpy.shutdown()

    def _finalize(self) -> None:
        if self.done:
            return
        self.done = True
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.get_logger().info(
            f"VIDEO SAVED: {self.output} frames={self.frame_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="record a ROS camera stream to a video file")
    parser.add_argument("--topic", default="/head_camera/color/image_raw")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--no-frame-timeout", type=float, default=10.0,
                        help="finalize after this many seconds without frames")
    parser.add_argument("--start-timeout", type=float, default=30.0,
                        help="wait this long for the first frame before "
                             "writing a placeholder")
    parser.add_argument("--width", type=int, default=0,
                        help="output width (0 = keep camera resolution)")
    parser.add_argument("--height", type=int, default=0,
                        help="output height (0 = keep camera resolution)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = RunRecorder(
        args.topic, args.output, args.fps,
        args.no_frame_timeout, args.start_timeout,
        args.width, args.height)

    stop = threading.Event()

    def _signal_handler(_signum, _frame):
        stop.set()
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._finalize()
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
