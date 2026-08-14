#!/usr/bin/env python3
"""Persistent formal-run YOLO and head-ArUco publishers.

The competition runner keeps this process alive across single-item motion
workers.  Each trip still owns exactly one motion controller and delivers one
item; only the read-only perception model is reused to avoid loading the same
checkpoint for every order.
"""

from __future__ import annotations

import argparse
import pathlib

# Importing the integrated module installs the MuJoCo XML compatibility shim
# needed by MMK2FK in the official client image before either detector is
# constructed.
import integrated_nav_pick_place as flow


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "perception" / "checkpoints" / "best.pt"
pick = flow.pick


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="persistent all-class supermarket perception")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument(
        "--max-inference-hz", type=float, default=12.0,
        help="maximum YOLO source-frame rate while perception is enabled")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--ready-file", required=True,
        help="runner-owned readiness sentinel written after both nodes load")
    parser.add_argument(
        "--publish-result-images", action="store_true",
        help="publish annotated YOLO/ArUco images for an explicit viewer")
    args = parser.parse_args()
    if not pathlib.Path(args.weights).is_file():
        parser.error(f"weights not found: {args.weights}")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if not 0.0 < args.max_inference_hz < float("inf"):
        parser.error("--max-inference-hz must be finite and positive")
    return args


def main() -> None:
    from run_log import start_run_log
    start_run_log("persistent_perception")
    args = parse_args()

    import rclpy
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import Bool

    rclpy.init()
    nodes = []
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        yolo_node = pick.KeleDetectNode(
            backend="yolo",
            pub_res_img=args.publish_result_images,
            device=args.device,
            weights=str(pathlib.Path(args.weights).resolve()),
            target_kind=None,
            confidence=args.confidence,
            show=False,
            camera_names=("head",),
            max_inference_hz=args.max_inference_hz,
        )
        aruco_node = pick.ArucoDetectNode(
            "head",
            marker_size=pick.MARKER_SIZE_M,
            publish_tf=False,
            publish_result_image=args.publish_result_images,
        )
        # Navigation and manipulation do not consume detections.  Keeping
        # YOLO active there competes with the GPU-rendered server for most of
        # every trip, so start paused and let the active motion worker request
        # perception only in SCAN/REVISIT/RECHECK states.
        yolo_node.set_enabled(False)
        aruco_node.set_enabled(False)
        control_node = Node("persistent_perception_control")

        def perception_control(message: Bool) -> None:
            yolo_node.set_enabled(message.data)
            aruco_node.set_enabled(message.data)

        control_node.create_subscription(
            Bool, "/supermarket_sorting/perception_enable",
            perception_control, 10)
        nodes = [yolo_node, aruco_node, control_node]
        for node in nodes:
            executor.add_node(node)
        ready_path = pathlib.Path(args.ready_file)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = ready_path.with_suffix(ready_path.suffix + ".tmp")
        temporary.write_text("ready\n", encoding="utf-8")
        temporary.replace(ready_path)
        yolo_node.get_logger().info(
            "persistent perception ready; model will be reused across "
            "single-item workers and inference is paused outside scan states")
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            executor.shutdown()
        except Exception:  # noqa: BLE001 - best-effort process cleanup
            pass
        for node in nodes:
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001 - best-effort process cleanup
                pass
        try:
            pathlib.Path(args.ready_file).unlink(missing_ok=True)
        except OSError:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
