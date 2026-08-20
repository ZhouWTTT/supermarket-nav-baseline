#!/usr/bin/env python3
"""Run the pick-controller unit tests on the host with stubbed ROS deps."""
import pathlib
import sys
import types
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


class FakeNode:
    def __init__(self, *a, **k):
        pass

    def get_logger(self):
        import logging
        return logging.getLogger("stub")

    def create_publisher(self, *a, **k):
        return None

    def create_subscription(self, *a, **k):
        return None

    def create_timer(self, *a, **k):
        return None

    def get_clock(self):
        class Clock:
            def now(self):
                class T:
                    nanoseconds = 0
                return T()
        return Clock()


def stub(mod_name, attrs=None):
    # Never shadow a real (container) ROS install: only add missing modules
    # and missing attributes.
    m = sys.modules.get(mod_name)
    if m is None:
        m = types.ModuleType(mod_name)
        sys.modules[mod_name] = m
    for k, v in (attrs or {}).items():
        if not hasattr(m, k):
            setattr(m, k, v)
    return m


class _Rotation:
    @staticmethod
    def from_euler(seq, angles, degrees=False):
        angles = np.atleast_1d(angles)
        a = float(angles[0])
        c, s = np.cos(a), np.sin(a)
        if seq == "x":
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif seq == "z":
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            R = np.eye(3)
        return type("Rot", (), {"as_matrix": lambda self: R})()


def install_stubs():
    rclpy = stub("rclpy", {
        "Node": FakeNode,
        "ExternalShutdownException": Exception,
        "MultiThreadedExecutor": object,
        "QoSProfile": object,
        "ReliabilityPolicy": object,
        "DurabilityPolicy": object,
        "ok": lambda: True,
        "shutdown": lambda: None,
        "init": lambda *a, **k: None,
    })
    stub("rclpy.executors", {
        "ExternalShutdownException": Exception,
        "MultiThreadedExecutor": object,
    })
    stub("rclpy.node", {"Node": FakeNode})
    stub("rclpy.qos", {
        "QoSProfile": object,
        "ReliabilityPolicy": object,
        "DurabilityPolicy": object,
    })
    stub("rclpy._rclpy_pybind11", {"RCLError": Exception})
    stub("rclpy.callback_groups", {
        "CallbackGroup": object,
        "MutuallyExclusiveCallbackGroup": object,
        "ReentrantCallbackGroup": object,
    })
    stub("rclpy.task", {})
    for m in ("geometry_msgs", "nav_msgs", "sensor_msgs", "std_msgs",
              "cv_bridge"):
        stub(m)
    stub("geometry_msgs.msg", {
        "Twist": type("Twist", (), {}),
        "PoseArray": type("PoseArray", (), {}),
        "PoseStamped": type("PoseStamped", (), {}),
        "Pose": type("Pose", (), {}),
        "TransformStamped": type("TransformStamped", (), {}),
    })
    stub("nav_msgs.msg", {
        "Odometry": type("Odometry", (), {}),
        "Path": type("Path", (), {}),
    })
    stub("sensor_msgs.msg", {
        "Image": type("Image", (), {}),
        "JointState": type("JointState", (), {}),
        "LaserScan": type("LaserScan", (), {}),
        "CameraInfo": type("CameraInfo", (), {}),
        "CompressedImage": type("CompressedImage", (), {}),
    })
    stub("std_msgs.msg", {
        "Float64MultiArray": type("Float64MultiArray", (), {}),
        "String": type("String", (), {}),
        "Int32MultiArray": type("Int32MultiArray", (), {}),
        "Float32MultiArray": type("Float32MultiArray", (), {}),
        "Bool": type("Bool", (), {}),
    })
    stub("cv_bridge", {"CvBridge": type("CvBridge", (), {})})
    stub("cv2", {
        "imdecode": None, "imwrite": None, "resize": None,
        "cvtColor": None, "COLOR_BGR2RGB": 4, "imread": None,
    })
    stub("scipy")
    stub("scipy.spatial")

    def _max_filter(input, footprint=None, size=None):
        return np.asarray(input)

    def _dist_edt(input, *a, **k):
        return np.asarray(input, dtype=float)

    stub("scipy.ndimage", {
        "maximum_filter": _max_filter,
        "distance_transform_edt": _dist_edt,
    })
    stub("scipy.spatial.transform", {"Rotation": _Rotation})
    stub("tf2_ros", {"TransformBroadcaster": object})
    stub("vision_msgs", {"msg": None})
    stub("vision_msgs.msg", {
        "Detection2DArray": type("Detection2DArray", (), {}),
        "Detection2D": type("Detection2D", (), {}),
        "ObjectHypothesisWithPose": type("ObjectHypothesisWithPose", (), {}),
        "BoundingBox2D": type("BoundingBox2D", (), {}),
        "Detection3DArray": type("Detection3DArray", (), {}),
        "Detection3D": type("Detection3D", (), {}),
    })
    stub("discoverse")
    stub("discoverse.robots")
    stub("discoverse.robots.mmk2")
    stub("discoverse.robots.mmk2.mmk2_fik", {
        "MMK2FIK": type("FakeMMK2FIK", (), {})})
    stub("discoverse.robots.mmk2.mmk2_fk", {
        "MMK2FK": type("FakeMMK2FK", (), {})})


if __name__ == "__main__":
    install_stubs()
    sys.path.insert(0, str(REPO / "examples" / "supermarket_sorting"))
    sys.path.insert(0, str(REPO / "tests"))
    names = sys.argv[1:] or [
        "test_dual_tissue_safety",
        "test_yolo_close_recheck",
        "test_snapshot_memory_reroute",
    ]
    suite = unittest.TestSuite()
    for name in names:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
