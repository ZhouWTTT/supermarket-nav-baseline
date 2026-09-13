#!/usr/bin/env python3
"""Run the image-owned Server with a sensor-backed display camera.

The official free-camera window needs an additional Gaussian render.  For a
smooth recording view, select an RGB observation camera that the Server must
already render and let the existing simulator display reuse that frame.
No simulation, sensor, control, FPS or task configuration is changed.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


OFFICIAL_SERVER = Path(
    "/workspace/supermarket_sorting_task/examples/supermarket_sorting/"
    "supermarket_sorting_server.py")


def load_official_server():
    if not OFFICIAL_SERVER.is_file():
        raise FileNotFoundError(f"official Server not found: {OFFICIAL_SERVER}")
    sys.path.insert(0, str(OFFICIAL_SERVER.parent))
    spec = importlib.util.spec_from_file_location(
        "official_supermarket_sorting_server", OFFICIAL_SERVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official Server: {OFFICIAL_SERVER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_sensor_display(server) -> None:
    original_init = server.TaskMMK2ROS2.__init__

    def init_with_sensor_display(self, config):
        original_init(self, config)
        camera_id = int(os.getenv("SUPERMARKET_DISPLAY_CAMERA_ID", "0"))
        rgb_camera_ids = tuple(config.obs_rgb_cam_id or ())
        if camera_id not in rgb_camera_ids:
            raise ValueError(
                "display camera must already be an RGB observation camera; "
                f"got {camera_id}, available={rgb_camera_ids}")
        if not 0 <= camera_id < len(self.camera_names):
            raise ValueError(
                f"display camera id {camera_id} outside camera list "
                f"0..{len(self.camera_names) - 1}")
        self.cam_id = camera_id
        self.camera_pose_changed = True
        print(
            "[server-display] reusing rendered sensor frame: "
            f"camera={self.camera_names[camera_id]} id={camera_id} "
            "additional_gs_pass=0")

    server.TaskMMK2ROS2.__init__ = init_with_sensor_display


def main() -> None:
    server = load_official_server()
    install_sensor_display(server)
    server.main()


if __name__ == "__main__":
    main()
