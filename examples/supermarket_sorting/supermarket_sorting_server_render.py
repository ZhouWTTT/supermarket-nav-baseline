#!/usr/bin/env python3
"""Optional render acceleration for the image-owned competition Server.

This wrapper deliberately changes only the Server render workload:

* publish selected RGB observation cameras (accelerated default: head only),
* render them at a configurable FPS (accelerated default: 12),
* allow batched/non-sequential Gaussian rendering.

Physics stepping, wheel control, task randomisation, referee state and lidar
remain owned by the official Server module in the container.  Launchers select
this wrapper only when their explicit ``server acceleration`` option is on;
otherwise they execute the official entry point directly.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import sys


OFFICIAL_SERVER = Path(
    "/workspace/supermarket_sorting_task/examples/supermarket_sorting/"
    "supermarket_sorting_server.py")
CAMERA_IDS = {"head": 0, "left": 1, "right": 2}


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def positive_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    result = float(default if value is None else value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"{name} must be a positive finite number, got {value!r}")
    return result


def sensor_camera_ids() -> list[int]:
    spec = os.getenv("SUPERMARKET_RGB_CAMERAS", "head").strip().lower()
    names = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [name for name in names if name not in CAMERA_IDS]
    if unknown:
        raise ValueError(
            "SUPERMARKET_RGB_CAMERAS contains unknown cameras: "
            + ", ".join(unknown))
    camera_ids = []
    for name in names:
        camera_id = CAMERA_IDS[name]
        if camera_id not in camera_ids:
            camera_ids.append(camera_id)
    if CAMERA_IDS["head"] not in camera_ids:
        raise ValueError(
            "SUPERMARKET_RGB_CAMERAS must include head for perception")
    return camera_ids


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


def install_render_overrides(server) -> None:
    """Install render-only overrides on an imported official Server module."""
    original_build_config = server.build_config
    original_init = server.TaskMMK2ROS2.__init__

    def build_config():
        config = original_build_config()
        config.obs_rgb_cam_id = sensor_camera_ids()
        config.render_set = dict(config.render_set)
        config.render_set["fps"] = positive_env_float(
            "SUPERMARKET_RENDER_FPS", 12.0)
        config.gs_render_sequential = env_flag(
            "SUPERMARKET_GS_SEQUENTIAL", False)
        print(
            "[server-acceleration] enabled: "
            f"rgb_cameras={config.obs_rgb_cam_id} "
            f"depth_cameras={config.obs_depth_cam_id} "
            f"fps={config.render_set['fps']:g} "
            f"gs_sequential={int(config.gs_render_sequential)}; "
            "physics_and_control=official")
        return config

    def init_with_optional_sensor_display(self, config):
        original_init(self, config)
        display_camera = os.getenv("SUPERMARKET_DISPLAY_CAMERA_ID")
        if display_camera is None:
            return
        camera_id = int(display_camera)
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
            "[server-acceleration] display reuses sensor frame: "
            f"camera={self.camera_names[camera_id]} id={camera_id} "
            "additional_gs_pass=0")

    server.build_config = build_config
    server.TaskMMK2ROS2.__init__ = init_with_optional_sensor_display


def main() -> None:
    server = load_official_server()
    install_render_overrides(server)
    server.main()


if __name__ == "__main__":
    main()
