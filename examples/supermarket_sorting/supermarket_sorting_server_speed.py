#!/usr/bin/env python3
"""Speed-only runtime overrides for the organizer's supermarket Server.

This wrapper deliberately loads the Server implementation already present in
the image.  It changes only render workload and straight-line wheel tracking;
layout, task generation, start pose, assets and referee behaviour remain the
organizer image's implementation.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import sys

import numpy as np


OFFICIAL_SERVER = Path(
    "/workspace/supermarket_sorting_task/examples/supermarket_sorting/"
    "supermarket_sorting_server.py")
CAMERA_IDS = {"head": 0, "left": 1, "right": 2}


def positive_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    result = float(default if value is None else value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"{name} must be a positive finite number, got {value!r}")
    return result


def sensor_camera_ids() -> list[int]:
    spec = os.getenv(
        "SUPERMARKET_RGB_CAMERAS", "head,left,right").strip().lower()
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
    if 0 not in camera_ids:
        raise ValueError(
            "SUPERMARKET_RGB_CAMERAS must include head for perception")
    return camera_ids


def load_official_server():
    if not OFFICIAL_SERVER.is_file():
        raise FileNotFoundError(
            f"organizer Server not found: {OFFICIAL_SERVER}")
    sys.path.insert(0, str(OFFICIAL_SERVER.parent))
    spec = importlib.util.spec_from_file_location(
        "organizer_supermarket_sorting_server", OFFICIAL_SERVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load organizer Server: {OFFICIAL_SERVER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_speed_overrides(server) -> None:
    original_build_config = server.build_config
    original_init = server.TaskMMK2ROS2.__init__
    original_render = server.TaskMMK2ROS2.render

    def build_config():
        config = original_build_config()
        config.obs_rgb_cam_id = sensor_camera_ids()
        config.render_set = dict(config.render_set)
        config.render_set["fps"] = positive_env_float(
            "SUPERMARKET_RENDER_FPS", float(config.render_set["fps"]))
        config.gs_render_sequential = os.getenv(
            "SUPERMARKET_GS_SEQUENTIAL", "1").strip().lower() in {
                "1", "true", "yes", "on"}
        print(
            "[server-speed] render profile: "
            f"rgb_cameras={config.obs_rgb_cam_id} depth_cameras="
            f"{config.obs_depth_cam_id} fps={config.render_set['fps']:g} "
            f"gs_sequential={int(config.gs_render_sequential)}")
        return config

    def init_with_speed_limits(self, config):
        original_init(self, config)
        # A GUI window otherwise asks the GS renderer for an additional free
        # camera.  Keep the window, but draw its diagnostic view with the much
        # cheaper MuJoCo renderer while the head sensor retains 3DGS.
        self.force_mujoco_display = os.getenv(
            "SUPERMARKET_FAST_MUJOCO_DISPLAY", "0").strip().lower() in {
                "1", "true", "yes", "on"}
        self.wheel_linear_error_limit = positive_env_float(
            "SUPERMARKET_WHEEL_LINEAR_ERROR_LIMIT_RADPS", 2.5)
        self.wheel_angular_error_limit = positive_env_float(
            "SUPERMARKET_WHEEL_ANGULAR_ERROR_LIMIT_RADPS", 2.5)
        print(
            "[server-speed] wheel tracking limits: "
            f"linear={self.wheel_linear_error_limit:g}rad/s "
            f"angular={self.wheel_angular_error_limit:g}rad/s "
            f"nonblocking_display={int(self.force_mujoco_display)}")

    def update_control_with_split_limits(self, action):
        # Common-mode wheel error controls translation; differential error
        # controls rotation.  Raising only the former preserves the official
        # angular response while giving straight travel more tracking force.
        wheel_error = self.tctr_base - self.sensor_wheel_qvel
        linear_error = float(0.5 * (wheel_error[0] + wheel_error[1]))
        angular_error = float(0.5 * (wheel_error[1] - wheel_error[0]))
        limited_linear = float(np.clip(
            linear_error,
            -self.wheel_linear_error_limit,
            self.wheel_linear_error_limit))
        limited_angular = float(np.clip(
            angular_error,
            -self.wheel_angular_error_limit,
            self.wheel_angular_error_limit))
        limited_wheel_error = np.array([
            limited_linear - limited_angular,
            limited_linear + limited_angular,
        ])
        wheel_force = self.pid_base_vel.output(
            limited_wheel_error, self.mj_model.opt.timestep)
        self.mj_data.ctrl[:2] = np.clip(
            wheel_force,
            self.mj_model.actuator_ctrlrange[:2, 0],
            self.mj_model.actuator_ctrlrange[:2, 1])
        self.mj_data.ctrl[2:self.njctrl] = np.clip(
            action[2:self.njctrl],
            self.mj_model.actuator_ctrlrange[2:self.njctrl, 0],
            self.mj_model.actuator_ctrlrange[2:self.njctrl, 1])

    def render_with_nonblocking_display(self):
        """Render sensors once, then draw a display without another GS pass.

        The organizer image's SimulatorBase.render does not know about the
        ``force_mujoco_display`` attribute.  In a windowed run it therefore
        asks the Gaussian renderer for the selected display camera in addition
        to the head sensor.  Selecting the head camera is particularly bad
        when the window size differs from the sensor size: the same camera is
        rendered through 3DGS twice at two resolutions and can stall CUDA.

        Hide the GLFW window only while the original method produces sensor
        frames.  Afterwards, reuse the already-rendered head RGB/depth image;
        non-sensor viewpoints use the inexpensive MuJoCo renderer.  Perception
        topics and their 3DGS images are unchanged.
        """
        fast_display = bool(
            getattr(self, "force_mujoco_display", False)
            and not self.config.headless
            and self.window is not None)
        if not fast_display:
            return original_render(self)

        import cv2
        import glfw
        from OpenGL import GL as gl
        import time

        display_window = self.window
        # SimulatorBase uses ``window is not None`` to decide whether to add a
        # display camera to the GS batch.  Temporarily hiding it leaves exactly
        # the configured sensor cameras in that batch.
        self.window = None
        try:
            original_render(self)
        finally:
            self.window = display_window

        try:
            if glfw.window_should_close(display_window):
                self.running = False
                return

            window_width, window_height = self.get_current_window_size()
            depth_mode = self.renderer._depth_rendering
            if not depth_mode:
                img_vis = self.img_rgb_obs_s.get(self.cam_id)
                if img_vis is None:
                    self.update_renderer_window_size(
                        window_width, window_height)
                    img_vis = self.getRgbImg(self.cam_id)
            else:
                img_depth = self.img_depth_obs_s.get(self.cam_id)
                if img_depth is None:
                    self.update_renderer_window_size(
                        window_width, window_height)
                    img_depth = self.getDepthImg(self.cam_id)
                img_vis = (
                    None if img_depth is None else cv2.applyColorMap(
                        cv2.convertScaleAbs(
                            img_depth, alpha=255.0 / self.config.max_render_depth),
                        cv2.COLORMAP_JET))

            if (img_vis is not None
                    and (img_vis.shape[1], img_vis.shape[0])
                    != (window_width, window_height)):
                img_vis = cv2.resize(
                    img_vis, (window_width, window_height),
                    interpolation=cv2.INTER_LINEAR)

            glfw.make_context_current(display_window)
            fb_width, fb_height = glfw.get_framebuffer_size(display_window)
            gl.glViewport(0, 0, fb_width, fb_height)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            if img_vis is not None:
                img_vis = np.ascontiguousarray(img_vis[::-1])
                gl.glWindowPos2i(0, 0)
                gl.glDrawPixels(
                    img_vis.shape[1], img_vis.shape[0], gl.GL_RGB,
                    gl.GL_UNSIGNED_BYTE, img_vis.tobytes())
            glfw.swap_buffers(display_window)
            glfw.poll_events()

            if self.config.sync:
                current_time = time.time()
                wait_time = max(
                    1.0 / self.render_fps
                    - (current_time - self.last_render_time), 0.0)
                if wait_time > 0.0:
                    time.sleep(wait_time)
                self.last_render_time = time.time()
        except Exception as exc:
            # Match the organizer renderer's non-fatal window error policy;
            # sensor publication must continue even if the display fails.
            print(f"[server-speed] display render error: {exc}")

    server.build_config = build_config
    server.TaskMMK2ROS2.__init__ = init_with_speed_limits
    server.TaskMMK2ROS2.updateControl = update_control_with_split_limits
    server.TaskMMK2ROS2.render = render_with_nonblocking_display


def main() -> None:
    server = load_official_server()
    install_speed_overrides(server)
    server.main()


if __name__ == "__main__":
    main()
