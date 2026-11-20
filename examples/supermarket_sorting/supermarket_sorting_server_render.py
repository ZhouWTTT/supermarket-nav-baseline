#!/usr/bin/env python3
"""Render-only runtime overrides for the organizer's supermarket Server.

This wrapper loads the Server implementation already present in the image and
changes only the render workload (which sensor cameras are rendered, FPS and
GS sequential order), plus an optional cheaper MuJoCo display.  It deliberately
does NOT touch wheel control/tracking, so chassis dynamics stay official.

Purpose: the organizer Server defaults to rendering head+left+right 3DGS
cameras at 640x480, which can exceed an 8 GB GPU.  Publishing only the head
camera (the one the client perception consumes) removes the two extra GS
passes and avoids CUDA out of memory.
"""

from __future__ import annotations

import importlib.util
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
    if result <= 0.0:
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


def install_render_overrides(server) -> None:
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
            "[server-render] render profile: "
            f"rgb_cameras={config.obs_rgb_cam_id} depth_cameras="
            f"{config.obs_depth_cam_id} fps={config.render_set['fps']:g} "
            f"gs_sequential={int(config.gs_render_sequential)}")
        return config

    def init_with_render_display(self, config):
        original_init(self, config)
        # A GUI window otherwise asks the GS renderer for an additional free
        # camera.  Keep the window, but draw its diagnostic view with the much
        # cheaper MuJoCo renderer while sensor cameras retain 3DGS.
        self.force_mujoco_display = os.getenv(
            "SUPERMARKET_FAST_MUJOCO_DISPLAY", "0").strip().lower() in {
                "1", "true", "yes", "on"}
        print(
            "[server-render] nonblocking_display="
            f"{int(self.force_mujoco_display)}")

    def render_with_nonblocking_display(self):
        """Render sensors once, then draw a display without another GS pass.

        Temporarily hides the display window so the GS batch contains exactly
        the configured sensor cameras, then draws the display from the already
        rendered sensor image with MuJoCo/OpenGL.  Perception topics keep the
        3DGS images.
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

        display_window = self.window
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
                            img_depth,
                            alpha=255.0 / self.config.max_render_depth),
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
        except Exception as exc:  # noqa: BLE001
            # Sensor publication must continue even if the display fails.
            print(f"[server-render] display render error: {exc}")

    server.build_config = build_config
    server.TaskMMK2ROS2.__init__ = init_with_render_display
    server.TaskMMK2ROS2.render = render_with_nonblocking_display


def main() -> None:
    server = load_official_server()
    install_render_overrides(server)
    server.main()


if __name__ == "__main__":
    main()
