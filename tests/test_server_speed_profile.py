"""Regression checks for the opt-in organizer Server speed wrapper."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).parents[1]
WRAPPER = (
    ROOT / "examples/supermarket_sorting/supermarket_sorting_server_speed.py")


def _module():
    spec = importlib.util.spec_from_file_location("server_speed", WRAPPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_head_only_camera_profile_is_selectable():
    module = _module()
    previous = os.environ.get("SUPERMARKET_RGB_CAMERAS")
    try:
        os.environ["SUPERMARKET_RGB_CAMERAS"] = "head"
        assert module.sensor_camera_ids() == [0]
    finally:
        if previous is None:
            os.environ.pop("SUPERMARKET_RGB_CAMERAS", None)
        else:
            os.environ["SUPERMARKET_RGB_CAMERAS"] = previous


def test_launchers_enable_split_linear_only_speed_profile():
    gui = (ROOT / "gui_competition_runner.py").read_text(encoding="utf-8")
    headless = (
        ROOT / "scripts/run_seed4_headless.py").read_text(encoding="utf-8")
    for source in (gui, headless):
        assert "SUPERMARKET_RGB_CAMERAS=head" in source
        assert "SUPERMARKET_RENDER_FPS=12" in source
        assert "SUPERMARKET_WHEEL_LINEAR_ERROR_LIMIT_RADPS=4.5" in source
        assert "SUPERMARKET_WHEEL_ANGULAR_ERROR_LIMIT_RADPS=2.5" in source
        assert "python3 /tmp/supermarket_sorting_server_speed.py" in source


def test_wrapper_keeps_angular_and_linear_error_limits_separate():
    source = WRAPPER.read_text(encoding="utf-8")
    assert "linear_error =" in source
    assert "angular_error =" in source
    assert "limited_linear - limited_angular" in source
    assert "limited_linear + limited_angular" in source


def test_windowed_camera_switch_does_not_request_a_second_gs_render():
    source = WRAPPER.read_text(encoding="utf-8")

    assert "original_render = server.TaskMMK2ROS2.render" in source
    assert "display_window = self.window" in source
    assert "self.window = None" in source
    assert "self.img_rgb_obs_s.get(self.cam_id)" in source
    assert "server.TaskMMK2ROS2.render = render_with_nonblocking_display" in source
