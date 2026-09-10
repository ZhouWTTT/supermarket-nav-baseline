"""Regression checks for centred, fully opened sandwich grasps."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/yolo_aruco_shelf_pick.py"
)
CONSTANTS = {
    "PRODUCT_GRASP_WIDTH_M",
    "GRIPPER_MAX_OPENING_M",
    "GRIP_PRESHAPE_CLEARANCE_M",
    "GRIP_PRESHAPE_CLEARANCE_BY_KIND_M",
    "GRIP_CLOSE",
    "GENERIC_EMPTY_GRIP_MARGIN",
    "GRIP_OPEN",
    "GRASP_TCP_X_OFFSET_BY_ARM",
    "SANMINGZHI_GRASP_X_SHIFT_BY_ARM_M",
    "GENERIC_CAPTURE_MAX_GRIP_BY_KIND",
    "GENERIC_DIRECT_FORWARD_SPEED_MPS",
    "GENERIC_DIRECT_FORWARD_SPEED_BY_KIND_MPS",
    "GRIP_CLOSE_MAX_STEP",
    "GRIP_CLOSE_MAX_STEP_BY_KIND",
    "GENERIC_POST_EXTEND_Z_DROP_M_BY_KIND",
    "SANMINGZHI_LATERAL_RECHECK_MIN_SAMPLES",
    "SANMINGZHI_LATERAL_RECHECK_SPREAD_MAX_M",
    "SANMINGZHI_LATERAL_RECHECK_DEADBAND_M",
    "SANMINGZHI_LATERAL_RECHECK_MAX_ADJUST_M",
    "SANMINGZHI_LATERAL_RECHECK_MAX_WAIT_S",
}


def _policy():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in CONSTANTS
            for target in node.targets)
    ]
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "grip_preshape_for_kind", "grasp_tcp_x_offset",
            "generic_direct_forward_speed_mps",
        }
    ]
    module = ast.fix_missing_locations(
        ast.Module(body=[*assignments, *functions], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


POLICY = _policy()


def test_sandwich_preshape_keeps_side_margin_while_engaging_body():
    assert math.isclose(POLICY["grip_preshape_for_kind"]("sanmingzhi"), 0.975)
    assert math.isclose(POLICY["grip_preshape_for_kind"]("shupian"), 0.975)


def test_sandwich_uses_the_validated_generic_arm_offset():
    offset = POLICY["grasp_tcp_x_offset"]

    assert math.isclose(offset("kele", "r"), 0.003)
    # 三明治左右傻瓜式补偿：左臂为负（向左/西）、右臂为正（向右/东）；
    # 具体毫米数允许直接调参，这里只校验方向与量级。
    assert offset("sanmingzhi", "l") < 0.0
    assert offset("sanmingzhi", "r") > 0.0
    assert abs(offset("sanmingzhi", "l")) <= 0.020
    assert abs(offset("sanmingzhi", "r")) <= 0.020


def test_sandwich_x_shift_parameter_is_outward_per_arm():
    shifts = POLICY["SANMINGZHI_GRASP_X_SHIFT_BY_ARM_M"]
    assert shifts["l"] < 0.0
    assert shifts["r"] > 0.0
    assert abs(shifts["l"]) <= 0.020
    assert abs(shifts["r"]) <= 0.020


def test_sandwich_rejects_loose_edge_holds_at_the_shelf():
    # 实测开度 >= 0.95 说明爪子只压住楔形盒的斜棱，带离货架必然中途滑脱；
    # 在货架侧拦截并让 runner 换目标重试。0.91~0.94 的历史样本可完成运输。
    shallow = POLICY["GENERIC_CAPTURE_MAX_GRIP_BY_KIND"]
    assert shallow["sanmingzhi"] == 0.95
    assert 0.0 < shallow["sanmingzhi"] < 1.0


def test_sandwich_uses_pre_speed_approach_without_slowing_spheres():
    speed = POLICY["generic_direct_forward_speed_mps"]
    assert math.isclose(speed("sanmingzhi"), 0.036)
    assert math.isclose(speed("pingguo"), 0.090)


def test_sandwich_keeps_the_old_gentle_close_and_level_extension():
    close_steps = POLICY["GRIP_CLOSE_MAX_STEP_BY_KIND"]
    assert close_steps["sanmingzhi"] == 0.006
    assert close_steps["sanmingzhi"] < POLICY["GRIP_CLOSE_MAX_STEP"]
    # 旧版三明治前伸保持水平，不下沉（下沉是后来的修复尝试，实测未改善）。
    assert "sanmingzhi" not in POLICY["GENERIC_POST_EXTEND_Z_DROP_M_BY_KIND"]


def test_sandwich_lateral_recheck_is_bounded_and_conservative():
    # 抓取前用 close-recheck 的深度世界 X 修正左右位置：要求多个一致采样，
    # 死区避免抖动，修正量封顶以免深度噪声把爪子带到盒外。
    assert POLICY["SANMINGZHI_LATERAL_RECHECK_MIN_SAMPLES"] >= 3
    assert 0.005 <= POLICY["SANMINGZHI_LATERAL_RECHECK_SPREAD_MAX_M"] <= 0.03
    # 左右对准是三明治夹持成败的关键，死区收紧到 2.5mm。
    assert POLICY["SANMINGZHI_LATERAL_RECHECK_DEADBAND_M"] == 0.0025
    assert 0.01 <= POLICY["SANMINGZHI_LATERAL_RECHECK_MAX_ADJUST_M"] <= 0.05
    assert 0.0 < POLICY["SANMINGZHI_LATERAL_RECHECK_MAX_WAIT_S"] <= 2.0
