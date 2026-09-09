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
    "SANMINGZHI_INWARD_X_OFFSET_BY_ARM_M",
    "GENERIC_CAPTURE_MAX_GRIP_BY_KIND",
    "GENERIC_DIRECT_FORWARD_SPEED_MPS",
    "GENERIC_DIRECT_FORWARD_SPEED_BY_KIND_MPS",
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
    assert math.isclose(POLICY["grip_preshape_for_kind"]("sanmingzhi"), 0.950)
    assert math.isclose(POLICY["grip_preshape_for_kind"]("shupian"), 0.975)


def test_sandwich_uses_the_validated_generic_arm_offset():
    offset = POLICY["grasp_tcp_x_offset"]

    assert math.isclose(offset("kele", "r"), 0.003)
    assert math.isclose(offset("sanmingzhi", "r"), 0.003)
    assert math.isclose(offset("sanmingzhi", "l"), 0.000)


def test_sandwich_keeps_the_generic_capture_policy():
    assert "sanmingzhi" not in POLICY["GENERIC_CAPTURE_MAX_GRIP_BY_KIND"]


def test_sandwich_uses_pre_speed_approach_without_slowing_spheres():
    speed = POLICY["generic_direct_forward_speed_mps"]
    assert math.isclose(speed("sanmingzhi"), 0.036)
    assert math.isclose(speed("pingguo"), 0.090)
