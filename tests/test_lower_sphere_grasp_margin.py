"""Regression checks for lower-shelf sphere seating depth."""

from __future__ import annotations

import ast
import math
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/yolo_aruco_shelf_pick.py"
)
CONSTANTS = {
    "SPHERE_RADIUS_M",
    "LOWER_GRASP_TCP_FORWARD_M",
    "LOWER_SPHERE_GRASP_TCP_FORWARD_M",
    "GENERIC_DIRECT_FORWARD_SPEED_MPS",
}


def _lower_depth_policy():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in CONSTANTS
            for target in node.targets)
    ]
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "lower_grasp_tcp_forward_m")
    module = ast.fix_missing_locations(
        ast.Module(body=[*assignments, function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


POLICY = _lower_depth_policy()
lower_grasp_tcp_forward_m = POLICY["lower_grasp_tcp_forward_m"]


def test_lower_apples_and_oranges_get_fifteen_mm_more_seating_depth():
    generic_depth = lower_grasp_tcp_forward_m("kele")

    assert math.isclose(generic_depth, 0.035)
    assert math.isclose(
        lower_grasp_tcp_forward_m("pingguo") - generic_depth, 0.015)
    assert math.isclose(
        lower_grasp_tcp_forward_m("chengzi") - generic_depth, 0.015)


def test_lower_sphere_margin_does_not_reduce_forward_speed():
    assert math.isclose(POLICY["GENERIC_DIRECT_FORWARD_SPEED_MPS"], 0.090)
