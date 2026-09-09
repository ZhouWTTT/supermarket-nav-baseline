"""Regression checks for centred sphere grasp height."""

from __future__ import annotations

import ast
import math
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/yolo_aruco_shelf_pick.py"
)
CONSTANTS = {
    "SPHERE_GRASP_TCP_Z_RAISE_M",
    "TOP_SHELF_SURFACE_Z_M",
    "TOP_SPHERE_MIN_TCP_TARGET_CLEARANCE_M",
}


def _height_policy():
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
        and node.name == "sphere_grasp_tcp_z")
    module = ast.fix_missing_locations(
        ast.Module(body=[*assignments, function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["sphere_grasp_tcp_z"]


sphere_grasp_tcp_z = _height_policy()


def test_middle_sphere_targets_five_millimetres_below_centre():
    assert math.isclose(sphere_grasp_tcp_z(0.903, "middle"), 0.898)


def test_top_sphere_uses_lower_dedicated_shelf_clearance():
    assert math.isclose(sphere_grasp_tcp_z(1.241, "top"), 1.249)


def test_top_clearance_never_lowers_a_naturally_high_sphere():
    assert math.isclose(sphere_grasp_tcp_z(1.270, "top"), 1.265)
