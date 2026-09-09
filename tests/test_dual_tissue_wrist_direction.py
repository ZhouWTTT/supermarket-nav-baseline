"""Regression checks for the operator-visible tissue wrist direction."""

from __future__ import annotations

import ast
import math
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/yolo_aruco_shelf_pick.py"
)


def _roll_pair():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    constant = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name)
                and target.id == "DUAL_TISSUE_TOP_WRIST_ROLL_RAD"
                for target in node.targets))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "dual_tissue_wrist_roll_pair")
    module = ast.fix_missing_locations(
        ast.Module(body=[constant, function], type_ignores=[]))
    namespace = {"math": math}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["dual_tissue_wrist_roll_pair"]


roll_pair = _roll_pair()


def test_outward_roll_is_left_minus_90_right_plus_90():
    left, right = roll_pair(True)
    assert math.isclose(left, -math.pi / 2.0)
    assert math.isclose(right, math.pi / 2.0)


def test_inward_roll_has_the_opposite_signs():
    left, right = roll_pair(False)
    assert math.isclose(left, math.pi / 2.0)
    assert math.isclose(right, -math.pi / 2.0)


def test_tissue_grasp_selects_operator_outward_roll():
    source = SOURCE.read_text(encoding="utf-8")
    assert "self.dual_top_wrist_inward = True" in source
    assert "roll_direction={'outward'" in source
