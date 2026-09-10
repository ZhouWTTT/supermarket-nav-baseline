"""Placement orders arm/slide motion according to the slide direction."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/integrated_nav_pick_place.py"
)


def _place_tick() -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "IntegratedNavPickPlace")
    return next(
        node for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name == "_place_tick")


def test_descent_keeps_transport_height_until_loaded_arm_target_is_sent():
    method = _place_tick()
    slide_assignment = next(
        node for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "des_slide"
            for target in node.targets)
        and "place_slide_cmd" in ast.unparse(node.value))
    arm_target_call = next(
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_selected_arm_target"
        and node.args
        and ast.unparse(node.args[0]) == "self.place_arm_joints")

    assert arm_target_call.lineno < slide_assignment.lineno


def test_only_slide_lift_branch_gates_arm_by_measured_planning_height():
    method = _place_tick()
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(method):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    arm_target_call = next(
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_selected_arm_target"
        and node.args
        and ast.unparse(node.args[0]) == "self.place_arm_joints")
    guards = []
    parent = parents[arm_target_call]
    while parent is not method:
        if isinstance(parent, ast.If):
            guards.append(ast.unparse(parent.test))
        parent = parents[parent]

    assert any(
        "slide_error <= PLACE_SAFE_IK_SLIDE_TOLERANCE_M" in guard
        for guard in guards)


def test_safe_slide_gate_allows_only_small_loaded_tracking_bias():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "PLACE_SAFE_IK_SLIDE_TOLERANCE_M"
            for target in node.targets))

    assert ast.literal_eval(assignment.value) <= 0.008


def test_arm_target_gate_returns_before_any_descent_can_run():
    method_source = ast.get_source_segment(
        SOURCE.read_text(encoding="utf-8"), _place_tick())

    assert "direction-aware loaded approach enabled" in method_source
    assert "safe IK slide height reached" in method_source
    assert "and not self._place_arm_target_sent" in method_source
    assert "loaded arm reached horizontal target at" in method_source
    assert "descending slide vertically" in method_source
