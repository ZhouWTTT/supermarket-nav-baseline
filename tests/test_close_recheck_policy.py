"""Regression tests for fast, guarded close-range verification."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
PICK_SOURCE = ROOT / "examples/supermarket_sorting/yolo_aruco_shelf_pick.py"


def _tree() -> ast.Module:
    return ast.parse(PICK_SOURCE.read_text(encoding="utf-8"))


def _literal(name: str):
    for node in _tree().body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant: {name}")


def _method(name: str) -> ast.FunctionDef:
    controller = next(
        node for node in _tree().body
        if isinstance(node, ast.ClassDef)
        and node.name == "ShelfPickController")
    return next(
        node for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name == name)


def test_close_recheck_uses_one_fast_confirmation():
    assert _literal("CLOSE_RECHECK_CONFIRMATIONS") == 1
    assert _literal("CLOSE_RECHECK_SWITCH_CONFIRMATIONS") == 3
    assert _literal("CLOSE_RECHECK_POSE_TIMEOUT_S") == 1.0
    assert _literal("CLOSE_RECHECK_CAMERA_MOTION_TIMEOUT_S") == 4.0
    assert _literal("CLOSE_RECHECK_ARUCO_PREFERENCE_S") == 0.25


def test_close_recheck_wrong_class_switch_has_its_own_stricter_gate():
    method = _method("_maybe_switch_recheck_target")
    switch_gate = next(
        node for node in ast.walk(method)
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.Name)
                and child.id == "CLOSE_RECHECK_SWITCH_CONFIRMATIONS"
                for child in ast.walk(node.test)))

    assert switch_gate.body


def test_close_recheck_geometry_cannot_accept_the_wrong_class():
    method = _method("_recheck_detection_matches")
    class_guard = next(
        node for node in ast.walk(method)
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.Constant) and child.value == "class"
                for child in ast.walk(node)))
    matched_assignment = next(
        node for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "matched"
                for target in node.targets))

    assert class_guard.lineno < matched_assignment.lineno


def test_direct_slot_failure_does_not_blindly_scan_adjacent_columns():
    constructor = _method("__init__")
    assignment = next(
        node for node in ast.walk(constructor)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute)
                and target.attr == "direct_slot_adjacent_max_retries"
                for target in node.targets))

    assert isinstance(assignment.value, ast.Constant)
    assert assignment.value.value == 0


def test_close_recheck_receives_all_classes_for_dynamic_switching():
    method = _method("yolo_cb")
    recheck_branch = next(
        node for node in ast.walk(method)
        if isinstance(node, ast.IfExp)
        and any(isinstance(child, ast.Name)
                and child.id == "STATE_RECHECK"
                for child in ast.walk(node.test)))

    assert isinstance(recheck_branch.body, ast.Name)
    assert recheck_branch.body.id == "head_records"
    assert isinstance(recheck_branch.orelse, ast.Name)
    assert recheck_branch.orelse.id == "target_records"


def test_aruco_candidate_only_blocks_inside_the_preference_window():
    method = _method("_recheck_detection_matches")
    preference_gate = next(
        node for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "aruco_preference_active"
        and any(isinstance(child, ast.Constant)
                and child.value == "aruco(candidate="
                for child in ast.walk(node)))

    assert preference_gate.body


def test_recheck_observation_timeout_starts_after_camera_ready():
    tick = _method("tick")
    timeout_uses_ready_since = any(
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and isinstance(node.right, ast.Attribute)
        and node.right.attr == "scan_camera_ready_since"
        and any(isinstance(child, ast.Name)
                and child.id == "CLOSE_RECHECK_POSE_TIMEOUT_S"
                for child in ast.walk(parent))
        for parent in ast.walk(tick)
        for node in ast.walk(parent)
    )

    assert timeout_uses_ready_since
