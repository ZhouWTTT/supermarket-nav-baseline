"""Regression tests for the empty-only linear cruise speedup."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
NAV_SOURCE = ROOT / "examples/supermarket_sorting/supermarket_navigation.py"
PICK_SOURCE = ROOT / "examples/supermarket_sorting/yolo_aruco_shelf_pick.py"
FLOW_SOURCE = ROOT / "examples/supermarket_sorting/integrated_nav_pick_place.py"


def _module_literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name
                for target in node.targets))
    return ast.literal_eval(assignment.value)


def _dict_entry(path: Path, name: str, key: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name
                for target in node.targets))
    assert isinstance(assignment.value, ast.Dict)
    for item_key, item_value in zip(
            assignment.value.keys, assignment.value.values):
        if ast.literal_eval(item_key) == key:
            return ast.literal_eval(item_value)
    raise AssertionError(f"missing {key!r} in {name}")


def _navigation_init_assignments() -> dict[str, object]:
    tree = ast.parse(NAV_SOURCE.read_text(encoding="utf-8"))
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "NavigationController")
    constructor = next(
        node for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    values = {}
    for node in constructor.body:
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.value, ast.Constant)):
            values[node.targets[0].attr] = node.value.value
    return values


def test_only_clear_path_cruise_ceiling_is_raised():
    values = _navigation_init_assignments()

    assert values["max_lin"] == 1.15
    assert values["near_goal_max_lin"] == 0.40
    assert values["max_lin_acc"] == 1.2
    assert values["_stop_dist"] == 0.32
    assert values["_slow_dist"] == 0.50


def test_angular_policy_is_unchanged():
    values = _navigation_init_assignments()

    assert values["max_ang"] == 2.0
    assert values["max_ang_acc"] == 5.0
    assert _module_literal(PICK_SOURCE, "NAV_ANGULAR_MAX_RADPS") == 2.0
    assert (_module_literal(PICK_SOURCE, "NAV_TRANSLATE_ANGULAR_MAX_RADPS")
            == 1.70)


def test_direct_motion_uses_the_same_linear_ceiling():
    assert _module_literal(PICK_SOURCE, "NAV_LINEAR_MAX_MPS") == 1.15

    tree = ast.parse(PICK_SOURCE.read_text(encoding="utf-8"))
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ShelfPickController")
    method = next(
        node for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name == "set_twist")
    source = ast.unparse(method)
    assert source.count("NAV_LINEAR_MAX_MPS") == 2


def test_every_loaded_product_keeps_the_previous_linear_ceiling():
    assert (_module_literal(FLOW_SOURCE, "LOADED_TRANSPORT_LINEAR_MAX_MPS")
            == 0.90)
    assert _dict_entry(
        FLOW_SOURCE, "LOADED_TRANSPORT_LIMITS", "kouxiangtang")[0] == 0.75
    assert _dict_entry(
        FLOW_SOURCE, "LOADED_TRANSPORT_LIMITS", "chengzi")[0] == 0.80
    assert _dict_entry(
        FLOW_SOURCE, "LOADED_TRANSPORT_LIMITS", "pingguo")[0] == 0.80
