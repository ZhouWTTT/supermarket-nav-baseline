"""Structural regressions for memory direct-stop and target-kind switching.

The production controllers import ROS/MuJoCo modules that are not present in
the lightweight host test environment, so these checks inspect their parsed
AST and verify the required state-machine handoffs without runtime mocks.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).parents[1]
PICK_SOURCE = ROOT / "examples/supermarket_sorting/yolo_aruco_shelf_pick.py"
INTEGRATED_SOURCE = (
    ROOT / "examples/supermarket_sorting/integrated_nav_pick_place.py")
RUNNER_SOURCE = ROOT / "examples/supermarket_sorting/competition_runner.py"


def _method(source: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(
        node for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _called_attributes(node: ast.AST) -> set[str]:
    return {
        call.func.attr
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }


def test_reliable_memory_direct_is_enabled_by_default():
    tree = ast.parse(RUNNER_SOURCE.read_text(encoding="utf-8"))
    parse_args = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    defaults = [
        keyword
        for call in ast.walk(parse_args)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "set_defaults"
        for keyword in call.keywords
    ]
    dynamic = next(item for item in defaults if item.arg == "dynamic_direct")
    assert isinstance(dynamic.value, ast.Constant)
    assert dynamic.value.value is True


def test_direct_arrival_hands_off_to_recheck_or_grasp_without_second_align():
    method = _method(
        INTEGRATED_SOURCE, "IntegratedNavPickPlace", "advance_direct_transit")
    calls = _called_attributes(method)

    assert "_start_grasp_settle" in calls
    assert "_start_close_recheck" in calls
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "STATE_ALIGN"
        for node in ast.walk(method)
    )


def test_coarse_reroute_can_still_promote_any_pending_kind_to_direct():
    tick = _method(
        INTEGRATED_SOURCE, "IntegratedNavPickPlace", "_memory_route_tick")
    promotion_call = next(
        node for node in ast.walk(tick)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_select_live_order_direct_hint")
    coarse_reroute_gate = next(
        node for node in ast.walk(tick)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "memory_rerouted")

    # Promotion must be checked before the one-shot coarse shelf-reroute gate.
    assert promotion_call.lineno < coarse_reroute_gate.lineno

    selector = _method(
        INTEGRATED_SOURCE,
        "IntegratedNavPickPlace",
        "_select_live_order_direct_hint")
    assert any(
        isinstance(node, ast.Attribute)
        and node.attr == "opportunistic_target_kinds"
        for node in ast.walk(selector)
    )
    assert any(
        isinstance(node, ast.keyword)
        and node.arg == "require_direct"
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        for node in ast.walk(selector)
    )


def test_direct_transit_keeps_ranking_live_order_candidates_until_arrival():
    tick = _method(
        INTEGRATED_SOURCE, "IntegratedNavPickPlace", "_memory_route_tick")
    calls = _called_attributes(tick)

    assert "_retarget_direct_order_from_memory" in calls
    assert any(
        isinstance(node, ast.Attribute)
        and node.attr == "STATE_DIRECT_TRANSIT"
        for node in ast.walk(tick)
    )

    arrival = _method(
        INTEGRATED_SOURCE, "IntegratedNavPickPlace", "advance_direct_transit")
    locks = [
        node for node in ast.walk(arrival)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute)
                and target.attr == "opportunistic_target_locked"
                for target in node.targets)
    ]
    assert any(
        isinstance(node.value, ast.Constant) and node.value.value is True
        for node in locks)


def _retarget_policy_class():
    tree = ast.parse(INTEGRATED_SOURCE.read_text(encoding="utf-8"))
    constants = {
        "DYNAMIC_DIRECT_RETARGET_MARGIN_M",
        "DYNAMIC_DIRECT_RETARGET_MIN_HOLD_S",
    }
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in constants
                for target in node.targets)
    ]
    method = _method(
        INTEGRATED_SOURCE,
        "IntegratedNavPickPlace",
        "_direct_retarget_is_better")
    policy = ast.ClassDef(
        name="RetargetPolicy",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[*assignments, policy], type_ignores=[]))
    namespace = {
        "np": np,
        "pick": SimpleNamespace(STATE_DIRECT_TRANSIT="direct_transit"),
        "time": SimpleNamespace(monotonic=lambda: 10.0),
    }
    exec(compile(module, str(INTEGRATED_SOURCE), "exec"), namespace)
    return namespace["RetargetPolicy"]


def test_new_reliable_e_shelf_order_replaces_farther_c_shelf_leg():
    policy = _retarget_policy_class()()
    policy.state = "direct_transit"
    policy.target_kind = "chengzi"
    policy.direct_transit_slot = ("C", "L1", "2")
    policy.direct_transit_started_at = 8.0
    policy.base_xy = np.array([1.52, 0.0])
    policy.align_base_x = -0.07
    policy.align_base_y = 2.475
    e_shelf_chips = {
        "kind": "shupian",
        "shelf": "E",
        "level": "L2",
        "column": "1",
        "target_xy": (1.43, 2.475),
    }

    assert policy._direct_retarget_is_better(e_shelf_chips)
    assert (e_shelf_chips["retarget_candidate_distance"]
            < e_shelf_chips["retarget_current_distance"])


def test_direct_retarget_hysteresis_rejects_same_or_barely_closer_slot():
    policy = _retarget_policy_class()()
    policy.state = "direct_transit"
    policy.target_kind = "chengzi"
    policy.direct_transit_slot = ("C", "L1", "2")
    policy.direct_transit_started_at = 8.0
    policy.base_xy = np.array([0.0, 0.0])
    policy.align_base_x = 0.0
    policy.align_base_y = 2.0

    same = {
        "kind": "chengzi", "shelf": "C", "level": "L1",
        "column": "2", "target_xy": (0.0, 2.0),
    }
    barely_closer = {
        "kind": "shupian", "shelf": "C", "level": "L2",
        "column": "1", "target_xy": (0.0, 1.95),
    }

    assert not policy._direct_retarget_is_better(same)
    assert not policy._direct_retarget_is_better(barely_closer)


def test_kind_switch_rebuilds_grasp_geometry_and_delivery_hook():
    switch = _method(
        PICK_SOURCE, "ShelfPickController", "_switch_recheck_target")
    calls = _called_attributes(switch)

    assert "_apply_target_kind" in calls
    assert "_commit_localised_target" in calls
    assert "set_state" in calls  # realign if the new kind needs another pose

    hook = _method(
        INTEGRATED_SOURCE, "IntegratedNavPickPlace", "_on_target_kind_changed")
    assigned_attributes = {
        node.value.attr
        for node in ast.walk(hook)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
    }
    assert "place_world" in assigned_attributes
