"""Regression checks for wall-safe sphere delivery approach geometry."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/integrated_nav_pick_place.py"
)
DELIVERY_APPROACH = (-1.80, -2.60, -math.pi / 2.0)


def _policy_class():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "IntegratedNavPickPlace")
    methods = [
        node for node in controller.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_target_is_sphere_product",
            "_delivery_slot_goal",
        }
    ]
    policy = ast.ClassDef(
        name="DeliveryGoalPolicy",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[policy], type_ignores=[]))
    namespace = {
        "DELIVERY_APPROACH": DELIVERY_APPROACH,
        "pick": SimpleNamespace(
            SPHERE_RADIUS_M={"pingguo": 0.035, "chengzi": 0.037}),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["DeliveryGoalPolicy"]


DeliveryGoalPolicy = _policy_class()


class Harness(DeliveryGoalPolicy):
    def __init__(self, *, kind: str, dedicated_sphere_grasp: bool):
        self.target_kind = kind
        self.use_sphere_grasp = dedicated_sphere_grasp
        self.place_world = np.array([-2.20, -3.52, 0.85])


def test_sphere_uses_table_centre_base_goal_instead_of_west_slot_x():
    goal = Harness(
        kind="pingguo", dedicated_sphere_grasp=True)._delivery_slot_goal()

    assert goal == DELIVERY_APPROACH


def test_lower_shelf_sphere_still_uses_table_centre_base_goal():
    goal = Harness(
        kind="pingguo", dedicated_sphere_grasp=False)._delivery_slot_goal()

    assert goal == DELIVERY_APPROACH


def test_non_sphere_keeps_assigned_slot_base_x():
    goal = Harness(
        kind="sanmingzhi", dedicated_sphere_grasp=False)._delivery_slot_goal()

    assert goal == (-2.20, DELIVERY_APPROACH[1], DELIVERY_APPROACH[2])


def test_transport_hold_and_drop_monitor_use_product_geometry():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "IntegratedNavPickPlace")
    methods = {
        node.name: node for node in controller.body
        if isinstance(node, ast.FunctionDef)
    }

    for name in ("_capture_transport_grip_command",
                 "_transport_drop_signature"):
        calls = {
            call.func.attr for call in ast.walk(methods[name])
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
        }
        assert "_target_is_sphere_product" in calls
