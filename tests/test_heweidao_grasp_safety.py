"""Heweidao grasp must keep the proven slower motion profile.

Heweidao is held by the open-limit jaws against its tapered body; after the
generic grasp speeds were raised, a transport drop appeared.  Its grasp
parameters stay on the old validated profile while other products keep the
faster generic values.
"""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/yolo_aruco_shelf_pick.py"
)


def _dict(name: str) -> dict:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
            assert isinstance(node.value, ast.Dict)
            return {
                ast.literal_eval(key): ast.literal_eval(value)
                for key, value in zip(node.value.keys, node.value.values)
            }
    raise AssertionError(f"missing dict: {name}")


def _literal(name: str):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant: {name}")


def test_heweidao_keeps_slower_forward_and_seating_profile():
    speeds = _dict("GENERIC_DIRECT_FORWARD_SPEED_BY_KIND_MPS")
    min_durations = _dict(
        "GENERIC_DIRECT_FORWARD_MIN_DURATION_BY_KIND_S")

    assert speeds["heweidao"] == 0.050
    assert min_durations["heweidao"] == 2.5
    # 通用档仍保持提速后的值，只有 heweidao 走旧档。
    assert _literal("GENERIC_DIRECT_FORWARD_SPEED_MPS") > speeds["heweidao"]


def test_heweidao_keeps_slower_close_and_retreat():
    close_steps = _dict("GRIP_CLOSE_MAX_STEP_BY_KIND")
    retreat_steps = _dict("RETREAT_ARM_MAX_STEP_BY_KIND_RAD")

    assert close_steps["heweidao"] == 0.006
    assert retreat_steps["heweidao"] == 0.010
    assert close_steps["heweidao"] < _literal("GRIP_CLOSE_MAX_STEP")
    assert retreat_steps["heweidao"] < _literal(
        "RETREAT_ARM_MAX_STEP_RAD")
