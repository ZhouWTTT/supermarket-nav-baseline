"""Regression tests for feedback-driven tissue clamp dwell."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/yolo_aruco_shelf_pick.py"
)
CONSTANTS = {
    "ARM_REACHED_TOLERANCE_RAD",
    "DUAL_TISSUE_GRIP_CONTACT_MAX",
    "DUAL_TISSUE_CLAMP_MIN_DWELL_S",
    "DUAL_TISSUE_CLAMP_DWELL_S",
}


def _release_gate():
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
        and node.name == "dual_tissue_clamp_release_gate")
    module = ast.fix_missing_locations(
        ast.Module(body=[*assignments, function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["dual_tissue_clamp_release_gate"]


release_gate = _release_gate()


def test_stable_clamp_still_gets_one_second_of_preload():
    assert release_gate(0.99, 0.010, 0.0, 0.0) is None
    assert release_gate(1.00, 0.010, 0.0, 0.0) == "stable-feedback"


def test_unsettled_arm_or_open_gripper_cannot_release_early():
    assert release_gate(2.0, 0.050, 0.0, 0.0) is None
    assert release_gate(2.0, 0.010, 0.10, 0.0) is None
    assert release_gate(2.0, 0.010, None, 0.0) is None


def test_previous_four_second_dwell_remains_the_hard_fallback():
    assert release_gate(4.0, 1.0, None, None) == "fallback-timeout"
