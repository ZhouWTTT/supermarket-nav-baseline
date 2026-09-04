"""Focused regression tests for the loaded single-arm placement escape gate.

The integrated controller imports ROS and simulator dependencies unavailable
in the lightweight host test environment.  Extracting the production method
and its constants through the AST tests the exact policy without mocking the
rest of that runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/integrated_nav_pick_place.py"
)
CONSTANTS = {
    "PLACE_ARM_SETTLE_TOLERANCE_RAD",
    "PLACE_XY_COMMAND_MIN_WAIT_S",
    "PLACE_XY_STATIONARY_SLIDE_ERROR_M",
    "PLACE_XY_MEASURED_TCP_TOLERANCE_M",
    "PLACE_XY_MEASURED_TCP_STABILITY_RADIUS_M",
    "PLACE_XY_MEASURED_TCP_SETTLE_S",
}


def _policy_class():
    tree = ast.parse(SOURCE.read_text())
    assignments = [
        node for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id in CONSTANTS
            for target in (
                node.targets if isinstance(node, ast.Assign)
                else [node.target]
            )
        )
    ]
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "IntegratedNavPickPlace")
    method = next(
        node for node in controller.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_place_refine_measured_tcp_ready")
    policy = ast.ClassDef(
        name="MeasuredTcpPolicy",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[*assignments, policy], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["MeasuredTcpPolicy"]


MeasuredTcpPolicy = _policy_class()


class Harness(MeasuredTcpPolicy):
    def __init__(self):
        self.joints = {"slide_joint": 0.4695}
        self.des_slide = 0.4635
        self._place_refine_target_sent_at = 0.0
        self._place_refine_tcp_stable_since = None
        self._place_refine_tcp_anchor_xy = None
        self.arm_error = 0.100
        self.over_table = True

    def selected_arm_error(self):
        return self.arm_error

    def _tcp_over_delivery_table(self, _tcp):
        return self.over_table


def test_stable_measured_tcp_bypasses_oscillating_loaded_joint():
    policy = Harness()
    tcp = np.array([-1.7855, -3.3770, 0.902])
    error = np.array([-1.81, -3.37]) - tcp[:2]

    assert not policy._place_refine_measured_tcp_ready(0.20, tcp, error)
    assert not policy._place_refine_measured_tcp_ready(0.74, tcp, error)
    assert policy._place_refine_measured_tcp_ready(1.01, tcp, error)


def test_tcp_motion_resets_the_sustained_stability_timer():
    policy = Harness()
    tcp = np.array([-1.7855, -3.3770, 0.902])
    error = np.array([-1.81, -3.37]) - tcp[:2]
    moved = tcp + np.array([0.0041, 0.0, 0.0])
    moved_error = np.array([-1.81, -3.37]) - moved[:2]

    assert not policy._place_refine_measured_tcp_ready(0.20, tcp, error)
    assert not policy._place_refine_measured_tcp_ready(
        0.90, moved, moved_error)
    assert not policy._place_refine_measured_tcp_ready(
        1.69, moved, moved_error)
    assert policy._place_refine_measured_tcp_ready(
        1.71, moved, moved_error)


def test_escape_gate_preserves_each_safety_boundary():
    tcp = np.array([-1.7855, -3.3770, 0.902])
    good_error = np.array([-1.81, -3.37]) - tcp[:2]

    cases = []
    outside_slot = Harness()
    cases.append((outside_slot, np.array([0.031, 0.0])))
    slide_unsettled = Harness()
    slide_unsettled.joints["slide_joint"] = 0.4760
    cases.append((slide_unsettled, good_error))
    joint_converged = Harness()
    joint_converged.arm_error = 0.020
    cases.append((joint_converged, good_error))
    off_table = Harness()
    off_table.over_table = False
    cases.append((off_table, good_error))
    no_refine_command = Harness()
    no_refine_command._place_refine_target_sent_at = None
    cases.append((no_refine_command, good_error))

    for policy, error in cases:
        assert not policy._place_refine_measured_tcp_ready(0.20, tcp, error)
        assert not policy._place_refine_measured_tcp_ready(2.00, tcp, error)
