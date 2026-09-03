"""Regression tests for feedback-gated placement speedups."""

from __future__ import annotations

import ast
import math
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/integrated_nav_pick_place.py"
)
RELEASE_CONSTANTS = {
    "HEWEIDAO_RELEASE_OPEN_MIN_S",
    "PLACE_RELEASE_MIN_DWELL_S",
    "PLACE_RELEASE_GRIP_OPEN_MIN",
    "PLACE_RELEASE_GRIP_OPEN_STABLE_S",
}


def _tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _literal(name):
    for node in _tree().body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant: {name}")


def _release_policy_class():
    tree = _tree()
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in RELEASE_CONSTANTS
            for target in node.targets)
    ]
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "IntegratedNavPickPlace")
    method = next(
        node for node in controller.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_single_place_release_ready")
    policy = ast.ClassDef(
        name="ReleasePolicy",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[*assignments, policy], type_ignores=[]))
    namespace = {"math": math}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["ReleasePolicy"]


ReleasePolicy = _release_policy_class()


class _Logger:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass


class Harness(ReleasePolicy):
    def __init__(self, *, kind="sanmingzhi", grip=1.0):
        self.target_kind = kind
        self.grip = grip
        self._place_release_started_at = 0.0
        self._place_grip_open_since = None
        self.place_release_dwell_s = 2.0
        self.logger = _Logger()

    def selected_gripper_position(self):
        return self.grip

    def get_logger(self):
        return self.logger


def test_feedback_ends_normal_release_after_short_safe_minimum():
    policy = Harness()

    assert not policy._single_place_release_ready(0.05)
    assert not policy._single_place_release_ready(0.39)
    assert policy._single_place_release_ready(0.40)


def test_missing_feedback_keeps_two_second_fallback():
    policy = Harness(grip=None)

    assert not policy._single_place_release_ready(1.99)
    assert policy._single_place_release_ready(2.00)


def test_heweidao_keeps_longer_release_minimum():
    policy = Harness(kind="heweidao")

    assert not policy._single_place_release_ready(0.05)
    assert not policy._single_place_release_ready(0.99)
    assert policy._single_place_release_ready(1.00)


def test_open_confirmation_must_be_continuous():
    policy = Harness()

    assert not policy._single_place_release_ready(0.10)
    policy.grip = 0.90
    assert not policy._single_place_release_ready(0.30)
    policy.grip = 1.0
    assert not policy._single_place_release_ready(0.40)
    assert policy._single_place_release_ready(0.51)


def test_loaded_motion_speedups_remain_below_generic_limits():
    per_kind = _literal("PLACE_LOADED_ARM_MAX_STEP_BY_KIND_RAD")

    assert per_kind["chengzi"] == 0.0045
    assert per_kind["pingguo"] == 0.0045
    assert per_kind["heweidao"] == 0.0105
    assert per_kind["chengzi"] < _literal("PLACE_LOADED_ARM_MAX_STEP_RAD")
    assert _literal("HEWEIDAO_PLACE_DESCENT_SLIDE_STEP_M") == 0.0006
    assert (_literal("HEWEIDAO_PLACE_DESCENT_SLIDE_STEP_M")
            < _literal("PLACE_DESCENT_SLIDE_STEP_M"))
    assert _literal("PLACE_EMPTY_DUAL_RECOVERY_MAX_STEP_RAD") == 0.015


def test_fast_empty_dual_recovery_starts_only_after_table_clearance():
    controller = next(
        node for node in _tree().body
        if isinstance(node, ast.ClassDef)
        and node.name == "IntegratedNavPickPlace")
    smooth = next(
        node for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name == "smooth_commands")
    assignment = next(
        node for node in ast.walk(smooth)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name)
                and target.id == "empty_dual_place_recovery"
                for target in node.targets))
    guard = ast.unparse(assignment.value)

    assert "self.flow_phase == 'place'" in guard
    assert "self.place_stage == 5" in guard
    assert "self._place_retreat_sent" in guard
    assert "self.use_dual_tissue_grasp" in guard
