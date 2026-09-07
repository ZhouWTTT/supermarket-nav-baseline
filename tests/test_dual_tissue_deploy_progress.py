"""Regression tests for progress-aware dual-tissue pregrasp deployment."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/yolo_aruco_shelf_pick.py"
)


def _tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _literal(name: str):
    for node in _tree().body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant: {name}")


def _policy_class():
    tree = _tree()
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ShelfPickController")
    method = next(
        node for node in controller.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "advance_dual_tissue_deploy")
    policy = ast.ClassDef(
        name="DeployPolicy",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[policy], type_ignores=[]))
    namespace = {
        "math": math,
        "ARM_REACHED_TOLERANCE_RAD": 0.03,
        "DUAL_TISSUE_DEPLOY_DWELL_S": _literal(
            "DUAL_TISSUE_DEPLOY_DWELL_S"),
        "DUAL_TISSUE_DEPLOY_TIMEOUT_S": _literal(
            "DUAL_TISSUE_DEPLOY_TIMEOUT_S"),
        "DUAL_TISSUE_DEPLOY_HARD_TIMEOUT_S": _literal(
            "DUAL_TISSUE_DEPLOY_HARD_TIMEOUT_S"),
        "DUAL_TISSUE_DEPLOY_STALL_TIMEOUT_S": _literal(
            "DUAL_TISSUE_DEPLOY_STALL_TIMEOUT_S"),
        "DUAL_TISSUE_DEPLOY_ARM_PROGRESS_RAD": _literal(
            "DUAL_TISSUE_DEPLOY_ARM_PROGRESS_RAD"),
        "DUAL_TISSUE_DEPLOY_SLIDE_PROGRESS_M": _literal(
            "DUAL_TISSUE_DEPLOY_SLIDE_PROGRESS_M"),
        "DUAL_TISSUE_DEPLOY_PROGRESS_LOG_PERIOD_S": _literal(
            "DUAL_TISSUE_DEPLOY_PROGRESS_LOG_PERIOD_S"),
        "STATE_ABORT": "abort",
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["DeployPolicy"]


DeployPolicy = _policy_class()


class _Logger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def info(self, _message):
        pass

    def warn(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)


class Harness(DeployPolicy):
    def __init__(self):
        self.clock = 0.0
        self.arm_error = 3.0
        self.joints = {"slide_joint": 0.0}
        self.des_slide = 0.0
        self.dual_surround_left_joints = np.zeros(6)
        self.dual_surround_right_joints = np.zeros(6)
        self.dual_deploy_best_arm_error = None
        self.dual_deploy_best_slide_error = None
        self.dual_deploy_last_progress_at = None
        self.dual_deploy_extension_last_log = None
        self.state = "deploy"
        self.logger = _Logger()

    def now(self):
        return self.clock

    def dual_arm_error(self):
        return self.arm_error

    def dual_commands_ready(self, _arm_tolerance, _slide_tolerance):
        return False

    def start_dual_tissue_surround(self):
        self.state = "surround"

    def set_state(self, state):
        self.state = state

    def get_logger(self):
        return self.logger


def test_normal_deadline_is_extended_while_arm_is_making_progress():
    controller = Harness()
    controller.advance_dual_tissue_deploy(0.0)
    controller.clock = 5.0
    controller.arm_error = 2.3

    controller.advance_dual_tissue_deploy(5.0)

    assert controller.state == "deploy"
    assert controller.logger.warnings
    assert not controller.logger.errors


def test_normal_deadline_still_aborts_a_genuinely_stalled_arm():
    controller = Harness()
    controller.advance_dual_tissue_deploy(0.0)
    controller.clock = 5.0

    controller.advance_dual_tissue_deploy(5.0)

    assert controller.state == "abort"
    assert "actuator stalled" in controller.logger.errors[-1]


def test_progress_extension_retains_a_finite_hard_ceiling():
    controller = Harness()
    controller.advance_dual_tissue_deploy(0.0)
    controller.clock = _literal("DUAL_TISSUE_DEPLOY_HARD_TIMEOUT_S")
    controller.arm_error = 1.0

    controller.advance_dual_tissue_deploy(controller.clock)

    assert controller.state == "abort"
    assert "hard timeout" in controller.logger.errors[-1]


def test_progress_policy_is_bounded_but_long_enough_for_slow_top_pose():
    assert _literal("DUAL_TISSUE_DEPLOY_TIMEOUT_S") == 5.0
    assert _literal("DUAL_TISSUE_DEPLOY_HARD_TIMEOUT_S") == 35.0
    assert _literal("DUAL_TISSUE_DEPLOY_STALL_TIMEOUT_S") == 4.0
