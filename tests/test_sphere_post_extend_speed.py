"""Regression tests for sphere forward landing and seating."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/yolo_aruco_shelf_pick.py"
)
CONSTANTS = {
    "GENERIC_DIRECT_FORWARD_SPEED_MPS",
    "GENERIC_DIRECT_FORWARD_MIN_DURATION_S",
    "GENERIC_DIRECT_FORWARD_MIN_DURATION_BY_KIND_S",
    "SPHERE_POST_CONTACT_EXTENSION_M",
    "SPHERE_POST_EXTEND_SPEED_MPS",
    "SPHERE_EXTENSION_ALIGN_FORWARD_M",
    "STATE_POST_EXTEND",
}


def _policy_class():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in CONSTANTS
            for target in node.targets)
    ]
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ShelfPickController")
    methods = [
        node for node in controller.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "prepare_sphere_post_extension",
            "start_post_extension",
        }
    ]
    policy = ast.ClassDef(
        name="PostExtendPolicy",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[*assignments, policy], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["PostExtendPolicy"]


PostExtendPolicy = _policy_class()


class _Logger:
    def info(self, _message):
        pass

    def error(self, _message):
        pass


class Harness(PostExtendPolicy):
    def __init__(self, *, sphere: bool):
        self.use_sphere_grasp = sphere
        self.target_kind = "pingguo"
        self.shelf_level = "middle"
        self.generic_direct_speed_mps = 0.090
        self.post_extend_nominal_world = np.zeros(3)
        self.post_extend_target_world = np.array([0.0, 0.05, 0.0])
        self.post_extend_arm_joints = np.ones(6)
        self.post_extend_endpoint_ready_since = None
        self.forward_contact_world = None
        self.state = None
        self.state_t0 = 0.0
        self.clock = 0.0
        self.actual_tcp = np.array([1.0, 2.0, 3.0])

    def now(self):
        return self.clock

    def selected_arm_positions(self):
        return np.zeros(6)

    def selected_tcp_world(self):
        return self.actual_tcp.copy()

    def solve_kdl_world(self, world, _reference):
        self.solved_world = world.copy()
        return np.ones(6)

    def set_selected_arm_target(self, joints):
        self.arm_target = joints.copy()

    def get_logger(self):
        return _Logger()

    def set_state(self, state):
        self.state = state
        self.state_t0 = self.clock


def test_sphere_post_extension_uses_gentle_twenty_mm_per_second_limit():
    policy = Harness(sphere=True)
    assert policy.prepare_sphere_post_extension()
    policy.start_post_extension()

    assert np.allclose(policy.post_extend_target_world, [1.0, 2.080, 3.0])
    assert policy.post_extend_speed_mps == 0.020
    assert math.isclose(policy.post_extend_duration_s, 6.0)
    assert policy.state == "post_extend"


def test_generic_post_extension_keeps_bounded_speed_and_duration():
    policy = Harness(sphere=False)
    policy.start_post_extension()

    assert policy.post_extend_speed_mps == 0.090
    assert math.isclose(policy.post_extend_duration_s, 1.2)
    assert policy.state == "post_extend"


def test_first_sphere_forward_landing_moves_eight_mm_deeper():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "SPHERE_FINGER_ENGAGEMENT_M"
            for target in node.targets))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "sphere_first_forward_tcp_y")
    module = ast.fix_missing_locations(
        ast.Module(body=[assignment, function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    endpoint = namespace["sphere_first_forward_tcp_y"]

    assert math.isclose(namespace["SPHERE_FINGER_ENGAGEMENT_M"], 0.020)
    assert math.isclose(endpoint(3.243, 0.035), 3.228)
    assert math.isclose(endpoint(3.243, 0.037), 3.226)
    assert math.isclose(endpoint(3.243, 0.035) - (3.243 - 0.035 + 0.012),
                        0.008)


def test_deeper_middle_sphere_endpoint_keeps_the_same_arm_reach_margin():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    value = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "SPHERE_EXTENSION_ALIGN_FORWARD_M"
            for target in node.targets))

    assert math.isclose(value, 0.015)
