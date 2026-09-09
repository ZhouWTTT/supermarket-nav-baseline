"""Regression guard for the apple/orange low-release gate."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/integrated_nav_pick_place.py"
)


def _sphere_bottom_high_tolerances() -> dict[str, float]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
                isinstance(target, ast.Name)
                and target.id == "PLACE_CONTACT_BOTTOM_HIGH_TOL_BY_KIND_M"
                for target in node.targets):
            return {
                key: float(value)
                for key, value in ast.literal_eval(node.value).items()
            }
    raise AssertionError("missing sphere release tolerance table")


def test_sphere_release_tolerance_covers_measured_kinematic_residual():
    tolerances = _sphere_bottom_high_tolerances()
    assert tolerances["pingguo"] >= 0.025
    assert tolerances["chengzi"] >= 0.025


def _constant(name: str):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
                isinstance(target, ast.Name)
                and target.id == name
                for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant: {name}")


def test_outer_slots_use_a_higher_loaded_approach_clearance():
    assert _constant("PLACE_OUTER_SLOT_INDEX") == 3
    outer = float(_constant("PLACE_APPROACH_CLEARANCE_OUTER_SLOT_M"))
    inner = float(_constant("PLACE_APPROACH_CLEARANCE_M"))
    assert outer >= 0.12
    assert outer > inner
    source = SOURCE.read_text(encoding="utf-8")
    assert "PLACE_APPROACH_CLEARANCE_OUTER_SLOT_M" in source


def test_outer_sphere_delivery_parks_near_the_slot_before_lowering():
    source = SOURCE.read_text(encoding="utf-8")
    assert "def _delivery_slot_goal" in source
    assert "PLACE_OUTER_SLOT_INDEX" in source
    assert "float(self.place_world[0])" in source
