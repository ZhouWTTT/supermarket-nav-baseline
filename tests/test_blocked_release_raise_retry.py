"""Regression tests for safe blocked-release recovery during placement.

A release can be blocked because the chassis drifts horizontally while the
slide lowers a product onto the delivery table.  The controller must never
respond by sweeping the clamped product sideways at near-table height (that
knocked over neighbouring goods in past runs); it has to raise back to the
verified overhead height, re-centre there, and only then descend again.  As a
last resort after bounded retries it may release in place while the product is
still supported on the table.
"""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/integrated_nav_pick_place.py"
)

REQUIRED_CONSTANTS = {
    "PLACE_XY_REFINE_RETRY_TIMEOUT_S": 15.0,
    "PLACE_BLOCKED_RELEASE_RAISE_RETRIES_MAX": 2,
    "PLACE_BLOCKED_RELEASE_INPLACE_MAX_ERROR_M": 0.110,
}


def _tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _literal(name: str):
    for node in _tree().body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant: {name}")


def _methods():
    controller = next(
        node for node in _tree().body
        if isinstance(node, ast.ClassDef)
        and node.name == "IntegratedNavPickPlace")
    return {
        node.name: node for node in controller.body
        if isinstance(node, ast.FunctionDef)
    }


def test_blocked_release_constants_exist_and_are_bounded():
    for name, expected in REQUIRED_CONSTANTS.items():
        assert _literal(name) == expected, name


def test_raise_retry_helpers_exist():
    methods = _methods()
    for name in (
            "_begin_blocked_release_raise",
            "_place_raise_retry_tick",
            "_blocked_release_inplace_safe"):
        assert name in methods, name


def test_blocked_release_never_refines_at_low_height():
    """Stage-2 release-blocked path must use raise/retry or in-place release,
    not an immediate low-height stage-1 horizontal correction."""
    methods = _methods()
    place_tick = methods["_place_tick"]
    calls = []
    for node in ast.walk(place_tick):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"):
            calls.append(node.func.attr)
    assert "_begin_blocked_release_raise" in calls
    assert "_blocked_release_inplace_safe" in calls
    assert "_place_contact_release" in calls


def test_retry_stage_is_kept_loaded():
    """Stage 6 (blocked-release raise) must keep holding the product and must
    be treated as base-locked vertical motion, not a horizontal sweep."""
    source = SOURCE.read_text(encoding="utf-8")
    assert "self.place_stage == 6" in source
    assert "and self.place_stage in {0, 1, 2, 6}" in source
    assert "blocked_release_raise" in source
