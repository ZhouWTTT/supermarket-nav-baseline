"""Regression tests for relaxed in-table release during placement.

The chassis can drift while the slide lowers a product onto the delivery
table, leaving the TCP a few centimetres off the assigned slot.  The referee
only requires the final position inside the delivery box, so the controller
must never sweep the clamped product sideways at low height, raise-and-retry
for minutes, or fail the order over a small offset -- it releases in place as
soon as the TCP is over the delivery table (spheres still need a verified low
bottom pose to avoid rolling).
"""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/integrated_nav_pick_place.py"
)


def _tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _methods():
    controller = next(
        node for node in _tree().body
        if isinstance(node, ast.ClassDef)
        and node.name == "IntegratedNavPickPlace")
    return {
        node.name: node for node in controller.body
        if isinstance(node, ast.FunctionDef)
    }


def _self_calls(method_name: str):
    method = _methods()[method_name]
    calls = []
    for node in ast.walk(method):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"):
            calls.append(node.func.attr)
    return calls


def test_release_gate_directly_releases_when_over_table():
    calls = _self_calls("_place_tick")
    assert "_place_contact_release" in calls


def test_blocked_release_no_longer_retries_or_refines_low():
    """The stage-2 gate must not fall back to raise/refine retries that made
    off-slot placements fail; in-table poses release in place."""
    calls = _self_calls("_place_tick")
    assert "_begin_blocked_release_raise" not in calls
    assert "_blocked_release_inplace_safe" not in calls


def test_sphere_bottom_safety_is_kept():
    source = SOURCE.read_text(encoding="utf-8")
    assert "sphere release bottom not verified" in source
