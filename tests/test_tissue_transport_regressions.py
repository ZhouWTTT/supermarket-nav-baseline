"""Regression guards for tissue transport and direct-retarget bugs."""

from __future__ import annotations

import ast
from pathlib import Path


PICK = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/yolo_aruco_shelf_pick.py"
)
INTEGRATED = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/integrated_nav_pick_place.py"
)


def _constant(name: str, path: Path) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant: {name}")


def test_transport_uses_a_separate_clearly_open_threshold():
    assert _constant("DUAL_TISSUE_TRANSPORT_EMPTY_MIN", PICK) == 0.25
    source = INTEGRATED.read_text(encoding="utf-8")
    assert "pick.DUAL_TISSUE_TRANSPORT_EMPTY_MIN" in source
    assert ("threshold = float(pick.DUAL_TISSUE_GRIP_CONTACT_MAX)"
            not in source)


def test_tissue_direct_leg_is_not_replaced_by_another_tissue_slot():
    source = INTEGRATED.read_text(encoding="utf-8")
    assert ('candidate_kind == "zhijin"' in source)
    assert ('self.target_kind == "zhijin"' in source)
    assert "self.direct_transit_slot is not None" in source


def test_loaded_arm_lift_is_completed_at_the_measured_lift_height():
    source = PICK.read_text(encoding="utf-8")
    assert 'self.dual_motion_label.startswith("arm_lift_")' in source
    assert "if lift_height_ready:" in source
    assert "endpoint_ready = True" in source
