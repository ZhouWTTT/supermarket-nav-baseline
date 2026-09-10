"""Non-sphere products release from a hover instead of pressing the table.

Boxes keep a commanded clearance above the delivery table and fall the last
few centimetres.  Heweidao instead seats its inverted wide rim on the table,
then uses a slow vertical unhook because its rim is wider than the gripper.
Spheres (chengzi/pingguo) retain their 3 mm low release.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/integrated_nav_pick_place.py"
)


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


def test_non_sphere_hover_clearance_is_at_least_twenty_mm():
    assert _literal("PLACE_PRODUCT_BOTTOM_CLEARANCE_M") == 0.020


def test_heweidao_uses_supported_low_release():
    """Heweidao's wide rim cannot fall through the open jaws, so it releases
    against the table instead of using a hover drop."""
    raise_m = _literal("HEWEIDAO_PLACE_RELEASE_RAISE_M")
    overtravel = _literal("HEWEIDAO_PLACE_CONTACT_OVERTRAVEL_M")
    assert math.isclose(raise_m - overtravel, -0.002)


def test_heweidao_lifts_vertically_instead_of_dragging_sideways():
    source = SOURCE.read_text(encoding="utf-8")
    assert "HEWEIDAO_RELEASE_BASE_BACKUP" not in source
    assert "before the slow vertical unhook" in source
    assert "self._start_place_vertical_clear(now)" in source


def test_only_spheres_require_bottom_at_table_for_release():
    source = SOURCE.read_text(encoding="utf-8")
    assert "_target_is_sphere_product()" in source
    assert "target_kind != \"heweidao\"" not in source
