"""Non-sphere products release from a hover instead of pressing the table.

Boxes and heweidao keep a commanded clearance above the delivery table and
open the gripper there, letting the product fall the last few centimetres.
Only spheres (chengzi/pingguo) still descend to their 3 mm low release and
require the product bottom at the table before opening.
"""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/integrated_nav_pick_place.py"
)


def _literal(name: str):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant: {name}")


def test_non_sphere_hover_clearance_is_at_least_twenty_mm():
    assert _literal("PLACE_PRODUCT_BOTTOM_CLEARANCE_M") == 0.020


def test_heweidao_uses_supported_low_release_with_backup():
    """Heweidao's wide rim cannot fall through the open jaws, so it releases
    just above the table (supported) and relies on the 100 mm fixed-arm
    chassis backup instead of a hover drop."""
    raise_m = _literal("HEWEIDAO_PLACE_RELEASE_RAISE_M")
    overtravel = _literal("HEWEIDAO_PLACE_CONTACT_OVERTRAVEL_M")
    assert 0.0 <= raise_m - overtravel <= 0.010


def test_only_spheres_require_bottom_at_table_for_release():
    source = SOURCE.read_text(encoding="utf-8")
    assert "_target_is_sphere_product()" in source
    assert "target_kind != \"heweidao\"" not in source
