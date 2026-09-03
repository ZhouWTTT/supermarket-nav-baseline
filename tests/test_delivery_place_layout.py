"""Regression checks for delivery-slot depth and tabletop containment."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "examples/supermarket_sorting/integrated_nav_pick_place.py"
)


def _literal(name):
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant: {name}")


def test_every_delivery_position_is_thirty_millimetres_deeper():
    old_slots = (
        (-2.20, -3.50),
        (-1.94, -3.48),
        (-1.68, -3.46),
        (-2.07, -3.39),
        (-1.81, -3.37),
    )
    new_slots = _literal("DELIVERY_PLACE_SLOTS_XY")

    assert len(new_slots) == len(old_slots)
    for (old_x, old_y), (new_x, new_y) in zip(old_slots, new_slots):
        assert new_x == old_x
        assert round(new_y - old_y, 3) == -0.030
    assert _literal("TISSUE_DEDICATED_PLACE_XY") == (-1.55, -3.33)
    assert _literal("DELIVERY_TABLE_PLACE_WORLD") == (-1.80, -3.38, 0.85)


def test_shifted_centres_keep_the_controller_table_margin():
    x_min, y_min, x_max, y_max = (-2.42, -3.63, -1.46, -3.19)
    margin = _literal("PLACE_RELEASE_TABLE_MARGIN_M")
    positions = [
        *_literal("DELIVERY_PLACE_SLOTS_XY"),
        _literal("TISSUE_DEDICATED_PLACE_XY"),
        _literal("DELIVERY_TABLE_PLACE_WORLD")[:2],
    ]

    assert all(
        x_min + margin <= x <= x_max - margin
        and y_min + margin <= y <= y_max - margin
        for x, y in positions
    )
    assert _literal("PLACE_CREEP_DISTANCE_M") == 0.28
