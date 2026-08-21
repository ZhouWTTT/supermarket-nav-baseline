import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name
               for target in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


class DeliverySlotLayoutTests(unittest.TestCase):
    def test_all_five_slots_are_shifted_south_and_clear_table_edges(self):
        slots = _literal_assignment(
            ROOT / "examples/supermarket_sorting/integrated_nav_pick_place.py",
            "DELIVERY_PLACE_SLOTS_XY",
        )
        bounds = _literal_assignment(
            ROOT / "examples/supermarket_sorting/supermarket_navigation.py",
            "DELIVERY_TABLE_XML_BOUNDS",
        )

        self.assertEqual(slots, (
            (-2.20, -3.50),
            (-1.94, -3.48),
            (-1.68, -3.46),
            (-2.07, -3.34),
            (-1.81, -3.32),
        ))

        x_min, y_min, x_max, y_max = bounds
        minimum_margin = 0.13 - 1e-9
        for x, y in slots:
            with self.subTest(slot=(x, y)):
                self.assertGreaterEqual(x - x_min, minimum_margin)
                self.assertGreaterEqual(x_max - x, minimum_margin)
                self.assertGreaterEqual(y - y_min, minimum_margin)
                self.assertGreaterEqual(y_max - y, minimum_margin)


if __name__ == "__main__":
    unittest.main()
