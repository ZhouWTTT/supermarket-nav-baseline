import ast
import math
import pathlib
import sys
import unittest

MODULE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting"
)
sys.path.insert(0, str(MODULE_DIR))

from supermarket_navigation import (  # noqa: E402
    AStarPlanner,
    Costmap2D,
    LETHAL,
    NavigationController,
    START_POSE,
)


GOAL = (1.80, 2.475, math.pi / 2.0)
WRAPPER_PATH = (
    MODULE_DIR / "integrated_nav_pick_place_terminal_first_scan.py"
)


def make_controller():
    costmap = Costmap2D()
    controller = NavigationController(costmap, AStarPlanner(costmap))
    return costmap, controller


class TerminalHeadingPlannerTests(unittest.TestCase):
    def test_default_configuration_preserves_original_direct_astar(self):
        costmap, controller = make_controller()
        controller.set_goal(*GOAL)
        actual = controller._try_plan_with_fallback(
            *START_POSE[:2], *GOAL[:2])
        expected = AStarPlanner(costmap).plan(
            *START_POSE[:2], *GOAL[:2])
        self.assertEqual(actual, expected)
        self.assertEqual(
            controller.terminal_heading_status()["mode"], "disabled")

    def test_first_scan_plan_ends_with_heading_aligned_segment(self):
        _, controller = make_controller()
        controller.configure_terminal_heading_approach(
            distance_m=0.65, merge_ahead_m=0.30, release_margin_m=0.12)
        controller.set_goal(*GOAL)
        path = controller._try_plan_with_fallback(
            *START_POSE[:2], *GOAL[:2])

        self.assertIsNotNone(path)
        self.assertEqual(controller.terminal_heading_status()["mode"], "active")
        self.assertAlmostEqual(path[-1][0], GOAL[0], places=9)
        self.assertAlmostEqual(path[-1][1], GOAL[1], places=9)
        self.assertAlmostEqual(path[-2][0], GOAL[0], places=9)
        self.assertAlmostEqual(path[-2][1], GOAL[1] - 0.65, places=9)

    def test_blocked_terminal_segment_falls_back_to_original_astar(self):
        costmap, controller = make_controller()
        costmap._fill_rect(
            costmap.dynamic_raw, 1.79, 2.18, 1.81, 2.22, LETHAL)
        costmap._rebuild_dynamic()
        controller.configure_terminal_heading_approach(
            distance_m=0.65, merge_ahead_m=0.30, release_margin_m=0.12)
        controller.set_goal(*GOAL)
        path = controller._try_plan_with_fallback(
            *START_POSE[:2], *GOAL[:2])

        self.assertIsNotNone(path)
        status = controller.terminal_heading_status()
        self.assertEqual(status["mode"], "fallback_direct")
        self.assertEqual(status["reason"], "terminal_segment_blocked")

    def test_replan_after_anchor_joins_ahead_instead_of_reversing(self):
        _, controller = make_controller()
        controller.configure_terminal_heading_approach(
            distance_m=0.65, merge_ahead_m=0.30, release_margin_m=0.12)
        controller.set_goal(*GOAL)
        base = (1.80, 2.05)
        path = controller._try_plan_with_fallback(
            *base, *GOAL[:2])

        self.assertIsNotNone(path)
        status = controller.terminal_heading_status()
        self.assertEqual(status["mode"], "active")
        self.assertGreater(status["join"][1], base[1])
        self.assertGreater(status["join_progress_m"], 0.0)
        self.assertLessEqual(
            status["join"][1],
            GOAL[1] - controller.terminal_heading_release_margin_m + 1e-9)

    def test_intermediate_recovery_goal_does_not_receive_final_constraint(self):
        _, controller = make_controller()
        controller.configure_terminal_heading_approach(
            distance_m=0.65, merge_ahead_m=0.30, release_margin_m=0.12)
        controller.set_goal(*GOAL)
        status_before = controller.terminal_heading_status()
        intermediate = (1.20, 0.50)
        path = controller._try_plan_with_fallback(
            *START_POSE[:2], *intermediate)

        self.assertIsNotNone(path)
        self.assertEqual(path[-1], intermediate)
        self.assertEqual(controller.terminal_heading_status(), status_before)


class TerminalFirstScanWrapperStructureTests(unittest.TestCase):
    def test_entrypoint_and_timing_are_at_expected_scopes(self):
        tree = ast.parse(WRAPPER_PATH.read_text(encoding="utf-8"))
        controller_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "TerminalFirstScanIntegratedNavPickPlace"
        )
        class_methods = {
            node.name for node in controller_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        module_functions = {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("tick", class_methods)
        self.assertIn("timing_snapshot", class_methods)
        self.assertNotIn("main", class_methods)
        self.assertIn("main", module_functions)


if __name__ == "__main__":
    unittest.main()
