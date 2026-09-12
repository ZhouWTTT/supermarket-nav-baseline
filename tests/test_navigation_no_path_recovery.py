import math
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace


MODULE_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

# These controller tests use a fake costmap and do not exercise NumPy/SciPy.
if "numpy" not in sys.modules:
    sys.modules["numpy"] = ModuleType("numpy")
if "scipy" not in sys.modules:
    scipy_module = ModuleType("scipy")
    ndimage_module = ModuleType("scipy.ndimage")
    ndimage_module.maximum_filter = None
    scipy_module.ndimage = ndimage_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.ndimage"] = ndimage_module

from supermarket_navigation import NavigationController  # noqa: E402


class MutablePlanner:
    def __init__(self, path=None):
        self.path = path
        self.failure_reason = None
        self.calls = 0

    def plan(self, *_args):
        self.calls += 1
        if self.path is None:
            self.failure_reason = "disconnected"
            return None
        self.failure_reason = None
        return list(self.path)


class FakeCostmap:
    resolution = 0.05

    def obstacle_counts(self):
        return 0, 0

    def line_is_free(self, *_args):
        return True

    def is_free_world(self, *_args):
        return True

    def is_static_motion_free_world(self, *_args):
        return True

    def is_static_raw_free_world(self, *_args):
        return True

    def is_dynamic_free_world(self, *_args):
        return True

    def raw_static_clearance_world(self, *_args):
        return float("inf")

    def raw_dynamic_clearance_world(self, *_args):
        return float("inf")


def full_scan(front_distance=4.0):
    count = 360
    ranges = [4.0] * count
    ranges[count // 2] = float(front_distance)
    return SimpleNamespace(
        ranges=ranges,
        angle_min=-math.pi,
        angle_increment=2.0 * math.pi / count,
        range_min=0.05,
        range_max=4.0,
    )


def make_controller(path=None):
    planner = MutablePlanner(path)
    return NavigationController(FakeCostmap(), planner), planner


class NavigationNoPathRecoveryTests(unittest.TestCase):
    def test_no_path_waits_five_seconds_then_starts_recovery(self):
        controller, planner = make_controller(None)
        controller.set_goal(1.0, 0.0, 0.0)
        scan = full_scan()

        for tick in range(50):
            controller.compute_velocity(
                0.0, 0.0, math.pi / 2.0, laser_msg=scan,
                time_now=tick * 0.1)
            self.assertIsNone(controller._reverse_recovery_phase)

        for tick in range(50, 53):
            controller.compute_velocity(
                0.0, 0.0, math.pi / 2.0, laser_msg=scan,
                time_now=tick * 0.1)
            if controller._reverse_recovery_phase is not None:
                break

        self.assertEqual(controller._reverse_recovery_phase, "rotate")
        self.assertEqual(controller.stop_reason, "rotate_recovery_start")
        # No-path retries are throttled well below the 50 Hz control rate.
        self.assertGreaterEqual(planner.calls, 10)
        self.assertLess(planner.calls, 20)

    def test_path_recovery_clears_the_no_path_timer(self):
        controller, planner = make_controller(None)
        controller.set_goal(1.0, 0.0, 0.0)
        scan = full_scan()

        for tick in range(25):
            controller.compute_velocity(
                0.0, 0.0, 0.0, laser_msg=scan,
                time_now=tick * 0.1)
        self.assertGreater(controller._no_path_recovery_time, 2.0)

        planner.path = [(0.0, 0.0), (1.0, 0.0)]
        for tick in range(25, 31):
            controller.compute_velocity(
                0.0, 0.0, 0.0, laser_msg=scan,
                time_now=tick * 0.1)
            if controller.path:
                break

        self.assertTrue(controller.path)
        self.assertEqual(controller._no_path_recovery_time, 0.0)
        self.assertIsNone(controller._reverse_recovery_phase)

    def test_lidar_block_keeps_existing_one_second_recovery_threshold(self):
        path = [(0.0, 0.0), (1.0, 0.0)]
        controller, _ = make_controller(path)
        controller.set_goal(1.0, 0.0, 0.0)
        scan = full_scan(front_distance=0.25)

        for tick in range(14):
            controller.compute_velocity(
                0.0, 0.0, 0.0, laser_msg=scan,
                time_now=tick * 0.1)
            if controller._reverse_recovery_phase is not None:
                break

        self.assertIsNotNone(controller._reverse_recovery_phase)
        self.assertLessEqual(tick * 0.1, 1.2)


if __name__ == "__main__":
    unittest.main()
