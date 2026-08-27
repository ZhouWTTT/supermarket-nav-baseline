"""Host-safe regression tests for live memory reroute admission."""

import pathlib
import sys
import unittest


MODULE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

from memory_matrix import select_memory_route_hint  # noqa: E402


class DynamicDirectAdmissionTests(unittest.TestCase):
    @staticmethod
    def _candidate(*, confidence, observations, sample_count):
        return {
            "slot_key": "L1|D|1",
            "shelf": "D",
            "level": "L1",
            "column": "1",
            "confidence": confidence,
            "closest_distance": 5.39,
            "last_seen": 100.0,
            "observations": observations,
            "sample_count": sample_count,
            "world_x": 0.70,
            "world_y": 3.35,
            "world_z": 0.56,
        }

    def test_first_weak_far_observation_cannot_redirect_live_route(self):
        weak = self._candidate(
            confidence=0.585, observations=1, sample_count=4)
        selected = select_memory_route_hint(
            "zhijin", [weak], [weak], (1.51, -1.62), 0.90,
            min_last_seen=99.0,
            reliable_only=True,
            require_direct=True)
        self.assertIsNone(selected)

    def test_stable_depth_candidate_can_redirect_to_concrete_slot(self):
        stable = self._candidate(
            confidence=0.95, observations=3, sample_count=12)
        selected = select_memory_route_hint(
            "zhijin", [stable], [stable], (1.51, -1.62), 0.90,
            min_last_seen=99.0,
            reliable_only=True,
            require_direct=True)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["slot_key"], "L1|D|1")


if __name__ == "__main__":
    unittest.main()
