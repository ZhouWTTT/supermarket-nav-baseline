import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

from path_memory import PathMemory  # noqa: E402


ENTRY = (-0.70, 1.45, -math.pi / 2.0)
EXIT = (-1.94, -2.40, -math.pi / 2.0)
PATH = [
    ENTRY[:2],
    (-0.90, 0.70),
    (-1.35, -0.30),
    (-1.55, -1.50),
    EXIT[:2],
]


class PathMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "paths.json"
        self.memory = PathMemory(enabled=True, storage_path=self.path)

    def tearDown(self):
        self.tempdir.cleanup()

    def _save_forward(self):
        self.memory.save_path(
            ENTRY[0], ENTRY[1], ENTRY[2],
            EXIT[0], EXIT[1], EXIT[2], PATH,
            source="delivery_trunk")

    def test_save_creates_forward_and_forward_driving_reverse(self):
        self._save_forward()
        document = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(document["version"], 2)
        self.assertEqual(len(document["sessions"]), 2)

        reverse_path, info = self.memory.load_path(
            EXIT[0], EXIT[1], math.pi / 2.0,
            ENTRY[0], ENTRY[1], math.pi / 2.0)
        self.assertTrue(info["cache_hit"])
        self.assertEqual(info["source"], "delivery_trunk_reverse")
        self.assertEqual(reverse_path, list(reversed(PATH)))

    def test_reverse_requires_heading_along_reversed_polyline(self):
        self._save_forward()
        reverse_path, info = self.memory.load_path(
            EXIT[0], EXIT[1], -math.pi / 2.0,
            ENTRY[0], ENTRY[1], -math.pi / 2.0)
        self.assertIsNone(reverse_path)
        self.assertFalse(info["cache_hit"])

    def test_invalidating_reverse_removes_both_directions_persistently(self):
        self._save_forward()
        _, reverse_info = self.memory.load_path(
            EXIT[0], EXIT[1], math.pi / 2.0,
            ENTRY[0], ENTRY[1], math.pi / 2.0)
        removed = self.memory.invalidate_path(
            reverse_info["matched_key"], "rotation_loop")
        self.assertEqual(len(removed), 2)
        self.assertEqual(self.memory.summary()["cache_size"], 0)

        reloaded = PathMemory(enabled=True, storage_path=self.path)
        self.assertEqual(reloaded.summary()["cache_size"], 0)
        self.assertEqual(reloaded.summary()["invalidation_count"], 1)
        document = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(
            document["invalidations"][-1]["reason"], "rotation_loop")

    def test_invalidating_version_one_entry_removes_legacy_reverse(self):
        forward_key = self.memory.make_key(*ENTRY, *EXIT)
        legacy_reverse_key = self.memory.make_key(
            EXIT[0], EXIT[1], EXIT[2],
            ENTRY[0], ENTRY[1], ENTRY[2])
        document = {
            "version": 1,
            "sessions": {
                forward_key: {
                    "key": forward_key,
                    "start_pose": list(ENTRY),
                    "goal_pose": list(EXIT),
                    "path": PATH,
                    "source": "planner",
                },
                legacy_reverse_key: {
                    "key": legacy_reverse_key,
                    "start_pose": list(EXIT),
                    "goal_pose": list(ENTRY),
                    "path": list(reversed(PATH)),
                    "source": "planner_reverse",
                },
            },
        }
        self.path.write_text(json.dumps(document), encoding="utf-8")
        memory = PathMemory(enabled=True, storage_path=self.path)
        removed = memory.invalidate_path(forward_key, "legacy_failure")
        self.assertIn(forward_key, removed)
        self.assertIn(legacy_reverse_key, removed)
        self.assertEqual(memory.summary()["cache_size"], 0)


if __name__ == "__main__":
    unittest.main()
