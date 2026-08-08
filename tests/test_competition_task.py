import json
import pathlib
import sys
import unittest


MODULE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

from competition_task import (  # noqa: E402
    CompetitionTask,
    TaskMessageError,
    associate_detection_marker,
    marker_arguments,
)


def task_message(targets, run_prefix="run_test"):
    return json.dumps({
        "schema_version": 1,
        "run_prefix": run_prefix,
        "count": len(targets),
        "targets": targets,
    })


class CompetitionTaskTests(unittest.TestCase):
    def test_parses_repeated_kinds_and_preserves_anonymous_ids(self):
        task = CompetitionTask.from_json(task_message([
            {"id": "item_a", "kind": "kele"},
            {"id": "item_b", "kind": "kele"},
            {"id": "item_c", "kind": "zhijin"},
        ]))
        self.assertEqual([order.id for order in task.orders],
                         ["item_a", "item_b", "item_c"])
        self.assertEqual([order.kind for order in task.orders],
                         ["kele", "kele", "zhijin"])

    def test_rejects_invalid_schema_count_kind_and_duplicate_id(self):
        invalid_documents = [
            {"schema_version": 2, "run_prefix": "run", "count": 1,
             "targets": [{"id": "x", "kind": "kele"}]},
            {"schema_version": 1, "run_prefix": "run", "count": 2,
             "targets": [{"id": "x", "kind": "kele"}]},
            {"schema_version": 1, "run_prefix": "run", "count": 1,
             "targets": [{"id": "x", "kind": "unknown"}]},
            {"schema_version": 1, "run_prefix": "run", "count": 2,
             "targets": [
                 {"id": "x", "kind": "kele"},
                 {"id": "x", "kind": "maidong"},
             ]},
        ]
        for document in invalid_documents:
            with self.subTest(document=document):
                with self.assertRaises(TaskMessageError):
                    CompetitionTask.from_json(json.dumps(document))

    def test_scheduler_optimizes_by_grasp_cost_not_source_order(self):
        task = CompetitionTask.from_json(task_message([
            {"id": "tissue", "kind": "zhijin"},
            {"id": "bottle", "kind": "kele"},
        ]))
        self.assertEqual(task.next_order(max_attempts=2).id, "bottle")

    def test_failure_retries_then_moves_on(self):
        task = CompetitionTask.from_json(task_message([
            {"id": "a", "kind": "kele"},
            {"id": "b", "kind": "maidong"},
        ]))
        order = task.next_order(max_attempts=2)
        task.finish_attempt(
            order, delivered=False, marker_id=12, error="ik",
            max_attempts=2)
        self.assertEqual(order.status, "pending")
        # Unattempted work is preferred before retrying a failed order.
        self.assertEqual(task.next_order(max_attempts=2).id, "b")
        task.finish_attempt(
            order, delivered=False, marker_id=13, error="timeout",
            max_attempts=2)
        self.assertEqual(order.status, "failed")
        self.assertEqual(task.excluded_markers("kele"), [12, 13])

    def test_success_and_summary(self):
        task = CompetitionTask.from_json(task_message([
            {"id": "a", "kind": "kele"},
        ]))
        order = task.next_order(max_attempts=2)
        task.finish_attempt(
            order, delivered=True, marker_id=9, max_attempts=2)
        self.assertTrue(task.terminal)
        self.assertEqual(task.summary()["delivered"], 1)
        self.assertEqual(marker_arguments([9, 2, 9]),
                         ["--exclude-marker-id", "2",
                          "--exclude-marker-id", "9"])

    def test_inventory_association_uses_measured_geometry(self):
        detection = {
            "class": "kele",
            "bbox_xyxy": [100, 100, 200, 200],
            "world": [0.0, 3.24, 0.9215],
        }
        markers = [
            {"id": 8, "pixel_center": [310, 220],
             "position_world": [0.5, 3.24, 0.85]},
            {"id": 7, "pixel_center": [150, 220],
             "position_world": [0.0, 3.24, 0.85]},
        ]
        marker = associate_detection_marker(detection, markers)
        self.assertIsNotNone(marker)
        self.assertEqual(marker["id"], 7)


if __name__ == "__main__":
    unittest.main()
