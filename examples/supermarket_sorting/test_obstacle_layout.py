#!/usr/bin/env python3
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from obstacle_layout import (
    EMPTY_CORRIDOR_PATH_LENGTH,
    MIN_DETOUR_METERS,
    OBSTACLE_BODIES,
    OBSTACLE_PAIR_GAP,
    OBSTACLE_HALF_SIZE,
    ROBOT_CLEARANCE_RADIUS,
    ROBOT_HALF_LENGTH,
    ROBOT_HALF_WIDTH,
    ROBOT_SAFETY_MARGIN,
    evaluate_obstacle_layout,
    generate_obstacle_layout,
)


TASK_DIR = Path(__file__).resolve().parent


class ObstacleLayoutTest(unittest.TestCase):
    def test_seed_is_reproducible(self):
        first = generate_obstacle_layout(12345)
        second = generate_obstacle_layout(12345)
        self.assertEqual(first, second)

    def test_many_seeds_are_safe_and_require_avoidance(self):
        for seed in range(200):
            generated = generate_obstacle_layout(seed)
            selected = tuple(
                (*generated.positions[body][:2], generated.yaws[body])
                for body in OBSTACLE_BODIES
            )
            evaluation = evaluate_obstacle_layout(selected)
            self.assertTrue(evaluation.valid, (seed, evaluation.reason))
            self.assertGreaterEqual(
                evaluation.path_length,
                EMPTY_CORRIDOR_PATH_LENGTH + MIN_DETOUR_METERS,
            )

    def test_overlapping_obstacles_are_rejected(self):
        # Bodies 01 and 02 overlap at these two distinct centers.
        selected = (
            (-0.10, -0.35, 0.0), (0.00, -0.35, 0.0), (0.35, -1.90, 0.0),
            (-0.95, 1.70, 0.0), (0.55, 2.25, 0.0),
        )
        evaluation = evaluate_obstacle_layout(selected)
        self.assertFalse(evaluation.valid)
        self.assertIn("overlap", evaluation.reason)

    def test_generated_boxes_respect_pairwise_gap(self):
        generated = generate_obstacle_layout(7)
        selected = [
            (*generated.positions[body][:2], generated.yaws[body])
            for body in OBSTACLE_BODIES
        ]
        for index, (x, y, _) in enumerate(selected):
            half_x, half_y = OBSTACLE_HALF_SIZE
            for other_index in range(index):
                other_x, other_y, _ = selected[other_index]
                other_half_x, other_half_y = OBSTACLE_HALF_SIZE
                separated_x = abs(x - other_x) >= half_x + other_half_x + OBSTACLE_PAIR_GAP
                separated_y = abs(y - other_y) >= half_y + other_half_y + OBSTACLE_PAIR_GAP
                self.assertTrue(separated_x or separated_y)

    def test_planning_dimensions_match_mjcf(self):
        scene_root = ET.parse(TASK_DIR / "mjcf" / "retail_competition.xml").getroot()
        for index in range(1, 6):
            body_name = f"dynamic_obstacle_box_{index:02d}"
            body = scene_root.find(f".//body[@name='{body_name}']")
            self.assertIsNotNone(body, body_name)
            geom = body.find("geom")
            size_x, size_y, _ = map(float, geom.attrib["size"].split())
            self.assertEqual((size_x, size_y), OBSTACLE_HALF_SIZE)

        robot_root = ET.parse(
            TASK_DIR / "models" / "mjcf" / "mobile_chassis" / "mmk2" / "mmk2.xml"
        ).getroot()
        agv_body = robot_root.find("./body[@name='agv_link']")
        self.assertIsNotNone(agv_body)
        body_x, body_y, _ = map(float, agv_body.attrib.get("pos", "0 0 0").split())
        base_boxes = [
            geom for geom in agv_body.findall("./geom")
            if geom.attrib.get("type") == "box" and geom.attrib.get("group") == "4"
        ]
        self.assertTrue(base_boxes)
        actual_half_length = max(
            abs(body_x + float(geom.attrib.get("pos", "0 0 0").split()[0]))
            + float(geom.attrib["size"].split()[0])
            for geom in base_boxes
        )
        actual_half_width = max(
            abs(body_y + float(geom.attrib.get("pos", "0 0 0").split()[1]))
            + float(geom.attrib["size"].split()[1])
            for geom in base_boxes
        )
        self.assertLessEqual(actual_half_length, ROBOT_HALF_LENGTH)
        self.assertLessEqual(actual_half_width, ROBOT_HALF_WIDTH)
        required_radius = (
            actual_half_length ** 2 + actual_half_width ** 2
        ) ** 0.5 + ROBOT_SAFETY_MARGIN
        self.assertGreaterEqual(ROBOT_CLEARANCE_RADIUS, required_radius)


if __name__ == "__main__":
    unittest.main()
