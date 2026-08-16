"""Host-safe tests for the YOLO-only memory matrix (no ROS/ArUco needed)."""

import pathlib
import sys
import unittest


MODULE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

from memory_matrix import (  # noqa: E402
    COLUMN_X_OFFSET,
    LEVEL_Z_RANGES,
    MemoryMatrix,
    MemoryMatrixTracker,
    SHELF_SCAN_X,
    fixed_slot_from_world,
    primary_candidates_from_document,
    select_memory_hint,
)


class FixedSlotMappingTests(unittest.TestCase):
    def test_all_fixed_grid_slots_map(self):
        for shelf, shelf_x in SHELF_SCAN_X.items():
            for column, offset in COLUMN_X_OFFSET.items():
                for level, (z_min, z_max) in LEVEL_Z_RANGES.items():
                    z = 0.5 * (z_min + z_max)
                    self.assertEqual(
                        fixed_slot_from_world(
                            shelf_x + offset, z, shelf=shelf),
                        (shelf, level, column))

    def test_dead_zone_z_is_rejected(self):
        # 层间死区：深度 z 无法明确归属任何一层时不猜层。
        self.assertIsNone(fixed_slot_from_world(
            SHELF_SCAN_X["E"], 0.74, shelf="E"))
        self.assertIsNone(fixed_slot_from_world(
            SHELF_SCAN_X["E"], 1.06, shelf="E"))

    def test_off_shelf_x_is_rejected(self):
        self.assertIsNone(fixed_slot_from_world(
            SHELF_SCAN_X["E"] + 0.40, 0.90, shelf="E"))


class MemoryMatrixTests(unittest.TestCase):
    def test_record_overwrite_margin_and_consume_slot(self):
        matrix = MemoryMatrix()
        self.assertTrue(matrix.record_at(
            "E", "L2", "2", -1, "kele", 0.95))
        # 同种类更低置信度不覆盖。
        self.assertFalse(matrix.record_at(
            "E", "L2", "2", -1, "kele", 0.90))
        # 换种类但置信度不够高不覆盖（防 aruco 时代同格反复改写的问题）。
        self.assertFalse(matrix.record_at(
            "E", "L2", "2", -1, "maidong", 0.96))
        self.assertTrue(matrix.record_at(
            "E", "L2", "2", -1, "maidong", 0.98))
        self.assertTrue(matrix.consume_slot("E", "L2", "2"))
        self.assertTrue(matrix.cells["L2|E|2"]["consumed"])
        # 已消费格不再接受新记录。
        self.assertFalse(matrix.record_at(
            "E", "L2", "2", -1, "shupian", 0.99))

    def test_to_json_shape(self):
        matrix = MemoryMatrix()
        matrix.record_at("E", "L2", "2", -1, "kele", 0.90)
        document = matrix.to_json()
        self.assertEqual(len(document["rows"]), 3)
        self.assertEqual(len(document["cols"]), 15)
        column_index = document["cols"].index("E2")
        self.assertEqual(
            document["grid"]["L2"][column_index],
            {"kind": "kele", "consumed": False})

    def test_near_distance_beats_far_higher_confidence(self):
        matrix = MemoryMatrix()
        self.assertTrue(matrix.record_at(
            "B", "L1", "3", -1, "heweidao", 0.80,
            observation_distance=1.00, sample_count=4))
        self.assertTrue(matrix.record_at(
            "B", "L1", "3", -1, "kele", 0.99,
            observation_distance=2.00, sample_count=4))

        # 主网格显示近距离候选；远处高置信度仍保存在候选证据中。
        self.assertEqual(
            matrix.cells["L1|B|3"]["kind"], "heweidao")
        self.assertEqual(
            {item["kind"] for item in matrix.candidates_for("kele")},
            {"kele"})
        self.assertIn(
            "kele", matrix.to_json()["candidates"]["L1|B|3"])

    def test_closer_band_refreshes_then_keeps_best_confidence(self):
        matrix = MemoryMatrix()
        matrix.record_at(
            "C", "L2", "2", -1, "heweidao", 0.98,
            observation_distance=2.00, sample_count=4)
        # 已明显进入更近距离带：旧的远距 0.98 不再代表这个候选。
        matrix.record_at(
            "C", "L2", "2", -1, "heweidao", 0.72,
            observation_distance=1.60, sample_count=4)
        candidate = matrix.candidates_for("heweidao")[0]
        self.assertEqual(candidate["closest_distance"], 1.6)
        self.assertEqual(candidate["confidence"], 0.72)

        # 同一最近 15cm 距离带内，保留置信度最高的一批。
        matrix.record_at(
            "C", "L2", "2", -1, "heweidao", 0.88,
            observation_distance=1.55, observer_xy=(0.1, 2.4),
            sample_count=4)
        matrix.record_at(
            "C", "L2", "2", -1, "heweidao", 0.99,
            observation_distance=2.30, observer_xy=(1.8, 2.4),
            sample_count=4)
        candidate = matrix.candidates_for("heweidao")[0]
        self.assertEqual(candidate["closest_distance"], 1.55)
        self.assertEqual(candidate["observation_distance"], 1.55)
        self.assertEqual(candidate["confidence"], 0.88)
        # 后来的远处观测只累计次数，不能冒充最佳近距观测的位置。
        self.assertEqual(candidate["observer_x"], 0.1)
        self.assertEqual(candidate["observations"], 4)
        self.assertEqual(candidate["sample_count"], 16)

    def test_navigation_uses_gui_primary_not_hidden_candidate(self):
        matrix = MemoryMatrix()
        # 历史上这格曾被识别为苹果，后来更近的脉动成为 GUI 主证据。
        matrix.record_at(
            "A", "L3", "2", -1, "pingguo", 0.976,
            observation_distance=0.491, sample_count=24)
        matrix.record_at(
            "A", "L3", "2", -1, "maidong", 0.976,
            observation_distance=0.353, sample_count=24)

        self.assertEqual(matrix.cells["L3|A|2"]["kind"], "maidong")
        self.assertEqual(len(matrix.candidates_for("pingguo")), 1)
        self.assertEqual(matrix.primary_candidates_for("pingguo"), [])
        self.assertEqual(
            matrix.primary_candidates_for("maidong")[0]["slot_key"],
            "L3|A|2")

    def test_consuming_item_invalidates_same_level_duplicates(self):
        matrix = MemoryMatrix()
        matrix.record_at(
            "A", "L3", "1", -1, "pingguo", 0.97,
            observation_distance=0.37, sample_count=20)
        matrix.record_at(
            "A", "L3", "2", -1, "pingguo", 0.976,
            observation_distance=0.49, sample_count=20)
        matrix.record_at(
            "B", "L3", "3", -1, "pingguo", 0.96,
            observation_distance=0.60, sample_count=20)

        self.assertTrue(matrix.consume_slot(
            "A", "L3", "1", kind="pingguo"))
        remaining = {
            item["slot_key"]
            for item in matrix.primary_candidates_for("pingguo")}
        self.assertEqual(remaining, {"L3|B|3"})
        self.assertNotIn(
            "pingguo", matrix.candidates.get("L3|A|2", {}))

        # 同层真有第二个苹果时，抓取后的新观测可重建候选。
        self.assertTrue(matrix.record_at(
            "A", "L3", "2", -1, "pingguo", 0.98,
            observation_distance=0.30, sample_count=4))
        remaining = {
            item["slot_key"]
            for item in matrix.primary_candidates_for("pingguo")}
        self.assertEqual(remaining, {"L3|A|2", "L3|B|3"})

    def test_consume_ignores_column_when_grasp_slot_drifted(self):
        """列不参与身份：抓取列漂移也必须清掉同货架同层的旧记录。"""
        matrix = MemoryMatrix()
        matrix.record_at(
            "C", "L1", "3", -1, "kele", 0.981,
            observation_distance=0.447, sample_count=40)

        # 抓取本地化落在 C-L1-2，但矩阵此前只记录了 C-L1-3。
        self.assertTrue(matrix.consume_slot(
            "C", "L1", "2", kind="kele"))
        self.assertEqual(matrix.primary_candidates_for("kele"), [])
        self.assertNotIn("L1|C|3", matrix.cells)

    def test_fourth_apple_replay_routes_to_b_not_hidden_a(self):
        """回放 09:00:59 实跑的第三/四单苹果证据。"""
        matrix = MemoryMatrix()
        # 第三单真正抓走的 A-L3-1 苹果。
        matrix.record_at(
            "A", "L3", "1", -1, "pingguo", 0.974,
            observation_distance=0.369, sample_count=584)
        # 同一苹果因 x 抖动留在 A-L3-2 的历史副本；该格
        # GUI 主证据已是更近的脉动，所以界面显示 A 无苹果。
        matrix.record_at(
            "A", "L3", "2", -1, "pingguo", 0.976,
            observation_distance=0.491, sample_count=24)
        matrix.record_at(
            "A", "L3", "2", -1, "maidong", 0.976,
            observation_distance=0.353, sample_count=24)
        # 第四单实际抓到的 B-L3-3 苹果。
        matrix.record_at(
            "B", "L3", "3", -1, "pingguo", 0.970,
            observation_distance=0.50, sample_count=24)

        self.assertTrue(matrix.consume_slot(
            "A", "L3", "1", kind="pingguo"))
        routes = matrix.primary_candidates_for("pingguo")
        self.assertEqual(
            [(item["shelf"], item["level"], item["column"])
             for item in routes],
            [("B", "L3", "3")])

    def test_serialized_navigation_reads_only_gui_primary(self):
        matrix = MemoryMatrix()
        matrix.record_at(
            "A", "L3", "2", -1, "pingguo", 0.97,
            observation_distance=0.60, sample_count=4)
        matrix.record_at(
            "A", "L3", "2", -1, "maidong", 0.96,
            observation_distance=0.30, sample_count=4)
        document = matrix.to_json()
        self.assertEqual(
            primary_candidates_from_document(document, "pingguo"), [])
        self.assertEqual(
            primary_candidates_from_document(
                document, "maidong")[0]["slot_key"],
            "L3|A|2")

    def test_direct_route_and_failed_level_failover(self):
        candidates = [
            {
                "slot_key": "L3|B|3", "shelf": "B", "level": "L3",
                "confidence": 0.94, "closest_distance": 0.67,
                "last_seen": 10.0,
            },
            {
                "slot_key": "L2|D|3", "shelf": "D", "level": "L2",
                "confidence": 0.97, "closest_distance": 0.58,
                "last_seen": 11.0,
            },
        ]
        initial = select_memory_hint(
            candidates, (-1.70, 2.40), 0.90, reliable_only=True)
        self.assertEqual((initial["shelf"], initial["level"]), ("B", "L3"))
        failover = select_memory_hint(
            candidates, (-0.91, 2.45), 0.90,
            exclude_shelf_levels={("B", "L3")}, reliable_only=True)
        self.assertEqual(
            (failover["shelf"], failover["level"]), ("D", "L2"))

    def test_dynamic_route_requires_fresh_evidence(self):
        candidates = [
            {
                "slot_key": "L1|A|1", "shelf": "A", "level": "L1",
                "confidence": 0.99, "closest_distance": 0.50,
                "last_seen": 99.0,
            },
            {
                "slot_key": "L1|C|2", "shelf": "C", "level": "L1",
                "confidence": 0.95, "closest_distance": 0.60,
                "last_seen": 101.0,
            },
        ]
        hint = select_memory_hint(
            candidates, (0.10, 2.40), 0.90,
            min_last_seen=100.0, reliable_only=True)
        self.assertEqual(hint["shelf"], "C")


class MultiShelfRecordingTests(unittest.TestCase):
    class _Logger:
        def info(self, _message):
            pass

        def warn(self, _message):
            pass

    @staticmethod
    def _tracker_at_e_station():
        # 绕过 ROS Node.__init__，只测试 host-safe 的录入核心。
        tracker = MemoryMatrixTracker.__new__(MemoryMatrixTracker)
        tracker.matrix = MemoryMatrix()
        tracker.confirmations = 3
        tracker._slot_acc = {}
        tracker._blocked_log_at = {}
        tracker.base_xy = (SHELF_SCAN_X["E"], 2.475)
        tracker._dirty = False
        tracker.get_logger = lambda: MultiShelfRecordingTests._Logger()
        return tracker

    def test_one_view_records_multiple_shelves_by_detection_world_x(self):
        tracker = self._tracker_at_e_station()
        detections = (
            ("kele", "A", "1", "L1", 0.60),
            ("maidong", "C", "2", "L2", 0.90),
            ("shupian", "E", "3", "L3", 1.24),
        )
        for frame_index in range(4):
            stamp_ns = 1_000_000_000 + frame_index
            for kind, shelf, column, _level, z in detections:
                tracker._record_yolo_only({
                    "class": kind,
                    "conf": 0.90,
                    "stamp_ns": stamp_ns,
                    "world": [
                        SHELF_SCAN_X[shelf] + COLUMN_X_OFFSET[column],
                        3.243,
                        z,
                    ],
                })

        # 观察者在 E 站，但 A/C/E 都应由各自世界 x 写入。
        self.assertEqual(
            tracker.matrix.cells["L1|A|1"]["kind"], "kele")
        self.assertEqual(
            tracker.matrix.cells["L2|C|2"]["kind"], "maidong")
        self.assertEqual(
            tracker.matrix.cells["L3|E|3"]["kind"], "shupian")

    def test_non_shelf_world_y_is_rejected(self):
        tracker = self._tracker_at_e_station()
        for frame_index in range(4):
            tracker._record_yolo_only({
                "class": "kele",
                "conf": 0.99,
                "stamp_ns": frame_index,
                "world": [SHELF_SCAN_X["A"], 2.40, 0.60],
            })
        self.assertEqual(tracker.matrix.cells, {})

    def test_front_depth_is_used_as_observation_distance(self):
        tracker = self._tracker_at_e_station()
        for frame_index in range(4):
            tracker._record_yolo_only({
                "class": "heweidao",
                "conf": 0.82,
                "stamp_ns": frame_index,
                "front_depth_m": 0.91 + 0.01 * frame_index,
                "world": [SHELF_SCAN_X["B"], 3.24, 0.58],
            })
        candidate = tracker.matrix.candidates_for("heweidao")[0]
        self.assertAlmostEqual(
            candidate["closest_distance"], 0.93, places=2)


if __name__ == "__main__":
    unittest.main()
