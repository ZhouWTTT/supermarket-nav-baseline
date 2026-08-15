import hashlib
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "examples" / "supermarket_sorting"
sys.path.insert(0, str(MODULE_DIR))

from replay_outcome_memory import (  # noqa: E402
    BACKUP_AVAILABLE, EXHAUSTED_UNTIL_MATERIAL_CHANGE,
    NO_FRESH_RGB, REACTIVATED, RGB_NOT_PROCESSED_BY_YOLO,
    RUNTIME_INVALID, SUCCEEDED, TARGET_ABSENT_IN_PROCESSED_FRAMES,
    UNTRIED, ReplayOutcome, ReplayOutcomeMemory,
    classify_fresh_frame_outcome, make_equivalence_key,
    fresh_frame_gate_should_hold,
    minimum_processed_frames_from_success_evidence, replay_state_priority,
)


def candidate(**changes):
    value = {
        "candidate_id": "candidate-41", "marker_id": 41,
        "context_source": "OBSERVED", "context_quality": "CONTEXT_COMPLETE",
        "observed_base_pose": [1.8, 2.41, 1.57],
        "observed_head_pose": ["lower_center", 0.35, 0.0, -0.46],
        "observed_scan_station": {"index": 0},
        "observed_pose_name": "lower_center",
        "head_pose_hint": ["lower_center", 0.35, 0.0, -0.46],
        "target_bbox_summary": [491.0, 40.0, 639.0, 242.0],
        "marker_pixel_summary": [520.0, 270.5],
        "association_summary": {"bottom_center_distance_px": 53.2},
        "source_yolo_stamp_ns": 10, "source_aruco_stamp_ns": 10,
        "confidence": 0.97, "confirmations": 3,
    }
    value.update(changes)
    return value


def key(run="run-a", marker=41, **changes):
    return make_equivalence_key(
        run_prefix=run, kind="maidong", marker_id=marker,
        candidate=candidate(marker_id=marker, **changes))


def outcome(eq, attempt, pose, failure=TARGET_ABSENT_IN_PROCESSED_FRAMES):
    return ReplayOutcome(
        equivalence_key=eq, candidate_id=f"candidate-{eq.marker_id}",
        attempt_id=attempt, pose_id=pose, outcome="FAILED",
        failure_class=failure, fresh_rgb_count=6,
        yolo_processed_count=6, target_detection_count=0,
        attempt_start_s=1.0, attempt_end_s=2.0,
        material_context_revision=eq.digest)


class ReplayOutcomeMemoryTests(unittest.TestCase):
    def test_stamp_only_change_is_equivalent(self):
        self.assertEqual(key().digest, key(
            source_yolo_stamp_ns=20, source_aruco_stamp_ns=21).digest)

    def test_confidence_change_is_equivalent(self):
        self.assertEqual(key().digest, key(confidence=0.51).digest)

    def test_confirmation_growth_is_equivalent(self):
        self.assertEqual(key().digest, key(confirmations=99).digest)

    def test_pose_outside_frozen_tolerance_changes_material_context(self):
        self.assertNotEqual(key().digest, key(
            observed_base_pose=[1.86, 2.41, 1.57]).digest)
        self.assertNotEqual(key().digest, key(
            observed_head_pose=["lower_center", 0.366, 0.0, -0.46]).digest)

    def test_new_backup_plan_can_reactivate_once(self):
        old = key(); memory = ReplayOutcomeMemory("run-a")
        memory.record(outcome(old, "a1", "lower_center"))
        memory.record(outcome(old, "a2", "lower_yaw_minus"))
        self.assertEqual(memory.state(old), EXHAUSTED_UNTIL_MATERIAL_CHANGE)
        new = key(head_pose_hint=["overview_mid", 0.35, 0.0, -0.46],
                  observed_pose_name="overview_mid")
        self.assertTrue(memory.reactivate(old, new))
        self.assertEqual(memory.state(new), REACTIVATED)
        self.assertFalse(memory.reactivate(old, new))

    def test_primary_then_backup_exhaustion(self):
        eq = key(); memory = ReplayOutcomeMemory("run-a")
        memory.record(outcome(eq, "a1", "lower_center"))
        self.assertEqual(memory.state(eq), BACKUP_AVAILABLE)
        memory.record(outcome(eq, "a2", "lower_yaw_minus"))
        self.assertEqual(memory.state(eq), EXHAUSTED_UNTIL_MATERIAL_CHANGE)

    def test_exhausted_never_outranks_untried(self):
        self.assertLess(replay_state_priority(UNTRIED),
                        replay_state_priority(EXHAUSTED_UNTIL_MATERIAL_CHANGE))

    def test_complete_context_reactivates_derived_context(self):
        old = key(context_quality="DERIVED"); memory = ReplayOutcomeMemory("run-a")
        memory.record(outcome(old, "a1", "lower_center"))
        memory.record(outcome(old, "a2", "lower_yaw_minus"))
        new = key(context_quality="CONTEXT_COMPLETE")
        self.assertTrue(memory.reactivate(old, new))

    def test_no_fresh_rgb_does_not_penalize_semantics(self):
        eq = key(); memory = ReplayOutcomeMemory("run-a")
        memory.record(outcome(eq, "a1", "lower_center", NO_FRESH_RGB))
        self.assertEqual(memory.state(eq), RUNTIME_INVALID)
        self.assertTrue(memory.decision(eq)[0])

    def test_unprocessed_rgb_does_not_penalize_semantics(self):
        eq = key(); memory = ReplayOutcomeMemory("run-a")
        memory.record(outcome(
            eq, "a1", "lower_center", RGB_NOT_PROCESSED_BY_YOLO))
        self.assertEqual(memory.state(eq), RUNTIME_INVALID)

    def test_processed_fresh_frames_record_real_no_target(self):
        self.assertEqual(classify_fresh_frame_outcome(
            raw_fresh_rgb_count=6, yolo_processed_count=6,
            target_detection_count=0), TARGET_ABSENT_IN_PROCESSED_FRAMES)
        self.assertEqual(minimum_processed_frames_from_success_evidence(), 6)
        self.assertTrue(fresh_frame_gate_should_hold(
            target_detection_count=0, yolo_processed_count=5,
            pose_elapsed_s=29.0))
        self.assertFalse(fresh_frame_gate_should_hold(
            target_detection_count=0, yolo_processed_count=6,
            pose_elapsed_s=1.0))
        self.assertFalse(fresh_frame_gate_should_hold(
            target_detection_count=1, yolo_processed_count=0,
            pose_elapsed_s=1.0))
        self.assertEqual(classify_fresh_frame_outcome(
            raw_fresh_rgb_count=1, yolo_processed_count=1,
            target_detection_count=1), SUCCEEDED)

    def test_attempt_count_never_hard_fails(self):
        eq = key(); memory = ReplayOutcomeMemory("run-a")
        for attempt in range(100):
            memory.record(outcome(eq, f"a{attempt}", "lower_center",
                                  NO_FRESH_RGB))
        self.assertTrue(memory.decision(eq)[0])
        self.assertNotEqual(memory.state(eq), "FAILED_HARD")

    def test_run_prefix_isolation(self):
        memory = ReplayOutcomeMemory("run-a")
        with self.assertRaises(ValueError):
            memory.record(outcome(key(run="run-b"), "a", "lower_center"))

    def test_same_kind_different_marker_is_independent(self):
        self.assertNotEqual(key(marker=41).digest, key(marker=42).digest)

    def test_same_marker_different_material_context_is_independent(self):
        self.assertNotEqual(key().digest, key(
            observed_base_pose=[1.8, 2.5, 1.57]).digest)

    def test_deterministic_key_and_ordering(self):
        self.assertEqual(key().digest, key().digest)
        values=[EXHAUSTED_UNTIL_MATERIAL_CHANGE, REACTIVATED, UNTRIED]
        self.assertEqual(sorted(values, key=replay_state_priority),
                         [UNTRIED, REACTIVATED,
                          EXHAUSTED_UNTIL_MATERIAL_CHANGE])

    def test_no_kind_hardcoding_in_priority(self):
        self.assertEqual(replay_state_priority(UNTRIED), 0)
        self.assertNotIn("maidong", replay_state_priority.__code__.co_consts)

    def test_strict_threshold_file_is_byte_identical(self):
        protected = MODULE_DIR / "yolo_aruco_shelf_pick.py"
        digest = hashlib.sha256(protected.read_bytes()).hexdigest().upper()
        self.assertEqual(
            digest,
            "FD6BE07C15C32641D7431D89999F2FFBC1E23C7FA9E365609B06D76A3AA7714C")

    def test_success_is_terminal_without_attempt_count(self):
        eq = key(); memory = ReplayOutcomeMemory("run-a")
        item = outcome(eq, "a1", "lower_center")
        memory.record(ReplayOutcome(
            **{**item.__dict__, "outcome": SUCCEEDED, "failure_class": None}))
        self.assertEqual(memory.state(eq), SUCCEEDED)


if __name__ == "__main__":
    unittest.main()
