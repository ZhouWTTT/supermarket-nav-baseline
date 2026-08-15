import pathlib
import sys
import unittest


MODULE_DIR = pathlib.Path(__file__).resolve().parents[1] / "examples" / "supermarket_sorting"
sys.path.insert(0, str(MODULE_DIR))

from candidate_attempt_memory import (  # noqa: E402
    CandidateAttemptMemory,
    CandidateAttemptOutcome,
    completion_estimate,
    make_fingerprint,
    reactivation_requirements,
)


def candidate(**updates):
    value = {
        "candidate_id": "candidate-7", "marker_id": 7,
        "kind": "maidong", "confirmations": 3,
        "source_yolo_stamp_ns": 10, "source_aruco_stamp_ns": 11,
        "observed_source_stamps": [10, 11],
        "context_type": "OBSERVED_CONTEXT",
        "context_quality": "CONTEXT_COMPLETE",
        "observed_base_pose": [1.0, 2.0, 0.0],
        "observed_head_pose": ["overview_mid", 0.1, 0.0, -0.45],
        "observed_scan_station": {"index": 1},
        "observed_pose_name": "overview_mid",
        "head_pose_hint": ["overview_mid", 0.1, 0.0, -0.45],
    }
    value.update(updates)
    return value


def fingerprint(value=None, **updates):
    return make_fingerprint(
        run_prefix="run", order_id="order", product_kind="maidong",
        candidate=candidate(**updates) if value is None else value)


def fail(memory, fp, stage="target_kind_detection", reason="no_target_kind"):
    memory.record(CandidateAttemptOutcome(
        fingerprint=fp, failure_stage=stage, failure_reason=reason,
        terminal_s=10.0, evidence_revision=fp.candidate_evidence_revision,
        candidate_state_after_failure="INVALIDATED",
        reactivation_requirements=reactivation_requirements(stage, reason)))


class CandidateAttemptMemoryTests(unittest.TestCase):
    def test_same_fingerprint_without_new_evidence_is_suppressed(self):
        memory = CandidateAttemptMemory(); fp = fingerprint(); fail(memory, fp)
        decision = memory.decision(fp)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "same_fingerprint_no_new_evidence")

    def test_new_source_stamped_evidence_reactivates(self):
        memory = CandidateAttemptMemory(); old = fingerprint(); fail(memory, old)
        new = fingerprint(observed_source_stamps=[20, 21],
                          source_yolo_stamp_ns=20, source_aruco_stamp_ns=21)
        decision = memory.decision(new)
        self.assertTrue(decision.allowed); self.assertTrue(decision.new_evidence)
        self.assertIn("new_source_pair", decision.reason)

    def test_new_observed_context_reactivates(self):
        memory = CandidateAttemptMemory(); old = fingerprint(); fail(memory, old)
        new = fingerprint(observed_base_pose=[1.2, 2.0, 0.0])
        self.assertTrue(memory.decision(new).allowed)
        self.assertIn("new_context", memory.decision(new).reason)

    def test_untried_backup_pose_reactivates(self):
        memory = CandidateAttemptMemory(); old = fingerprint(); fail(memory, old)
        new = fingerprint(head_pose_hint=["lower_center", 0.45, 0.0, -0.45])
        self.assertTrue(memory.decision(new).allowed)
        self.assertIn("new_backup_pose", memory.decision(new).reason)

    def test_spread_reject_same_pose_and_evidence_is_suppressed(self):
        memory = CandidateAttemptMemory(); fp = fingerprint()
        fail(memory, fp, "sample_spread", "spread_reject")
        self.assertFalse(memory.decision(fp).allowed)

    def test_navigation_failure_requires_navigation_revision(self):
        memory = CandidateAttemptMemory(); fp = fingerprint()
        fail(memory, fp, "navigation", "no_path")
        self.assertFalse(memory.decision(fp).allowed)
        changed = make_fingerprint(
            run_prefix="run", order_id="order", product_kind="maidong",
            candidate=candidate(), navigation_request_revision=1)
        self.assertTrue(memory.decision(changed).allowed)
        self.assertIn("navigation_revision_changed", memory.decision(changed).reason)

    def test_worker_crash_does_not_punish_candidate_semantics(self):
        memory = CandidateAttemptMemory(); fp = fingerprint()
        fail(memory, fp, "process", "worker_crash")
        self.assertTrue(memory.decision(fp).allowed)

    def test_sensor_invalid_requires_recovery_not_stamp_growth(self):
        self.assertEqual(
            reactivation_requirements("yolo_processing",
                                      "rgb_not_processed_by_yolo"),
            ("sensor_recovered",))

    def test_duplicate_outcome_is_idempotent(self):
        memory = CandidateAttemptMemory(); fp = fingerprint()
        outcome = CandidateAttemptOutcome(
            fp, "association", "no_association", 10.0,
            fp.candidate_evidence_revision, "INVALIDATED", ())
        self.assertTrue(memory.record(outcome)); self.assertFalse(memory.record(outcome))
        self.assertEqual(len(memory.outcomes), 1)

    def test_attempt_number_is_not_in_fingerprint(self):
        self.assertEqual(fingerprint().digest, fingerprint().digest)

    def test_deadline_estimate_uses_real_stage_summary_and_negative_slack(self):
        estimate = completion_estimate(True)
        self.assertIn("R8-B", estimate.estimate_source)
        self.assertEqual(estimate.estimated_completion_s, 233.0)
        self.assertTrue(estimate.feasibility(233.0)["deadline_feasible"])
        self.assertEqual(estimate.feasibility(200.0)["deadline_slack_s"], -33.0)


if __name__ == "__main__":
    unittest.main()
