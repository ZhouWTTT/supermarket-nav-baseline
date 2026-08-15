import pathlib
import sys
import unittest


MODULE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

from replay_observation_controller import (  # noqa: E402
    ADVANCE,
    HOLD,
    ReplayObservationController,
    ReplayObservationSnapshot,
)


def snapshot(**values):
    return ReplayObservationSnapshot(**values)


class ReplayObservationControllerTests(unittest.TestCase):
    def make_controller(self):
        return ReplayObservationController(
            required_samples=5,
            min_wait_s=0.8,
            max_wait_s=6.0,
            progress_grace_s=1.0)

    def test_holds_minimum_observation_then_advances_no_signal(self):
        controller = self.make_controller()
        controller.start_pose(10.0)
        self.assertEqual(controller.observe(10.79, snapshot()).action, HOLD)
        decision = controller.observe(10.8, snapshot())
        self.assertEqual(decision.action, ADVANCE)
        self.assertEqual(decision.reason, "no_target_kind")

    def test_association_and_sample_progress_extend_pose(self):
        controller = self.make_controller()
        controller.start_pose(0.0)
        building = snapshot(
            target_kind_detection_count=3,
            aruco_detection_count=9,
            fresh_synchronized_pair_count=2,
            association_candidate_count=1,
            association_success_rate=0.5,
            association_confirmation_count=2)
        self.assertEqual(controller.observe(0.9, building).reason,
                         "building_association_confirmations")
        collecting = snapshot(
            target_kind_detection_count=5,
            aruco_detection_count=15,
            fresh_synchronized_pair_count=4,
            association_candidate_count=3,
            association_success_rate=0.75,
            association_confirmation_count=3,
            accepted_sample_count=3)
        decision = controller.observe(1.7, collecting)
        self.assertEqual(decision.action, HOLD)
        self.assertEqual(decision.reason, "collecting_accepted_samples")

    def test_stalled_partial_samples_advance(self):
        controller = self.make_controller()
        controller.start_pose(0.0)
        partial = snapshot(
            target_kind_detection_count=5,
            aruco_detection_count=12,
            fresh_synchronized_pair_count=4,
            association_candidate_count=3,
            association_success_rate=0.75,
            association_confirmation_count=3,
            accepted_sample_count=3)
        controller.observe(0.9, partial)
        decision = controller.observe(2.0, partial)
        self.assertEqual(decision.action, ADVANCE)
        self.assertEqual(decision.reason,
                         "accepted_sample_collection_stalled")

    def test_duplicate_polls_do_not_extend_progress(self):
        controller = self.make_controller()
        controller.start_pose(0.0)
        initial = snapshot(
            target_kind_detection_count=2,
            aruco_detection_count=8,
            fresh_synchronized_pair_count=1,
            duplicate_count=1,
            association_candidate_count=1,
            association_success_rate=1.0,
            association_confirmation_count=1)
        controller.observe(0.8, initial)
        duplicate_only = snapshot(
            target_kind_detection_count=2,
            aruco_detection_count=8,
            fresh_synchronized_pair_count=1,
            duplicate_count=50,
            association_candidate_count=1,
            association_success_rate=1.0,
            association_confirmation_count=1)
        decision = controller.observe(1.81, duplicate_only)
        self.assertEqual(decision.action, ADVANCE)
        self.assertEqual(decision.reason,
                         "accepted_sample_collection_stalled")

    def test_freshness_rejection_has_specific_reason(self):
        controller = self.make_controller()
        controller.start_pose(0.0)
        decision = controller.observe(0.8, snapshot(
            target_kind_detection_count=2,
            aruco_detection_count=8,
            freshness_rejection_count=3))
        self.assertEqual(decision.action, ADVANCE)
        self.assertEqual(decision.reason, "freshness_or_sync_rejections")

    def test_zero_association_success_rate_advances(self):
        controller = self.make_controller()
        controller.start_pose(0.0)
        decision = controller.observe(0.8, snapshot(
            target_kind_detection_count=2,
            aruco_detection_count=8,
            fresh_synchronized_pair_count=3,
            association_candidate_count=1,
            association_success_rate=0.0))
        self.assertEqual(decision.action, ADVANCE)
        self.assertEqual(decision.reason, "no_association")

    def test_max_budget_advances_even_with_recent_progress(self):
        controller = self.make_controller()
        controller.start_pose(0.0)
        decision = controller.observe(6.0, snapshot(
            target_kind_detection_count=8,
            aruco_detection_count=20,
            fresh_synchronized_pair_count=8,
            association_candidate_count=6,
            association_success_rate=0.75,
            association_confirmation_count=3,
            accepted_sample_count=4))
        self.assertEqual(decision.action, ADVANCE)
        self.assertEqual(decision.reason, "max_observation_budget")

    def test_observation_budget_availability_is_bounded(self):
        controller = self.make_controller()
        self.assertFalse(controller.observation_budget_available(0.0))
        controller.start_pose(10.0)
        self.assertTrue(controller.observation_budget_available(15.999))
        self.assertFalse(controller.observation_budget_available(16.0))

    def test_authoritative_localization_is_never_overridden(self):
        controller = self.make_controller()
        controller.start_pose(0.0)
        decision = controller.observe(7.0, snapshot(
            accepted_sample_count=5, localized=True))
        self.assertEqual(decision.action, HOLD)
        self.assertEqual(decision.reason, "authoritative_localization")


if __name__ == "__main__":
    unittest.main()
