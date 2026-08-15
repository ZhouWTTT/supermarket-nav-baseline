from __future__ import annotations

from pathlib import Path
import sys


MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples" / "supermarket_sorting")
sys.path.insert(0, str(MODULE_DIR))

from place_retry_manager import (  # noqa: E402
    PlaceFailureReason,
    PlaceIKCandidate,
    PlaceRetryManager,
    RetryDisposition,
    ordered_unique_candidates,
)


def candidate(index: int, *, delta: float = 0.0) -> PlaceIKCandidate:
    return PlaceIKCandidate(
        approach_world_pose=(-1.8, -3.20 - 0.01 * index, 0.86),
        arm_joints=(index + delta, 0.1, 0.2, 0.3, 0.4, 0.5),
        slide_target=0.20,
        release_world_pose=(-1.8, -3.20 - 0.01 * index, 0.82),
        release_slide=0.24,
    )


def test_candidate_order_preserves_planner_priority():
    planned = (candidate(2), candidate(0), candidate(1))
    manager = PlaceRetryManager(planned, lambda _candidate: True)

    first = manager.start()
    second = manager.retry(PlaceFailureReason.ARM_SETTLE_TIMEOUT)
    third = manager.retry(PlaceFailureReason.ARM_SETTLE_TIMEOUT)

    assert first.candidate == planned[0]
    assert second.candidate == planned[1]
    assert third.candidate == planned[2]


def test_candidate_deduplication_is_tolerant_and_stable():
    first = candidate(0)
    numerical_duplicate = candidate(0, delta=5e-5)
    distinct = candidate(1)

    unique = ordered_unique_candidates(
        (first, numerical_duplicate, distinct, first))

    assert unique == (first, distinct)


def test_settle_timeout_switches_to_revalidated_candidate():
    validation_calls = []

    def validate(value):
        validation_calls.append(value)
        return True

    first, second = candidate(0), candidate(1)
    manager = PlaceRetryManager((first, second), validate)
    assert manager.start().candidate == first

    decision = manager.retry(PlaceFailureReason.ARM_SETTLE_TIMEOUT)

    assert decision.disposition is RetryDisposition.ACTIVATE_CANDIDATE
    assert decision.candidate == second
    assert manager.retry_count == 1
    assert validation_calls == [first, second]


def test_retry_contract_holds_gripper_closed_and_stops_base():
    manager = PlaceRetryManager(
        (candidate(0), candidate(1)), lambda _candidate: True)
    manager.start()

    decision = manager.retry(
        PlaceFailureReason.TEMPORARY_CONTROLLER_TIMEOUT)

    assert decision.should_activate
    assert decision.hold_gripper_closed is True
    assert decision.stop_base is True


def test_candidate_exhaustion_returns_recoverable_failure():
    manager = PlaceRetryManager((candidate(0),), lambda _candidate: True)
    manager.start()

    decision = manager.retry(PlaceFailureReason.ARM_SETTLE_TIMEOUT)

    assert decision.disposition is RetryDisposition.RECOVERABLE_FAILURE
    assert decision.recoverable is True
    assert decision.reason is PlaceFailureReason.CANDIDATES_EXHAUSTED
    assert decision.candidate is None
    assert decision.hold_gripper_closed is True
    assert decision.stop_base is True


def test_unsafe_pose_rejects_retry_without_advancing_candidate():
    first, second = candidate(0), candidate(1)
    manager = PlaceRetryManager((first, second), lambda _candidate: True)
    manager.start()

    decision = manager.retry(PlaceFailureReason.UNSAFE_TCP)

    assert decision.disposition is RetryDisposition.FATAL_SAFETY_FAILURE
    assert decision.recoverable is False
    assert decision.candidate is None
    assert manager.active_candidate == first
    assert manager.retry_count == 0
