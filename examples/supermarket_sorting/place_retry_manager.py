"""Fail-closed candidate management for delivery-table placement.

This module is deliberately independent from ROS and the manipulation
controller.  It owns only deterministic candidate ordering, numerical
deduplication, and the policy decision to activate another pre-validated IK
candidate.  The caller remains responsible for stopping the base, holding the
gripper closed, validating the candidate against the unchanged delivery-table
envelope, and sending arm/slide commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Iterable, Optional, Sequence, Tuple


Vector = Tuple[float, ...]


def _float_tuple(value: Sequence[float]) -> Vector:
    return tuple(float(component) for component in value)


@dataclass(frozen=True)
class PlaceIKCandidate:
    """One complete approach/release command pair.

    Candidate coordinates describe the same pre-existing safe release region;
    they may differ only in how the arm reaches that region.
    """

    approach_world_pose: Vector
    arm_joints: Vector
    slide_target: float
    release_world_pose: Vector
    release_slide: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "approach_world_pose",
            _float_tuple(self.approach_world_pose))
        object.__setattr__(self, "arm_joints", _float_tuple(self.arm_joints))
        object.__setattr__(self, "slide_target", float(self.slide_target))
        object.__setattr__(
            self, "release_world_pose",
            _float_tuple(self.release_world_pose))
        object.__setattr__(self, "release_slide", float(self.release_slide))

    @property
    def finite(self) -> bool:
        values = (
            self.approach_world_pose + self.arm_joints
            + (self.slide_target,) + self.release_world_pose
            + (self.release_slide,))
        return all(math.isfinite(value) for value in values)


class PlaceFailureReason(str, Enum):
    ARM_SETTLE_TIMEOUT = "arm_settle_timeout"
    IK_CANDIDATE_INVALID = "ik_candidate_invalid"
    TEMPORARY_CONTROLLER_TIMEOUT = "temporary_controller_timeout"
    UNSAFE_TCP = "unsafe_tcp"
    TABLETOP_COLLISION = "tabletop_collision"
    INVALID_RELEASE_POSE = "invalid_release_pose"
    CANDIDATES_EXHAUSTED = "candidates_exhausted"


RETRYABLE_FAILURES = frozenset({
    PlaceFailureReason.ARM_SETTLE_TIMEOUT,
    PlaceFailureReason.IK_CANDIDATE_INVALID,
    PlaceFailureReason.TEMPORARY_CONTROLLER_TIMEOUT,
})

NON_RETRYABLE_SAFETY_FAILURES = frozenset({
    PlaceFailureReason.UNSAFE_TCP,
    PlaceFailureReason.TABLETOP_COLLISION,
    PlaceFailureReason.INVALID_RELEASE_POSE,
})


class RetryDisposition(str, Enum):
    ACTIVATE_CANDIDATE = "activate_candidate"
    RECOVERABLE_FAILURE = "recoverable_failure"
    FATAL_SAFETY_FAILURE = "fatal_safety_failure"


@dataclass(frozen=True)
class PlaceRetryDecision:
    disposition: RetryDisposition
    candidate: Optional[PlaceIKCandidate]
    candidate_index: Optional[int]
    reason: Optional[PlaceFailureReason]
    detail: str
    stop_base: bool = True
    hold_gripper_closed: bool = True

    @property
    def should_activate(self) -> bool:
        return self.disposition is RetryDisposition.ACTIVATE_CANDIDATE

    @property
    def recoverable(self) -> bool:
        return self.disposition is RetryDisposition.RECOVERABLE_FAILURE


class RecoverablePlaceFailure(RuntimeError):
    """Placement stopped safely and may be retried by the order runner."""

    def __init__(self, blocker: str, detail: str):
        self.blocker = str(blocker)
        self.detail = str(detail)
        super().__init__(f"{self.blocker}: {self.detail}")


def _vectors_close(left: Vector, right: Vector, tolerance: float) -> bool:
    return (
        len(left) == len(right)
        and all(abs(a - b) <= tolerance for a, b in zip(left, right)))


def candidates_equivalent(
        left: PlaceIKCandidate,
        right: PlaceIKCandidate,
        *,
        pose_tolerance: float = 1e-5,
        joint_tolerance: float = 1e-4,
        slide_tolerance: float = 1e-5) -> bool:
    """Return whether two candidates would issue the same safe command."""
    return (
        _vectors_close(
            left.approach_world_pose, right.approach_world_pose,
            pose_tolerance)
        and _vectors_close(
            left.arm_joints, right.arm_joints, joint_tolerance)
        and abs(left.slide_target - right.slide_target) <= slide_tolerance
        and _vectors_close(
            left.release_world_pose, right.release_world_pose,
            pose_tolerance)
        and abs(left.release_slide - right.release_slide)
        <= slide_tolerance)


def ordered_unique_candidates(
        candidates: Iterable[PlaceIKCandidate]) -> Tuple[PlaceIKCandidate, ...]:
    """Deduplicate candidates without changing planner preference order."""
    result = []
    for candidate in candidates:
        if not any(candidates_equivalent(candidate, item) for item in result):
            result.append(candidate)
    return tuple(result)


CandidateValidator = Callable[[PlaceIKCandidate], bool]


class PlaceRetryManager:
    """Deterministic, fail-closed place candidate retry policy."""

    def __init__(
            self,
            candidates: Iterable[PlaceIKCandidate],
            validator: CandidateValidator,
            *,
            max_retries: int = 3):
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._candidates = ordered_unique_candidates(candidates)
        self._validator = validator
        self._max_retries = int(max_retries)
        self._next_index = 0
        self._active_index: Optional[int] = None
        self._retry_count = 0

    @property
    def candidates(self) -> Tuple[PlaceIKCandidate, ...]:
        return self._candidates

    @property
    def active_candidate(self) -> Optional[PlaceIKCandidate]:
        if self._active_index is None:
            return None
        return self._candidates[self._active_index]

    @property
    def retry_count(self) -> int:
        return self._retry_count

    def start(self) -> PlaceRetryDecision:
        """Validate and activate the first candidate in planner order."""
        return self._next_valid_candidate(
            PlaceFailureReason.IK_CANDIDATE_INVALID)

    def retry(self, reason: PlaceFailureReason) -> PlaceRetryDecision:
        """Return the next safe action for a classified placement failure."""
        reason = PlaceFailureReason(reason)
        if reason in NON_RETRYABLE_SAFETY_FAILURES:
            return PlaceRetryDecision(
                RetryDisposition.FATAL_SAFETY_FAILURE,
                None,
                self._active_index,
                reason,
                "safety failure forbids candidate retry")
        if reason not in RETRYABLE_FAILURES:
            raise ValueError(f"unsupported retry reason: {reason}")
        if self._retry_count >= self._max_retries:
            return self._exhausted(reason, "place retry budget exhausted")
        self._retry_count += 1
        return self._next_valid_candidate(reason)

    def _next_valid_candidate(
            self, reason: PlaceFailureReason) -> PlaceRetryDecision:
        while self._next_index < len(self._candidates):
            index = self._next_index
            candidate = self._candidates[index]
            self._next_index += 1
            try:
                valid = bool(self._validator(candidate))
            except Exception:  # Validator failures are candidate-local.
                valid = False
            if not valid:
                continue
            self._active_index = index
            return PlaceRetryDecision(
                RetryDisposition.ACTIVATE_CANDIDATE,
                candidate,
                index,
                reason,
                "candidate validated for activation")
        return self._exhausted(reason, "no validated place candidate remains")

    def _exhausted(
            self, source_reason: PlaceFailureReason,
            detail: str) -> PlaceRetryDecision:
        return PlaceRetryDecision(
            RetryDisposition.RECOVERABLE_FAILURE,
            None,
            self._active_index,
            PlaceFailureReason.CANDIDATES_EXHAUSTED,
            f"{detail}; source_reason={source_reason.value}")


__all__ = [
    "NON_RETRYABLE_SAFETY_FAILURES",
    "PlaceFailureReason",
    "PlaceIKCandidate",
    "PlaceRetryDecision",
    "PlaceRetryManager",
    "RecoverablePlaceFailure",
    "RETRYABLE_FAILURES",
    "RetryDisposition",
    "candidates_equivalent",
    "ordered_unique_candidates",
]
