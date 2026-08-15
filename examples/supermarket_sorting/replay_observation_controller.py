#!/usr/bin/env python3
"""Evidence-driven observation timing for provisional candidate replay.

This module is deliberately independent of ROS and perception.  It can only
decide whether the current replay pose should be held or advanced; the strict
localizer remains the sole authority for accepting samples and localization.
"""

from __future__ import annotations

from dataclasses import dataclass


HOLD = "hold"
ADVANCE = "advance"


@dataclass(frozen=True)
class ReplayObservationSnapshot:
    """Read-only progress counters produced by the existing localizer."""

    target_kind_detection_count: int = 0
    aruco_detection_count: int = 0
    fresh_synchronized_pair_count: int = 0
    duplicate_count: int = 0
    freshness_rejection_count: int = 0
    association_candidate_count: int = 0
    association_confirmation_count: int = 0
    association_success_rate: float = 0.0
    accepted_sample_count: int = 0
    localized: bool = False


@dataclass(frozen=True)
class ReplayObservationDecision:
    action: str
    reason: str
    pose_elapsed_s: float
    progress_age_s: float


class ReplayObservationController:
    """Hold a productive pose and advance a stalled candidate replay pose."""

    def __init__(
            self,
            *,
            required_samples: int,
            min_wait_s: float = 0.8,
            max_wait_s: float = 6.0,
            progress_grace_s: float = 1.0):
        if required_samples <= 0:
            raise ValueError("required_samples must be positive")
        if not 0.0 <= min_wait_s < max_wait_s:
            raise ValueError("require 0 <= min_wait_s < max_wait_s")
        if progress_grace_s <= 0.0:
            raise ValueError("progress_grace_s must be positive")
        self.required_samples = int(required_samples)
        self.min_wait_s = float(min_wait_s)
        self.max_wait_s = float(max_wait_s)
        self.progress_grace_s = float(progress_grace_s)
        self.pose_started_at: float | None = None
        self.last_progress_at: float | None = None
        self.last_snapshot = ReplayObservationSnapshot()

    def start_pose(
            self, now_s: float,
            snapshot: ReplayObservationSnapshot | None = None) -> None:
        self.pose_started_at = float(now_s)
        self.last_progress_at = float(now_s)
        self.last_snapshot = snapshot or ReplayObservationSnapshot()

    def observation_budget_available(self, now_s: float) -> bool:
        """Whether the current pose still owns adaptive observation time."""
        if self.pose_started_at is None:
            return False
        return float(now_s) - self.pose_started_at < self.max_wait_s

    @staticmethod
    def _progress_key(snapshot: ReplayObservationSnapshot) -> tuple[int, ...]:
        return (
            snapshot.target_kind_detection_count,
            snapshot.aruco_detection_count,
            snapshot.fresh_synchronized_pair_count,
            snapshot.association_candidate_count,
            snapshot.association_confirmation_count,
            snapshot.accepted_sample_count,
        )

    def observe(
            self, now_s: float,
            snapshot: ReplayObservationSnapshot) -> ReplayObservationDecision:
        now_s = float(now_s)
        if self.pose_started_at is None:
            self.start_pose(now_s, snapshot)
        assert self.pose_started_at is not None
        assert self.last_progress_at is not None

        current_progress = self._progress_key(snapshot)
        previous_progress = self._progress_key(self.last_snapshot)
        if any(current > previous for current, previous in zip(
                current_progress, previous_progress)):
            self.last_progress_at = now_s
        self.last_snapshot = snapshot

        elapsed = max(0.0, now_s - self.pose_started_at)
        progress_age = max(0.0, now_s - self.last_progress_at)

        if snapshot.localized:
            return ReplayObservationDecision(
                HOLD, "authoritative_localization", elapsed, progress_age)
        if elapsed < self.min_wait_s:
            return ReplayObservationDecision(
                HOLD, "minimum_observation", elapsed, progress_age)
        if elapsed >= self.max_wait_s:
            return ReplayObservationDecision(
                ADVANCE, "max_observation_budget", elapsed, progress_age)

        productive = (
            snapshot.association_success_rate > 0.0
            or 0 < snapshot.association_confirmation_count
            or 0 < snapshot.accepted_sample_count < self.required_samples)
        if productive and progress_age <= self.progress_grace_s:
            reason = (
                "collecting_accepted_samples"
                if snapshot.accepted_sample_count > 0
                else "building_association_confirmations")
            return ReplayObservationDecision(HOLD, reason, elapsed, progress_age)

        if snapshot.target_kind_detection_count <= 0:
            reason = "no_target_kind"
        elif snapshot.aruco_detection_count <= 0:
            reason = "no_aruco"
        elif snapshot.fresh_synchronized_pair_count <= 0:
            reason = (
                "freshness_or_sync_rejections"
                if snapshot.freshness_rejection_count > 0
                else "no_fresh_synchronized_pair")
        elif snapshot.association_success_rate <= 0.0:
            reason = "no_association"
        elif snapshot.accepted_sample_count < self.required_samples:
            reason = "accepted_sample_collection_stalled"
        else:
            # The strict localizer has enough samples but has not accepted
            # them.  Do not second-guess its spread/bounds decision.
            reason = "strict_localization_not_accepted"
        return ReplayObservationDecision(ADVANCE, reason, elapsed, progress_age)
