"""Bounded, primitive-only telemetry for provisional candidate admission."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


INVENTORY_SYNC_TOLERANCE_NS = 200_000_000


def nearest_synchronized_frame(target_stamp: int, frames, *, tolerance_ns: int):
    """Return the nearest stamped frame without relaxing the sync gate."""
    if not frames:
        return None
    nearest = min(frames, key=lambda frame: abs(int(frame[0]) - target_stamp))
    if abs(int(nearest[0]) - target_stamp) > int(tolerance_ns):
        return None
    return nearest


@dataclass
class PoseFunnel:
    attempt_id: str
    station_id: int
    pose_name: str
    shelf_band: str
    pending_kinds: tuple[str, ...]
    fresh_rgb_stamps: set[int] = field(default_factory=set)
    fresh_aruco_stamps: set[int] = field(default_factory=set)
    yolo_detection_count_by_kind: Counter = field(default_factory=Counter)
    aruco_detection_count: int = 0
    aruco_seen_ids: set[int] = field(default_factory=set)
    synchronized_frame_pair_count: int = 0
    pair_desync_reject_count: int = 0
    association_attempt_count: int = 0
    association_success_count: int = 0
    association_reject_reason_counts: Counter = field(default_factory=Counter)
    duplicate_pair_count: int = 0
    confirmation_reset_count: int = 0
    confirmation_reset_reasons: Counter = field(default_factory=Counter)
    threshold_reached_count: int = 0
    candidate_constructed_count: int = 0
    candidate_received_by_runner_count: int = 0
    candidate_inserted_into_inventory_count: int = 0
    candidate_rejected_reason_counts: Counter = field(default_factory=Counter)
    seen_pair_keys: set[tuple[int, int]] = field(default_factory=set)

    def summary(self) -> dict:
        return {
            "attempt_id": self.attempt_id,
            "station_id": self.station_id,
            "pose_name": self.pose_name,
            "shelf_band": self.shelf_band,
            "pending_task_kinds": list(self.pending_kinds),
            "fresh_rgb_frame_count": len(self.fresh_rgb_stamps),
            "fresh_aruco_frame_count": len(self.fresh_aruco_stamps),
            "yolo_detection_count_by_kind": dict(
                sorted(self.yolo_detection_count_by_kind.items())),
            "target_task_kind_detection_count": sum(
                self.yolo_detection_count_by_kind[kind]
                for kind in self.pending_kinds),
            "aruco_detection_count": self.aruco_detection_count,
            "aruco_seen_ids": sorted(self.aruco_seen_ids)[:45],
            "synchronized_frame_pair_count": self.synchronized_frame_pair_count,
            "pair_desync_reject_count": self.pair_desync_reject_count,
            "association_attempt_count": self.association_attempt_count,
            "association_success_count": self.association_success_count,
            "association_reject_reason_counts": dict(
                sorted(self.association_reject_reason_counts.items())),
            "duplicate_pair_count": self.duplicate_pair_count,
            "confirmation_reset_count": self.confirmation_reset_count,
            "confirmation_reset_reasons": dict(
                sorted(self.confirmation_reset_reasons.items())),
            "threshold_reached_count": self.threshold_reached_count,
            "candidate_constructed_count": self.candidate_constructed_count,
            "candidate_received_by_runner_count": (
                self.candidate_received_by_runner_count),
            "candidate_inserted_into_inventory_count": (
                self.candidate_inserted_into_inventory_count),
            "candidate_rejected_reason_counts": dict(
                sorted(self.candidate_rejected_reason_counts.items())),
        }


class CandidateAdmissionTrace:
    """Retain at most ``max_poses`` completed pose summaries per run."""

    def __init__(self, run_prefix: str, *, max_poses: int = 8):
        self.run_prefix = str(run_prefix)
        self.max_poses = int(max_poses)
        self.active: dict[str, PoseFunnel] = {}
        self.completed_count = 0

    def start_pose(
            self, *, attempt_id: str, station_id: int, pose_name: str,
            shelf_band: str, pending_kinds) -> PoseFunnel | None:
        if self.completed_count >= self.max_poses:
            return None
        pose = PoseFunnel(
            str(attempt_id), int(station_id), str(pose_name), str(shelf_band),
            tuple(sorted({str(kind) for kind in pending_kinds})))
        self.active[str(attempt_id)] = pose
        return pose

    def current(self, attempt_id: str | None) -> PoseFunnel | None:
        return None if attempt_id is None else self.active.get(str(attempt_id))

    def end_pose(self, attempt_id: str | None) -> dict | None:
        if attempt_id is None:
            return None
        pose = self.active.pop(str(attempt_id), None)
        if pose is None:
            return None
        self.completed_count += 1
        return pose.summary()


def first_loss_stage(summary: dict) -> str:
    """Classify the earliest failed funnel stage from one pose aggregate."""
    if int(summary.get("target_task_kind_detection_count", 0)) <= 0:
        return "NO_TARGET_DETECTION"
    if int(summary.get("aruco_detection_count", 0)) <= 0:
        return "NO_ARUCO"
    if int(summary.get("synchronized_frame_pair_count", 0)) <= 0:
        return "NO_SYNCHRONIZED_PAIR"
    if int(summary.get("association_success_count", 0)) <= 0:
        return "NO_ASSOCIATION"
    if int(summary.get("confirmation_reset_count", 0)) > 0:
        return "CONFIRMATION_RESET"
    if int(summary.get("candidate_constructed_count", 0)) <= 0:
        return "CANDIDATE_NOT_CONSTRUCTED"
    if int(summary.get("candidate_received_by_runner_count", 0)) <= 0:
        return "RUNNER_EVENT_NOT_RECEIVED"
    if int(summary.get("candidate_inserted_into_inventory_count", 0)) <= 0:
        return "INVENTORY_INSERT_REJECTED"
    return "ADMITTED"
