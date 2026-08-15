"""Pure run-scoped scan coverage, cursor, and deadline semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable


UNSEEN = "UNSEEN"
PARTIAL = "PARTIAL"
COVERED_VALID = "COVERED_VALID"
INVALIDATED = "INVALIDATED"


@dataclass(frozen=True, order=True)
class CoverageKey:
    run_prefix: str
    station_id: int
    pose_name: str
    shelf_band: str


@dataclass
class CoverageRecord:
    key: CoverageKey
    coverage_generation: int
    state: str = UNSEEN
    scan_started_at: float | None = None
    scan_completed_at: float | None = None
    fresh_rgb_frame_count: int = 0
    fresh_aruco_frame_count: int = 0
    observed_kinds: list[str] = field(default_factory=list)
    candidate_ids_created: list[str] = field(default_factory=list)
    completion_reason: str | None = None
    estimated_duration_s: float | None = None
    actual_duration_s: float | None = None

    def as_dict(self) -> dict:
        result = asdict(self)
        result.update(asdict(self.key))
        result.pop("key")
        return result


def shelf_band_for_pose(pose_name: str) -> str:
    name = str(pose_name)
    if name == "overview_high":
        return "top"
    if name.startswith("lower"):
        return "lower"
    return "middle"


def stable_attempt_id(run_prefix: str, order_id: str, ordinal: int) -> str:
    return f"{run_prefix}:{order_id}:attempt-{int(ordinal)}"


def deadline_action(
        elapsed_s: float, phase: str | None, *,
        soft_deadline_s: float = 570.0,
        hard_deadline_s: float = 600.0) -> str:
    """Return ALLOW_NEW, ALLOW_INFLIGHT, BLOCK_NEW, or HARD_STOP."""
    if elapsed_s >= hard_deadline_s:
        return "HARD_STOP"
    if elapsed_s < soft_deadline_s:
        return "ALLOW_NEW"
    if phase in {
            "GRASPED", "BACKUP", "NAV_TO_DELIVERY", "DELIVERING",
            "PLACE_APPROACH", "PLACING", "PLACE_RETRY"}:
        return "ALLOW_INFLIGHT"
    return "BLOCK_NEW"


class RunScanCoverage:
    """Coverage authority isolated to exactly one run prefix."""

    def __init__(
            self, run_prefix: str,
            route: Iterable[tuple[int, str, str]], *, generation: int = 1):
        self.run_prefix = str(run_prefix)
        self.coverage_generation = int(generation)
        self.route = tuple(CoverageKey(
            self.run_prefix, int(station), str(pose), str(band))
            for station, pose, band in route)
        self.records = {
            key: CoverageRecord(key, self.coverage_generation)
            for key in self.route
        }
        self.cursor_index = 0
        self.initial_discovery_segments = 0
        self.resumed_uncovered_scan_segments = 0
        self.coverage_reuse_count = 0
        self.repeated_covered_pose_count = 0

    def _owned(self, key: CoverageKey) -> bool:
        return key.run_prefix == self.run_prefix and key in self.records

    def start(
            self, key: CoverageKey, *, stamp: float, resumed: bool,
            estimated_duration_s: float | None = None) -> bool:
        if not self._owned(key):
            return False
        record = self.records[key]
        if record.state == COVERED_VALID:
            self.repeated_covered_pose_count += 1
            return False
        record.state = PARTIAL
        record.scan_started_at = float(stamp)
        record.scan_completed_at = None
        record.actual_duration_s = None
        record.estimated_duration_s = (
            None if estimated_duration_s is None else
            max(0.0, float(estimated_duration_s)))
        if resumed:
            self.resumed_uncovered_scan_segments += 1
        else:
            self.initial_discovery_segments += 1
        return True

    def complete(
            self, key: CoverageKey, *, stamp: float,
            fresh_rgb_frame_count: int, fresh_aruco_frame_count: int,
            observed_kinds: Iterable[str] = (),
            candidate_ids_created: Iterable[str] = (),
            completion_reason: str,
            camera_settled: bool, pose_completed: bool,
            interrupted: bool = False) -> bool:
        if not self._owned(key):
            return False
        record = self.records[key]
        record.fresh_rgb_frame_count = max(0, int(fresh_rgb_frame_count))
        record.fresh_aruco_frame_count = max(0, int(fresh_aruco_frame_count))
        record.observed_kinds = sorted({str(item) for item in observed_kinds})
        record.candidate_ids_created = sorted({
            str(item) for item in candidate_ids_created})
        record.completion_reason = str(completion_reason)
        if record.scan_started_at is not None:
            record.actual_duration_s = max(
                0.0, float(stamp) - record.scan_started_at)
        valid = (
            not interrupted and camera_settled and pose_completed
            and record.fresh_rgb_frame_count > 0
            and record.fresh_aruco_frame_count > 0)
        record.state = COVERED_VALID if valid else PARTIAL
        record.scan_completed_at = float(stamp) if valid else None
        if valid:
            # Advance only through a contiguous covered prefix.  A PARTIAL
            # earlier key remains the resume point even if a later callback
            # arrives first.
            while (self.cursor_index < len(self.route)
                   and self.records[self.route[self.cursor_index]].state
                   == COVERED_VALID):
                self.cursor_index += 1
        return valid

    def invalidate(self, key: CoverageKey, reason: str) -> None:
        if self._owned(key):
            record = self.records[key]
            record.state = INVALIDATED
            record.completion_reason = str(reason)

    def reusable(self, key: CoverageKey) -> bool:
        if self._owned(key) and self.records[key].state == COVERED_VALID:
            self.coverage_reuse_count += 1
            return True
        return False

    def needs_scan(self, key: CoverageKey) -> bool:
        return self._owned(key) and self.records[key].state != COVERED_VALID

    def next_uncovered_key(self) -> CoverageKey | None:
        if not self.route:
            return None
        # Cursor is monotonic within one run.  Returning to route start would
        # silently turn resume into another full discovery pass.
        ordered = self.route[self.cursor_index:]
        return next((key for key in ordered if self.needs_scan(key)), None)

    def uncovered_keys_from_cursor(self) -> tuple[CoverageKey, ...]:
        return tuple(key for key in self.route[self.cursor_index:]
                     if self.needs_scan(key))

    def snapshot(self) -> dict:
        next_key = self.next_uncovered_key()
        return {
            "run_prefix": self.run_prefix,
            "coverage_generation": self.coverage_generation,
            "cursor_index": self.cursor_index,
            "next_uncovered_key": (None if next_key is None else asdict(next_key)),
            "records": [self.records[key].as_dict() for key in self.route],
            "metrics": {
                "initial_discovery_segments": self.initial_discovery_segments,
                "resumed_uncovered_scan_segments": (
                    self.resumed_uncovered_scan_segments),
                "covered_pose_count": sum(
                    record.state == COVERED_VALID
                    for record in self.records.values()),
                "partial_pose_count": sum(
                    record.state == PARTIAL
                    for record in self.records.values()),
                "repeated_covered_pose_count": self.repeated_covered_pose_count,
                "coverage_reuse_count": self.coverage_reuse_count,
                "discovery_segments_started": sum(
                    record.scan_started_at is not None
                    for record in self.records.values()),
                "discovery_segments_completed": sum(
                    record.state == COVERED_VALID
                    for record in self.records.values()),
                "discovery_segments_producing_candidates": sum(
                    bool(record.candidate_ids_created)
                    for record in self.records.values()),
            },
        }

    @classmethod
    def from_snapshot(
            cls, snapshot: dict,
            route: Iterable[tuple[int, str, str]]) -> "RunScanCoverage":
        coverage = cls(
            snapshot["run_prefix"], route,
            generation=int(snapshot.get("coverage_generation", 1)))
        coverage.cursor_index = int(snapshot.get("cursor_index", 0))
        by_key = {
            (int(item["station_id"]), str(item["pose_name"]),
             str(item["shelf_band"])): item
            for item in snapshot.get("records", [])
            if item.get("run_prefix") == coverage.run_prefix
        }
        for key, record in coverage.records.items():
            saved = by_key.get((key.station_id, key.pose_name, key.shelf_band))
            if saved is None:
                continue
            for name in (
                    "state", "scan_started_at", "scan_completed_at",
                    "fresh_rgb_frame_count", "fresh_aruco_frame_count",
                    "observed_kinds", "candidate_ids_created",
                    "completion_reason", "estimated_duration_s",
                    "actual_duration_s"):
                setattr(record, name, saved.get(name, getattr(record, name)))
        metrics = snapshot.get("metrics", {})
        for name in (
                "initial_discovery_segments", "resumed_uncovered_scan_segments",
                "coverage_reuse_count", "repeated_covered_pose_count"):
            setattr(coverage, name, int(metrics.get(name, 0)))
        return coverage
