#!/usr/bin/env python3
"""Run-scoped, material-context replay outcomes.

This module is deliberately free of ROS and perception authority.  It records
what an already-defined replay plan observed; it cannot relax any detection,
association, localization, grasp, navigation, or place gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


UNTRIED = "UNTRIED"
PRIMARY_FAILED_NO_TARGET = "PRIMARY_FAILED_NO_TARGET"
BACKUP_AVAILABLE = "BACKUP_AVAILABLE"
EXHAUSTED_UNTIL_MATERIAL_CHANGE = "EXHAUSTED_UNTIL_MATERIAL_CHANGE"
REACTIVATED = "REACTIVATED"
SUCCEEDED = "SUCCEEDED"
RUNTIME_INVALID = "RUNTIME_INVALID"

NO_FRESH_RGB = "NO_FRESH_RGB"
RGB_NOT_PROCESSED_BY_YOLO = "RGB_NOT_PROCESSED_BY_YOLO"
TARGET_ABSENT_IN_PROCESSED_FRAMES = "TARGET_ABSENT_IN_PROCESSED_FRAMES"

BASE_POSITION_TOLERANCE_M = 0.055
BASE_YAW_TOLERANCE_RAD = 0.035
HEAD_SLIDE_TOLERANCE_M = 0.015
HEAD_YAW_TOLERANCE_RAD = 0.030
HEAD_PITCH_TOLERANCE_RAD = 0.030
REPLAY_POLICY_VERSION = "material-context-r15-v1"


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bucket(value: Any, tolerance: float) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return math.floor(number / tolerance + 0.5)


def _pose_plan(candidate: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    requested = candidate.get("head_pose_hint")
    primary = (str(requested[0]) if isinstance(requested, (list, tuple))
               and len(requested) == 4 else "derived")
    backup = {
        "overview_high": "overview_mid",
        "overview_mid": "overview_high",
        "overview_down": "overview_mid",
        "lower_center": "lower_yaw_minus",
        "lower_yaw_minus": "lower_center",
        "lower_yaw_plus": "lower_center",
    }.get(primary, "derived_backup")
    return primary, (backup,)


def _station_id(candidate: Mapping[str, Any]) -> int | None:
    for name in ("observed_scan_station", "scan_station_hint"):
        station = candidate.get(name)
        if isinstance(station, Mapping):
            try:
                return int(station["index"])
            except (KeyError, TypeError, ValueError):
                pass
    return None


def _shelf_band(pose_name: str) -> str:
    if pose_name.startswith("lower"):
        return "lower"
    if pose_name in {"overview_high", "overview_down"}:
        return "upper"
    return "middle"


def _geometry_signature(candidate: Mapping[str, Any]) -> str:
    # Exact observed pixel geometry is intentional.  No new spatial tolerance
    # is introduced here; stamps, confidence, and confirmation count are absent.
    return _digest({
        "bbox": candidate.get("target_bbox_summary"),
        "marker_pixel": candidate.get("marker_pixel_summary"),
        "association": candidate.get("association_summary"),
    })


@dataclass(frozen=True)
class ReplayEquivalenceKey:
    run_prefix: str
    kind: str
    marker_id: int
    station_id: int | None
    shelf_band: str
    pose_name: str
    base_pose_bucket: tuple[int | None, int | None, int | None]
    head_pose_bucket: tuple[int | None, int | None, int | None]
    primary_replay_pose_id: str
    backup_replay_pose_ids: tuple[str, ...]
    context_quality: str
    geometry_signature: str
    replay_policy_version: str = REPLAY_POLICY_VERSION

    @property
    def digest(self) -> str:
        return _digest(asdict(self))

    def as_dict(self) -> dict[str, Any]:
        return _canonical(asdict(self))


def make_equivalence_key(
        *, run_prefix: str, kind: str, marker_id: int,
        candidate: Mapping[str, Any]) -> ReplayEquivalenceKey:
    base = candidate.get("observed_base_pose") or candidate.get(
        "observation_base_pose_hint") or (None, None, None)
    head = candidate.get("observed_head_pose") or candidate.get(
        "head_pose_hint") or ("derived", None, None, None)
    base = tuple(base) if isinstance(base, (list, tuple)) else (None,) * 3
    head = tuple(head) if isinstance(head, (list, tuple)) else (None,) * 4
    primary, backups = _pose_plan(candidate)
    pose_name = str(candidate.get("observed_pose_name") or primary)
    return ReplayEquivalenceKey(
        run_prefix=str(run_prefix), kind=str(kind), marker_id=int(marker_id),
        station_id=_station_id(candidate), shelf_band=_shelf_band(pose_name),
        pose_name=pose_name,
        base_pose_bucket=(
            _bucket(base[0] if len(base) > 0 else None,
                    BASE_POSITION_TOLERANCE_M),
            _bucket(base[1] if len(base) > 1 else None,
                    BASE_POSITION_TOLERANCE_M),
            _bucket(base[2] if len(base) > 2 else None,
                    BASE_YAW_TOLERANCE_RAD)),
        head_pose_bucket=(
            _bucket(head[1] if len(head) > 1 else None,
                    HEAD_SLIDE_TOLERANCE_M),
            _bucket(head[2] if len(head) > 2 else None,
                    HEAD_YAW_TOLERANCE_RAD),
            _bucket(head[3] if len(head) > 3 else None,
                    HEAD_PITCH_TOLERANCE_RAD)),
        primary_replay_pose_id=primary,
        backup_replay_pose_ids=backups,
        context_quality=str(candidate.get("context_quality") or "DERIVED"),
        geometry_signature=_geometry_signature(candidate))


def classify_fresh_frame_outcome(
        *, raw_fresh_rgb_count: int, yolo_processed_count: int,
        target_detection_count: int) -> str:
    if int(target_detection_count) > 0:
        return SUCCEEDED
    if int(raw_fresh_rgb_count) <= 0:
        return NO_FRESH_RGB
    if int(yolo_processed_count) < minimum_processed_frames_from_success_evidence():
        return RGB_NOT_PROCESSED_BY_YOLO
    return TARGET_ABSENT_IN_PROCESSED_FRAMES


@dataclass(frozen=True)
class ReplayOutcome:
    equivalence_key: ReplayEquivalenceKey
    candidate_id: str
    attempt_id: str
    pose_id: str
    outcome: str
    failure_class: str | None
    fresh_rgb_count: int
    yolo_processed_count: int
    target_detection_count: int
    attempt_start_s: float
    attempt_end_s: float
    material_context_revision: str
    reactivation_requirements: tuple[str, ...] = ()


@dataclass
class _MemoryEntry:
    key: ReplayEquivalenceKey
    candidate_ids: set[str] = field(default_factory=set)
    attempt_ids: set[str] = field(default_factory=set)
    tried_pose_ids: set[str] = field(default_factory=set)
    state: str = UNTRIED
    outcomes: list[ReplayOutcome] = field(default_factory=list)


class ReplayOutcomeMemory:
    """Outcome state isolated by run prefix and material context key."""

    def __init__(self, run_prefix: str):
        self.run_prefix = str(run_prefix)
        self._entries: dict[str, _MemoryEntry] = {}

    def _entry(self, key: ReplayEquivalenceKey) -> _MemoryEntry:
        if key.run_prefix != self.run_prefix:
            raise ValueError("replay outcome cannot cross run_prefix")
        return self._entries.setdefault(key.digest, _MemoryEntry(key=key))

    def state(self, key: ReplayEquivalenceKey) -> str:
        entry = self._entries.get(key.digest)
        return UNTRIED if entry is None else entry.state

    def record(self, outcome: ReplayOutcome) -> str:
        entry = self._entry(outcome.equivalence_key)
        if any(item.attempt_id == outcome.attempt_id
               and item.pose_id == outcome.pose_id for item in entry.outcomes):
            return entry.state
        entry.outcomes.append(outcome)
        entry.candidate_ids.add(str(outcome.candidate_id))
        entry.attempt_ids.add(str(outcome.attempt_id))
        if outcome.outcome == SUCCEEDED:
            entry.state = SUCCEEDED
            return entry.state
        if outcome.failure_class in {NO_FRESH_RGB,
                                     RGB_NOT_PROCESSED_BY_YOLO}:
            # Runtime-invalid observations do not consume a semantic pose.
            entry.state = RUNTIME_INVALID
            return entry.state
        entry.tried_pose_ids.add(str(outcome.pose_id))
        plan = (entry.key.primary_replay_pose_id,
                *entry.key.backup_replay_pose_ids)
        if all(pose in entry.tried_pose_ids for pose in plan):
            entry.state = EXHAUSTED_UNTIL_MATERIAL_CHANGE
        elif entry.key.primary_replay_pose_id in entry.tried_pose_ids:
            entry.state = BACKUP_AVAILABLE
        else:
            entry.state = PRIMARY_FAILED_NO_TARGET
        return entry.state

    def decision(self, key: ReplayEquivalenceKey) -> tuple[bool, str]:
        state = self.state(key)
        if state == EXHAUSTED_UNTIL_MATERIAL_CHANGE:
            return False, "exhausted_until_material_change"
        if state == SUCCEEDED:
            return False, "already_succeeded"
        if state == RUNTIME_INVALID:
            return True, "sensor_recovered_retry_allowed"
        return True, state.lower()

    def reactivate(self, old_key: ReplayEquivalenceKey,
                   new_key: ReplayEquivalenceKey) -> bool:
        if old_key.run_prefix != new_key.run_prefix:
            return False
        old = self._entries.get(old_key.digest)
        if old is None or old.state != EXHAUSTED_UNTIL_MATERIAL_CHANGE:
            return False
        if old_key.digest == new_key.digest:
            return False
        existing = self._entries.get(new_key.digest)
        if existing is not None and existing.state == REACTIVATED:
            return False
        self._entry(new_key).state = REACTIVATED
        return True


def replay_state_priority(state: str) -> int:
    """Deterministic, kind-agnostic scheduler priority."""
    return {
        UNTRIED: 0, REACTIVATED: 1, BACKUP_AVAILABLE: 2,
        RUNTIME_INVALID: 3, PRIMARY_FAILED_NO_TARGET: 4,
        EXHAUSTED_UNTIL_MATERIAL_CHANGE: 9, SUCCEEDED: 10,
    }.get(str(state), 8)


def minimum_processed_frames_from_success_evidence() -> int:
    """Frozen lower bound observed in successful R11-A/R13 direct replays."""
    return 6


def processed_frame_wait_budget_s_from_success_evidence() -> float:
    """Bounded by the slowest successful first-frame latency (29.318 s)."""
    return 30.0


def fresh_frame_gate_should_hold(
        *, target_detection_count: int, yolo_processed_count: int,
        pose_elapsed_s: float) -> bool:
    """Hold only a still-empty target funnel, within the empirical budget."""
    return (
        int(target_detection_count) <= 0
        and int(yolo_processed_count)
        < minimum_processed_frames_from_success_evidence()
        and float(pose_elapsed_s)
        < processed_frame_wait_budget_s_from_success_evidence())
