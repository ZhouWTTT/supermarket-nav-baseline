#!/usr/bin/env python3
"""Run-scoped strict-localization failure memory.

The module is pure bookkeeping.  It never changes perception thresholds,
selects samples, removes outliers, or authorizes a grasp.  Material identity
deliberately excludes source stamps, confidence, confirmation growth, and the
attempt number so frame churn cannot manufacture retry opportunities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


UNTRIED = "UNTRIED"
PRIMARY_NO_ASSOCIATION = "PRIMARY_NO_ASSOCIATION"
BACKUP_AVAILABLE = "BACKUP_AVAILABLE"
SPREAD_RECOVERY_AVAILABLE = "SPREAD_RECOVERY_AVAILABLE"
EXHAUSTED_UNTIL_MATERIAL_CHANGE = "EXHAUSTED_UNTIL_MATERIAL_CHANGE"
REACTIVATED = "REACTIVATED"
LOCALIZED = "LOCALIZED"
PREGRASP_FAILED = "PREGRASP_FAILED"
RUNTIME_INVALID = "RUNTIME_INVALID"

NO_ASSOCIATION = "NO_ASSOCIATION"
SPREAD_REJECT = "SPREAD_REJECT"
LOCALIZATION_VALIDATED = "LOCALIZATION_VALIDATED"
PREGRASP_ARM_CONVERGENCE_TIMEOUT = "PREGRASP_ARM_CONVERGENCE_TIMEOUT"
SENSOR_INVALID = "SENSOR_INVALID"
PROCESS_FAILURE = "PROCESS_FAILURE"

BASE_POSITION_BUCKET_M = 0.055
BASE_YAW_BUCKET_RAD = 0.035
HEAD_SLIDE_BUCKET_M = 0.015
HEAD_YAW_BUCKET_RAD = 0.030
HEAD_PITCH_BUCKET_RAD = 0.030
STRICT_POLICY_VERSION = "strict-localization-r15b-v1"


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


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bucket(value: Any, width: float) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return math.floor(number / width + 0.5)


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
    explicit = candidate.get("replay_backup_pose_ids")
    if isinstance(explicit, (list, tuple)):
        backups = tuple(dict.fromkeys(str(item) for item in explicit))
    else:
        backups = (backup,)
    return primary, backups[:1]


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


def _pose_values(candidate: Mapping[str, Any]) -> tuple[tuple, tuple]:
    base = (candidate.get("observed_base_pose")
            or candidate.get("observation_base_pose_hint")
            or (None, None, None))
    head = (candidate.get("observed_head_pose")
            or candidate.get("head_pose_hint")
            or ("derived", None, None, None))
    base = tuple(base) if isinstance(base, (list, tuple)) else (None,) * 3
    head = tuple(head) if isinstance(head, (list, tuple)) else (None,) * 4
    return base, head


@dataclass(frozen=True)
class StrictReplayEquivalenceKey:
    run_prefix: str
    kind: str
    marker_id: int
    station_id: int | None
    shelf_band: str
    base_pose_bucket: tuple[int | None, int | None, int | None]
    head_pose_bucket: tuple[int | None, int | None, int | None]
    primary_pose_id: str
    backup_pose_ids: tuple[str, ...]
    context_quality: str
    bbox_marker_geometry_signature: str
    association_geometry_signature: str
    validated_viewpoint_signature: str | None
    strict_policy_version: str = STRICT_POLICY_VERSION

    @property
    def digest(self) -> str:
        return stable_digest(asdict(self))

    @property
    def family(self) -> tuple[str, str, int]:
        return self.run_prefix, self.kind, self.marker_id

    def as_dict(self) -> dict[str, Any]:
        return _canonical(asdict(self))


def make_equivalence_key(
        *, run_prefix: str, kind: str, marker_id: int,
        candidate: Mapping[str, Any]) -> StrictReplayEquivalenceKey:
    base, head = _pose_values(candidate)
    primary, backups = _pose_plan(candidate)
    pose_name = str(candidate.get("observed_pose_name") or primary)
    validated = candidate.get("validated_station_context")
    return StrictReplayEquivalenceKey(
        run_prefix=str(run_prefix), kind=str(kind), marker_id=int(marker_id),
        station_id=_station_id(candidate), shelf_band=_shelf_band(pose_name),
        base_pose_bucket=(
            _bucket(base[0] if len(base) > 0 else None,
                    BASE_POSITION_BUCKET_M),
            _bucket(base[1] if len(base) > 1 else None,
                    BASE_POSITION_BUCKET_M),
            _bucket(base[2] if len(base) > 2 else None,
                    BASE_YAW_BUCKET_RAD)),
        head_pose_bucket=(
            _bucket(head[1] if len(head) > 1 else None,
                    HEAD_SLIDE_BUCKET_M),
            _bucket(head[2] if len(head) > 2 else None,
                    HEAD_YAW_BUCKET_RAD),
            _bucket(head[3] if len(head) > 3 else None,
                    HEAD_PITCH_BUCKET_RAD)),
        primary_pose_id=primary,
        backup_pose_ids=backups,
        context_quality=str(candidate.get("context_quality") or "DERIVED"),
        bbox_marker_geometry_signature=stable_digest({
            "bbox": candidate.get("target_bbox_summary"),
            "marker_pixel": candidate.get("marker_pixel_summary"),
        }),
        association_geometry_signature=stable_digest(
            candidate.get("association_summary")),
        validated_viewpoint_signature=(
            None if validated is None else stable_digest(validated)),
    )


@dataclass(frozen=True)
class StrictReplayDecision:
    allowed: bool
    state: str
    reason: str
    material_revision: int
    recovery_slot: str | None = None
    reactivation_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Entry:
    key: StrictReplayEquivalenceKey
    material_revision: int
    state: str = UNTRIED
    tried_pose_ids: set[str] = field(default_factory=set)
    spread_reject_count: int = 0
    outcomes: list[tuple[str, str]] = field(default_factory=list)
    reactivation_reason: str | None = None
    localized: bool = False


def _normalize_pose_id(value: Any) -> str:
    return str(value).split(":")[-1]


def strict_state_priority(state: str) -> int:
    """Frozen kind-agnostic scheduling tier for control mode."""
    return {
        UNTRIED: 0,
        REACTIVATED: 1,
        SPREAD_RECOVERY_AVAILABLE: 2,
        BACKUP_AVAILABLE: 3,
        PRIMARY_NO_ASSOCIATION: 4,
        RUNTIME_INVALID: 4,
        LOCALIZED: 4,
        PREGRASP_FAILED: 4,
        EXHAUSTED_UNTIL_MATERIAL_CHANGE: 99,
    }.get(str(state), 98)


class StrictReplayOutcomeMemory:
    """Bounded strict-failure state isolated by ``run_prefix``."""

    def __init__(self, run_prefix: str):
        self.run_prefix = str(run_prefix)
        self._entries: dict[str, _Entry] = {}
        self._latest_by_family: dict[tuple[str, str, int], _Entry] = {}

    def _check(self, key: StrictReplayEquivalenceKey) -> None:
        if key.run_prefix != self.run_prefix:
            raise ValueError("strict replay outcome cannot cross run_prefix")

    def _material_change_reason(
            self, old: StrictReplayEquivalenceKey,
            new: StrictReplayEquivalenceKey) -> str:
        changes = []
        if (old.station_id, old.shelf_band) != (new.station_id, new.shelf_band):
            changes.append("station_or_shelf_changed")
        if (old.base_pose_bucket, old.head_pose_bucket) != (
                new.base_pose_bucket, new.head_pose_bucket):
            changes.append("pose_bucket_changed")
        if old.backup_pose_ids != new.backup_pose_ids:
            changes.append("new_backup_pose")
        if old.context_quality != new.context_quality:
            changes.append("context_quality_changed")
        if (old.bbox_marker_geometry_signature
                != new.bbox_marker_geometry_signature
                or old.association_geometry_signature
                != new.association_geometry_signature):
            changes.append("geometry_changed")
        if old.validated_viewpoint_signature != new.validated_viewpoint_signature:
            changes.append("validated_viewpoint_changed")
        return "+".join(changes) or "material_key_changed"

    def _entry(self, key: StrictReplayEquivalenceKey) -> _Entry:
        self._check(key)
        exact = self._entries.get(key.digest)
        if exact is not None:
            return exact
        previous = self._latest_by_family.get(key.family)
        revision = 1 if previous is None else previous.material_revision + 1
        state = UNTRIED
        reason = None
        if (previous is not None
                and previous.state == EXHAUSTED_UNTIL_MATERIAL_CHANGE):
            state = REACTIVATED
            reason = self._material_change_reason(previous.key, key)
        entry = _Entry(
            key=key, material_revision=revision, state=state,
            reactivation_reason=reason)
        self._entries[key.digest] = entry
        self._latest_by_family[key.family] = entry
        return entry

    def state(self, key: StrictReplayEquivalenceKey) -> str:
        return self._entry(key).state

    def decision(self, key: StrictReplayEquivalenceKey) -> StrictReplayDecision:
        entry = self._entry(key)
        state = entry.state
        if state == EXHAUSTED_UNTIL_MATERIAL_CHANGE:
            return StrictReplayDecision(
                False, state, "exhausted_until_material_change",
                entry.material_revision)
        slot = None
        if state == SPREAD_RECOVERY_AVAILABLE:
            slot = "spread_recovery_1_of_1"
        elif state == BACKUP_AVAILABLE:
            slot = "backup_1_of_1"
        reasons = {
            UNTRIED: "untried_material_context",
            REACTIVATED: "material_context_reactivated",
            SPREAD_RECOVERY_AVAILABLE: "bounded_spread_recovery_available",
            BACKUP_AVAILABLE: "untried_backup_available",
            PRIMARY_NO_ASSOCIATION: "primary_outcome_incomplete",
            RUNTIME_INVALID: "runtime_recovery_allowed",
            LOCALIZED: "validated_candidate_history_retained",
            PREGRASP_FAILED: "pregrasp_execution_failure_not_strict_exhaustion",
        }
        return StrictReplayDecision(
            True, state, reasons.get(state, "strict_retry_allowed"),
            entry.material_revision, recovery_slot=slot,
            reactivation_reason=entry.reactivation_reason)

    def last_outcome(self, key: StrictReplayEquivalenceKey) -> str | None:
        entry = self._entry(key)
        return None if not entry.outcomes else entry.outcomes[-1][0]

    def record_no_association(
            self, key: StrictReplayEquivalenceKey, *, attempt_id: str,
            pose_ids: Sequence[str] = ()) -> StrictReplayDecision:
        entry = self._entry(key)
        attempted = tuple(_normalize_pose_id(item) for item in pose_ids)
        if not attempted:
            attempted = (key.primary_pose_id,)
        entry.tried_pose_ids.update(attempted)
        entry.outcomes.append((NO_ASSOCIATION, str(attempt_id)))
        plan = (key.primary_pose_id, *key.backup_pose_ids)
        if all(pose in entry.tried_pose_ids for pose in plan):
            entry.state = EXHAUSTED_UNTIL_MATERIAL_CHANGE
        elif key.primary_pose_id in entry.tried_pose_ids and any(
                pose not in entry.tried_pose_ids for pose in key.backup_pose_ids):
            entry.state = BACKUP_AVAILABLE
        else:
            entry.state = PRIMARY_NO_ASSOCIATION
        return self.decision(key)

    def record_spread_reject(
            self, key: StrictReplayEquivalenceKey, *,
            attempt_id: str) -> StrictReplayDecision:
        entry = self._entry(key)
        entry.spread_reject_count += 1
        entry.outcomes.append((SPREAD_REJECT, str(attempt_id)))
        entry.state = (
            SPREAD_RECOVERY_AVAILABLE
            if entry.spread_reject_count == 1
            else EXHAUSTED_UNTIL_MATERIAL_CHANGE)
        return self.decision(key)

    def record_localized(
            self, key: StrictReplayEquivalenceKey, *,
            attempt_id: str) -> StrictReplayDecision:
        entry = self._entry(key)
        entry.localized = True
        entry.outcomes.append((LOCALIZATION_VALIDATED, str(attempt_id)))
        entry.state = LOCALIZED
        return self.decision(key)

    def record_pregrasp_failed(
            self, key: StrictReplayEquivalenceKey, *,
            attempt_id: str) -> StrictReplayDecision:
        entry = self._entry(key)
        entry.localized = True
        entry.outcomes.append(
            (PREGRASP_ARM_CONVERGENCE_TIMEOUT, str(attempt_id)))
        entry.state = PREGRASP_FAILED
        return self.decision(key)

    def record_runtime_invalid(
            self, key: StrictReplayEquivalenceKey, *,
            attempt_id: str) -> StrictReplayDecision:
        entry = self._entry(key)
        entry.outcomes.append((SENSOR_INVALID, str(attempt_id)))
        if entry.state in {UNTRIED, REACTIVATED, RUNTIME_INVALID}:
            entry.state = RUNTIME_INVALID
        return self.decision(key)

    def record_process_failure(
            self, key: StrictReplayEquivalenceKey, *,
            attempt_id: str) -> StrictReplayDecision:
        entry = self._entry(key)
        entry.outcomes.append((PROCESS_FAILURE, str(attempt_id)))
        return self.decision(key)

    def record(
            self, key: StrictReplayEquivalenceKey, *, attempt_id: str,
            outcome: str, pose_ids: Sequence[str] = ()) -> StrictReplayDecision:
        outcome = str(outcome).upper()
        if outcome == NO_ASSOCIATION:
            return self.record_no_association(
                key, attempt_id=attempt_id, pose_ids=pose_ids)
        if outcome == SPREAD_REJECT:
            return self.record_spread_reject(key, attempt_id=attempt_id)
        if outcome == LOCALIZATION_VALIDATED:
            return self.record_localized(key, attempt_id=attempt_id)
        if outcome == PREGRASP_ARM_CONVERGENCE_TIMEOUT:
            return self.record_pregrasp_failed(key, attempt_id=attempt_id)
        if outcome == SENSOR_INVALID:
            return self.record_runtime_invalid(key, attempt_id=attempt_id)
        if outcome == PROCESS_FAILURE:
            return self.record_process_failure(key, attempt_id=attempt_id)
        raise ValueError(f"unknown strict replay outcome: {outcome}")
