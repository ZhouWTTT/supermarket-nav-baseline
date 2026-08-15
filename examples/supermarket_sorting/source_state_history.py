"""Bounded, past-only source-state evidence for inventory candidates.

The history stores detached finite primitives only.  It never retains ROS
messages, NumPy arrays, controller references, or interpolated state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Iterable


COMPLETE = "COMPLETE"
PARTIAL = "PARTIAL"
UNAVAILABLE = "UNAVAILABLE"


def _finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a state coordinate")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("state coordinate must be finite")
    return result


def _stamp(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a source stamp")
    result = int(value)
    if result < 0:
        raise ValueError("source stamp must be non-negative")
    return result


@dataclass(frozen=True)
class OdomStateSample:
    source_stamp_ns: int
    callback_receipt_monotonic_ns: int
    x: float
    y: float
    yaw: float

    @classmethod
    def create(cls, *, source_stamp_ns: Any,
               callback_receipt_monotonic_ns: Any,
               x: Any, y: Any, yaw: Any) -> "OdomStateSample":
        return cls(_stamp(source_stamp_ns),
                   _stamp(callback_receipt_monotonic_ns),
                   _finite(x), _finite(y), _finite(yaw))

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.yaw


@dataclass(frozen=True)
class JointStateSample:
    source_stamp_ns: int
    callback_receipt_monotonic_ns: int
    slide: float
    head_yaw: float
    head_pitch: float

    @classmethod
    def create(cls, *, source_stamp_ns: Any,
               callback_receipt_monotonic_ns: Any,
               slide: Any, head_yaw: Any,
               head_pitch: Any) -> "JointStateSample":
        return cls(_stamp(source_stamp_ns),
                   _stamp(callback_receipt_monotonic_ns),
                   _finite(slide), _finite(head_yaw), _finite(head_pitch))

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.slide, self.head_yaw, self.head_pitch


def latest_not_after(history: Iterable[Any], source_stamp_ns: Any, *,
                     max_delta_ns: int = 500_000_000):
    """Return ``(sample, delta_ns)`` using a past/same-stamp sample only."""
    target = _stamp(source_stamp_ns)
    selected = None
    for sample in history:
        stamp = int(sample.source_stamp_ns)
        if stamp <= target and (selected is None
                                or stamp > selected.source_stamp_ns):
            selected = sample
    if selected is None:
        return None
    delta = target - int(selected.source_stamp_ns)
    if delta > int(max_delta_ns):
        return None
    return selected, delta


class BoundedSourceStateHistory:
    """Two bounded histories with monotonicity diagnostics."""

    def __init__(self, *, odom_maxlen: int = 512,
                 joint_maxlen: int = 512,
                 max_lookup_delta_ns: int = 500_000_000):
        if odom_maxlen <= 0 or joint_maxlen <= 0:
            raise ValueError("history bounds must be positive")
        self._odom = deque(maxlen=int(odom_maxlen))
        self._joint = deque(maxlen=int(joint_maxlen))
        self.max_lookup_delta_ns = int(max_lookup_delta_ns)
        self.odom_stamp_regression_count = 0
        self.joint_stamp_regression_count = 0

    def clear(self) -> None:
        self._odom.clear()
        self._joint.clear()
        self.odom_stamp_regression_count = 0
        self.joint_stamp_regression_count = 0

    def append_odom(self, **values: Any) -> OdomStateSample:
        sample = OdomStateSample.create(**values)
        if self._odom and sample.source_stamp_ns < self._odom[-1].source_stamp_ns:
            self.odom_stamp_regression_count += 1
        self._odom.append(sample)
        return sample

    def append_joint(self, **values: Any) -> JointStateSample:
        sample = JointStateSample.create(**values)
        if self._joint and sample.source_stamp_ns < self._joint[-1].source_stamp_ns:
            self.joint_stamp_regression_count += 1
        self._joint.append(sample)
        return sample

    @property
    def odom_samples(self) -> tuple[OdomStateSample, ...]:
        return tuple(self._odom)

    @property
    def joint_samples(self) -> tuple[JointStateSample, ...]:
        return tuple(self._joint)

    def odom_at(self, source_stamp_ns: Any):
        return latest_not_after(
            self._odom, source_stamp_ns,
            max_delta_ns=self.max_lookup_delta_ns)

    def joint_at(self, source_stamp_ns: Any):
        return latest_not_after(
            self._joint, source_stamp_ns,
            max_delta_ns=self.max_lookup_delta_ns)


def _wrap_to_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def pose_delta(base_a, head_a, base_b, head_b):
    if base_a is None or head_a is None or base_b is None or head_b is None:
        return None
    return (
        ("base_position_m", math.hypot(
            float(base_a[0]) - float(base_b[0]),
            float(base_a[1]) - float(base_b[1]))),
        ("base_yaw_rad", _wrap_to_pi(
            float(base_a[2]) - float(base_b[2]))),
        ("head_slide_m", float(head_a[0]) - float(head_b[0])),
        ("head_yaw_rad", _wrap_to_pi(
            float(head_a[1]) - float(head_b[1]))),
        ("head_pitch_rad", _wrap_to_pi(
            float(head_a[2]) - float(head_b[2]))),
    )


def _pose_or_none(value: Any):
    if value is None:
        return None
    try:
        result = tuple(_finite(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result if len(result) == 3 else None


def classify_candidate_outcome(result: Any) -> str:
    """Map a replayed candidate terminal result to the R13 join vocabulary."""
    if not isinstance(result, dict):
        return "OTHER_TERMINAL"
    if result.get("validated_marker_id") is not None:
        return "LOCALIZATION_VALIDATED"
    reason = str(result.get("candidate_first_failure_reason") or "")
    if reason == "no_target_kind":
        return "NO_TARGET_KIND"
    if reason in {"no_association", "insufficient_confirmations"}:
        return "NO_ASSOCIATION"
    if reason == "spread_reject":
        return "SPREAD_REJECT"
    if int(result.get("target_kind_detection_count") or 0) > 0:
        return "TARGET_KIND_REACQUIRED"
    return "OTHER_TERMINAL"


@dataclass(frozen=True)
class CandidateSourceStateEvidence:
    fields: tuple[tuple[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        def thaw(value):
            if isinstance(value, tuple):
                if all(isinstance(item, tuple) and len(item) == 2
                       and isinstance(item[0], str) for item in value):
                    return {key: thaw(item) for key, item in value}
                return [thaw(item) for item in value]
            return value
        return {key: thaw(value) for key, value in self.fields}


def build_candidate_source_state_evidence(
        *, history: BoundedSourceStateHistory, run_prefix: Any,
        candidate_id: Any, kind: Any, marker_id: Any,
        confirmation_count: Any, yolo_source_stamp_ns: Any,
        aruco_source_stamp_ns: Any, callback_latest_base_pose: Any,
        callback_latest_head_pose: Any) -> CandidateSourceStateEvidence:
    yolo_stamp = _stamp(yolo_source_stamp_ns)
    aruco_stamp = _stamp(aruco_source_stamp_ns)
    callback_base = _pose_or_none(callback_latest_base_pose)
    callback_head = _pose_or_none(callback_latest_head_pose)
    yo = history.odom_at(yolo_stamp)
    yj = history.joint_at(yolo_stamp)
    ao = history.odom_at(aruco_stamp)
    aj = history.joint_at(aruco_stamp)
    base_yolo = None if yo is None else yo[0].pose
    head_yolo = None if yj is None else yj[0].pose
    base_aruco = None if ao is None else ao[0].pose
    head_aruco = None if aj is None else aj[0].pose
    present = sum(value is not None for value in (yo, yj, ao, aj))
    availability = COMPLETE if present == 4 else PARTIAL if present else UNAVAILABLE
    fields = (
        ("run_prefix", str(run_prefix)),
        ("candidate_id", str(candidate_id)),
        ("kind", str(kind)),
        ("marker_id", int(marker_id)),
        ("confirmation_count", int(confirmation_count)),
        ("yolo_source_stamp_ns", yolo_stamp),
        ("aruco_source_stamp_ns", aruco_stamp),
        ("pair_delta_ns", abs(yolo_stamp - aruco_stamp)),
        ("callback_latest_base_pose", callback_base),
        ("callback_latest_head_pose", callback_head),
        ("base_pose_at_yolo_source", base_yolo),
        ("head_pose_at_yolo_source", head_yolo),
        ("base_pose_at_aruco_source", base_aruco),
        ("head_pose_at_aruco_source", head_aruco),
        ("callback_yolo_pose_delta", pose_delta(
            callback_base, callback_head, base_yolo, head_yolo)),
        ("callback_aruco_pose_delta", pose_delta(
            callback_base, callback_head, base_aruco, head_aruco)),
        ("yolo_aruco_pose_delta", pose_delta(
            base_yolo, head_yolo, base_aruco, head_aruco)),
        ("odom_lookup_delta_ns", (
            ("yolo", None if yo is None else yo[1]),
            ("aruco", None if ao is None else ao[1]))),
        ("joint_lookup_delta_ns", (
            ("yolo", None if yj is None else yj[1]),
            ("aruco", None if aj is None else aj[1]))),
        ("source_context_availability", availability),
        ("creation_context_source", "CALLBACK_LATEST"),
        ("state_stamp_regression_count", (
            ("odom", history.odom_stamp_regression_count),
            ("joint", history.joint_stamp_regression_count))),
    )
    return CandidateSourceStateEvidence(fields)
