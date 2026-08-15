#!/usr/bin/env python3
"""Read-only evidence ledgers for strict shelf-localisation stability.

This module deliberately owns no acceptance thresholds and makes no control
decision.  It freezes primitive snapshots, bounds in-memory history, and
computes counterfactual transforms for later diagnosis.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


TRACE_OFF = "off"
TRACE_SUMMARY = "summary"
TRACE_FULL = "full"
TRACE_MODES = frozenset((TRACE_OFF, TRACE_SUMMARY, TRACE_FULL))


def strict_trace_mode(value: Any = None) -> str:
    """Validate the competition-default trace mode."""
    selected = (os.getenv("SUPERMARKET_STRICT_STABILITY_TRACE", TRACE_OFF)
                if value is None else str(value)).strip().lower()
    if selected not in TRACE_MODES:
        raise ValueError(
            "SUPERMARKET_STRICT_STABILITY_TRACE must be off, summary, or full")
    return selected


class StrictTraceRuntime:
    """Mode gate and counters kept separate from production decisions."""

    def __init__(self, mode: Any = None, *, pair_maxlen: int = 256,
                 sample_maxlen: int = 128):
        self.mode = strict_trace_mode(mode)
        self._pair_maxlen = int(pair_maxlen)
        self._sample_maxlen = int(sample_maxlen)
        self.ledger = (StabilityAttemptLedger(
            pair_maxlen=self._pair_maxlen, sample_maxlen=self._sample_maxlen)
                       if self.mode == TRACE_FULL else None)
        self.reset_attempt()

    def reset_attempt(self) -> None:
        if self.ledger is not None:
            self.ledger.reset_attempt()
        self.pair_count = 0
        self.accepted_sample_count = 0
        self.final_spread_m = 0.0
        self.strict_trace_pair_records_built = 0
        self.strict_trace_sample_records_built = 0
        self.strict_trace_events_emitted = 0

    @property
    def pair_records_enabled(self) -> bool:
        return self.mode == TRACE_FULL

    @property
    def sample_records_enabled(self) -> bool:
        return self.mode == TRACE_FULL

    def observe_pair(self) -> None:
        if self.mode != TRACE_OFF:
            self.pair_count += 1

    def append_pair(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if self.ledger is None or self.mode != TRACE_FULL:
            raise RuntimeError("per-pair trace is disabled")
        self.strict_trace_pair_records_built += 1
        return self.ledger.append_pair(record)

    def observe_sample(self, *, accepted: bool, spread_m: float) -> None:
        if self.mode == TRACE_OFF:
            return
        if accepted:
            self.accepted_sample_count += 1
        value = float(spread_m)
        if math.isfinite(value):
            self.final_spread_m = value

    def append_sample(self, record: Mapping[str, Any]) -> tuple[dict, dict]:
        if self.ledger is None or self.mode != TRACE_FULL:
            raise RuntimeError("per-sample trace is disabled")
        self.strict_trace_sample_records_built += 1
        return self.ledger.append_sample(record)

    def emit(self, sink, event: str, **payload) -> bool:
        if self.mode != TRACE_FULL or sink is None:
            return False
        sink(event, **payload)
        self.strict_trace_events_emitted += 1
        return True

    def emit_summary(self, sink, **payload) -> bool:
        if self.mode != TRACE_SUMMARY or sink is None:
            return False
        sink("strict_localization_trace_summary",
             pair_count=self.pair_count,
             accepted_sample_count=self.accepted_sample_count,
             final_spread_m=self.final_spread_m,
             **payload)
        self.strict_trace_events_emitted += 1
        return True

    def counters(self) -> dict[str, int]:
        return {
            "strict_trace_pair_records_built":
                self.strict_trace_pair_records_built,
            "strict_trace_sample_records_built":
                self.strict_trace_sample_records_built,
            "strict_trace_events_emitted": self.strict_trace_events_emitted,
        }


def _finite_number(value: Any) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a finite numeric primitive")
    if not math.isfinite(float(value)):
        raise ValueError("non-finite diagnostic value")
    return value


def freeze_primitive(value: Any, *, max_sequence: int = 128) -> Any:
    """Return an immutable, finite, bounded primitive tree."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return _finite_number(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        if len(value) > max_sequence:
            raise ValueError("mapping exceeds diagnostic bound")
        return tuple(
            (str(key), freeze_primitive(item, max_sequence=max_sequence))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) > max_sequence:
            raise ValueError("sequence exceeds diagnostic bound")
        return tuple(freeze_primitive(item, max_sequence=max_sequence)
                     for item in value)
    raise ValueError(f"unsupported live diagnostic value: {type(value).__name__}")


def thaw_primitive(value: Any) -> Any:
    """Create a detached JSON-compatible copy of a frozen primitive tree."""
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2
               and isinstance(item[0], str) for item in value):
            return {key: thaw_primitive(item) for key, item in value}
        return [thaw_primitive(item) for item in value]
    return value


@dataclass(frozen=True)
class FrozenRecord:
    fields: tuple[tuple[str, Any], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenRecord":
        frozen = freeze_primitive(value)
        if not isinstance(frozen, tuple):
            raise ValueError("record must be a mapping")
        return cls(frozen)

    def to_dict(self) -> dict[str, Any]:
        return thaw_primitive(self.fields)


class BoundedLedger:
    """Bounded store that never retains caller-owned arrays or messages."""

    def __init__(self, maxlen: int):
        if int(maxlen) <= 0:
            raise ValueError("maxlen must be positive")
        self._records: deque[FrozenRecord] = deque(maxlen=int(maxlen))

    def append(self, record: Mapping[str, Any]) -> FrozenRecord:
        frozen = FrozenRecord.from_mapping(record)
        self._records.append(frozen)
        return frozen

    def snapshot(self) -> tuple[FrozenRecord, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)


class IdentityIsolationGuard:
    """Pure helper describing when an evidence epoch must be isolated."""

    def __init__(self):
        self._identity: tuple[int, str] | None = None

    def observe(self, marker_id: int, position_source: str) -> bool:
        identity = (int(marker_id), str(position_source))
        changed = self._identity is not None and identity != self._identity
        self._identity = identity
        return changed

    @property
    def identity(self) -> tuple[int, str] | None:
        return self._identity


@dataclass(frozen=True)
class AssociationDecision:
    accepted: bool
    diagnostic_reason: str
    marker_index: int | None
    immutable_inputs: FrozenRecord


def source_stamp_ns(message: Any) -> int | None:
    try:
        stamp = message.header.stamp
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def latest_not_after(history: Iterable[Mapping[str, Any]], stamp_ns: int,
                     *, max_delta_ns: int | None = None) -> tuple[dict, int] | None:
    """Select only a past-or-equal source state; never select a future state."""
    candidates = []
    for state in history:
        try:
            state_stamp = int(state["source_stamp_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        if state_stamp <= int(stamp_ns):
            candidates.append((state_stamp, state))
    if not candidates:
        return None
    state_stamp, state = max(candidates, key=lambda item: item[0])
    delta = int(stamp_ns) - state_stamp
    if max_delta_ns is not None and delta > int(max_delta_ns):
        return None
    return thaw_primitive(freeze_primitive(state)), delta


def _xyz(value: Any) -> tuple[float, float, float] | None:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        return None
    return values


def transform_xyz(matrix: Any, xyz: Any) -> tuple[float, float, float] | None:
    point = _xyz(xyz)
    try:
        rows = tuple(tuple(float(item) for item in row) for row in matrix)
    except (TypeError, ValueError):
        return None
    if (point is None or len(rows) != 4 or any(len(row) != 4 for row in rows)
            or not all(math.isfinite(item) for row in rows for item in row)):
        return None
    augmented = (*point, 1.0)
    result = tuple(sum(rows[row][column] * augmented[column]
                       for column in range(4)) for row in range(3))
    return result if all(math.isfinite(item) for item in result) else None


def point_to_point_metrics(points: Iterable[Any]) -> tuple[list[float], float]:
    parsed = [point for value in points if (point := _xyz(value)) is not None]
    if len(parsed) < 2:
        return [0.0, 0.0, 0.0], 0.0
    ptp = [max(point[axis] for point in parsed)
           - min(point[axis] for point in parsed) for axis in range(3)]
    return ptp, max(ptp)


def marker_geometry(corners: Any, *, image_width: float | None = None,
                    image_height: float | None = None) -> dict[str, Any]:
    try:
        points = [tuple(map(float, point)) for point in corners]
    except (TypeError, ValueError):
        return {}
    if (len(points) != 4 or any(len(point) != 2 for point in points)
            or not all(math.isfinite(v) for point in points for v in point)):
        return {}
    center = [sum(point[axis] for point in points) / 4.0 for axis in range(2)]
    edges = [math.hypot(points[(i + 1) % 4][0] - points[i][0],
                        points[(i + 1) % 4][1] - points[i][1]) for i in range(4)]
    area = abs(sum(points[i][0] * points[(i + 1) % 4][1]
                   - points[(i + 1) % 4][0] * points[i][1]
                   for i in range(4))) * 0.5
    mean_edge = sum(edges) / 4.0
    skew = 0.0 if mean_edge <= 0.0 else (max(edges) - min(edges)) / mean_edge
    margin = None
    if image_width is not None and image_height is not None:
        values = [*(point[0] for point in points),
                  *(float(image_width) - point[0] for point in points),
                  *(point[1] for point in points),
                  *(float(image_height) - point[1] for point in points)]
        margin = min(values)
    return {
        "marker_corners_px": points,
        "marker_center_px": center,
        "marker_area_px": area,
        "marker_perimeter_px": sum(edges),
        "marker_image_margin_px": margin,
        "marker_corner_edge_lengths_px": edges,
        "marker_corner_skew_metric": skew,
    }


class StabilityAttemptLedger:
    """Per-controller bounded pair/sample ledger and variance decomposition."""

    def __init__(self, *, pair_maxlen: int = 256, sample_maxlen: int = 128):
        self.pairs = BoundedLedger(pair_maxlen)
        self.samples = BoundedLedger(sample_maxlen)
        self._fixed_transform_by_epoch: dict[tuple[Any, Any], Any] = {}

    def reset_attempt(self) -> None:
        self.__init__(pair_maxlen=self.pairs._records.maxlen,
                      sample_maxlen=self.samples._records.maxlen)

    def append_pair(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self.pairs.append(record).to_dict()

    def append_sample(self, record: Mapping[str, Any]) -> tuple[dict, dict]:
        detached = thaw_primitive(freeze_primitive(record))
        epoch = (detached.get("scan_epoch_id"), detached.get("pose_id"))
        runtime_tmat = detached.pop("_runtime_camera_world_tmat", None)
        source_tmat = detached.pop("_source_time_camera_world_tmat", None)
        raw = detached.get("raw_marker_camera_xyz_m")
        runtime = detached.get("marker_world_xyz_m")
        if epoch not in self._fixed_transform_by_epoch and runtime_tmat is not None:
            self._fixed_transform_by_epoch[epoch] = freeze_primitive(runtime_tmat)
        fixed_tmat = thaw_primitive(self._fixed_transform_by_epoch.get(epoch))
        detached["world_xyz_using_current_runtime_binding"] = runtime
        detached["world_xyz_using_nearest_source_time_state"] = transform_xyz(
            source_tmat, raw) if source_tmat is not None else None
        detached["world_xyz_using_fixed_first_sample_transform"] = transform_xyz(
            fixed_tmat, raw) if fixed_tmat is not None else None
        prior = [item.to_dict() for item in self.samples.snapshot()
                 if (item.to_dict().get("scan_epoch_id"),
                     item.to_dict().get("pose_id")) == epoch
                 and item.to_dict().get("accepted")]
        prospective = prior + ([detached] if detached.get("accepted") else [])
        ptp, spread = point_to_point_metrics(
            item.get("marker_world_xyz_m") for item in prospective)
        detached["cumulative_world_ptp_xyz_m"] = ptp
        detached["cumulative_world_spread_m"] = spread
        frozen = self.samples.append(detached).to_dict()
        return frozen, self.summary(epoch=epoch)

    def summary(self, *, epoch: tuple[Any, Any] | None = None) -> dict[str, Any]:
        records = [item.to_dict() for item in self.samples.snapshot()
                   if item.to_dict().get("accepted")]
        if epoch is not None:
            records = [item for item in records
                       if (item.get("scan_epoch_id"), item.get("pose_id")) == epoch]
        metrics = {}
        for name, field in (
                ("camera_frame", "raw_marker_camera_xyz_m"),
                ("runtime_world", "world_xyz_using_current_runtime_binding"),
                ("source_time_world", "world_xyz_using_nearest_source_time_state"),
                ("fixed_transform_world", "world_xyz_using_fixed_first_sample_transform")):
            ptp, spread = point_to_point_metrics(item.get(field) for item in records)
            metrics[f"{name}_ptp_xyz_m"] = ptp
            metrics[f"{name}_spread_m"] = spread
        scalar_ranges = {
            "base_motion_ptp_m": "base_pose_used_for_transform",
            "base_yaw_ptp_rad": "_base_yaw_scalar",
            "head_slide_ptp_m": "_head_slide_scalar",
            "head_yaw_ptp_rad": "_head_yaw_scalar",
            "head_pitch_ptp_rad": "_head_pitch_scalar",
            "marker_area_range_px": "marker_area_px",
            "marker_skew_range": "marker_corner_skew_metric",
        }
        for output, field in scalar_ranges.items():
            if field == "base_pose_used_for_transform":
                values = [item.get(field) for item in records]
                xy = [tuple(map(float, value[:2])) for value in values
                      if isinstance(value, list) and len(value) >= 2]
                value = max((math.hypot(a[0] - b[0], a[1] - b[1])
                             for index, a in enumerate(xy)
                             for b in xy[index + 1:]), default=0.0)
            else:
                values = [float(item[field]) for item in records
                          if isinstance(item.get(field), (int, float))]
                value = max(values) - min(values) if len(values) >= 2 else 0.0
            metrics[output] = value
        centers = [item.get("marker_center_px") for item in records]
        valid_centers = [center for center in centers
                         if isinstance(center, list) and len(center) == 2]
        metrics["marker_center_range_px"] = (
            [max(c[i] for c in valid_centers) - min(c[i] for c in valid_centers)
             for i in range(2)] if len(valid_centers) >= 2 else [0.0, 0.0])
        metrics["position_source_unique_values"] = sorted({
            str(item.get("position_source")) for item in records
            if item.get("position_source") is not None})
        metrics["marker_id_unique_values"] = sorted({
            int(item["marker_id"]) for item in records
            if isinstance(item.get("marker_id"), int)})
        metrics["accepted_sample_count"] = len(records)
        return metrics
