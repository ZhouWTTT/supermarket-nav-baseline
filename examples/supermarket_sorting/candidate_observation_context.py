"""Immutable, JSON-safe snapshots for candidate observation replay.

Only sensor/controller values sampled at the association instant are marked
observed.  Geometry-based fallbacks stay explicitly derived.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


OBSERVED_CONTEXT = "OBSERVED_CONTEXT"
DERIVED_CONTEXT = "DERIVED_CONTEXT"
CONTEXT_COMPLETE = "CONTEXT_COMPLETE"
CONTEXT_DERIVED = "CONTEXT_DERIVED"
CONTEXT_INCOMPLETE = "CONTEXT_INCOMPLETE"


def normalize_kind(value: object) -> str:
    """Normalize spelling only; never alias one competition class to another."""
    return str(value or "").strip().lower()


def _finite_tuple(values: object, length: int) -> tuple[float, ...] | None:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        return None
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(value) for value in result) else None


def nearest_pose_name(
        head_pose: Sequence[float],
        camera_poses: Iterable[Sequence[object]]) -> str | None:
    """Name the closest configured pose while retaining measured joint values."""
    measured = _finite_tuple(head_pose, 3)
    if measured is None:
        return None
    candidates: list[tuple[float, str]] = []
    for pose in camera_poses:
        if len(pose) != 4:
            continue
        configured = _finite_tuple(pose[1:], 3)
        if configured is None:
            continue
        error = sum((actual - target) ** 2
                    for actual, target in zip(measured, configured))
        candidates.append((error, str(pose[0])))
    return min(candidates)[1] if candidates else None


def association_summary(detection: Mapping, marker: Mapping) -> tuple | None:
    """Copy the image association geometry into immutable primitives."""
    bbox = _finite_tuple(detection.get("bbox_xyxy"), 4)
    pixel = _finite_tuple(marker.get("pixel_center"), 2)
    if bbox is None or pixel is None:
        return None
    x0, y0, x1, y1 = bbox
    bottom = ((x0 + x1) * 0.5, y1)
    return (
        ("bbox_center", ((x0 + x1) * 0.5, (y0 + y1) * 0.5)),
        ("bbox_bottom_center", bottom),
        ("marker_pixel", pixel),
        ("bottom_center_distance_px", math.hypot(
            pixel[0] - bottom[0], pixel[1] - bottom[1])),
    )


@dataclass(frozen=True)
class ObservationContext:
    context_type: str
    context_source: str
    context_quality: str
    observed_base_pose: tuple[float, float, float] | None
    observed_head_pose: tuple[str, float, float, float] | None
    observed_scan_station: tuple | None
    observed_pose_name: str | None
    observed_source_stamps: tuple[int, int] | None
    target_bbox_summary: tuple[float, float, float, float] | None
    marker_pixel_summary: tuple[float, float] | None
    association_summary: tuple | None
    controller_state: str | None
    scan_index: int | None
    pitch_index: int | None
    camera_settled: bool | None

    def as_dict(self) -> dict:
        """Return a new JSON-safe structure with no live mutable references."""
        association = (
            None if self.association_summary is None
            else {key: (list(value) if isinstance(value, tuple) else value)
                  for key, value in self.association_summary})
        station = None
        if self.observed_scan_station is not None:
            station = dict(self.observed_scan_station)
        return {
            "context_type": self.context_type,
            "context_source": self.context_source,
            "context_quality": self.context_quality,
            "observed_base_pose": (None if self.observed_base_pose is None
                                   else list(self.observed_base_pose)),
            "observed_head_pose": (None if self.observed_head_pose is None
                                   else list(self.observed_head_pose)),
            "observed_scan_station": station,
            "observed_pose_name": self.observed_pose_name,
            "observed_source_stamps": (None if self.observed_source_stamps is None
                                       else list(self.observed_source_stamps)),
            "target_bbox_summary": (None if self.target_bbox_summary is None
                                    else list(self.target_bbox_summary)),
            "marker_pixel_summary": (None if self.marker_pixel_summary is None
                                     else list(self.marker_pixel_summary)),
            "association_summary": association,
            "controller_state": self.controller_state,
            "scan_index": self.scan_index,
            "pitch_index": self.pitch_index,
            "camera_settled": self.camera_settled,
        }


def make_observed_context(
        *, base_pose: object, head_pose: object,
        camera_poses: Iterable[Sequence[object]], station_index: int | None,
        station_x: float | None, station_y: float | None,
        yolo_stamp: object, aruco_stamp: object,
        detection: Mapping, marker: Mapping,
        controller_state: str | None = None,
        scan_index: int | None = None, pitch_index: int | None = None,
        camera_settled: bool | None = None) -> ObservationContext:
    base = _finite_tuple(base_pose, 3)
    measured_head = _finite_tuple(head_pose, 3)
    pose_name = (None if measured_head is None
                 else nearest_pose_name(measured_head, camera_poses))
    head = (None if measured_head is None or pose_name is None
            else (pose_name, *measured_head))
    bbox = _finite_tuple(detection.get("bbox_xyxy"), 4)
    pixel = _finite_tuple(marker.get("pixel_center"), 2)
    geometry = association_summary(detection, marker)
    try:
        stamps = (int(yolo_stamp), int(aruco_stamp))
    except (TypeError, ValueError):
        stamps = None
    station = None
    if station_index is not None and base is not None:
        station = (
            ("index", int(station_index)),
            ("nominal_x", None if station_x is None else float(station_x)),
            ("nominal_y", None if station_y is None else float(station_y)),
            ("observed_x", base[0]), ("observed_y", base[1]),
            ("observed_yaw", base[2]),
        )
    complete = all(value is not None for value in (
        base, head, station, stamps, bbox, pixel, geometry))
    return ObservationContext(
        context_type=(OBSERVED_CONTEXT if complete else DERIVED_CONTEXT),
        context_source=("OBSERVED" if complete else "DERIVED"),
        context_quality=(CONTEXT_COMPLETE if complete else CONTEXT_INCOMPLETE),
        observed_base_pose=base,
        observed_head_pose=head,
        observed_scan_station=station,
        observed_pose_name=pose_name,
        observed_source_stamps=stamps,
        target_bbox_summary=bbox,
        marker_pixel_summary=pixel,
        association_summary=geometry,
        controller_state=(None if controller_state is None
                          else str(controller_state)),
        scan_index=(None if scan_index is None else int(scan_index)),
        pitch_index=(None if pitch_index is None else int(pitch_index)),
        camera_settled=(None if camera_settled is None else bool(camera_settled)),
    )
