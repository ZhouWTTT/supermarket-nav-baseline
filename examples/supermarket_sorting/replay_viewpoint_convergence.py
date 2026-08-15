"""Fail-closed convergence and epoch lifecycle for candidate replay views.

The controller has no perception authority.  It only records whether the
commanded base/head viewpoint is stably reached and whether subsequently
received frames belong to the current pose epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


def select_replay_head_poses(
        hint: dict, configured_poses: Sequence[Sequence[object]],
        *, top_shelf_z_m: float, middle_shelf_z_min_m: float) -> tuple:
    """Preserve an exact observed pose, followed by one configured backup."""
    poses = tuple(tuple(pose) for pose in configured_poses)
    requested = hint.get("head_pose_hint")
    main_index = None
    if isinstance(requested, (list, tuple)) and len(requested) == 4:
        requested_name = str(requested[0])
        main_index = next((
            index for index, pose in enumerate(poses)
            if pose[0] == requested_name), None)
    if main_index is None:
        try:
            z = float(hint["provisional_marker_world"][2])
        except (KeyError, TypeError, ValueError, IndexError):
            z = 0.85
        if z >= top_shelf_z_m:
            main_index = 0
        elif z >= middle_shelf_z_min_m:
            main_index = 2
        else:
            main_index = 3
    backup_index = {0: 1, 1: 0, 2: 1, 3: 4, 4: 3, 5: 3}[main_index]
    if (hint.get("context_type") == "OBSERVED_CONTEXT"
            and isinstance(requested, (list, tuple))
            and len(requested) == 4):
        try:
            observed = (str(requested[0]), *(float(value)
                         for value in requested[1:]))
            if all(math.isfinite(value) for value in observed[1:]):
                return (observed, poses[backup_index])
        except (TypeError, ValueError):
            pass
    return (poses[main_index], poses[backup_index])


@dataclass(frozen=True)
class ViewpointConvergenceSnapshot:
    pose_id: str
    scan_epoch_id: int
    base_target_reached: bool
    head_target_reached: bool
    camera_settled_after_target_reached: bool
    strict_scan_allowed: bool
    stable_since_monotonic_s: float | None
    convergence_monotonic_s: float | None


class ReplayViewpointConvergenceController:
    """Track one replay pose without weakening any localization gate."""

    def __init__(self, *, base_position_tolerance_m: float,
                 base_yaw_tolerance_rad: float,
                 slide_tolerance_m: float, head_tolerance_rad: float,
                 stable_window_s: float):
        values = (
            base_position_tolerance_m, base_yaw_tolerance_rad,
            slide_tolerance_m, head_tolerance_rad, stable_window_s)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("convergence tolerances must be finite and positive")
        self.base_position_tolerance_m = float(base_position_tolerance_m)
        self.base_yaw_tolerance_rad = float(base_yaw_tolerance_rad)
        self.slide_tolerance_m = float(slide_tolerance_m)
        self.head_tolerance_rad = float(head_tolerance_rad)
        self.stable_window_s = float(stable_window_s)
        self.scan_epoch_id = 0
        self.pose_id = "inactive"
        self.epoch_started_monotonic_s: float | None = None
        self.convergence_monotonic_s: float | None = None
        self._stable_since: float | None = None
        self._base_reached = False
        self._head_reached = False
        self._fresh_after_source_stamp_ns: int | None = None

    def start_pose(self, *, pose_id: str, now_s: float) -> int:
        self.scan_epoch_id += 1
        self.pose_id = str(pose_id)
        self.epoch_started_monotonic_s = float(now_s)
        self.convergence_monotonic_s = None
        self._stable_since = None
        self._base_reached = False
        self._head_reached = False
        self._fresh_after_source_stamp_ns = None
        return self.scan_epoch_id

    def observe(self, *, now_s: float, base_position_error_m: float,
                base_yaw_error_rad: float, head_error: tuple[float, float, float]
                ) -> ViewpointConvergenceSnapshot:
        if self.epoch_started_monotonic_s is None:
            raise RuntimeError("start_pose must be called before observe")
        now_s = float(now_s)
        self._base_reached = (
            abs(float(base_position_error_m))
            <= self.base_position_tolerance_m
            and abs(float(base_yaw_error_rad))
            <= self.base_yaw_tolerance_rad)
        slide, yaw, pitch = (abs(float(value)) for value in head_error)
        self._head_reached = (
            slide <= self.slide_tolerance_m
            and yaw <= self.head_tolerance_rad
            and pitch <= self.head_tolerance_rad)
        if self._base_reached and self._head_reached:
            if self._stable_since is None:
                self._stable_since = now_s
        else:
            self._stable_since = None
            self.convergence_monotonic_s = None
        settled = (
            self._stable_since is not None
            and now_s - self._stable_since >= self.stable_window_s)
        if settled and self.convergence_monotonic_s is None:
            self.convergence_monotonic_s = now_s
        return self.snapshot()

    def set_source_stamp_boundary(self, *source_stamps_ns: int) -> None:
        """Freeze the newest pre-convergence sensor stamp for this epoch."""
        if self.convergence_monotonic_s is None:
            raise RuntimeError("viewpoint must converge before frame boundary")
        stamps = [int(value) for value in source_stamps_ns if value is not None]
        self._fresh_after_source_stamp_ns = max(stamps) if stamps else -1

    def frame_is_fresh(self, source_stamp_ns: int) -> bool:
        """A frame is usable only after convergence in the active epoch."""
        if (self.convergence_monotonic_s is None
                or self._fresh_after_source_stamp_ns is None):
            return False
        return int(source_stamp_ns) > self._fresh_after_source_stamp_ns

    def snapshot(self) -> ViewpointConvergenceSnapshot:
        settled = self.convergence_monotonic_s is not None
        return ViewpointConvergenceSnapshot(
            pose_id=self.pose_id,
            scan_epoch_id=self.scan_epoch_id,
            base_target_reached=self._base_reached,
            head_target_reached=self._head_reached,
            camera_settled_after_target_reached=settled,
            strict_scan_allowed=settled,
            stable_since_monotonic_s=self._stable_since,
            convergence_monotonic_s=self.convergence_monotonic_s)
