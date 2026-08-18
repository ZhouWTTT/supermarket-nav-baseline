#!/usr/bin/env python3
"""Opt-in worker with a heading-constrained first scan-station approach.

The original integrated worker remains the baseline.  This wrapper enables a
short heading-aligned terminal path only during the initial GO_SCAN transit,
then restores the previous controller settings when the first SCAN begins.
It accepts the same CLI arguments as integrated_nav_pick_place.py.
"""

from __future__ import annotations

import json
import math
import os
import time

import integrated_nav_pick_place as integrated
import yolo_aruco_shelf_pick as pick


def _finite_environment_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class TerminalFirstScanIntegratedNavPickPlace(
        integrated.IntegratedNavPickPlace):
    """Integrated worker with a reversible first-transit terminal path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_scan_terminal_distance_m = _finite_environment_float(
            "SUPERMARKET_FIRST_SCAN_TERMINAL_DISTANCE", 0.65)
        self._first_scan_terminal_merge_ahead_m = _finite_environment_float(
            "SUPERMARKET_FIRST_SCAN_TERMINAL_MERGE_AHEAD", 0.30)
        self._first_scan_terminal_release_margin_m = (
            _finite_environment_float(
                "SUPERMARKET_FIRST_SCAN_TERMINAL_RELEASE_MARGIN", 0.12))
        self._first_scan_terminal_previous = None
        self._first_scan_terminal_active = False
        self._first_scan_terminal_complete = False
        self._first_scan_terminal_started_at = None
        self._first_scan_terminal_elapsed_s = None
        self._first_scan_terminal_last_mode = None

    def _initial_scan_transit_active(self) -> bool:
        return bool(
            not self._first_scan_terminal_complete
            and self.flow_phase == "grab"
            and self.state == pick.STATE_GO_SCAN
            and self.target_marker_id is None
            and self.base_xy is not None
            and self.joints
        )

    def _enable_first_scan_terminal(self) -> None:
        if self._first_scan_terminal_active:
            return
        controller = self.nav.controller
        self._first_scan_terminal_previous = (
            controller.terminal_heading_distance_m,
            controller.terminal_heading_merge_ahead_m,
            controller.terminal_heading_release_margin_m,
        )
        controller.configure_terminal_heading_approach(
            self._first_scan_terminal_distance_m,
            self._first_scan_terminal_merge_ahead_m,
            self._first_scan_terminal_release_margin_m,
        )
        self._first_scan_terminal_active = True
        if self._first_scan_terminal_started_at is None:
            self._first_scan_terminal_started_at = time.monotonic()
        self.get_logger().info(
            "[first-scan-terminal] enabled "
            f"distance={self._first_scan_terminal_distance_m:.2f}m "
            f"merge_ahead={self._first_scan_terminal_merge_ahead_m:.2f}m "
            f"release_margin={self._first_scan_terminal_release_margin_m:.2f}m")

    def _disable_first_scan_terminal(self, reason: str) -> None:
        if not self._first_scan_terminal_active:
            return
        controller = self.nav.controller
        previous = self._first_scan_terminal_previous
        controller.configure_terminal_heading_approach(*previous)
        self._first_scan_terminal_previous = None
        self._first_scan_terminal_active = False
        self.get_logger().info(
            "[first-scan-terminal] restored previous navigation settings "
            f"reason={reason}")

    def _log_terminal_plan_mode_change(self) -> None:
        if not self._first_scan_terminal_active:
            return
        status = self.nav.controller.terminal_heading_status()
        mode = status.get("mode")
        if mode == self._first_scan_terminal_last_mode:
            return
        self._first_scan_terminal_last_mode = mode
        self.get_logger().info(
            "[first-scan-terminal] plan="
            + json.dumps(status, ensure_ascii=False, sort_keys=True))

    def timing_snapshot(self) -> dict:
        snapshot = super().timing_snapshot()
        elapsed = self._first_scan_terminal_elapsed_s
        if (elapsed is None
                and self._first_scan_terminal_started_at is not None):
            elapsed = max(
                0.0, time.monotonic() - self._first_scan_terminal_started_at)
        snapshot["first_scan_terminal_approach"] = {
            "completed": self._first_scan_terminal_complete,
            "elapsed_s": None if elapsed is None else round(elapsed, 3),
            "distance_m": self._first_scan_terminal_distance_m,
            "last_plan_mode": self._first_scan_terminal_last_mode,
        }
        return snapshot

    def tick(self) -> None:
        if self._initial_scan_transit_active():
            self._enable_first_scan_terminal()
        else:
            self._disable_first_scan_terminal("outside_initial_go_scan")

        super().tick()
        self._log_terminal_plan_mode_change()

        # The parent changes GO_SCAN -> SCAN only after navigation and camera
        # pose both settle, so this is the end of the measured first transit.
        if (not self._first_scan_terminal_complete
                and self.state == pick.STATE_SCAN):
            self._first_scan_terminal_complete = True
            if self._first_scan_terminal_started_at is not None:
                self._first_scan_terminal_elapsed_s = max(
                    0.0,
                    time.monotonic() - self._first_scan_terminal_started_at)
            self.get_logger().info(
                "[first-scan-terminal] first scan station reached "
                f"elapsed={self._first_scan_terminal_elapsed_s:.3f}s")
            self._disable_first_scan_terminal("first_scan_station_reached")


def main() -> int:
    integrated.IntegratedNavPickPlace = (
        TerminalFirstScanIntegratedNavPickPlace)
    return integrated.main()


if __name__ == "__main__":
    raise SystemExit(main())
