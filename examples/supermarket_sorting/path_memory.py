#!/usr/bin/env python3
"""Lightweight path memory for repeated shelf/delivery transits."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PathMemory:
    def __init__(
        self,
        enabled: bool = False,
        storage_path: str | Path = "/tmp/supermarket_path_memory.json",
        position_bin_m: float = 0.25,
        yaw_bin_rad: float = math.pi / 4.0,
        max_start_offset_m: float = 0.35,
        max_goal_offset_m: float = 0.20,
    ) -> None:
        self.enabled = bool(enabled)
        self.storage_path = Path(storage_path)
        self.position_bin_m = float(position_bin_m)
        self.yaw_bin_rad = float(yaw_bin_rad)
        self.max_start_offset_m = float(max_start_offset_m)
        self.max_goal_offset_m = float(max_goal_offset_m)
        self._catalog: dict[str, dict] = {}
        if self.enabled:
            self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except Exception:
            return
        sessions = raw.get("sessions", {})
        if isinstance(sessions, dict):
            self._catalog = {
                str(key): value for key, value in sessions.items()
                if isinstance(value, dict)
            }

    def _save(self) -> None:
        if not self.enabled:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "sessions": self._catalog,
        }
        self.storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _q(self, value: float, step: float) -> float:
        return round(round(value / step) * step, 3)

    def _pose_bin(self, x: float, y: float, yaw: float | None) -> tuple[float, float, float | None]:
        return (
            self._q(float(x), self.position_bin_m),
            self._q(float(y), self.position_bin_m),
            None if yaw is None else self._q(wrap_to_pi(float(yaw)), self.yaw_bin_rad),
        )

    def make_key(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        goal_x: float,
        goal_y: float,
        goal_yaw: float | None,
    ) -> str:
        sx, sy, syaw = self._pose_bin(start_x, start_y, start_yaw)
        gx, gy, gyaw = self._pose_bin(goal_x, goal_y, goal_yaw)
        return f"s=({sx},{sy},{syaw})__g=({gx},{gy},{gyaw})"

    @staticmethod
    def _distance(ax: float, ay: float, bx: float, by: float) -> float:
        return math.hypot(ax - bx, ay - by)

    def _normalize_path(
        self,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        path: list[tuple[float, float]] | list[list[float]],
    ) -> list[tuple[float, float]]:
        normalized = [(float(x), float(y)) for x, y in path]
        if len(normalized) < 2:
            return normalized
        first = normalized[0]
        last = normalized[-1]
        forward_score = (
            self._distance(start_x, start_y, first[0], first[1])
            + self._distance(goal_x, goal_y, last[0], last[1])
        )
        reverse_score = (
            self._distance(start_x, start_y, last[0], last[1])
            + self._distance(goal_x, goal_y, first[0], first[1])
        )
        if reverse_score < forward_score:
            normalized.reverse()
        return normalized

    def _store_entry(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        goal_x: float,
        goal_y: float,
        goal_yaw: float | None,
        path: list[tuple[float, float]],
        source: str,
    ) -> None:
        key = self.make_key(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw)
        self._catalog[key] = {
            "key": key,
            "saved_at": time.time(),
            "source": source,
            "start_pose": [float(start_x), float(start_y), float(start_yaw)],
            "goal_pose": [
                float(goal_x),
                float(goal_y),
                None if goal_yaw is None else float(goal_yaw),
            ],
            "path": path,
        }

    def _entry_pose(self, entry: dict, which: str) -> tuple[float, float, float | None] | None:
        pose = entry.get(which)
        if not isinstance(pose, list) or len(pose) != 3:
            return None
        try:
            x = float(pose[0])
            y = float(pose[1])
            yaw = None if pose[2] is None else float(pose[2])
        except (TypeError, ValueError):
            return None
        return x, y, yaw

    def _yaw_close(self, want: float | None, have: float | None) -> bool:
        if want is None or have is None:
            return True
        return abs(wrap_to_pi(float(want) - float(have))) <= max(
            self.yaw_bin_rad, math.pi / 6.0)

    def _find_nearby_entry(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        goal_x: float,
        goal_y: float,
        goal_yaw: float | None,
    ) -> tuple[dict | None, dict]:
        best_entry = None
        best_info = {
            "enabled": True,
            "cache_hit": False,
            "key": self.make_key(
                start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw),
            "reason": "no_nearby_match",
        }
        best_score = None
        start_limit = self.max_start_offset_m + self.position_bin_m
        goal_limit = self.max_goal_offset_m + self.position_bin_m
        for key, entry in self._catalog.items():
            if not isinstance(entry, dict):
                continue
            start_pose = self._entry_pose(entry, "start_pose")
            goal_pose = self._entry_pose(entry, "goal_pose")
            if start_pose is None or goal_pose is None:
                continue
            if not self._yaw_close(start_yaw, start_pose[2]):
                continue
            if not self._yaw_close(goal_yaw, goal_pose[2]):
                continue
            start_delta = self._distance(start_x, start_y, start_pose[0], start_pose[1])
            goal_delta = self._distance(goal_x, goal_y, goal_pose[0], goal_pose[1])
            if start_delta > start_limit or goal_delta > goal_limit:
                continue
            score = start_delta + goal_delta
            if best_score is None or score < best_score:
                best_score = score
                best_entry = entry
                best_info = {
                    "enabled": True,
                    "cache_hit": False,
                    "key": self.make_key(
                        start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw),
                    "matched_key": key,
                    "reason": "nearby_match",
                    "start_delta_m": round(start_delta, 3),
                    "goal_delta_m": round(goal_delta, 3),
                }
        return best_entry, best_info

    def load_path(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        goal_x: float,
        goal_y: float,
        goal_yaw: float | None,
    ) -> tuple[list[tuple[float, float]] | None, dict]:
        if not self.enabled:
            return None, {"enabled": False, "cache_hit": False}
        key = self.make_key(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw)
        entry = self._catalog.get(key)
        info = {"enabled": True, "cache_hit": False, "key": key}
        if not isinstance(entry, dict):
            entry, info = self._find_nearby_entry(
                start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw)
            if not isinstance(entry, dict):
                return None, info

        path = entry.get("path", [])
        if not isinstance(path, list) or len(path) < 2:
            info["reason"] = "bad_path"
            return None, info

        # Some planners expose the path from goal->start.  Accept cached paths
        # in either direction and normalize them to start->goal on load.
        try:
            path = self._normalize_path(
                start_x, start_y, goal_x, goal_y, path)
        except Exception:
            pass

        first = path[0]
        last = path[-1]
        if (
            not isinstance(first, (list, tuple))
            or len(first) != 2
            or not isinstance(last, (list, tuple))
            or len(last) != 2
        ):
            info["reason"] = "bad_points"
            return None, info

        start_offset = self._distance(start_x, start_y, float(first[0]), float(first[1]))
        goal_offset = self._distance(goal_x, goal_y, float(last[0]), float(last[1]))
        if start_offset > self.max_start_offset_m:
            info["reason"] = "start_offset"
            info["start_offset_m"] = round(start_offset, 3)
            return None, info
        if goal_offset > self.max_goal_offset_m:
            info["reason"] = "goal_offset"
            info["goal_offset_m"] = round(goal_offset, 3)
            return None, info

        remembered = [(float(x), float(y)) for x, y in path]
        info["cache_hit"] = True
        info["waypoint_count"] = len(remembered)
        info["saved_at"] = entry.get("saved_at")
        return remembered, info

    def save_path(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        goal_x: float,
        goal_y: float,
        goal_yaw: float | None,
        path: Iterable[tuple[float, float]],
        source: str = "planner",
    ) -> None:
        if not self.enabled:
            return
        normalized = self._normalize_path(
            start_x, start_y, goal_x, goal_y, list(path))
        if len(normalized) < 2:
            return
        self._store_entry(
            start_x, start_y, start_yaw,
            goal_x, goal_y, goal_yaw,
            normalized, source)
        self._store_entry(
            goal_x, goal_y,
            0.0 if goal_yaw is None else float(goal_yaw),
            start_x, start_y, start_yaw,
            list(reversed(normalized)),
            f"{source}_reverse")
        self._save()

    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "storage_path": str(self.storage_path),
            "cache_size": len(self._catalog),
        }
