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
        if not isinstance(entry, dict):
            return None, {"enabled": True, "cache_hit": False, "key": key}

        path = entry.get("path", [])
        if not isinstance(path, list) or len(path) < 2:
            return None, {"enabled": True, "cache_hit": False, "key": key, "reason": "bad_path"}

        # Some planners expose the path from goal->start.  Accept cached paths
        # in either direction and normalize them to start->goal on load.
        if len(path) >= 2:
            try:
                first = path[0]
                last = path[-1]
                forward_score = (
                    self._distance(start_x, start_y, float(first[0]), float(first[1]))
                    + self._distance(goal_x, goal_y, float(last[0]), float(last[1]))
                )
                reverse_score = (
                    self._distance(start_x, start_y, float(last[0]), float(last[1]))
                    + self._distance(goal_x, goal_y, float(first[0]), float(first[1]))
                )
                if reverse_score < forward_score:
                    path = list(reversed(path))
            except Exception:
                pass

        first = path[0]
        last = path[-1]
        if (
            not isinstance(first, list)
            or len(first) != 2
            or not isinstance(last, list)
            or len(last) != 2
        ):
            return None, {"enabled": True, "cache_hit": False, "key": key, "reason": "bad_points"}

        start_offset = self._distance(start_x, start_y, float(first[0]), float(first[1]))
        goal_offset = self._distance(goal_x, goal_y, float(last[0]), float(last[1]))
        if start_offset > self.max_start_offset_m:
            return None, {
                "enabled": True,
                "cache_hit": False,
                "key": key,
                "reason": "start_offset",
                "start_offset_m": round(start_offset, 3),
            }
        if goal_offset > self.max_goal_offset_m:
            return None, {
                "enabled": True,
                "cache_hit": False,
                "key": key,
                "reason": "goal_offset",
                "goal_offset_m": round(goal_offset, 3),
            }

        remembered = [(float(x), float(y)) for x, y in path]
        return remembered, {
            "enabled": True,
            "cache_hit": True,
            "key": key,
            "waypoint_count": len(remembered),
            "saved_at": entry.get("saved_at"),
        }

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
        normalized = [(float(x), float(y)) for x, y in path]
        if len(normalized) < 2:
            return
        # Normalize to start->goal before saving so future runs can match the
        # first waypoint against the live start pose.
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
        key = self.make_key(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw)
        self._catalog[key] = {
            "key": key,
            "saved_at": time.time(),
            "source": source,
            "path": normalized,
        }
        self._save()

    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "storage_path": str(self.storage_path),
            "cache_size": len(self._catalog),
        }
