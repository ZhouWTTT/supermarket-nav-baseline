"""Sensor-only persistent obstacle evidence for per-run static boxes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass
class CellEvidence:
    first_hit_s: float
    last_hit_s: float
    hits: int = 0
    free_rays: int = 0
    confirmed: bool = False
    last_observation_id: object | None = None


@dataclass(frozen=True)
class StaticOccupancyMask:
    """Read-only view of the locally submitted static OccupancyGrid."""

    width: int
    height: int
    resolution_m: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    data: Sequence[int]
    occupied_threshold: int = 65

    def is_occupied(self, world_x: float, world_y: float, radius_m: float = 0.0) -> bool:
        if self.resolution_m <= 0.0 or self.width <= 0 or self.height <= 0:
            return False
        dx = float(world_x) - self.origin_x
        dy = float(world_y) - self.origin_y
        cosine = math.cos(self.origin_yaw)
        sine = math.sin(self.origin_yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        center_x = math.floor(local_x / self.resolution_m)
        center_y = math.floor(local_y / self.resolution_m)
        cell_radius = max(0, math.ceil(float(radius_m) / self.resolution_m))
        for gy in range(center_y - cell_radius, center_y + cell_radius + 1):
            if not 0 <= gy < self.height:
                continue
            for gx in range(center_x - cell_radius, center_x + cell_radius + 1):
                if not 0 <= gx < self.width:
                    continue
                if math.hypot(gx - center_x, gy - center_y) > cell_radius + 0.25:
                    continue
                value = int(self.data[gy * self.width + gx])
                if value >= self.occupied_threshold:
                    return True
        return False


class ConfirmedObstacleGrid:
    def __init__(
        self,
        resolution_m: float = 0.10,
        hits_required: int = 3,
        hit_window_s: float = 1.0,
        free_rays_to_clear: int = 5,
    ) -> None:
        if resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive")
        self.resolution_m = float(resolution_m)
        self.hits_required = int(hits_required)
        self.hit_window_s = float(hit_window_s)
        self.free_rays_to_clear = int(free_rays_to_clear)
        self.run_prefix = ""
        self._cells: dict[tuple[int, int], CellEvidence] = {}

    def reset_for_run(self, run_prefix: str) -> bool:
        run_prefix = str(run_prefix)
        if run_prefix == self.run_prefix:
            return False
        self.run_prefix = run_prefix
        self._cells.clear()
        return True

    def key(self, x: float, y: float) -> tuple[int, int]:
        return (
            math.floor(float(x) / self.resolution_m),
            math.floor(float(y) / self.resolution_m),
        )

    def observe_hit(
        self, x: float, y: float, now_s: float, observation_id: object | None = None
    ) -> bool:
        key = self.key(x, y)
        evidence = self._cells.get(key)
        if evidence is None or now_s - evidence.first_hit_s > self.hit_window_s:
            evidence = CellEvidence(now_s, now_s)
            self._cells[key] = evidence
        if observation_id is not None and evidence.last_observation_id == observation_id:
            return evidence.confirmed
        evidence.last_hit_s = float(now_s)
        evidence.hits += 1
        evidence.free_rays = 0
        evidence.last_observation_id = observation_id
        if evidence.hits >= self.hits_required:
            evidence.confirmed = True
        return evidence.confirmed

    def observe_free(self, x: float, y: float) -> bool:
        key = self.key(x, y)
        evidence = self._cells.get(key)
        if evidence is None:
            return False
        evidence.free_rays += 1
        if evidence.free_rays >= self.free_rays_to_clear:
            del self._cells[key]
            return True
        return False

    def confirmed_points(self) -> list[tuple[float, float]]:
        points = []
        half = 0.5 * self.resolution_m
        for (gx, gy), evidence in self._cells.items():
            if evidence.confirmed:
                points.append(
                    (gx * self.resolution_m + half, gy * self.resolution_m + half)
                )
        return points

    def clustered_confirmed_points(
        self, minimum_cells: int = 2
    ) -> list[tuple[float, float]]:
        """Return confirmed cells belonging to a spatially valid point cluster."""

        confirmed = {
            key for key, evidence in self._cells.items() if evidence.confirmed
        }
        if minimum_cells <= 1:
            selected = confirmed
        else:
            selected = set()
            for gx, gy in confirmed:
                neighborhood = sum(
                    (gx + dx, gy + dy) in confirmed
                    for dx in (-1, 0, 1)
                    for dy in (-1, 0, 1)
                )
                if neighborhood >= minimum_cells:
                    selected.add((gx, gy))
        half = 0.5 * self.resolution_m
        return [
            (gx * self.resolution_m + half, gy * self.resolution_m + half)
            for gx, gy in sorted(selected)
        ]

    def is_confirmed(self, x: float, y: float) -> bool:
        evidence = self._cells.get(self.key(x, y))
        return bool(evidence and evidence.confirmed)
