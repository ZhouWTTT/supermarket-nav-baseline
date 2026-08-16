"""Pure-Python task models for the supermarket competition runner.

This module deliberately has no ROS imports.  Task validation and scheduling
can therefore be tested on the host as well as inside the official Client
image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


VALID_KINDS = frozenset({
    "sanmingzhi", "heweidao", "shupian", "zhijin", "maidong",
    "kouxiangtang", "pingguo", "chengzi", "kele",
})

# Prefer reliable, quick single-arm products when no shelf inventory is known.
# This is intentionally a grasp-cost estimate, not a fixed-location lookup.
GRASP_COST = {
    "kele": 1.00,
    "maidong": 1.05,
    "kouxiangtang": 1.10,
    "shupian": 1.15,
    "heweidao": 1.20,
    "sanmingzhi": 1.25,
    "pingguo": 1.40,
    "chengzi": 1.45,
    "zhijin": 1.80,
}

PRODUCT_CENTER_ABOVE_MARKER_M = {
    "sanmingzhi": 0.0434,
    "heweidao": 0.0355,
    "shupian": 0.054,
    "zhijin": 0.043,
    "maidong": 0.104,
    "kele": 0.0715,
    "kouxiangtang": 0.030,
    "pingguo": 0.034,
    "chengzi": 0.036,
}

# 货架 marker 是固定场景件（不随商品随机化移动），每个 ID 的世界 Z 已知。
# 关联时用固定 Z 做"层一致性"校验：marker 固定 Z + 货物中心高 必须与
# YOLO 深度 Z 接近（层间相距 0.35m，0.20 容差可容忍深度偏差又不会跨层）。
# 这样下层货物不会配到顶层 marker，同时不受 marker 实测 Z 噪声影响。
MARKER_Z_CONSISTENCY_TOLERANCE_M = 0.20
_FIXED_MARKER_Z: dict[int, float] | None = None


def fixed_marker_z(marker_id: int) -> float | None:
    """Return the static world Z of an ArUco marker from the fixed layout."""
    global _FIXED_MARKER_Z
    if _FIXED_MARKER_Z is None:
        _FIXED_MARKER_Z = {}
        try:
            layout = json.loads(
                Path(__file__).with_name(
                    "retail_competition_layout.json").read_text())
        except (OSError, ValueError):
            return None
        for slot in layout:
            try:
                marker_id_key = int(slot["aruco_id"])
                level = str(slot["level"])
                z = {
                    "L1": 0.500, "L2": 0.852, "L3": 1.190,
                }.get(level)
                if z is not None:
                    _FIXED_MARKER_Z[marker_id_key] = z
            except (KeyError, TypeError, ValueError):
                continue
    return _FIXED_MARKER_Z.get(int(marker_id))


class TaskMessageError(ValueError):
    """Raised when a competition task message violates schema version 1."""


@dataclass
class Order:
    id: str
    kind: str
    source_index: int
    status: str = "pending"
    attempts: int = 0
    marker_id: int | None = None
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source_index": self.source_index,
            "status": self.status,
            "attempts": self.attempts,
            "marker_id": self.marker_id,
            "errors": list(self.errors),
        }


@dataclass
class CompetitionTask:
    run_prefix: str
    orders: list[Order]
    schema_version: int = 1
    marker_history: dict[str, set[int]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: str) -> "CompetitionTask":
        try:
            document = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise TaskMessageError(f"task is not valid JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise TaskMessageError("task root must be a JSON object")
        if document.get("schema_version") != 1:
            raise TaskMessageError(
                f"unsupported schema_version={document.get('schema_version')!r}")

        run_prefix = document.get("run_prefix")
        if not isinstance(run_prefix, str) or not run_prefix.strip():
            raise TaskMessageError("run_prefix must be a non-empty string")

        targets = document.get("targets")
        count = document.get("count")
        if not isinstance(targets, list):
            raise TaskMessageError("targets must be an array")
        if not isinstance(count, int) or isinstance(count, bool):
            raise TaskMessageError("count must be an integer")
        if count != len(targets):
            raise TaskMessageError(
                f"count={count} does not match targets length={len(targets)}")
        if not 1 <= count <= 45:
            raise TaskMessageError("count must be between 1 and 45")

        orders: list[Order] = []
        seen_ids: set[str] = set()
        for index, target in enumerate(targets):
            if not isinstance(target, dict):
                raise TaskMessageError(f"targets[{index}] must be an object")
            order_id = target.get("id")
            kind = target.get("kind")
            if not isinstance(order_id, str) or not order_id.strip():
                raise TaskMessageError(
                    f"targets[{index}].id must be a non-empty string")
            if order_id in seen_ids:
                raise TaskMessageError(f"duplicate target id: {order_id}")
            if kind not in VALID_KINDS:
                raise TaskMessageError(
                    f"targets[{index}].kind={kind!r} is not supported")
            seen_ids.add(order_id)
            orders.append(Order(order_id, kind, index))
        return cls(run_prefix=run_prefix.strip(), orders=orders)

    def next_order(self, max_attempts: int) -> Order | None:
        candidates = [
            order for order in self.orders
            if order.status == "pending" and order.attempts < max_attempts
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda order: (
                order.attempts,
                GRASP_COST.get(order.kind, 10.0),
                order.source_index,
            ),
        )

    def excluded_markers(self, kind: str) -> list[int]:
        return sorted(self.marker_history.get(kind, set()))

    def finish_attempt(
            self, order: Order, *, delivered: bool,
            marker_id: int | None = None, error: str | None = None,
            max_attempts: int = 2) -> None:
        order.attempts += 1
        if marker_id is not None:
            order.marker_id = int(marker_id)
            self.marker_history.setdefault(order.kind, set()).add(int(marker_id))
        if delivered:
            order.status = "delivered"
            return
        if error:
            order.errors.append(str(error))
        order.status = (
            "failed" if order.attempts >= max_attempts else "pending")

    @property
    def terminal(self) -> bool:
        return all(order.status in {"delivered", "failed"}
                   for order in self.orders)

    def summary(self) -> dict[str, Any]:
        delivered = sum(order.status == "delivered" for order in self.orders)
        failed = sum(order.status == "failed" for order in self.orders)
        return {
            "schema_version": self.schema_version,
            "run_prefix": self.run_prefix,
            "count": len(self.orders),
            "delivered": delivered,
            "failed": failed,
            "pending": len(self.orders) - delivered - failed,
            "orders": [order.as_dict() for order in self.orders],
        }


def marker_arguments(marker_ids: Iterable[int]) -> list[str]:
    arguments: list[str] = []
    for marker_id in sorted(set(int(value) for value in marker_ids)):
        arguments.extend(["--exclude-marker-id", str(marker_id)])
    return arguments


def associate_detection_marker(
        detection: dict[str, Any], markers: Iterable[dict[str, Any]],
        z_tolerance: float = MARKER_Z_CONSISTENCY_TOLERANCE_M,
        planar_tolerance: float = 0.20,
        prefer_measured_marker_z: bool = False) -> dict | None:
    """Associate one YOLO box with the shelf marker directly below it.

    Only public image geometry and measured world coordinates are used.  No
    fixed marker-to-product layout is consulted.

    ``z_tolerance`` / ``planar_tolerance`` relax the world-coordinate gates for
    consumers (e.g. the memory-matrix tracker) whose YOLO depth is noisier than
    the formal alignment pipeline; the defaults keep the original behaviour.

    ``prefer_measured_marker_z``: 布局表里部分 marker ID 与仿真实际摆放错位，
    固定 z 会给出错误的高度一致性校验。设为 True 时改用 marker 实测世界
    高度（位置准确），配合收紧的 z_tolerance 能拒绝跨层/邻列的错误关联。
    """
    try:
        x0, y0, x1, y1 = map(float, detection["bbox_xyxy"])
    except (KeyError, TypeError, ValueError):
        return None
    width, height = x1 - x0, y1 - y0
    if width <= 2.0 or height <= 2.0:
        return None

    centre_x = 0.5 * (x0 + x1)
    minimum_y = y0 + 0.50 * height
    maximum_y = y1 + max(65.0, 1.50 * height)
    detection_world = _finite_xyz(detection.get("world"))
    product_height = PRODUCT_CENTER_ABOVE_MARKER_M.get(detection.get("class"))
    candidates = []
    for marker in markers:
        try:
            marker_id = int(marker["id"])
            marker_x, marker_y = map(float, marker["pixel_center"])
        except (KeyError, TypeError, ValueError):
            continue
        marker_world = _finite_xyz(marker.get("position_world"))
        if not 0 <= marker_id <= 44 or marker_world is None:
            continue
        if marker_y < minimum_y or marker_y > maximum_y:
            continue
        horizontal_margin = 0.35 * width
        if marker_x < x0 - horizontal_margin or marker_x > x1 + horizontal_margin:
            continue
        if detection_world is not None and product_height is not None:
            if prefer_measured_marker_z:
                marker_z_use = float(marker_world[2])
            else:
                marker_z_use = (
                    fixed_marker_z(marker_id)
                    if fixed_marker_z(marker_id) is not None
                    else float(marker_world[2]))
            if (abs(marker_z_use + product_height - detection_world[2])
                    > z_tolerance):
                continue
            planar = ((marker_world[0] - detection_world[0]) ** 2
                      + (marker_world[1] - detection_world[1]) ** 2) ** 0.5
            if planar > planar_tolerance:
                continue
        distance = ((marker_x - centre_x) ** 2 + (marker_y - y1) ** 2) ** 0.5
        candidates.append((distance, marker_id, marker))
    return min(candidates, default=(None, None, None))[2]


def _finite_xyz(value: Any) -> tuple[float, float, float] | None:
    try:
        import math
        xyz = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    if len(xyz) != 3 or not all(math.isfinite(component) for component in xyz):
        return None
    return xyz
