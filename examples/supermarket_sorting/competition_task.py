"""Pure-Python task models for the supermarket competition runner.

This module deliberately has no ROS imports.  Task validation and scheduling
can therefore be tested on the host as well as inside the official Client
image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
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


def prefer_non_rerouted(
        candidates: list[Order], rerouted_order_ids: Iterable[str]) -> list[Order]:
    """Prefer orders that were not just abandoned after a wrong localisation.

    A mislocalised order is deferred instead of retried immediately, so the
    next worker grabs the nearest *other* pending product.  The deferred
    order is still returned when it is the only candidate left, so it is
    never permanently lost.
    """
    excluded = set(rerouted_order_ids or ())
    if not excluded:
        return candidates
    fresh = [order for order in candidates if order.id not in excluded]
    return fresh if fresh else candidates


def associate_detection_marker(
        detection: dict[str, Any], markers: Iterable[dict[str, Any]]) -> dict | None:
    """Associate one YOLO box with the shelf marker directly below it.

    Only public image geometry and measured world coordinates are used.  No
    fixed marker-to-product layout is consulted.
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
            if abs(marker_world[2] + product_height - detection_world[2]) > 0.16:
                continue
            planar = ((marker_world[0] - detection_world[0]) ** 2
                      + (marker_world[1] - detection_world[1]) ** 2) ** 0.5
            if planar > 0.20:
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
