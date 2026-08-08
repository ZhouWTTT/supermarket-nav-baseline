#!/usr/bin/env python3
"""Observe-only quality gate for supermarket perception topics.

The node records YOLO and three-camera ArUco observations as JSON lines.  It
never sends robot commands.  ArUco's JSON topic has no ROS header, so its
timestamp is explicitly recorded as receive time and its frame comes from the
documented topic contract (head_camera/left_camera/right_camera).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray


STALE_AFTER_S = 2.0
MIN_YOLO_SAMPLES = 3
MAX_YOLO_SPREAD_M = 0.15
MIN_YOLO_CONFIDENCE = 0.25
HISTORY_SIZE = 10

ARUCO_SOURCES = {
    "aruco_head": ("/aruco/head/detections", "head_camera"),
    "aruco_left": ("/aruco/left/detections", "left_camera"),
    "aruco_right": ("/aruco/right/detections", "right_camera"),
}


def finite_position(position: Iterable[float]) -> bool:
    values = list(position)
    return len(values) == 3 and all(math.isfinite(float(value)) for value in values)


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def position_spread(history: Deque[Tuple[float, float, float]]) -> float:
    """Return the largest axis range in a recent position history."""
    if len(history) < 2:
        return 0.0
    axes = zip(*history)
    return max(max(axis) - min(axis) for axis in axes)


class PerceptionObserver(Node):
    def __init__(self) -> None:
        super().__init__("perception_observer")

        self.last_receive: Dict[str, Optional[float]] = {
            "kele": None,
            **{source: None for source in ARUCO_SOURCES},
        }
        self.health_state: Dict[str, Optional[str]] = {
            source: None for source in self.last_receive
        }
        self.last_empty_emit: Dict[str, float] = defaultdict(lambda: -math.inf)
        self.histories: Dict[str, Deque[Tuple[float, float, float]]] = defaultdict(
            lambda: deque(maxlen=HISTORY_SIZE)
        )

        self.create_subscription(
            Detection3DArray, "/kele/detections", self.on_kele, 10
        )
        for source, (topic, frame) in ARUCO_SOURCES.items():
            self.create_subscription(
                String,
                topic,
                lambda message, src=source, frm=frame: self.on_aruco(
                    src, frm, message
                ),
                10,
            )

        self.create_timer(0.5, self.check_source_health)
        self.get_logger().info(
            "observe-only perception quality gate started; stale_after=%.1fs"
            % STALE_AFTER_S
        )

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def emit(
        self,
        *,
        source: str,
        stamp: float,
        stamp_source: str,
        frame: str,
        class_or_marker,
        position,
        quality: dict,
        accepted: bool,
        reject_reason: str,
    ) -> None:
        record = {
            "receive_time": round(self.now_seconds(), 6),
            "source": source,
            "stamp": round(float(stamp), 6),
            "stamp_source": stamp_source,
            "frame": frame,
            "class_or_marker": class_or_marker,
            "position": position,
            "quality": quality,
            "accepted": bool(accepted),
            "reject_reason": reject_reason,
        }
        print(json.dumps(record, ensure_ascii=False, separators=(",", ":")), flush=True)

    def mark_received(self, source: str, receive_time: float) -> None:
        self.last_receive[source] = receive_time
        self.health_state[source] = "fresh"

    def should_emit_empty(self, source: str, now: float) -> bool:
        """Keep heartbeat evidence without writing empty frames at camera rate."""
        if now - self.last_empty_emit[source] < 1.0:
            return False
        self.last_empty_emit[source] = now
        return True

    def on_kele(self, message: Detection3DArray) -> None:
        receive_time = self.now_seconds()
        self.mark_received("kele", receive_time)
        frame = message.header.frame_id
        stamp = stamp_to_seconds(message.header.stamp)
        age = max(0.0, receive_time - stamp) if stamp > 0.0 else float("inf")

        if not message.detections:
            if not self.should_emit_empty("kele", receive_time):
                return
            self.emit(
                source="kele",
                stamp=stamp,
                stamp_source="message_header",
                frame=frame,
                class_or_marker=None,
                position=None,
                quality={"age_s": round(age, 4), "observation_count": 0},
                accepted=False,
                reject_reason="no_detection",
            )
            return

        for detection in message.detections:
            if not detection.results:
                self.emit(
                    source="kele",
                    stamp=stamp,
                    stamp_source="message_header",
                    frame=frame,
                    class_or_marker=None,
                    position=None,
                    quality={"age_s": round(age, 4), "observation_count": 0},
                    accepted=False,
                    reject_reason="missing_hypothesis",
                )
                continue

            result = detection.results[0]
            class_id = str(result.hypothesis.class_id)
            score = float(result.hypothesis.score)
            point = result.pose.pose.position
            position = (float(point.x), float(point.y), float(point.z))
            key = "kele:%s" % class_id
            if finite_position(position):
                self.histories[key].append(position)
            history = self.histories[key]
            spread = position_spread(history)

            reasons: List[str] = []
            if frame != "world":
                reasons.append("wrong_frame")
            if stamp <= 0.0:
                reasons.append("invalid_stamp")
            elif age > STALE_AFTER_S:
                reasons.append("stale")
            if class_id != "kele":
                reasons.append("task_class_mismatch")
            if not finite_position(position):
                reasons.append("non_finite_position")
            elif not (-5.0 <= position[0] <= 5.0 and -5.0 <= position[1] <= 5.0
                      and 0.1 <= position[2] <= 2.5):
                reasons.append("outside_workspace")
            if not math.isfinite(score) or score < MIN_YOLO_CONFIDENCE:
                reasons.append("low_confidence")
            if len(history) < MIN_YOLO_SAMPLES:
                reasons.append("insufficient_history")
            elif spread > MAX_YOLO_SPREAD_M:
                reasons.append("unstable_multiframe")

            self.emit(
                source="kele",
                stamp=stamp,
                stamp_source="message_header",
                frame=frame,
                class_or_marker=class_id,
                position=[round(value, 6) for value in position],
                quality={
                    "confidence": round(score, 4),
                    "age_s": round(age, 4),
                    "depth_valid": finite_position(position) and position[2] > 0.0,
                    "observation_count": len(history),
                    "spread_m": round(spread, 6),
                },
                accepted=not reasons,
                reject_reason="accepted" if not reasons else ",".join(reasons),
            )

    def on_aruco(self, source: str, frame: str, message: String) -> None:
        receive_time = self.now_seconds()
        self.mark_received(source, receive_time)
        try:
            records = json.loads(message.data)
            if not isinstance(records, list):
                raise ValueError("root is not a list")
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            self.emit(
                source=source,
                stamp=receive_time,
                stamp_source="receive_time",
                frame=frame,
                class_or_marker=None,
                position=None,
                quality={"parse_error": str(error)},
                accepted=False,
                reject_reason="invalid_json",
            )
            return

        if not records:
            if not self.should_emit_empty(source, receive_time):
                return
            self.emit(
                source=source,
                stamp=receive_time,
                stamp_source="receive_time",
                frame=frame,
                class_or_marker=None,
                position=None,
                quality={"observation_count": 0},
                accepted=False,
                reject_reason="no_detection",
            )
            return

        for record in records:
            marker_id = record.get("id")
            raw_position = record.get("position")
            valid_id = isinstance(marker_id, int) and 0 <= marker_id <= 44
            valid_point = isinstance(raw_position, list) and finite_position(raw_position)
            position = [float(value) for value in raw_position] if valid_point else None
            reasons: List[str] = []
            if not valid_id:
                reasons.append("invalid_marker_id")
            if not valid_point:
                reasons.append("non_finite_position")
            elif position[2] <= 0.0 or position[2] > 5.0:
                reasons.append("outside_camera_range")

            self.emit(
                source=source,
                stamp=receive_time,
                stamp_source="receive_time",
                frame=frame,
                class_or_marker=marker_id,
                position=[round(value, 6) for value in position] if position else None,
                quality={
                    "depth_valid": bool(valid_point and position[2] > 0.0),
                    "observation_count": 1,
                    "note": "ArUco JSON has no header; frame uses topic contract",
                },
                accepted=not reasons,
                reject_reason="accepted" if not reasons else ",".join(reasons),
            )

    def check_source_health(self) -> None:
        now = self.now_seconds()
        for source, last_time in self.last_receive.items():
            if last_time is None:
                state = "source_missing"
                age = None
            else:
                age = max(0.0, now - last_time)
                state = "stale" if age > STALE_AFTER_S else "fresh"

            if state == self.health_state[source]:
                continue
            self.health_state[source] = state
            if state == "fresh":
                continue
            self.emit(
                source=source,
                stamp=last_time or 0.0,
                stamp_source="last_receive_time",
                frame="",
                class_or_marker=None,
                position=None,
                quality={"age_s": None if age is None else round(age, 4)},
                accepted=False,
                reject_reason=state,
            )


def main() -> None:
    rclpy.init()
    node = PerceptionObserver()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
