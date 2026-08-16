#!/usr/bin/env python3
"""3x15 memory matrix tracker for the supermarket sorting task.

Rows  = shelf levels (L1, L2, L3)
Cols  = shelf x column (A1..A3, B1..B3, C1..C3, D1..D3, E1..E3) -> 15 cols

记忆矩阵只依赖 YOLO + 深度世界坐标 + 固定货架几何，不依赖 ArUco：

* 机器人只需位于货架前的观察走廊；一帧里每个 YOLO 检测都按自身
  世界 x/z 独立映射到全局货架网格，不再把“当前站点”当成“目标货架”；
* 因此画面同时看到 A/C/E 等多个货架时，可以在同一批 YOLO 结果中
  同时累积多个货架的记忆；
* 同一 (货架, 层, 列, 种类) 仍需多帧中位数确认后才写入；
* 每格按种类分别保留候选证据。观测距离优先：进入更近的距离带就更新，
  同一近距带内保留置信度最高的一次；远处观测只累计次数，不能覆盖近距证据；
* ArUco/marker 完全不参与矩阵：marker 只在抓取货物时用于精确定位。

矩阵写入 ``<repo>/logs/memory_matrix.json`` 供宿主机 GUI 渲染。
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

try:  # ROS only needed for the live tracker; pure matrix logic stays host-safe
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from nav_msgs.msg import Odometry
except ImportError:  # pragma: no cover - host / pure-python contexts
    rclpy = None
    Node = object
    String = None
    Odometry = None


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LAYOUT_PATH = HERE / "retail_competition_layout.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "logs" / "memory_matrix.json"

SHELVES = ("A", "B", "C", "D", "E")
COLUMNS = ("1", "2", "3")
LEVELS = ("L1", "L2", "L3")

# 货架字母 -> 扫描站点 x（货架中线，与导航 SCAN_X 一致）
SHELF_SCAN_X = {
    "A": -1.735, "B": -0.850, "C": 0.035,
    "D": 0.920, "E": 1.800,
}
# 层 -> 固定层高（用于导航直达提示的 z，也是层判定的参照）
LEVEL_MARKER_Z = {"L1": 0.500, "L2": 0.852, "L3": 1.190}
# 同一货架内 3 列的固定 x 偏移（相对货架中线）。
COLUMN_X_OFFSET = {"1": -0.22, "2": 0.00, "3": 0.22}
# 列判定容差：YOLO 深度 x 与固定列中心偏差超过该值就不采信（丢弃斜视角
# 余光/邻架干扰）。列中心间距 0.22m、列间边界距最近列中心 0.11m，容差取
# 0.14 可覆盖整架宽度，但对站点容差边缘之外的离群 x 仍能拦截。
COLUMN_X_TOLERANCE_M = 0.14
# 层判定阈值（YOLO 深度 z，单位 m）：按固定层中心取层间中点
# （L1≈0.58, L2≈0.92, L3≈1.24）。
LEVEL_Z_L3_MIN = 1.09
LEVEL_Z_L2_MIN = 0.75
# 每层可采信的深度区间：只有深度中位数落在该层固定几何带内才记录。
# 落在层间死区（0.72~0.78 / 1.02~1.12）的帧视为层判定不确定，直接丢弃，
# 避免同一个箱子因深度抖动被记到相邻层导致导航跑错层。
LEVEL_Z_RANGES = {
    "L1": (0.50, 0.72),
    "L2": (0.78, 1.02),
    "L3": (1.12, 1.34),
}

# 只在机器人位于货架前观察走廊时才信任 YOLO 深度。横向范围
# 覆盖 A..E 整条通道，不要求机器人停在某个货架中心。
STATION_Y_MIN = 2.20
STATION_Y_MAX = 2.75
OBSERVATION_X_MARGIN_M = 0.55
# 货物都在货架平面 y≈3.24m 上。YOLO 深度得到的是可见前表面，因此给
# 予较宽容差，但拒绝走廊、机器人和配送区中的同类物体污染矩阵。
SHELF_OBSERVATION_Y_MIN = 2.95
SHELF_OBSERVATION_Y_MAX = 3.50
# 同槽位最少样本数（多帧中位数抑制单帧深度噪声）。
DEPTH_MIN_SAMPLES = 4
# 近距带宽：小于该差值视为同一观测距离带，带内按置信度选代表值。
# 这样既不会因 1~2cm 深度抖动来回替换，也会在机器人明显靠近后刷新证据。
OBSERVATION_DISTANCE_BAND_M = 0.15
# 无距离信息的旧兼容调用仍使用原防覆盖余量。
OVERWRITE_CONF_MARGIN = 0.02


def _load_slot_map() -> dict[int, tuple[str, str, str]]:
    """marker_id -> (shelf, level, column) from the fixed layout JSON.

    保留仅为兼容旧调用（competition_runner 等）；矩阵写入本身不再使用。
    """
    result: dict[int, tuple[str, str, str]] = {}
    try:
        layout = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return result
    for slot in layout:
        try:
            marker_id = int(slot["aruco_id"])
            column_text = str(slot["column"])
            column = (
                column_text[-1] if column_text[-1:].isdigit()
                else column_text)
            result[marker_id] = (
                str(slot["shelf"]), str(slot["level"]), column)
        except (KeyError, TypeError, ValueError):
            continue
    return result


SLOT_BY_MARKER = _load_slot_map()
COL_LABELS = [
    f"{shelf}{column}" for shelf in SHELVES for column in COLUMNS]


def fixed_slot_from_world(
        x: float, z: float,
        shelf: str | None = None) -> tuple[str, str, str] | None:
    """把 YOLO 深度世界坐标映射到固定货架网格 (shelf, level, column)。

    矩阵主路径不传 ``shelf``，使每个检测按自身世界 x 选择最近的
    全局货架；显式传入仅保留给抓取阶段的局部校验。层/列都用固定几何
    判定，列必须落在固定列中心容差内，否则返回 None。
    """
    try:
        x = float(x)
        z = float(z)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(z)):
        return None
    if shelf is None:
        shelf = min(SHELF_SCAN_X, key=lambda s: abs(SHELF_SCAN_X[s] - x))
    if shelf not in SHELF_SCAN_X:
        return None
    shelf_x = SHELF_SCAN_X[shelf]
    level = (
        "L3" if z >= LEVEL_Z_L3_MIN
        else ("L2" if z >= LEVEL_Z_L2_MIN else "L1"))
    z_min, z_max = LEVEL_Z_RANGES[level]
    if not (z_min <= z <= z_max):
        return None
    column = min(
        COLUMNS,
        key=lambda c: abs(x - (shelf_x + COLUMN_X_OFFSET[c])))
    if abs(x - (shelf_x + COLUMN_X_OFFSET[column])) > COLUMN_X_TOLERANCE_M:
        return None
    return (shelf, level, column)


def slot_key(shelf: str, level: str, column: str) -> str:
    return f"{level}|{shelf}|{column}"


class MemoryMatrix:
    """3x15 主网格，以及每个槽位按品类保存的 YOLO 候选证据。"""

    def __init__(self) -> None:
        # cell_key = f"{level}|{shelf}|{column}"
        self.cells: dict[str, dict[str, Any]] = {}
        # cell_key -> kind -> candidate。cells 是 GUI 与导航共用的当前
        # 主证据；candidates 仅保留诊断/重新观测所需的多品类历史。
        self.candidates: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _cell_key(level: str, shelf: str, column: str) -> str:
        return f"{level}|{shelf}|{column}"

    def record(
            self, marker_id: int, kind: str, confidence: float) -> bool:
        """按 marker 布局映射记录（兼容旧调用；矩阵主线不再走这里）。"""
        slot = SLOT_BY_MARKER.get(int(marker_id))
        if slot is None:
            return False
        shelf, level, column = slot
        return self.record_at(
            shelf, level, column,
            marker_id, kind, confidence)

    def record_at(
            self, shelf: str, level: str, column: str,
            marker_id: int = -1, kind: str = "", confidence: float = 0.0,
            observation_distance: float | None = None,
            observer_xy: tuple[float, float] | None = None,
            observed_at: float | None = None,
            sample_count: int = 1,
    ) -> bool:
        """按显式槽位记录一批已完成多帧确认的观测。

        有距离的 YOLO 主路径采用“最近距离带优先、带内置信度优先”。同格的
        不同品类不会相互删除，而是分别保留候选。没有距离的旧调用继续使用
        原来的置信度防覆盖规则，以免改变兼容调用的行为。
        """
        key = self._cell_key(level, shelf, column)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        if not math.isfinite(confidence):
            confidence = 0.0
        try:
            distance = float(observation_distance)
        except (TypeError, ValueError):
            distance = None
        if distance is not None and (
                not math.isfinite(distance) or distance <= 0.0):
            distance = None
        now = float(observed_at if observed_at is not None else time.time())

        with self._lock:
            previous = self.cells.get(key)
            if previous is not None and previous.get("consumed"):
                return False

            # 兼容无距离的旧调用：仍按原单赢家规则决定是否接受。
            if distance is None and previous is not None:
                prev_kind = str(previous.get("kind", ""))
                prev_conf = float(previous.get("confidence", 0.0))
                if prev_kind != kind:
                    if confidence < prev_conf + OVERWRITE_CONF_MARGIN:
                        return False
                elif confidence < prev_conf:
                    return False

            slot_candidates = self.candidates.setdefault(key, {})
            candidate = slot_candidates.get(kind)
            representative_updated = False
            if candidate is None:
                candidate = {
                    "kind": kind,
                    "marker_id": int(marker_id),
                    "shelf": shelf,
                    "level": level,
                    "column": column,
                    "confidence": round(confidence, 3),
                    "closest_distance": (
                        None if distance is None else round(distance, 3)),
                    # 代表置信度是在哪个距离获得的；它不一定恰好等于最短
                    # 距离，但一定处于当前最短距离 + 近距带之内。
                    "observation_distance": (
                        None if distance is None else round(distance, 3)),
                    "observations": 1,
                    "sample_count": max(1, int(sample_count)),
                    "last_seen": now,
                }
                slot_candidates[kind] = candidate
                representative_updated = True
            else:
                candidate["observations"] = int(
                    candidate.get("observations", 0)) + 1
                candidate["sample_count"] = int(
                    candidate.get("sample_count", 0)) + max(
                        1, int(sample_count))
                candidate["last_seen"] = now
                candidate["marker_id"] = int(marker_id)
                old_closest = candidate.get("closest_distance")
                old_rep_distance = candidate.get("observation_distance")
                old_confidence = float(candidate.get("confidence", 0.0))

                if distance is None:
                    if confidence > old_confidence:
                        candidate["confidence"] = round(confidence, 3)
                        representative_updated = True
                else:
                    try:
                        old_closest = float(old_closest)
                    except (TypeError, ValueError):
                        old_closest = None
                    new_closest = (
                        distance if old_closest is None
                        else min(old_closest, distance))
                    candidate["closest_distance"] = round(new_closest, 3)
                    try:
                        old_rep_distance = float(old_rep_distance)
                    except (TypeError, ValueError):
                        old_rep_distance = None
                    old_still_near = (
                        old_rep_distance is not None
                        and old_rep_distance
                        <= new_closest + OBSERVATION_DISTANCE_BAND_M)
                    new_is_near = (
                        distance
                        <= new_closest + OBSERVATION_DISTANCE_BAND_M)
                    if (not old_still_near
                            or (new_is_near and confidence > old_confidence)):
                        candidate["confidence"] = round(confidence, 3)
                        candidate["observation_distance"] = round(
                            distance, 3)
                        representative_updated = True

            if representative_updated and observer_xy is not None:
                try:
                    observer_x, observer_y = (
                        float(observer_xy[0]), float(observer_xy[1]))
                    if math.isfinite(observer_x) and math.isfinite(observer_y):
                        candidate["observer_x"] = round(observer_x, 3)
                        candidate["observer_y"] = round(observer_y, 3)
                except (TypeError, ValueError, IndexError):
                    pass

            self._select_primary(key)
            return True

    def _select_primary(self, key: str) -> None:
        """把槽位候选折叠成 GUI 使用的单个主候选。调用方须持有锁。"""
        values = list(self.candidates.get(key, {}).values())
        if not values:
            self.cells.pop(key, None)
            return
        with_distance = [
            item for item in values
            if item.get("closest_distance") is not None]
        if with_distance:
            nearest = min(float(item["closest_distance"])
                          for item in with_distance)
            near_values = [
                item for item in with_distance
                if float(item["closest_distance"])
                <= nearest + OBSERVATION_DISTANCE_BAND_M]
            primary = max(
                near_values,
                key=lambda item: (
                    float(item.get("confidence", 0.0)),
                    int(item.get("sample_count", 0)),
                    float(item.get("last_seen", 0.0))))
        else:
            primary = max(
                values,
                key=lambda item: float(item.get("confidence", 0.0)))
        consumed = bool(self.cells.get(key, {}).get("consumed", False))
        self.cells[key] = {
            name: value for name, value in primary.items()
            if name not in ("observations", "sample_count", "last_seen")
        }
        self.cells[key]["consumed"] = consumed

    def candidates_for(self, kind: str) -> list[dict[str, Any]]:
        """返回某品类所有未消费的历史候选（诊断用）。"""
        result = []
        with self._lock:
            for key, by_kind in self.candidates.items():
                if self.cells.get(key, {}).get("consumed"):
                    continue
                candidate = by_kind.get(kind)
                if candidate is None:
                    continue
                item = dict(candidate)
                item["slot_key"] = key
                result.append(item)
        return result

    def primary_candidates_for(self, kind: str) -> list[dict[str, Any]]:
        """返回与 GUI 主矩阵一致的未消费导航证据。

        隐藏历史候选不得单独触发导航。这避免同一物体因深度 x
        抖动跨到相邻列后，GUI 已显示为其他品类，导航却仍然读到
        旧品类的矛盾。
        """
        result = []
        with self._lock:
            for key, cell in self.cells.items():
                if (cell.get("consumed")
                        or str(cell.get("kind", "")) != kind):
                    continue
                candidate = self.candidates.get(key, {}).get(kind)
                if candidate is None:
                    continue
                item = dict(candidate)
                item["slot_key"] = key
                result.append(item)
        return result

    def consume(self, marker_id: int) -> bool:
        """按 marker_id 消费（兼容旧调用；YOLO 路径用 consume_slot）。"""
        marker_id = int(marker_id)
        with self._lock:
            for cell in self.cells.values():
                if cell.get("marker_id") == marker_id:
                    cell["consumed"] = True
                    return True
        return False

    def consume_slot(
            self, shelf: str, level: str, column: str,
            kind: str | None = None) -> bool:
        """按固定槽位消费，并清理同层同品类的同期副本。

        列只是录入网格，不是导航身份。同一物体在抓取前可能因
        深度 x 抖动被写入同货架同层的相邻列；已确认抓走后，
        这些旧副本一并失效。若该层确有第二件同类货物，后续
        YOLO 新观测会在未消费槽位重新建立候选。
        """
        key = self._cell_key(level, shelf, column)
        with self._lock:
            cell = self.cells.get(key)
            changed = False
            if cell is not None:
                cell["consumed"] = True
                changed = True
            if kind:
                prefix = f"{level}|{shelf}|"
                affected = []
                for other_key, by_kind in list(self.candidates.items()):
                    if not other_key.startswith(prefix):
                        continue
                    if other_key == key and cell is not None:
                        continue
                    if by_kind.pop(kind, None) is None:
                        continue
                    changed = True
                    affected.append(other_key)
                    if not by_kind:
                        self.candidates.pop(other_key, None)
                for other_key in affected:
                    self._select_primary(other_key)
            return changed

    def clear(self) -> None:
        with self._lock:
            self.cells.clear()
            self.candidates.clear()

    def to_json(self, updated_at: float | None = None) -> dict[str, Any]:
        grid = {level: [None] * len(COL_LABELS) for level in LEVELS}
        cells: dict[str, dict[str, Any]] = {}
        with self._lock:
            for key, cell in self.cells.items():
                cells[key] = dict(cell)
                try:
                    level, shelf, column = key.split("|")
                    col_index = COL_LABELS.index(f"{shelf}{column}")
                    grid[level][col_index] = {
                        "kind": cell["kind"],
                        "consumed": bool(cell.get("consumed", False)),
                    }
                except (ValueError, KeyError):
                    continue
            candidates = {
                key: {kind: dict(candidate)
                      for kind, candidate in by_kind.items()}
                for key, by_kind in self.candidates.items()
            }
        return {
            "updated_at": (
                updated_at if updated_at is not None else time.time()),
            "rows": list(LEVELS),
            "cols": COL_LABELS,
            "cells": cells,
            "candidates": candidates,
            "grid": grid,
        }


class MemoryMatrixTracker(Node):
    """Subscribe to YOLO + odom and keep the matrix fresh (no ArUco)."""

    def __init__(
            self, confirmations: int = 2,
            output_path: Path | None = None,
            write_interval_s: float = 0.5) -> None:
        super().__init__("memory_matrix_tracker")
        self.matrix = MemoryMatrix()
        self.confirmations = max(1, int(confirmations))
        self.output_path = Path(
            output_path or DEFAULT_OUTPUT_PATH)
        self.write_interval_s = float(write_interval_s)
        # 全局深度采样累加器：
        # (shelf, level, column, kind) ->
        # {x[], y[], z[], distance[], conf, last_stamp_ns}
        self._slot_acc: dict[
            tuple[str, str, str, str], dict[str, Any]] = {}
        # 防覆盖日志限流：slot_key -> 上次记录时间
        self._blocked_log_at: dict[str, float] = {}
        self.base_xy: tuple[float, float] | None = None
        self._last_write_at = 0.0
        self._dirty = False
        self.create_subscription(
            String, "/goods/yolo_detections", self._yolo_cb, 10)
        if Odometry is not None:
            self.create_subscription(
                Odometry, "/slamware_ros_sdk_server_node/odom",
                self._odom_cb, 10)
        self.create_timer(write_interval_s, self.tick_write)
        self.get_logger().info(
            "memory matrix tracker ready; yolo-only (no ArUco); output="
            f"{self.output_path} confirmations={self.confirmations} "
            f"depth_min_samples={DEPTH_MIN_SAMPLES}")

    def _odom_cb(self, message) -> None:
        try:
            x = float(message.pose.pose.position.x)
            y = float(message.pose.pose.position.y)
        except (AttributeError, TypeError, ValueError):
            return
        self.base_xy = (x, y)

    def _in_shelf_observation_corridor(self) -> bool:
        """机器人是否位于可观察整排货架的通道内。"""
        if self.base_xy is None:
            return False
        x, y = self.base_xy
        return (
            STATION_Y_MIN <= y <= STATION_Y_MAX
            and min(SHELF_SCAN_X.values()) - OBSERVATION_X_MARGIN_M
            <= x
            <= max(SHELF_SCAN_X.values()) + OBSERVATION_X_MARGIN_M
        )

    @staticmethod
    def _decode_records(message: String) -> list[dict]:
        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError, ValueError):
            return []
        return [item for item in value if isinstance(item, dict)] \
            if isinstance(value, list) else []

    def _yolo_cb(self, message: String) -> None:
        records = [
            item for item in self._decode_records(message)
            if item.get("camera", "head") == "head"]
        if not records:
            return
        for detection in records:
            self._record_yolo_only(detection)

    def _record_yolo_only(self, detection: dict) -> None:
        """纯 YOLO 录制：每个检测的世界坐标独立映射到全局固定网格。

        marker 完全不参与。机器人只需位于货架观察走廊，不需要停在被检测
        货架的标准站点；因此同一帧可以同时给多个货架累积样本。
        """
        kind = detection.get("class")
        if not isinstance(kind, str):
            return
        if not self._in_shelf_observation_corridor():
            return
        # ``world`` 是 YOLO 框中心的同步深度点，保持为主来源以延续现有
        # 层高标定；中心深度缺失时再使用 ROI 前景中位点。
        measured_world = detection.get("world")
        if measured_world is None:
            measured_world = detection.get("front_world")
        try:
            world = [float(v) for v in measured_world]
        except (TypeError, ValueError, KeyError):
            return
        if len(world) != 3 or not all(
                math.isfinite(v) for v in world):
            return
        x, y, z = world
        if not (SHELF_OBSERVATION_Y_MIN <= y <= SHELF_OBSERVATION_Y_MAX):
            return
        # 关键：不传当前站点货架，由检测自身的全局 x 决定 A..E。
        slot = fixed_slot_from_world(x, z)
        if slot is None:
            return
        shelf_name, level, column = slot
        acc_key = (shelf_name, level, column, kind)
        acc = self._slot_acc.setdefault(
            acc_key, {"x": [], "y": [], "z": [], "distance": [],
                      "conf": 0.0,
                      "last_stamp_ns": None})
        try:
            stamp_ns = int(detection.get("stamp_ns"))
        except (TypeError, ValueError):
            stamp_ns = None
        # 同一源图像中的重复框不能冒充多帧确认。
        if stamp_ns is not None and acc["last_stamp_ns"] == stamp_ns:
            return
        acc["x"].append(x)
        acc["y"].append(y)
        acc["z"].append(z)
        # front_depth_m 是同一 YOLO ROI 的前景深度，优先用作实际观测距离；
        # 缺失时才退回相机世界坐标到检测世界坐标的欧氏距离，最后退回
        # 底盘到货物的水平距离。这里只衡量“这次看得多近”，不参与槽位映射。
        observation_distance = None
        try:
            front_depth = float(detection.get("front_depth_m"))
            if math.isfinite(front_depth) and front_depth > 0.0:
                observation_distance = front_depth
        except (TypeError, ValueError):
            pass
        if observation_distance is None:
            try:
                camera_world = [
                    float(v) for v in detection["camera_world_position"]]
                if len(camera_world) == 3 and all(
                        math.isfinite(v) for v in camera_world):
                    observation_distance = math.sqrt(sum(
                        (world[index] - camera_world[index]) ** 2
                        for index in range(3)))
            except (KeyError, TypeError, ValueError):
                pass
        if observation_distance is None and self.base_xy is not None:
            observation_distance = math.hypot(
                x - self.base_xy[0], y - self.base_xy[1])
        if observation_distance is not None:
            acc["distance"].append(observation_distance)
        acc["last_stamp_ns"] = stamp_ns
        try:
            confidence = float(detection.get("conf", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        acc["conf"] = max(acc["conf"], confidence)
        if len(acc["x"]) < max(
                self.confirmations, DEPTH_MIN_SAMPLES):
            return
        z_med = float(sorted(acc["z"])[len(acc["z"]) // 2])
        x_med = float(sorted(acc["x"])[len(acc["x"]) // 2])
        y_med = float(sorted(acc["y"])[len(acc["y"]) // 2])
        distance_med = (
            float(sorted(acc["distance"])[len(acc["distance"]) // 2])
            if acc["distance"] else None)
        sample_count = len(acc["x"])
        self._slot_acc[acc_key] = {
            "x": [], "y": [], "z": [], "distance": [], "conf": 0.0,
            "last_stamp_ns": None,
        }
        # 中位数必须仍落在同一固定格：抑制帧间抖动导致的列/层摇摆。
        if (fixed_slot_from_world(x_med, z_med) != slot
                or not SHELF_OBSERVATION_Y_MIN
                <= y_med
                <= SHELF_OBSERVATION_Y_MAX):
            return
        if self.matrix.record_at(
                shelf_name, level, column,
                -1, kind, acc["conf"],
                observation_distance=distance_med,
                observer_xy=self.base_xy,
                sample_count=sample_count):
            primary = self.matrix.cells.get(
                f"{level}|{shelf_name}|{column}", {})
            distance_text = (
                f"{distance_med:.3f}m"
                if distance_med is not None else "?")
            self.get_logger().info(
                f"[memory] observed kind={kind} "
                f"slot=({shelf_name}, {level}, {column}) "
                f"[yolo-only global-map observer_x={self.base_xy[0]:.3f} "
                f"xyz_med=({x_med:.3f},{y_med:.3f},{z_med:.3f}) "
                f"distance={distance_text} conf={acc['conf']:.3f}; "
                f"primary={primary.get('kind')} "
                f"conf={primary.get('confidence')} "
                f"closest={primary.get('closest_distance')}]")
            self._dirty = True

    def consume_slot(
            self, shelf: str, level: str, column: str,
            kind: str | None = None) -> None:
        if self.matrix.consume_slot(shelf, level, column, kind=kind):
            self.get_logger().info(
                f"[memory] consumed slot=({shelf}, {level}, {column})"
                + ("" if not kind else
                   f"; invalidated sibling {kind} candidates on "
                   f"{shelf}-{level}"))
            self._dirty = True
            self._write_now()

    def consume(self, marker_id: int) -> None:
        """按 marker_id 消费（兼容旧调用，YOLO 路径请用 consume_slot）。"""
        if self.matrix.consume(int(marker_id)):
            self.get_logger().info(
                f"[memory] consumed marker={marker_id} "
                f"slot={SLOT_BY_MARKER.get(int(marker_id))}")
            self._dirty = True
            self._write_now()

    def reset(self) -> None:
        self.matrix.clear()
        self._slot_acc.clear()
        self._blocked_log_at.clear()
        self.base_xy = None
        self._dirty = True
        self._write_now()

    def _write_now(self) -> None:
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output_path.with_suffix(
                self.output_path.suffix + ".tmp")
            payload = self.matrix.to_json()
            # GUI 兼容：由已确认格子生成"近似记录"汇总（不再有独立深度回退）。
            approx: dict[str, dict[str, float]] = {}
            approx_cols: dict[str, dict[str, str]] = {}
            for key, cell in payload["cells"].items():
                if cell.get("consumed"):
                    continue
                try:
                    level, shelf, column = key.split("|")
                except (ValueError, KeyError):
                    continue
                level_shelf = f"{level}|{shelf}"
                approx.setdefault(level_shelf, {})[
                    str(cell.get("kind", ""))] = float(
                        cell.get("confidence", 0.0))
                approx_cols.setdefault(level_shelf, {})[
                    str(cell.get("kind", ""))] = column
            payload["approx"] = approx
            payload["approx_cols"] = approx_cols
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=1),
                encoding="utf-8")
            temporary.replace(self.output_path)
        except OSError as exc:
            self.get_logger().warn(
                f"[memory] failed to write matrix: {exc}")

    def tick_write(self) -> None:
        now = time.time()
        if (self._dirty
                and now - self._last_write_at >= self.write_interval_s):
            self._write_now()
            self._last_write_at = now
            self._dirty = False


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="publish the 3x15 memory matrix JSON")
    parser.add_argument(
        "--confirmations", type=int, default=3)
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    rclpy.init()
    node = MemoryMatrixTracker(
        confirmations=args.confirmations, output_path=Path(args.output))
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            node.tick_write()
    except KeyboardInterrupt:
        pass
    finally:
        node.tick_write()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
