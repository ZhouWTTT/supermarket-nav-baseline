#!/usr/bin/env python3
"""快照式全架扫描实验客户端：复用正式客户端的扫描状态机，不抓取、不送货。

流程与正式入口一致：导航到每个货架 → 扫描位姿 → 记录，只是：
  * 扫描位姿换成"整架一帧"的远距离快照位姿（y=2.0，整架可入一帧）；
  * 关闭目标关联/抓取/放置，只让 MemoryMatrixTracker 独立记录 45 槽种类；
  * 全部货架扫描完成后正常退出，写出 logs/memory_matrix.json。
"""

from __future__ import annotations

import argparse
import pathlib
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

import integrated_nav_pick_place as integrated  # noqa: E402
import yolo_aruco_shelf_pick as pick  # noqa: E402
from memory_matrix import MemoryMatrixTracker  # noqa: E402


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "perception" / "checkpoints" / "best.pt"

# 整架快照站位：退到 y≈2.0（距货架前缘约 1.17m）。用抬头/低头/平视三个
# 动作分别覆盖上/下/全层，宽视场会同时看到相邻货架，深度 x 与 marker
# 会把每个商品分到正确的货架，不必一帧只看一个货架。
SNAPSHOT_Y = 2.00
# 站间安全通道：走廊挡板北端在 y=1.70，x≈0.5；直行取 y=2.45 让车身
# （后部约 0.5m）始终在挡板北侧，避免压到挡板角导致基座打滑卡死。
TRANSIT_Y = 2.45
SNAPSHOT_POSES = (
    ("snap_up", 0.60, 0.00, 0.15),       # 抬头（俯仰上限约 +0.16）：上/中层
    ("snap_down", 0.60, 0.00, -0.35),    # 低头：覆盖中/下层
    ("snap_center", 0.60, 0.00, 0.00),   # 平视：全层补拍
)


class SnapshotClient(integrated.IntegratedNavPickPlace):
    """复用父级扫描状态机，仅替换位姿并禁用抓取。"""

    def __init__(self, stations=None, passes: int = 1) -> None:
        super().__init__(
            "kele", max_scan_cycles=1,
            tcp_diagnostic_ground_truth=False, scan_skip_lower=False,
            nav_during_scan=True, close_recheck=False)
        self.stations = [float(x) for x in (stations or pick.SCAN_X)]
        # 覆盖扫描位姿为整架快照位姿
        self.default_scan_poses = SNAPSHOT_POSES
        self.scan_poses = SNAPSHOT_POSES
        self.max_scan_cycles = max(1, int(passes))
        self.finished = False
        self.snapshot_stations_done = 0
        self._last_finished_station = None
        self._coarse_logged_for = None
        self._transit_stage = None   # 当前站的 x，用于两段式导航
        self._transit_phase = "lane"  # lane(通道直行) -> approach(退到快照位)

    def _nearest_scan_stations(self):
        """按指定货架顺序访问（父级按 SCAN_X 全 5 架）。"""
        return list(range(len(self.stations)))

    def _publish_perception_request(self, enabled, force=False):
        """快照实验感知全程常开：避免冷启动漏帧，也记录相邻货架。"""
        return

    def current_scan_station_x(self) -> float:
        if self.scan_station_order is not None:
            idx = self.scan_station_order[self.scan_index]
        else:
            idx = self.scan_index
        return float(self.stations[idx % len(self.stations)])

    def _advance_scan_pose(self) -> bool:
        """用 len(self.stations) 代替父级的 len(SCAN_X)，支持子集货架。"""
        self.scan_pose_index += 1
        self.scan_camera_ready_since = None
        if self.scan_pose_index >= len(self.scan_poses):
            self.scan_pose_index = 0
            self.scan_index += 1
            if self.scan_index >= len(self.stations):
                self.scan_index = 0
                self.scan_cycles += 1
                if self.scan_cycles >= self.max_scan_cycles:
                    self.get_logger().info(
                        f"[snapshot] swept {len(self.stations)} stations "
                        f"x {self.scan_cycles} pass(es); stopping")
                    self.set_state(pick.STATE_ABORT)
                    return False
        if self.state != pick.STATE_ABORT:
            self.set_state(pick.STATE_GO_SCAN)
        return self.state != pick.STATE_ABORT

    def drive_to(
            self, target_xy, final_yaw: float,
            position_tolerance: float = 0.055,
            linear_min_mps: float | None = None) -> bool:
        """站间两段式导航：先走 y=2.45 安全通道，再到 y=2.0 快照位。"""
        station_x = float(target_xy[0])
        if self._transit_stage != station_x:
            self._transit_stage = station_x
            self._transit_phase = "lane"
        if self._transit_phase == "lane":
            arrived = pick.ShelfPickController.drive_to(
                self, [station_x, TRANSIT_Y], final_yaw,
                position_tolerance, linear_min_mps=linear_min_mps)
            if arrived:
                self._transit_phase = "approach"
                self.get_logger().info(
                    f"[snapshot] reached transit lane x={station_x:.2f}")
            return False
        # approach：从通道退到快照位
        arrived = pick.ShelfPickController.drive_to(
            self, [station_x, SNAPSHOT_Y], final_yaw,
            position_tolerance, linear_min_mps=linear_min_mps)
        if arrived:
            self._transit_phase = "lane"
            return True
        if self.base_xy is not None:
            distance = float(
                (self.base_xy[0] - station_x) ** 2
                + (self.base_xy[1] - SNAPSHOT_Y) ** 2) ** 0.5
            yaw_error = pick.wrap_to_pi(final_yaw - self.base_yaw)
            if distance < 0.20 and abs(yaw_error) < 0.20:
                self.set_twist(0.0, 0.0)
                self.cmd_linear = 0.0
                self.cmd_angular = 0.0
                if self._coarse_logged_for != station_x:
                    self._coarse_logged_for = station_x
                    self.get_logger().info(
                        f"[snapshot] coarse arrival at x={station_x:.2f} "
                        f"y={SNAPSHOT_Y:.2f} dist={distance:.3f}")
                return True
        return False

    def try_association_locked(self) -> None:
        """快照实验不做目标关联/定位/抓取，只由 MemoryMatrixTracker 记录。"""
        return

    def _maybe_lock_yolo_only_target_locked(self) -> None:
        """快照阶段不锁定目标、不抓取（记忆矩阵由 tracker 独立记录）。"""
        return

    def _start_revisit(self) -> None:
        """快照实验跳过补拍（补拍失败会经 position-fallback 进入抓取）。"""
        return

    def _try_position_fallback(self) -> bool:
        """禁用位置回退：它会把 YOLO 框直接定位并进入 ALIGN 抓取。"""
        return False

    def set_state(self, new_state: str) -> None:
        if new_state == pick.STATE_ABORT:
            # 扫描按计划跑完（父级在全部站点扫完后会走到 ABORT）。
            self.finished = True
            self.get_logger().info(
                "[snapshot] sweep complete; stopping client")
            return
        super().set_state(new_state)

    def tick(self) -> None:
        super().tick()
        if self.finished:
            self.set_twist(0.0, 0.0)
            self.smooth_commands()
            self.publish_commands()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="full-shelf snapshot experiment client")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--passes", type=int, default=1,
        help="how many full sweeps over all five shelves")
    parser.add_argument(
        "--dwell", type=float, default=1.5,
        help="seconds per snapshot pose (absorbs perception warm-up)")
    parser.add_argument(
        "--stations", type=float, nargs="*", default=None,
        help="station x values to sweep (default: all five shelves)")
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("--passes must be >= 1")
    if args.dwell <= 0.0:
        parser.error("--dwell must be positive")
    if not pathlib.Path(args.weights).is_file():
        parser.error(f"weights not found: {args.weights}")
    return args


def main() -> None:
    from run_log import start_run_log
    start_run_log("snapshot_client")
    args = parse_args()
    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=4)
    nodes = []
    spin_thread = None
    try:
        client = SnapshotClient(
            stations=args.stations, passes=args.passes)
        # 快照实验用更长的每姿态驻留，让感知冷启动不浪费前几个位姿
        pick.SCAN_DWELL_S = float(args.dwell)
        yolo_node = pick.KeleDetectNode(
            backend="yolo", pub_res_img=False, device=args.device,
            weights=str(pathlib.Path(args.weights).resolve()),
            target_kind=None, confidence=args.confidence, show=False,
            camera_names=("head",), max_inference_hz=12.0)
        aruco_node = pick.ArucoDetectNode(
            "head", marker_size=pick.MARKER_SIZE_M, publish_tf=False,
            publish_result_image=False)
        tracker = MemoryMatrixTracker(confirmations=2)
        client.configure_local_perception(yolo_node, aruco_node)
        # 感知全程常开（父级 duty cycle 已被快照客户端禁用）
        yolo_node.set_enabled(True)
        aruco_node.set_enabled(True)
        # 每轮实验从干净矩阵开始
        tracker.reset()
        nodes = [client, yolo_node, aruco_node, tracker]
        for node in nodes:
            executor.add_node(node)
        # 客户端 tick 定时器（50Hz）与 odom/joints/laser 订阅必须由同一个
        # 执行器公平派发；逐个 spin_once 会让定时器霸占回调，激光被饿死。
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        client.get_logger().info(
            f"[snapshot] client ready; sweeping "
            f"{len(pick.SCAN_X)} shelves x {args.passes} pass(es)")
        while rclpy.ok() and not client.finished:
            time.sleep(0.005)
        tracker.tick_write()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for node in nodes:
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        executor.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
