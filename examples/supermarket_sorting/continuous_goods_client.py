#!/usr/bin/env python3
"""Continuous multi-order supermarket sorting client.

整合 ``integrated_nav_pick_place.py`` 的单货物抓取流程，连续处理一批随机生成、
不重复的货物订单（默认 5 单）：

    抓货区抓取 -> 导航到终点 -> 抬升释放（把手臂抬高到上层货架抓取高度，
    伸长手臂后松爪） -> 收臂倒车离开 -> 导航返回抓货区 -> 开始下一个订单

防死锁设计：
  * 每次卸货后强制导航回到抓货区才进入下一单，绝不在终点停住；
  * 导航阶段有进度看门狗：长时间无位移时强制重新设置导航目标并重规划；
    持续卡死则放弃当前订单，继续后面的订单；
  * 抓取失败（找不到/抓取掉落）允许按订单重试，重试耗尽后跳过，整个客户端
    不中断；
  * 控制层运行期间屏蔽 ``rclpy.shutdown``（由顶层 main 统一收尾），防止单次
    流程/中止把整个客户端杀掉；
  * 卸货后先倒车一小段再返回，避免刚扔下的货物挡住激光并触发安全急停。

用法（与 integrated_nav_pick_place.py 相同的容器环境）::

    python3 examples/supermarket_sorting/continuous_goods_client.py \
        --orders-count 5 --seed 11 --max-scan-cycles 2

也可用 ``--orders kele,maidong,...`` 指定订单列表（不能重复）。
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import threading
import time
from contextlib import contextmanager

import numpy as np

# integrated_nav_pick_place 在导入时完成两件事：把本目录放进 sys.path
# （supermarket_navigation.py 已复制到本目录，从这里直接 import），
# 还会打上 MuJoCo 兼容补丁。必须先于 supermarket_navigation 导入。
import integrated_nav_pick_place as integrated  # noqa: E402
import yolo_aruco_shelf_pick as pick  # noqa: E402
from integrated_nav_pick_place import (  # noqa: E402
    IntegratedNavPickPlace,
    PLACE_RETREAT_ARM_L,
    PLACE_RETREAT_ARM_R,
)
from supermarket_navigation import DELIVERY_APPROACH  # noqa: E402


# ---------------------------------------------------------------------------
# 订单生成
# ---------------------------------------------------------------------------
ALL_GOODS_KINDS = sorted(pick.PRODUCT_CENTER_ABOVE_MARKER_M.keys())
DEFAULT_ORDERS_COUNT = 5


def parse_orders_arg(text: str) -> list[str]:
    """解析显式订单列表（逗号分隔，允许重复，必须在已知货物类别内）。"""
    kinds = [part.strip().lower() for part in text.split(",") if part.strip()]
    for kind in kinds:
        if kind not in ALL_GOODS_KINDS:
            raise ValueError(
                f"unknown goods kind {kind!r}; valid kinds: "
                f"{', '.join(ALL_GOODS_KINDS)}")
    return kinds


def generate_orders(count: int, seed: int | None) -> list[str]:
    """随机生成 count 个货物订单（random.choices 允许重复货物）。"""
    if count < 1:
        raise ValueError("orders count must be >= 1")
    rng = random.Random(seed)
    return rng.choices(ALL_GOODS_KINDS, k=count)


# ---------------------------------------------------------------------------
# 连续订单流程常量
# ---------------------------------------------------------------------------
# 终点卸货点：沿用 baseline 导航的配送台到达点（桌面北侧的空地），
# 到点后直接把货物扔下，不做桌面放置。
DROP_GOAL = DELIVERY_APPROACH  # (-1.80, -2.60, -pi/2)
# 返回抓货区：回到黄线走廊东端起点（货架扫描站附近），面北等待下一单。
SHELF_RETURN_GOAL = (1.92, pick.SCAN_Y, pick.YAW_NORTH)
# 后续订单（从 A 货架开始扫）返回西端起点，避免先回 E 再折返 A。
SHELF_RETURN_GOAL_WEST = (pick.SCAN_X[-1], pick.SCAN_Y, pick.YAW_NORTH)

DROP_RELEASE_DWELL_S = 1.5     # 张爪后停留时间，让货物完全脱手
DROP_RETREAT_DWELL_S = 1.0     # 手臂收回后的停留时间
DROP_BACKUP_DIST_M = 0.18      # 卸货后倒车距离，离开刚扔下的货物
DROP_BACKUP_SPEED_MPS = 0.10
DROP_BACKUP_TIMEOUT_S = 8.0
# 终点抬升释放：把手臂抬高到上层货架抓取高度（slide 收到顶、TCP z≈1.2m），
# 再伸长手臂、松爪释放（不做桌面放置）。
DROP_RAISE_TCP_Z = 1.20         # 上层货架抓取高度（TCP z，米）
DROP_RAISE_EXTEND_M = 0.55      # 伸长距离（朝南越过配送区）
DROP_RAISE_SLIDE = -0.04        # 抬升 slide（顶层抓取姿态，= SLIDE_MIN）
DROP_RAISE_TIMEOUT_S = 12.0     # 抬升到位超时兜底
# 松爪前最后延展：把手臂从抬升姿态再往外伸一段，让货物越过桌面更深处
DROP_RAISE_EXTEND_FINAL_M = 0.75
DROP_RAISE_EXTEND_TIMEOUT_S = 10.0
# 放货前朝桌子方向前进一小段（不怕撞桌；带超时防死锁）
DROP_CREEP_DIST_M = 0.50
DROP_CREEP_SPEED_MPS = 0.06
DROP_CREEP_TIMEOUT_S = 8.0

ABORT_SETTLE_S = 1.0           # 抓取中止后等待手臂收回稳定
ABORT_SETTLE_TIMEOUT_S = 15.0  # 收臂超时兜底：即使没完全到位也放行下一单

# 导航防死锁看门狗
NAV_STALL_CHECK_S = 8.0        # 每隔这么久检查一次位移
NAV_STALL_MIN_PROGRESS_M = 0.05
NAV_STALL_MAX_RESETS = 6       # 连续多次无进展 -> 放弃当前订单

# 卸货收臂超时兜底：货物已经扔下，手臂没完全收回来也不阻塞流程
DROP_RETREAT_TIMEOUT_S = 15.0


@contextmanager
def _suppress_rclpy_shutdown():
    """临时屏蔽 rclpy.shutdown，防止抓取状态机中止时杀掉整个连续流程。"""
    import rclpy
    real_shutdown = rclpy.shutdown
    rclpy.shutdown = lambda *args, **kwargs: None
    try:
        yield
    finally:
        rclpy.shutdown = real_shutdown


class ContinuousOrderController(IntegratedNavPickPlace):
    """单订单的连续流程控制器：抓取 -> 终点抬升释放 -> 返回抓货区。

    扩展自 IntegratedNavPickPlace：抓取阶段完全复用父级状态机；抓取完成后
    不再做精细桌面放置，而是在 DELIVERY_APPROACH 直接把货物扔下，然后导航
    回抓货区。整个流程不自动 rclpy.shutdown，交由顶层编排器管理。
    """

    def __init__(self, target_kind: str, max_scan_cycles: int,
                 tcp_diagnostic_ground_truth: bool, scan_skip_lower: bool,
                 place_release_dwell_s: float = DROP_RELEASE_DWELL_S,
                 place_retreat_dwell_s: float = DROP_RETREAT_DWELL_S,
                 nav_during_scan: bool = True,
                 backup_after_grab_m: float = 0.20,
                 close_recheck: bool = True):
        super().__init__(
            target_kind, max_scan_cycles,
            tcp_diagnostic_ground_truth, scan_skip_lower,
            place_release_dwell_s=place_release_dwell_s,
            place_retreat_dwell_s=place_retreat_dwell_s,
            nav_during_scan=nav_during_scan,
            backup_after_grab_m=backup_after_grab_m,
            close_recheck=close_recheck)

        # 顶层编排器轮询的完成/失败标记
        self.order_index = 0
        self.order_count = 1
        self.order_done = False
        self.order_done_at = None
        self.order_aborted = False
        self.abort_reason = None
        self._abort_handled = False
        self._abort_reason = None
        self._abort_settle_t0 = 0.0

        # 卸货倒车状态
        self._drop_backup_start_xy = None
        self._drop_backup_start_yaw = 0.0
        self._drop_backup_t0 = 0.0
        # 终点抬升释放状态
        self.drop_raise_joints = None
        self.drop_raise_slide = None
        self._drop_raise_t0 = 0.0
        self.drop_extend_joints = None
        self._drop_extend_t0 = 0.0
        self._drop_creep_start_y = None
        self._drop_creep_t0 = 0.0

        # 导航防死锁看门狗状态
        self._watchdog_phase = None
        self._watchdog_t0 = self.now()
        self._watchdog_last_xy = None
        self._watchdog_resets = 0
        self._watchdog_goal = None
        self._last_unknown_phase_log = 0.0

        self.get_logger().info(
            f"[continuous] order controller ready kind={target_kind} "
            f"drop_goal={DROP_GOAL} return_goal={SHELF_RETURN_GOAL_WEST}")

    # ------------------------------------------------------------------
    # 抓取阶段（复用父级状态机，但屏蔽 abort 时的全局 shutdown）
    # ------------------------------------------------------------------
    def _tick_grab(self) -> None:
        with _suppress_rclpy_shutdown():
            super().tick()   # IntegratedNavPickPlace.tick -> ShelfPickController.tick
        if self.state == pick.STATE_ABORT:
            self._handle_abort()

    def _handle_abort(self) -> None:
        if self._abort_handled:
            return
        self._abort_handled = True
        self._abort_reason = f"grab aborted (state={self.state})"
        self.flow_phase = "aborted"
        self._abort_settle_t0 = self.now()
        # 收回手臂并张开夹爪，让机器人以安全姿态等待编排器决定重试/跳过。
        if self.use_dual_tissue_grasp:
            self.des_left_arm = np.asarray(PLACE_RETREAT_ARM_L, dtype=float)
            self.des_right_arm = np.asarray(PLACE_RETREAT_ARM_R, dtype=float)
            self.des_left_grip = pick.GRIP_OPEN
            self.des_right_grip = pick.GRIP_OPEN
        else:
            joints = (
                PLACE_RETREAT_ARM_R if self.grasp_arm == "r"
                else PLACE_RETREAT_ARM_L)
            self.set_selected_arm_target(np.asarray(joints, dtype=float))
            self._set_selected_grip(pick.GRIP_OPEN)
        self.des_slide = pick.SLIDE_REFERENCE_COMMAND
        self.get_logger().error(
            f"[order] grab aborted; retracting arm and settling: "
            f"{self._abort_reason}")

    def _abort_settle_tick(self) -> None:
        self.set_twist(0.0, 0.0)
        if self.use_dual_tissue_grasp:
            self.des_left_grip = pick.GRIP_OPEN
            self.des_right_grip = pick.GRIP_OPEN
            settled = self.dual_commands_ready(
                arm_tolerance=0.10, slide_tolerance=0.06)
        else:
            self._set_selected_grip(pick.GRIP_OPEN)
            settled = self.commands_ready(
                arm_tolerance=0.10, slide_tolerance=0.06)
        elapsed = self.now() - self._abort_settle_t0
        if (elapsed >= ABORT_SETTLE_S and settled) or \
                elapsed >= ABORT_SETTLE_TIMEOUT_S:
            self.order_aborted = True
            self.abort_reason = self._abort_reason
            self.flow_phase = "order_failed"
            if settled:
                self.get_logger().warn(
                    "[order] abort motion settled; handing control back to "
                    "the orchestrator")
            else:
                self.get_logger().warn(
                    f"[order] abort arm did not fully settle within "
                    f"{elapsed:.1f}s; continuing anyway (anti-deadlock)")

    # ------------------------------------------------------------------
    # 终点抬升释放（朝桌子前进一小段 -> 抬高到上层货架抓取高度 ->
    # 伸长手臂 -> 松爪）
    # ------------------------------------------------------------------
    def _drop_tick(self) -> None:
        now = self.now()
        if self.use_dual_tissue_grasp:
            self._drop_tick_dual(now)
            return

        if self.place_stage == 0:
            # 先把手臂抬高到上层货架抓取高度并伸长（前进时货物已在高位，不会碰桌）
            if self.drop_raise_joints is None:
                self.drop_raise_joints = self._solve_drop_raise()
                if self.drop_raise_joints is not None:
                    self.set_selected_arm_target(self.drop_raise_joints)
                    if self.drop_raise_slide is not None:
                        self.des_slide = self.drop_raise_slide
                    self._place_arm_target_sent = True
                    self._drop_raise_t0 = now
                    self.get_logger().info(
                        f"[drop-raise] raising arm to top-shelf height "
                        f"z={DROP_RAISE_TCP_Z:.2f}m slide="
                        f"{self.drop_raise_slide:.2f}")
                else:
                    self.get_logger().warn(
                        "[drop-raise] IK failed; falling back to "
                        "extend/release")
                    # 不跳过向桌子的最后一段前进：右臂 raise 失败时也要先
                    # 走完 stage 1 的 0.5m creep，否则会在离桌一小段处直接
                    # 释放（左臂 raise 成功会走完这段，右臂却放不到位）。
                    self.place_stage = 1
                    self.place_t0 = now
                    return
            if (self.commands_ready(
                    arm_tolerance=0.06, slide_tolerance=0.03)
                    or now - self._drop_raise_t0 >= DROP_RAISE_TIMEOUT_S):
                self.place_stage = 1
                self.place_t0 = now
                self.get_logger().info(
                    "[drop-raise] arm raised; creeping towards the table")
            return
        if self.place_stage == 1:
            # 手臂已抬高，再朝桌子方向前进最后一段（带超时防死锁）
            if self._drop_creep_start_y is None:
                self._drop_creep_start_y = float(self.base_xy[1])
                self._drop_creep_t0 = now
            crept = float(self._drop_creep_start_y - self.base_xy[1])
            if (crept >= DROP_CREEP_DIST_M
                    or now - self._drop_creep_t0 >= DROP_CREEP_TIMEOUT_S):
                self.set_twist(0.0, 0.0)
                self.place_stage = 2
                self.place_t0 = now
                self.get_logger().info(
                    f"[drop-creep] advanced {crept:.3f}m towards the table")
            else:
                yaw_err = pick.wrap_to_pi(-math.pi / 2.0 - self.base_yaw)
                angular = float(np.clip(2.0 * yaw_err, -0.3, 0.3))
                linear = (
                    DROP_CREEP_SPEED_MPS if abs(yaw_err) <= 0.10 else 0.0)
                self.set_twist(linear, angular)
            return
        if self.place_stage == 2:
            # 松爪前最后延展：把手臂再往外伸一段（带超时兜底）
            if self.drop_extend_joints is None:
                self.drop_extend_joints = self._solve_drop_raise(
                    DROP_RAISE_EXTEND_FINAL_M)
                if self.drop_extend_joints is not None:
                    self.set_selected_arm_target(self.drop_extend_joints)
                    if self.drop_raise_slide is not None:
                        self.des_slide = self.drop_raise_slide
                    self._place_arm_target_sent = True
                    self._drop_extend_t0 = now
                    self.get_logger().info(
                        f"[drop-extend] extending arm to "
                        f"{DROP_RAISE_EXTEND_FINAL_M:.2f}m")
                else:
                    self.get_logger().warn(
                        "[drop-extend] IK failed; releasing without "
                        "further extension")
                    self.place_stage = 3
                    self.place_t0 = now
                    return
            if (self.commands_ready(
                    arm_tolerance=0.06, slide_tolerance=0.03)
                    or now - self._drop_extend_t0
                    >= DROP_RAISE_EXTEND_TIMEOUT_S):
                self.place_stage = 3
                self.place_t0 = now
                self.get_logger().info(
                    "[drop-extend] arm extended; opening gripper")
            return
        if self.place_stage == 3:
            self._set_selected_grip(pick.GRIP_OPEN)
            if now - self.place_t0 >= self.place_release_dwell_s:
                self.get_logger().info(
                    "[drop] gripper opened; goods released from height")
                self.place_stage = 4
                self.place_t0 = now
                self._place_retreat_sent = False
        elif self.place_stage == 4:
            self._set_selected_grip(pick.GRIP_OPEN)
            if not self._place_retreat_sent:
                self._place_retreat_sent = True
                joints = (
                    PLACE_RETREAT_ARM_R if self.grasp_arm == "r"
                    else PLACE_RETREAT_ARM_L)
                self.set_selected_arm_target(np.asarray(joints, dtype=float))
                self.des_slide = pick.SLIDE_REFERENCE_COMMAND
            if (now - self.place_t0 >= self.place_retreat_dwell_s
                    and self.commands_ready(
                        arm_tolerance=0.08, slide_tolerance=0.05)) or \
                    now - self.place_t0 >= DROP_RETREAT_TIMEOUT_S:
                self.flow_phase = "drop_backup"
                self._drop_backup_start_xy = None
                if now - self.place_t0 < DROP_RETREAT_TIMEOUT_S:
                    self.get_logger().info(
                        "[drop] arm retracted; backing away from the goods")
                else:
                    self.get_logger().warn(
                        "[drop] arm retreat timed out; backing away anyway "
                        "(anti-deadlock)")

    def _solve_drop_raise(self, extend_m=None):
        """解一个“上层货架抓取高度 + 伸长”的末端姿态（终点朝南伸长）。"""
        if extend_m is None:
            extend_m = DROP_RAISE_EXTEND_M
        bx, by = float(self.base_xy[0]), float(self.base_xy[1])
        world = np.array(
            [bx, by - extend_m, DROP_RAISE_TCP_Z], dtype=float)
        reference = self.selected_arm_positions()
        for slide in (DROP_RAISE_SLIDE, 0.0, 0.05, 0.10):
            joints = self._solve_place_world(
                world, reference, float(slide))
            if joints is not None:
                self.drop_raise_slide = float(slide)
                return joints
        return None

    def _drop_tick_dual(self, now: float) -> None:
        """双机械臂（zhijin）：先抬升 slide，再朝桌子前进并张开夹爪释放。"""
        if self.place_stage == 0:
            self.des_slide = DROP_RAISE_SLIDE
            slide = self.joints.get("slide_joint")
            if (slide is not None
                    and abs(float(slide) - DROP_RAISE_SLIDE) < 0.03):
                self.place_stage = 1
                self.place_t0 = now
                self.get_logger().info("[drop-dual] raised; creeping forward")
            return
        if self.place_stage == 1:
            # 手臂已抬高，朝桌子方向前进最后一段（带超时防死锁）
            if self._drop_creep_start_y is None:
                self._drop_creep_start_y = float(self.base_xy[1])
                self._drop_creep_t0 = now
            crept = float(self._drop_creep_start_y - self.base_xy[1])
            if (crept >= DROP_CREEP_DIST_M
                    or now - self._drop_creep_t0 >= DROP_CREEP_TIMEOUT_S):
                self.set_twist(0.0, 0.0)
                self.place_stage = 2
                self.place_t0 = now
                self.get_logger().info(
                    f"[drop-dual-creep] advanced {crept:.3f}m towards the "
                    "table")
            else:
                yaw_err = pick.wrap_to_pi(-math.pi / 2.0 - self.base_yaw)
                angular = float(np.clip(2.0 * yaw_err, -0.3, 0.3))
                linear = (
                    DROP_CREEP_SPEED_MPS if abs(yaw_err) <= 0.10 else 0.0)
                self.set_twist(linear, angular)
            return
        if self.place_stage == 2:
            self.des_left_grip = pick.GRIP_OPEN
            self.des_right_grip = pick.GRIP_OPEN
            if now - self.place_t0 >= self.place_release_dwell_s:
                self.get_logger().info(
                    "[drop-dual] both grippers opened; goods dropped")
                self.place_stage = 3
                self.place_t0 = now
        elif self.place_stage == 3:
            self.des_left_arm = np.asarray(PLACE_RETREAT_ARM_L, dtype=float)
            self.des_right_arm = np.asarray(PLACE_RETREAT_ARM_R, dtype=float)
            self.des_left_grip = pick.GRIP_OPEN
            self.des_right_grip = pick.GRIP_OPEN
            self.des_slide = pick.SLIDE_REFERENCE_COMMAND
            if (now - self.place_t0 >= self.place_retreat_dwell_s
                    and self.dual_commands_ready(
                        arm_tolerance=0.08, slide_tolerance=0.05)) or \
                    now - self.place_t0 >= DROP_RETREAT_TIMEOUT_S:
                self.flow_phase = "drop_backup"
                self._drop_backup_start_xy = None
                if now - self.place_t0 < DROP_RETREAT_TIMEOUT_S:
                    self.get_logger().info(
                        "[drop-dual] arms retracted; backing away")
                else:
                    self.get_logger().warn(
                        "[drop-dual] arm retreat timed out; backing away "
                        "anyway (anti-deadlock)")

    def _drop_backup_tick(self) -> None:
        """卸货后沿当前朝向倒车一小段，避免刚扔下的货物挡住激光。"""
        now = self.now()
        if self._drop_backup_start_xy is None:
            self._drop_backup_start_xy = self.base_xy.copy()
            self._drop_backup_start_yaw = float(self.base_yaw)
            self._drop_backup_t0 = now

        heading = np.array([
            math.cos(self.base_yaw), math.sin(self.base_yaw)])
        moved_back = float(np.dot(
            self._drop_backup_start_xy - self.base_xy, heading))
        yaw_err = pick.wrap_to_pi(self._drop_backup_start_yaw - self.base_yaw)
        elapsed = now - self._drop_backup_t0

        if (moved_back >= DROP_BACKUP_DIST_M
                or elapsed > DROP_BACKUP_TIMEOUT_S):
            self.set_twist(0.0, 0.0)
            self._start_return_to_shelf()
            return

        linear = float(np.clip(DROP_BACKUP_SPEED_MPS, 0.0, 0.15))
        angular = float(np.clip(2.0 * yaw_err, -0.6, 0.6))
        self.set_twist(-linear, angular)   # 负数 = 倒车

    # ------------------------------------------------------------------
    # 返回抓货区
    # ------------------------------------------------------------------
    def _return_goal(self):
        """卸货后一律回西端（A 货架旁）：下一单从 A 开始扫，避免回 E 再折返。"""
        return SHELF_RETURN_GOAL_WEST

    def _start_return_to_shelf(self) -> None:
        self.flow_phase = "return_to_shelf"
        self._nav_goal = None
        self._nav_last_log = 0.0
        self._watchdog_phase = None   # 强制看门狗重新计时
        goal = self._return_goal()
        self.nav.set_goal(*goal)
        self.get_logger().info(
            f"[flow] drop done; navigating back to the grab area "
            f"{goal}")

    def _return_to_shelf_tick(self) -> None:
        now = self.now()
        self._nav_watchdog_check(self._return_goal())
        if self.order_aborted:
            return
        if self._laser_stale(now):
            self.set_twist(0.0, 0.0)
            if now - self._laser_warn_log > 1.0:
                self.get_logger().warn(
                    "waiting for fresh laser scan on the way back to the shelf")
                self._laser_warn_log = now
            return

        v, w, reached = self.nav.update(
            self.base_xy[0], self.base_xy[1], self.base_yaw,
            laser_msg=self.laser_msg, time_now=now)
        self.set_twist(v, w)

        ctrl = self.nav.controller
        if (ctrl.stop_reason is not None
                and ctrl.stop_reason != self._last_nav_reason):
            self._last_nav_reason = ctrl.stop_reason
            self.get_logger().info(
                f"[nav→shelf] stop_reason={ctrl.stop_reason} "
                f"lidar={ctrl.lidar_clearance:.2f}m "
                f"rear={ctrl.rear_clearance:.2f}m")

        if now - self._nav_last_log >= integrated.NAV_PROGRESS_LOG_S:
            self._nav_last_log = now
            self.get_logger().info(
                f"[nav→shelf] pos=({self.base_xy[0]:.2f},"
                f"{self.base_xy[1]:.2f}) "
                f"yaw={math.degrees(self.base_yaw):.0f}° "
                f"v={v:.2f} w={w:.2f} reached={reached}")

        if reached:
            # 导航器 yaw 容差 0.15 rad，再细化到面北。
            yaw_err = pick.wrap_to_pi(pick.YAW_NORTH - self.base_yaw)
            if abs(yaw_err) > 0.03:
                self.set_twist(0.0, 2.0 * yaw_err)
                return
            self.set_twist(0.0, 0.0)
            self.flow_phase = "order_done"
            self.order_done = True
            self.order_done_at = now
            self.get_logger().info(
                f"[flow] ORDER {self.order_index + 1}/{self.order_count} "
                f"({self.target_kind}) COMPLETE — back at the grab area "
                f"({self.base_xy[0]:.2f},{self.base_xy[1]:.2f})")

    # ------------------------------------------------------------------
    # 导航防死锁看门狗
    # ------------------------------------------------------------------
    def _nav_watchdog_check(self, goal) -> None:
        now = self.now()
        if self._watchdog_phase != self.flow_phase:
            self._watchdog_phase = self.flow_phase
            self._watchdog_t0 = now
            self._watchdog_last_xy = (
                None if self.base_xy is None else self.base_xy.copy())
            self._watchdog_resets = 0
            self._watchdog_goal = goal
            return
        if self.base_xy is None or self._watchdog_last_xy is None:
            return

        goal_xy = np.asarray(goal[:2], dtype=float)
        dist_to_goal = float(np.linalg.norm(goal_xy - self.base_xy))
        if dist_to_goal < 0.30:
            # 已接近终点，机器人可能正在原地对准航向，不算停滞。
            self._watchdog_t0 = now
            self._watchdog_last_xy = self.base_xy.copy()
            return

        elapsed = now - self._watchdog_t0
        moved = float(np.linalg.norm(
            self.base_xy - self._watchdog_last_xy))
        if elapsed >= NAV_STALL_CHECK_S and moved < NAV_STALL_MIN_PROGRESS_M:
            self._watchdog_resets += 1
            self.get_logger().warn(
                f"[anti-deadlock] flow={self.flow_phase} no progress for "
                f"{elapsed:.1f}s (moved={moved:.3f}m); resetting nav goal "
                f"{goal} (reset={self._watchdog_resets}/"
                f"{NAV_STALL_MAX_RESETS})")
            self.nav.set_goal(*goal)
            self._watchdog_t0 = now
            self._watchdog_last_xy = self.base_xy.copy()
            if self._watchdog_resets >= NAV_STALL_MAX_RESETS:
                self._order_fail(
                    f"navigation stalled in {self.flow_phase} towards "
                    f"{goal} ({self._watchdog_resets} resets)")
        else:
            self._watchdog_t0 = now
            self._watchdog_last_xy = self.base_xy.copy()

    def _order_fail(self, reason: str) -> None:
        if not self.order_aborted:
            self.order_aborted = True
            self.abort_reason = reason
            self.flow_phase = "order_failed"
            self.set_twist(0.0, 0.0)
            self.get_logger().error(
                f"[order] FAILED {self.target_kind}: {reason}")

    # ------------------------------------------------------------------
    # 主控制循环（重写：支持连续流程且不自动退出）
    # ------------------------------------------------------------------
    def tick(self) -> None:
        if self.base_xy is None or not self.joints:
            return
        if not self.initialized:
            self.initialize_commands()

        if self.flow_phase == "grab":
            self._tick_grab()
            return

        self.set_twist(0.0, 0.0)
        if self.flow_phase == "order_failed":
            # 已原地停车；只需保持输出，等待编排器接管。
            pass
        elif self.flow_phase == "aborted":
            self._abort_settle_tick()
        elif self.flow_phase == "backup":
            self._backup_tick()
        elif self.flow_phase == "restore_height":
            self._restore_height_tick()
        elif self.flow_phase == "nav_to_delivery":
            self._nav_watchdog_check(DROP_GOAL)
            if not self.order_aborted:
                super()._nav_to_delivery_tick()
        elif self.flow_phase == "place":
            self._drop_tick()
        elif self.flow_phase == "drop_backup":
            self._drop_backup_tick()
        elif self.flow_phase == "return_to_shelf":
            self._return_to_shelf_tick()
        elif self.flow_phase == "order_done":
            self.set_twist(0.0, 0.0)
        else:
            now = self.now()
            if now - self._last_unknown_phase_log > 5.0:
                self._last_unknown_phase_log = now
                self.get_logger().warn(
                    f"[flow] unknown flow_phase={self.flow_phase}; "
                    "stopping motors")

        self.apply_manip_base_hold()
        self.smooth_commands()
        self.publish_commands()

        if self.now() - self.last_status_log > 1.0:
            self.get_logger().info(
                f"[flow] order={self.order_index + 1}/{self.order_count} "
                f"kind={self.target_kind} phase={self.flow_phase} "
                f"state={self.state} place_stage={self.place_stage} "
                f"base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
                f"yaw={math.degrees(self.base_yaw):.0f}°")
            self.last_status_log = self.now()


def _cv_gui_available() -> bool:
    """True if this OpenCV build supports HighGUI windows (GTK etc.)."""
    try:
        import cv2
        cv2.namedWindow("__cv_gui_probe__")
        cv2.destroyWindow("__cv_gui_probe__")
        return True
    except cv2.error:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="continuous multi-order supermarket sorting client "
                    "(grab -> drop -> return -> next)")
    parser.add_argument(
        "--orders-count", type=int, default=DEFAULT_ORDERS_COUNT,
        help=f"number of random orders (default: {DEFAULT_ORDERS_COUNT}, "
             "may contain duplicate goods)")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for order generation (default: random)")
    parser.add_argument(
        "--orders", default=None,
        help="explicit comma-separated order list, e.g. kele,maidong,... "
             "(duplicates allowed; overrides --orders-count/--seed)")
    parser.add_argument(
        "--weights", default=str(integrated.REPO_ROOT / "best.pt"),
        help="multi-class Ultralytics checkpoint (default: repository best.pt)")
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--show", action="store_true",
                        help="show the YOLO result window")
    parser.add_argument("--max-scan-cycles", type=int, default=3)
    parser.add_argument("--tcp-diagnostic-ground-truth", action="store_true")
    parser.add_argument("--scan-skip-lower", action="store_true")
    parser.add_argument("--no-nav-during-scan", action="store_true")
    parser.add_argument("--no-close-recheck", action="store_true")
    parser.add_argument("--backup-after-grab", type=float, default=0.20)
    parser.add_argument("--drop-release-dwell", type=float,
                        default=DROP_RELEASE_DWELL_S)
    parser.add_argument("--drop-retreat-dwell", type=float,
                        default=DROP_RETREAT_DWELL_S)
    parser.add_argument("--max-retries", type=int, default=2,
                        help="retries per order before skipping it")
    args = parser.parse_args()
    if args.orders_count < 1:
        parser.error("--orders-count must be >= 1")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if args.max_scan_cycles < 1:
        parser.error("--max-scan-cycles must be >= 1")
    if args.backup_after_grab < 0.0:
        parser.error("--backup-after-grab must be >= 0")
    if args.drop_release_dwell < 0.0 or args.drop_retreat_dwell < 0.0:
        parser.error("drop dwells must be >= 0")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if args.orders:
        try:
            parse_orders_arg(args.orders)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main() -> None:
    from run_log import start_run_log
    start_run_log("gui_client_continuous")
    args = parse_args()
    weights = str(pathlib.Path(args.weights).expanduser().resolve())
    if not pathlib.Path(weights).is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    orders = (
        parse_orders_arg(args.orders)
        if args.orders
        else generate_orders(args.orders_count, args.seed))

    import rclpy
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=4)
    nodes = []
    spin_thread = None
    controller = None
    viewer = None
    try:
        # 一个 YOLO 节点发布全部货物类别，各订单控制器按自身 target_kind 过滤，
        # 避免每单重载模型。
        yolo_node = pick.KeleDetectNode(
            backend="yolo", pub_res_img=True, device=args.device,
            weights=weights, target_kind=None,
            confidence=args.confidence, show=False,
            camera_names=("head",))
        aruco_node = pick.ArucoDetectNode(
            "head", marker_size=pick.MARKER_SIZE_M, publish_tf=False,
            publish_result_image=True)
        nodes += [yolo_node, aruco_node]
        executor.add_node(yolo_node)
        executor.add_node(aruco_node)

        def make_controller(
                kind: str, index: int,
                exclude_marker_ids=None):
            ctrl = ContinuousOrderController(
                kind, args.max_scan_cycles,
                args.tcp_diagnostic_ground_truth, args.scan_skip_lower,
                place_release_dwell_s=args.drop_release_dwell,
                place_retreat_dwell_s=args.drop_retreat_dwell,
                nav_during_scan=not args.no_nav_during_scan,
                backup_after_grab_m=args.backup_after_grab,
                close_recheck=not args.no_close_recheck)
            ctrl.order_index = index
            ctrl.order_count = len(orders)
            # 只有第一单从最近的 E 货架开始扫；之后每单从最西侧 A 货架开始
            ctrl.scan_prefer_west_start = index > 0
            # 重试时排除上次抓取失败的槽位 marker，让扫描去找同类商品的
            # 另一个位置，避免每次重试都撞同一个卡住的槽位（死循环感）。
            if exclude_marker_ids:
                ctrl.excluded_marker_ids = set(exclude_marker_ids)
            return ctrl

        def start_order(
                kind: str, index: int,
                exclude_marker_ids=None):
            nonlocal controller, viewer
            ctrl = make_controller(
                kind, index,
                exclude_marker_ids=exclude_marker_ids)
            executor.add_node(ctrl)
            nodes.append(ctrl)
            controller = ctrl
            if viewer is not None:
                viewer.controller = ctrl
            ctrl.get_logger().info(
                f"[orders] START order {index + 1}/{len(orders)}: {kind}")

        def finish_controller(ctrl):
            executor.remove_node(ctrl)
            # 给执行器一点时间停止派发该节点的回调，再安全销毁。
            time.sleep(0.05)
            try:
                nodes.remove(ctrl)
            except ValueError:
                pass
            ctrl.destroy_node()

        start_order(orders[0], 0)

        if args.show:
            if _cv_gui_available():
                viewer = pick.MainThreadResultViewer(controller)
                executor.add_node(viewer)
                nodes.append(viewer)
            else:
                # 官方 client 镜像的 OpenCV 无窗口支持，仿真窗口仍在 server 侧。
                controller.get_logger().warn(
                    "OpenCV has no GUI support; skipping the YOLO window "
                    "(the server-side simulation window still shows motion)")

        controller.get_logger().info(
            f"[orders] 本批订单({len(orders)}单): " + " -> ".join(orders) +
            f"  seed={args.seed}")

        def spin_in_background():
            try:
                executor.spin()
            except ExternalShutdownException:
                pass

        spin_thread = threading.Thread(
            target=spin_in_background, name="ros2_executor", daemon=True)
        spin_thread.start()

        results = []          # (kind, status)
        retries = {kind: 0 for kind in orders}
        failed_markers = {kind: set() for kind in orders}
        order_idx = 0

        while rclpy.ok():
            if viewer is not None:
                key = viewer.show()
                if key in (ord("q"), 27):
                    controller.get_logger().info(
                        "q/Esc pressed in result window; stopping")
                    rclpy.shutdown()
                    break
            if controller is None:
                time.sleep(0.02)
                continue

            if controller.order_done:
                kind = controller.target_kind
                results.append((kind, "done"))
                controller.get_logger().info(
                    f"[orders] COMPLETE order {controller.order_index + 1}: "
                    f"{kind}")
                finish_controller(controller)
                order_idx += 1
                if order_idx >= len(orders):
                    controller = None
                    break
                start_order(orders[order_idx], order_idx)
            elif controller.order_aborted:
                kind = controller.target_kind
                reason = controller.abort_reason or "unknown"
                failed_marker = controller.target_marker_id
                if failed_marker is not None:
                    failed_markers[kind].add(int(failed_marker))
                retries[kind] += 1
                controller.get_logger().error(
                    f"[orders] FAILED order: {kind} ({reason}); "
                    f"retry={retries[kind]}/{args.max_retries} "
                    f"excluded={sorted(failed_markers[kind])}")
                finish_controller(controller)
                if retries[kind] <= args.max_retries:
                    start_order(
                        kind, order_idx,
                        exclude_marker_ids=failed_markers[kind])
                else:
                    results.append((kind, f"failed({reason})"))
                    order_idx += 1
                    if order_idx >= len(orders):
                        controller = None
                        break
                    start_order(orders[order_idx], order_idx)
            else:
                time.sleep(0.02)

        summary = "; ".join(f"{k}:{s}" for k, s in results)
        log_node = (
            controller
            if controller is not None
            else (nodes[0] if nodes else None))
        if log_node is not None:
            if order_idx >= len(orders):
                log_node.get_logger().info(
                    f"[orders] ALL ORDERS PROCESSED ({len(results)}): "
                    f"{summary}")
            else:
                log_node.get_logger().info(
                    f"[orders] STOPPED EARLY at order "
                    f"{order_idx + 1}/{len(orders)}; results so far: "
                    f"{summary}")
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        executor.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        for node in nodes:
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        if args.show:
            import cv2
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
