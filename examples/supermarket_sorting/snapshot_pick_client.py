#!/usr/bin/env python3
"""正式行走录入 + 记忆矩阵直达的连续多单超市分拣客户端。

这是当前唯一的 GUI 多单客户端入口，在抓取订单前增加"逐架行走录入"阶段：

    阶段一（--record-first，默认开启）：
        完全按照正式版的行走流程：导航器逐架走到每个货架前的标准扫描站点
        （SCAN_X × SCAN_Y=2.475），用正式扫描位姿（7 个视角）在近距录入
        该架全部商品，E→D→C→B→A 走完一遍，写入 3×15 记忆矩阵
        logs/memory_matrix.json。近距录入保证层高（L1/L2/L3）判断准确，
        也不走快照位的远距离两段式通道（省掉不必要的距离调整/旋转）；
    阶段二：
        连续处理订单列表，每单先查记忆矩阵：命中则用 scan hint 直达该货架/
        该层做局部定位（避免整排货架扫描），未命中才回退全量扫描。

抓取阶段执行完整的连续多单流程：

    抓货区抓取 -> 导航到终点 -> 抬升释放 -> 收臂倒车离开 ->
    导航返回抓货区 -> 开始下一个订单

防死锁设计：
  * 每次卸货后强制导航回到抓货区才进入下一单，绝不在终点停住；
  * 导航阶段有进度看门狗：长时间无位移时强制重新设置导航目标并重规划；
    持续卡死则放弃当前订单，继续后面的订单；
  * 抓取失败（找不到/抓取掉落）允许按订单重试，重试耗尽后跳过，整个客户端
    不中断；
  * 控制层运行期间屏蔽 ``rclpy.shutdown``（由顶层 main 统一收尾），防止单次
    流程/中止把整个客户端杀掉；
  * 卸货后先倒车一小段再返回，避免刚扔下的货物挡住激光并触发安全急停。

用法::

    python3 examples/supermarket_sorting/snapshot_pick_client.py \
        --orders-count 5 --seed 11 --record-first --max-scan-cycles 2

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
from memory_matrix import (  # noqa: E402
    LEVEL_MARKER_Z,
    SHELF_SCAN_X,
    MemoryMatrixTracker,
    grasp_eligible_candidates,
)
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
# 已完成多帧确认且在该距离内观察到的候选，即使置信度略低于常规阈值，
# 也允许作为一次近处局部核验目标。导航仍只使用货架+层，不使用列。
MEMORY_CLOSE_OBSERVATION_M = 1.35
MEMORY_CLOSE_CONFIDENCE_MIN = 0.70

# 行进中发现更近的可靠货架时，至少节省这些路程才改道；每单最多一次。
MEMORY_REROUTE_SAVING_M = 0.30


def _shelf_for_scan_x(world_x: float) -> str:
    """把扫描站 x 归一到货架字母。"""
    return min(
        SHELF_SCAN_X,
        key=lambda shelf: abs(SHELF_SCAN_X[shelf] - float(world_x)))


def _memory_candidate_allowed(
        candidate: dict, excluded_slots=(), excluded_shelves=(),
        excluded_shelf_levels=(),
        min_last_seen: float | None = None) -> bool:
    """应用记忆候选的槽位、货架、层和新鲜度门槛。"""
    if str(candidate.get("slot_key", "")) in set(excluded_slots or ()):
        return False
    shelf = str(candidate.get("shelf", ""))
    level = str(candidate.get("level", ""))
    if shelf in set(excluded_shelves or ()):
        return False
    if (shelf, level) in set(excluded_shelf_levels or ()):
        return False
    if min_last_seen is not None:
        try:
            last_seen = float(candidate.get("last_seen", 0.0))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(last_seen) or last_seen < float(min_last_seen):
            return False
    return True


def _memory_direct_hint_ok(hint: dict) -> bool:
    """取消可靠门槛：任何记忆候选都允许直达抓取站位。

    距离/深度 y/样本数/新鲜度不再拦截；若直达槽位与实际不符，
    由抓取前的 close-recheck 复核失败后排除该槽位并回退扫描兜底。
    """
    return True


def _select_memory_hint(
        candidates, base_xy, conf_threshold: float,
        exclude_slots=(), exclude_shelves=(), exclude_shelf_levels=(),
        min_last_seen: float | None = None,
        reliable_only: bool = False,
        require_direct: bool = False):
    """从 GUI 主候选中选择货架+层提示，返回选择细节。

    可靠候选按机器人当前行程最近优先；只有没有可靠候选且允许兜底时，
    才按置信度选择。独立成纯函数，便于用实跑矩阵回放导航决策。
    """
    if base_xy is None:
        base_xy = (float("nan"), float("nan"))

    def dist_to(x: float) -> float:
        try:
            dx = float(base_xy[0]) - x
            dy = float(base_xy[1]) - pick.SCAN_Y
        except (TypeError, ValueError, IndexError):
            return float("inf")
        distance = math.hypot(dx, dy)
        return distance if math.isfinite(distance) else float("inf")

    best = None
    nearest = None
    for candidate in candidates:
        if not _memory_candidate_allowed(
                candidate, exclude_slots, exclude_shelves,
                exclude_shelf_levels, min_last_seen):
            continue
        shelf = str(candidate.get("shelf", ""))
        level = str(candidate.get("level", ""))
        x = SHELF_SCAN_X.get(shelf)
        z = LEVEL_MARKER_Z.get(level)
        if x is None or z is None:
            continue
        confidence = float(candidate.get("confidence", 0.0))
        observed_distance = candidate.get("closest_distance")
        try:
            observed_distance = float(observed_distance)
        except (TypeError, ValueError):
            observed_distance = float("inf")
        travel = dist_to(x)
        item = {
            "x": x,
            "z": z,
            "shelf": shelf,
            "level": level,
            "column": str(candidate.get("column", "")),
            "slot_key": str(candidate.get("slot_key", "")),
            "confidence": confidence,
            "observed_distance": observed_distance,
            "world_y": candidate.get("world_y"),
            "world_z": candidate.get("world_z"),
            "sample_count": candidate.get("sample_count"),
            "last_seen": candidate.get("last_seen"),
            "travel": travel,
            "close_relaxed": False,
            "reliable": False,
        }
        fallback_key = (travel, -confidence, observed_distance, x, z)
        if best is None or fallback_key < best[0]:
            best = (fallback_key, item)
        close_reliable = (
            observed_distance <= MEMORY_CLOSE_OBSERVATION_M
            and confidence >= MEMORY_CLOSE_CONFIDENCE_MIN)
        if confidence >= conf_threshold or close_reliable:
            item = dict(item)
            item["close_relaxed"] = (
                close_reliable and confidence < conf_threshold)
            item["reliable"] = True
            if require_direct and not _memory_direct_hint_ok(item):
                continue
            reliable_key = (travel, observed_distance, -confidence, x, z)
            if nearest is None or reliable_key < nearest[0]:
                nearest = (reliable_key, item)
    if nearest is not None:
        return nearest[1]
    if reliable_only or best is None:
        return None
    return best[1]


def _update_memory_scan_progress(controller) -> str | None:
    """记录本单已经完整扫完并离开的货架。

    控制器在同一货架的摄像头位姿之间也会反复进入 GO_SCAN，
    因此只有扫描站 x 真正变化时才标记上一架已扫完。
    """
    if controller.scan_station_order is None:
        return None
    current_x = float(controller.current_scan_station_x())
    previous_x = getattr(controller, "memory_last_scan_station_x", None)
    if previous_x is None:
        controller.memory_last_scan_station_x = current_x
        return None
    if abs(current_x - float(previous_x)) <= 0.40:
        return None
    shelf = _shelf_for_scan_x(float(previous_x))
    controller.memory_exhausted_shelves.add(shelf)
    controller.memory_last_scan_station_x = current_x
    controller.get_logger().info(
        f"[memory] shelf {shelf} fully scanned without localisation; "
        "suppressing dynamic revisit for this order")
    return shelf


def _consume_grabbed_memory(controller, matrix_tracker) -> bool:
    """幂等地把刚确认离架的货物从矩阵中立即标为已取走。"""
    if getattr(controller, "memory_consumed", False):
        return False
    slot = controller.target_slot()
    if slot is not None:
        matrix_tracker.consume_slot(
            *slot, kind=controller.target_kind)
        controller.memory_consumed = True
        controller.get_logger().info(
            f"[memory] immediate post-grasp consume kind="
            f"{controller.target_kind} slot={slot}")
        return True
    if controller.target_marker_id is not None:
        # 兼容旧目标；不参与矩阵导航或抓取调整逻辑。
        matrix_tracker.consume(controller.target_marker_id)
        controller.memory_consumed = True
        return True
    return False


class FormalWalkRecorder(integrated.IntegratedNavPickPlace):
    """正式行走式录入：按正式流程逐架走到货架前标准站点录入全部商品。

    复用正式 client 的导航器 drive_to（不走远距离两段式通道），站点和扫描
    位姿都用正式常量（SCAN_X / SCAN_Y=2.475 / SCAN_CAMERA_POSES），只在
    关联/补拍/位置回退上保持"只记录不抓取"，感知全程常开。
    """

    def __init__(self, passes: int = 1) -> None:
        super().__init__(
            "kele", max_scan_cycles=1,
            tcp_diagnostic_ground_truth=False, scan_skip_lower=False,
            nav_during_scan=True, close_recheck=False)
        self.stations = [float(x) for x in pick.SCAN_X]
        self.default_scan_poses = pick.SCAN_CAMERA_POSES
        self.scan_poses = pick.SCAN_CAMERA_POSES
        self.max_scan_cycles = max(1, int(passes))
        self.finished = False
        self._last_finished_station = None

    def _nearest_scan_stations(self) -> list[int]:
        """按 E→D→C→B→A 固定顺序逐架走（与正式流程的站点顺序一致）。"""
        return list(range(len(self.stations)))

    def _publish_perception_request(self, enabled, force=False):
        """录入阶段感知全程常开：避免冷启动漏帧。"""
        return

    def current_scan_station_x(self) -> float:
        if self.scan_station_order is not None:
            idx = self.scan_station_order[self.scan_index]
        else:
            idx = self.scan_index
        return float(self.stations[idx % len(self.stations)])

    def _advance_scan_pose(self) -> bool:
        """用 len(self.stations) 支持逐架顺序推进；扫完一架进下一架。"""
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
                        f"[record] swept {len(self.stations)} stations "
                        f"x {self.scan_cycles} pass(es); stopping")
                    self.set_state(pick.STATE_ABORT)
                    return False
        if self.state != pick.STATE_ABORT:
            self.set_state(pick.STATE_GO_SCAN)
        return self.state != pick.STATE_ABORT

    def try_association_locked(self) -> None:
        """录入阶段不做目标关联/定位/抓取，只由 MemoryMatrixTracker 记录。"""
        return

    def _maybe_lock_yolo_only_target_locked(self) -> None:
        """录入阶段不锁定目标、不抓取（记忆矩阵由 tracker 独立记录）。"""
        return

    def _start_revisit(self) -> None:
        """跳过补拍（补拍失败会经 position-fallback 进入抓取）。"""
        return

    def _try_position_fallback(self) -> bool:
        """禁用位置回退：它会把 YOLO 框直接定位并进入 ALIGN 抓取。"""
        return False

    def set_state(self, new_state: str) -> None:
        if new_state == pick.STATE_ABORT:
            # 录入按计划跑完（父级在全部站点扫完后会走到 ABORT）。
            self.finished = True
            self.get_logger().info(
                "[record] 录入完成; stopping recorder")
            return
        super().set_state(new_state)

    def tick(self) -> None:
        super().tick()
        if self.finished:
            self.set_twist(0.0, 0.0)
            self.smooth_commands()
            self.publish_commands()


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
DROP_BACKUP_SPEED_MPS = 0.25
DROP_BACKUP_TIMEOUT_S = 8.0
# 终点抬升释放：把手臂抬高到上层货架抓取高度（slide 收到顶、TCP z≈1.2m），
# 再伸长手臂、松爪释放（不做桌面放置）。
DROP_RAISE_TCP_Z = 1.20         # 上层货架抓取高度（TCP z，米）
DROP_RAISE_EXTEND_M = 0.55      # 伸长距离（朝南越过配送区）
DROP_RAISE_SLIDE = -0.04        # 抬升 slide（顶层抓取姿态，= SLIDE_MIN）
# 抬臂/伸臂是"高位持货"姿态，误差可容忍：达到软门槛就提前进入下一步，
# 不再干等收敛（KDL 与仿真的运动学偏差常让 0.06 rad 硬门槛永远过不了）。
DROP_RAISE_SOFT_S = 3.0
DROP_RAISE_SOFT_ARM_TOLERANCE_RAD = 0.10
DROP_RAISE_SOFT_SLIDE_TOLERANCE_M = 0.05
DROP_RAISE_TIMEOUT_S = 8.0      # 抬升到位硬超时兜底
# 松爪前最后延展：把手臂从抬升姿态再往外伸一段，让货物越过桌面更深处
DROP_RAISE_EXTEND_FINAL_M = 0.75
DROP_RAISE_EXTEND_TIMEOUT_S = 6.0
# 放货前朝桌子方向前进一小段（不怕撞桌；带超时防死锁）
DROP_CREEP_DIST_M = 0.50
DROP_CREEP_SPEED_MPS = 0.10
DROP_CREEP_TIMEOUT_S = 6.0

ABORT_SETTLE_S = 1.0           # 抓取中止后等待手臂收回稳定
ABORT_SETTLE_TIMEOUT_S = 15.0  # 收臂超时兜底：即使没完全到位也放行下一单

# 导航防死锁看门狗
NAV_STALL_CHECK_S = 8.0        # 每隔这么久检查一次位移
NAV_STALL_MIN_PROGRESS_M = 0.05
NAV_STALL_MAX_RESETS = 6       # 连续多次无进展 -> 放弃当前订单

# nav_to_delivery 局部死锁时的独立倒车恢复。
NAV_RECOVERY_BACKUP_DIST_M = 0.25
NAV_RECOVERY_BACKUP_SPEED_MPS = 0.15
NAV_RECOVERY_TIMEOUT_S = 4.0
NAV_RECOVERY_MAX_ATTEMPTS = 3
NAV_RECOVERY_REAR_STOP_M = 0.45

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
                 backup_after_grab_m: float = 0.15,
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
        # 记忆矩阵必须在抓取完成事件发生时立即消费，不能等送货并返回。
        self.memory_consume_callback = None
        self.memory_consumed = False
        # 一条矩阵提示只核验对应的“货架+层”。提示层失败后由编排器
        # 立即选择下一条矩阵证据，不在当前架自动展开七视角全扫描。
        self.memory_active_hint = None
        self.memory_failed_hint = None
        self.memory_failed_hint_levels = set()
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
        # nav_to_delivery 独立倒车恢复状态
        self._nav_recovery_phase = None
        self._nav_recovery_start_xy = None
        self._nav_recovery_start_yaw = 0.0
        self._nav_recovery_started_at = 0.0
        self._nav_recovery_goal = None
        self._nav_recovery_attempts = 0

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

    def _on_grab_complete(self) -> None:
        """抓取成功后立刻同步矩阵，再开始送货阶段。"""
        super()._on_grab_complete()
        callback = self.memory_consume_callback
        if callback is None or self.memory_consumed:
            return
        try:
            callback(self)
        except Exception as exc:  # 矩阵写盘失败不能让已抓货流程中断
            self.get_logger().warn(
                f"[memory] immediate consume failed; will retry at order "
                f"completion: {exc}")

    def _restore_full_scan_after_inventory_hint(self) -> None:
        """把提示层失败事件交给矩阵导航，避免原地展开全架扫描。"""
        was_active = self.inventory_scan_hint_active
        failed_hint = self.memory_active_hint
        super()._restore_full_scan_after_inventory_hint()
        if was_active and failed_hint is not None:
            self.memory_failed_hint = failed_hint
            self.memory_active_hint = None

    def _handle_abort(self) -> None:
        if self._abort_handled:
            return
        self._abort_handled = True
        if getattr(self, "no_middle_tissue", False):
            self._abort_reason = "no middle-column tissue"
        else:
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
    def _place_base_motion_active(self) -> bool:
        """The continuous drop flow performs its table creep in stage 1."""
        return self.place_stage == 1

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
            raise_elapsed = now - self._drop_raise_t0
            measured_slide = self.joints.get("slide_joint")
            slide_error = (
                float("inf") if measured_slide is None
                else abs(float(measured_slide) - self.des_slide))
            soft_ready = (
                raise_elapsed >= DROP_RAISE_SOFT_S
                and self.selected_arm_error()
                <= DROP_RAISE_SOFT_ARM_TOLERANCE_RAD
                and slide_error <= DROP_RAISE_SOFT_SLIDE_TOLERANCE_M)
            if (self.commands_ready(
                    arm_tolerance=0.06, slide_tolerance=0.03)
                    or soft_ready
                    or raise_elapsed >= DROP_RAISE_TIMEOUT_S):
                self.place_stage = 1
                self.place_t0 = now
                self.get_logger().info(
                    "[drop-raise] arm raised; creeping towards the table"
                    + (" (soft)" if soft_ready else ""))
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
            extend_elapsed = now - self._drop_extend_t0
            measured_slide = self.joints.get("slide_joint")
            slide_error = (
                float("inf") if measured_slide is None
                else abs(float(measured_slide) - self.des_slide))
            soft_ready = (
                extend_elapsed >= DROP_RAISE_SOFT_S
                and self.selected_arm_error()
                <= DROP_RAISE_SOFT_ARM_TOLERANCE_RAD
                and slide_error <= DROP_RAISE_SOFT_SLIDE_TOLERANCE_M)
            if (self.commands_ready(
                    arm_tolerance=0.06, slide_tolerance=0.03)
                    or soft_ready
                    or extend_elapsed >= DROP_RAISE_EXTEND_TIMEOUT_S):
                self.place_stage = 3
                self.place_t0 = now
                self.get_logger().info(
                    "[drop-extend] arm extended; opening gripper"
                    + (" (soft)" if soft_ready else ""))
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
        # 卸货后基座往往仍停在送货桌禁入区内：原始导航器遇到 table_keepout
        # 会原地停车不恢复（v=0,w=0）。与集成 drive_to 的逃逸逻辑一致：
        # 先向北倒出禁入区，再交给导航器。
        table_clearance = integrated.point_to_rect_clearance(
            float(self.base_xy[0]), float(self.base_xy[1]),
            integrated.DELIVERY_TABLE_COSTMAP_BOUNDS)
        safe_clearance = (
            integrated.WHOLE_BODY_KEEP_OUT_RADIUS
            + integrated.PLACE_CLEAR_TABLE_MARGIN_M)
        if table_clearance < safe_clearance:
            yaw_error = pick.wrap_to_pi(-math.pi / 2.0 - self.base_yaw)
            if abs(yaw_error) <= 0.20:
                self.set_twist(
                    -integrated.PLACE_CLEAR_TABLE_SPEED_MPS,
                    float(np.clip(2.0 * yaw_error, -0.25, 0.25)))
            else:
                self.set_twist(0.0, 0.0)
            if not self._table_escape_logged:
                self._table_escape_logged = True
                self.get_logger().warn(
                    "[return-shelf] inside delivery-table keep-out; "
                    f"clearance={table_clearance:.3f}m "
                    f"required={safe_clearance:.3f}m; reversing north "
                    "before normal navigation")
            return
        if self._table_escape_logged:
            self._table_escape_logged = False
            self.get_logger().info(
                "[return-shelf] table keep-out escape complete; "
                f"clearance={table_clearance:.3f}m")
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
        goal = tuple(float(value) for value in goal)
        if (self._watchdog_phase != self.flow_phase
                or self._watchdog_goal != goal):
            self._watchdog_phase = self.flow_phase
            self._watchdog_t0 = now
            self._watchdog_last_xy = (
                None if self.base_xy is None else self.base_xy.copy())
            self._watchdog_resets = 0
            self._watchdog_goal = goal
            self._nav_recovery_attempts = 0
            return
        if self.base_xy is None or self._watchdog_last_xy is None:
            return


        # Heweidao's deliberately slow first turn can spend the whole initial
        # window rotating with almost no XY displacement.  That is commanded
        # progress, not a navigation stall; start the normal stall clock only
        # after the initial loaded-turn watchdog grace expires.
        if (self.flow_phase == "nav_to_delivery"
                and self._post_grab_slow_turn_active()):
            self._watchdog_t0 = now
            self._watchdog_last_xy = self.base_xy.copy()
            self._nav_recovery_attempts = 0
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
            # 先尝试让底层导航器执行安全倒车/侧向脱困；不行再试独立倒车。
            if self._try_force_nav_recovery(now):
                self.get_logger().warn(
                    f"[anti-deadlock] flow={self.flow_phase} no progress; "
                    "requesting safe reverse/lateral recovery before goal reset")
                self._watchdog_t0 = now
                self._watchdog_last_xy = self.base_xy.copy()
                return
            if self._start_nav_recovery(goal):
                self.get_logger().warn(
                    f"[anti-deadlock] flow={self.flow_phase} no progress; "
                    "starting measured backup recovery before goal reset")
                self._watchdog_t0 = now
                self._watchdog_last_xy = self.base_xy.copy()
                return
            self._watchdog_resets += 1
            self.get_logger().warn(
                f"[anti-deadlock] flow={self.flow_phase} no progress for "
                f"{elapsed:.1f}s (moved={moved:.3f}m); resetting nav goal "
                f"{goal} (reset={self._watchdog_resets}/"
                f"{NAV_STALL_MAX_RESETS})")
            self.nav.set_goal(*goal)
            self._watchdog_t0 = now
            self._watchdog_last_xy = self.base_xy.copy()
            self._nav_recovery_attempts = 0
            if self._watchdog_resets >= NAV_STALL_MAX_RESETS:
                self._order_fail(
                    f"navigation stalled in {self.flow_phase} towards "
                    f"{goal} ({self._watchdog_resets} resets)")
        elif moved >= NAV_STALL_MIN_PROGRESS_M:
            # 只有真正移动了才重置停滞计时；原地不动时让计时持续累积，
            # 否则每 tick 重置 t0 会导致看门狗永远到不了检查间隔。
            self._watchdog_t0 = now
            self._watchdog_last_xy = self.base_xy.copy()
            self._nav_recovery_attempts = 0

    def _try_force_nav_recovery(self, now: float) -> bool:
        """Ask the navigation controller to execute its existing recovery."""
        if self.flow_phase != "nav_to_delivery" or self.laser_msg is None:
            return False
        controller = getattr(self.nav, "controller", None)
        if controller is None or not hasattr(
                controller, "_maybe_start_reverse_recovery"):
            return False
        try:
            return bool(controller._maybe_start_reverse_recovery(
                "arc_blocked",
                float(self.base_xy[0]),
                float(self.base_xy[1]),
                float(self.base_yaw),
                self.laser_msg,
                now))
        except Exception:
            return False

    def _nav_rear_clearance(self) -> float | None:
        if self.laser_msg is None:
            return None
        controller = getattr(self.nav, "controller", None)
        if controller is None or not hasattr(controller, "_rear_clearance"):
            return None
        try:
            return float(controller._rear_clearance(self.laser_msg))
        except Exception:
            return None

    def _nav_recovery_static_path_free(
            self, start_xy, start_yaw, signed_distance) -> bool:
        """只检查配送台禁入区；其余静态障碍由后向激光距离兜底。"""
        distance = abs(float(signed_distance))
        direction = 1.0 if signed_distance > 0.0 else -1.0
        cosine, sine = math.cos(start_yaw), math.sin(start_yaw)
        sample_step = 0.05
        steps = max(1, int(math.ceil(distance / sample_step)))
        for index in range(1, steps + 1):
            travelled = distance * index / steps
            x = float(start_xy[0]) + direction * travelled * cosine
            y = float(start_xy[1]) + direction * travelled * sine
            if (integrated.point_to_rect_clearance(
                    x, y, integrated.DELIVERY_TABLE_COSTMAP_BOUNDS)
                    <= integrated.WHOLE_BODY_KEEP_OUT_RADIUS):
                return False
        return True

    def _start_nav_recovery(self, goal) -> bool:
        """在 nav_to_delivery 停滞时开始一段独立测距倒车。"""
        if (self.flow_phase != "nav_to_delivery"
                or self._nav_recovery_phase is not None
                or self.base_xy is None
                or self._nav_recovery_attempts >= NAV_RECOVERY_MAX_ATTEMPTS):
            return False
        rear = self._nav_rear_clearance()
        if rear is None or rear <= NAV_RECOVERY_REAR_STOP_M:
            self.get_logger().warn(
                f"[nav-recovery] backup skipped: rear clearance="
                f"{rear if rear is None else round(rear, 2)}m "
                f"required>{NAV_RECOVERY_REAR_STOP_M:.2f}m")
            return False
        if not self._nav_recovery_static_path_free(
                self.base_xy, self.base_yaw,
                -NAV_RECOVERY_BACKUP_DIST_M):
            self.get_logger().warn(
                "[nav-recovery] backup skipped: table keep-out on reverse path")
            return False
        self._nav_recovery_phase = "backup"
        self._nav_recovery_start_xy = self.base_xy.copy()
        self._nav_recovery_start_yaw = float(self.base_yaw)
        self._nav_recovery_started_at = self.now()
        self._nav_recovery_goal = goal
        self._nav_recovery_attempts += 1
        self.set_twist(0.0, 0.0)
        self.get_logger().info(
            f"[nav-recovery] starting measured backup "
            f"{NAV_RECOVERY_BACKUP_DIST_M:.2f}m at "
            f"{NAV_RECOVERY_BACKUP_SPEED_MPS:.2f}m/s "
            f"(attempt={self._nav_recovery_attempts}/"
            f"{NAV_RECOVERY_MAX_ATTEMPTS}) rear={rear:.2f}m")
        return True

    def _finish_nav_recovery(self, reason: str) -> None:
        now = self.now()
        goal = self._nav_recovery_goal
        self._nav_recovery_phase = None
        self._nav_recovery_start_xy = None
        self._nav_recovery_start_yaw = 0.0
        self._nav_recovery_started_at = 0.0
        self._nav_recovery_goal = None
        self.set_twist(0.0, 0.0)
        if goal is not None:
            self.nav.set_goal(*goal)
        self._watchdog_phase = None
        self._watchdog_t0 = now
        self._watchdog_last_xy = (
            None if self.base_xy is None else self.base_xy.copy())
        self.get_logger().info(
            f"[nav-recovery] finished ({reason}); replanning towards "
            f"{goal}")

    def _nav_recovery_tick(self) -> None:
        if (self._nav_recovery_phase != "backup"
                or self._nav_recovery_start_xy is None):
            self._finish_nav_recovery("invalid_state")
            return
        now = self.now()
        heading = np.array([
            math.cos(self._nav_recovery_start_yaw),
            math.sin(self._nav_recovery_start_yaw),
        ])
        moved_back = float(max(
            0.0,
            np.dot(self._nav_recovery_start_xy - self.base_xy, heading)))
        elapsed = max(0.0, now - self._nav_recovery_started_at)
        remaining = max(
            0.0, NAV_RECOVERY_BACKUP_DIST_M - moved_back)
        if moved_back >= NAV_RECOVERY_BACKUP_DIST_M:
            self._finish_nav_recovery("distance_reached")
            return
        if elapsed >= NAV_RECOVERY_TIMEOUT_S:
            self._finish_nav_recovery("timeout")
            return

        rear = self._nav_rear_clearance()
        if rear is None or rear <= NAV_RECOVERY_REAR_STOP_M:
            self._finish_nav_recovery(
                f"rear_stop rear={rear:.2f}m" if rear is not None
                else "rear_scan_unavailable")
            return
        if not self._nav_recovery_static_path_free(
                self.base_xy, self._nav_recovery_start_yaw, -remaining):
            self._finish_nav_recovery("static_path_blocked")
            return

        yaw_error = pick.wrap_to_pi(
            self._nav_recovery_start_yaw - self.base_yaw)
        angular = float(np.clip(
            2.0 * yaw_error, -0.4, 0.4))
        self.set_twist(-NAV_RECOVERY_BACKUP_SPEED_MPS, angular)

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
            self._nav_watchdog_check(self._delivery_watchdog_goal())
            if self._nav_recovery_phase is not None:
                self._nav_recovery_tick()
            elif not self.order_aborted:
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
    parser.add_argument("--backup-after-grab", type=float, default=0.15)
    parser.add_argument("--drop-release-dwell", type=float,
                        default=DROP_RELEASE_DWELL_S)
    parser.add_argument("--drop-retreat-dwell", type=float,
                        default=DROP_RETREAT_DWELL_S)
    parser.add_argument("--max-retries", type=int, default=2,
                        help="retries per order before skipping it")
    parser.add_argument(
        "--record-first", action="store_true",
        help="开始抓取前先按正式行走流程逐架录入，记录记忆矩阵")
    parser.add_argument(
        "--record-passes", type=int, default=1,
        help="行走录入趟数（默认 1）")
    parser.add_argument(
        "--record-dwell", type=float, default=1.0,
        help="录入每扫描位姿驻留秒数（默认 1.0）")
    parser.add_argument(
        "--matrix-confirmations", type=int, default=2,
        help="记忆矩阵每格最少确认帧数（默认 2，另受深度中位数 "
             "depth_min_samples=4 约束）")
    parser.add_argument(
        "--memory-conf-threshold", type=float, default=0.90,
        help="记忆矩阵常规可靠阈值；近距多帧记录可用较低置信度参与就近"
             "核验（默认 0.90）")
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
    if args.record_passes < 1:
        parser.error("--record-passes must be >= 1")
    if args.record_dwell <= 0.0:
        parser.error("--record-dwell must be positive")
    if args.matrix_confirmations < 1:
        parser.error("--matrix-confirmations must be >= 1")
    if not 0.0 <= args.memory_conf_threshold <= 1.0:
        parser.error("--memory-conf-threshold must be in [0, 1]")
    if args.orders:
        try:
            parse_orders_arg(args.orders)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main() -> None:
    from run_log import start_run_log
    start_run_log("snapshot_pick_client")
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
        matrix_tracker = MemoryMatrixTracker(
            confirmations=args.matrix_confirmations)
        nodes += [yolo_node, aruco_node, matrix_tracker]
        executor.add_node(yolo_node)
        executor.add_node(aruco_node)
        executor.add_node(matrix_tracker)

        def _ensure_spin():
            nonlocal spin_thread
            if spin_thread is None:
                def spin_in_background():
                    try:
                        executor.spin()
                    except ExternalShutdownException:
                        pass

                spin_thread = threading.Thread(
                    target=spin_in_background, name="ros2_executor",
                    daemon=True)
                spin_thread.start()

        # ── 阶段一：正式行走逐架录入，记录记忆矩阵 ──
        # 每局开始时矩阵干净起步；订单阶段 tracker 常开，抓取过程中的扫描
        # 会持续录入（"余光"模式：不专门行走，靠订单扫描逐步填矩阵）。
        matrix_tracker.reset()
        if args.record_first:
            recorder = FormalWalkRecorder(passes=args.record_passes)
            recorder.configure_local_perception(yolo_node, aruco_node)
            yolo_node.set_enabled(True)
            aruco_node.set_enabled(True)
            original_dwell = pick.SCAN_DWELL_S
            pick.SCAN_DWELL_S = float(args.record_dwell)
            executor.add_node(recorder)
            nodes.append(recorder)
            recorder.get_logger().info(
                f"[record] 正式行走录入开始: "
                f"{len(pick.SCAN_X)} 架 x {args.record_passes} 趟, "
                f"dwell={args.record_dwell}s (站点 SCAN_Y=2.475)")
            _ensure_spin()
            while rclpy.ok() and not recorder.finished:
                time.sleep(0.005)
            pick.SCAN_DWELL_S = original_dwell
            executor.remove_node(recorder)
            try:
                nodes.remove(recorder)
            except ValueError:
                pass
            recorder.destroy_node()
            recorder.get_logger().info(
                "[record] 行走录入完成，进入订单抓取")

        def _memory_hint_for(
                kind: str, exclude_slots=None, log_decision: bool = True,
                exclude_shelves=None, min_last_seen: float | None = None,
                reliable_only: bool = False,
                exclude_shelf_levels=None,
                require_direct: bool = False):
            """从记忆矩阵里找同类且未取走、未排除的槽位作为扫描直达提示。

            矩阵由 YOLO + 固定货架几何写入（不含 marker），导航直达直接用
            固定货架/层坐标（SHELF_SCAN_X / LEVEL_MARKER_Z）。

            每个物理格可保留多个品类候选，避免一次错分类抹掉真实记录。
            常规高置信度候选与近距离多帧候选都可参与；在可靠候选中按当前
            行程最近优先。列只用于排除具体失败槽位，不参与导航目的地。

            返回含 x/z/shelf/level 的选择结果或 None。
            """
            # 纸巾必须先排除侧列，再按距离选择。否则更近但不可抓的侧列
            # 候选会先胜出、随后被作废，并遮住稍远但可抓的中列候选。
            primary_candidates = grasp_eligible_candidates(
                kind, matrix_tracker.matrix.primary_candidates_for(kind))
            all_candidates = grasp_eligible_candidates(
                kind, matrix_tracker.matrix.candidates_for(kind))

            # 导航与 GUI 读取同一份主证据；被更近/更可靠的
            # 其他品类覆盖的隐藏历史候选不能单独触发直达。
            selected = _select_memory_hint(
                primary_candidates,
                matrix_tracker.base_xy,
                args.memory_conf_threshold,
                exclude_slots=exclude_slots,
                exclude_shelves=exclude_shelves,
                exclude_shelf_levels=exclude_shelf_levels,
                min_last_seen=min_last_seen,
                reliable_only=reliable_only,
                require_direct=require_direct)
            if selected is None:
                # 主证据里没有可用候选时，允许用“历史候选”按最近优先做
                # 扫描提示（不用于直达）。这些候选可能被同格的其他品类盖住，
                # 但作为“先去哪个货架看一眼”的提示比跑到最远货架更合理。
                if require_direct:
                    return None
                selected = _select_memory_hint(
                    all_candidates,
                    matrix_tracker.base_xy,
                    args.memory_conf_threshold,
                    exclude_slots=exclude_slots,
                    exclude_shelves=exclude_shelves,
                    exclude_shelf_levels=exclude_shelf_levels,
                    min_last_seen=min_last_seen,
                    reliable_only=False)
                if selected is None:
                    return None
                selected["hidden_fallback"] = True
            else:
                # 无论直达还是扫描提示，只要更近的货架上有“被同格其他品类
                # 盖住”的历史候选，就优先去近处看一眼。
                nearest_all = _select_memory_hint(
                    all_candidates,
                    matrix_tracker.base_xy,
                    args.memory_conf_threshold,
                    exclude_slots=exclude_slots,
                    exclude_shelves=exclude_shelves,
                    exclude_shelf_levels=exclude_shelf_levels,
                    min_last_seen=min_last_seen,
                    reliable_only=False)
                if (nearest_all is not None
                        and float(selected.get("travel", float("inf")))
                        > float(nearest_all.get("travel", float("inf")))
                        + 0.60):
                    nearest_all["hidden_fallback"] = True
                    selected = nearest_all
            if selected["reliable"]:
                if log_decision:
                    matrix_tracker.get_logger().info(
                        f"[memory] {kind}: 可靠候选就近导航 "
                        f"x={selected['x']:.3f} "
                        f"travel={selected['travel']:.2f}m "
                        f"observed={selected['observed_distance']:.2f}m "
                        f"conf={selected['confidence']:.3f} "
                        f"close_relaxed={int(selected['close_relaxed'])}")
            elif log_decision:
                matrix_tracker.get_logger().info(
                    f"[memory] {kind}: 无可靠候选，置信度兜底 "
                    f"x={selected['x']:.3f} "
                    f"conf={selected['confidence']:.3f} "
                    f"observed={selected['observed_distance']:.2f}m "
                    f"(threshold={args.memory_conf_threshold})")
            return selected

        def make_controller(
                kind: str, index: int,
                exclude_marker_ids=None, exclude_slots=None):
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
            ctrl.memory_rerouted = False
            ctrl.memory_exhausted_shelves = set()
            ctrl.memory_last_scan_station_x = None
            ctrl.memory_consume_callback = (
                lambda completed: _consume_grabbed_memory(
                    completed, matrix_tracker))
            # 只有第一单从最近的 E 货架开始扫；之后每单从最西侧 A 货架开始
            ctrl.scan_prefer_west_start = index > 0
            # 重试时排除上次抓取失败的槽位 marker，让扫描去找同类商品的
            # 另一个位置，避免每次重试都撞同一个卡住的槽位（死循环感）。
            if exclude_marker_ids:
                ctrl.excluded_marker_ids = set(exclude_marker_ids)
            # YOLO-only 路径按固定槽位排除（level|shelf|column）。
            if exclude_slots:
                ctrl.excluded_slot_keys = set(exclude_slots)
            # 记忆矩阵直达：如果本局扫描已记录过该品类且尚未取走的槽位，
            # 直接把扫描提示设到那个货架/层，控制器先直达该站做局部定位，
            # 失效再回退全量扫描。
            try:
                hint = _memory_hint_for(
                    kind, exclude_slots, require_direct=True)
            except Exception:  # noqa: BLE001 - 记忆提示失败不影响主流程
                hint = None
            direct_hint_ok = (
                hint is not None
                and (kind != "zhijin" or hint.get("column") == "2"))
            if not direct_hint_ok:
                try:
                    hint = _memory_hint_for(kind, exclude_slots)
                except Exception:  # noqa: BLE001 - 记忆提示失败不影响主流程
                    hint = None
                if (kind == "zhijin"
                        and hint is not None
                        and hint.get("column") != "2"):
                    hint = None
            if hint is not None:
                _hint_x, _hint_z = hint["x"], hint["z"]
                ctrl.memory_active_hint = (hint["shelf"], hint["level"])
                ctrl.memory_last_scan_station_x = _hint_x
                # 取消可靠门槛：只要记忆里有该品类主候选就直达抓取站位，
                # 由 close-recheck 复核兜底，不再要求 reliable/置信度。
                # 被同格其他品类盖住的 hidden_fallback 仍不直达，避免把
                # 误检/被覆盖的历史记录当成真实槽位（例如 A L2 C1 主品类
                # 是脉动，却残留一条误检可乐记录导致第 4 单跑回 A 货架）。
                if (direct_hint_ok
                        and not hint.get("hidden_fallback")
                        and hint.get("column") in {"1", "2", "3"}
                        and not args.no_close_recheck):
                    accepted = ctrl.configure_direct_slot_target(
                        hint["shelf"], hint["level"], hint["column"],
                        product_y=hint.get("world_y"),
                        product_z=hint.get("world_z"))
                    if accepted:
                        # 直达失败回退扫描时，从同一货架继续，而不是按
                        # scan_prefer_west_start 跑回最西侧 A 货架重扫。
                        ctrl.scan_preferred_x = _hint_x
                        ctrl.get_logger().info(
                            f"[memory] order {index + 1} kind={kind}: "
                            f"direct slot -> "
                            f"{hint['shelf']}-{hint['level']}-"
                            f"{hint['column']} conf="
                            f"{hint['confidence']:.3f} "
                            f"observed={hint['observed_distance']:.2f}m")
                        ctrl.memory_reroute_not_before = time.time()
                        return ctrl
                    ctrl.get_logger().warn(
                        f"[memory] direct slot rejected for kind={kind}; "
                        "using shelf-level scan hint")
                ctrl.configure_inventory_scan_hint(_hint_x, _hint_z)
                hint_z_text = (
                    "?" if _hint_z is None else f"{_hint_z:.3f}")
                ctrl.get_logger().info(
                    f"[memory] order {index + 1} kind={kind}: "
                    f"direct scan hint -> shelf x={_hint_x:.3f} "
                    f"marker_z={hint_z_text}")
            # last_seen 使用 time.time() 的墙钟时间。动态改道只处理
            # 本单建立后的新观测，避免同一份旧矩阵在途中反复决策。
            ctrl.memory_reroute_not_before = time.time()
            return ctrl

        def start_order(
                kind: str, index: int,
                exclude_marker_ids=None, exclude_slots=None):
            nonlocal controller, viewer
            ctrl = make_controller(
                kind, index,
                exclude_marker_ids=exclude_marker_ids,
                exclude_slots=exclude_slots)
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

        _ensure_spin()

        results = []          # (kind, status)
        retries = {kind: 0 for kind in orders}
        failed_markers = {kind: set() for kind in orders}
        failed_slots = {kind: set() for kind in orders}
        order_idx = 0
        last_memory_reroute_check = 0.0

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

            # 矩阵提示只负责核验它记录的货架+层。该层未找到目标时，
            # 立即切到剩余矩阵候选；不要先在当前架展开全层扫描，也不要
            # 落入无矩阵依据的相邻货架。
            memory_failover_applied = False
            failed_hint = controller.memory_failed_hint
            if (failed_hint is not None
                    and controller.state == pick.STATE_GO_SCAN
                    and controller.target_marker_id is None):
                controller.memory_failed_hint = None
                controller.memory_failed_hint_levels.add(failed_hint)
                try:
                    next_hint = _memory_hint_for(
                        controller.target_kind,
                        failed_slots[controller.target_kind],
                        log_decision=False,
                        exclude_shelves=(
                            controller.memory_exhausted_shelves),
                        exclude_shelf_levels=(
                            controller.memory_failed_hint_levels),
                        reliable_only=True)
                    if next_hint is not None:
                        hint_x, hint_z = next_hint["x"], next_hint["z"]
                        controller.configure_inventory_scan_hint(
                            hint_x, hint_z)
                        controller.memory_active_hint = (
                            next_hint["shelf"], next_hint["level"])
                        controller.scan_station_order = None
                        controller.scan_index = 0
                        controller.scan_pose_index = 0
                        controller.scan_camera_ready_since = None
                        controller.memory_last_scan_station_x = hint_x
                        memory_failover_applied = True
                        controller.get_logger().info(
                            f"[memory] hinted {failed_hint[0]}-"
                            f"{failed_hint[1]} did not confirm "
                            f"{controller.target_kind}; immediate matrix "
                            f"failover -> {next_hint['shelf']}-"
                            f"{next_hint['level']} x={hint_x:.3f}")
                    else:
                        controller.get_logger().info(
                            f"[memory] hinted {failed_hint[0]}-"
                            f"{failed_hint[1]} did not confirm "
                            f"{controller.target_kind}; no reliable matrix "
                            "candidate remains, resuming fallback scan")
                except Exception as exc:
                    matrix_tracker.get_logger().warn(
                        f"[memory] immediate matrix failover skipped: {exc}")

            # 在动态改道之前先记录完整扫描进度。例如 A 真正扫完后转去 B，
            # 此时 A 必须立即进入本单禁止回访集合。
            _update_memory_scan_progress(controller)

            # 去目标货架途中 YOLO 仍持续更新记忆。若新出现的近距可靠记录
            # 指向一个明显更近的货架，则本单最多改道一次；目标一旦锁定或
            # 已进入定点扫描便不再干预。这里只改货架+层导航提示。
            now = time.monotonic()
            if (not memory_failover_applied
                    and not controller.memory_rerouted
                    and controller.state == pick.STATE_GO_SCAN
                    and controller.target_marker_id is None
                    and matrix_tracker.base_xy is not None
                    and now - last_memory_reroute_check >= 0.25):
                last_memory_reroute_check = now
                try:
                    refreshed_hint = _memory_hint_for(
                        controller.target_kind,
                        failed_slots[controller.target_kind],
                        log_decision=False,
                        exclude_shelves=(
                            controller.memory_exhausted_shelves),
                        exclude_shelf_levels=(
                            controller.memory_failed_hint_levels),
                        min_last_seen=(
                            controller.memory_reroute_not_before),
                        reliable_only=True)
                    current_x = (
                        float(controller.scan_preferred_x)
                        if (controller.scan_station_order is None
                            and controller.scan_preferred_x is not None)
                        else controller.current_scan_station_x())
                    if refreshed_hint is not None:
                        hint_x = refreshed_hint["x"]
                        hint_z = refreshed_hint["z"]
                        current_travel = math.hypot(
                            matrix_tracker.base_xy[0] - current_x,
                            matrix_tracker.base_xy[1] - pick.SCAN_Y)
                        new_travel = math.hypot(
                            matrix_tracker.base_xy[0] - hint_x,
                            matrix_tracker.base_xy[1] - pick.SCAN_Y)
                        if (abs(hint_x - current_x) > 0.40
                                and new_travel + MEMORY_REROUTE_SAVING_M
                                < current_travel):
                            controller.configure_inventory_scan_hint(
                                hint_x, hint_z)
                            controller.memory_active_hint = (
                                refreshed_hint["shelf"],
                                refreshed_hint["level"])
                            controller.scan_station_order = None
                            controller.scan_index = 0
                            controller.scan_pose_index = 0
                            controller.scan_camera_ready_since = None
                            controller.memory_rerouted = True
                            controller.memory_last_scan_station_x = hint_x
                            controller.get_logger().info(
                                f"[memory] dynamic reroute "
                                f"{current_x:.3f}->{hint_x:.3f}; "
                                f"travel {current_travel:.2f}->"
                                f"{new_travel:.2f}m (once per order)")
                except Exception as exc:  # 记忆改道失败不能影响抓取主流程
                    matrix_tracker.get_logger().warn(
                        f"[memory] dynamic reroute skipped: {exc}")

            if controller.order_done:
                kind = controller.target_kind
                results.append((kind, "done"))
                # 正常路径已在抓取完成事件中消费；这里仅作异常回调兜底。
                _consume_grabbed_memory(controller, matrix_tracker)
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
                if (kind == "zhijin"
                        and getattr(controller, "no_middle_tissue", False)):
                    results.append((kind, "skipped(no-middle-tissue)"))
                    controller.get_logger().warn(
                        f"[orders] SKIP order {controller.order_index + 1}: "
                        "no middle-column tissue is available")
                    finish_controller(controller)
                    order_idx += 1
                    if order_idx >= len(orders):
                        controller = None
                        break
                    start_order(orders[order_idx], order_idx)
                    continue
                failed_marker = controller.target_marker_id
                if failed_marker is not None:
                    failed_markers[kind].add(int(failed_marker))
                failed_slot = controller.target_slot_key()
                if failed_slot is not None:
                    failed_slots[kind].add(failed_slot)
                retries[kind] += 1
                controller.get_logger().error(
                    f"[orders] FAILED order: {kind} ({reason}); "
                    f"retry={retries[kind]}/{args.max_retries} "
                    f"excluded_markers={sorted(failed_markers[kind])} "
                    f"excluded_slots={sorted(failed_slots[kind])}")
                finish_controller(controller)
                if retries[kind] <= args.max_retries:
                    start_order(
                        kind, order_idx,
                        exclude_marker_ids=failed_markers[kind],
                        exclude_slots=failed_slots[kind])
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
        try:
            matrix_tracker.tick_write()
        except (NameError, Exception):  # noqa: BLE001 - best-effort flush
            pass
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
