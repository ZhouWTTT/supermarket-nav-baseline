#!/usr/bin/env python3
"""Integrate the baseline SupermarketNavigator with the current shelf-pick pipeline.

This is a NEW orchestrator script only — it does not modify any existing file.
It subclasses ``ShelfPickController`` (yolo_aruco_shelf_pick.py) and reuses the
baseline navigation module (supermarket_navigation.py) purely by import.

Attempted end-to-end flow:

    start
      -> navigator drives to the shelf scan stations (GO_SCAN transit)
      -> YOLO depth + optional ArUco visual localisation
      -> grasp the requested goods (unchanged parent states)
      -> navigator drives through the obstacle corridor to the delivery table
      -> arm extends, lowers the held product near the table, releases, retreats

Usage (inside the client container, mirroring yolo_aruco_shelf_pick.py)::

    python3 examples/supermarket_sorting/integrated_nav_pick_place.py \
        --target-kind kele --max-scan-cycles 2
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# sys.path: current pick pipeline first (its own module-level sys.path setup
# then wins for mmk2_kdl / perception imports), baseline dir appended after.
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import yolo_aruco_shelf_pick as pick  # noqa: E402  (parent pipeline, unmodified)
from memory_matrix import (  # noqa: E402
    LEVEL_MARKER_Z,
    MEMORY_CONSUME_TOPIC,
    MEMORY_REROUTE_SAVING_M,
    STATION_Y_MAX,
    STATION_Y_MIN,
    candidates_from_document,
    memory_direct_candidate_allowed,
    primary_candidates_from_document,
    read_memory_document,
    select_memory_route_hint,
    shelf_for_scan_x,
)

from sensor_msgs.msg import LaserScan  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402
from supermarket_navigation import (  # noqa: E402  (baseline nav, unmodified)
    DELIVERY_APPROACH,
    DELIVERY_TRUNK_ENTRY,
    DELIVERY_TRUNK_EXIT,
    DELIVERY_TABLE_COSTMAP_BOUNDS,
    DELIVERY_TABLE_XML_BOUNDS,
    SHELF_APPROACH,
    START_POSE,
    SupermarketNavigator,
    WHOLE_BODY_KEEP_OUT_RADIUS,
    point_to_rect_clearance,
)

# ---------------------------------------------------------------------------
# MuJoCo compatibility shim.
#
# The official client image bundles mujoco 3.2.7, whose XML schema has no mesh
# ``inertia`` attribute and rejects the flat aruco marker quad used by the
# scene ("mesh volume is too small").  The FK model is only used to compute
# camera/site poses, so we replace the flat 3 cm quad with an equivalent
# 2 mm-thick box at runtime — valid on every mujoco version and identical for
# forward kinematics.  This only monkey-patches the class in this process; no
# repository file is modified.
# ---------------------------------------------------------------------------
import re as _re  # noqa: E402

_ARUCO_MESH_RE = _re.compile(
    r'<mesh name="aruco_marker_3cm_mesh".*?/>', _re.S)
_ARUCO_MESH_BOX = (
    '<mesh name="aruco_marker_3cm_mesh"\n'
    '          vertex="-0.015 -0.015 0  0.015 -0.015 0  0.015 0.015 0  '
    '-0.015 0.015 0  -0.015 -0.015 0.002  0.015 -0.015 0.002  '
    '0.015 0.015 0.002  -0.015 0.015 0.002"\n'
    '          texcoord="0 0  1 0  1 1  0 1  0 0  1 0  1 1  0 1"\n'
    '          face="0 1 2  0 2 3  4 5 6  4 6 7  0 4 5  0 5 1  '
    '1 5 6  1 6 2  2 6 7  2 7 3  3 7 4  3 4 0"/>')


def _mujoco_compat_xml(text: str) -> str:
    """Sanitise the scene XML so old client mujoco versions can load it."""
    return _ARUCO_MESH_RE.sub(_ARUCO_MESH_BOX, text)


try:  # noqa: E402
    import discoverse.robots.mmk2.mmk2_fk as _mmk2_fk_mod
except ImportError:
    _mmk2_fk_mod = None

if _mmk2_fk_mod is not None:
    _orig_mmk2fk_init = _mmk2_fk_mod.MMK2FK.__init__

    def _mmk2fk_compat_init(self, mjcf_path=None):
        if mjcf_path is None:
            task_dir = (
                pathlib.Path(_mmk2_fk_mod.DISCOVERSE_ROOT_DIR)
                / "examples" / "supermarket_sorting")
            src = task_dir / "mjcf" / "retail_competition.xml"
            runtime = pathlib.Path("/tmp/retail_competition_fk.xml")
            runtime.write_text(
                _mujoco_compat_xml(
                    src.read_text().replace(
                        "__REPO_ROOT__", str(task_dir))))
            mjcf_path = str(runtime)
        _orig_mmk2fk_init(self, mjcf_path)

    _mmk2_fk_mod.MMK2FK.__init__ = _mmk2fk_compat_init


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DELIVERY_TABLE_PLACE_WORLD = (-1.80, -3.35, 0.85)  # x, y, minimum approach z
DELIVERY_TABLE_TOP_Z_M = 0.767

# Five deterministic delivery slots.  The robot approaches from +Y and faces
# south, so more-negative Y is deeper on the table.  Three staggered inner
# slots are filled first, followed by two outer slots.  A literal five-item
# depth-only row would leave less than 90 mm between centres on this 440 mm
# deep table and would overlap the larger products.  All slots are shifted
# 50 mm south from the original layout so the fourth and fifth products are
# not released at the north edge of the tabletop.  A further 50 mm south shift
# pushed the deepest inner slot out of arm reach (IK no-solution), so the inner
# slots are reverted to the original depth while the outer slots stay shifted.
DELIVERY_PLACE_SLOTS_XY = (
    (-2.20, -3.50),  # 1: deepest, inner-left
    (-1.94, -3.48),  # 2: inner-centre
    (-1.68, -3.46),  # 3: inner-right
    (-2.07, -3.39),  # 4: outer-left
    (-1.81, -3.37),  # 5: outer-right / nearest
)
# 纸巾专用放置位：桌子最东侧。最深内排槽位会超出双臂前伸可达范围
# （IK 无解），故纸巾固定放东侧较浅处，机器人导航到对应 x 后直接下降。
TISSUE_DEDICATED_PLACE_XY = (-1.55, -3.30)
PLACE_SLOT_IK_NUDGE_M = 0.020
PLACE_SLOT_XY_TOLERANCE_M = 0.020

# Product centre heights above their supporting surface (half heights of the
# collision geometry).  The placement controller targets the product centre
# at table top + this value + a small clearance, rather than opening the
# gripper at one fixed TCP height for every product.
# 数值与 yolo_aruco_shelf_pick.py 保持一致（2026-08-17 合并 wxj v2 值）。
PRODUCT_HALF_HEIGHT_M = {
    "sanmingzhi": 0.0494,
    "heweidao": 0.00,
    "shupian": 0.0350,
    "zhijin": 0.0440,
    "maidong": 0.03550,
    "kele": 0.0325,
    "kouxiangtang": 0.0400,
    "pingguo": 0.0350,
    "chengzi": 0.0370,
}
# Heweidao uses a tapered 105 mm-tall collision mesh.  Keep the shared table
# above untouched because grasp/perception inherited its historical value;
# placement alone needs the physical half-height to put the product bottom on
# the delivery table and to validate contact there.
HEWEIDAO_PLACE_HALF_HEIGHT_M = 0.0525
# The original target commanded a few millimetres beyond the geometric table
# plane.  Raise only heweidao's release by 20 mm so its wide rim opens 14 mm
# above the tabletop; the existing contact detector remains a fallback if the
# product touches the table before reaching that target.
HEWEIDAO_PLACE_CONTACT_OVERTRAVEL_M = 0.006
HEWEIDAO_PLACE_RELEASE_RAISE_M = 0.020
HEWEIDAO_PLACE_DESCENT_SLIDE_STEP_M = 0.0004
HEWEIDAO_PLACE_DESCENT_TIMEOUT_S = 20.0
# Product bottom clearance above the delivery table top at release: the arm
# lowers until the held product bottom is 1 cm above the table, then opens the
# gripper.  The product drops the remaining 1 cm onto the table, then the arm
# raises vertically and the chassis backs away horizontally.
PLACE_PRODUCT_BOTTOM_CLEARANCE_M = 0.010
# Spheres can roll after even a short free fall.  Lower them to 3 mm above the
# measured tabletop while boxes retain the original 10 mm clearance.
PLACE_PRODUCT_BOTTOM_CLEARANCE_BY_KIND_M = {
    "heweidao": (
        -HEWEIDAO_PLACE_CONTACT_OVERTRAVEL_M
        + HEWEIDAO_PLACE_RELEASE_RAISE_M),
    "chengzi": 0.003,
    "pingguo": 0.003,
}
PLACE_APPROACH_CLEARANCE_M = 0.060
PLACE_DESCENT_SLIDE_STEP_M = 0.0015
PLACE_BASE_SETTLE_S = 0.20
PLACE_ARM_SETTLE_TOLERANCE_RAD = 0.025
PLACE_SLIDE_SETTLE_TOLERANCE_M = 0.004
# stage0 overhead approach 的 slide 收敛容差。半高表回退后 release_z 目标
# 比实际低约 6mm，slide 下降时商品底部先触桌被顶住，实测卡位误差 ≈6mm，
# 4mm 的全局收敛门永远过不了导致 approach 死等 30s（实测 kele 在桌子上方
# 停留 18s 等待收敛）。approach 只是把 slide 放到 overhead 高度，触桌卡位
# 6mm 不影响后续 stage1 水平精修（slide 不动）与 stage2 下降（触桌检测
# 依力矩饱和触发松爪），故单独放宽到 8mm，全局常量保持不变。
PLACE_APPROACH_SLIDE_TOLERANCE_M = 0.008
PLACE_XY_REFINE_STEP_M = 0.020
PLACE_XY_REFINE_SETTLE_S = 0.30
PLACE_XY_REFINE_TIMEOUT_S = 12.0
PLACE_XY_COMMAND_MIN_WAIT_S = 0.75
PLACE_XY_STATIONARY_SETTLE_S = 0.30
# 精调阶段的“臂静止”判据使用实测关节位置漂移而不是速度话题：负载关节
# 在力矩限位附近静止时，位置几乎不动，但速度话题仍会报 ±0.1 rad/s 的
# 噪声（实测 j6 位置 0.003 rad 内抖动时速度 ±0.12）。用速度阈值会把
# 已经静止的臂误判为运动中，导致水平精调空等满 12 s 超时后才兜底下降。
PLACE_XY_STATIONARY_ARM_POS_M = 0.006
PLACE_XY_STATIONARY_SLIDE_MPS = 0.005
# Under a held product the passive slide settles about 6 mm away from its
# command even though its velocity is effectively zero.  Keep the normal
# 4 mm command-ready gate above, but let the Cartesian refinement consume a
# genuinely stationary loaded pose inside this separate safety envelope.
PLACE_XY_STATIONARY_SLIDE_ERROR_M = 0.012
# During the loaded overhead approach a joint can stop a few hundredths of a
# radian short because the product/arm is already at its effort limit.  If the
# arm and slide have then been stationary for a short command interval, the
# measured pose is safe to hand to the existing XY refinement stage.  This is
# deliberately a separate, looser gate: it never authorises release and the
# normal slot/height checks still guard the subsequent descent.
# Keep this aligned with the existing vertical-descent arm gate.  The latest
# fifth delivery settled at 0.0489 rad on joint 5: it is mechanically static
# and therefore safe for measured-TCP refinement, but just outside 0.040.
PLACE_APPROACH_STATIONARY_ARM_ERROR_RAD = 0.050
PLACE_APPROACH_STATIONARY_ARM_RAD_S = 0.020
PLACE_APPROACH_STATIONARY_SLIDE_VEL_MPS = 0.005
PLACE_APPROACH_STATIONARY_MIN_AGE_S = 0.75
# A placement timeout must end by lowering onto the table, never by sweeping a
# clamped product back through the table edge.  This envelope is only a last
# resort; normal refinement still targets the 20 mm assigned-slot tolerance.
PLACE_XY_TIMEOUT_FALLBACK_TOLERANCE_M = 0.080
PLACE_DESCENT_TIMEOUT_S = 10.0
PLACE_VERTICAL_CLEARANCE_M = 0.070
PLACE_VERTICAL_CLEAR_TIMEOUT_S = 5.0
# Heweidao is wider at its top (95 mm) than the gripper's nominal maximum
# opening (80 mm).  Gravity release through the open fingers is therefore not
# reliable: after table contact, open the fingers, keep the arm fixed and back
# the chassis straight away from the table before any vertical lift.
HEWEIDAO_RELEASE_OPEN_MIN_S = 1.0
HEWEIDAO_RELEASE_OPEN_TIMEOUT_S = 3.0
HEWEIDAO_RELEASE_GRIP_OPEN_MIN = 0.85
HEWEIDAO_RELEASE_BASE_BACKUP_DISTANCE_M = 0.100
HEWEIDAO_RELEASE_BASE_BACKUP_SPEED_MPS = 0.10
HEWEIDAO_RELEASE_BASE_BACKUP_TIMEOUT_S = 5.0
# 下降接触检测：长商品/夹持偏低导致商品底部先触桌时，slide 被桌面顶住、
# 反馈不再跟随命令（实测卡死时 slide 力矩饱和 ≈ -306 N·m，正常运动时 ≈ 0）。
# 检测到"slide 力矩饱和 + 位置停滞"后停止下压并就地释放（商品底部已在桌面，
# 松爪即完成放置），避免 30 s 硬超时判失败。
PLACE_CONTACT_STALL_S = 0.5            # slide 误差无改善的观察窗口
PLACE_CONTACT_STALL_IMPROVEMENT_M = 0.002  # 窗口内误差改善 < 2 mm 视为停滞
PLACE_SLIDE_STALL_EFFORT_NM = 50.0     # slide 力矩超过此值视为被物理顶住
PLACE_SLIDE_STALL_VEL_MPS = 0.002      # slide 速度低于此值才算"没有在移动"
PLACE_STALL_CMD_MIN_AGE_S = 2.0        # 命令发出至少这么久才允许判停滞（防起步误判）
PLACE_CONTACT_BOTTOM_LOW_TOL_M = 0.005     # 商品底部允许低于桌面 5 mm
PLACE_CONTACT_BOTTOM_HIGH_TOL_M = 0.020    # 商品底部允许高于桌面 20 mm
# Spheres are deliberately released only 3 mm above the table.  Reusing the
# 20 mm box tolerance for an emergency/drop decision leaves enough free fall
# for an orange or apple to bounce and roll, so keep their low-release gate
# much closer to the surface without slowing the commanded descent.
PLACE_CONTACT_BOTTOM_HIGH_TOL_BY_KIND_M = {
    "chengzi": 0.008,
    "pingguo": 0.008,
}
PLACE_CLEAR_TABLE_MARGIN_M = 0.060
PLACE_CLEAR_TABLE_SPEED_MPS = 0.60
PLACE_CLEAR_TABLE_TIMEOUT_S = 15.0
# Keep the fast, obstacle-aware navigator active until its 0.10 m coarse
# tolerance.  The parent controller is retained only for the final few
# centimetres needed by perception/manipulation alignment.  The old 0.35 m
# hand-off made every shelf station spend roughly 15--20 s in low-speed trim.
NAV_TRANSIT_GATE_M = 0.10
NAV_PRECISE_HANDOFF_MARGIN_M = 0.02
# Final direct-slot precision.  The obstacle-aware navigator deliberately
# keeps its 0.10 m terminal envelope; the short shelf-local controller closes
# the remaining error to this tolerance without asking A* to resolve a pose
# more finely than its occupancy grid.
DIRECT_GRASP_POSITION_TOLERANCE_M = 0.025
NAV_LASER_STALE_S = 0.50           # fail safe if the 12 Hz scan stops
NAV_STATE_STALE_S = 0.50           # odom/joints must also remain live
FEEDBACK_LOSS_HARD_TIMEOUT_S = 10.0
NAV_PROGRESS_LOG_S = 3.0

# Reusable delivery trunk.  Shelf-specific and slot-specific motion remains
# live-planned on either side of the shared navigation anchors.
DELIVERY_TRUNK_REVERSE_START = (
    DELIVERY_TRUNK_EXIT[0], DELIVERY_TRUNK_EXIT[1], math.pi / 4.0)
DELIVERY_TRUNK_REVERSE_GOAL = (
    DELIVERY_TRUNK_ENTRY[0], DELIVERY_TRUNK_ENTRY[1], math.pi / 2.0)
DELIVERY_TRUNK_CACHE_START_TOLERANCE_M = 0.18
DELIVERY_TRUNK_CACHE_GOAL_TOLERANCE_M = 0.12
ROUTE_LEG_PROGRESS_M = 0.10
ROUTE_LEG_REPLAN_STALL_S = 20.0
ROUTE_LEG_REPLAN_MAX = 1
ROUTE_LEG_STALL_TIMEOUT_S = 35.0
# A leg stuck behind a dynamic box now fails via the widened stall check
# below (any persistent stop reason).  This hard timeout is only the final
# ceiling for slow-but-progressing legs; 150 s at the observed slowest real
# sim rate still covers every route leg in this arena.
ROUTE_LEG_HARD_TIMEOUT_S = 150.0

# 运行中"动态直达"（go_scan 途中改道到具体槽位）与订单开始直达一致的
# 置信度下限：拦截不可能成为矩阵主证据的病理记录（wxj v2 语义）。
DYNAMIC_DIRECT_CONF_MIN = 0.70

# Keep the held product clear of the shelf before delivery navigation starts
# turning the base.  The arms and product still protrude toward the shelf at
# the end of the parent grasp state machine.
BACKUP_SPEED_MPS = 0.30
BACKUP_TIMEOUT_S = 8.0
# Clear the shelf straight first, then fold a small part of the delivery turn
# into the remainder of the reverse.  The bounded angle keeps the loaded arm's
# swept volume close to the already-safe straight-withdrawal corridor; the
# normal obstacle-aware navigator still owns the rest of the turn and route.
BACKUP_TURN_CLEARANCE_M = 0.08
BACKUP_TURN_MAX_ANGLE_RAD = math.radians(12.0)
BACKUP_TURN_GAIN = 2.0
BACKUP_TURN_MAX_RPS = 0.30
# Heweidao can slide in the fingers during the first delivery turn while it is
# still settling after grasp.  Limit only that product and loaded interval;
# every other product and all later delivery/return turns retain the normal
# navigator limits.
POST_GRAB_SLOW_TURN_WATCHDOG_GRACE_S = 10.0
HEWEIDAO_LOADED_TURN_MAX_RPS = 0.55
# Product-specific loaded transit limits.  These are not blanket navigation
# slowdowns: they apply only after a successful grasp and before delivery.
# Kouxiangtang slipped in two independent runs exactly as the route requested
# 1.0--2.0 rad/s turns; limiting yaw removes that lateral impulse while the
# 0.75 m/s straight cap keeps the match-time cost modest.
LOADED_TRANSPORT_LIMITS = {
    "kouxiangtang": (0.75, 0.55),
    "heweidao": (None, HEWEIDAO_LOADED_TURN_MAX_RPS),
    "chengzi": (0.80, 0.55),
    "pingguo": (0.80, 0.55),
}
TRANSIT_SLIDE_TARGET_M = 0.006
TRANSIT_SLIDE_TOLERANCE_M = 0.010
TRANSIT_SLIDE_TIMEOUT_S = 8.0
TRANSIT_SLIDE_HARD_TIMEOUT_S = 12.0
TRANSIT_SLIDE_DEGRADED_MAX_ERROR_M = 0.050
# Gripper commands use 1.0=open and 0.0=fully closed.  Add holding preload
# only after the arm has withdrawn from the shelf, so capture stability/empty
# grasp checks remain unchanged.  This moves sandwich 0.16 -> 0.12; generic
# and dual grasps are already at the 0.0 limit.  Spheres use the gentler
# explicit 0.06 target below to avoid excessive squeeze.
TRANSPORT_GRIP_PRELOAD_COMMAND = 0.04
SPHERE_TRANSPORT_GRIP_COMMAND = 0.06
# The measured gripper is the only feedback that remains available after the
# shelf cameras no longer see the carried product.  Require a sustained empty
# signature before acting so one noisy JointState frame cannot fail an order.
TRANSPORT_DROP_MONITOR_GRACE_S = 0.50
TRANSPORT_DROP_CONFIRM_S = 0.30
TRANSPORT_DROP_RECOVERY_TIMEOUT_S = 4.0
TRANSPORT_DROP_FAILURE_SETTLE_S = 0.15

# A* stops outside the table's inflated costmap.  From that safe pose, make a
# short, yaw-controlled final approach before extending the arm.  Keep the
# speed well below normal navigation while avoiding an unnecessarily long
# low-speed hand-off.  The physical chassis front remains clear of the table
# at the nominal endpoint.
PLACE_CREEP_DISTANCE_M = 0.25
PLACE_CREEP_SPEED_MPS = 0.50
# The final 25 cm is driven while the product is extended over the tabletop.
# Reduce only that short segment for weak lateral and spherical grasps; the
# route to the table retains the limits above.
PLACE_CREEP_SPEED_BY_KIND_MPS = {
    "kouxiangtang": 0.25,
    "chengzi": 0.20,
    "pingguo": 0.20,
}
PLACE_CREEP_FRONT_STOP_M = 0.25
# Preserve the successful longitudinal arm reach measured on the deepest
# slot, but do not drive the same 0.25 m for outer slots that are substantially
# closer to the aisle.  The normal configured creep remains a hard upper cap.
PLACE_BASE_TO_SLOT_LONGITUDINAL_M = 0.62
PLACE_CREEP_GOAL_TOLERANCE_M = 0.01
PLACE_CREEP_YAW_GAIN = 2.0
PLACE_CREEP_MAX_ANGULAR_RPS = 0.30
# The overhead IK target can be solved against the deterministic end pose of
# the final creep.  Starting that loaded-arm motion while the base covers its
# last centimetres removes a serial wait without lowering the product during
# chassis motion.  The target is solved again against measured odometry after
# the base settles, so creep/lidar tolerances cannot shift the release slot.
PLACE_PARALLEL_SLIDE_ARM_ERROR_RAD = 0.20
PLACE_PARALLEL_SLIDE_MIN_BOTTOM_CLEARANCE_M = 0.12
# The parent limiter permits 0.022 rad every 20 ms (about 1.1 rad/s) and can
# apply that full step as soon as placement starts.  Ramp the loaded arm from
# rest and cap it at the gentler shelf-contact rate so the held product is not
# shocked loose.  Empty-arm recovery after release keeps the normal speed.
PLACE_LOADED_ARM_MAX_STEP_RAD = 0.006
PLACE_LOADED_ARM_STEP_RAMP_RAD = 0.00025
# Round products can remain secure for long chassis transit yet slip when a
# large multi-joint placement reconfiguration accelerates the fingers.  Give
# spheres a product-specific quarter-speed cap; correctness is worth the
# roughly 15--20 s controlled move and other products keep the faster rate.
PLACE_LOADED_ARM_MAX_STEP_BY_KIND_RAD = {
    "chengzi": 0.0030,
    "pingguo": 0.0030,
    # 纸巾双臂在精校准 XY 平移时，左右臂关节路径不对称会撕扯纸盒导致
    # 掉落；用半速上限压低每 tick 关节步长，减小双臂笛卡尔轨迹差异。
    "zhijin": 0.0030,
}
PLACE_LOADED_ARM_STEP_RAMP_BY_KIND_RAD = {
    "chengzi": 0.00010,
    "pingguo": 0.00010,
    "zhijin": 0.00010,
}
# The capture gate remains deliberately strict while lifting from the shelf.
# After that capture has been verified, a sphere can settle deeper between the
# fingers during a long route without being empty.  Reserve transport-drop
# recovery for feedback close to the empty-gripper position.
SPHERE_TRANSPORT_HELD_MINIMUM = {
    "chengzi": 0.30,
    "pingguo": 0.30,
}
PLACE_RELEASE_TABLE_MARGIN_M = 0.04
PLACE_APPROACH_HARD_TIMEOUT_S = 30.0
# 纸巾双臂 overhead 阶段若被顶住（slide stall 且接触几何无法验证），
# 原来的 30s 会在“slide stall rejected”循环里空等约 27s；缩短到 10s，
# 尽早触发“原地下降+释放”兜底（该兜底实测 ~3s 内完成），减少放置
# 后的长时间停顿。仅影响双臂（zhijin）放置，单臂仍用 30s。
DUAL_TISSUE_APPROACH_HARD_TIMEOUT_S = 10.0
PLACE_APPROACH_PROGRESS_LOG_S = 2.0
# 慢速仿真/高负载下载货摆臂可能超过旧的 15 s 上限但仍在正常收敛。硬超时
# 只作为最终上限：超过后若手臂误差仍在改善（窗口内改善 ≥ 阈值）就继续等，
# 只有"超时且长时间无改善"（真卡死）才判失败。
PLACE_APPROACH_PROGRESS_GATE_S = 6.0
PLACE_APPROACH_PROGRESS_IMPROVEMENT_RAD = 0.01
# 放置阶段逐关节运动诊断日志：基座/两臂六关节(measured/command/desired)/
# slide/夹爪/TCP/商品底部高度，每 PLACE_MOTION_LOG_PERIOD_S 一条
# [place-motion]，用于排查“商品掉落/被挤压到桌面”等放置问题。
PLACE_MOTION_LOG_PERIOD_S = 0.25
# The table-clear state already verifies the arms and chassis clearance.  One
# short zero-command interval is sufficient before worker shutdown; the old
# unconditional 3 s dwell accumulated once per order without adding safety.
FLOW_DONE_SETTLE_S = 0.25

# After the first delivered item, the runner may ask this worker to return to
# shelf A and record a complete stationary inventory view before it exits.
# Use every normal shelf view, but finish at the high overview posture so the
# next worker does not inherit a lowered camera assembly for long transit.
RETURN_WEST_SCAN_POSES = (
    pick.SCAN_CAMERA_POSES[1:] + pick.SCAN_CAMERA_POSES[:1])
RETURN_WEST_GOAL = (pick.SCAN_X[-1], pick.SCAN_Y, pick.YAW_NORTH)
RETURN_WEST_SCAN_POSE_TIMEOUT_S = 4.0
RETURN_WEST_RECOVERY_TIMEOUT_S = 4.0

# Neutral-posture recovery ceilings.  Every terminal path — successful
# placement, transport drop, grasp abort, fatal placement error — restores
# both arms, both grippers, slide and head to the initial pose before the
# worker exits.  The simulator resets only the base pose between workers, so
# without this a failed predecessor leaves the next worker with a closed
# gripper / deployed arm inherited through the joint feedback.
ABORT_RECOVERY_TIMEOUT_S = 8.0
FATAL_RECOVERY_TIMEOUT_S = 8.0

# Fixed initial whole-body manipulation posture.  Both arms and the head return
# here after every release; restoring only the selected arm or leaving the
# previous shelf-view pitch can leak a stale pose into the next order.
PLACE_RETREAT_ARM_L = [0.0, -0.166, 0.032, 0.0, 1.571, 2.223]
PLACE_RETREAT_ARM_R = [0.0, -0.166, 0.032, 0.0, -1.571, -2.223]
PLACE_RETREAT_HEAD_YAW_PITCH = [0.0, 0.0]
PLACE_RETREAT_HEAD_TOLERANCE_RAD = 0.05


class IntegratedNavPickPlace(pick.ShelfPickController):
    """Shelf-pick controller whose driving is done by the baseline navigator.

    Flow phases (``flow_phase``):
      "grab"            — parent state machine (GO_SCAN/SCAN/ALIGN/.../DONE);
                          its drive_to() is overridden to use the navigator for
                          long-range transit while keeping the precise final
                          alignment untouched.
      "backup"          — reverse with yaw hold to clear the shelf.
      "restore_height"  — restore the lift to its startup height.
      "nav_to_delivery" — navigator to DELIVERY_APPROACH with goods held.
      "place"           — extend, descend near the table, release, retreat.
      "return_to_west"  — after the first delivery, navigate back to shelf A.
      "return_west_scan" — hold at A and record all shelf camera views.
      "return_west_recover" — restore the neutral transit posture.
      "drop_success_recover" — low slot release detected; clear vertically.
      "drop_failed_recover" — premature product loss; recover before retry.
      "drop_failed"     — terminal failed attempt awaiting worker shutdown.
      "done"            — flow finished.
    """

    def __init__(
            self, target_kind: str, max_scan_cycles: int,
            tcp_diagnostic_ground_truth: bool, scan_skip_lower: bool,
            place_x: float = DELIVERY_TABLE_PLACE_WORLD[0],
            place_y: float = DELIVERY_TABLE_PLACE_WORLD[1],
            place_z: float = DELIVERY_TABLE_PLACE_WORLD[2],
            place_slot: int | None = None,
            place_release_dwell_s: float = 1.0,
            place_retreat_dwell_s: float = 0.5,
            nav_during_scan: bool = True,
            backup_after_grab_m: float = 0.20,
            place_creep_m: float = PLACE_CREEP_DISTANCE_M,
            close_recheck: bool = True,
            return_west_after_place: bool = False,
            return_start_after_place: bool = False):
        super().__init__(
            target_kind, max_scan_cycles,
            tcp_diagnostic_ground_truth, scan_skip_lower,
            close_recheck=close_recheck)

        # Wall-clock phase telemetry is intentionally observational only.  It
        # makes the next formal run actionable without changing any motion,
        # perception, or safety decision.
        self._pick_state_started_at = time.monotonic()
        self._pick_state_elapsed_s: dict[str, float] = {}
        self._flow_phase_started_at = time.monotonic()
        self._flow_phase_elapsed_s: dict[str, float] = {}
        self._flow_phase_distance_m: dict[str, float] = {}
        self._telemetry_last_base_xy = None

        self.nav_during_scan = nav_during_scan
        self.backup_after_grab_m = float(backup_after_grab_m)
        self.place_creep_m = float(place_creep_m)
        self.place_slot = place_slot
        self.place_world = np.array(
            [place_x, place_y, place_z], dtype=float)
        if target_kind == "zhijin":
            # 纸巾固定放东侧专用位，避免最深槽位双臂 IK 无解。
            self.place_world[0] = TISSUE_DEDICATED_PLACE_XY[0]
            self.place_world[1] = TISSUE_DEDICATED_PLACE_XY[1]
        self.place_min_approach_z = float(place_z)
        self.place_release_dwell_s = place_release_dwell_s
        self.place_retreat_dwell_s = place_retreat_dwell_s
        self.return_west_after_place = bool(return_west_after_place)
        self.return_start_after_place = bool(return_start_after_place)
        self.completion_file: str | None = None
        self.completion_order_id: str | None = None
        self._all_orders_completion_signalled = False
        self.placement_completed = False
        self.post_delivery_warnings: list[str] = []
        self.delivery_completed_by_drop = False
        self.drop_event = None
        self.terminal_error = None
        self._fatal_error = None
        self._fatal_match = False
        self._fatal_recovery_started_at = 0.0
        self._startup_posture_recovered = False

        # ── laser for the navigator ──
        self.laser_msg = None
        self.last_scan_time = None
        self.create_subscription(
            LaserScan, "/slamware_ros_sdk_server_node/scan",
            self._scan_cb, 10)
        self.perception_enable_pub = self.create_publisher(
            Bool, "/supermarket_sorting/perception_enable", 10)
        self.memory_consume_pub = self.create_publisher(
            String, MEMORY_CONSUME_TOPIC, 10)
        self.manage_external_perception = False
        self.local_perception_nodes = ()
        # Formal runners may keep one detector alive for the complete match so
        # the matrix continues updating during navigation, placement and
        # worker hand-off.  This flag turns every local disable request into
        # an enable request without changing detector ownership.
        self.perception_always_on = False
        self._perception_requested = None
        self._perception_request_last_at = float("-inf")

        # ── baseline navigator (same interface as the demo) ──
        self.nav = SupermarketNavigator()
        self.get_logger().info(
            "path_memory="
            + json.dumps(self.nav.path_memory_status(), ensure_ascii=False)
        )

        # ── our flow state ──
        self.flow_phase = "grab"
        self._post_grab_slow_turn_until = 0.0
        self._post_grab_slow_turn_logged = False
        self._nav_goal = None
        self._nav_last_log = 0.0
        self._last_nav_reason = None
        self._nav_memory_logged = False
        self._route_leg_name = None
        self._route_leg_goal = None
        self._route_leg_started_at = 0.0
        self._route_leg_last_progress_at = 0.0
        self._route_leg_best_distance = float("inf")
        self._route_leg_replans = 0
        self._route_leg_position_tolerance = None
        self._route_leg_yaw_tolerance = None
        self.delivery_nav_stage = None
        self.delivery_direct_fallback_used = False
        self.scan_trunk_route_stage = None
        self.scan_trunk_route_done = False
        self.scan_direct_fallback_used = False
        self.scan_route_final_goal = None
        self.place_stage = 0
        self.place_t0 = 0.0
        self.place_arm_joints = None
        self.place_approach_world = None
        self.place_slide_cmd = None
        self.place_release_world = None
        self.place_release_slide_cmd = None
        self.place_ik_ref_source = None
        self.place_ik_reference_joints = None
        self._place_ik_attempted = False
        self._place_arm_target_sent = False
        self._place_slide_target_sent = False
        self._place_creep_preposition_attempted = False
        self._place_creep_preposition_started = False
        self._place_creep_preposition_finalized = False
        self._place_base_settle_started_at = None
        self._place_base_reference_xy = None
        self._place_base_reference_yaw = None
        self._place_refine_started_at = None
        self._place_refine_target_sent = False
        self._place_refine_target_sent_at = None
        self._place_refine_motion_stable_since = None
        self._place_refine_motion_anchor = None
        self._place_refine_stable_since = None
        self._place_refine_iterations = 0
        self._place_release_started_at = None
        self._heweidao_release_phase = None
        self._heweidao_release_phase_started_at = 0.0
        self._heweidao_release_base_start_xy = None
        self._place_slide_stall_snapshot = None
        self._place_stall_warn_log = 0.0
        self._place_retreat_sent = False
        self._place_retreat_start_clearance = None
        self._dual_descent_sent = False
        self._dual_place_target_sent = False
        self._dual_release_spread_sent = False
        self.dual_release_slide_cmd = None
        self.place_creep_start_y = None
        self.place_creep_done = False
        self._place_stage0_wait_log = 0.0
        self._place_approach_best_error = float("inf")
        self._place_approach_best_error_at = 0.0
        self._place_motion_last_log = 0.0
        self._place_loaded_arm_step_rad = 0.0
        self._backup_start_xy = None
        self._backup_start_yaw = 0.0
        self._backup_turn_target_yaw = None
        self._backup_t0 = 0.0
        self._backup_logged = False
        self._height_restore_t0 = 0.0
        self._height_restore_monotonic_t0 = 0.0
        self._height_restore_timeout_logged = False
        self._transport_grip_command = None
        self._drop_monitor_armed_at = None
        self._drop_signature_since = None
        self._drop_candidate_reference_world = None
        self._drop_recovery_started_at = 0.0
        self._drop_recovery_vertical_clear = False
        self._drop_recovery_vertical_clear_started_at = 0.0
        self._flow_done_logged = False
        self._table_escape_logged = False
        self._table_escape_started_at = None
        self._laser_warn_log = 0.0
        self._state_warn_log = 0.0
        self._feedback_stale_since = None
        self._feedback_warn_monotonic = 0.0
        self._feedback_timeout_triggered = False
        self.return_scan_pose_index = 0
        self.return_scan_pose_started_at = 0.0
        self.return_scan_camera_ready_since = None
        self.return_recovery_started_at = 0.0

        # Runner-owned matrix routing is optional for standalone workers.  In
        # formal mode the runner passes one atomic JSON file shared across
        # process boundaries; this controller only reads it and publishes an
        # immediate consume event after the product leaves the shelf.
        self.memory_file = None
        self.memory_confidence_threshold = 0.90
        self.memory_active_hint = None
        self.memory_failed_hint = None
        self.memory_failed_hint_levels = set()
        self.memory_failed_hint_levels_by_kind = {}
        self.memory_exhausted_shelves = set()
        self.memory_exhausted_shelves_by_kind = {}
        self.excluded_slot_keys_by_kind = {}
        self.memory_last_scan_station_x = None
        self.memory_rerouted = False
        self.dynamic_direct_enabled = False
        self.direct_transit_slot = None
        self.direct_transit_started_at = None
        self.memory_reroute_not_before = time.time()
        self._memory_last_reroute_check = 0.0
        self._memory_consumed = False

        self.get_logger().info(
            "integrated nav+pick+place ready; "
            f"nav_during_scan={nav_during_scan} "
            f"close_recheck={int(close_recheck)} "
            f"place_slot={None if place_slot is None else place_slot + 1} "
            f"place_world={np.round(self.place_world, 3)} "
            f"backup_after_grab={self.backup_after_grab_m:.2f}m "
            f"place_creep={self.place_creep_m:.2f}m "
            f"release_dwell={place_release_dwell_s}s "
            f"retreat_dwell={place_retreat_dwell_s}s "
            f"return_west_after_place="
            f"{int(self.return_west_after_place)}")

    def set_state(self, new_state: str) -> None:
        """Accumulate parent pick-state wall time without altering its FSM."""
        previous = getattr(self, "state", None)
        started = getattr(self, "_pick_state_started_at", None)
        now = time.monotonic()
        if previous is not None and started is not None and previous != new_state:
            self._pick_state_elapsed_s[previous] = (
                self._pick_state_elapsed_s.get(previous, 0.0)
                + max(0.0, now - started))
        super().set_state(new_state)
        if previous != self.state and started is not None:
            self._pick_state_started_at = now

    def _heweidao_loaded_turn_limit_active(self) -> bool:
        """Whether heweidao is still being carried towards delivery."""
        return (
            getattr(self, "target_kind", None) == "heweidao"
            and getattr(self, "flow_phase", None) in {
                "backup", "restore_height", "nav_to_delivery"
            })

    def _post_grab_slow_turn_active(self) -> bool:
        """Whether the initial slow-turn watchdog grace is still active."""
        until = float(getattr(self, "_post_grab_slow_turn_until", 0.0))
        return (
            until > 0.0
            and self._heweidao_loaded_turn_limit_active()
            and self.now() < until)

    def _loaded_transport_limits(
            self) -> tuple[float | None, float | None]:
        """Return product-specific velocity caps while carrying to table."""
        if getattr(self, "flow_phase", None) not in {
                "backup", "restore_height", "nav_to_delivery"}:
            return None, None
        return LOADED_TRANSPORT_LIMITS.get(
            getattr(self, "target_kind", None), (None, None))

    def set_twist(self, linear: float, angular: float) -> None:
        """Apply normal limits plus product-specific loaded transit caps."""
        requested_linear = float(linear)
        requested_angular = float(angular)
        linear_cap, angular_cap = self._loaded_transport_limits()
        if linear_cap is not None:
            linear = float(np.clip(
                requested_linear, -linear_cap, linear_cap))
        if angular_cap is not None:
            angular = float(np.clip(
                requested_angular, -angular_cap, angular_cap))
        if ((abs(float(linear) - requested_linear) > 1e-9
             or abs(float(angular) - requested_angular) > 1e-9)
                and not getattr(self, "_loaded_motion_limit_logged", False)):
            self._loaded_motion_limit_logged = True
            self.get_logger().info(
                "[loaded-motion] limiting carried product motion "
                f"kind={self.target_kind} "
                f"v={requested_linear:.2f}->{float(linear):.2f}m/s "
                f"w={requested_angular:.2f}->{float(angular):.2f}rad/s")
        if self._post_grab_slow_turn_active():
            angular = float(np.clip(
                requested_angular,
                -HEWEIDAO_LOADED_TURN_MAX_RPS,
                HEWEIDAO_LOADED_TURN_MAX_RPS))
            if (abs(angular - requested_angular) > 1e-9
                    and not getattr(
                        self, "_post_grab_slow_turn_logged", False)):
                self._post_grab_slow_turn_logged = True
                self.get_logger().info(
                    "[loaded-turn] limiting heweidao delivery turn "
                    f"w={requested_angular:.2f}->{angular:.2f}rad/s "
                    "until delivery arrival")
        super().set_twist(linear, angular)

    def _set_flow_phase(self, new_phase: str) -> None:
        """Change the outer phase and account for elapsed wall-clock time."""
        if new_phase == self.flow_phase:
            return
        now = time.monotonic()
        previous = self.flow_phase
        self._flow_phase_elapsed_s[previous] = (
            self._flow_phase_elapsed_s.get(previous, 0.0)
            + max(0.0, now - self._flow_phase_started_at))
        self.flow_phase = new_phase
        self._flow_phase_started_at = now

    def timing_snapshot(self) -> dict[str, dict[str, float]]:
        """Return accumulated timings including the currently active states."""
        now = time.monotonic()
        pick_states = dict(self._pick_state_elapsed_s)
        pick_states[self.state] = (
            pick_states.get(self.state, 0.0)
            + max(0.0, now - self._pick_state_started_at))
        flow_phases = dict(self._flow_phase_elapsed_s)
        flow_phases[self.flow_phase] = (
            flow_phases.get(self.flow_phase, 0.0)
            + max(0.0, now - self._flow_phase_started_at))
        return {
            "pick_state_elapsed_s": {
                key: round(value, 3)
                for key, value in sorted(pick_states.items())
            },
            "flow_phase_elapsed_s": {
                key: round(value, 3)
                for key, value in sorted(flow_phases.items())
            },
            "flow_phase_distance_m": {
                key: round(value, 3)
                for key, value in sorted(
                    self._flow_phase_distance_m.items())
            },
        }

    def _record_motion_telemetry(self) -> None:
        """Accumulate measured travel per phase without affecting control."""
        current = np.asarray(self.base_xy, dtype=float).copy()
        previous = self._telemetry_last_base_xy
        self._telemetry_last_base_xy = current
        if previous is None:
            return
        distance = float(np.linalg.norm(current - previous))
        # Ignore an odometry reset/teleport across server restarts; normal
        # 50 Hz motion is orders of magnitude below this threshold.
        if not math.isfinite(distance) or distance > 0.50:
            return
        self._flow_phase_distance_m[self.flow_phase] = (
            self._flow_phase_distance_m.get(self.flow_phase, 0.0)
            + distance)

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _scan_cb(self, msg) -> None:
        self.laser_msg = msg
        self.last_scan_time = self.now()

    def _laser_stale(self, now: float) -> bool:
        return (
            self.last_scan_time is None
            or now - self.last_scan_time > NAV_LASER_STALE_S)

    def configure_external_perception(self, enabled: bool) -> None:
        """Let this worker gate the shared detector around scan states."""
        self.manage_external_perception = bool(enabled)
        self._perception_requested = None
        self._perception_request_last_at = float("-inf")
        self._publish_perception_request(False, force=True)

    def configure_local_perception(self, *nodes) -> None:
        """Apply the same duty cycle when persistent perception is absent."""
        self.local_perception_nodes = tuple(nodes)
        self._perception_requested = None
        self._perception_request_last_at = float("-inf")
        self._publish_perception_request(False, force=True)

    def configure_memory_routing(
            self, path: str | None, confidence_threshold: float,
            initial_x: float | None = None,
            initial_z: float | None = None) -> None:
        """Read runner-owned matrix hints without sharing process state."""
        self.memory_file = (
            None if not path else pathlib.Path(path).expanduser().resolve())
        self.memory_confidence_threshold = float(confidence_threshold)
        self.memory_reroute_not_before = time.time()
        if initial_x is not None:
            shelf = shelf_for_scan_x(initial_x)
            level = None
            if initial_z is not None:
                level = min(
                    LEVEL_MARKER_Z,
                    key=lambda name: abs(
                        LEVEL_MARKER_Z[name] - float(initial_z)))
            if level is not None:
                self.memory_active_hint = (shelf, level)
            self.memory_last_scan_station_x = float(initial_x)
        self.get_logger().info(
            f"[memory] routing file={self.memory_file} "
            f"threshold={self.memory_confidence_threshold:.2f} "
            f"initial_hint={self.memory_active_hint}")

    def _select_live_memory_hint(
            self, *, reliable_only: bool,
            min_last_seen: float | None = None,
            require_direct: bool = False):
        if self.memory_file is None:
            return None
        document = read_memory_document(self.memory_file)
        # Rank every still-pending class carried by this worker, not merely
        # the class chosen when the process was dispatched.  This lets a live
        # matrix update replace a stale target with the nearest pending item.
        kinds = tuple(dict.fromkeys(getattr(
            self, "opportunistic_target_kinds", (self.target_kind,))))
        failed_by_kind = getattr(
            self, "memory_failed_hint_levels_by_kind", {})
        exhausted_by_kind = getattr(
            self, "memory_exhausted_shelves_by_kind", None)
        ranked = []
        for priority, kind in enumerate(kinds):
            failed_levels = set(failed_by_kind.get(kind, set()))
            # Backward compatibility for standalone/tests created before the
            # per-kind failure map existed.
            if kind == self.target_kind:
                failed_levels.update(self.memory_failed_hint_levels)
            if exhausted_by_kind is None:
                exhausted_shelves = (
                    self.memory_exhausted_shelves
                    if kind == self.target_kind else set())
            else:
                exhausted_shelves = set(
                    exhausted_by_kind.get(kind, set()))
            hint = select_memory_route_hint(
                kind,
                primary_candidates_from_document(document, kind),
                candidates_from_document(document, kind),
                self.base_xy,
                self.memory_confidence_threshold,
                exclude_slots=(
                    set(self.excluded_slot_keys)
                    | set(self.excluded_slot_keys_by_kind.get(kind, set()))),
                exclude_shelves=exhausted_shelves,
                exclude_shelf_levels=failed_levels,
                min_last_seen=min_last_seen,
                reliable_only=reliable_only,
                require_direct=require_direct)
            if hint is None:
                continue
            candidate = dict(hint)
            candidate["target_kind"] = kind
            try:
                travel = float(candidate.get("travel", float("inf")))
                confidence = float(candidate.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(travel):
                continue
            ranked.append(((travel, -confidence, priority), candidate))
        return min(ranked, key=lambda item: item[0])[1] if ranked else None

    def _activate_memory_target_kind(self, hint: dict, reason: str) -> str:
        """Switch to the pending class selected by a matrix route hint."""
        kind = str(hint.get("target_kind", self.target_kind))
        if kind == self.target_kind:
            return self.target_kind
        previous = self.target_kind
        self._set_pregrasp_target_kind(kind)
        self.get_logger().info(
            f"[order-select] matrix nearest switched kind={previous}->{kind} "
            f"reason={reason} travel={float(hint['travel']):.2f}m")
        return previous

    def _direct_retry_slot_supported(
            self, shelf: str, level: str, column: str) -> bool:
        """Only retry columns currently backed by stable target evidence."""
        if self.memory_file is None:
            return False
        document = read_memory_document(self.memory_file)
        wanted_key = f"{level}|{shelf}|{column}"
        for candidate in primary_candidates_from_document(
                document, self.target_kind):
            if (str(candidate.get("slot_key", "")) == wanted_key
                    and memory_direct_candidate_allowed(candidate)):
                return True
        self.get_logger().info(
            f"[direct-slot-retry] skip unsupported matrix slot "
            f"{shelf}-{level}-{column} kind={self.target_kind}")
        return False

    def configure_direct_slot_target(
            self, shelf: str, level: str, column: str,
            marker_id: int | None = None,
            product_y: float | None = None,
            product_z: float | None = None) -> bool:
        """Resolve a remembered slot, then navigate before precision align.

        ``ShelfPickController.configure_direct_slot_target`` historically
        entered ``STATE_ALIGN`` immediately.  That made ALIGN's 45 second
        precision watchdog include the entire cross-store transit.  Keep the
        resolved grasp geometry, but defer ALIGN until the obstacle-aware
        navigator reaches the shelf-side handoff pose.
        """
        accepted = super().configure_direct_slot_target(
            shelf, level, column,
            marker_id=marker_id,
            product_y=product_y,
            product_z=product_z,
            defer_align=True)
        if not accepted:
            return False

        self.direct_transit_slot = (
            str(shelf).upper(), str(level).upper(), str(column)[-1:])
        self.direct_transit_started_at = time.monotonic()
        self.scan_trunk_route_stage = None
        self.scan_trunk_route_done = False
        self.scan_route_final_goal = None
        self._route_leg_name = None
        self._route_leg_goal = None
        self._nav_goal = None
        self.set_state(pick.STATE_DIRECT_TRANSIT)
        self.get_logger().info(
            "[direct-transit] queued slot="
            f"{self.direct_transit_slot} handoff="
            f"({self.align_base_x:.3f},{self.align_base_y:.3f},"
            f"{math.degrees(pick.YAW_NORTH):.0f}deg)")
        return True

    def advance_direct_transit(self) -> None:
        """直达记忆槽位抓取位，最终停稳后立即复核/抓取。

        长距离段由障碍导航收敛到稳定的 0.10m 货架包络，最后几厘米在
        同一个 DIRECT_TRANSIT 状态内交给货架局部控制器完成。这样不会
        把 2.5cm 精度强塞给栅格规划器，也不会重新进入一次 ALIGN 状态；
        close-recheck 关闭时仍然是到最终站位、停稳、直接抓取。
        """
        if self.target_world is None or self.direct_transit_slot is None:
            raise RuntimeError(
                "direct transit entered without a resolved slot target")

        arrived = self.drive_to(
            [self.align_base_x, self.align_base_y],
            pick.YAW_NORTH, DIRECT_GRASP_POSITION_TOLERANCE_M,
            linear_min_mps=pick.NAV_ALIGN_LINEAR_MIN_MPS,
            linear_gain=pick.NAV_ALIGN_LINEAR_GAIN,
            rotate_gate_rad=pick.NAV_ALIGN_ROTATE_GATE_RAD,
            translate_angular_max_rps=(
                pick.NAV_ALIGN_TRANSLATE_ANGULAR_MAX_RADPS))
        if not arrived:
            return

        elapsed = (
            0.0 if self.direct_transit_started_at is None
            else max(0.0, time.monotonic() - self.direct_transit_started_at))
        position_error = float(np.linalg.norm(
            self.base_xy - np.array([
                self.align_base_x, self.align_base_y], dtype=float)))
        yaw_error = abs(pick.wrap_to_pi(pick.YAW_NORTH - self.base_yaw))
        slot = self.direct_transit_slot
        self.direct_transit_slot = None
        self.direct_transit_started_at = None
        self.set_twist(0.0, 0.0)
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        if self.close_recheck and not self._recheck_passed:
            self.set_state(pick.STATE_RECHECK)
            self._start_close_recheck()
            self.get_logger().info(
                f"[direct-transit] arrived slot={slot} after {elapsed:.1f}s; "
                f"pose_error={position_error:.3f}m/"
                f"{yaw_error:.3f}rad; "
                "stop-and-grasp handoff (close-recheck enabled)")
        else:
            self._start_grasp_settle()
            self.get_logger().info(
                f"[direct-transit] arrived slot={slot} after {elapsed:.1f}s; "
                f"pose_error={position_error:.3f}m/"
                f"{yaw_error:.3f}rad; "
                "stop-and-grasp handoff (no recheck)")

    def _try_apply_direct_memory_hint(
            self, hint: dict, reason: str) -> bool:
        hint_x = float(hint["x"])
        column = str(hint.get("column", ""))
        confidence = float(hint.get("confidence", 0.0) or 0.0)
        # Snapshot 直达门槛：隐藏历史候选不直达；纸巾三列均由双臂流程
        # 支持；动态直达还需持续证据与置信度门槛。close-recheck 由调用方按需开启
        # （关闭时不再作为直达的强制前置条件，抓取直接进入 ALIGN→grasp）。
        if (not hint.get("hidden_fallback")
                and column in {"1", "2", "3"}
                and memory_direct_candidate_allowed(hint)
                and confidence >= DYNAMIC_DIRECT_CONF_MIN):
            previous_kind = self._activate_memory_target_kind(hint, reason)
            accepted = self.configure_direct_slot_target(
                str(hint["shelf"]), str(hint["level"]), column,
                product_y=hint.get("world_y"),
                product_z=hint.get("world_z"))
            if not accepted:
                if previous_kind != self.target_kind:
                    self._set_pregrasp_target_kind(previous_kind)
                return False
            self.scan_preferred_x = hint_x
            self.memory_active_hint = (
                str(hint["shelf"]), str(hint["level"]))
            self.memory_last_scan_station_x = hint_x
            self.get_logger().info(
                f"[memory] {reason} -> direct slot "
                f"{hint['shelf']}-{hint['level']}-{column} "
                f"x={hint_x:.3f} conf={float(hint['confidence']):.3f}")
            return True
        return False

    def _apply_memory_hint(self, hint: dict, reason: str) -> str:
        if self._try_apply_direct_memory_hint(hint, reason):
            return "direct"
        self._activate_memory_target_kind(hint, reason)
        hint_x = float(hint["x"])
        hint_z = float(hint["z"])
        self.configure_inventory_scan_hint(hint_x, hint_z)
        self.memory_active_hint = (
            str(hint["shelf"]), str(hint["level"]))
        self.scan_station_order = None
        self.scan_index = 0
        self.scan_pose_index = 0
        self.scan_camera_ready_since = None
        self.memory_last_scan_station_x = hint_x
        self.get_logger().info(
            f"[memory] {reason} -> {self.memory_active_hint} "
            f"x={hint_x:.3f} travel={float(hint['travel']):.2f}m "
            f"observed={float(hint['observed_distance']):.2f}m "
            f"conf={float(hint['confidence']):.3f}")
        return "scan"

    def _update_memory_scan_progress(self) -> None:
        if self.scan_station_order is None:
            return
        current_x = float(self.current_scan_station_x())
        previous_x = self.memory_last_scan_station_x
        if previous_x is None:
            self.memory_last_scan_station_x = current_x
            return
        if abs(current_x - float(previous_x)) <= 0.40:
            return
        shelf = shelf_for_scan_x(float(previous_x))
        self.memory_exhausted_shelves.add(shelf)
        self.memory_exhausted_shelves_by_kind.setdefault(
            self.target_kind, set()).add(shelf)
        self.memory_last_scan_station_x = current_x
        self.get_logger().info(
            f"[memory] shelf {shelf} fully scanned without localisation; "
            "suppressing dynamic revisit for this order")

    def _memory_route_tick(self) -> None:
        if self.memory_file is None:
            return
        if self.state == pick.STATE_DIRECT_TRANSIT:
            self._memory_direct_transit_conflict_tick()
            return
        if self.state != pick.STATE_GO_SCAN:
            return
        if self.target_world is not None:
            return

        if self.memory_failed_hint is not None:
            failed_kind, failed_hint = self.memory_failed_hint
            self.memory_failed_hint = None
            self.memory_failed_hint_levels_by_kind.setdefault(
                failed_kind, set()).add(failed_hint)
            if failed_kind == self.target_kind:
                self.memory_failed_hint_levels.add(failed_hint)
            next_hint = self._select_live_memory_hint(reliable_only=True)
            if next_hint is not None:
                self._apply_memory_hint(
                    next_hint,
                    f"hint {failed_kind}:{failed_hint} failed; nearest "
                    "pending failover")
                return
            self.get_logger().info(
                f"[memory] hint {failed_hint} failed; no reliable matrix "
                "candidate remains, resuming fallback scan")

        self._update_memory_scan_progress()
        now = time.monotonic()
        if now - self._memory_last_reroute_check < 0.25:
            return
        self._memory_last_reroute_check = now
        # ``--dynamic-direct`` may change an in-flight goal only when the
        # candidate already satisfies the same persistent-evidence contract
        # as an order-start direct slot.  A weak/far first observation is a
        # useful matrix hint, but must not redirect the route or turn a single
        # trip into E -> guessed shelf -> real shelf.
        refreshed = self._select_live_memory_hint(
            reliable_only=True,
            min_last_seen=self.memory_reroute_not_before,
            require_direct=self.dynamic_direct_enabled)
        if refreshed is None:
            return
        current_x = (
            float(self.scan_preferred_x)
            if self.scan_station_order is None
            and self.scan_preferred_x is not None
            else float(self.current_scan_station_x()))
        if self.dynamic_direct_enabled:
            if self._try_apply_direct_memory_hint(
                    refreshed, "dynamic reroute"):
                self.memory_rerouted = True
                self.memory_reroute_not_before = time.time()
            # Dynamic-direct is intentionally direct-only.  If a candidate
            # cannot become a concrete slot target, keep the current route;
            # do not silently degrade it into a shelf-level scan reroute.
            return

        hint_shelf = str(refreshed.get("shelf", ""))
        current_shelf = shelf_for_scan_x(current_x)
        hint_kind = str(refreshed.get("target_kind", self.target_kind))
        exhausted_for_hint = getattr(
            self, "memory_exhausted_shelves_by_kind", {}).get(
                hint_kind, set())
        if (not hint_shelf
                or hint_shelf == current_shelf
                or hint_shelf in exhausted_for_hint
                or abs(float(refreshed["x"]) - current_x) <= 0.40):
            return
        if not self.dynamic_direct_enabled:
            current_travel = math.hypot(
                float(self.base_xy[0]) - current_x,
                float(self.base_xy[1]) - pick.SCAN_Y)
            new_travel = float(refreshed["travel"])
            if new_travel + MEMORY_REROUTE_SAVING_M >= current_travel:
                return
        self._apply_memory_hint(
            refreshed,
            f"dynamic reroute {current_x:.3f}->{float(refreshed['x']):.3f}")
        self.memory_rerouted = True
        self.memory_reroute_not_before = time.time()

    def _memory_direct_transit_conflict_tick(self) -> None:
        """Drop a direct target when the matrix reliably replaces its class."""
        now = time.monotonic()
        if now - self._memory_last_reroute_check < 0.25:
            return
        self._memory_last_reroute_check = now
        slot_key = self.target_slot_key()
        if slot_key is None:
            return
        document = read_memory_document(self.memory_file)
        active_still_primary = any(
            str(candidate.get("slot_key", "")) == slot_key
            for candidate in primary_candidates_from_document(
                document, self.target_kind))
        if active_still_primary:
            return
        replacement_kind = None
        for kind in self.opportunistic_target_kinds:
            if kind == self.target_kind:
                continue
            if any(
                    str(candidate.get("slot_key", "")) == slot_key
                    and memory_direct_candidate_allowed(candidate)
                    for candidate in primary_candidates_from_document(
                        document, kind)):
                replacement_kind = kind
                break
        # Mere disappearance is not enough to abandon a route; a stable,
        # different pending class at the same slot is explicit stale-memory
        # evidence and is safe to act on while still in transit.
        if replacement_kind is None:
            return

        stale_kind = self.target_kind
        self.excluded_slot_keys_by_kind.setdefault(
            stale_kind, set()).add(slot_key)
        self.target_marker_id = None
        self.target_physical_marker_id = None
        self.target_world = None
        self.target_localisation_source = None
        self.committed_slot = None
        self.direct_slot_target_active = False
        self.direct_transit_slot = None
        self.direct_transit_started_at = None
        self.memory_active_hint = None
        self.memory_rerouted = False
        self.set_state(pick.STATE_GO_SCAN)
        next_hint = self._select_live_memory_hint(
            reliable_only=True, require_direct=True)
        self.get_logger().warn(
            f"[order-select] in-transit matrix conflict slot={slot_key} "
            f"expected={stale_kind} observed={replacement_kind}; dropping "
            "stale target and selecting nearest pending item")
        if next_hint is not None:
            self._try_apply_direct_memory_hint(
                next_hint, "in-transit stale-memory failover")

    def _restore_full_scan_after_inventory_hint(self) -> None:
        """Expose a failed matrix shelf/level to immediate route failover."""
        was_active = self.inventory_scan_hint_active
        failed_hint = self.memory_active_hint
        super()._restore_full_scan_after_inventory_hint()
        if was_active and failed_hint is not None:
            self.memory_failed_hint = (self.target_kind, failed_hint)
            self.memory_active_hint = None

    def _recheck_fail(self) -> None:
        """After a stale direct slot, choose another nearest pending item."""
        missed_kind = self.target_kind
        missed_slot = self.target_slot_key()
        dynamic_was_enabled = bool(self.dynamic_direct_enabled)
        super()._recheck_fail()
        # A valid YOLO/depth fallback proceeds directly to grasp and must not
        # be disturbed merely because optional ArUco was inconclusive.
        if self.target_world is not None or self.state != pick.STATE_GO_SCAN:
            return
        if missed_slot is not None:
            self.excluded_slot_keys_by_kind.setdefault(
                missed_kind, set()).add(missed_slot)
        self.dynamic_direct_enabled = dynamic_was_enabled
        self.memory_rerouted = False
        self.memory_reroute_not_before = time.time()
        self._memory_last_reroute_check = 0.0
        self.get_logger().warn(
            f"[order-select] stale target dropped kind={missed_kind} "
            f"slot={missed_slot}; selecting nearest remaining pending item")

    def _publish_perception_request(
            self, enabled: bool, force: bool = False) -> None:
        if (not self.manage_external_perception
                and not self.local_perception_nodes):
            return
        enabled = bool(enabled)
        if self.perception_always_on:
            enabled = True
        now = self.now()
        if (not force
                and enabled == self._perception_requested
                and now - self._perception_request_last_at < 0.5):
            return
        if self.manage_external_perception:
            self.perception_enable_pub.publish(Bool(data=enabled))
        for node in self.local_perception_nodes:
            node.set_enabled(enabled)
        self._perception_requested = enabled
        self._perception_request_last_at = now

    def initialize_commands(self) -> None:
        """Initialize commands while keeping a fixed post-grasp transit height."""
        super().initialize_commands()
        measured_slide = self.joints.get("slide_joint")
        self.get_logger().info(
            f"[flow] configured post-grasp transit slide="
            f"{TRANSIT_SLIDE_TARGET_M:.3f} "
            f"(initial measured={measured_slide})")

    @staticmethod
    def _transit_slide_target() -> float:
        return float(TRANSIT_SLIDE_TARGET_M)

    def _start_route_leg(
            self, name: str, goal, *, use_memory: bool,
            lock_cached_path: bool = False,
            position_tolerance: float | None = None,
            yaw_tolerance: float | None = None) -> None:
        """Start one explicit leg of a composite shelf/delivery route."""
        goal = tuple(float(value) for value in goal)
        now = self.now()
        self._route_leg_name = str(name)
        self._route_leg_goal = goal
        self._route_leg_started_at = now
        self._route_leg_last_progress_at = now
        self._route_leg_best_distance = float(np.linalg.norm(
            np.asarray(goal[:2], dtype=float) - self.base_xy))
        self._route_leg_replans = 0
        self._route_leg_position_tolerance = position_tolerance
        self._route_leg_yaw_tolerance = yaw_tolerance
        self._nav_last_log = 0.0
        self._last_nav_reason = None
        self._nav_memory_logged = False
        self._nav_goal = None
        self.nav.set_goal(
            *goal,
            cached_start_offset_limit=(
                DELIVERY_TRUNK_CACHE_START_TOLERANCE_M
                if use_memory else None),
            cached_goal_offset_limit=(
                DELIVERY_TRUNK_CACHE_GOAL_TOLERANCE_M
                if use_memory else None),
            use_path_memory=use_memory,
            lock_cached_path=lock_cached_path,
            position_tolerance=position_tolerance,
            yaw_tolerance=yaw_tolerance)
        self.get_logger().info(
            f"[route] start leg={name} goal="
            f"({goal[0]:.2f},{goal[1]:.2f},"
            f"{math.degrees(goal[2]):.0f}deg) "
            f"memory={int(use_memory)} lock={int(lock_cached_path)} "
            f"pos_tol={self.nav.controller.pos_tol:.3f} "
            f"yaw_tol={self.nav.controller.yaw_tol:.3f}")

    def _route_leg_tick(self) -> tuple[bool, str | None]:
        """Drive the active route leg and report a terminal failure reason."""
        now = self.now()
        if self._route_leg_goal is None or self._route_leg_name is None:
            return False, "route_leg_not_configured"
        if self._laser_stale(now):
            self.set_twist(0.0, 0.0)
            if now - self._laser_warn_log > 1.0:
                self.get_logger().warn(
                    f"[route:{self._route_leg_name}] waiting for fresh laser")
                self._laser_warn_log = now
            return False, None

        v, w, reached = self.nav.update(
            self.base_xy[0], self.base_xy[1], self.base_yaw,
            laser_msg=self.laser_msg, time_now=now)
        self.set_twist(v, w)
        if not self._nav_memory_logged:
            self._nav_memory_logged = True
            self.get_logger().info(
                f"[route:{self._route_leg_name}] path_memory="
                + json.dumps(
                    self.nav.path_memory_status(), ensure_ascii=False))

        ctrl = self.nav.controller
        stop_reason = ctrl.stop_reason
        if (stop_reason is not None
                and stop_reason != self._last_nav_reason):
            self._last_nav_reason = stop_reason
            self.get_logger().info(
                f"[route:{self._route_leg_name}] stop_reason={stop_reason} "
                f"lidar={ctrl.lidar_clearance:.2f}m "
                f"rear={ctrl.rear_clearance:.2f}m")

        distance = float(np.linalg.norm(
            np.asarray(self._route_leg_goal[:2], dtype=float) - self.base_xy))
        if distance + ROUTE_LEG_PROGRESS_M < self._route_leg_best_distance:
            self._route_leg_best_distance = distance
            self._route_leg_last_progress_at = now
        elapsed = now - self._route_leg_started_at
        stalled = now - self._route_leg_last_progress_at

        if now - self._nav_last_log >= NAV_PROGRESS_LOG_S:
            self._nav_last_log = now
            self.get_logger().info(
                f"[route:{self._route_leg_name}] "
                f"pos=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
                f"yaw={math.degrees(self.base_yaw):.0f}deg "
                f"dist={distance:.2f}m v={v:.2f} w={w:.2f} "
                f"elapsed={elapsed:.1f}s stalled={stalled:.1f}s "
                f"reached={reached}")

        if reached:
            self.set_twist(0.0, 0.0)
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0
            return True, None

        no_path = isinstance(stop_reason, str) and (
            stop_reason.startswith("no_path")
            or stop_reason.startswith("stuck_no_path"))
        recovery_exhausted = self.nav.recovery_exhausted()
        # A loaded product can legitimately need more than 20 s to complete a
        # large initial heading change because set_twist() caps its angular
        # speed.  The goal distance is constant during that manoeuvre, but the
        # robot is not stalled.  Keep the replan budget for an actual obstacle
        # stop later in the route.
        active_heading_alignment = (
            (stop_reason == "heading_alignment" and abs(float(w)) >= 0.05)
            or stop_reason == "rotate_recovery")
        if active_heading_alignment:
            self._route_leg_last_progress_at = now
            stalled = 0.0
        # A leg can stall behind a dynamic box with stop_reason
        # lidar_stop/arc_blocked/rotation_loop while the planner still finds
        # a path (so no_path is False and recovery may not be exhausted).
        # Treat any persistent stop reason the same as no_path: after
        # ROUTE_LEG_STALL_TIMEOUT_S without 0.10 m of progress the leg fails
        # and the caller can fall back instead of waiting out a 150 s ceiling.
        persistent_stop = (
            not active_heading_alignment
            and (no_path or recovery_exhausted or stop_reason is not None))
        if (stalled >= ROUTE_LEG_REPLAN_STALL_S
                and persistent_stop
                and self._route_leg_replans < ROUTE_LEG_REPLAN_MAX):
            self._route_leg_replans += 1
            self.nav.invalidate_active_cached_path(
                f"route_leg:{self._route_leg_name}:live_replan", now=now)
            # Replan from the measured pose against the current lidar map.
            # Do not reload the same cached route that led into the stop.
            self.nav.set_goal(
                *self._route_leg_goal,
                use_path_memory=False,
                lock_cached_path=False,
                position_tolerance=self._route_leg_position_tolerance,
                yaw_tolerance=self._route_leg_yaw_tolerance)
            self._route_leg_last_progress_at = now
            self._route_leg_best_distance = distance
            self._nav_memory_logged = False
            self._last_nav_reason = None
            self.get_logger().warn(
                f"[route:{self._route_leg_name}] no progress for "
                f"{stalled:.1f}s with stop_reason={stop_reason}; "
                f"forcing live replan {self._route_leg_replans}/"
                f"{ROUTE_LEG_REPLAN_MAX}")
            return False, None
        failure = None
        if elapsed >= ROUTE_LEG_HARD_TIMEOUT_S:
            failure = (
                f"hard_timeout:{elapsed:.1f}s:{stop_reason or 'moving'}")
        elif (stalled >= ROUTE_LEG_STALL_TIMEOUT_S and persistent_stop):
            failure = (
                f"stalled:{stalled:.1f}s:{stop_reason or 'no_progress'}:"
                f"recovery_exhausted={int(recovery_exhausted)}")
        if failure is not None:
            self.nav.invalidate_active_cached_path(
                f"route_leg:{self._route_leg_name}:{failure}", now=now)
            return False, failure
        return False, None

    def _is_shelf_scan_transit(self, target: np.ndarray) -> bool:
        return bool(
            self.flow_phase == "grab"
            and self.state in {
                pick.STATE_GO_SCAN, pick.STATE_DIRECT_TRANSIT}
            and abs(float(target[1]) - pick.SCAN_Y) <= 0.20)

    def _scan_trunk_route_tick(
            self, target: np.ndarray, final_yaw: float) -> bool:
        """Drive to the baseline shelf-approach line in one planned leg.

        The multi-stage delivery-trunk routing (table -> trunk exit ->
        trunk entry -> shelf) is removed: with five random corridor boxes per
        match, trunk anchors and cached routes routinely forced detours and
        stalls near the delivery table (216 s first-scan detour, 150 s trunk
        timeout, post-delivery 135-degree turn beside the table).  One direct
        A* leg to the requested station replaces all trunk stages for GO_SCAN
        transits and for the post-delivery return to shelf A.
        """
        final_goal = (
            float(target[0]), float(target[1]), float(final_yaw))
        # The baseline navigation map defines y=2.40 as the obstacle-aware
        # shelf approach line.  Manipulation targets are 7--15 cm farther
        # north and intentionally sit outside the global planner's remit.
        # Preserve the target column X, but make A* terminate on that safe
        # line; drive_to then performs the short measured shelf-local advance.
        shelf_approach_y = float(SHELF_APPROACH["A"][1])
        route_goal = (
            final_goal[0], min(final_goal[1], shelf_approach_y),
            final_goal[2])
        self.scan_route_final_goal = final_goal
        if self.scan_trunk_route_done:
            return True

        if self.scan_trunk_route_stage is None:
            self.scan_trunk_route_stage = "direct_to_shelf"
            self.get_logger().info(
                "[route] single direct leg to shelf station goal="
                f"({route_goal[0]:.2f},{route_goal[1]:.2f},"
                f"{math.degrees(route_goal[2]):.0f}deg) "
                f"final=({final_goal[0]:.2f},{final_goal[1]:.2f}) "
                "terminal=coarse_then_local")
            self._start_route_leg(
                "scan_direct_to_shelf", route_goal,
                use_memory=False)
            return False

        if (self.scan_trunk_route_stage == "direct_to_shelf"
                and self._route_leg_goal is not None
                and np.linalg.norm(
                    np.asarray(self._route_leg_goal[:2])
                    - np.asarray(route_goal[:2])) > 0.05):
            self._start_route_leg(
                "scan_direct_to_shelf", route_goal,
                use_memory=False)

        reached, failure = self._route_leg_tick()
        if failure is not None:
            raise RuntimeError(
                "shelf transit direct leg failed: " + failure)
        if not reached:
            return False

        self.scan_trunk_route_done = True
        self._route_leg_name = None
        self._route_leg_goal = None
        self._nav_goal = None
        return True

    # ------------------------------------------------------------------
    # drive_to override — shelf transit uses one obstacle-aware coarse leg,
    # followed by the parent's short local trim.  The second stage is a
    # continuous controller hand-off, not another route or ALIGN state.
    # ------------------------------------------------------------------
    def drive_to(self, target_xy, final_yaw: float,
                 position_tolerance: float = 0.055,
                 linear_min_mps: float | None = None,
                 linear_gain: float = pick.NAV_LINEAR_GAIN,
                 rotate_gate_rad: float = pick.NAV_ROTATE_GATE_RAD,
                 translate_angular_max_rps: float = (
                     pick.NAV_TRANSLATE_ANGULAR_MAX_RADPS)) -> bool:
        target = np.asarray(target_xy, dtype=float)
        distance = float(np.linalg.norm(target - self.base_xy))
        shelf_transit = self._is_shelf_scan_transit(target)

        # A previous delivery may leave the chassis inside the conservative
        # whole-body table keep-out.  The normal navigator correctly forbids
        # turning there, but that also means it cannot align toward the next
        # shelf.  First retrace the final south-facing approach in reverse;
        # this moves directly away from known static geometry without an
        # unsafe in-place arm sweep.
        table_clearance = point_to_rect_clearance(
            float(self.base_xy[0]), float(self.base_xy[1]),
            DELIVERY_TABLE_COSTMAP_BOUNDS)
        safe_clearance = (
            WHOLE_BODY_KEEP_OUT_RADIUS + PLACE_CLEAR_TABLE_MARGIN_M)
        if table_clearance < safe_clearance:
            # This branch is also the fail-safe exit for a new worker that
            # inherits a deployed placement arm from a failed predecessor.
            # Any fail-safe retreat from the delivery table must restore both
            # arms and the neutral head posture; reassert it on every tick
            # until the chassis is outside the whole-body keep-out.
            self._command_initial_arm_posture()
            if self._table_escape_started_at is None:
                self._table_escape_started_at = self.now()
            escape_elapsed = self.now() - self._table_escape_started_at
            if escape_elapsed >= PLACE_CLEAR_TABLE_TIMEOUT_S:
                self.set_twist(0.0, 0.0)
                raise RuntimeError(
                    "delivery-table startup escape timed out after "
                    f"{escape_elapsed:.1f}s clearance={table_clearance:.3f}m "
                    f"required={safe_clearance:.3f}m")
            yaw_error = pick.wrap_to_pi(-math.pi / 2.0 - self.base_yaw)
            posture_ready = (
                self.dual_commands_ready(
                    arm_tolerance=0.08, slide_tolerance=0.05)
                and self._initial_head_posture_ready())
            if not posture_ready:
                # Never rotate a deployed whole body beside the table.  First
                # make the inherited/failed-worker posture compact, then use
                # wxj's rotate-into-gate escape below.
                self.set_twist(0.0, 0.0)
            elif abs(yaw_error) <= 0.20:
                self.set_twist(
                    -PLACE_CLEAR_TABLE_SPEED_MPS,
                    float(np.clip(2.0 * yaw_error, -0.25, 0.25)))
            else:
                self.set_twist(
                    0.0, float(np.clip(2.0 * yaw_error, -0.25, 0.25)))
            if not self._table_escape_logged:
                self._table_escape_logged = True
                self.get_logger().warn(
                    "starting inside delivery-table keep-out; "
                    f"clearance={table_clearance:.3f}m "
                    f"required={safe_clearance:.3f}m; reversing north "
                    "and restoring the initial manipulation posture before "
                    "normal navigation")
            return False
        if self._table_escape_logged:
            self._table_escape_logged = False
            self._table_escape_started_at = None
            self.get_logger().info(
                f"delivery-table startup escape complete; "
                f"clearance={table_clearance:.3f}m")

        if shelf_transit:
            if not self._scan_trunk_route_tick(target, final_yaw):
                return False

        if (self.nav_during_scan
                and not shelf_transit
                and distance > max(NAV_TRANSIT_GATE_M,
                                   position_tolerance
                                   + NAV_PRECISE_HANDOFF_MARGIN_M)):
            now = self.now()
            goal = (float(target[0]), float(target[1]), float(final_yaw))
            if self._nav_goal != goal:
                self._nav_goal = goal
                self.nav.set_goal(
                    *goal,
                    use_path_memory=True)
                self._nav_last_log = 0.0
                self._nav_memory_logged = False
                self.get_logger().info(
                    "[nav] new_goal="
                    + json.dumps(
                        {
                            "goal": [round(goal[0], 3), round(goal[1], 3), round(goal[2], 3)],
                            "path_memory": self.nav.path_memory_status(),
                        },
                        ensure_ascii=False,
                    )
                )

            if self._laser_stale(now):
                self.set_twist(0.0, 0.0)
                if now - self._laser_warn_log > 1.0:
                    self.get_logger().warn(
                        "waiting for fresh laser scan during transit "
                        f"(last={self.last_scan_time})")
                    self._laser_warn_log = now
                return False

            v, w, reached = self.nav.update(
                self.base_xy[0], self.base_xy[1], self.base_yaw,
                laser_msg=self.laser_msg, time_now=now)
            self.set_twist(v, w)
            if not self._nav_memory_logged:
                self._nav_memory_logged = True
                self.get_logger().info(
                    "[nav] path_memory_runtime="
                    + json.dumps(self.nav.path_memory_status(), ensure_ascii=False)
                )

            ctrl = self.nav.controller
            if (ctrl.stop_reason is not None
                    and ctrl.stop_reason != self._last_nav_reason):
                self._last_nav_reason = ctrl.stop_reason
                self.get_logger().info(
                    f"[nav] stop_reason={ctrl.stop_reason} "
                    f"lidar={ctrl.lidar_clearance:.2f}m "
                    f"rear={ctrl.rear_clearance:.2f}m "
                    f"v={v:.2f} w={w:.2f}")

            if now - self._nav_last_log >= NAV_PROGRESS_LOG_S:
                self._nav_last_log = now
                self.get_logger().info(
                    f"[nav] to=({goal[0]:.2f},{goal[1]:.2f},"
                    f"{math.degrees(goal[2]):.0f}°) "
                    f"pos=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
                    f"yaw={math.degrees(self.base_yaw):.0f}° "
                    f"dist={distance:.2f}m v={v:.2f} w={w:.2f} "
                    f"reached={reached}")

            if not reached:
                return False
            # Navigator coarse arrival → fall through to precise alignment.
            # Discard the transit command before the centimetre-scale parent
            # controller takes over; otherwise its second velocity ramp can
            # carry momentum through the hand-off and create an avoidable
            # overshoot/reverse correction cycle.
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0

        arrived = super().drive_to(
            target_xy, final_yaw, position_tolerance,
            linear_min_mps=linear_min_mps,
            linear_gain=linear_gain,
            rotate_gate_rad=rotate_gate_rad,
            translate_angular_max_rps=translate_angular_max_rps)
        if arrived:
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0
        return arrived

    # ------------------------------------------------------------------
    # flow hooks
    # ------------------------------------------------------------------
    def _on_grab_complete(self) -> None:
        # The parent reaches DONE only after horizontal withdrawal is complete.
        # Increase holding preload now, then restore during straight backup;
        # rotation waits for verified height feedback.
        grabbed_at = self.now()
        self._post_grab_slow_turn_until = (
            grabbed_at + POST_GRAB_SLOW_TURN_WATCHDOG_GRACE_S
            if self.target_kind == "heweidao" else 0.0)
        self._post_grab_slow_turn_logged = False
        self._loaded_motion_limit_logged = False
        if not self._memory_consumed:
            slot = self.target_slot()
            if slot is not None:
                shelf, level, column = slot
                event = {
                    "shelf": shelf,
                    "level": level,
                    "column": column,
                    "kind": self.target_kind,
                    "marker_id": self.target_marker_id,
                }
                self.memory_consume_pub.publish(String(
                    data=json.dumps(event, ensure_ascii=False)))
                self._memory_consumed = True
                self.get_logger().info(
                    f"[memory] immediate post-grasp consume "
                    f"kind={self.target_kind} slot={slot}")
        self._capture_transport_grip_command()
        self._drop_monitor_armed_at = (
            self.now() + TRANSPORT_DROP_MONITOR_GRACE_S)
        self._drop_signature_since = None
        self._drop_candidate_reference_world = None
        initial_empty, initial_feedback = self._transport_drop_signature()
        self.get_logger().info(
            "[drop-monitor] armed after post-grasp grace "
            f"grace={TRANSPORT_DROP_MONITOR_GRACE_S:.2f}s "
            f"initial_empty={int(initial_empty)} "
            f"feedback={initial_feedback}")
        self.des_slide = self._transit_slide_target()
        self.get_logger().info(
            f"[flow] goods grabbed (marker={self.target_marker_id}, "
            f"kind={self.target_kind}, state={self.state}); "
            "preparing delivery transit")
        if self.backup_after_grab_m > 1e-4:
            self._set_flow_phase("backup")
            self._backup_start_xy = self.base_xy.copy()
            self._backup_start_yaw = float(self.base_yaw)
            delivery_goal = self._delivery_slot_goal()
            delivery_heading = math.atan2(
                float(delivery_goal[1]) - float(self.base_xy[1]),
                float(delivery_goal[0]) - float(self.base_xy[0]))
            delivery_turn = pick.wrap_to_pi(
                delivery_heading - self._backup_start_yaw)
            bounded_delivery_turn = float(np.clip(
                delivery_turn,
                -BACKUP_TURN_MAX_ANGLE_RAD,
                BACKUP_TURN_MAX_ANGLE_RAD))
            self._backup_turn_target_yaw = pick.wrap_to_pi(
                self._backup_start_yaw + bounded_delivery_turn)
            self._backup_t0 = self.now()
            self._backup_logged = False
            self.get_logger().info(
                f"[flow] backing up {self.backup_after_grab_m:.2f}m "
                "with a bounded delivery turn after shelf clearance "
                f"target_delta={math.degrees(bounded_delivery_turn):.1f}°")
            return
        self._start_height_restore()

    def _start_height_restore(self) -> None:
        self._set_flow_phase("restore_height")
        self._height_restore_t0 = self.now()
        self._height_restore_monotonic_t0 = time.monotonic()
        self._height_restore_timeout_logged = False
        self.des_slide = self._transit_slide_target()
        self.set_twist(0.0, 0.0)
        self.get_logger().info(
            f"[flow] restoring post-grasp transit slide: measured="
            f"{self.joints.get('slide_joint')} "
            f"target={self.des_slide:.3f}")

    def _restore_height_tick(self) -> None:
        now = self.now()
        elapsed = max(
            0.0, time.monotonic() - self._height_restore_monotonic_t0)
        target_slide = self._transit_slide_target()
        self.set_twist(0.0, 0.0)
        self.des_slide = target_slide
        measured_slide = self.joints.get("slide_joint")
        if measured_slide is not None and math.isfinite(float(measured_slide)):
            error = abs(float(measured_slide) - target_slide)
            if error <= TRANSIT_SLIDE_TOLERANCE_M:
                self.get_logger().info(
                    f"[flow] post-grasp transit slide restored: measured="
                    f"{float(measured_slide):.3f} error={error:.3f}m; "
                    "starting delivery navigation")
                self._start_delivery_navigation()
                return
        if (not self._height_restore_timeout_logged
                and elapsed >= TRANSIT_SLIDE_TIMEOUT_S):
            self._height_restore_timeout_logged = True
            self.get_logger().warn(
                f"[flow] post-grasp transit slide restore timed out after "
                f"{TRANSIT_SLIDE_TIMEOUT_S:.1f}s "
                f"(measured={measured_slide}, target={target_slide:.3f}); "
                "remaining stopped through the hard safety ceiling")
        if elapsed < TRANSIT_SLIDE_HARD_TIMEOUT_S:
            return

        measured_finite = False
        error = float("inf")
        try:
            measured = float(measured_slide)
            measured_finite = math.isfinite(measured)
            if measured_finite:
                error = abs(measured - target_slide)
        except (TypeError, ValueError):
            measured = float("nan")
        if (measured_finite
                and error <= TRANSIT_SLIDE_DEGRADED_MAX_ERROR_M):
            self.get_logger().warn(
                "[flow] transit slide missed the normal tolerance but is "
                f"inside the degraded safety envelope after {elapsed:.1f}s "
                f"(measured={measured:.3f}, error={error:.3f}m); "
                "continuing delivery navigation")
            self._start_delivery_navigation()
            return

        self._enter_fatal_recovery(RuntimeError(
            "post-grasp transit slide remained outside the safe envelope "
            f"for {elapsed:.1f}s (measured={measured_slide}, "
            f"target={target_slide:.3f}, error={error:.3f}m)"))

    def _start_delivery_navigation(self) -> None:
        self._set_flow_phase("nav_to_delivery")
        self.des_slide = self._transit_slide_target()
        self._nav_goal = None
        self._nav_last_log = 0.0
        self._nav_memory_logged = False
        # Single direct leg to the assigned slot; the delivery-trunk stages
        # (trunk entry -> trunk forward -> slot) are removed because trunk
        # anchors and cached routes routinely stalled beside the delivery
        # table under random box layouts (see _scan_trunk_route_tick note).
        self.delivery_nav_stage = "direct_to_slot"
        self._start_route_leg(
            "delivery_direct_to_slot", self._delivery_slot_goal(),
            use_memory=False)
        self.get_logger().info(
            f"[nav→delivery] assigned slot="
            f"{None if self.place_slot is None else self.place_slot + 1} "
            f"target={np.round(self.place_world[:2], 3)} "
            f"direct_single_leg=True")

    def _delivery_slot_goal(self) -> tuple[float, float, float]:
        return (
            float(self.place_world[0]),
            DELIVERY_APPROACH[1],
            DELIVERY_APPROACH[2],
        )

    def _delivery_watchdog_goal(self) -> tuple[float, float, float]:
        """Return the goal of the delivery leg that is actually active.

        Continuous-order clients add a flow-level watchdog around this
        multi-leg route.  Resetting the navigator to the final drop goal while
        an entry/trunk leg is active desynchronizes the route state machine and
        makes it execute already-passed legs at the table.
        """
        route_goal = getattr(self, "_route_leg_goal", None)
        if (getattr(self, "delivery_nav_stage", None) != "slot_refine"
                and route_goal is not None):
            return tuple(float(value) for value in route_goal)
        return self._delivery_slot_goal()

    def _backup_tick(self) -> None:
        """Clear the shelf straight, then reverse through a bounded arc."""
        now = self.now()
        self.des_slide = self._transit_slide_target()
        if self._backup_start_xy is None:
            self._backup_start_xy = self.base_xy.copy()
            self._backup_start_yaw = float(self.base_yaw)
            self._backup_turn_target_yaw = self._backup_start_yaw
            self._backup_t0 = now

        heading = np.array([
            math.cos(self._backup_start_yaw),
            math.sin(self._backup_start_yaw),
        ])
        moved_back = float(np.dot(
            self._backup_start_xy - self.base_xy, heading))
        yaw_err = pick.wrap_to_pi(self._backup_start_yaw - self.base_yaw)
        elapsed = now - self._backup_t0

        reached = moved_back >= self.backup_after_grab_m
        timed_out = elapsed > BACKUP_TIMEOUT_S
        if reached or timed_out:
            self.set_twist(0.0, 0.0)
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0
            turned_deg = math.degrees(pick.wrap_to_pi(
                self.base_yaw - self._backup_start_yaw))
            message = (
                f"[flow] backup finished (moved={moved_back:.3f}m, "
                f"turned={turned_deg:.1f}°, "
                f"elapsed={elapsed:.1f}s); verifying transit height")
            if timed_out and not reached:
                self.get_logger().warn(message + " after timeout")
            else:
                self.get_logger().info(message)
            self._start_height_restore()
            return

        if (moved_back >= BACKUP_TURN_CLEARANCE_M
                and self._backup_turn_target_yaw is not None):
            yaw_err = pick.wrap_to_pi(
                self._backup_turn_target_yaw - self.base_yaw)
            angular = float(np.clip(
                BACKUP_TURN_GAIN * yaw_err,
                -BACKUP_TURN_MAX_RPS,
                BACKUP_TURN_MAX_RPS))
        else:
            # Preserve the original straight withdrawal until the protruding
            # product and arm have cleared the shelf face.
            angular = float(np.clip(2.0 * yaw_err, -0.6, 0.6))
        self.set_twist(-BACKUP_SPEED_MPS, angular)
        if not self._backup_logged and elapsed >= 1.0:
            self._backup_logged = True
            self.get_logger().info(
                f"[backup] dist={moved_back:.3f}/"
                f"{self.backup_after_grab_m:.2f}m "
                f"yaw_err={math.degrees(yaw_err):.1f}° "
                f"turn_active={int(moved_back >= BACKUP_TURN_CLEARANCE_M)}")

    def _nav_to_delivery_tick(self) -> None:
        now = self.now()
        self.des_slide = self._transit_slide_target()
        if self.delivery_nav_stage != "slot_refine":
            reached, failure = self._route_leg_tick()
            if failure is not None:
                raise RuntimeError(
                    "delivery direct navigation failed: " + failure)
            if not reached:
                return
            self.delivery_nav_stage = "slot_refine"

        # Navigator yaw tolerance is 0.15 rad; refine to face south before the
        # guarded final creep and loaded-arm motion begin.
        self.cmd_linear = 0.0
        yaw_err = pick.wrap_to_pi(-math.pi / 2.0 - self.base_yaw)
        if abs(yaw_err) > 0.03:
            self.set_twist(0.0, 2.0 * yaw_err)
            return
        self.set_twist(0.0, 0.0)
        self.cmd_angular = 0.0
        self._set_flow_phase("place")
        self.place_stage = 0
        self.place_t0 = now
        self.get_logger().info(
            f"[flow] arrived at delivery approach via=direct "
            f"pos=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
            f"yaw={math.degrees(self.base_yaw):.0f}deg; placing with "
            f"grip_command={self._transport_grip_command} "
            f"measured_grip={self.selected_gripper_position()}")

    def _set_selected_grip(self, value: float) -> None:
        if self.grasp_arm == "r":
            self.des_right_grip = float(value)
        else:
            self.des_left_grip = float(value)

    @staticmethod
    def _rounded_list(values, decimals: int = 4) -> list[float] | None:
        """Return a compact JSON-safe vector for placement diagnostics."""
        if values is None:
            return None
        array = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(array)):
            return None
        return np.round(array, decimals).tolist()

    def _place_joint_snapshot(self) -> dict:
        """Capture measured, streamed and target placement state."""
        measured = self.selected_arm_positions()
        commanded = (
            self.cmd_right_arm if self.grasp_arm == "r"
            else self.cmd_left_arm)
        desired = (
            self.des_right_arm if self.grasp_arm == "r"
            else self.des_left_arm)
        tcp = self.selected_tcp_world()
        measured_slide = self.joints.get("slide_joint")
        return {
            "stage": int(self.place_stage),
            "arm": self.grasp_arm,
            "ref_source": self.place_ik_ref_source,
            "measured_joints": self._rounded_list(measured),
            "commanded_joints": self._rounded_list(commanded),
            "desired_joints": self._rounded_list(desired),
            "desired_minus_measured": self._rounded_list(
                np.asarray(desired, dtype=float) - measured),
            "max_joint_delta_from_measured": (
                None if not np.all(np.isfinite(measured))
                else round(float(np.max(np.abs(
                    np.asarray(desired, dtype=float) - measured))), 4)),
            "measured_slide": (
                None if measured_slide is None
                else round(float(measured_slide), 4)),
            "commanded_slide": round(float(self.cmd_slide), 4),
            "desired_slide": round(float(self.des_slide), 4),
            "tcp_world": self._rounded_list(tcp),
            "grip_command": (
                None if self._transport_grip_command is None
                else round(float(self._transport_grip_command), 4)),
            "measured_grip": self.selected_gripper_position(),
            **self._place_base_diagnostic(),
        }

    def _arm_tcp_pose_diagnostic(
            self, side: str, joints: np.ndarray,
            slide: float | None) -> dict | None:
        """Return FK endpoint position and orientation in the world frame."""
        joints = np.asarray(joints, dtype=float)
        if (slide is None or not math.isfinite(float(slide))
                or joints.shape != (6,) or not np.all(np.isfinite(joints))):
            return None
        left, right = self.kdl.forward_kinematics(
            np.concatenate(([float(slide)], joints)), index=side)
        transform = left if side == "left" else right
        world_rotation = (
            pick.Rotation.from_euler("z", self.base_yaw).as_matrix()
            @ transform[:3, :3])
        return {
            "xyz": self._rounded_list(
                self.footprint_to_world(transform[:3, 3])),
            "rpy": self._rounded_list(
                pick.Rotation.from_matrix(world_rotation).as_euler("xyz")),
        }

    def _log_place_motion(self, now: float, stage: int) -> None:
        """Periodic per-joint motion log during the placement phase.

        每条 [place-motion] 记录基座位姿、左右臂六关节的 measured/command/
        desired、slide、夹爪、TCP 世界坐标与商品底部高度，便于逐 tick 还原
        摆臂/精修/下降/释放阶段的机械臂运动状态。
        """
        if now - self._place_motion_last_log < PLACE_MOTION_LOG_PERIOD_S:
            return
        self._place_motion_last_log = now
        slide = self.joints.get("slide_joint")
        tcp = self.selected_tcp_world()
        half_height = PRODUCT_HALF_HEIGHT_M.get(self.target_kind, 0.0)
        bottom_z = None
        if tcp is not None and np.all(np.isfinite(tcp)):
            bottom_z = round(float(
                tcp[2] - self._tcp_above_product_center() - half_height), 4)
        stage_names = {
            0: "overhead_approach",
            1: "horizontal_refine",
            2: "vertical_descent",
            3: "release",
            4: "vertical_clear",
            5: "horizontal_retreat",
        }
        joint_positions = {
            name: round(float(value), 5)
            for name, value in sorted(self.joints.items())
            if math.isfinite(float(value))
        }
        joint_velocities = {
            name: round(float(value), 5)
            for name, value in sorted(self.joint_velocities.items())
            if math.isfinite(float(value))
        }
        joint_efforts = {
            name: round(float(value), 4)
            for name, value in sorted(self.joint_efforts.items())
            if math.isfinite(float(value))
        }
        left_measured = self.arm_positions("left")
        right_measured = self.arm_positions("right")
        document = {
            "time": round(float(now), 4),
            "phase": self.flow_phase,
            "stage": int(stage),
            "stage_name": stage_names.get(int(stage), "unknown"),
            "stage_elapsed": round(max(0.0, float(now - self.place_t0)), 4),
            "arm": self.grasp_arm,
            "base": self._rounded_list(
                np.concatenate((self.base_xy, [self.base_yaw]))),
            "base_twist_meas": [
                round(float(self.base_measured_linear), 5),
                round(float(self.base_measured_angular), 5),
            ],
            "base_twist_cmd": [
                round(float(self.cmd_linear), 5),
                round(float(self.cmd_angular), 5),
            ],
            "base_twist_des": [
                round(float(self.des_linear), 5),
                round(float(self.des_angular), 5),
            ],
            "slide_meas": (
                None if slide is None else round(float(slide), 4)),
            "slide_cmd": round(float(self.cmd_slide), 4),
            "slide_des": round(float(self.des_slide), 4),
            "head_meas": self._rounded_list(np.array([
                self.joints.get("head_yaw_joint", float("nan")),
                self.joints.get("head_pitch_joint", float("nan")),
            ])),
            "head_cmd": self._rounded_list(self.cmd_head),
            "head_des": self._rounded_list(self.des_head),
            "tcp": self._rounded_list(tcp),
            "product_bottom_z": bottom_z,
            "table_clearance": (
                None if bottom_z is None
                else round(float(bottom_z - DELIVERY_TABLE_TOP_Z_M), 4)),
            "transport_grip_cmd": (
                None if self._transport_grip_command is None
                else round(float(self._transport_grip_command), 4)),
            "left_arm_meas": self._rounded_list(left_measured),
            "left_arm_cmd": self._rounded_list(self.cmd_left_arm),
            "left_arm_des": self._rounded_list(self.des_left_arm),
            "right_arm_meas": self._rounded_list(right_measured),
            "right_arm_cmd": self._rounded_list(self.cmd_right_arm),
            "right_arm_des": self._rounded_list(self.des_right_arm),
            "left_grip_meas": self.joints.get(
                "left_arm_eef_gripper_joint"),
            "left_grip_cmd": round(float(self.cmd_left_grip), 5),
            "left_grip_des": round(float(self.des_left_grip), 5),
            "right_grip_meas": self.joints.get(
                "right_arm_eef_gripper_joint"),
            "right_grip_cmd": round(float(self.cmd_right_grip), 5),
            "right_grip_des": round(float(self.des_right_grip), 5),
            "left_tcp_meas": self._arm_tcp_pose_diagnostic(
                "left", left_measured, slide),
            "left_tcp_cmd": self._arm_tcp_pose_diagnostic(
                "left", self.cmd_left_arm, self.cmd_slide),
            "left_tcp_des": self._arm_tcp_pose_diagnostic(
                "left", self.des_left_arm, self.des_slide),
            "right_tcp_meas": self._arm_tcp_pose_diagnostic(
                "right", right_measured, slide),
            "right_tcp_cmd": self._arm_tcp_pose_diagnostic(
                "right", self.cmd_right_arm, self.cmd_slide),
            "right_tcp_des": self._arm_tcp_pose_diagnostic(
                "right", self.des_right_arm, self.des_slide),
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_efforts": joint_efforts,
        }
        if self.use_dual_tissue_grasp:
            document["centre_world"] = self._rounded_list(
                self._dual_release_world())
        self.get_logger().info(
            "[place-motion] "
            + json.dumps(document, ensure_ascii=False, separators=(",", ":")))

    def _capture_transport_grip_command(self) -> None:
        """Strengthen and remember the closed command after shelf retreat."""
        if self.use_dual_tissue_grasp:
            self._transport_grip_command = float(
                pick.DUAL_TISSUE_GRIP_COMMAND)
            self.des_left_grip = self._transport_grip_command
            self.des_right_grip = self._transport_grip_command
            self.get_logger().info(
                f"[grip-hold] dual transport command="
                f"{self._transport_grip_command:.3f} (position limit)")
            return
        grasp_command = float(
            self.des_right_grip
            if self.grasp_arm == "r" else self.des_left_grip)
        if self.use_sphere_grasp:
            self._transport_grip_command = SPHERE_TRANSPORT_GRIP_COMMAND
        else:
            self._transport_grip_command = float(np.clip(
                grasp_command - TRANSPORT_GRIP_PRELOAD_COMMAND,
                0.0, pick.GRIP_OPEN))
        self._set_selected_grip(self._transport_grip_command)
        self.get_logger().info(
            f"[grip-hold] arm={self.grasp_arm} "
            f"capture_command={grasp_command:.3f} -> "
            f"transport_command={self._transport_grip_command:.3f} "
            f"measured={self.selected_gripper_position()}")

    def _hold_grasp_during_transport(self) -> None:
        """Reassert the closed command until the verified release stage."""
        if self._transport_grip_command is None:
            return
        if self.use_dual_tissue_grasp:
            self.des_left_grip = self._transport_grip_command
            self.des_right_grip = self._transport_grip_command
        else:
            self._set_selected_grip(self._transport_grip_command)

    def _held_product_reference_world(self) -> np.ndarray | None:
        """Return the best measured proxy for the carried product centre."""
        if self.use_dual_tissue_grasp:
            return self._dual_release_world()
        return self.selected_tcp_world()

    def _transport_drop_signature(self) -> tuple[bool, dict]:
        """Infer an empty grasp from measured gripper positions.

        The three grasp families have deliberately different feedback
        semantics, so sharing one numeric threshold would be unsafe:
        spheres remain visibly open around the fruit, generic fingers close
        almost to their command after losing the item, and an unloaded dual
        tissue clamp springs both measured joints above its contact range.
        """
        if self.use_dual_tissue_grasp:
            left = self.joints.get("left_arm_eef_gripper_joint")
            right = self.joints.get("right_arm_eef_gripper_joint")
            if left is None or right is None:
                return False, {"mode": "dual", "feedback": "missing"}
            left = float(left)
            right = float(right)
            if not math.isfinite(left) or not math.isfinite(right):
                return False, {"mode": "dual", "feedback": "invalid"}
            threshold = float(pick.DUAL_TISSUE_GRIP_CONTACT_MAX)
            return (
                left > threshold and right > threshold,
                {
                    "mode": "dual",
                    "left_grip": round(left, 4),
                    "right_grip": round(right, 4),
                    "empty_threshold": round(threshold, 4),
                })

        measured = self.selected_gripper_position()
        if measured is None:
            return False, {"mode": "single", "feedback": "missing"}
        if self.use_sphere_grasp:
            threshold = float(SPHERE_TRANSPORT_HELD_MINIMUM.get(
                self.target_kind, self.sphere_capture_minimum()))
            return (
                measured <= threshold,
                {
                    "mode": "sphere",
                    "measured_grip": round(measured, 4),
                    "held_minimum": round(threshold, 4),
                })

        close_command = float(pick.GRIP_CLOSE_BY_CLASS.get(
            self.target_kind, pick.GENERIC_GRIP_CLOSE))
        threshold = close_command + float(pick.GENERIC_EMPTY_GRIP_MARGIN)
        return (
            measured <= threshold,
            {
                "mode": "generic",
                "measured_grip": round(measured, 4),
                "empty_maximum": round(threshold, 4),
                "close_command": round(close_command, 4),
            })

    def _start_transport_drop_recovery(
            self, now: float, *, delivered: bool, over_table: bool,
            details: dict,
            reference: np.ndarray | None = None) -> None:
        """Stop motion and recover posture after a confirmed product loss."""
        if reference is None:
            reference = self._held_product_reference_world()
        reference_list = self._rounded_list(reference)
        self.drop_event = {
            "outcome": "delivered_above_table" if delivered else "retry",
            "phase": self.flow_phase,
            "place_stage": int(self.place_stage),
            "product_reference_world": reference_list,
            "over_delivery_table": bool(over_table),
            "feedback": details,
        }
        self.set_twist(0.0, 0.0)
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        # If the item has just fallen onto the table, immediately sweeping the
        # wrist toward the compact posture can hit it before it settles.  Lock
        # both measured arms and raise only the common slide first; neutral arm
        # recovery begins in _drop_recovery_tick after vertical clearance.
        self._drop_recovery_vertical_clear = bool(over_table)
        self._drop_recovery_vertical_clear_started_at = now
        if self._drop_recovery_vertical_clear:
            measured_slide = self.joints.get("slide_joint")
            measured_left = self.arm_positions("left")
            measured_right = self.arm_positions("right")
            if (measured_slide is not None
                    and math.isfinite(float(measured_slide))
                    and np.all(np.isfinite(measured_left))
                    and np.all(np.isfinite(measured_right))):
                self.des_left_arm = measured_left.copy()
                self.des_right_arm = measured_right.copy()
                self.des_left_grip = pick.GRIP_OPEN
                self.des_right_grip = pick.GRIP_OPEN
                self.des_slide = max(
                    pick.SLIDE_MIN,
                    float(measured_slide) - PLACE_VERTICAL_CLEARANCE_M)
                self.commands_ready_since = None
                self.get_logger().warn(
                    "[drop-monitor] product loss occurred over the table; "
                    "raising the empty wrists vertically before neutral "
                    f"recovery slide={float(measured_slide):.3f}->"
                    f"{self.des_slide:.3f}")
            else:
                self._drop_recovery_vertical_clear = False
                self._command_initial_arm_posture()
        else:
            self._command_initial_arm_posture()
        self.commands_ready_since = None
        self._drop_recovery_started_at = now
        self._drop_signature_since = None
        self._drop_candidate_reference_world = None

        event_json = json.dumps(
            self.drop_event, ensure_ascii=False, separators=(",", ":"))
        if delivered:
            # Delivery is irreversible at this point.  Mark it immediately so
            # a later posture/return-scan fault cannot schedule a duplicate.
            self.delivery_completed_by_drop = True
            self.placement_completed = True
            self.get_logger().info(
                "[drop-monitor] product released at the assigned low table "
                "pose; counting order as delivered and skipping remaining "
                f"loaded placement stages event={event_json}")
            self._set_flow_phase("drop_success_recover")
            return

        self.terminal_error = (
            "product dropped before verified low table release; retry this "
            f"order event={event_json}")
        self.get_logger().error(
            "[drop-monitor] product loss did not satisfy the assigned-slot "
            "low-release gate; stopping worker for same-order retry "
            f"event={event_json}")
        self._set_flow_phase("drop_failed_recover")

    def _monitor_held_product(self, now: float) -> bool:
        """Debounce gripper feedback and handle a confirmed product loss."""
        active = (
            self.flow_phase in {
                "backup", "restore_height", "nav_to_delivery"}
            or (self.flow_phase == "place" and self.place_stage in {0, 1, 2}))
        if not active:
            self._drop_signature_since = None
            self._drop_candidate_reference_world = None
            return False
        if (self._drop_monitor_armed_at is None
                or now < self._drop_monitor_armed_at):
            return False

        lost, details = self._transport_drop_signature()
        if not lost:
            self._drop_signature_since = None
            self._drop_candidate_reference_world = None
            return False
        if self._drop_signature_since is None:
            self._drop_signature_since = now
            reference = self._held_product_reference_world()
            self._drop_candidate_reference_world = (
                None if reference is None
                else np.asarray(reference, dtype=float).copy())
            self.get_logger().warn(
                "[drop-monitor] possible product loss; waiting for "
                f"{TRANSPORT_DROP_CONFIRM_S:.2f}s confirmation "
                f"feedback={details} reference="
                f"{self._rounded_list(self._drop_candidate_reference_world)}")
        elif self._drop_candidate_reference_world is None:
            reference = self._held_product_reference_world()
            self._drop_candidate_reference_world = (
                None if reference is None
                else np.asarray(reference, dtype=float).copy())
        # Stop immediately on the first empty-grasp signature.  Confirmation
        # still debounces the order decision, but the base/arm no longer move
        # the reference point across the table boundary in that interval.
        self.set_twist(0.0, 0.0)
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        if now - self._drop_signature_since < TRANSPORT_DROP_CONFIRM_S:
            return True

        reference = self._drop_candidate_reference_world
        # A high wrist above the official box is not proof of a stable table
        # placement: the orange run lost its grip about 147 mm above the table
        # and could still bounce or roll off.  Accept an uncommanded release
        # only during the vertical-descent stage, inside the conservative
        # table inset and assigned slot, with the estimated product bottom at
        # table height.  Keep the official rectangle separately so every
        # over-table loss receives a vertical-first arm recovery.
        over_table = self._tcp_over_delivery_table_official(reference)
        delivered = bool(
            self.flow_phase == "place"
            and self._tcp_over_delivery_table(reference)
            and self._tcp_at_assigned_slot(reference))
        self._start_transport_drop_recovery(
            now, delivered=delivered, over_table=over_table, details=details,
            reference=reference)
        return True

    def _drop_recovery_tick(self, now: float, *, delivered: bool) -> None:
        """Recover a neutral footprint, then finish or fail the worker."""
        self.set_twist(0.0, 0.0)
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        if self._drop_recovery_vertical_clear:
            self.des_left_grip = pick.GRIP_OPEN
            self.des_right_grip = pick.GRIP_OPEN
            ready = self.dual_commands_ready(
                arm_tolerance=0.05, slide_tolerance=0.020)
            timed_out = (
                now - self._drop_recovery_vertical_clear_started_at
                >= PLACE_VERTICAL_CLEAR_TIMEOUT_S)
            if not ready and not timed_out:
                return
            if timed_out:
                self.get_logger().warn(
                    "[drop-monitor] vertical wrist clearance did not fully "
                    f"settle within {PLACE_VERTICAL_CLEAR_TIMEOUT_S:.1f}s; "
                    "continuing neutral recovery from the raised command")
            else:
                self.get_logger().info(
                    "[drop-monitor] vertical wrist clearance reached; "
                    "neutral arm recovery may begin")
            self._drop_recovery_vertical_clear = False
            self._drop_recovery_started_at = now
            self._command_initial_arm_posture()
            return
        self._command_initial_arm_posture()
        ready = (
            self.dual_commands_ready(
                arm_tolerance=0.08, slide_tolerance=0.05)
            and self._initial_head_posture_ready())
        timed_out = (
            now - self._drop_recovery_started_at
            >= TRANSPORT_DROP_RECOVERY_TIMEOUT_S)
        if not ready and not timed_out:
            return
        if timed_out:
            message = (
                "neutral posture recovery timed out after confirmed "
                "product loss")
            if delivered:
                self._post_delivery_warning(message)
            else:
                self.terminal_error = f"{self.terminal_error}; {message}"

        if not delivered:
            self.place_t0 = now
            self._set_flow_phase("drop_failed")
            self.get_logger().error(
                "[drop-monitor] failed transport attempt finished; "
                "shutting down for runner retry")
            return

        # Reuse the existing obstacle-aware table clearance and first-order
        # return-to-A path.  The product is already gone, so begin with both
        # arms neutral instead of replaying descent/release/vertical-clear.
        clearance = point_to_rect_clearance(
            float(self.base_xy[0]), float(self.base_xy[1]),
            DELIVERY_TABLE_COSTMAP_BOUNDS)
        self.place_stage = 5
        self.place_t0 = now
        self._place_retreat_start_clearance = clearance
        self._place_retreat_sent = True
        self._set_flow_phase("place")
        self.get_logger().info(
            "[drop-monitor] neutral posture recovered; entering table "
            f"clearance only (clearance={clearance:.3f}m)")

    def _command_initial_arm_posture(self) -> None:
        """Return the manipulators and camera assembly to the initial pose."""
        self.des_left_arm = np.asarray(PLACE_RETREAT_ARM_L, dtype=float)
        self.des_right_arm = np.asarray(PLACE_RETREAT_ARM_R, dtype=float)
        self.des_left_grip = pick.GRIP_OPEN
        self.des_right_grip = pick.GRIP_OPEN
        self.des_slide = pick.SLIDE_REFERENCE_COMMAND
        self.des_head = np.asarray(
            PLACE_RETREAT_HEAD_YAW_PITCH, dtype=float)

    def _command_fatal_recovery_posture(self) -> None:
        """Retract safely while retaining any not-yet-delivered payload."""
        self._command_initial_arm_posture()
        if (self._transport_grip_command is not None
                and not self.placement_completed
                and not self.delivery_completed_by_drop):
            self._hold_grasp_during_transport()

    def _neutral_posture_ready(self) -> bool:
        """Whether both arms, slide and head reached the initial pose."""
        return bool(
            self.dual_commands_ready(
                arm_tolerance=0.08, slide_tolerance=0.05)
            and self._initial_head_posture_ready())

    def _abort_recovery_ready(self) -> bool:
        """Restore the neutral posture before an abort shuts the worker down.

        Every grasp-failure path (STATE_ABORT) first streams both arms, both
        grippers, slide and head back to the initial pose; the worker exits
        only after the posture settles or the recovery ceiling is reached.
        """
        self._command_initial_arm_posture()
        if self._neutral_posture_ready():
            self.get_logger().info(
                "[abort] neutral posture recovered before exit")
            return True
        if self.now() - self.state_t0 >= ABORT_RECOVERY_TIMEOUT_S:
            self.get_logger().warn(
                "[abort] neutral posture recovery timed out after "
                f"{ABORT_RECOVERY_TIMEOUT_S:.1f}s; shutting down anyway")
            return True
        return False

    def _enter_fatal_recovery(self, exc: Exception) -> None:
        """Freeze the base and recover without an unverified payload drop.

        rclpy swallows timer-callback exceptions, so without this a placement
        RuntimeError would leave the worker parked mid-motion until the runner
        kills it on timeout.  Catch it here, restore the posture and shut the
        worker down cleanly with the error reported to the runner.
        """
        # A descent-stage timeout at an already verified low slot is a release
        # condition, not a reason to open and retract simultaneously.  Reuse
        # the normal in-place release state so the loaded arm remains fixed
        # through the full gripper dwell.  If the exception happened after the
        # release state had already begun, use the same vertical-first recovery
        # as a confirmed low drop before any neutral arm motion.
        if (self._transport_grip_command is not None
                and self.flow_phase == "place"
                and self.place_stage in {2, 3}
                and not self.placement_completed
                and not self.delivery_completed_by_drop):
            now = self.now()
            reference = self._held_product_reference_world()
            verified_low_slot = bool(
                self._tcp_over_delivery_table(reference)
                and self._tcp_at_assigned_slot(reference)
                and self._product_bottom_at_table(reference))
            if verified_low_slot:
                if self.place_stage == 2:
                    self.get_logger().warn(
                        "[place-recovery] descent failed after the product "
                        "reached its verified low slot; stopping the slide "
                        "and entering the normal fixed-arm release dwell "
                        f"(cause={type(exc).__name__}: {exc})")
                    self._place_contact_release(now, reference)
                else:
                    details = {
                        "mode": "fatal_recovery",
                        "reason": "exception_during_verified_release",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    self.get_logger().warn(
                        "[place-recovery] release-stage error at a verified "
                        "low slot; clearing the open wrists vertically before "
                        f"neutral recovery (cause={type(exc).__name__}: {exc})")
                    self._start_transport_drop_recovery(
                        now, delivered=True, over_table=True,
                        details=details, reference=reference)
                return

        # Once a loaded wrist is already safely above its assigned table area,
        # retracting toward the neutral pose can drag or sweep the product over
        # the table edge.  Convert any stage-0/1 placement failure into the
        # normal vertical descent instead.  The measured arm pose is frozen,
        # the common slide lowers the product, and the existing contact/release
        # gates complete the placement.  Off-table failures still use the
        # conservative clamped recovery below.
        if (self._transport_grip_command is not None
                and self.flow_phase == "place"
                and self.place_stage in {0, 1}
                and not self.placement_completed
                and not self.delivery_completed_by_drop):
            now = self.now()
            reference = (
                self._dual_release_world()
                if self.use_dual_tissue_grasp
                else self.selected_tcp_world())
            release_geometry_ready = (
                self.use_dual_tissue_grasp
                or self.place_release_world is not None)
            if (release_geometry_ready
                    and self._place_timeout_fallback_safe(reference)):
                error_xy = (
                    np.asarray(reference, dtype=float)[:2]
                    - self.place_world[:2])
                self.get_logger().warn(
                    "[place-recovery] loaded placement error occurred over "
                    "the table; locking the measured arm pose and descending "
                    "in place instead of retracting with the product "
                    f"(stage={self.place_stage} "
                    f"slot_error={np.linalg.norm(error_xy):.3f}m "
                    f"cause={type(exc).__name__}: {exc})")
                try:
                    if self.use_dual_tissue_grasp:
                        self._begin_dual_place_descent(now, reference)
                    else:
                        self._begin_single_place_descent(now, reference)
                    return
                except Exception as fallback_exc:  # noqa: BLE001
                    self.get_logger().error(
                        "[place-recovery] vertical fallback could not start; "
                        "continuing to the clamped fatal recovery: "
                        f"{type(fallback_exc).__name__}: {fallback_exc}")
                    exc = RuntimeError(
                        f"{exc}; vertical fallback failed: {fallback_exc}")

        if self._fatal_error is None:
            self._fatal_error = f"{type(exc).__name__}: {exc}"
            self._fatal_recovery_started_at = self.now()
            self.get_logger().error(
                f"[fatal] phase={self.flow_phase} state={self.state} "
                f"place_stage={self.place_stage}: {self._fatal_error}; "
                "recovering posture before exit")
            self._command_fatal_recovery_posture()
            self.set_twist(0.0, 0.0)
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0
        self.flow_phase = "fatal_recover"

    def _fatal_recovery_tick(self, now: float) -> bool:
        """Hold the recovery posture; return True when the worker may exit."""
        self.set_twist(0.0, 0.0)
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        self._command_fatal_recovery_posture()
        if self._neutral_posture_ready():
            self.get_logger().error(
                f"[fatal] posture recovered; worker finished after "
                f"{self._fatal_error}")
            return True
        if now - self._fatal_recovery_started_at >= FATAL_RECOVERY_TIMEOUT_S:
            self.get_logger().warn(
                "[fatal] posture recovery timed out after "
                f"{FATAL_RECOVERY_TIMEOUT_S:.1f}s; exiting anyway")
            self.get_logger().error(
                f"[fatal] worker finished after {self._fatal_error}")
            return True
        return False

    def _initial_head_posture_ready(self) -> bool:
        """Whether measured head joints have reached the neutral transit pose."""
        yaw = self.joints.get("head_yaw_joint")
        pitch = self.joints.get("head_pitch_joint")
        if yaw is None or pitch is None:
            return False
        measured = np.asarray([yaw, pitch], dtype=float)
        return bool(
            np.all(np.isfinite(measured))
            and np.max(np.abs(
                measured - PLACE_RETREAT_HEAD_YAW_PITCH))
            <= PLACE_RETREAT_HEAD_TOLERANCE_RAD)

    def _start_place_retreat(self, now: float) -> None:
        """Back fully clear while preserving the vertically raised pose."""
        self.place_stage = 5
        self.place_t0 = now
        self._place_retreat_sent = False
        self.commands_ready_since = None
        self._place_retreat_start_clearance = point_to_rect_clearance(
            float(self.base_xy[0]), float(self.base_xy[1]),
            DELIVERY_TABLE_COSTMAP_BOUNDS)
        required = WHOLE_BODY_KEEP_OUT_RADIUS + PLACE_CLEAR_TABLE_MARGIN_M
        self.get_logger().info(
            "[place] vertical raise complete; holding arm, slide and "
            "open-gripper "
            "commands unchanged "
            "until the chassis is fully clear of the table "
            f"(clearance={self._place_retreat_start_clearance:.3f}m, "
            f"required={required:.3f}m)")

    def _start_place_arm_recovery(self, clearance: float) -> None:
        """Recover the whole arm only after horizontal retreat completes."""
        if self._place_retreat_sent:
            return
        self._place_retreat_sent = True
        self.place_t0 = self.now()
        self.commands_ready_since = None
        self._command_initial_arm_posture()
        self.get_logger().info(
            "[place] horizontal retreat complete and base stopped; now "
            f"restoring both arms, grippers, slide and head "
            f"(clearance={clearance:.3f}m)")

    def _tcp_above_product_center(self) -> float:
        """TCP height above the held product centre measured at grasp time.

        ``forward_contact_world`` is the measured TCP position when the arm
        first contacted the product; the dual-arm equivalent is
        ``dual_contact_tcp_z``.  Long goods and top-shelf picks hold the wrist
        well above the centre, so this offset keeps the product bottom -- not
        the wrist -- at the configured table clearance during placement.
        """
        if self.target_world is None:
            return 0.0
        if (self.use_dual_tissue_grasp
                and self.dual_contact_tcp_z is not None):
            return (float(self.dual_contact_tcp_z)
                    - float(self.target_world[2]))
        if self.forward_contact_world is not None:
            return (float(self.forward_contact_world[2])
                    - float(self.target_world[2]))
        return 0.0

    def _place_tcp_offset(self) -> float:
        """放置时 TCP 高出商品中心的高度。

        heweidao 在放置时把夹爪绕腕轴旋转 180°，TCP 相对商品中心的 z 偏移
        随之符号反转（抓取时下移 1cm 夹窄处，旋转后变成上移 1cm）。
        """
        offset = self._tcp_above_product_center()
        if self.target_kind == "heweidao":
            offset = -offset
        return offset

    def _product_release_z(self) -> float:
        """Return TCP height that leaves the product just above the table.

        The TCP does not always pass through the product centre.  Short top
        goods, for example, deliberately use a higher wrist pose to clear the
        shelf.  Preserve that grasp-time vertical offset so the product bottom
        -- rather than the wrist -- receives the configured table clearance.
        """
        half_height = self._placement_product_half_height()
        bottom_clearance = PLACE_PRODUCT_BOTTOM_CLEARANCE_BY_KIND_M.get(
            self.target_kind, PLACE_PRODUCT_BOTTOM_CLEARANCE_M)
        return (
            DELIVERY_TABLE_TOP_Z_M
            + half_height
            + bottom_clearance
            + self._place_tcp_offset())

    def _placement_product_half_height(self) -> float:
        """Return geometry used only by the delivery-table placement flow."""
        if self.target_kind == "heweidao":
            return HEWEIDAO_PLACE_HALF_HEIGHT_M
        return PRODUCT_HALF_HEIGHT_M[self.target_kind]

    def _product_bottom_at_table(self, tcp: np.ndarray | None) -> bool:
        """True when the held product's bottom is at/near the table surface.

        商品底部世界 z = TCP z − 抓取时 TCP 高出商品中心的高度 − 商品半高。
        盒装/长商品使用 [桌面−5mm, 桌面+20mm]，球形商品把上界收紧到
        +8mm，避免应急释放仍产生足以反弹滚动的自由落差。判定覆盖两条路径：
        * 正常到位：商品底部悬空 10mm（PLACE_PRODUCT_BOTTOM_CLEARANCE_M）；
        * 触桌接触：长商品/夹持偏低导致商品底部先碰桌面、slide 被顶住，
          TCP 高于标称 release_z，但商品底部已在桌面。
        """
        if tcp is None or np.asarray(tcp).shape != (3,):
            return False
        if not np.all(np.isfinite(tcp)):
            return False
        half_height = self._placement_product_half_height()
        bottom_z = (
            float(tcp[2])
            - self._place_tcp_offset()
            - half_height)
        high_tolerance = PLACE_CONTACT_BOTTOM_HIGH_TOL_BY_KIND_M.get(
            self.target_kind, PLACE_CONTACT_BOTTOM_HIGH_TOL_M)
        return (
            DELIVERY_TABLE_TOP_Z_M - PLACE_CONTACT_BOTTOM_LOW_TOL_M
            <= bottom_z
            <= DELIVERY_TABLE_TOP_Z_M + high_tolerance)

    def _place_slide_stalled(self, now: float) -> bool:
        """True once the slide is physically blocked while commanded to move.

        位置控制下物理阻挡（商品底部触桌顶住 slide，或夹爪/摆臂压着桌面）表现为：
        命令继续朝目标走、反馈停住、误差不再缩小，同时 slide 力矩大幅饱和
        （实测卡死 ≈ -306 N·m，正常运动/静止 ≈ 0）。窗口内误差改善 < 2 mm 且
        力矩超过阈值视为被顶住。与命令误差大小无关——kele/zhijin 实测都卡在
        6 mm 左右，旧实现按"接近到位"放过导致 30 s 硬超时。

        防误判保护：
        * slide 仍在移动（速度 > 2 mm/s）→ 正常运动，不算停滞；
        * 命令发出不足 PLACE_STALL_CMD_MIN_AGE_S（大行程起步期力矩饱和且
          位置短暂不动，实测 1.3 s 内被误判过）→ 不算停滞；
        * 已到位（容差内）→ 不算停滞；力矩未饱和 → 没有被顶住。
        """
        measured_slide = self.joints.get("slide_joint")
        effort = self.joint_efforts.get("slide_joint")
        slide_vel = self.joint_velocities.get("slide_joint")
        if (measured_slide is None or effort is None or slide_vel is None
                or not math.isfinite(float(measured_slide))
                or not math.isfinite(float(effort))
                or not math.isfinite(float(slide_vel))):
            return False
        error = abs(float(measured_slide) - self.des_slide)
        # 已到位（容差内）不算停滞；力矩未饱和说明没有被顶住。
        if error <= PLACE_SLIDE_SETTLE_TOLERANCE_M:
            self._place_slide_stall_snapshot = None
            return False
        if abs(float(effort)) < PLACE_SLIDE_STALL_EFFORT_NM:
            self._place_slide_stall_snapshot = None
            return False
        # slide 正在移动 → 正常运动；命令发出太短 → 还在起步。
        if abs(float(slide_vel)) > PLACE_SLIDE_STALL_VEL_MPS:
            self._place_slide_stall_snapshot = None
            return False
        if now - self.place_t0 < PLACE_STALL_CMD_MIN_AGE_S:
            self._place_slide_stall_snapshot = None
            return False
        if self._place_slide_stall_snapshot is None:
            self._place_slide_stall_snapshot = (now, error)
            return False
        start_time, start_error = self._place_slide_stall_snapshot
        if start_error - error >= PLACE_CONTACT_STALL_IMPROVEMENT_M:
            # 仍在下降：刷新快照，继续观察。
            self._place_slide_stall_snapshot = (now, error)
            return False
        if now - start_time >= PLACE_CONTACT_STALL_S:
            reference = self._held_product_reference_world()
            verified_table_contact = bool(
                self._place_timeout_fallback_safe(reference)
                and self._product_bottom_at_table(reference))
            if verified_table_contact:
                return True
            # High effort can also mean that an arm link or gripper body hit
            # the table edge while the product itself is still high.  Never
            # turn that collision into an intentional high drop.  Keep the
            # grip closed and let the bounded stage timeout select the safe
            # vertical/fatal recovery path instead.
            if now - self._place_stall_warn_log >= 1.0:
                self._place_stall_warn_log = now
                self.get_logger().warn(
                    "[place] slide stall rejected: product contact geometry "
                    "is not verified; keeping grip closed "
                    f"reference={self._rounded_list(reference)} "
                    f"table={int(self._tcp_over_delivery_table(reference))} "
                    f"slot={int(self._tcp_at_assigned_slot(reference))} "
                    f"fallback_safe="
                    f"{int(self._place_timeout_fallback_safe(reference))} "
                    f"bottom={int(self._product_bottom_at_table(reference))}")
            return False
        return False

    def _place_contact_release(self, now: float, tcp: np.ndarray | None) -> None:
        """Stop the descent and release in place once goods touch the table.

        把 arm 和 des_slide 都锁回当前反馈值：停止下压，也停止尚未完全收敛
        的横向臂运动，避免松爪期间继续推商品。随后进入 stage 3 的原地释放
        流程；tcp 仅用于日志，可能为 None（双臂流程或反馈缺失时）。
        """
        measured_slide = self.joints.get("slide_joint")
        if measured_slide is not None and math.isfinite(float(measured_slide)):
            self.des_slide = float(measured_slide)
        if self.use_dual_tissue_grasp:
            measured_left = self.arm_positions("left")
            measured_right = self.arm_positions("right")
            if (np.all(np.isfinite(measured_left))
                    and np.all(np.isfinite(measured_right))):
                self.des_left_arm = measured_left.copy()
                self.des_right_arm = measured_right.copy()
        else:
            measured_arm = self.selected_arm_positions()
            if np.all(np.isfinite(measured_arm)):
                self.place_arm_joints = measured_arm.copy()
                self.set_selected_arm_target(measured_arm)
        self.commands_ready_since = None
        self.place_stage = 3
        self.place_t0 = now
        self._place_release_started_at = now
        self._place_slide_stall_snapshot = None
        self.get_logger().warn(
            "[place] verified low table release; locking the measured arms, "
            "stopping descent and releasing in place tcp="
            f"{None if tcp is None else np.round(tcp, 3)}")

    def _projected_place_creep_pose(self) -> tuple[np.ndarray, float]:
        """Predict the deterministic base pose at the end of final creep."""
        if self.place_creep_start_y is None:
            self.place_creep_start_y = float(self.base_xy[1])
        distance_cap_y = (
            float(self.place_creep_start_y) - float(self.place_creep_m))
        slot_stop_y = (
            float(self.place_world[1])
            + PLACE_BASE_TO_SLOT_LONGITUDINAL_M
            + PLACE_CREEP_GOAL_TOLERANCE_M)
        # Facing south, the first (larger) Y threshold reached terminates the
        # creep.  X is not commanded during this short segment.
        projected_xy = np.array([
            float(self.base_xy[0]), max(distance_cap_y, slot_stop_y)],
            dtype=float)
        return projected_xy, -math.pi / 2.0

    def _command_loaded_place_arm(
            self, now: float, *, projected_base_pose: tuple | None,
            phase_label: str) -> bool:
        """Solve and command the loaded single arm while slide stays high."""
        self.place_arm_joints = self._compute_place_arm_joints(
            projected_base_pose=projected_base_pose)
        if self.place_arm_joints is None:
            return False
        self.des_slide = self._transit_slide_target()
        self.set_selected_arm_target(self.place_arm_joints)
        self._place_slide_target_sent = False
        self.place_t0 = now
        self._place_stage0_wait_log = 0.0
        self._place_approach_best_error = float("inf")
        self._place_approach_best_error_at = now
        if not self._place_arm_target_sent:
            self._place_loaded_arm_step_rad = 0.0
        self._place_arm_target_sent = True
        loaded_arm_max_step = PLACE_LOADED_ARM_MAX_STEP_BY_KIND_RAD.get(
            self.target_kind, PLACE_LOADED_ARM_MAX_STEP_RAD)
        self.get_logger().info(
            f"[place] {phase_label}; slide remains at transport height; "
            f"loaded-arm soft start max_step={loaded_arm_max_step:.4f}rad/"
            "tick")
        self.get_logger().info(
            "[place-joints] motion_start="
            + json.dumps(
                self._place_joint_snapshot(), ensure_ascii=False,
                separators=(",", ":")))
        return True

    def _maybe_start_place_creep_preposition(self, now: float) -> None:
        """Start single-arm overhead positioning during the final creep."""
        if (self.use_dual_tissue_grasp
                or self.place_creep_done
                or self._place_creep_preposition_attempted):
            return
        self._place_creep_preposition_attempted = True
        projected_pose = self._projected_place_creep_pose()
        if self._command_loaded_place_arm(
                now, projected_base_pose=projected_pose,
                phase_label="parallel arm pre-position started during creep"):
            self._place_creep_preposition_started = True
            self.get_logger().info(
                "[place] projected creep endpoint="
                f"({projected_pose[0][0]:.3f},"
                f"{projected_pose[0][1]:.3f},"
                f"{math.degrees(projected_pose[1]):.1f}°); "
                "base and loaded arm may now move in parallel")
            return

        # A projected-pose miss is not a placement failure.  The measured
        # settled pose remains authoritative and gets the normal IK attempt.
        self._place_ik_attempted = False
        self.place_arm_joints = None
        self.get_logger().warn(
            "[place] projected creep-end IK unavailable; deferring arm "
            "positioning until measured base settle")

    def _finalize_place_creep_preposition(self, now: float) -> None:
        """Re-solve against settled odometry before any slide descent."""
        if (not self._place_creep_preposition_started
                or self._place_creep_preposition_finalized):
            return
        self._place_ik_attempted = False
        self.place_arm_joints = None
        if not self._command_loaded_place_arm(
                now, projected_base_pose=None,
                phase_label=(
                    "parallel pre-position refined at measured base pose")):
            raise RuntimeError(
                "place IK failed at settled base after creep pre-position; "
                "refusing to release goods off-table")
        self._place_creep_preposition_finalized = True

    def _place_parallel_slide_safe(self, arm_error: float) -> bool:
        """Whether the near-final arm is in a safe vertical descent corridor."""
        if (not math.isfinite(float(arm_error))
                or arm_error > PLACE_PARALLEL_SLIDE_ARM_ERROR_RAD):
            return False
        reference = self._held_product_reference_world()
        if (reference is None
                or not self._tcp_over_delivery_table(reference)):
            return False
        bottom_z = (
            float(reference[2])
            - self._place_tcp_offset()
            - self._placement_product_half_height())
        return bool(
            math.isfinite(bottom_z)
            and bottom_z
            >= (DELIVERY_TABLE_TOP_Z_M
                + PLACE_PARALLEL_SLIDE_MIN_BOTTOM_CLEARANCE_M))

    def _compute_place_arm_joints(
            self, *, projected_base_pose: tuple | None = None
    ) -> np.ndarray | None:
        """Solve an approach pose with enough slide travel for a low release.

        The numeric IK depends heavily on the reference joints.  Start from
        the measured loaded-arm branch, retain pregrasp and compact only as
        fallbacks, then use feedback-driven XY refinement above the table.
        The final vertical descent keeps the arm joints fixed and increases
        the downward-facing slide joint.  The result (including failure) is
        cached to avoid per-tick recomputation.
        """
        if self._place_ik_attempted:
            return self.place_arm_joints
        self._place_ik_attempted = True

        measured = self.selected_arm_positions()
        compact = np.asarray(
            PLACE_RETREAT_ARM_R if self.grasp_arm == "r"
            else PLACE_RETREAT_ARM_L, dtype=float)
        # The loaded arm must stay on the branch nearest its actual starting
        # posture.  Compact remains a last-resort fallback, never the first
        # solution accepted for a held product.
        refs = [("measured", measured)]
        if self.pregrasp_arm_joints is not None:
            refs.append((
                "pregrasp",
                np.asarray(self.pregrasp_arm_joints, dtype=float)))
        refs.append(("compact", compact))

        # The formal runner assigns an absolute table slot.  Manual runs use
        # --place-x/--place-y in exactly the same way.  Small bounded nudges
        # are IK fallbacks only; they cannot move a product into another slot.
        target_x = float(self.place_world[0])
        target_y = float(self.place_world[1])
        xy_candidates = (
            (target_x, target_y),
            (target_x, target_y + PLACE_SLOT_IK_NUDGE_M),
            (target_x, target_y - PLACE_SLOT_IK_NUDGE_M),
            (target_x + PLACE_SLOT_IK_NUDGE_M, target_y),
            (target_x - PLACE_SLOT_IK_NUDGE_M, target_y),
        )
        release_z = self._product_release_z()
        minimum_approach_z = max(
            self.place_min_approach_z,
            release_z + PLACE_APPROACH_CLEARANCE_M)
        z_candidates = tuple(
            minimum_approach_z + offset for offset in (0.0, 0.02, 0.04))
        # Top-shelf grasps pin the slide at SLIDE_MIN, which leaves the arm too
        # high to reach the table; raising the slide lowers the whole arm into
        # reach.  Middle/lower grasps keep their grasp slide.
        slide_candidates = []
        for slide in (self.slide_grasp, 0.20, 0.30, 0.35, 0.40, 0.45):
            slide = float(np.clip(slide, pick.SLIDE_MIN, pick.SLIDE_MAX))
            if not any(abs(slide - item) < 1e-6
                       for item in slide_candidates):
                slide_candidates.append(slide)

        for x, y in xy_candidates:
            for z in z_candidates:
                descent = z - release_z
                for slide in slide_candidates:
                    release_slide = slide + descent
                    if release_slide > pick.SLIDE_MAX + 1e-6:
                        continue
                    world = np.array([x, y, z], dtype=float)
                    for ref_source, ref in refs:
                        joints = self._solve_place_world(
                            world, ref, slide,
                            projected_base_pose=projected_base_pose)
                        if joints is None:
                            continue
                        comparison = {}
                        for other_source, other_ref in refs:
                            if other_source == ref_source:
                                other_joints = joints
                            else:
                                other_joints = self._solve_place_world(
                                    world, other_ref, slide,
                                    projected_base_pose=projected_base_pose)
                            if other_joints is None:
                                comparison[other_source] = None
                                continue
                            comparison[other_source] = {
                                "target_joints": self._rounded_list(
                                    other_joints),
                                "max_delta_from_measured": round(float(
                                    np.max(np.abs(other_joints - measured))),
                                    4),
                                "delta_from_measured": self._rounded_list(
                                    other_joints - measured),
                            }
                        self.place_approach_world = world.copy()
                        self.place_arm_joints = joints
                        self.place_slide_cmd = slide
                        self.place_release_world = np.array(
                            [target_x, target_y, release_z], dtype=float)
                        self.place_release_slide_cmd = release_slide
                        self.place_ik_ref_source = ref_source
                        self.place_ik_reference_joints = ref.copy()
                        self.get_logger().info(
                            f"[place] approach IK={np.round(world, 3)} "
                            f"release={np.round(self.place_release_world, 3)} "
                            f"slide={slide:.3f}->{release_slide:.3f} "
                            f"descent={descent:.3f}m "
                            f"slot={None if self.place_slot is None else self.place_slot + 1} "
                            f"ref_source={ref_source}")
                        self.get_logger().info(
                            "[place-joints] ik_selection="
                            + json.dumps({
                                "world": self._rounded_list(world),
                                "slide": round(float(slide), 4),
                                "release_slide": round(
                                    float(release_slide), 4),
                                "selected_ref_source": ref_source,
                                "measured_at_solve": self._rounded_list(
                                    measured),
                                "selected_reference": self._rounded_list(
                                    ref),
                                "selected_target": self._rounded_list(
                                    joints),
                                "selected_delta_from_measured": (
                                    self._rounded_list(joints - measured)),
                                "selected_max_delta_from_measured": round(
                                    float(np.max(np.abs(
                                        joints - measured))), 4),
                                "candidate_by_reference": comparison,
                                "measured_slide_at_solve": self.joints.get(
                                    "slide_joint"),
                                "commanded_slide_at_solve": round(
                                    float(self.cmd_slide), 4),
                                "grip_command": self._transport_grip_command,
                                "measured_grip": (
                                    self.selected_gripper_position()),
                                "projected_base_pose": (
                                    None if projected_base_pose is None
                                    else self._rounded_list(np.concatenate((
                                        np.asarray(
                                            projected_base_pose[0],
                                            dtype=float),
                                        [float(projected_base_pose[1])])))),
                            }, ensure_ascii=False, separators=(",", ":")))
                        return joints

        self.get_logger().error(
            "[place] no approach IK with enough downward slide travel; "
            "keeping gripper closed")
        return None

    def _solve_place_world(
            self, world: np.ndarray, reference: np.ndarray,
            slide: float, *, projected_base_pose: tuple | None = None
    ) -> np.ndarray | None:
        """Solve the selected arm to ``world`` at a given slide height."""
        target = np.eye(4)
        if self.target_kind == "heweidao":
            # heweidao 放置：夹爪绕左右轴(x)旋转 180°，让锥形杯上下颠倒
            # ——宽口朝下接触桌面，松爪后夹爪直接从窄底脱离，不再需要
            # "开爪后底盘水平抽离"。(绕 y 轴腕轴翻滚会导致放置 IK 无解)
            target[:3, :3] = pick.Rotation.from_euler("x", math.pi).as_matrix()
        if projected_base_pose is None:
            target[:3, 3] = self.world_to_footprint(world)
        else:
            projected_xy, projected_yaw = projected_base_pose
            delta = np.asarray(world, dtype=float) - np.array([
                float(projected_xy[0]), float(projected_xy[1]), 0.0])
            cosine = math.cos(-float(projected_yaw))
            sine = math.sin(-float(projected_yaw))
            target[:3, 3] = np.array([
                cosine * delta[0] - sine * delta[1],
                sine * delta[0] + cosine * delta[1],
                delta[2],
            ])
        reference = np.asarray(reference, dtype=float)
        ref_with_slide = np.concatenate(([slide], reference))
        try:
            if self.grasp_arm == "r":
                solutions = self.kdl.inverse_kinematics(
                    T_right=target, target_height=slide,
                    ref_pos=ref_with_slide)
            else:
                solutions = self.kdl.inverse_kinematics(
                    T_left=target, target_height=slide,
                    ref_pos=ref_with_slide)
        except Exception:  # noqa: BLE001 - try next candidate
            return None
        if solutions is None or len(solutions) == 0:
            return None
        candidates = [np.asarray(item[1:], dtype=float) for item in solutions]
        return min(
            candidates,
            key=lambda item: float(np.max(np.abs(item - reference))))

    def _laser_front_range(self) -> float | None:
        msg = self.laser_msg
        if msg is None or not msg.ranges:
            return None
        n = len(msg.ranges)
        half = max(1, int(round(n * 0.10)))
        centre = n // 2
        window = msg.ranges[max(0, centre - half):centre + half]
        valid = [float(r) for r in window
                 if r is not None and math.isfinite(float(r))
                 and r > 0.05 and r < 8.0]
        return min(valid) if valid else None

    def _advance_place_creep(self) -> bool:
        """Move slowly from the navigation endpoint to the arm-place pose."""
        if self.place_creep_done:
            return True
        if self.place_creep_start_y is None:
            self.place_creep_start_y = float(self.base_xy[1])

        front = self._laser_front_range()
        crept = float(self.place_creep_start_y - self.base_xy[1])
        slot_base_goal_y = float(
            self.place_world[1] + PLACE_BASE_TO_SLOT_LONGITUDINAL_M)
        slot_reached = (
            float(self.base_xy[1])
            <= slot_base_goal_y + PLACE_CREEP_GOAL_TOLERANCE_M)
        distance_reached = crept >= self.place_creep_m
        front_reached = (
            front is not None and front <= PLACE_CREEP_FRONT_STOP_M)
        if (self.place_creep_m <= 1e-4 or slot_reached
                or distance_reached or front_reached):
            self.set_twist(0.0, 0.0)
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0
            self.place_creep_done = True
            reason = (
                "slot_depth" if slot_reached else
                "distance_cap" if distance_reached else
                "lidar" if front_reached else "disabled")
            self.get_logger().info(
                f"[place] final approach finished (reason={reason} "
                f"crept={crept:.3f}m front={front} "
                f"slot_base_goal_y={slot_base_goal_y:.3f} "
                f"base=({self.base_xy[0]:.3f},{self.base_xy[1]:.3f}))")
            return True

        yaw_err = pick.wrap_to_pi(-math.pi / 2.0 - self.base_yaw)
        angular = float(np.clip(
            PLACE_CREEP_YAW_GAIN * yaw_err,
            -PLACE_CREEP_MAX_ANGULAR_RPS,
            PLACE_CREEP_MAX_ANGULAR_RPS))
        # Pause translation if odometry shows an unexpectedly large yaw error;
        # correct it first so a nominally southward creep cannot cut sideways.
        creep_speed = PLACE_CREEP_SPEED_BY_KIND_MPS.get(
            self.target_kind, PLACE_CREEP_SPEED_MPS)
        linear = creep_speed if abs(yaw_err) <= 0.10 else 0.0
        self.set_twist(linear, angular)
        return False

    @staticmethod
    def _tcp_over_delivery_table(tcp: np.ndarray | None) -> bool:
        """Require the measured release point to lie inside the tabletop."""
        if tcp is None or np.asarray(tcp).shape != (3,):
            return False
        x_min, y_min, x_max, y_max = DELIVERY_TABLE_XML_BOUNDS
        margin = PLACE_RELEASE_TABLE_MARGIN_M
        return (
            np.all(np.isfinite(tcp))
            and x_min + margin <= float(tcp[0]) <= x_max - margin
            and y_min + margin <= float(tcp[1]) <= y_max - margin)

    @staticmethod
    def _tcp_over_delivery_table_official(tcp: np.ndarray | None) -> bool:
        """Whether the reference point lies above the tabletop per the
        official referee ``delivery_box`` (referee.py), which has NO inset
        margin: x ∈ [-2.42, -1.46], y ∈ [-3.63, -3.19].

        Z is intentionally left unconstrained because this helper is used only
        to decide whether an empty wrist must clear vertically before neutral
        recovery.  Delivery still requires the conservative inset, assigned
        slot and low product-bottom checks in ``_monitor_held_product``.
        """
        if tcp is None or np.asarray(tcp).shape != (3,):
            return False
        x_min, y_min, x_max, y_max = DELIVERY_TABLE_XML_BOUNDS
        return (
            np.all(np.isfinite(tcp))
            and x_min <= float(tcp[0]) <= x_max
            and y_min <= float(tcp[1]) <= y_max)

    def _tcp_at_assigned_slot(self, tcp: np.ndarray | None) -> bool:
        """Require the measured release XY to remain near its own slot."""
        if tcp is None or np.asarray(tcp).shape != (3,):
            return False
        error = np.asarray(tcp[:2], dtype=float) - self.place_world[:2]
        return bool(
            np.all(np.isfinite(error))
            and np.linalg.norm(error) <= PLACE_SLOT_XY_TOLERANCE_M)

    def _dual_release_world(self) -> np.ndarray | None:
        """Approximate the held tissue centre by the two measured TCPs."""
        left = self.arm_tcp_world("left")
        right = self.arm_tcp_world("right")
        if left is None or right is None:
            return None
        return 0.5 * (np.asarray(left) + np.asarray(right))

    def _place_base_settled(self, now: float) -> bool:
        """Hold a zero base command briefly before starting placement."""
        self.set_twist(0.0, 0.0)
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        if self._place_base_settle_started_at is None:
            self._place_base_settle_started_at = now
            self._place_base_reference_xy = self.base_xy.copy()
            self._place_base_reference_yaw = float(self.base_yaw)
            self.get_logger().info(
                f"[place] base locked before overhead positioning; "
                f"settle={PLACE_BASE_SETTLE_S:.2f}s "
                f"reference=({self.base_xy[0]:.3f},"
                f"{self.base_xy[1]:.3f},"
                f"{math.degrees(self.base_yaw):.1f}deg)")
            return False
        return now - self._place_base_settle_started_at >= PLACE_BASE_SETTLE_S

    def _place_approach_stationary_ready(
            self, now: float, arm_error: float, slide_error: float) -> bool:
        """Accept a physically stationary loaded overhead pose.

        The arm effort limit can leave one joint just outside the strict
        command-settled tolerance even though all measured velocities are
        zero.  Waiting for that error to disappear cannot make progress and
        used to leave the final (fifth) delivery parked at the table.  Treat a
        short, stationary, low-error interval as ready for XY refinement;
        that stage works from measured TCP feedback and retains all release
        safety gates.
        """
        if (not self._place_slide_target_sent
                or now - self.place_t0 < PLACE_APPROACH_STATIONARY_MIN_AGE_S
                or not math.isfinite(float(arm_error))
                or float(arm_error) > PLACE_APPROACH_STATIONARY_ARM_ERROR_RAD
                or not math.isfinite(float(slide_error))
                or slide_error > PLACE_XY_STATIONARY_SLIDE_ERROR_M):
            return False
        sides = ("left",) if self.grasp_arm == "l" else ("right",)
        arm_velocities = [
            self.joint_velocities.get(f"{side}_arm_joint{index}")
            for side in sides for index in range(1, 7)]
        slide_velocity = self.joint_velocities.get("slide_joint")
        if (slide_velocity is None
                or any(value is None for value in arm_velocities)):
            return False
        return bool(
            max(abs(float(value)) for value in arm_velocities)
            <= PLACE_APPROACH_STATIONARY_ARM_RAD_S
            and abs(float(slide_velocity))
            <= PLACE_APPROACH_STATIONARY_SLIDE_VEL_MPS)

    def _place_refine_command_settled(
            self, now: float, *, dual: bool) -> tuple[bool, bool]:
        """Finish a refine step by target error or measured standstill.

        Returns ``(settled, residual_standstill)``.  A loaded joint can stop
        short when its effort limit is reached; the commanded pose is then
        unreachable and retrying cannot make progress.  Standstill is
        detected from the measured joint positions (negligible drift over a
        short window) instead of the velocity topic, which reports ±0.1
        rad/s on an effort-saturated but physically static joint.  When
        ``residual_standstill`` is True the caller should stop retrying and
        descend from the measured TCP if that pose is still slot-safe.
        """
        ready = (
            self.dual_commands_ready(
                arm_tolerance=PLACE_ARM_SETTLE_TOLERANCE_RAD,
                slide_tolerance=PLACE_SLIDE_SETTLE_TOLERANCE_M)
            if dual else self.commands_ready(
                arm_tolerance=PLACE_ARM_SETTLE_TOLERANCE_RAD,
                slide_tolerance=PLACE_SLIDE_SETTLE_TOLERANCE_M))
        if ready:
            self._place_refine_motion_stable_since = None
            self._place_refine_motion_anchor = None
            return True, False
        if (self._place_refine_target_sent_at is None
                or now - self._place_refine_target_sent_at
                < PLACE_XY_COMMAND_MIN_WAIT_S):
            self._place_refine_motion_stable_since = None
            self._place_refine_motion_anchor = None
            return False, False
        slide_velocity = self.joint_velocities.get("slide_joint")
        measured_slide = self.joints.get("slide_joint")
        if slide_velocity is None or measured_slide is None:
            self._place_refine_motion_stable_since = None
            self._place_refine_motion_anchor = None
            return False, False
        if (abs(float(slide_velocity)) > PLACE_XY_STATIONARY_SLIDE_MPS
                or abs(float(measured_slide) - self.des_slide)
                > PLACE_XY_STATIONARY_SLIDE_ERROR_M):
            self._place_refine_motion_stable_since = None
            self._place_refine_motion_anchor = None
            return False, False
        arm = (
            np.concatenate([
                self.arm_positions("left"),
                self.arm_positions("right")])
            if dual else self.selected_arm_positions())
        if not np.all(np.isfinite(arm)):
            self._place_refine_motion_stable_since = None
            self._place_refine_motion_anchor = None
            return False, False
        if self._place_refine_motion_anchor is None:
            self._place_refine_motion_anchor = arm.copy()
            self._place_refine_motion_stable_since = now
            return False, False
        drift = float(np.max(
            np.abs(arm - self._place_refine_motion_anchor)))
        if drift > PLACE_XY_STATIONARY_ARM_POS_M:
            self._place_refine_motion_anchor = arm.copy()
            self._place_refine_motion_stable_since = now
            return False, False
        if (now - self._place_refine_motion_stable_since
                < PLACE_XY_STATIONARY_SETTLE_S):
            return False, False
        residual = self.dual_arm_error() if dual else self.selected_arm_error()
        self.get_logger().warn(
            "[place] refine actuator stationary with residual joint error; "
            "stopping retry and using the measured TCP "
            f"(dual={int(dual)} residual={residual:.4f}rad "
            f"slide_error="
            f"{abs(float(measured_slide) - self.des_slide):.4f}m)")
        self._place_refine_motion_stable_since = None
        self._place_refine_motion_anchor = None
        return True, True

    def _place_timeout_fallback_safe(
            self, reference: np.ndarray | None) -> bool:
        """Whether a loaded pose may descend in place after refinement fails."""
        if reference is None or np.asarray(reference).shape != (3,):
            return False
        reference = np.asarray(reference, dtype=float)
        if not np.all(np.isfinite(reference)):
            return False
        error_xy = reference[:2] - self.place_world[:2]
        return bool(
            self._tcp_over_delivery_table(reference)
            and np.linalg.norm(error_xy)
            <= PLACE_XY_TIMEOUT_FALLBACK_TOLERANCE_M)

    def _place_base_diagnostic(self) -> dict:
        """Report odometry displacement while the physical base is held."""
        delta_xy = None
        delta_yaw = None
        if self._place_base_reference_xy is not None:
            delta_xy = self.base_xy - self._place_base_reference_xy
        if self._place_base_reference_yaw is not None:
            delta_yaw = pick.wrap_to_pi(
                self.base_yaw - self._place_base_reference_yaw)
        return {
            "base_xy": self._rounded_list(self.base_xy),
            "base_delta_xy": self._rounded_list(delta_xy),
            "base_delta_yaw": (
                None if delta_yaw is None else round(float(delta_yaw), 4)),
            "base_command": [
                round(float(self.cmd_linear), 4),
                round(float(self.cmd_angular), 4),
            ],
        }

    def _send_single_place_refine_step(
            self, tcp: np.ndarray, error_xy: np.ndarray) -> None:
        """Send one bounded horizontal correction at the measured height."""
        measured_slide = self.joints.get("slide_joint")
        if measured_slide is None:
            raise RuntimeError("slide feedback unavailable during place refine")
        error_norm = float(np.linalg.norm(error_xy))
        step_xy = error_xy.copy()
        if error_norm > PLACE_XY_REFINE_STEP_M:
            step_xy *= PLACE_XY_REFINE_STEP_M / error_norm
        target_world = np.asarray(tcp, dtype=float).copy()
        target_world[:2] += step_xy
        measured = self.selected_arm_positions()
        joints = self._solve_place_world(
            target_world, measured, float(measured_slide))
        if joints is None:
            raise RuntimeError(
                "horizontal place refinement IK failed; keeping goods clamped "
                f"tcp={np.round(tcp, 3)} step={np.round(step_xy, 3)}")
        self.place_arm_joints = joints
        self.place_ik_ref_source = "measured_refine"
        self.place_ik_reference_joints = measured.copy()
        self.set_selected_arm_target(joints)
        self.des_slide = float(measured_slide)
        self.commands_ready_since = None
        self._place_loaded_arm_step_rad = 0.0
        self._place_refine_target_sent = True
        self._place_refine_target_sent_at = self.now()
        self._place_refine_motion_stable_since = None
        self._place_refine_motion_anchor = None
        self._place_refine_iterations += 1
        self.get_logger().info(
            "[place] horizontal refine step "
            f"iteration={self._place_refine_iterations} "
            f"error={np.round(error_xy, 4)}m "
            f"step={np.round(step_xy, 4)}m "
            f"height={float(tcp[2]):.3f}m")

    def _begin_single_place_descent(
            self, now: float, tcp: np.ndarray) -> None:
        """Freeze the arm and lower only the slide to the release height."""
        measured_slide = self.joints.get("slide_joint")
        if measured_slide is None or self.place_release_world is None:
            raise RuntimeError("release geometry unavailable before descent")
        # Freeze the exact measured arm configuration.  From this point until
        # release, only the slide may change so the carried product keeps its
        # measured orientation and horizontal pose.
        measured_arm = self.selected_arm_positions()
        if not np.all(np.isfinite(measured_arm)):
            raise RuntimeError("arm feedback unavailable before descent")
        self.place_arm_joints = measured_arm.copy()
        self.set_selected_arm_target(measured_arm)
        target_z = float(self.place_release_world[2])
        target_slide = float(measured_slide) + (float(tcp[2]) - target_z)
        if not pick.SLIDE_MIN <= target_slide <= pick.SLIDE_MAX:
            raise RuntimeError(
                "safe release height is outside slide range: "
                f"current={float(measured_slide):.3f} "
                f"target={target_slide:.3f} tcp_z={float(tcp[2]):.3f} "
                f"release_z={target_z:.3f}")
        self.place_release_slide_cmd = target_slide
        self.des_slide = target_slide
        self.commands_ready_since = None
        self._place_slide_stall_snapshot = None
        self.place_stage = 2
        self.place_t0 = now
        self.get_logger().info(
            f"[place] horizontal target settled; descending vertically "
            f"with measured arm pose locked, tcp={np.round(tcp, 3)} "
            f"slide={float(measured_slide):.3f}->{target_slide:.3f}")

    def _begin_dual_place_descent(
            self, now: float, release_world: np.ndarray) -> None:
        """Lock both loaded arms and lower tissue with the common slide."""
        measured_slide = self.joints.get("slide_joint")
        if measured_slide is None:
            raise RuntimeError("slide feedback unavailable before dual descent")
        target_z = self._product_release_z()
        target_slide = float(measured_slide) + (
            float(release_world[2]) - target_z)
        if not pick.SLIDE_MIN <= target_slide <= pick.SLIDE_MAX:
            raise RuntimeError(
                "dual-arm safe release is outside slide range: "
                f"target={target_slide:.3f}")
        measured_left = self.arm_positions("left")
        measured_right = self.arm_positions("right")
        if (not np.all(np.isfinite(measured_left))
                or not np.all(np.isfinite(measured_right))):
            raise RuntimeError(
                "dual-arm feedback unavailable before descent")
        self.dual_release_slide_cmd = target_slide
        self.des_left_arm = measured_left.copy()
        self.des_right_arm = measured_right.copy()
        self.des_slide = target_slide
        self.commands_ready_since = None
        self._place_slide_stall_snapshot = None
        self._dual_descent_sent = True
        self.place_stage = 2
        self.place_t0 = now
        self.get_logger().info(
            f"[place-dual] horizontal target settled; descending "
            "vertically with measured arm poses locked "
            f"centre={np.round(release_world, 3)} "
            f"slide={float(measured_slide):.3f}->{target_slide:.3f}")

    def _start_dual_place_release_spread(self, now: float) -> None:
        """Reverse the measured tissue squeeze while keeping both jaws shut.

        抓取时以双侧实测接触位置为锚，每侧继续向内预压
        ``dual_squeeze_m``。放置时严格反向：从当前实测 TCP 出发，每侧仅
        沿当前双臂 TCP 的水平连线向外移动同样距离。保留当前 TCP 姿态
        以及各自的 Z，不再张爪，也不再移动到固定 0.150m 半跨度；纸巾
        卸载后再垂直抬升和后退。
        """
        left_tcp = self.arm_tcp_world("left")
        right_tcp = self.arm_tcp_world("right")
        left_reference = self.arm_positions("left")
        right_reference = self.arm_positions("right")
        measured_slide = self.joints.get("slide_joint")
        if (left_tcp is None or right_tcp is None
                or left_reference is None or right_reference is None
                or measured_slide is None):
            raise RuntimeError(
                "dual release spread lacks arm/slide feedback")

        release_outward_m = float(self.dual_squeeze_m)
        # 抓取货架时底盘朝北，放置时底盘朝南，左右臂在世界 X 轴上的
        # 大小关系会反转。不能固定用“左减 X、右加 X”判断外侧；应以
        # 当前左右 TCP 的水平连线为准，保证两臂间距确实增大。
        release_axis_xy = (
            np.asarray(right_tcp[:2], dtype=float)
            - np.asarray(left_tcp[:2], dtype=float))
        release_axis_norm = float(np.linalg.norm(release_axis_xy))
        if (not math.isfinite(release_axis_norm)
                or release_axis_norm <= 1e-6):
            raise RuntimeError(
                "dual release spread has invalid measured TCP separation")
        release_axis_xy /= release_axis_norm

        left_goal = np.asarray(left_tcp, dtype=float).copy()
        right_goal = np.asarray(right_tcp, dtype=float).copy()
        left_goal[:2] -= release_axis_xy * release_outward_m
        right_goal[:2] += release_axis_xy * release_outward_m

        # Preserve the exact measured endpoint orientations.  Starting from
        # identity rotations here can make the wrist/elbow joints unwind while
        # the intended release is only a short lateral inverse-squeeze.
        slide = float(measured_slide)
        reference = np.concatenate((
            [slide],
            np.asarray(left_reference, dtype=float),
            np.asarray(right_reference, dtype=float),
        ))
        left_target, right_target = self.kdl.forward_kinematics(reference)
        left_target = np.asarray(left_target, dtype=float).copy()
        right_target = np.asarray(right_target, dtype=float).copy()
        left_target[:3, 3] = self.world_to_footprint(left_goal)
        right_target[:3, 3] = self.world_to_footprint(right_goal)
        try:
            solutions = self.kdl.inverse_kinematics(
                T_left=left_target,
                T_right=right_target,
                target_height=slide,
                ref_pos=reference)
        except Exception as exc:  # noqa: BLE001 - report a safe place failure
            self.get_logger().error(
                f"[place-dual] release-spread IK raised: {exc}")
            raise RuntimeError("dual release spread IK failed") from exc
        if solutions is None or len(solutions) == 0:
            raise RuntimeError(
                "no IK solution for inverse-squeeze tissue release: "
                f"left={np.round(left_goal, 3)} "
                f"right={np.round(right_goal, 3)}")

        candidates = [
            np.asarray(item[1:], dtype=float) for item in solutions]
        arms_reference = reference[1:]
        best = min(
            candidates,
            key=lambda item: float(np.max(np.abs(item - arms_reference))))
        self.des_left_arm = best[:6].copy()
        self.des_right_arm = best[6:].copy()
        self.commands_ready_since = None
        self._dual_release_spread_sent = True
        self.place_t0 = now
        self.start_dual_tissue_motion(
            "release_open",
            self.des_left_arm,
            self.des_right_arm,
            release_outward_m,
            pick.DUAL_TISSUE_RELEASE_SPEED_MPS,
            self.state,
            require_convergence=True)
        self.get_logger().info(
            "[place-dual] reversing the grasp squeeze with grippers unchanged "
            f"outward={release_outward_m:.3f}m/side "
            f"axis_xy={np.round(release_axis_xy, 3)} "
            f"left={np.round(left_tcp, 3)}->{np.round(left_goal, 3)} "
            f"right={np.round(right_tcp, 3)}->{np.round(right_goal, 3)}")

    def _start_place_vertical_clear(self, now: float) -> None:
        """Raise vertically after release before arm or chassis retreat."""
        measured_slide = self.joints.get("slide_joint")
        if measured_slide is None:
            raise RuntimeError("slide feedback unavailable after release")
        target_slide = max(
            pick.SLIDE_MIN,
            float(measured_slide) - PLACE_VERTICAL_CLEARANCE_M)
        self.des_slide = target_slide
        self.commands_ready_since = None
        self.place_stage = 4
        self.place_t0 = now
        self.get_logger().info(
            f"[place] goods released; raising vertically before retreat "
            f"slide={float(measured_slide):.3f}->{target_slide:.3f}")

    def _start_heweidao_release_base_backup(self, now: float) -> None:
        """Back the fixed arm and open fingers away from supported heweidao."""
        measured_slide = self.joints.get("slide_joint")
        measured_arm = self.selected_arm_positions()
        if (measured_slide is None
                or not math.isfinite(float(measured_slide))
                or not np.all(np.isfinite(measured_arm))):
            raise RuntimeError(
                "heweidao release base backup lacks arm/slide feedback")

        self.place_arm_joints = measured_arm.copy()
        self.set_selected_arm_target(measured_arm)
        self.des_slide = float(measured_slide)
        self.commands_ready_since = None
        self._heweidao_release_phase = "base_backing"
        self._heweidao_release_phase_started_at = now
        self._heweidao_release_base_start_xy = self.base_xy.copy()
        self.get_logger().info(
            "[place-heweidao] product supported and gripper open; "
            "holding arm/slide fixed and backing chassis horizontally "
            f"distance={HEWEIDAO_RELEASE_BASE_BACKUP_DISTANCE_M:.3f}m "
            f"start_xy={np.round(self._heweidao_release_base_start_xy, 3)} "
            f"measured_grip={self.selected_gripper_position()}")

    def _heweidao_place_release_tick(self, now: float) -> None:
        """Open, back the fixed arm with the chassis, then raise vertically."""
        self._set_selected_grip(pick.GRIP_OPEN)
        if self._heweidao_release_phase is None:
            self._heweidao_release_phase = "opening"
            self._heweidao_release_phase_started_at = now
            self.get_logger().info(
                "[place-heweidao] table support verified; opening gripper "
                "before the fixed-arm chassis backup")
            return

        elapsed = now - self._heweidao_release_phase_started_at
        measured_grip = self.selected_gripper_position()
        grip_open = (
            measured_grip is not None
            and math.isfinite(float(measured_grip))
            and float(measured_grip) >= HEWEIDAO_RELEASE_GRIP_OPEN_MIN)

        if self._heweidao_release_phase == "opening":
            if elapsed < HEWEIDAO_RELEASE_OPEN_MIN_S:
                return
            if not grip_open and elapsed < HEWEIDAO_RELEASE_OPEN_TIMEOUT_S:
                return
            if not grip_open:
                self.get_logger().warn(
                    "[place-heweidao] gripper did not reach the open "
                    "threshold while stationary; starting the 100 mm chassis "
                    f"backup to clear the tapered shoulder grip="
                    f"{measured_grip}")
            self._start_heweidao_release_base_backup(now)
            return

        if self._heweidao_release_phase != "base_backing":
            raise RuntimeError(
                "invalid heweidao release phase "
                f"{self._heweidao_release_phase!r}")

        if self._heweidao_release_base_start_xy is None:
            raise RuntimeError(
                "heweidao release base backup lacks its odometry start pose")
        moved = float(np.linalg.norm(
            self.base_xy - self._heweidao_release_base_start_xy))
        if moved >= HEWEIDAO_RELEASE_BASE_BACKUP_DISTANCE_M:
            self.set_twist(0.0, 0.0)
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0
            if not grip_open:
                if elapsed < HEWEIDAO_RELEASE_BASE_BACKUP_TIMEOUT_S:
                    return
                raise RuntimeError(
                    "heweidao gripper did not verify open after the 100 mm "
                    f"chassis backup (measured_grip={measured_grip})")
            self.get_logger().info(
                "[place-heweidao] 100 mm chassis backup complete with "
                "arm/slide fixed; "
                f"moved={moved:.3f}m grip={float(measured_grip):.3f}; "
                "starting vertical clearance")
            self._start_place_vertical_clear(now)
            return
        if elapsed >= HEWEIDAO_RELEASE_BASE_BACKUP_TIMEOUT_S:
            self.set_twist(0.0, 0.0)
            raise RuntimeError(
                "heweidao chassis release backup did not reach "
                f"{HEWEIDAO_RELEASE_BASE_BACKUP_DISTANCE_M:.3f}m within "
                f"{HEWEIDAO_RELEASE_BASE_BACKUP_TIMEOUT_S:.1f}s "
                f"(moved={moved:.3f}m, measured_grip={measured_grip})")
        self.set_twist(-HEWEIDAO_RELEASE_BASE_BACKUP_SPEED_MPS, 0.0)

    def _place_vertical_clear_tick(self, now: float) -> None:
        """Complete post-release vertical clearance with the base locked."""
        self.set_twist(0.0, 0.0)
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        if self.use_dual_tissue_grasp:
            # The inverse-squeeze has already released the tissue.  Keep the
            # jaws unchanged while lifting so opening fingers cannot sweep the
            # box; they open only after the chassis is clear of the table.
            self._hold_grasp_during_transport()
        else:
            self.des_left_grip = pick.GRIP_OPEN
            self.des_right_grip = pick.GRIP_OPEN
        ready = (
            self.dual_commands_ready(
                arm_tolerance=0.05, slide_tolerance=0.020)
            if self.use_dual_tissue_grasp
            else self.commands_ready(
                arm_tolerance=0.05, slide_tolerance=0.020))
        if ready:
            self.get_logger().info(
                "[place] vertical clearance reached; arm/base retreat may begin")
            self._start_place_retreat(now)
            return
        if now - self.place_t0 >= PLACE_VERTICAL_CLEAR_TIMEOUT_S:
            raise RuntimeError(
                "post-release vertical clearance did not settle within "
                f"{PLACE_VERTICAL_CLEAR_TIMEOUT_S:.1f}s")

    def _place_tick(self) -> None:
        now = self.now()
        self._log_place_motion(now, self.place_stage)
        if self.place_stage == 5:
            self._clear_delivery_table_tick(now)
            return
        if self.use_dual_tissue_grasp:
            self._place_tick_dual(now)
            return

        if self.place_stage == 0:
            # Solve against the deterministic creep endpoint and begin the
            # loaded-arm motion while the chassis covers its final short
            # segment.  Slide remains at transport height throughout base
            # motion.  Settled odometry is authoritative and is re-solved
            # below before any descent.
            self._maybe_start_place_creep_preposition(now)
            if not self._advance_place_creep():
                return
            if not self._place_base_settled(now):
                return
            self._finalize_place_creep_preposition(now)
            if self.place_arm_joints is None:
                if not self._command_loaded_place_arm(
                        now, projected_base_pose=None,
                        phase_label=(
                            "loaded arm positioning started after base "
                            "settle")):
                    raise RuntimeError(
                        "place IK failed; refusing to release goods off-table")
            if (self._place_arm_target_sent
                    and not self._place_slide_target_sent):
                arm_error = self.selected_arm_error()
                # 阶段 1：只等臂到位。slide 命令尚未发出（des_slide 仍是
                # 运输值），此时按 commands_ready 检查 slide 误差会因旧
                # des_slide 与实测的差 > 容差而死锁（实测卡死：
                # arm_error=0.0072rad 但 slide=0.0121 永不过 4mm 门）。
                # slide 到位检查留给阶段 2（slide 命令发出之后）。
                parallel_slide_safe = self._place_parallel_slide_safe(
                    arm_error)
                if (arm_error <= PLACE_ARM_SETTLE_TOLERANCE_RAD
                        or parallel_slide_safe):
                    self.des_slide = float(self.place_slide_cmd)
                    self._place_slide_target_sent = True
                    self.commands_ready_since = None
                    self.place_t0 = now
                    self._place_approach_best_error = arm_error
                    self._place_approach_best_error_at = now
                    self._place_slide_stall_snapshot = None
                    if parallel_slide_safe:
                        self.get_logger().info(
                            "[place] arm entered verified table-overhead "
                            "corridor; moving arm and slide in parallel "
                            f"arm_error={arm_error:.4f}rad "
                            f"slide={self.cmd_slide:.3f}->"
                            f"{self.des_slide:.3f}")
                    else:
                        self.get_logger().info(
                            "[place] loaded arm pre-positioned at transport "
                            f"height; lowering slide vertically "
                            f"{self.cmd_slide:.3f}->{self.des_slide:.3f}")
                elif (now - self._place_stage0_wait_log
                        >= PLACE_APPROACH_PROGRESS_LOG_S):
                    self._place_stage0_wait_log = now
                    self.get_logger().info(
                        "[place] waiting for loaded arm pre-position "
                        f"arm_error={arm_error:.4f}rad "
                        f"slide={self.joints.get('slide_joint')}")
                return
            # 商品底部已触桌（slide 被顶住）：不必等 approach 到位，就地松爪。
            if self._place_slide_stalled(now):
                self.get_logger().warn(
                    "[place] slide blocked during overhead approach; "
                    "goods already at table height — releasing in place")
                self._place_contact_release(now, self.selected_tcp_world())
                return
            arm_error = self.selected_arm_error()
            measured_slide = self.joints.get("slide_joint")
            slide_error = (
                float("inf") if measured_slide is None
                else abs(float(measured_slide) - self.des_slide))
            approach_elapsed = (
                0.0 if not self._place_arm_target_sent
                else now - self.place_t0)
            converged = self.commands_ready(
                arm_tolerance=PLACE_ARM_SETTLE_TOLERANCE_RAD,
                slide_tolerance=PLACE_APPROACH_SLIDE_TOLERANCE_M)
            stationary_ready = (
                not converged
                and self._place_approach_stationary_ready(
                    now, arm_error, slide_error))
            if converged or stationary_ready:
                tcp = self.selected_tcp_world()
                if tcp is None:
                    return
                if stationary_ready:
                    self.get_logger().warn(
                        "[place] loaded arm reached a stationary overhead "
                        "pose just outside strict tolerance; continuing "
                        "with measured-pose XY refinement "
                        f"arm_error={arm_error:.4f}rad "
                        f"slide_error={slide_error:.4f}m")
                self.get_logger().info(
                    f"[place] goods reached overhead pose "
                    f"elapsed={approach_elapsed:.2f}s "
                    f"arm_error={arm_error:.4f}rad "
                    f"slide_error={slide_error:.4f}m tcp="
                    f"{None if tcp is None else np.round(tcp, 3)}; "
                    "starting fixed-height horizontal refinement")
                self.get_logger().info(
                    "[place-joints] approach_reached="
                    + json.dumps(
                        self._place_joint_snapshot(),
                        ensure_ascii=False, separators=(",", ":")))
                self.place_stage = 1
                self.place_t0 = now
                self._place_refine_started_at = now
                self._place_refine_target_sent = False
                self._place_refine_stable_since = None
                self._place_refine_motion_anchor = None
                self.commands_ready_since = None
            elif (self._place_arm_target_sent
                    and now - self._place_stage0_wait_log
                    >= PLACE_APPROACH_PROGRESS_LOG_S):
                self._place_stage0_wait_log = now
                self.get_logger().info(
                    f"[place] waiting for approach pose "
                    f"elapsed={approach_elapsed:.2f}s "
                    f"arm_error={arm_error:.4f}rad "
                    f"slide_error={slide_error:.4f}m")
                self.get_logger().info(
                    "[place-joints] approach_progress="
                    + json.dumps({
                        "elapsed": round(float(approach_elapsed), 3),
                        **self._place_joint_snapshot(),
                    }, ensure_ascii=False, separators=(",", ":")))
            if self._place_arm_target_sent:
                if (arm_error
                        <= self._place_approach_best_error
                        - PLACE_APPROACH_PROGRESS_IMPROVEMENT_RAD):
                    self._place_approach_best_error = arm_error
                    self._place_approach_best_error_at = now
                if (approach_elapsed >= PLACE_APPROACH_HARD_TIMEOUT_S
                        and not converged
                        and now - self._place_approach_best_error_at
                        >= PLACE_APPROACH_PROGRESS_GATE_S):
                    raise RuntimeError(
                        "[place] approach pose did not settle within "
                        f"{PLACE_APPROACH_HARD_TIMEOUT_S:.0f}s "
                        f"(arm_error={arm_error:.4f}rad "
                        f"slide_error={slide_error:.4f}m)")
        elif self.place_stage == 1:
            # Keep Z and the gripper fixed while correcting XY in small steps.
            if (self._place_refine_started_at is not None
                    and now - self._place_refine_started_at
                    >= PLACE_XY_REFINE_TIMEOUT_S):
                raise RuntimeError(
                    "horizontal place refinement timed out; keeping goods "
                    f"clamped after {PLACE_XY_REFINE_TIMEOUT_S:.1f}s")
            if self._place_refine_target_sent:
                settled, residual_standstill = (
                    self._place_refine_command_settled(
                        now, dual=False))
                if not settled:
                    return
                self._place_refine_target_sent = False
                self._place_refine_target_sent_at = None
                self.commands_ready_since = None
                if residual_standstill:
                    tcp = self.selected_tcp_world()
                    if (tcp is not None
                            and self._place_timeout_fallback_safe(tcp)):
                        self.get_logger().warn(
                            "[place] refine target unreachable at the arm "
                            "effort limit; measured TCP already over the "
                            "assigned table slot; descending in place "
                            f"tcp={np.round(tcp, 3)}")
                        self._begin_single_place_descent(now, tcp)
                        return
            tcp = self.selected_tcp_world()
            if tcp is None:
                return
            error_xy = self.place_world[:2] - tcp[:2]
            if np.linalg.norm(error_xy) <= PLACE_SLOT_XY_TOLERANCE_M:
                if self._place_refine_stable_since is None:
                    self._place_refine_stable_since = now
                if (now - self._place_refine_stable_since
                        >= PLACE_XY_REFINE_SETTLE_S):
                    self.get_logger().info(
                        "[place] horizontal refinement verified "
                        f"error={np.round(error_xy, 4)}m "
                        f"settle={PLACE_XY_REFINE_SETTLE_S:.2f}s")
                    self._begin_single_place_descent(now, tcp)
                return
            self._place_refine_stable_since = None
            self._send_single_place_refine_step(tcp, error_xy)
        elif self.place_stage == 2:
            # XY is now fixed.  Keep the arm and gripper unchanged and wait
            # only for the commanded vertical slide motion to finish.
            # 商品底部触桌顶住 slide：停止下压，就地松爪完成放置。
            if self._place_slide_stalled(now):
                self._place_contact_release(now, self.selected_tcp_world())
                return
            if not self.commands_ready(
                    arm_tolerance=0.05, slide_tolerance=0.010):
                descent_timeout = (
                    HEWEIDAO_PLACE_DESCENT_TIMEOUT_S
                    if self.target_kind == "heweidao"
                    else PLACE_DESCENT_TIMEOUT_S)
                if now - self.place_t0 >= descent_timeout:
                    raise RuntimeError(
                        "vertical place descent did not settle within "
                        f"{descent_timeout:.1f}s")
                return
            tcp = self.selected_tcp_world()
            self.get_logger().info(
                "[place] vertical descent complete; opening gripper at "
                f"tcp={None if tcp is None else np.round(tcp, 3)}")
            self.get_logger().info(
                "[place-joints] descent_reached="
                + json.dumps(
                    self._place_joint_snapshot(),
                    ensure_ascii=False, separators=(",", ":")))
            self.place_stage = 3
            self.place_t0 = now
            self._place_release_started_at = now
            self._place_slide_stall_snapshot = None
        elif self.place_stage == 3:
            # Keep arm and slide fixed while the gripper opens.
            self._set_selected_grip(pick.GRIP_OPEN)
            if (now - self._place_release_started_at
                    >= self.place_release_dwell_s):
                self._start_place_vertical_clear(now)
        elif self.place_stage == 4:
            self._place_vertical_clear_tick(now)

    def _configure_dual_place_target(
            self, max_step_m: float | None = None) -> bool:
        """Translate the clamped tissue to its slot without changing wrist pose."""
        left_tcp = self.arm_tcp_world("left")
        right_tcp = self.arm_tcp_world("right")
        left_reference = self.arm_positions("left")
        right_reference = self.arm_positions("right")
        measured_slide = self.joints.get("slide_joint")
        if (left_tcp is None or right_tcp is None
                or left_reference is None or right_reference is None
                or measured_slide is None):
            return False

        centre = 0.5 * (np.asarray(left_tcp) + np.asarray(right_tcp))
        offset_xy = self.place_world[:2] - centre[:2]
        offset_norm = float(np.linalg.norm(offset_xy))
        if (max_step_m is not None and offset_norm > max_step_m):
            offset_xy *= max_step_m / offset_norm
        left_goal = np.asarray(left_tcp, dtype=float).copy()
        right_goal = np.asarray(right_tcp, dtype=float).copy()
        left_goal[:2] += offset_xy
        right_goal[:2] += offset_xy

        slide = float(measured_slide)
        reference = np.concatenate((
            [slide],
            np.asarray(left_reference, dtype=float),
            np.asarray(right_reference, dtype=float),
        ))
        # Middle-column and side-column tissue grasps deliberately use
        # different wrist orientations.  Derive both target transforms from
        # the measured FK and replace translation only; an identity transform
        # here would silently unroll a side-column grasp during placement.
        left_target, right_target = self.kdl.forward_kinematics(reference)
        left_target = np.asarray(left_target, dtype=float).copy()
        right_target = np.asarray(right_target, dtype=float).copy()
        left_target[:3, 3] = self.world_to_footprint(left_goal)
        right_target[:3, 3] = self.world_to_footprint(right_goal)
        try:
            solutions = self.kdl.inverse_kinematics(
                T_left=left_target,
                T_right=right_target,
                target_height=slide,
                ref_pos=reference)
        except Exception as exc:  # noqa: BLE001 - report a safe place failure
            self.get_logger().error(
                f"[place-dual] assigned-slot IK raised: {exc}")
            return False
        if solutions is None or len(solutions) == 0:
            self.get_logger().error(
                "[place-dual] no IK solution for assigned slot "
                f"{np.round(self.place_world[:2], 3)}")
            return False

        candidates = [
            np.asarray(item[1:], dtype=float) for item in solutions]
        arms_reference = reference[1:]
        best = min(
            candidates,
            key=lambda item: float(np.max(
                np.abs(item - arms_reference))))
        self.des_left_arm = best[:6].copy()
        self.des_right_arm = best[6:].copy()
        self.place_ik_ref_source = "measured_dual"
        self.place_ik_reference_joints = arms_reference.copy()
        self.commands_ready_since = None
        self._place_loaded_arm_step_rad = 0.0
        self._dual_place_target_sent = True
        self.place_t0 = self.now()
        self._place_refine_target_sent_at = self.place_t0
        self._place_refine_motion_stable_since = None
        self._place_refine_motion_anchor = None
        self._place_approach_best_error = float("inf")
        self._place_approach_best_error_at = self.place_t0
        self.get_logger().info(
            f"[place-dual] moving clamped tissue to slot="
            f"{None if self.place_slot is None else self.place_slot + 1} "
            f"centre={np.round(centre, 3)} "
            f"target={np.round(self.place_world[:2], 3)} "
            f"offset={np.round(offset_xy, 3)}")
        self.get_logger().info(
            "[place-joints] dual_ik_selection="
            + json.dumps({
                "ref_source": self.place_ik_ref_source,
                "measured_left": self._rounded_list(left_reference),
                "measured_right": self._rounded_list(right_reference),
                "target_left": self._rounded_list(best[:6]),
                "target_right": self._rounded_list(best[6:]),
                "delta_left": self._rounded_list(
                    best[:6] - left_reference),
                "delta_right": self._rounded_list(
                    best[6:] - right_reference),
                "max_joint_delta_from_measured": round(float(np.max(
                    np.abs(best - arms_reference))), 4),
                "measured_slide": round(float(slide), 4),
                "grip_command": self._transport_grip_command,
                "tcp_orientation": "preserved_measured",
            }, ensure_ascii=False, separators=(",", ":")))
        return True

    def _place_tick_dual(self, now: float) -> None:
        """Place dual-arm tissue through overhead, XY, descent and release."""
        if self.place_stage == 0:
            self.des_left_grip = pick.DUAL_TISSUE_GRIP_COMMAND
            self.des_right_grip = pick.DUAL_TISSUE_GRIP_COMMAND
            if not self._advance_place_creep():
                return
            if not self._place_base_settled(now):
                return
            if not self._dual_place_target_sent:
                if not self._configure_dual_place_target():
                    raise RuntimeError(
                        "dual-arm IK failed for assigned delivery slot")
                return
            if not self.dual_commands_ready(
                    arm_tolerance=PLACE_ARM_SETTLE_TOLERANCE_RAD,
                    slide_tolerance=PLACE_APPROACH_SLIDE_TOLERANCE_M):
                # 商品底部已触桌（slide 被顶住）：就地松爪完成放置。
                if self._place_slide_stalled(now):
                    self.get_logger().warn(
                        "[place-dual] slide blocked during overhead "
                        "approach; tissue already at table height — "
                        "releasing in place")
                    self._place_contact_release(now, None)
                    return
                dual_error = self.dual_arm_error()
                if (dual_error
                        <= self._place_approach_best_error
                        - PLACE_APPROACH_PROGRESS_IMPROVEMENT_RAD):
                    self._place_approach_best_error = dual_error
                    self._place_approach_best_error_at = now
                if (now - self._place_stage0_wait_log
                        >= PLACE_APPROACH_PROGRESS_LOG_S):
                    self._place_stage0_wait_log = now
                    self.get_logger().info(
                        "[place-joints] dual_approach_progress="
                        + json.dumps({
                            "elapsed": round(float(now - self.place_t0), 3),
                            "max_joint_delta_from_measured": round(
                                float(dual_error), 4),
                            "measured_slide": self.joints.get("slide_joint"),
                            "centre_world": self._rounded_list(
                                self._dual_release_world()),
                            **self._place_base_diagnostic(),
                        }, ensure_ascii=False, separators=(",", ":")))
                if (now - self.place_t0
                        >= DUAL_TISSUE_APPROACH_HARD_TIMEOUT_S
                        and now - self._place_approach_best_error_at
                        >= PLACE_APPROACH_PROGRESS_GATE_S):
                    raise RuntimeError(
                        "dual-arm overhead pose did not settle within "
                        f"{DUAL_TISSUE_APPROACH_HARD_TIMEOUT_S:.0f}s")
                return
            release_world = self._dual_release_world()
            if release_world is None:
                return
            if self.target_kind == "zhijin":
                # 纸巾专用位固定，overhead 到位后直接下降，跳过水平精调，
                # 避免双臂 XY 平移不一致导致纸巾掉落。
                self._begin_dual_place_descent(now, release_world)
                return
            self.place_stage = 1
            self.place_t0 = now
            self._place_refine_started_at = now
            self._place_refine_target_sent = False
            self._place_refine_stable_since = None
            self._place_refine_motion_anchor = None
            self._dual_place_target_sent = False
            self.commands_ready_since = None
            self.get_logger().info(
                f"[place-dual] tissue reached safe overhead pose; "
                f"starting horizontal refinement centre="
                f"{np.round(release_world, 3)}")
        elif self.place_stage == 1:
            if (self._place_refine_started_at is not None
                    and now - self._place_refine_started_at
                    >= PLACE_XY_REFINE_TIMEOUT_S):
                raise RuntimeError(
                    "dual-arm horizontal place refinement timed out; "
                    "keeping tissue clamped")
            if self._dual_place_target_sent:
                settled, residual_standstill = (
                    self._place_refine_command_settled(
                        now, dual=True))
                if not settled:
                    return
                self._dual_place_target_sent = False
                self._place_refine_target_sent_at = None
                self.commands_ready_since = None
                if residual_standstill:
                    release_world = self._dual_release_world()
                    if (release_world is not None
                            and self._place_timeout_fallback_safe(
                                release_world)):
                        self.get_logger().warn(
                            "[place-dual] refine target unreachable at the "
                            "arm effort limit; measured centre already over "
                            "the assigned table slot; descending in place "
                            f"centre={np.round(release_world, 3)}")
                        self._begin_dual_place_descent(now, release_world)
                        return
            release_world = self._dual_release_world()
            if release_world is None:
                return
            error_xy = self.place_world[:2] - release_world[:2]
            if np.linalg.norm(error_xy) <= PLACE_SLOT_XY_TOLERANCE_M:
                if self._place_refine_stable_since is None:
                    self._place_refine_stable_since = now
                if (now - self._place_refine_stable_since
                        < PLACE_XY_REFINE_SETTLE_S):
                    return
                self._begin_dual_place_descent(now, release_world)
                return
            self._place_refine_stable_since = None
            if not self._configure_dual_place_target(
                    max_step_m=PLACE_XY_REFINE_STEP_M):
                raise RuntimeError(
                    "dual-arm horizontal refinement IK failed")
            self._place_refine_iterations += 1
        elif self.place_stage == 2:
            # 商品底部触桌顶住 slide：停止下压，就地松爪完成放置。
            if self._place_slide_stalled(now):
                self._place_contact_release(now, None)
                return
            if not self.dual_commands_ready(
                    arm_tolerance=0.05, slide_tolerance=0.010):
                if now - self.place_t0 >= PLACE_DESCENT_TIMEOUT_S:
                    raise RuntimeError(
                        "dual-arm vertical descent did not settle within "
                        f"{PLACE_DESCENT_TIMEOUT_S:.1f}s")
                return
            release_world = self._dual_release_world()
            target_z = self._product_release_z()
            self.place_stage = 3
            self.place_t0 = now
            self._place_release_started_at = now
            self._place_slide_stall_snapshot = None
            self.get_logger().info(
                "[place-dual] vertical descent complete; starting "
                "inverse-squeeze arm opening with grippers unchanged "
                f"at centre={None if release_world is None else np.round(release_world, 3)}")
            self.get_logger().info(
                "[place-joints] dual_descent_reached="
                + json.dumps({
                    "measured_left": self._rounded_list(
                        self.arm_positions("left")),
                    "measured_right": self._rounded_list(
                        self.arm_positions("right")),
                    "measured_slide": self.joints.get("slide_joint"),
                    "centre_world": self._rounded_list(release_world),
                    "target_z": round(float(target_z), 4),
                    "measured_left_grip": self.joints.get(
                        "left_arm_eef_gripper_joint"),
                    "measured_right_grip": self.joints.get(
                        "right_arm_eef_gripper_joint"),
                }, ensure_ascii=False, separators=(",", ":")))
        elif self.place_stage == 3:
            # 反向复用抓取预压：夹爪命令保持运输值不变，左右 TCP 各向外
            # 移动 dual_squeeze_m。张臂到位后先垂直上抬，再后退离桌。
            self._hold_grasp_during_transport()
            if not self._dual_release_spread_sent:
                self._start_dual_place_release_spread(now)
                return
            release_status = self.advance_dual_tissue_motion()
            if release_status == "failed":
                raise RuntimeError(
                    "inverse-squeeze tissue release did not converge")
            if release_status != "reached":
                if now - self.place_t0 >= PLACE_DESCENT_TIMEOUT_S:
                    raise RuntimeError(
                        "inverse-squeeze tissue release did not settle within "
                        f"{PLACE_DESCENT_TIMEOUT_S:.1f}s")
                return
            self._dual_release_spread_sent = False
            self._start_place_vertical_clear(now)
        elif self.place_stage == 4:
            self._place_vertical_clear_tick(now)

    def _clear_delivery_table_tick(self, now: float) -> None:
        """Back away horizontally, then recover the robot posture."""
        clearance = point_to_rect_clearance(
            float(self.base_xy[0]), float(self.base_xy[1]),
            DELIVERY_TABLE_COSTMAP_BOUNDS)
        required = WHOLE_BODY_KEEP_OUT_RADIUS + PLACE_CLEAR_TABLE_MARGIN_M
        if self._place_retreat_start_clearance is None:
            self._place_retreat_start_clearance = min(clearance, required)

        if clearance >= required:
            self.set_twist(0.0, 0.0)
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0
            if not self._place_retreat_sent:
                self._start_place_arm_recovery(clearance)
                return
            # Reassert the default posture only after the horizontal backing
            # motion has ended.  Completion is feedback-gated below.
            self._command_initial_arm_posture()
            if (now - self.place_t0 < self.place_retreat_dwell_s
                    or not self.dual_commands_ready(
                        arm_tolerance=0.08,
                        slide_tolerance=0.05)
                    or not self._initial_head_posture_ready()):
                return
            self.placement_completed = True
            self.place_t0 = now
            self.get_logger().info(
                f"[flow] PLACE COMPLETE — {self.target_kind} delivered; "
                f"table cleared (clearance={clearance:.3f}m); "
                f"base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f})")
            if self.return_start_after_place:
                self._start_return_to_start(now)
            elif self.return_west_after_place:
                self._start_return_to_west(now)
            else:
                self._set_flow_phase("done")
            return

        # Vertical raising is already complete.  Tissue keeps its unchanged
        # closed-jaw command throughout horizontal retreat; normal products
        # keep their already-open command.  Once the chassis is clear,
        # _start_place_arm_recovery opens both grippers away from the goods.
        if self.use_dual_tissue_grasp:
            self._hold_grasp_during_transport()
        else:
            self.des_left_grip = pick.GRIP_OPEN
            self.des_right_grip = pick.GRIP_OPEN

        elapsed = now - self.place_t0
        if elapsed >= PLACE_CLEAR_TABLE_TIMEOUT_S:
            self.set_twist(0.0, 0.0)
            raise RuntimeError(
                "could not back out of delivery-table keep-out after place: "
                f"clearance={clearance:.3f}m required={required:.3f}m")

        yaw_error = pick.wrap_to_pi(-math.pi / 2.0 - self.base_yaw)
        self.set_twist(
            -PLACE_CLEAR_TABLE_SPEED_MPS,
            float(np.clip(2.0 * yaw_error, -0.25, 0.25)))

    def _post_delivery_warning(self, message: str) -> None:
        """Record a non-fatal failure after the product is already delivered."""
        message = str(message)
        self.post_delivery_warnings.append(message)
        self.get_logger().warn(f"[post-delivery] {message}")

    def _on_rotation_recovery_exhausted(self) -> None:
        """Preserve delivery if optional post-place refinement cannot rotate."""
        if self.placement_completed:
            self._post_delivery_warning(
                "rotation recovery budget exhausted after delivery")
            self._finish_after_return_scan(self.now())
            return
        super()._on_rotation_recovery_exhausted()

    def _finish_after_return_scan(self, now: float) -> None:
        self.set_twist(0.0, 0.0)
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        self._set_flow_phase("done")
        self.place_t0 = now

    def _start_return_to_start(self, now: float) -> None:
        """Start the final-delivery return from the table to the start pose."""
        self._signal_all_orders_completed(now)
        self._set_flow_phase("return_to_start")
        self._nav_goal = None
        self._nav_last_log = 0.0
        self._last_nav_reason = None
        self._nav_memory_logged = False
        self.get_logger().info(
            "[return-start] final delivery complete; returning to start "
            f"goal=({START_POSE[0]:.3f},{START_POSE[1]:.3f},"
            f"{math.degrees(START_POSE[2]):.0f}deg)")

    def configure_all_orders_completion_signal(
            self, order_id: str, completion_file: str | None) -> None:
        """Configure the atomic hand-off from delivery to final return."""
        self.completion_order_id = str(order_id)
        self.completion_file = completion_file

    def _signal_all_orders_completed(self, now: float) -> None:
        """Persist final delivery before entering the independent return."""
        if self._all_orders_completion_signalled:
            return
        if not self.completion_file:
            # Manual single-worker runs have no supervising runner.  They
            # still retain the same phase boundary and human-facing log, but
            # do not need the file handshake used to disable the runner's
            # per-order timeout.
            self._all_orders_completion_signalled = True
            self.get_logger().info("All order completed")
            return
        slot = self.target_slot()
        document = {
            "schema_version": 1,
            "milestone": "all_orders_completed",
            "status": "delivered",
            "order_id": self.completion_order_id,
            "kind": self.target_kind,
            "marker_id": self.target_marker_id,
            "slot": None if slot is None else list(slot),
            "slot_key": self.target_slot_key(),
            "flow_phase": "return_to_start",
            "signalled_at_monotonic": round(float(now), 3),
        }
        destination = pathlib.Path(self.completion_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8")
        temporary.replace(destination)
        self._all_orders_completion_signalled = True
        self.get_logger().info("All order completed")

    def _return_to_start_tick(self, now: float) -> None:
        target = np.asarray(START_POSE[:2], dtype=float)
        arrived = self.drive_to(target, START_POSE[2], 0.08)
        if not arrived:
            return
        self.set_twist(0.0, 0.0)
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        self._set_flow_phase("done")
        self.get_logger().info(
            f"[return-start] reached start pose base=({self.base_xy[0]:.2f},"
            f"{self.base_xy[1]:.2f}) yaw={math.degrees(self.base_yaw):.0f}deg")

    def _start_return_to_west(self, now: float) -> None:
        """Start the first-delivery return from the table to shelf A."""
        self._set_flow_phase("return_to_west")
        # The first shelf trip may already have completed this route state.
        # Reset it so the delivery trunk is genuinely traversed in reverse.
        self.scan_trunk_route_stage = None
        self.scan_trunk_route_done = False
        self.scan_direct_fallback_used = False
        self.scan_route_final_goal = None
        self._route_leg_name = None
        self._route_leg_goal = None
        self._nav_goal = None
        self._nav_last_log = 0.0
        self._last_nav_reason = None
        self._nav_memory_logged = False
        self.return_scan_pose_index = 0
        self.return_scan_pose_started_at = now
        self.return_scan_camera_ready_since = None
        self.get_logger().info(
            "[return-west] first delivery complete; returning to shelf A "
            f"goal=({RETURN_WEST_GOAL[0]:.3f},"
            f"{RETURN_WEST_GOAL[1]:.3f},north)")

    def _return_to_west_tick(self, now: float) -> None:
        target = np.asarray(RETURN_WEST_GOAL[:2], dtype=float)
        try:
            trunk_ready = self._scan_trunk_route_tick(
                target, RETURN_WEST_GOAL[2])
        except RuntimeError as exc:
            self._post_delivery_warning(
                f"return to shelf A failed: {exc}")
            self._finish_after_return_scan(now)
            return
        if not trunk_ready:
            return

        # The route navigator has a deliberately broad arrival tolerance.
        # Refine position and north-facing yaw before recording shelf A.
        arrived = pick.ShelfPickController.drive_to(
            self, target, RETURN_WEST_GOAL[2], 0.08)
        if not arrived:
            return
        self.set_twist(0.0, 0.0)
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        # 临时要求：只停定在 A 货架前，不做整体库存扫描；停稳即收工，
        # 让 runner 派发下一单。return_west_scan 相关代码保留未动。
        self.get_logger().info(
            "[return-west] stopped at shelf A; finishing worker "
            "(stop-only, no inventory scan)")
        self._finish_after_return_scan(now)

    def _advance_return_scan_pose(self, now: float) -> None:
        completed_name = RETURN_WEST_SCAN_POSES[
            self.return_scan_pose_index][0]
        self.get_logger().info(
            f"[return-west-scan] completed pose={completed_name} "
            f"{self.return_scan_pose_index + 1}/"
            f"{len(RETURN_WEST_SCAN_POSES)}")
        self.return_scan_pose_index += 1
        self.return_scan_pose_started_at = now
        self.return_scan_camera_ready_since = None
        if self.return_scan_pose_index < len(RETURN_WEST_SCAN_POSES):
            return
        self._set_flow_phase("return_west_recover")
        self.return_recovery_started_at = now
        self._command_initial_arm_posture()
        self.get_logger().info(
            "[return-west-scan] shelf A inventory dwell complete; "
            "restoring neutral transit posture")

    def _return_west_scan_tick(self, now: float) -> None:
        self.set_twist(0.0, 0.0)
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        pose = RETURN_WEST_SCAN_POSES[self.return_scan_pose_index]
        pose_name, slide_target, yaw_target, pitch_target = pose
        self.des_slide = slide_target
        self.des_head[:] = [yaw_target, pitch_target]

        if not self.scan_camera_ready(pose):
            self.return_scan_camera_ready_since = None
            if (now - self.return_scan_pose_started_at
                    >= RETURN_WEST_SCAN_POSE_TIMEOUT_S):
                self._post_delivery_warning(
                    f"shelf A scan pose {pose_name} did not settle within "
                    f"{RETURN_WEST_SCAN_POSE_TIMEOUT_S:.1f}s; continuing")
                self._advance_return_scan_pose(now)
            return
        if self.return_scan_camera_ready_since is None:
            self.return_scan_camera_ready_since = now
            return
        required = (
            pick.SCAN_CAMERA_STABLE_S
            + pick.SCAN_SETTLE_S
            + pick.SCAN_DWELL_S)
        if now - self.return_scan_camera_ready_since < required:
            return
        self._advance_return_scan_pose(now)

    def _return_west_recover_tick(self, now: float) -> None:
        self.set_twist(0.0, 0.0)
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        self._command_initial_arm_posture()
        ready = (
            self.dual_commands_ready(
                arm_tolerance=0.08, slide_tolerance=0.05)
            and self._initial_head_posture_ready())
        if ready:
            self.get_logger().info(
                "[return-west] shelf A scan complete; posture recovered")
            self._finish_after_return_scan(now)
            return
        if (now - self.return_recovery_started_at
                >= RETURN_WEST_RECOVERY_TIMEOUT_S):
            self._post_delivery_warning(
                "neutral posture recovery timed out after shelf A scan")
            self._finish_after_return_scan(now)

    def _place_base_motion_active(self) -> bool:
        """Return whether the active placement stage intentionally moves the base.

        The refined table-placement flow performs its final creep in stage 0
        and its post-release table retreat in stage 5; all motion in stages 1-4
        runs against one fixed world-frame base.
        """
        if self.place_stage == 0:
            return not self.place_creep_done
        return self.place_stage == 5

    def smooth_commands(self) -> None:
        """Limit loaded placement arm motion and the final slide descent."""
        previous_slide = self.cmd_slide
        previous_left_arm = self.cmd_left_arm.copy()
        previous_right_arm = self.cmd_right_arm.copy()
        super().smooth_commands()

        # Do not let a command accumulated before a sharp path turn coast
        # above the carried-product caps while the normal slew limiter catches
        # up.  This is the exact transition that preceded both observed
        # kouxiangtang drops.
        loaded_linear_cap, loaded_angular_cap = (
            self._loaded_transport_limits())
        if loaded_linear_cap is not None:
            self.cmd_linear = float(np.clip(
                self.cmd_linear,
                -loaded_linear_cap, loaded_linear_cap))
        if loaded_angular_cap is not None:
            self.cmd_angular = float(np.clip(
                self.cmd_angular,
                -loaded_angular_cap, loaded_angular_cap))

        # Enforce the one-shot cap on the published command too.  This keeps a
        # previously accumulated angular command from coasting above the new
        # desired limit during the first loaded turn.
        if self._post_grab_slow_turn_active():
            self.cmd_angular = float(np.clip(
                self.cmd_angular,
                -HEWEIDAO_LOADED_TURN_MAX_RPS,
                HEWEIDAO_LOADED_TURN_MAX_RPS))

        loaded_place_extension = (
            self.flow_phase == "place"
            and self.place_stage in {0, 1}
            and (
                (self.use_dual_tissue_grasp
                 and self._dual_place_target_sent)
                or (not self.use_dual_tissue_grasp
                    and self._place_arm_target_sent)))
        if loaded_place_extension:
            loaded_arm_max_step = (
                PLACE_LOADED_ARM_MAX_STEP_BY_KIND_RAD.get(
                    self.target_kind, PLACE_LOADED_ARM_MAX_STEP_RAD))
            loaded_arm_step_ramp = (
                PLACE_LOADED_ARM_STEP_RAMP_BY_KIND_RAD.get(
                    self.target_kind, PLACE_LOADED_ARM_STEP_RAMP_RAD))
            self._place_loaded_arm_step_rad = min(
                loaded_arm_max_step,
                self._place_loaded_arm_step_rad
                + loaded_arm_step_ramp)
            arm_step = self._place_loaded_arm_step_rad
            if self.use_dual_tissue_grasp:
                combined = self.synchronized_slew(
                    np.concatenate((previous_left_arm, previous_right_arm)),
                    np.concatenate((self.des_left_arm, self.des_right_arm)),
                    arm_step)
                self.cmd_left_arm = combined[:6]
                self.cmd_right_arm = combined[6:]
            elif self.grasp_arm == "l":
                self.cmd_left_arm = self.synchronized_slew(
                    previous_left_arm, self.des_left_arm, arm_step)
            else:
                self.cmd_right_arm = self.synchronized_slew(
                    previous_right_arm, self.des_right_arm, arm_step)

        # NavigationController already applies acceleration ramps during
        # normal motion.  Do not apply the parent's second ramp in the unsafe
        # direction when the navigator has explicitly requested a stop: at
        # 0.90 m/s, a 0.03-per-tick decay could otherwise preserve forward
        # motion for roughly 0.6 s after a lidar/trajectory stop.  Angular
        # motion is cancelled only for reasons that require the complete base
        # to hold; obstacle stops may still rotate in place to find a route.
        nav_reason = self.nav.controller.stop_reason
        if (self.flow_phase in {
                "grab", "nav_to_delivery", "return_to_west",
                "return_to_start"}
                and abs(self.des_linear) <= 1e-9
                and nav_reason is not None):
            self.cmd_linear = 0.0
        full_hold = (
            nav_reason == "table_keepout"
            or nav_reason == "rotation_loop"
            or nav_reason in {
                "reverse_recovery_start", "lateral_escape_replan"
            }
            or (isinstance(nav_reason, str)
                and (nav_reason.startswith("no_path")
                     or nav_reason.startswith("stuck_no_path"))))
        if (self.flow_phase in {
                "grab", "nav_to_delivery", "return_to_west",
                "return_to_start"}
                and abs(self.des_angular) <= 1e-9
                and full_hold):
            self.cmd_angular = 0.0

        # Once the final creep has stopped, arm positioning, refinement,
        # descent and release all run against one fixed world-frame base.
        if (self.flow_phase == "place"
                and not self._place_base_motion_active()):
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0

        single_descent = (
            not self.use_dual_tissue_grasp and self.place_stage == 2)
        dual_descent = (
            self.use_dual_tissue_grasp
            and self.place_stage == 2
            and self._dual_descent_sent)
        vertical_clear = self.place_stage == 4
        single_loaded_slide_positioning = (
            self.flow_phase == "place"
            and not self.use_dual_tissue_grasp
            and self.place_stage == 0
            and self._place_slide_target_sent)
        if (self.flow_phase == "place"
                and (single_loaded_slide_positioning
                     or single_descent or dual_descent or vertical_clear)):
            slide_step = (
                HEWEIDAO_PLACE_DESCENT_SLIDE_STEP_M
                if (single_descent and self.target_kind == "heweidao")
                else PLACE_DESCENT_SLIDE_STEP_M)
            self.cmd_slide = float(self.slew(
                previous_slide,
                self.des_slide,
                slide_step))

    # ------------------------------------------------------------------
    # main control loop
    # ------------------------------------------------------------------
    def _stop_for_stale_feedback(
            self, *, odom_stale: bool, joints_stale: bool,
            laser_stale: bool) -> None:
        """Stop immediately and terminate the match after persistent loss."""
        monotonic_now = time.monotonic()
        if self._feedback_stale_since is None:
            self._feedback_stale_since = monotonic_now
        elapsed = max(0.0, monotonic_now - self._feedback_stale_since)

        # Bypass both navigation and the parent's velocity smoothing.  Arm and
        # gripper commands are deliberately left at their most recent values:
        # without joint feedback it is unsafe to begin an unverified recovery
        # trajectory or open a possibly loaded gripper.
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        self.cmd_vel_pub.publish(pick.Twist())
        if monotonic_now - self._feedback_warn_monotonic >= 1.0:
            self.get_logger().warn(
                "stopping for stale robot feedback "
                f"(odom_stale={odom_stale}, joints_stale={joints_stale}, "
                f"laser_stale={laser_stale}, elapsed={elapsed:.1f}s)")
            self._feedback_warn_monotonic = monotonic_now

        if (elapsed < FEEDBACK_LOSS_HARD_TIMEOUT_S
                or self._feedback_timeout_triggered):
            return
        self._feedback_timeout_triggered = True
        self._fatal_match = True
        self.terminal_error = (
            "fatal-match: robot feedback unavailable for "
            f"{elapsed:.1f}s (odom_stale={odom_stale}, "
            f"joints_stale={joints_stale}, laser_stale={laser_stale})")
        self.get_logger().error(
            "[feedback-watchdog] persistent robot feedback loss; stopping "
            "the match without starting another motion worker: "
            f"{self.terminal_error}")
        import rclpy
        rclpy.shutdown()

    def tick(self) -> None:
        if self.base_xy is None or not self.joints:
            self._publish_perception_request(False)
            self._stop_for_stale_feedback(
                odom_stale=self.base_xy is None,
                joints_stale=not self.joints,
                laser_stale=self.last_scan_time is None)
            return
        self._record_motion_telemetry()
        now = self.now()
        odom_stale = (
            self.last_odom_time is None
            or now - self.last_odom_time > NAV_STATE_STALE_S)
        joints_stale = (
            self.last_joint_time is None
            or now - self.last_joint_time > NAV_STATE_STALE_S)
        laser_stale = self._laser_stale(now)
        memory_corridor_transit = (
            self.memory_file is not None
            and self.state == pick.STATE_GO_SCAN
            and STATION_Y_MIN <= float(self.base_xy[1]) <= STATION_Y_MAX)
        stable_perception_state = (
            self.state == pick.STATE_SCAN
            or (self.state in {pick.STATE_REVISIT, pick.STATE_RECHECK}
                and self.scan_camera_ready_since is not None)
            or memory_corridor_transit)
        stable_return_scan = (
            self.flow_phase == "return_west_scan"
            and self.return_scan_camera_ready_since is not None
            and now - self.return_scan_camera_ready_since
            >= pick.SCAN_CAMERA_STABLE_S)
        perception_needed = (
            not (odom_stale or joints_stale or laser_stale)
            and ((self.flow_phase == "grab" and stable_perception_state)
                 or stable_return_scan))
        self._publish_perception_request(perception_needed)
        if odom_stale or joints_stale or laser_stale:
            self._stop_for_stale_feedback(
                odom_stale=odom_stale,
                joints_stale=joints_stale,
                laser_stale=laser_stale)
            return
        if self._feedback_stale_since is not None:
            stale_elapsed = max(
                0.0, time.monotonic() - self._feedback_stale_since)
            self.get_logger().info(
                "[feedback-watchdog] robot feedback recovered after "
                f"{stale_elapsed:.1f}s")
            self._feedback_stale_since = None
            self._feedback_timeout_triggered = False
        if not self.initialized:
            self.initialize_commands()

        # The simulator resets only the base pose between workers, so a failed
        # predecessor can leave a closed gripper / deployed arm behind through
        # the joint feedback.  Restore the neutral posture once at startup,
        # before any shelf motion begins.
        if (not self._startup_posture_recovered
                and self.state in {
                    pick.STATE_GO_SCAN, pick.STATE_DIRECT_TRANSIT}
                and self.flow_phase == "grab"):
            self._command_initial_arm_posture()
            if self._neutral_posture_ready():
                self._startup_posture_recovered = True
                self.get_logger().info(
                    "[startup] neutral manipulation posture verified")

        try:
            if self.flow_phase == "grab":
                self._memory_route_tick()
                super().tick()
                # STATE_DONE is terminal for the pick FSM but not for the
                # integrated delivery flow.  Gate on flow_phase so a missed
                # transition is repaired exactly once on the next tick.
                if (self.state == pick.STATE_DONE
                        and self.flow_phase == "grab"):
                    self._on_grab_complete()
                return
        except Exception as exc:  # noqa: BLE001 - report, recover, exit
            self._enter_fatal_recovery(exc)

        # Post-grab phases: same command pipeline tail as the parent tick.
        self.set_twist(0.0, 0.0)
        # A release command is legal only after the corresponding placement
        # controller has verified the assigned slot and low release pose.
        # Reassert the captured holding command through base motion and arm
        # positioning so no stale/default command can loosen the gripper while
        # the loaded robot turns or reaches over the table.
        single_place_hold = (
            self.flow_phase == "place"
            and not self.use_dual_tissue_grasp
            and self.place_stage in {0, 1, 2})
        dual_place_hold = (
            self.flow_phase == "place"
            and self.use_dual_tissue_grasp
            and self.place_stage in {0, 1, 2})
        if (self.flow_phase in {
                "backup", "restore_height", "nav_to_delivery"}
                or single_place_hold or dual_place_hold):
            self._hold_grasp_during_transport()
        drop_paused = self._monitor_held_product(now)
        drop_candidate_hold = (
            drop_paused
            and (self.flow_phase in {
                "backup", "restore_height", "nav_to_delivery"}
                 or (self.flow_phase == "place"
                     and self.place_stage in {0, 1, 2})))
        if self.flow_phase == "fatal_recover":
            # A phase tick raised (e.g. place IK/timeout RuntimeError).  rclpy
            # swallows callback exceptions, so without this branch the worker
            # would stay parked mid-motion until the runner kills it.  Restore
            # the neutral posture, then shut down with the error recorded.
            if self._fatal_recovery_tick(now):
                import rclpy
                rclpy.shutdown()
                return
        else:
            try:
                if drop_paused:
                    # The first empty-grasp signature already locks the base;
                    # after debounce the monitor may also have selected a
                    # recovery phase.  In either case, do not run one more
                    # tick of the old phase.
                    pass
                elif self.flow_phase == "backup":
                    self._backup_tick()
                elif self.flow_phase == "restore_height":
                    self._restore_height_tick()
                elif self.flow_phase == "nav_to_delivery":
                    self._nav_to_delivery_tick()
                elif self.flow_phase == "place":
                    self._place_tick()
                elif self.flow_phase == "return_to_start":
                    self._return_to_start_tick(now)
                elif self.flow_phase == "return_to_west":
                    self._return_to_west_tick(now)
                elif self.flow_phase == "return_west_scan":
                    self._return_west_scan_tick(now)
                elif self.flow_phase == "return_west_recover":
                    self._return_west_recover_tick(now)
                elif self.flow_phase == "drop_success_recover":
                    self._drop_recovery_tick(now, delivered=True)
                elif self.flow_phase == "drop_failed_recover":
                    self._drop_recovery_tick(now, delivered=False)
                elif not self._flow_done_logged:
                    self._flow_done_logged = True
                    self.get_logger().info(
                        f"[flow] done — final base=({self.base_xy[0]:.2f},"
                        f"{self.base_xy[1]:.2f})")
            except Exception as exc:  # noqa: BLE001 - report, recover, exit
                self._enter_fatal_recovery(exc)
        if (self.flow_phase == "done"
                and self.now() - self.place_t0 > FLOW_DONE_SETTLE_S):
            self.get_logger().info("[flow] flow finished; shutting down")
            import rclpy
            rclpy.shutdown()
            return
        if (self.flow_phase == "drop_failed"
                and self.now() - self.place_t0
                > TRANSPORT_DROP_FAILURE_SETTLE_S):
            self.get_logger().error(
                "[drop-monitor] transport-drop worker finished; requesting "
                "runner restart")
            import rclpy
            rclpy.shutdown()
            return

        if drop_candidate_hold:
            # Keep the last arm/slide command frozen during the short
            # confirmation window.  Calling smooth_commands here would keep
            # advancing a previously requested loaded placement trajectory
            # even though the base has already stopped.
            self.cmd_linear = 0.0
            self.cmd_angular = 0.0
            self.publish_commands()
            return

        self.apply_manip_base_hold()
        self.smooth_commands()
        self.publish_commands()

        if self.now() - self.last_status_log > 1.0:
            self.get_logger().info(
                f"[flow] phase={self.flow_phase} "
                f"state={self.state} place_stage={self.place_stage} "
                f"base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
                f"yaw={math.degrees(self.base_yaw):.0f}°")
            self.last_status_log = self.now()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="integrated nav + YOLO/ArUco pick + place client")
    parser.add_argument(
        "--target-kind", required=True,
        choices=sorted(pick.PRODUCT_CENTER_ABOVE_MARKER_M),
        help="exact goods class to remove from the shelf")
    parser.add_argument(
        "--order-id", default="manual",
        help="anonymous competition order id recorded in the worker result")
    parser.add_argument(
        "--candidate-kind", action="append", default=[],
        choices=sorted(pick.PRODUCT_CENTER_ABOVE_MARKER_M),
        help="another pending order class that may become this trip's sole "
             "target after repeated scan detections")
    parser.add_argument(
        "--result-file",
        help="write a machine-readable worker result for competition_runner")
    parser.add_argument(
        "--completion-file",
        help="atomically signal final delivery before independent return")
    parser.add_argument(
        "--exclude-marker-id", action="append", type=int, default=[],
        help="ignore a marker already delivered or failed in this match")
    parser.add_argument(
        "--exclude-slot-key", action="append", default=[],
        help="ignore a failed YOLO-only slot formatted as Lx|Shelf|Column")
    parser.add_argument(
        "--exclude-kind-slot", action="append", default=[],
        help="ignore a class-scoped failed slot formatted as KIND=Lx|Shelf|Column")
    parser.add_argument(
        "--memory-file",
        help="runner-owned atomic memory-matrix JSON used for live routing")
    parser.add_argument(
        "--memory-confidence-threshold", type=float, default=0.90)
    parser.add_argument(
        "--dynamic-direct", action="store_true",
        help="allow reliable fresh memory candidates to redirect this worker "
             "directly to a guarded fixed slot")
    parser.add_argument(
        "--perception-always-on", action="store_true",
        help="ignore state-level perception disable requests so a formal "
             "run can record the matrix during every phase")
    parser.add_argument("--memory-initial-shelf", choices=list("ABCDE"))
    parser.add_argument(
        "--memory-initial-level", choices=["L1", "L2", "L3"])
    parser.add_argument(
        "--memory-initial-column", choices=["1", "2", "3"])
    parser.add_argument("--memory-initial-product-y", type=float)
    parser.add_argument("--memory-initial-product-z", type=float)
    parser.add_argument(
        "--formal-mode", action="store_true",
        help="disable all fixed-layout diagnostic shortcuts")
    parser.add_argument(
        "--external-perception", action="store_true",
        help="consume the persistent runner-owned YOLO/ArUco topics instead "
             "of loading another detector model in this worker")
    parser.add_argument(
        "--weights", default=str(REPO_ROOT / "examples" / "supermarket_sorting" / "perception" / "checkpoints" / "best.pt"),
        help="multi-class Ultralytics checkpoint (default: repository best.pt)")
    parser.add_argument(
        "--confidence", type=float, default=0.45)
    parser.add_argument(
        "--max-inference-hz", type=float, default=12.0,
        help="maximum YOLO source-frame rate during active scan states")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--show", action="store_true", help="show the YOLO result window")
    parser.add_argument(
        "--max-scan-cycles", type=int, default=3)
    parser.add_argument(
        "--tcp-diagnostic-ground-truth", action="store_true",
        help="fixed-geometry diagnostic only (no product truth); "
             "use fixed public shelf geometry for the target centre")
    parser.add_argument(
        "--scan-skip-lower", action="store_true",
        help="deprecated no-op: the fixed-layout lower-shelf skip was "
             "removed for rule compliance; scans always use full poses")
    parser.add_argument(
        "--no-close-recheck", action="store_true",
        help="disable close-range class verification before grasping")
    parser.add_argument(
        "--place-x", type=float, default=DELIVERY_TABLE_PLACE_WORLD[0])
    parser.add_argument(
        "--place-y", type=float, default=DELIVERY_TABLE_PLACE_WORLD[1])
    parser.add_argument(
        "--place-z", type=float, default=DELIVERY_TABLE_PLACE_WORLD[2],
        help="minimum TCP height for the pre-release approach pose")
    parser.add_argument(
        "--place-slot", type=int,
        choices=range(len(DELIVERY_PLACE_SLOTS_XY)),
        help="zero-based deterministic delivery slot; overrides "
             "--place-x/--place-y")
    parser.add_argument(
        "--place-release-dwell", type=float, default=2.0,
        help="seconds the gripper stays open before retreating")
    parser.add_argument(
        "--place-retreat-dwell", type=float, default=1.0)
    parser.add_argument(
        "--backup-after-grab", type=float, default=0.20,
        help="base backup distance in metres after grasp and before delivery "
             "navigation (0 disables)")
    parser.add_argument(
        "--place-creep-distance", type=float,
        default=PLACE_CREEP_DISTANCE_M,
        help="guarded final approach distance toward the delivery table in "
             "metres (0 disables)")
    parser.add_argument(
        "--no-nav-during-scan", action="store_true",
        help="use the parent straight-line drive_to between scan stations")
    parser.add_argument(
        "--scan-start-west", action="store_true",
        help="scan from the westmost shelf (A) first; used for orders after "
             "the first in a match")
    parser.add_argument(
        "--return-west-after-place", action="store_true",
        help="after a successful placement, return to shelf A and perform "
             "one stationary full-view inventory scan before exiting")
    parser.add_argument(
        "--return-start-after-place", action="store_true",
        help="after the final placement of a match, return to the start pose "
             "before exiting")
    parser.add_argument(
        "--scan-start-x", type=float,
        help="measured product world X from cross-order inventory; chooses "
             "the nearest first scan station without bypassing perception")
    parser.add_argument(
        "--scan-marker-z", type=float,
        help="measured shelf-marker Z paired with --scan-start-x; prioritises "
             "that camera level with automatic full-scan fallback")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if not 0.0 <= args.memory_confidence_threshold <= 1.0:
        parser.error("--memory-confidence-threshold must be in [0, 1]")
    if not 0.0 < args.max_inference_hz < float("inf"):
        parser.error("--max-inference-hz must be finite and positive")
    if args.max_scan_cycles < 1:
        parser.error("--max-scan-cycles must be >= 1")
    if args.backup_after_grab < 0.0:
        parser.error("--backup-after-grab must be >= 0")
    if args.place_creep_distance < 0.0:
        parser.error("--place-creep-distance must be >= 0")
    if args.scan_start_x is not None and not math.isfinite(args.scan_start_x):
        parser.error("--scan-start-x must be finite")
    if args.scan_marker_z is not None and not math.isfinite(args.scan_marker_z):
        parser.error("--scan-marker-z must be finite")
    if args.scan_marker_z is not None and args.scan_start_x is None:
        parser.error("--scan-marker-z requires --scan-start-x")
    initial_direct_slot = (
        args.memory_initial_shelf,
        args.memory_initial_level,
        args.memory_initial_column,
    )
    if any(value is not None for value in initial_direct_slot):
        if not all(value is not None for value in initial_direct_slot):
            parser.error(
                "initial memory shelf/level/column must be supplied "
                "together")
    elif (args.memory_initial_product_y is not None
          or args.memory_initial_product_z is not None):
        parser.error(
            "initial memory product coordinates require a direct slot")
    for value in (
            args.memory_initial_product_y,
            args.memory_initial_product_z):
        if value is not None and not math.isfinite(value):
            parser.error("initial memory product coordinates must be finite")
    if args.formal_mode and (
            args.tcp_diagnostic_ground_truth or args.scan_skip_lower):
        parser.error(
            "formal mode forbids fixed-layout ground truth and scan shortcuts")
    invalid_markers = [value for value in args.exclude_marker_id
                       if value < 0 or value > 44]
    if invalid_markers:
        parser.error(f"invalid ArUco marker ids: {invalid_markers}")
    invalid_slots = []
    for value in args.exclude_slot_key:
        parts = str(value).split("|")
        if (len(parts) != 3
                or parts[0] not in {"L1", "L2", "L3"}
                or parts[1] not in {"A", "B", "C", "D", "E"}
                or parts[2] not in {"1", "2", "3"}):
            invalid_slots.append(value)
    if invalid_slots:
        parser.error(f"invalid memory slot keys: {invalid_slots}")
    invalid_kind_slots = []
    for value in args.exclude_kind_slot:
        kind, separator, slot_key = str(value).partition("=")
        parts = slot_key.split("|")
        if (not separator
                or kind not in pick.PRODUCT_CENTER_ABOVE_MARKER_M
                or len(parts) != 3
                or parts[0] not in {"L1", "L2", "L3"}
                or parts[1] not in {"A", "B", "C", "D", "E"}
                or parts[2] not in {"1", "2", "3"}):
            invalid_kind_slots.append(value)
    if invalid_kind_slots:
        parser.error(
            f"invalid class-scoped memory slots: {invalid_kind_slots}")
    return args


def _cv_gui_available() -> bool:
    """True if this OpenCV build supports HighGUI windows (GTK etc.)."""
    try:
        import cv2
        cv2.namedWindow("__cv_gui_probe__")
        cv2.destroyWindow("__cv_gui_probe__")
        return True
    except cv2.error:
        return False


def _write_result(path: str | None, document: dict) -> None:
    if not path:
        return
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def main() -> int:
    from run_log import start_run_log
    start_run_log("worker")
    args = parse_args()
    started_at = time.monotonic()
    place_x, place_y = args.place_x, args.place_y
    if args.place_slot is not None:
        place_x, place_y = DELIVERY_PLACE_SLOTS_XY[args.place_slot]
    weights = str(pathlib.Path(args.weights).expanduser().resolve())
    if not pathlib.Path(weights).is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    import rclpy
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

    rclpy.init()
    nodes = []
    spin_thread = None
    controller = None
    caught_error = None
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        controller = IntegratedNavPickPlace(
            args.target_kind, args.max_scan_cycles,
            args.tcp_diagnostic_ground_truth, args.scan_skip_lower,
            place_x=place_x, place_y=place_y, place_z=args.place_z,
            place_slot=args.place_slot,
            place_release_dwell_s=args.place_release_dwell,
            place_retreat_dwell_s=args.place_retreat_dwell,
            nav_during_scan=not args.no_nav_during_scan,
            backup_after_grab_m=args.backup_after_grab,
            place_creep_m=args.place_creep_distance,
            close_recheck=not args.no_close_recheck,
            return_west_after_place=args.return_west_after_place,
            return_start_after_place=args.return_start_after_place)
        controller.configure_all_orders_completion_signal(
            args.order_id, args.completion_file)
        controller.perception_always_on = bool(args.perception_always_on)
        controller.dynamic_direct_enabled = bool(args.dynamic_direct)
        controller.configure_external_perception(args.external_perception)
        controller.configure_opportunistic_targets(args.candidate_kind)
        controller.scan_prefer_west_start = args.scan_start_west
        if args.scan_start_x is not None:
            controller.configure_inventory_scan_hint(
                args.scan_start_x, args.scan_marker_z)
        controller.excluded_marker_ids = set(args.exclude_marker_id)
        controller.excluded_slot_keys = set(args.exclude_slot_key)
        controller.excluded_slot_keys_by_kind = {}
        for value in args.exclude_kind_slot:
            kind, _, slot_key = str(value).partition("=")
            controller.excluded_slot_keys_by_kind.setdefault(
                kind, set()).add(slot_key)
        controller.configure_memory_routing(
            args.memory_file,
            args.memory_confidence_threshold,
            initial_x=args.scan_start_x,
            initial_z=args.scan_marker_z)
        if args.memory_initial_shelf is not None:
            accepted = controller.configure_direct_slot_target(
                args.memory_initial_shelf,
                args.memory_initial_level,
                args.memory_initial_column,
                product_y=args.memory_initial_product_y,
                product_z=args.memory_initial_product_z)
            if accepted:
                controller.memory_active_hint = (
                    args.memory_initial_shelf,
                    args.memory_initial_level)
                controller.memory_last_scan_station_x = args.scan_start_x
                controller.memory_reroute_not_before = time.time()
                controller.get_logger().info(
                    "[memory] runner initial direct slot accepted: "
                    f"{args.memory_initial_shelf}-"
                    f"{args.memory_initial_level}-"
                    f"{args.memory_initial_column}")
            else:
                controller.get_logger().warn(
                    "[memory] runner initial direct slot rejected; "
                    "keeping shelf-level scan hint fallback")
        if controller.excluded_marker_ids:
            controller.get_logger().info(
                "excluding markers from earlier attempts: "
                f"{sorted(controller.excluded_marker_ids)}")
        nodes = [controller]
        if args.external_perception:
            controller.get_logger().info(
                "using runner-owned persistent YOLO/ArUco perception")
        else:
            yolo_node = pick.KeleDetectNode(
                backend="yolo", pub_res_img=args.show, device=args.device,
                # Multi-order scans publish every detected class so the
                # controller can select one visible pending order.
                weights=weights,
                target_kind=(
                    None if args.formal_mode or args.candidate_kind
                    else args.target_kind),
                confidence=args.confidence, show=False,
                camera_names=("head",),
                max_inference_hz=args.max_inference_hz)
            aruco_node = pick.ArucoDetectNode(
                "head", marker_size=pick.MARKER_SIZE_M, publish_tf=False,
                publish_result_image=args.show)
            controller.configure_local_perception(yolo_node, aruco_node)
            nodes[0:0] = [yolo_node, aruco_node]
        viewer = None
        if args.show:
            if _cv_gui_available():
                viewer = pick.MainThreadResultViewer(controller)
                nodes.append(viewer)
            else:
                # Official client image ships an OpenCV without GTK/HighGUI;
                # the simulation window (server side) is still available.
                controller.get_logger().warn(
                    "OpenCV has no GUI support; skipping the YOLO window "
                    "(the server-side simulation window still shows motion)")
        for node in nodes:
            executor.add_node(node)

        if viewer is None:
            executor.spin()
        else:
            def spin_in_background():
                try:
                    executor.spin()
                except ExternalShutdownException:
                    pass

            spin_thread = threading.Thread(
                target=spin_in_background,
                name="ros2_executor", daemon=True)
            spin_thread.start()
            while rclpy.ok():
                key = viewer.show()
                if key in (ord("q"), 27):
                    controller.get_logger().info(
                        "q/Esc pressed in result window; stopping")
                    rclpy.shutdown()
                    break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:  # noqa: BLE001 - worker must report the failure
        caught_error = f"{type(exc).__name__}: {exc}"
        if controller is not None and controller.placement_completed:
            # Placement is irreversible.  A navigation/camera failure during
            # the optional return scan must not schedule the same product for
            # another grasp attempt.
            controller._post_delivery_warning(caught_error)
            controller._finish_after_return_scan(controller.now())
        else:
            raise
    finally:
        delivered = bool(
            controller is not None
            and controller.placement_completed)
        state = None if controller is None else controller.state
        phase = None if controller is None else controller.flow_phase
        marker_id = (
            None if controller is None else controller.target_marker_id)
        slot = None if controller is None else controller.target_slot()
        error = None if delivered else (
            caught_error
            or (None if controller is None else controller._fatal_error)
            or (None if controller is None else controller.terminal_error)
            or (None if controller is None else controller.abort_reason)
            or f"worker stopped in phase={phase} state={state}")
        result_document = {
            "schema_version": 1,
            "order_id": args.order_id,
            "kind": (
                args.target_kind if controller is None
                else controller.target_kind),
            "requested_kind": args.target_kind,
            "status": "delivered" if delivered else "failed",
            "marker_id": marker_id,
            "slot": None if slot is None else list(slot),
            "slot_key": (
                None if controller is None
                else controller.target_slot_key()),
            "phase": phase,
            "state": state,
            "error": error,
            "no_middle_tissue": bool(
                controller is not None
                and getattr(controller, "no_middle_tissue", False)),
            "elapsed_s": round(time.monotonic() - started_at, 3),
            "formal_mode": bool(args.formal_mode),
            "place_slot": args.place_slot,
            "place_xy": [place_x, place_y],
            "return_west_after_place": bool(
                args.return_west_after_place),
            "post_delivery_warnings": (
                [] if controller is None
                else list(controller.post_delivery_warnings)),
            "delivery_completed_by_drop": bool(
                controller is not None
                and controller.delivery_completed_by_drop),
            "drop_event": (
                None if controller is None else controller.drop_event),
        }
        if controller is not None:
            result_document.update(controller.timing_snapshot())
        _write_result(args.result_file, result_document)
        if rclpy.ok():
            rclpy.shutdown()
        try:
            executor.shutdown()
        except Exception:  # noqa: BLE001 - result is already persisted
            pass
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        for node in nodes:
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001 - best-effort worker cleanup
                pass
        if args.show:
            import cv2
            try:
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001 - headless OpenCV lacks HighGUI
                # 结果文件已写入；退出时清理窗口失败不能改变返回码，
                # 否则已交付订单会被 runner 误判为失败。
                pass
    return 0 if delivered else 2


if __name__ == "__main__":
    raise SystemExit(main())
