#!/usr/bin/env python3
"""Find one requested goods class, localise it with YOLO/optional ArUco, and pick it.

Pipeline:

1. Scan the shelf row with the head camera and YOLO ``best.pt``.
2. Keep only ``--target-kind`` YOLO boxes.
3. Lock a stable multi-frame YOLO depth position into the fixed shelf grid;
   when a synchronized ArUco is visible it remains an optional refinement.
4. Recheck the selected class and 3-D slot at close range before grasping.
5. Compose shelf level and object geometry: generic goods retain their layer
   motion; top and middle apples/oranges share a feedback-driven sphere
   approach and stable closure.  Top spheres lift with arm IK, while middle
   spheres trial-lift and fully lift with the slide before arm retreat.

``--tcp-diagnostic-ground-truth`` is an explicit fixed-layout experiment.  It
still uses YOLO and ArUco to select the product, but replaces the measured
centre with the scene's exact centre and removes perception/tuning offsets.
The fixed top-gripper TCP transform remains active because it converts the
wrist endpoint to the finger contact region.  If that experiment misses,
perception is no longer a possible cause.

The simulator server must already be running.  Example from the repo root::

    source /opt/ros/humble/setup.bash
    python3 examples/supermarket_sorting/yolo_aruco_shelf_pick.py \
        --target-kind maidong --weights best.pt --show

This example intentionally stops after removing the item from the shelf; it
does not navigate to the delivery table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, String


HERE = Path(__file__).resolve().parent
PERCEPTION_DIR = HERE / "perception"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PERCEPTION_DIR) not in sys.path:
    # perception/kele_detect.py uses ``from backends import ...``.
    sys.path.insert(0, str(PERCEPTION_DIR))

from discoverse.robots.mmk2.mmk2_fik import MMK2FIK
from mmk2_kdl import MMK2Kdl
from memory_matrix import SLOT_BY_MARKER, fixed_slot_from_world
from perception.aruco_detect import ArucoDetectNode
from perception.kele_detect import KeleDetectNode


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = HERE / "perception" / "checkpoints" / "best.pt"
FIXED_LAYOUT_PATH = HERE / "retail_competition_layout.json"


@lru_cache(maxsize=1)
def fixed_layout_by_marker():
    """Load fixed-layout truth only for explicitly requested diagnostics."""
    return {
        int(slot["aruco_id"]): slot
        for slot in json.loads(FIXED_LAYOUT_PATH.read_text())
    }

PRODUCT_CENTER_ABOVE_MARKER_M = {
    "sanmingzhi": 0.0434,
    "heweidao": -0.015,
    "shupian": 0.040,
    "zhijin": 0.043,
    "maidong": 0.034,
    "kele": 0.0215,
    "kouxiangtang": 0.020,
    "pingguo": 0.034,
    "chengzi": 0.036,
}

# Product centre heights above the shelf board surface (half heights of the
# collision geometry).  Used to bound depth-only fallback estimates: the
# measured "world Z" can land on the front/top face of a tall product and
# must never be treated as the product centre above this envelope.
# 数值来自 wxj continuous-multi-order-v2 分支（2026-08-17 合并）。
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
# Allowance above surface + half height when clamping depth-only Z estimates.
PRODUCT_CENTER_Z_TOLERANCE_M = 0.015

# Physical lateral widths from the scene collision geometry.  Gripper
# pre-shaping uses one formula for every class; these are dimensions, not
# empirically tuned grasp offsets.
PRODUCT_GRASP_WIDTH_M = {
    "sanmingzhi": 0.066,
    "heweidao": 0.096,
    "shupian": 0.066,
    "zhijin": 0.172,
    "maidong": 0.066,
    "kele": 0.053,
    "kouxiangtang": 0.049,
    "pingguo": 0.070,
    "chengzi": 0.074,
}

# Half-depth of each product along the shelf approach axis (world Y).  The
# depth deprojection at the YOLO bbox centre lands on the front surface;
# adding the half-depth recovers the product centre.  Values come from the
# scene collision geoms (cylinder radius / box half-Y / mesh half-Y / sphere
# radius), not from empirically tuned grasp offsets.
PRODUCT_HALF_DEPTH_M = {
    "sanmingzhi": 0.0500,
    "heweidao": 0.0325,
    "shupian": 0.0325,
    "zhijin": 0.0425,
    "maidong": 0.0325,
    "kele": 0.0265,
    "kouxiangtang": 0.0245,
    "pingguo": 0.0350,
    "chengzi": 0.0370,
}

# Depth-measured target selection.  The depth estimate is accepted only when
# enough synchronized frames agree, stay near the associated marker in the
# horizontal plane, and lie inside a plausible shelf-height band; otherwise
# the controller falls back to the marker + fixed-offset target.
DEPTH_TARGET_MIN_SAMPLES = 5
DEPTH_TARGET_SPREAD_MAX_M = 0.04
DEPTH_TARGET_MAX_DELTA_MS = 150.0
DEPTH_TARGET_MARKER_XY_MAX_M = 0.20
DEPTH_TARGET_Z_MIN_M = 0.40
DEPTH_TARGET_Z_MAX_M = 1.35
# Horizontal depth-to-marker distance below which the product is considered
# still in its nominal slot.  In that case the depth-measured Z is biased high
# by the downward camera angle and the marker-derived Z is biased low by the
# marker pose refinement; averaging the two cancels most of both biases.
# A displaced product (distance above this threshold) keeps the measured Z.
DEPTH_TARGET_IN_SLOT_XY_MAX_M = 0.10

# 无码锁定：YOLO 目标类在固定网格内多帧稳定即锁定，不需要 ArUco。
# ArUco 仍保留在抓取前复核里做精化，但不再作为扫描锁定的前提。
YOLO_ONLY_TARGET_CONFIRMATIONS = 4
YOLO_ONLY_TARGET_SPREAD_MAX_M = 0.09
YOLO_ONLY_TARGET_CONF_MIN = 0.80

# 固定货架几何中商品中心的世界 Y。记忆矩阵 / 无码直导路径只依赖货架几何，
# 不读取裁判 ground-truth；所有槽位的商品中心在这条固定货架平面上。
SHELF_PRODUCT_CENTER_Y_M = 3.243

# 抓取前基座停稳门控：导航“到达”后先确认底盘不再漂移，再进入臂部动作。
# 这是导航到抓取站位后最后一道安全门，不改变抓取动作本身。
GRASP_BASE_SETTLE_S = 0.20
GRASP_BASE_SETTLE_MAX_XY_M = 0.004
GRASP_BASE_SETTLE_MAX_YAW_RAD = 0.010
GRASP_BASE_SETTLE_TIMEOUT_S = 1.5

# The shelf marker is on the front rail and the product centre is behind it.
PRODUCT_BEHIND_MARKER_M = 0.075
MARKER_SIZE_M = 0.03
@lru_cache(maxsize=1)
def fixed_marker_world_by_id():
    return {
        marker_id: np.asarray(slot["world_position"], dtype=float) - np.array([
            0.0,
            PRODUCT_BEHIND_MARKER_M,
            PRODUCT_CENTER_ABOVE_MARKER_M[slot["object_kind"]],
        ])
        for marker_id, slot in fixed_layout_by_marker().items()
    }


# East-to-west scan stations, all in the unobstructed shelf approach aisle.
SCAN_X = (1.80, 0.92, 0.035, -0.85, -1.735)
SCAN_Y = 2.475
# Preserve the three established overview views.  Only if they do not bind the
# requested target do the added lower poses move the complete slide assembly
# down and sweep head yaw.  Top/middle targets therefore leave the scan before
# any lower-shelf motion is requested.
# 检索货架时只保留正面视角，不再做左右偏航（±0.15）摆动。
# 边缘列改由更近的扫描站/记忆直达/close-recheck 覆盖。
SCAN_CAMERA_POSES = (
    ("overview_high", 0.11, 0.00, -0.20),
    ("middle_center", 0.60, 0.00, 0.16),
    ("lower_center", 0.60, 0.00, -0.45),
)
# 定点补拍（revisit）：主扫描某位姿已关联到候选但样本不足时，不直接切走，
# 而是留在当前站点对目标槽位用多个角度补拍。锁定门槛完全不变（2 帧确认 +
# 5 个 marker 样本 + 4cm 扩散），只是给原有门槛更多采集机会；仅当正常扫描
# 未锁定目标时触发，不影响"样本足够直接抓取"的原路径。
REVISIT_POSES = SCAN_CAMERA_POSES
REVISIT_MAX_POSES = len(REVISIT_POSES)
REVISIT_DWELL_S = 1.0
# 补拍首姿态（已知最后关联成功的视角）驻留更长：稀疏检测槽位（如 B 货架
# 顶层薯片）往往只在特定姿态偶尔冒出一两帧，多停一会儿能显著提高补拍命中率；
# 其余兜底姿态仍按 REVISIT_DWELL_S 快速过。
REVISIT_FIRST_POSE_DWELL_S = 2.5
REVISIT_MAX_ROUNDS_PER_MARKER = 1
REVISIT_MAX_ROUNDS_PER_SCAN = 4
SCAN_CAMERA_REACHED_SLIDE_M = 0.015
SCAN_CAMERA_REACHED_HEAD_RAD = 0.030
# Shortened settle/dwell: association needs 3 synchronized confirmations and
# 5 marker samples, which complete in well under a second at the perception
# frame rate.  Previously the 2.5 s dwell per pose added ~75 s of pure wait
# per full shelf sweep.
SCAN_CAMERA_STABLE_S = 0.10
SCAN_SETTLE_S = 0.10
SCAN_DWELL_S = 0.45
# During a formal multi-order run, the first *graspable* pending class at a scan
# pose is usually much cheaper to finish than crossing several shelf stations
# for the class chosen before perception started.  Do not commit on YOLO alone:
# require the same class/ArUco pair in three distinct synchronized frames.  This
# avoids spending a revisit cycle on a visible box whose shelf marker is hidden.
# After this one-shot choice, the normal localisation, close recheck and
# single-item grasp pipeline are unchanged, and the target cannot switch again.
OPPORTUNISTIC_TARGET_CONFIRMATIONS = 3
OPPORTUNISTIC_TARGET_WINDOW_NS = 1_000_000_000
# Lower-shelf camera poses (slide down + head sweep) are only useful when the
# requested kind can actually sit on the lower shelf.  With the fixed layout
# this is known statically; --scan-skip-lower restricts the sweep to the three
# overview poses for kinds that never appear below the middle shelf.
SCAN_OVERVIEW_POSES = SCAN_CAMERA_POSES[:1]


def kind_never_on_lower_shelf(kind: str) -> bool:
    """True when every fixed-layout slot of this kind is above the lower shelf."""
    slots = [s for s in fixed_layout_by_marker().values()
             if s["object_kind"] == kind]
    return bool(slots) and all(
        s["world_position"][2] >= MIDDLE_SHELF_Z_MIN_M for s in slots)

YAW_NORTH = math.pi / 2.0
ARM_LATERAL_BIAS_M = 0.10
NAV_X_MIN = -2.05
NAV_X_MAX = 2.05
# 普通货物实机抓取普遍偏“左”（TCP 相对目标点偏东 9~36mm，用户视角为左）。
# 球体（苹果/橙子）与纸巾不应用该横向补偿，因此本次调整只影响普通货物：
# 统一把抓取目标点往右（-x）移，抵消偏左。顶层左臂不再额外往东补偿。
TOP_LEFT_GRASP_X_BIAS_M = 0.000
# drive_to cruise/rotation limits.  The old profile rotated in place whenever
# the heading error exceeded 0.18 rad and then crept at min(0.36, distance);
# measured ALIGN phases took 9--20 s for base moves under 0.35 m.  Translate
# while correcting heading once the error is moderate, keep a minimum approach
# speed, and accept the final heading with a deadband so odom yaw noise cannot
# stall the phase.
NAV_LINEAR_MAX_MPS = 0.90
NAV_LINEAR_MIN_MPS = 0.10
# 货架对齐的最后一段使用更低的进给下限：对齐容差只有 2.5cm，高速停车
# 过冲会偏移抓取位姿。仅提高距离比例增益，最后 50mm 仍保持原 0.10m/s
# 下限，因此快速收敛不改变 2.5cm 到位精度和末端制动速度。
NAV_ALIGN_LINEAR_MIN_MPS = 0.10
NAV_LINEAR_GAIN = 1.20
NAV_ALIGN_LINEAR_GAIN = 1.80
NAV_ROTATE_GATE_RAD = 0.45
# ALIGN is a short, feedback-controlled pose correction.  Let moderate
# heading error converge on an arc instead of paying for rotate -> translate
# -> rotate as three serial motions.  Large errors still rotate in place.
NAV_ALIGN_ROTATE_GATE_RAD = 0.85
# 这些转动发生在抓取前，不携货。与外层安全导航的 2.0rad/s 上限
# 对齐，缩短微调前后的转向时间；抓取后载货导航仍由外层的独立限速管理。
NAV_ANGULAR_MAX_RADPS = 2.00
NAV_TRANSLATE_ANGULAR_MAX_RADPS = 1.00
NAV_ALIGN_TRANSLATE_ANGULAR_MAX_RADPS = 1.50
NAV_YAW_DEADBAND_RAD = 0.035
# Preserve the same final-yaw deadband, but remove the long exponential tail
# caused by the old 2.0 * error command under a slow simulator real-time rate.
NAV_FINAL_YAW_GAIN = 3.50
# 原地旋转卡死恢复：旋转指令发出后若 yaw 长时间无变化（被西墙/货架顶住），
# 先短距离倒车解除卡死再继续旋转，避免在西侧货架贴墙处永久卡住。
NAV_ROT_STALL_S = 2.5            # 旋转无进展判定时间（秒）
NAV_ROT_STALL_MIN_CHANGE_RAD = 0.03
NAV_ROT_UNSTICK_DIST_M = 0.15    # 解除卡死时的倒车距离
NAV_ROT_UNSTICK_SPEED_MPS = 0.08
NAV_ROT_UNSTICK_TIMEOUT_S = 4.0
NAV_ROT_UNSTICK_MAX = 3          # 最多解除次数，超过后中止当前尝试
# A table-to-shelf route may legitimately consume the navigator's full 150 s
# leg budget under the measured simulator real-time rate.  Leave room for the
# final camera posture/stability gate while retaining a finite state exit.
GO_SCAN_HARD_TIMEOUT_S = 180.0
# Direct inventory localisation can legitimately move ALIGN to an adjacent
# shelf (roughly 1 m) before the centimetre-scale correction.  At the measured
# simulator real-time rate that takes about 30 s while making steady progress,
# so retain a finite failure exit without rejecting that valid path.
ALIGN_HARD_TIMEOUT_S = 45.0

ARUCO_SYNC_TOLERANCE_NS = 200_000_000
ARUCO_MAX_VERTICAL_GAP_BOX_HEIGHTS = 1.50
ARUCO_MAX_VERTICAL_GAP_MIN_PX = 65.0
ARUCO_MAX_HORIZONTAL_MARGIN_BOX_WIDTHS = 0.60
ARUCO_PRODUCT_LEVEL_TOLERANCE_M = 0.16
FIXED_MARKER_POSITION_TOLERANCE_M = 0.12
ASSOCIATION_CONFIRMATIONS_REQUIRED = 2
MARKER_SAMPLES_REQUIRED = 5
MARKER_SAMPLE_SPREAD_MAX_M = 0.04

# Close-range verification before committing the arm.  Far-view YOLO and
# ArUco association remains the localisation source; this second view only
# verifies that the requested class is still present at the selected slot.
CLOSE_RECHECK_CONFIRMATIONS = 2
CLOSE_RECHECK_WINDOW_S = 2.0
CLOSE_RECHECK_POSE_TIMEOUT_S = 3.0
REVISIT_POSE_COMMAND_TIMEOUT_S = 5.0
CLOSE_RECHECK_XY_MAX_M = 0.12
CLOSE_RECHECK_Z_MAX_M = 0.16

SLIDE_REFERENCE_Z_M = 0.9235
SLIDE_REFERENCE_COMMAND = 0.11
SLIDE_MIN = -0.04
SLIDE_MAX = 0.60
PREGRASP_BACKOFF_M = 0.18
TOP_SHELF_Z_M = 1.10
MIDDLE_SHELF_Z_MIN_M = 0.75
# Top-shelf goods sit above the slide joint's usable height range, so the arm
# itself raises the TCP to the product centre.  Keep enough chassis clearance
# for the elbow, form a front-facing pregrasp at the same height, and make one
# horizontal insertion.  This avoids the low-tolerance pitched descent and its
# shoulder-branch change.
TOP_GRASP_CENTER_DISTANCE_M = 0.69
TOP_PREGRASP_BACKOFF_M = 0.12
# The short top goods previously drove the wrist TCP only 26--28 mm above the
# 1.189 m shelf surface, while the collision-free cola grasp measured about
# 63 mm.  Use one shelf-geometry clearance rule instead of class-specific Z
# offsets.  A 70 mm commanded clearance leaves roughly 60 mm after the
# observed ~9 mm tracking sag.  Tall goods such as cola already exceed it and
# are therefore unchanged.
TOP_SHELF_SURFACE_Z_M = 1.189
TOP_MIN_TCP_TARGET_CLEARANCE_M = 0.070
LOWER_PREGRASP_BACKOFF_M = 0.16
LOWER_GRASP_TCP_FORWARD_M = 0.035
# Every supported profile first reaches its established close point with the
# gripper open, then keeps the same endpoint orientation and executes one
# additional monotonic 50 mm arm segment before closing.  Generic profiles
# solve both endpoints during configuration; sphere profiles build the second
# endpoint once from the measured physical-contact pose.
GENERIC_POST_CONTACT_EXTENSION_M = 0.050
# 按货物类型覆盖接触后前伸量（米）：薯片罐用 25mm——既让罐子坐进夹爪
# 更深、夹得更稳，又避免 50mm 长前伸把罐子推倒。
GENERIC_POST_EXTEND_M_BY_KIND = {"shupian": 0.025}
# 核桃味刀越靠下夹持越稳定。利用已有的闭合前延伸段同步下降 10mm，
# 不增加独立状态或停顿；其余商品保持水平前伸。
GENERIC_POST_EXTEND_Z_DROP_M_BY_KIND = {"heweidao": 0.010}
# The extended middle/lower endpoint is outside the analytic arm envelope from
# SCAN_Y once navigation tolerance is included.  Move only those non-spherical
# profiles closer; sphere alignment and the top profile remain unchanged.
GENERIC_EXTENSION_ALIGN_FORWARD_M = 0.040
# The generic middle/lower front endpoints sit near the selected arm's
# workspace edge: a ~1.5 cm base arrival deviation can push the IK chain past
# the boundary.  On failure, nudge the base forward and re-drive instead of
# aborting immediately.
GENERIC_IK_RETRY_STEP_M = 0.03
GENERIC_IK_RETRY_MAX_M = 0.06
GENERIC_ALIGN_Y_MAX_M = 2.60
LOWER_LIFT_M = 0.04
LOWER_LIFT_DWELL_S = 1.2
LOWER_RETREAT_DWELL_S = 2.5
# The generic middle/top retreat streams to the pregrasp joints solved at the
# align pose.  The mobile base is not held during middle manipulation, so a
# small base shift can leave one joint a few hundredths of a radian short of
# the commanded vector forever.  Treat that as a completed retreat only when
# the measured TCP is clearly south of the shelf front rail.
GENERIC_RETREAT_TIMEOUT_S = 8.0
GENERIC_RETREAT_CLEAR_MARGIN_M = 0.02
# The mobile base is velocity controlled, so publishing zero velocity does not
# make it rigid against arm contact forces.  During top/middle/lower
# manipulation keep the measured post-alignment longitudinal pose and yaw with
# a soft, saturated odometry loop.  This does not gate, pause or replan the arm
# trajectory.
MANIP_BASE_HOLD_LINEAR_KP = 2.0
MANIP_BASE_HOLD_LINEAR_MAX_MPS = 0.10
MANIP_BASE_HOLD_LINEAR_DEADBAND_M = 0.003
MANIP_BASE_HOLD_YAW_KP = 2.0
MANIP_BASE_HOLD_YAW_MAX_RADPS = 0.30
MANIP_BASE_HOLD_YAW_DEADBAND_RAD = 0.005
MANIP_BASE_HOLD_LOG_PERIOD_S = 0.50
SPHERE_PREGRASP_BACKOFF_M = 0.12
# The IK endpoint is at the wrist, while the useful finger contact region is
# behind it along the approach axis.  This top-profile-only transform places
# the product between the fingers instead of driving the palm into its centre.
# It remains active in ground-truth diagnostic mode because it is gripper
# geometry, not a perception correction.
TOP_GRASP_TCP_FORWARD_M = 0.035
# Parallel-jaw sphere geometry is composed with a shelf-level motion profile.
# Top and middle spheres share their approach/contact/closure logic; the layer
# selects base alignment and whether the arm or slide performs trial/full lift.
# Lower-shelf sphere support remains intentionally disabled until that layer is
# implemented and tested separately.
SPHERE_RADIUS_M = {"pingguo": 0.035, "chengzi": 0.037}
SPHERE_FINGER_ENGAGEMENT_M = 0.012
SPHERE_FAST_SPEED_MPS = 0.12
SPHERE_TERMINAL_SPEED_MPS = 0.055
SPHERE_TERMINAL_ZONE_M = 0.045
SPHERE_OPEN_GRIP_DROP_CONTACT = 0.08
SPHERE_CONTACT_CREEP_SPEED_MPS = 0.020
SPHERE_CONTACT_CREEP_TIMEOUT_S = 1.5
SPHERE_CREEP_CAPTURE_GRIP_DROP = 0.15
SPHERE_CREEP_MIN_ADVANCE_M = 0.002
SPHERE_CREEP_STALL_WINDOW_S = 0.40
SPHERE_CREEP_STALL_MIN_SPAN_S = 0.30
SPHERE_CREEP_STALL_RANGE_M = 0.0015
SPHERE_CREEP_CONTACT_TOLERANCE_M = np.array([0.006, 0.010, 0.015])
SPHERE_TCP_TOLERANCE_M = np.array([0.006, 0.010, 0.010])
MIDDLE_SPHERE_CREEP_GOAL_TOLERANCE_M = np.array([0.006, 0.002, 0.006])
SPHERE_FORWARD_TIMEOUT_S = 3.0
SPHERE_GRIP_MIN_CAPTURE_POSITION = {"pingguo": 0.50, "chengzi": 0.55}
SPHERE_GRIP_STABILITY_WINDOW_S = 0.50
SPHERE_GRIP_STABILITY_MIN_SPAN_S = 0.40
SPHERE_GRIP_STABILITY_RANGE = 0.020
SPHERE_CLOSE_SAMPLE_AFTER_S = 1.0
SPHERE_CLOSE_TIMEOUT_S = 3.0
SPHERE_TRIAL_LIFT_M = 0.010
SPHERE_TRIAL_LIFT_TIMEOUT_S = 3.0
SPHERE_LIFT_M = 0.04
SPHERE_LIFT_TIMEOUT_S = 4.0
# The measured-TCP correction must not sit behind a stricter convergence gate
# than the normal controller.  These tolerances merely decide when it is safe
# to take a useful TCP sample; the Cartesian check below remains authoritative.
MIDDLE_SPHERE_CORRECTION_ARM_TOLERANCE_RAD = 0.025
MIDDLE_SPHERE_CORRECTION_SLIDE_TOLERANCE_M = 0.012
MIDDLE_SPHERE_PREGRASP_TCP_TOLERANCE_M = np.array([0.008, 0.010, 0.008])
MIDDLE_SPHERE_SLIDE_CORRECTION_MAX_STEP_M = 0.015
MIDDLE_SPHERE_SLIDE_CORRECTIONS_MAX = 2
# A small residual pregrasp error is not a reason to throw away an otherwise
# useful experiment.  Continue with a warning after the soft deadline.  The
# wider hard envelope only rejects a missing or clearly implausible pose.
MIDDLE_SPHERE_DEPLOY_SOFT_CONTINUE_S = 5.0
MIDDLE_SPHERE_PREGRASP_SOFT_LIMIT_M = np.array([0.025, 0.035, 0.025])
MIDDLE_SPHERE_PREGRASP_HARD_LIMIT_M = np.array([0.060, 0.080, 0.060])
MIDDLE_SPHERE_DEPLOY_TIMEOUT_S = 7.0
# Move the wrist slightly past the inferred product centre so the fingers,
# rather than only their tips, surround the product before closing.  The base
# remains stationary during this motion.  Only the pregrasp and contact IK
# endpoints are generated; the controller streams continuously between them.
FRONT_GRASP_OVERSHOOT_M = 0.03
ARM_COMMAND_MAX_STEP_RAD = 0.022
# 收回阶段（STATE_RETREAT）使用更小的关节步长：抓取后带着货物往回撤时
# 速度放慢约一半，避免快速收臂让货物晃动/刮擦货架。
RETREAT_ARM_MAX_STEP_RAD = 0.010
# Every product profile uses the same measured Cartesian terminal slowdown.
# This is a command-rate limit, not a per-class position correction: outside
# the final 50 mm the arm keeps its current speed, then smoothly ramps down to
# the smaller joint step as the TCP approaches the contact target.
FORWARD_TERMINAL_ZONE_M = 0.050
FORWARD_TERMINAL_ARM_STEP_RAD = 0.006
ARM_REACHED_TOLERANCE_RAD = 0.025
ARM_READY_SETTLE_S = 0.08
DEPLOY_TIMEOUT_S = 5.0
ABORT_SHUTDOWN_TIMEOUT_S = 4.0
LIFT_AMOUNT_M = 0.06

# Generic goods use one fixed trajectory after visual localisation.  The
# controller solves the contact endpoint once, then plays a monotonic eased
# joint interpolation without measured-TCP correction, convergence gates, or
# replanning.  Keeping the peak Cartesian-equivalent speed low prevents light
# goods from being struck while eliminating feedback-induced shaking.
GENERIC_DIRECT_FORWARD_SPEED_MPS = 0.036
GENERIC_DIRECT_FORWARD_MIN_DURATION_S = 2.5
GENERIC_DIRECT_FORWARD_SETTLE_S = 0.5
GENERIC_DIRECT_FORWARD_MIN_SETTLE_S = 0.20
# A single feedback sample can cross the endpoint tolerance while the arm is
# still ringing.  Early completion therefore requires a short continuous
# in-tolerance window; the original fixed settle remains the hard fallback.
MOTION_ENDPOINT_STABILITY_S = 0.12
# The fixed deploy dwell is a safety ceiling; with the convergence gate the
# arm normally starts forward as soon as the pregrasp is reached.
GENERIC_DIRECT_DEPLOY_DWELL_S = 2.0
GENERIC_DEPLOY_SOFT_ARM_TOLERANCE_RAD = 0.12
GENERIC_DEPLOY_HARD_TIMEOUT_S = 8.0
# 部署期 pregrasp 未收敛时，向前微调基座、回到 ALIGN 重新解算并重试，
# 而不是直接放弃整单（避免整单重扫 + 重新定位 + 重复核）。达到上限仍
# 失败才中止。
GENERIC_DEPLOY_RETRY_MAX = 2
GENERIC_DEPLOY_RETRY_STEP_M = 0.02
GENERIC_TOP_LIFT_M = 0.045
GENERIC_TOP_LIFT_TIMEOUT_S = 5.0
# Middle-layer goods are already captured when this state runs.  Give the
# slide substantially longer than the normal convergence window, then prefer
# a horizontal retreat at the measured height over waiting indefinitely or
# opening the gripper in STATE_ABORT.
GENERIC_MIDDLE_LIFT_TIMEOUT_S = 10.0

# A tissue box is 172 mm wide, more than twice one gripper's 80 mm opening.
# In the fixed layout every tissue target is on the middle shelf, so a separate
# symmetric two-arm profile uses the closed grippers as padded side supports:
# reach around both sides, move inward together, lift with the slide, and
# retreat together while maintaining the lateral squeeze.
DUAL_TISSUE_PREGRASP_BACKOFF_M = 0.160
# Stop the first top-shelf pull with the box centre safely behind the 3.173 m
# board edge.  A 120 mm backoff put the measured centre at y=3.112 m, so the
# box fell as soon as one side released.  A 52 mm backoff keeps the centre of
# mass just inside the board edge while exposing enough of its front lip for a
# support bar that is kept safely in front of the shelf.
DUAL_TISSUE_TOP_EDGE_BACKOFF_M = 0.052
# During the old in-place wrist rotation, link6's long collision box swept
# through both the tissue and the L3 board.  The low unroll below removes that
# vertical sweep collision; this depth leaves about 20 mm of the horizontal
# bar below the exposed tissue lip while its rear tip stays before the board.
DUAL_TISSUE_TOP_FORK_FRONT_BACKOFF_M = 0.185
DUAL_TISSUE_TOP_FORK_RELEASE_M = 0.025
DUAL_TISSUE_TOP_FORK_BAR_LATERAL_OFFSET_M = 0.070
DUAL_TISSUE_TOP_FORK_CENTRE_M = 0.215
# The rolled-to-horizontal sweep reaches about 46 mm above the commanded TCP.
# Put that complete sweep below the 1.169 m board underside, then raise the
# already-horizontal bar to the original support height.
DUAL_TISSUE_TOP_FORK_TCP_BELOW_SURFACE_M = 0.075
DUAL_TISSUE_TOP_FORK_PRELOAD_M = 0.050
DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M = 0.030
# Route above the tissue before crossing laterally, then descend well behind
# it.  This makes the left wrist a true centred rear stop instead of a side
# contact that lets the box remain on the shelf when the chassis retreats.
DUAL_TISSUE_TOP_FORK_PUSH_M = 0.120
DUAL_TISSUE_TOP_FORK_PUSHER_RELEASE_M = 0.040
# 原 0.220 m 的后侧余量让左腕在 stage 9 的目标点 y 达到约 3.460 m，超出
# 左臂在 slide=0.05 m 时的 KDL 可达空间，导致 support trajectory IK failed。
# 收紧到 0.160 m 后，仍位于纸巾盒后表面外侧约 117 mm，足以完成越顶绕行，
# 同时让最后 push 位姿落到盒体后表面附近，而不是停在盒后方 57 mm 处。
DUAL_TISSUE_TOP_FORK_PUSHER_REAR_CLEARANCE_M = 0.160
DUAL_TISSUE_TOP_FORK_PUSHER_OVERHEAD_M = 0.060
DUAL_TISSUE_TOP_FORK_PUSHER_X_LEAD_M = 0.055
# Link5/link6 contact is about 28 mm below the commanded TCP.  Lowering this
# pose places that contact near the box centre instead of its top edge, which
# otherwise creates enough pitch torque to tip the tissue backward.
DUAL_TISSUE_TOP_FORK_PUSHER_LOWER_M = 0.045
DUAL_TISSUE_TOP_FORK_LIFT_M = 0.075
DUAL_TISSUE_TOP_FORK_SLIDE_M = 0.080
# 平转 90°（2026-08-18 第三版）：探入前先把纸巾盒在货架平面内转 90°，
# 让长边（172mm）从前向横向变为前后纵深，短边（85mm）面向机器人。
# 左臂先贴住纸盒左侧作为支点（只横向抵住、不前后推），右臂手侧面对准
# 中心偏右的前脸位置持续前伸——单侧推力绕左支点形成 CCW 力矩，避免
# 单臂直推把整个纸盒向后平移（实测整体后移，转不动）。
# 重新启用平转 90° 预调整：旋转后再进入双臂抓取。
TISSUE_ROTATE_ENABLED = True
TISSUE_ROTATE_ANCHOR_SPAN_M = 0.065
TISSUE_ROTATE_RIGHT_OFFSET_M = 0.070
TISSUE_ROTATE_PUSH_M = 0.160
TISSUE_ROTATE_SPEED_MPS = 0.015
TISSUE_ROTATE_DWELL_S = 1.5
# 转完 90° 后纸盒面向机器人宽度只剩 85mm（半宽 0.0425m），双臂探入/夹持
# 半跨度相应收窄。
TISSUE_ROTATED_PROBE_HALF_SPAN_M = 0.065
TISSUE_ROTATED_CLAMP_HALF_SPAN_M = 0.050
# 前立柱绕过：货架前立柱是 40mm 方块，顶高 1.30m。侧列（C1/C3）双臂
# 宽跨度直探会在立柱高度带横穿立柱，导致手臂被挡住。保持手侧面姿态
# （不旋转腕部），先抬到立柱顶以上，越过立柱 y 带后再下降。
DUAL_TISSUE_POST_TOP_Z_M = 1.300
DUAL_TISSUE_POST_CLEARANCE_M = 0.025
DUAL_TISSUE_POST_Y_CLEAR_M = 0.045
# 侧列越柱改用“横向收窄→窄跨度前伸→到位后张开”，不抬升高度：垂直抬升
# 会让 KDL 选到远分支（右臂 joint4/5/6 一次跳 3 rad），实机手臂甩飞。
DUAL_TISSUE_POST_NARROW_SPAN_M = 0.065
# 侧列绕柱狗腿：立柱外侧绕行半跨度。左/右臂先移到立柱外侧，越过立柱 y
# 带后再回到纸盒侧边，避开“再偏左就撞柱”的几何上限。
DUAL_TISSUE_POST_OUTER_SPAN_M = 0.280
# 侧列取出：抬升后水平撤退的前几秒让底盘向远离立柱的一侧横向漂移，
# 宽跨度横条不用经过立柱就能退出货架。
TISSUE_POST_RETREAT_LATERAL_MPS = 0.010
TISSUE_POST_RETREAT_SHIFT_S = 5.0
# 探入深度：TCP 只需越过纸盒中心 +5mm——手指（钳子）覆盖纸盒后部约
# 38mm 接触纸盒侧面，而 link6 碰撞盒前缘（TCP 前方 55mm）距纸盒后表面
# 仍有 17mm——"钳子接触纸巾而不是臂膀"（用户反馈探入太深、臂膀碰纸盒，
# 原 +30mm 深度下 link6 前缘距纸盒后表面 42mm 仍有余量，但检测误差/
# 纸盒偏移下偏深，统一降到 +5mm 并把夹持靠前）。
# 探入深度：TCP 越过纸盒中心 5mm，保持“钳子接触纸巾而不是臂膀”。
DUAL_TISSUE_INSERT_FORWARD_M = 0.005
# Keep the complete gripper bodies clear during insertion, not merely their
# TCP centres.  The 140 mm surround half-span leaves 54 mm around a nominal
# 86 mm tissue half-width, so the measured ~20 mm lateral vision error does
# not turn one arm into an early frontal contact.
DUAL_TISSUE_PREGRASP_HALF_SPAN_M = 0.150
# "直接探入"的初始双臂间距（单侧半跨度）：0.105m——手侧面（0° 滚转，
# 手指竖直）姿态下 link6 碰撞盒 160mm 横向，横条外缘 = span+0.08，
# 距立柱内缘（0.19m）余量 5mm——侧面探入全程不旋转腕部（用户要求
# "一开始探入就是 90° 旋转后的样子，不再做额外旋转"），同时手指大面
# 在探入时就插进纸盒两侧约 13mm（纸盒被对称挤压居中），到位后合拢
# 深压夹持。
DUAL_TISSUE_DIRECT_PROBE_SPAN_M = 0.105
DUAL_TISSUE_SURROUND_HALF_SPAN_M = 0.105
# 侧列的邻侧探入半跨度（与立柱侧一致：0.105，link6 横条立柱余量 5mm）。
DUAL_TISSUE_NEIGHBOUR_PROBE_SPAN_M = 0.140
# 侧列立柱侧探入半跨度：比总跨度的一半收 5mm，使左/右臂横条整体远离
# 前立柱（横条外缘从贴柱变为约 5mm 余量），同时邻货侧放 5mm。
DUAL_TISSUE_POST_SIDE_SPAN_M = 0.140
# Near a side-column box, keep the rolled wrists slow enough that position
# tracking cannot overshoot the roughly 29 mm lateral clearance.
DUAL_TISSUE_SIDE_ROLLED_MAX_STEP_RAD = 0.005
# 侧滚腕专用探入深度：固定板（TCP 前 55~85mm）的接触带中心在
# insert_y-0.07 处，insert=0.025 时接触带中心落在盒心附近（约前 2.5mm），
# 从"抓前端"改为"抓中间"，固定板对盒侧整面贴合（85mm 深）。该值只
# 作用于侧滚腕路径，中列手侧面路径保持原 5mm。
DUAL_TISSUE_SIDE_ROLLED_INSERT_FORWARD_M = 0.025
# 场景边缘墙壁（x=±2.47）避让：目标列贴近墙壁时，把对齐基座向货架中心
# 平移，避免双臂部署/预抓路径扫墙（实测 A C1 左臂扫西墙 21mm，东移
# 0.06m 后干净；E C3 右臂扫东墙 91mm，西移 0.12m 后干净）。
DUAL_TISSUE_WALL_CLEAR_X_THRESHOLD_M = 1.90
DUAL_TISSUE_WEST_WALL_SHIFT_M = 0.06
DUAL_TISSUE_EAST_WALL_SHIFT_M = 0.12
# 最终夹持半跨度（侧面大面夹持，过盈约 27mm/侧）。
DUAL_TISSUE_CLAMP_HALF_SPAN_M = 0.090
# Default tissue TCP shelf clearance; top and lower levels have additional
# independently tuned vertical offsets below.
# 中层/下层探入高度按"手托托底"标定（复刻 20260817 153144 顶层成功局
# 的几何）：link6 碰撞盒在 TCP 垂直下方 70mm（垂直半宽 15mm），探入
# 高度取 板面+8.5cm，使碰撞盒下缘恰好贴板（153144 顶层即 -1mm 贴板并
# 成功），闭合手指底端（TCP-70mm）位于纸盒底（板面）上方约 15mm——
# 纸盒悬空时只需下滑约 15mm 即坐落在手指底端，形成"侧面夹持 + 底部
# 托住"的稳定抱持。旧的 1.5cm/9cm 净空让纸盒需要下滑 55mm/40mm，
# 抬升时纸盒在手指间长距离滑落，会歪斜/掉出（实测"没举起纸巾、纸盒
# 留在货架上"）。
DUAL_TISSUE_TCP_CLEARANCE_M = 0.085
# Horizontal wrists place their collision box 70 mm below the TCP.
# 恢复 20260817 153144 顶层成功局的高度：raise 0.035（命令 TCP 1.283），
# 手指底端 1.213 位于纸盒底（1.199）上方约 14mm，纸盒悬空下滑 14mm 即
# 坐落在手指底端（"夹+托"）。0.055 让下滑量达 34mm，抬升时纸盒长距离
# 滑落易掉出。
DUAL_TISSUE_TOP_TCP_RAISE_M = 0.035
# E-L3-C1 puts the nominal 105 mm left probe only 10 mm from the front
# upright.  The hand-side (unrolled) pose keeps link6's 160 mm collision
# width at span+0.08 = 185 mm from the box centre, leaving 5 mm to the
# post inner face (0.19 m) — the side pose inserts directly (no overhead
# rise/widen/descend and no wrist roll): the overhead path forced the
# right wrist joint6 to jump branches (~326 deg) and look like it was
# rotating constantly, while the direct probe keeps joint6 in the same
# branch the whole way.
DUAL_TISSUE_TOP_DIRECT_PROBE_SPAN_M = 0.105
DUAL_TISSUE_TOP_WRIST_ROLL_RAD = math.pi / 2.0
DUAL_TISSUE_ENDPOINT_TCP_TOLERANCE_M = 0.018
# Side-column narrow-wrist profile.  Both wrists roll outside the shelf so the
# fixed link6 plates present their narrow horizontal section to the front post;
# the same orientation is retained through insertion, clamp, lift and retreat.
DUAL_TISSUE_SIDE_ROLLED_ENABLED = True
DUAL_TISSUE_SIDE_ROLLED_TCP_CLEARANCE_M = 0.100
DUAL_TISSUE_SIDE_ROLLED_SQUEEZE_M = 0.025
DUAL_TISSUE_SIDE_ROLLED_MAX_SEGMENT_JOINT_DELTA_RAD = 0.90
# 手背（outward）探入到位后，旋转 90° 到“手侧面”夹持姿态的路径长度
# 与速度。旋转点位于纸盒后侧之外，不会扫到立柱/邻货。
DUAL_TISSUE_UNROLL_PATH_M = 0.120
DUAL_TISSUE_UNROLL_SPEED_MPS = 0.050
# On the lower shelf the arms reach further down and their solved joints sag
# below the target Z by 10-25 mm; the left arm then drags on the board and
# stalls 50-70 mm short, skewing the contact line by ~12 deg.  Use a larger
# clearance on the lower shelf so the fingers stay above the board.  FK
# diagnosis showed the closed grippers ended ~25-45 mm ABOVE the tissue box
# top (wrist bodies resting on the top corners, box never lifted).  Lower the
# contact height so the wrist/fingers engage the box side below its top: with
# a 0.10 clearance the commanded TCP is ~0.599 m, the sim-real ~0.574 m, below
# the 0.587 m box top and above the 0.499 m board.
# 下层与中层一致按"手托托底"标定：探入高度 = 板面 + 8.5cm（碰撞盒下缘
# 贴 L1 板面，同 153144 顶层成功几何），纸盒悬空下滑约 15mm 坐落在
# 手指底端。此前的 9cm 净空让纸盒下滑 40mm，抬升时易从手指间掉出。
DUAL_TISSUE_LOWER_TCP_CLEARANCE_M = 0.085
# The rolled-wrist overhead path needs elbow room during its vertical rise.
# At the old +40 mm forward-compensated pose (base y ~= 2.58), both wrist joints
# saturated and pulled the TCPs inward by 30--50 mm.  The repeatedly successful
# pose was y ~= 2.53, so stand 20 mm behind the generic top approach instead.
DUAL_TISSUE_ALIGN_FORWARD_M = -0.020
# 夹爪完全闭合（0 为最紧），配合侧夹预压增大，避免纸盒从指间滑脱。
DUAL_TISSUE_GRIP_COMMAND = 0.0
DUAL_TISSUE_FORWARD_SPEED_MPS = 0.036
DUAL_TISSUE_CLOSE_SPEED_MPS = 0.05
DUAL_TISSUE_CONTACT_SEARCH_HALF_SPAN_M = 0.045
DUAL_TISSUE_CONTACT_SEARCH_SPEED_MPS = 0.015
# 方案 A2 的接触搜索目标（"一侧停靠、一侧慢推"，20260817 153144 机制）：
# 右臂先合拢到停靠跨度（span 0.085，手指内面距纸盒右缘约 32mm，不碰
# 纸盒）并停住；左臂继续慢速推进（上限 span 0.040），手指压住纸盒左缘
# 把纸盒推向右侧，纸盒顶到停靠的右手指后被双侧夹住，左臂 stall 即真实
# 接触（纸盒被两侧手指挤压，不会再打滑）。双侧同时合拢会让纸盒被推着
# 打滑、手指永远合拢不到位（实测空夹）。
DUAL_TISSUE_PARK_SPAN_M = 0.085
DUAL_TISSUE_PUSH_SPAN_M = 0.040
# Minimum inward travel before a grip/stall contact signal is accepted.  The
# measured travel to the box side varies widely with the box's lateral offset
# (18-70 mm observed), so keep the bar low; the measured-centre anchor below
# is what makes the final clamp symmetric.
DUAL_TISSUE_CONTACT_MIN_ADVANCE_M = 0.006
DUAL_TISSUE_CONTACT_COMMAND_LEAD_M = 0.004
DUAL_TISSUE_CONTACT_STALL_WINDOW_S = 0.55
DUAL_TISSUE_CONTACT_STALL_MIN_SPAN_S = 0.40
DUAL_TISSUE_CONTACT_STALL_RANGE_M = 0.0015
DUAL_TISSUE_CONTACT_ENDPOINT_TOLERANCE_M = 0.003
# A closed unloaded gripper tracks 0.08 in prior runs; side contact in the
# successful-but-unstable run forced both measured joints down near 0.012.
DUAL_TISSUE_GRIP_CONTACT_MAX = 0.045
# 侧夹预压拉到 25mm（接近压力上限）：双臂合拢后继续内压，纸盒挡住即由
# 位置控制器持续加压，扭矩到执行器上限封顶，不会导致物理发散。
# 双臂闭合时的单侧预压量（米）：从实测接触位置再向内压入的量。加大后
# 两侧钳爪对纸盒的夹紧力更强，纸盒不易在抬/撤过程中滑动。
# The previous 30 mm/side position preload kept driving after contact and
# toppled a tissue box.  25 mm keeps the clamp firm without toppling;
# the following clamp pose retains roughly 4 mm/side geometric interference.
DUAL_TISSUE_SQUEEZE_M = 0.025
DUAL_TISSUE_SQUEEZE_SPEED_MPS = 0.012
DUAL_TISSUE_RETREAT_SPEED_MPS = 0.018
DUAL_TISSUE_MIN_MOTION_DURATION_S = 2.0
DUAL_TISSUE_MOTION_SETTLE_S = 0.75
DUAL_TISSUE_MOTION_MIN_SETTLE_S = 0.20
# The old deploy transition advanced after a fixed two-second dwell even when
# one measured arm was still short of the symmetric pregrasp.  Keep the same
# minimum dwell, but give a lagging arm a bounded feedback-driven window to
# reach the pose before the insertion trajectory starts.
DUAL_TISSUE_DEPLOY_DWELL_S = 2.0
DUAL_TISSUE_DEPLOY_TIMEOUT_S = 5.0
DUAL_TISSUE_CLAMP_DWELL_S = 4.0
DUAL_TISSUE_LIFT_M = 0.060
# The slide cannot raise a top-shelf grasp, so the arms perform a shorter
# guarded lift before retreating at the raised height.
DUAL_TISSUE_TOP_ARM_LIFT_M = 0.035
# Lift a tissue slowly enough for the side friction contacts to carry its mass.
# The old one-step 60 mm joint target completed in ~0.5 s and left the box on
# the shelf even though both arms had established a symmetric side contact.
DUAL_TISSUE_ARM_LIFT_SPEED_MPS = 0.012
# A single 60 mm Cartesian IK solve can select a distant redundant wrist
# branch even when the starting clamp pose is sound.  In simulation that made
# the right wrist retain a 1.04 rad error and cross almost 220 mm toward the
# left arm.  Plan the lift and the raised retreat as short, branch-continuous
# waypoints, and reject any segment that still asks for a large joint jump.
DUAL_TISSUE_ARM_LIFT_STEP_M = 0.015
DUAL_TISSUE_ARM_LIFT_MIN_CLEARANCE_M = 0.025
DUAL_TISSUE_ARM_RETREAT_STEP_M = 0.055
DUAL_TISSUE_ARM_SEGMENT_MAX_JOINT_DELTA_RAD = 0.45
DUAL_TISSUE_TOP_MIDDLE_LIFT_INWARD_PRELOAD_M = 0.002
# 顶层侧滚腕抬升撤退限速 0.018（与中层/下层手侧面撤退同速）。实测 0.030
# 下带载撤退时双侧手臂差约 0.07 rad 不收敛（左臂 TCP 差 50mm），10s 超时
# 中止丢盒；放慢后跟踪误差显著减小（2026-08-24 顶层 A C1 现场复现）。
DUAL_TISSUE_TOP_ARM_RETREAT_SPEED_MPS = 0.018
DUAL_TISSUE_ARM_LIFT_Z_TOLERANCE_M = 0.006
DUAL_TISSUE_LIFT_DWELL_S = 2.5
DUAL_TISSUE_SLIDE_LIFT_TOLERANCE_M = 0.015
DUAL_TISSUE_SLIDE_LIFT_STABLE_S = 0.25
DUAL_TISSUE_SLIDE_LIFT_TIMEOUT_S = 5.5
DUAL_TISSUE_LIFT_SLIDE_STEP_M = 0.0015
DUAL_TISSUE_RETREAT_DWELL_S = 2.0
# The dual side-clamp solves both arm endpoints at the level-adjusted TCP
# height.  Unlike the single-arm front profiles, moving the base closer to the
# shelf shrinks the lateral room around the box and breaks the IK, so a failed
# dual solve backs the base up a step instead of driving forward.
SHELF_SURFACE_Z_M = {"top": 1.189, "middle": 0.851, "lower": 0.499}
DUAL_IK_RETRY_STEP_M = 0.03
DUAL_IK_RETRY_MAX_M = 0.06
DUAL_ALIGN_Y_MIN_M = 2.42
DUAL_ALIGN_Y_MAX_M = 2.58
# A dual side clamp can only hold a tissue box that is square to the approach.
# When one arm is blocked early (shelf board, rotated box corner), the two
# contact TCPs end up skewed; the old code completed such a grasp and reported
# success while the box was only caught on one side.  Gate on the line angle
# between the two grippers and abort instead of faking a straight hold.
DUAL_TISSUE_MAX_LINE_ANGLE_DEG = 8.0
# 墙侧列跳过已废弃（wxj v2 全列统一直接探入，墙侧列也获得足够横向余量）。
MIDDLE_SHELF_SURFACE_Z_M = 0.851

GRIP_OPEN = 1.0
# Sphere handling keeps the established close target.  Generic goods receive
# a deeper target and a longer force-building dwell before lift so the fingers
# finish seating the object instead of continuing to close during retreat.
GRIP_CLOSE = 0.08
GENERIC_GRIP_CLOSE = 0.0
GRIP_CLOSE_BY_CLASS = {"sanmingzhi": 0.16}
GRIP_OPEN_MAX_STEP = 0.025
GRIP_CLOSE_MAX_STEP = 0.006
GENERIC_CLOSE_DWELL_S = 6.0
GENERIC_EMPTY_GRIP_MARGIN = 0.02
# Generic close is gated on the measured gripper instead of a fixed 6 s dwell:
# a held product stops jaw travel once it is seated, and a stability window
# confirms that before lift.  The old dwell kept running ~3 s after the jaws
# had stopped moving.  Stage 1 (gentle intermediate opening) still runs first
# so cylinders can settle before the final squeeze; both stages keep their
# time-based ceilings as safety.
GENERIC_CLOSE_STAGE1_MIN_S = 0.6
GENERIC_CLOSE_STAGE1_STABLE_WINDOW_S = 0.35
GENERIC_CLOSE_STAGE1_STABLE_RANGE = 0.020
GENERIC_CLOSE_MIN_S = 1.2
GENERIC_CLOSE_STABLE_WINDOW_S = 0.5
GENERIC_CLOSE_STABLE_RANGE = 0.012
# Generic close is split into two stages: close to a gentle intermediate
# opening, hold while the product settles between the jaws, then close fully.
GENERIC_GRIP_STAGE1 = 0.50
GENERIC_CLOSE_STAGE1_DWELL_S = 2.5
# The 50 mm open-finger post extension was observed knocking light/tall
# cylinders away from the grasp (maidong, kouxiangtang): the fingers clear the
# product by only a few mm when open, and a small lateral estimate error lets
# one finger catch and push it during the forward sweep.  For these goods,
# close immediately at the contact point instead of extending first.
# 脉动、可乐已移除出名单（v2 实测恢复正常 50mm 接触后前伸），口香糖保留。
GENERIC_NO_POST_EXTEND_KINDS = {"kouxiangtang"}
# The generic front profiles keep the wrist at the product centre.  For short
# goods (kouxiangtang is only ~5 mm above the middle board) the finger tips
# then scrape or jam against the shelf board during the approach, deflecting
# the arm and throwing the grasp off target.  Raise the TCP so the fingers
# clear the board; tall goods are unaffected because their centre already
# exceeds the clearance.
GENERIC_TCP_FINGER_CLEARANCE_M = 0.06
# Kouxiangtang was repeatedly contacted near its upper edge: one grasp was
# already empty after retreat and two nominal captures later slipped during
# delivery.  Its 80 mm body still leaves 10 mm between this 50 mm TCP height
# and the board, while moving the jaws 10 mm closer to the product centre.
GENERIC_TCP_FINGER_CLEARANCE_BY_KIND_M = {
    "kouxiangtang": 0.050,
}
# 所有货物抓取姿态的目标 Z 统一抬升 1cm：让指尖略高于测得的货物中心，
# 避免指尖落在货架导轨/前缘高度（D 架苹果指尖撞导轨问题），对深度测量
# 的偏低误差更宽容。对球体（苹果/橙子）与普通货物、各层货架一致生效。
GRASP_TCP_Z_RAISE_M = 0.010
# 按货物类型的抓取高度额外偏移（米）：负值降低。可乐、核桃味刀调低 1cm。
GRASP_TCP_Z_OFFSET_BY_KIND = {"kele": -0.010, "heweidao": -0.010}
# KDL/仿真运动学的 X 方向执行偏差：普通货物右臂 TCP 实测比指令偏东约
# 1-2cm（用户视角为偏左），把指令目标再往西（-x）10mm，右臂合计 -20mm；
# 左臂不再往东补偿，保持目标点居中。球体与纸巾不走该补偿，不受影响。
GRASP_TCP_X_OFFSET_BY_ARM = {"r": 0.003, "l": 0.000}
GRIPPER_MAX_OPENING_M = 0.080
GRIP_PRESHAPE_CLEARANCE_M = 0.012
GRIP_PRESHAPE_REACHED_TOLERANCE = 0.04

STATE_GO_SCAN = "go_scan"
STATE_SCAN = "scan"
STATE_REVISIT = "revisit"
STATE_DIRECT_TRANSIT = "direct_transit"
STATE_ALIGN = "align"
STATE_RECHECK = "recheck"
STATE_GRASP_SETTLE = "grasp_settle"
STATE_TISSUE_ROTATE = "tissue_rotate"
STATE_DEPLOY = "deploy"
STATE_ARM_FORWARD = "arm_forward"
STATE_POST_EXTEND = "post_extend"
STATE_DUAL_CONTACT = "dual_contact"
STATE_DUAL_SQUEEZE = "dual_squeeze"
STATE_CLOSE = "close"
STATE_TRIAL_LIFT = "trial_lift"
STATE_LIFT = "lift"
STATE_RETREAT = "retreat"
STATE_DONE = "done"
STATE_ABORT = "abort"


def generic_post_extend_world(
        nominal_contact_world: np.ndarray, target_kind: str) -> np.ndarray:
    """Return the open-gripper continuation endpoint for one product kind."""
    extended = np.asarray(nominal_contact_world, dtype=float).copy()
    extended[1] += GENERIC_POST_EXTEND_M_BY_KIND.get(
        target_kind, GENERIC_POST_CONTACT_EXTENSION_M)
    extended[2] -= GENERIC_POST_EXTEND_Z_DROP_M_BY_KIND.get(target_kind, 0.0)
    return extended


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def stamp_from_record(record: dict) -> int | None:
    try:
        return int(record["stamp_ns"])
    except (KeyError, TypeError, ValueError):
        return None


def decode_list(message: String) -> list[dict]:
    try:
        value = json.loads(message.data)
    except (json.JSONDecodeError, TypeError):
        return []
    return [record for record in value if isinstance(record, dict)] \
        if isinstance(value, list) else []


def marker_below_yolo(detection: dict, markers: list[dict]) -> dict | None:
    """Bind a product to its nearest marker that is not above it.

    Image Y increases downward.  Candidates must lie in the lower half of the
    YOLO rectangle or below it, remain within the adjacent label-rail band, and
    agree with the product's YOLO world-Z when depth is available.  The final
    choice is the Euclidean-nearest marker to the box's bottom centre.
    """
    try:
        x0, y0, x1, y1 = map(float, detection["bbox_xyxy"])
    except (KeyError, TypeError, ValueError):
        return None
    width, height = x1 - x0, y1 - y0
    if width <= 2.0 or height <= 2.0:
        return None

    centre_x = 0.5 * (x0 + x1)
    minimum_marker_y = y0 + 0.50 * height
    maximum_marker_y = y1 + max(
        ARUCO_MAX_VERTICAL_GAP_MIN_PX,
        ARUCO_MAX_VERTICAL_GAP_BOX_HEIGHTS * height)
    detection_world = None
    try:
        candidate_world = np.asarray(detection.get("world"), dtype=float)
        if candidate_world.shape == (3,) and np.all(np.isfinite(candidate_world)):
            detection_world = candidate_world
    except (TypeError, ValueError):
        pass
    product_height = PRODUCT_CENTER_ABOVE_MARKER_M.get(
        detection.get("class"))
    candidates = []
    for marker in markers:
        try:
            marker_id = int(marker["id"])
            marker_x, marker_y = map(float, marker["pixel_center"])
            world = np.asarray(marker["position_world"], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if marker_id not in range(45) or world.shape != (3,):
            continue
        if not np.all(np.isfinite(world)):
            continue
        if marker_y < minimum_marker_y or marker_y > maximum_marker_y:
            continue  # reject codes above the product or on a lower shelf rail
        horizontal_margin = ARUCO_MAX_HORIZONTAL_MARGIN_BOX_WIDTHS * width
        if (marker_x < x0 - horizontal_margin
                or marker_x > x1 + horizontal_margin):
            continue
        if detection_world is not None and product_height is not None:
            marker_product_z = world[2] + product_height
            if (abs(marker_product_z - detection_world[2])
                    > ARUCO_PRODUCT_LEVEL_TOLERANCE_M):
                continue
        if detection_world is not None:
            # Keep the product's depth estimate near the marker horizontally.
            # This rejects markers that are pixel-aligned by chance but belong
            # to a neighbouring slot when products are crowded or displaced.
            if (np.linalg.norm(world[:2] - detection_world[:2])
                    > DEPTH_TARGET_MARKER_XY_MAX_M):
                continue
        pixel_distance = math.hypot(marker_x - centre_x, marker_y - y1)
        candidates.append((pixel_distance, marker_id, marker))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def fixed_slot_nearest_marker(marker_world: np.ndarray):
    """Resolve a physical fixed-layout slot from measured marker position.

    The rendered marker texture can decode to a different raw ID.  Geometry is
    unambiguous because shelf marker locations are separated by more than this
    tolerance in X/Z, so diagnostic mode uses position instead of trusting the
    raw decoded ID.
    """
    marker_world = np.asarray(marker_world, dtype=float)
    if marker_world.shape != (3,) or not np.all(np.isfinite(marker_world)):
        return None, None, float("inf")
    distances = [
        (float(np.linalg.norm(marker_world - expected)), marker_id)
        for marker_id, expected in fixed_marker_world_by_id().items()
    ]
    distance, physical_marker_id = min(distances)
    if distance > FIXED_MARKER_POSITION_TOLERANCE_M:
        return None, None, distance
    return (physical_marker_id,
            fixed_layout_by_marker()[physical_marker_id], distance)


class ShelfPickController(Node):
    """Motion state machine driven only by YOLO and measured ArUco records."""

    def __init__(self, target_kind: str, max_scan_cycles: int,
                 tcp_diagnostic_ground_truth: bool,
                 scan_skip_lower: bool,
                 close_recheck: bool = True):
        super().__init__("yolo_aruco_shelf_pick_controller")
        self.target_kind = target_kind
        self.product_height = PRODUCT_CENTER_ABOVE_MARKER_M[target_kind]
        self.product_grasp_width = PRODUCT_GRASP_WIDTH_M[target_kind]
        self.grip_preshape_command = float(np.clip(
            (self.product_grasp_width + GRIP_PRESHAPE_CLEARANCE_M)
            / GRIPPER_MAX_OPENING_M,
            GRIP_CLOSE + GENERIC_EMPTY_GRIP_MARGIN,
            GRIP_OPEN))
        self.max_scan_cycles = max_scan_cycles
        self.tcp_diagnostic_ground_truth = tcp_diagnostic_ground_truth
        self.scan_skip_lower = bool(scan_skip_lower)
        self.lock = threading.Lock()
        self.opportunistic_target_kinds = (target_kind,)
        self.opportunistic_target_priority = {target_kind: 0}
        self.opportunistic_target_locked = True
        self.opportunistic_yolo_frames = deque(maxlen=30)
        self.opportunistic_target_pairs = {}

        self.base_xy = None
        self.base_yaw = 0.0
        self.joints = {}
        self.last_odom_time = None
        self.last_joint_time = None
        self.initialized = False
        self.ik = MMK2FIK()
        self.kdl = MMK2Kdl()

        self.state = STATE_GO_SCAN
        self.state_t0 = self.now()
        self.state_monotonic_t0 = time.monotonic()
        self.abort_reason = None
        self.scan_index = 0
        self.scan_pose_index = 0
        self.scan_camera_ready_since = None
        self.scan_cycles = 0
        self.scan_station_order = None
        # 第一单之后从最西侧（shelf A）开始扫描；第一单保持从最近站点（E）开始
        self.scan_prefer_west_start = False
        # The match runner may have already observed this requested kind while
        # a previous order was scanning.  Use that measured world X only to
        # choose the first scan station; normal perception must still confirm
        # and localise the product before any grasp is attempted.
        self.scan_preferred_x = None
        # 定点补拍状态：当前位姿关联到但未锁定的候选（marker_id -> 关联帧数）
        self.scan_unlocked_markers = {}
        # 有框无码候选：YOLO 检测到目标类但没关联到码（box_key -> 帧数/姿态）
        self.scan_unlocked_boxes = {}
        self.revisit_marker_id = None
        self.revisit_pose_index = 0
        self.revisit_pose_t0 = 0.0
        self.revisit_pose_monotonic_t0 = 0.0
        self.revisit_poses = REVISIT_POSES
        self.revisit_rounds = {}
        self.revisit_total_rounds = 0
        # 方向3备选：box 补拍失败时用 YOLO 框世界坐标推断槽位
        self.revisit_box_world = None
        self.revisit_box_conf = 0.0
        self.revisit_box_confirmations = 0
        self.revisit_box_key = None
        self.scan_diag_last_log = 0.0
        self.default_scan_poses = (
            SCAN_OVERVIEW_POSES
            if self.scan_skip_lower and kind_never_on_lower_shelf(target_kind)
            else SCAN_CAMERA_POSES)
        self.scan_poses = self.default_scan_poses
        self.inventory_scan_hint_active = False
        self.nav_target = None
        self.ik_retry_forward_m = 0.0
        self.deploy_retry_count = 0
        # 原地旋转卡死恢复状态
        self._rot_stall_target = None
        self._rot_stall_anchor_yaw = None
        self._rot_stall_anchor_t = 0.0
        self._rot_stall_anchor_xy = None
        self._rot_stall_unsticks = 0
        self._rot_unstick_phase = False

        self.yolo_frames = deque(maxlen=24)
        # 每个 head 帧 YOLO 输出的总框数（所有类别），用于区分"YOLO 全没检测
        # 到"与"YOLO 检测到其他货但没检测到目标类"。
        self.yolo_total_frames = deque(maxlen=24)
        self.aruco_frames = deque(maxlen=24)
        self.marker_positions = deque(maxlen=15)
        self.depth_target_samples = deque(maxlen=15)
        self.association_candidate_id = None
        self.association_confirmation_count = 0
        self.last_association_pair = None
        self.last_association_reject_log = 0.0
        self.skipped_tissue_markers = set()
        self.skipped_tissue_slots = set()
        self.no_middle_tissue = False
        # Formal multi-order runs can blacklist a shelf marker after a failed
        # attempt, so a retry searches for another physical item of the same
        # kind instead of repeatedly selecting the same slot.
        self.excluded_marker_ids = set()
        # YOLO-only localisation has no decoded marker identity.  Failed
        # attempts therefore blacklist the fixed matrix slot instead.
        self.excluded_slot_keys = set()
        self.close_recheck = bool(close_recheck)
        self.recheck_marker_skips = set()
        self.recheck_confirmation_times = deque(maxlen=12)
        self.recheck_last_yolo_stamp = None
        self.recheck_started_at = 0.0
        self.recheck_pose_started_at = 0.0
        self.recheck_pose_index = 0
        self.recheck_poses = ()
        self._recheck_passed = False
        self.last_generic_close_log = 0.0
        self.target_marker_id = None
        self.target_physical_marker_id = None
        self.target_world = None
        self.committed_slot = None
        # 记忆直达槽位复核失败后的同货架相邻列重试（方案 B）：
        # 列证据通常只差一格，相邻列命中可省掉整轮“回货架中心全量扫描”。
        self.direct_slot_target_active = False
        self.direct_slot_adjacent_retries = 0
        self.direct_slot_adjacent_max_retries = 2
        self.grasp_arm = "r"
        self.align_base_x = None
        self.align_base_y = SCAN_Y
        self._grasp_settle_anchor_xy = None
        self._grasp_settle_anchor_yaw = None
        self._grasp_settle_started_at = None
        self._grasp_settle_logged = False
        self.shelf_level = "unselected"
        self.object_geometry = "generic"
        self.is_top_shelf = False
        self.use_sphere_grasp = False
        self.use_dual_tissue_grasp = target_kind == "zhijin"
        self.slide_grasp = SLIDE_REFERENCE_COMMAND
        self.pregrasp_arm_joints = None
        self.approach_arm_joints = []
        self.approach_index = 0
        self.forward_contact_world = None
        self.forward_terminal_entered_at = None
        self.forward_terminal_slow_logged = False
        self.forward_start_tcp = None
        self.forward_start_base_xy = None
        self.generic_close_start_grip = None
        self.generic_close_stage = 1
        self.generic_close_grip_samples = deque(maxlen=120)
        self.generic_forward_start_world = None
        self.generic_direct_start_joints = None
        self.generic_direct_contact_joints = None
        self.generic_direct_duration_s = 0.0
        self.generic_direct_endpoint_ready_since = None
        self.post_extend_nominal_world = None
        self.post_extend_target_world = None
        self.post_extend_arm_joints = None
        self.post_extend_start_joints = None
        self.post_extend_duration_s = 0.0
        self.post_extend_endpoint_ready_since = None
        self.generic_top_lift_arm_joints = None
        self.generic_top_retreat_arm_joints = None
        self.dual_pregrasp_left_joints = None
        self.dual_pregrasp_right_joints = None
        self.dual_lift_use_arm = False
        self.dual_lift_left_joints = None
        self.dual_lift_right_joints = None
        self.dual_lift_retreat_left_joints = None
        self.dual_lift_retreat_right_joints = None
        self.dual_lift_arm_waypoints = []
        self.dual_lift_arm_stage = 0
        self.dual_lift_arm_achieved_m = 0.0
        self.dual_lift_retreat_waypoints = []
        self.dual_lift_retreat_stage = 0
        self.dual_lift_settled_since = None
        # 平转 90° 预调整：锁定目标后先原地旋转纸盒，转完再进入双臂抓取。
        self.tissue_rotated_90 = False
        self.tissue_rotate_stage = 0
        self.tissue_rotate_targets = {}
        self._grasp_arm_switch_count = 0
        # Top-shelf fork sequence; see advance_dual_tissue_top_fork_sequence.
        self.dual_top_extract_stage = 0
        self.dual_top_fork_targets = {}
        self.dual_surround_close_left_joints = None
        self.dual_surround_close_right_joints = None
        self.dual_surround_left_joints = None
        self.dual_surround_right_joints = None
        self.dual_surround_pass_left_joints = None
        self.dual_surround_pass_right_joints = None
        self.dual_surround_forward_left_joints = None
        self.dual_surround_forward_right_joints = None
        self.dual_surround_return_left_joints = None
        self.dual_surround_return_right_joints = None
        self.dual_surround_unroll_left_joints = None
        self.dual_surround_unroll_right_joints = None
        self.dual_surround_stage = 0
        self.dual_overhead_route = False
        self.dual_surround_half_span = DUAL_TISSUE_SURROUND_HALF_SPAN_M
        self.dual_middle_extend_close = False
        self.dual_clamp_half_span = DUAL_TISSUE_CLAMP_HALF_SPAN_M
        self.dual_insert_forward_m = DUAL_TISSUE_INSERT_FORWARD_M
        # 中间列纸巾"直接探入"：不走宽环绕，双臂以夹持跨度直接前探后压紧
        self.dual_direct_probe = False
        self.dual_top_wrist_rolled = False
        self.dual_top_wrist_inward = False
        self.dual_side_rolled = False
        self.dual_contact_push_side = "left"
        self.dual_pregrasp_half_span = DUAL_TISSUE_PREGRASP_HALF_SPAN_M
        self.dual_squeeze_m = DUAL_TISSUE_SQUEEZE_M
        self.dual_contact_tcp_z = None
        self.dual_clamp_left_joints = None
        self.dual_clamp_right_joints = None
        self.dual_retreat_left_joints = None
        self.dual_retreat_right_joints = None
        self.dual_motion_start_left = None
        self.dual_motion_start_right = None
        self.dual_motion_target_left = None
        self.dual_motion_target_right = None
        self.dual_motion_duration_s = 0.0
        self.dual_motion_label = "idle"
        self.dual_motion_require_convergence = False
        self.dual_motion_endpoint_ready_since = None
        self.dual_motion_path_distances = None
        self.dual_motion_path_left = None
        self.dual_motion_path_right = None
        self.dual_contact_start_left_joints = None
        self.dual_contact_start_right_joints = None
        self.dual_contact_target_left_joints = None
        self.dual_contact_target_right_joints = None
        self.dual_contact_start_left_world = None
        self.dual_contact_start_right_world = None
        self.dual_contact_goal_left_world = None
        self.dual_contact_goal_right_world = None
        self.dual_contact_duration_s = 0.0
        self.dual_contact_left_duration_s = 0.0
        self.dual_contact_right_duration_s = 0.0
        self.dual_left_contacted = False
        self.dual_right_contacted = False
        self.dual_left_contact_hold_joints = None
        self.dual_right_contact_hold_joints = None
        self.dual_left_contact_samples = deque(maxlen=100)
        self.dual_right_contact_samples = deque(maxlen=100)
        self.manip_base_hold_xy = None
        self.manip_base_hold_yaw = None
        self.manip_base_hold_last_log = 0.0
        self.sphere_pregrasp_world = None
        self.sphere_contact_world = None
        self.sphere_forward_reference = None
        self.sphere_forward_last_progress = -1.0
        self.sphere_open_grip_reference = None
        self.sphere_creep_start_world = None
        self.sphere_creep_goal_world = None
        self.sphere_creep_started_at = None
        self.sphere_creep_progress_samples = deque(maxlen=100)
        self.sphere_close_grip_samples = deque(maxlen=100)
        self.sphere_trial_grip_samples = deque(maxlen=100)
        self.sphere_trial_lift_arm_joints = None
        self.sphere_lift_arm_joints = None
        self.sphere_retreat_arm_joints = None
        self.sphere_trial_slide = None
        self.sphere_lift_slide = None
        self.sphere_slide_command = None
        self.middle_sphere_slide_corrections = 0
        self.sphere_grip_verified = False

        # Desired and smoothly published commands.
        self.des_slide = 0.0
        self.des_head = np.zeros(2)
        self.des_left_arm = np.zeros(6)
        self.des_right_arm = np.zeros(6)
        self.des_left_grip = GRIP_OPEN
        self.des_right_grip = GRIP_OPEN
        self.cmd_slide = 0.0
        self.cmd_head = np.zeros(2)
        self.cmd_left_arm = np.zeros(6)
        self.cmd_right_arm = np.zeros(6)
        self.cmd_left_grip = GRIP_OPEN
        self.cmd_right_grip = GRIP_OPEN
        self.des_linear = self.des_angular = 0.0
        self.cmd_linear = self.cmd_angular = 0.0
        self.commands_ready_since = None
        # Preserve the complete feedback carried by JointState/Odometry for
        # placement diagnostics.  The controller itself only needs joint
        # positions, but velocity/effort and measured chassis twist are needed
        # to distinguish a commanded move from oscillation, contact or stall.
        self.joint_velocities = {}
        self.joint_efforts = {}
        self.base_measured_linear = 0.0
        self.base_measured_angular = 0.0

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.slide_pub = self.create_publisher(
            Float64MultiArray, "/spine_forward_position_controller/commands", 5)
        self.head_pub = self.create_publisher(
            Float64MultiArray, "/head_forward_position_controller/commands", 5)
        self.left_pub = self.create_publisher(
            Float64MultiArray, "/left_arm_forward_position_controller/commands", 5)
        self.right_pub = self.create_publisher(
            Float64MultiArray, "/right_arm_forward_position_controller/commands", 5)

        self.create_subscription(
            Odometry, "/slamware_ros_sdk_server_node/odom", self.odom_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 10)
        self.create_subscription(
            String, "/goods/yolo_detections", self.yolo_cb, 10)
        self.create_subscription(
            String, "/aruco/head/detections", self.aruco_cb, 10)
        self.create_timer(0.02, self.tick)
        self.last_status_log = 0.0

        self.get_logger().info(
            f"requested class={target_kind}; tcp_diagnostic_ground_truth="
            f"{int(tcp_diagnostic_ground_truth)}; "
            f"scan_poses={len(self.scan_poses)} "
            f"physical_width={self.product_grasp_width:.3f}m "
            f"preshape={self.grip_preshape_command:.3f}; "
            "waiting for odom/joints/perception")

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, message: Odometry) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        self.base_xy = np.array([position.x, position.y], dtype=float)
        self.base_yaw = float(Rotation.from_quat([
            orientation.x, orientation.y, orientation.z, orientation.w,
        ]).as_euler("xyz")[2])
        self.base_measured_linear = float(message.twist.twist.linear.x)
        self.base_measured_angular = float(message.twist.twist.angular.z)
        self.last_odom_time = self.now()

    def joint_cb(self, message: JointState) -> None:
        self.joints = {
            name: float(message.position[index])
            for index, name in enumerate(message.name)
            if index < len(message.position)
        }
        self.joint_velocities = {
            name: float(message.velocity[index])
            for index, name in enumerate(message.name)
            if index < len(message.velocity)
        }
        self.joint_efforts = {
            name: float(message.effort[index])
            for index, name in enumerate(message.name)
            if index < len(message.effort)
        }
        self.last_joint_time = self.now()

    def yolo_cb(self, message: String) -> None:
        all_records = decode_list(message)
        head_records = [
            record for record in all_records
            if record.get("camera", "head") == "head"]
        stamp_ns = (
            stamp_from_record(head_records[0]) if head_records else None)
        if stamp_ns is None:
            return
        with self.lock:
            self.yolo_total_frames.append((stamp_ns, len(head_records)))
            self.opportunistic_yolo_frames.append(
                (stamp_ns, head_records))
            self._maybe_lock_opportunistic_target_locked()
            records = [
                record for record in head_records
                if record.get("class") == self.target_kind]
            if not records:
                return
            self.yolo_frames.append((stamp_ns, records))
            collect_yolo_only = (
                self.state == STATE_SCAN
                or (self.state == STATE_REVISIT
                    and self.revisit_box_key is not None))
            if collect_yolo_only:
                # 有框无码候选：记录 YOLO 目标类框（按世界 x/z 或像素归并），
                # 让"检测到但没解到码"的货物也能触发定点补拍。
                for record in records:
                    world = self._detection_world(record)
                    slot = None
                    if world is not None:
                        try:
                            # Stable fixed-grid grouping prevents one product
                            # from becoming several rounded-coordinate boxes.
                            slot = fixed_slot_from_world(
                                float(world[0]), float(world[2]))
                        except (TypeError, ValueError):
                            slot = None
                    if slot is not None:
                        box_key = slot
                    elif world is not None:
                        box_key = (
                            round(float(world[0]), 1),
                            round(float(world[2]), 1))
                    else:
                        pixel = record.get("pixel_center") or (0, 0)
                        box_key = (
                            "px",
                            self.current_scan_camera_pose()[0],
                            int(round(float(pixel[0]) / 40)),
                            int(round(float(pixel[1]) / 40)))
                    if (self.state == STATE_REVISIT
                            and self.revisit_box_key is not None
                            and box_key != self.revisit_box_key):
                        continue
                    entry = self.scan_unlocked_boxes.setdefault(
                        box_key, {"confirmations": 0, "pose": None,
                                  "world": None, "max_conf": 0.0,
                                  "worlds": deque(maxlen=15),
                                  "last_stamp_ns": None})
                    if entry.get("last_stamp_ns") != stamp_ns:
                        entry["confirmations"] += 1
                        entry["last_stamp_ns"] = stamp_ns
                    entry["pose"] = self.current_scan_camera_pose()[0]
                    try:
                        entry["max_conf"] = max(
                            entry.get("max_conf", 0.0),
                            float(record.get("conf", 0.0)))
                    except (TypeError, ValueError):
                        pass
                    if world is not None:
                        entry["world"] = world
                        entry["worlds"].append(world)
                    self._maybe_lock_yolo_only_target_locked()
            self.try_association_locked()
            self.try_recheck_locked()

    def configure_opportunistic_targets(self, kinds) -> None:
        """Allow one graspable pending class to become this trip's target.

        This is deliberately a pre-grasp, one-shot choice.  It never changes
        a target after localisation begins and therefore cannot turn one trip
        into a multi-item collection run.  A class is graspable here only
        after repeated synchronized YOLO/ArUco associations.
        """
        ordered = []
        for kind in (self.target_kind, *tuple(kinds or ())):
            if kind not in PRODUCT_CENTER_ABOVE_MARKER_M:
                raise ValueError(f"unsupported opportunistic kind: {kind!r}")
            if kind not in ordered:
                ordered.append(kind)
        self.opportunistic_target_kinds = tuple(ordered)
        self.opportunistic_target_priority = {
            kind: index for index, kind in enumerate(ordered)}
        self.opportunistic_target_locked = len(ordered) <= 1
        self.opportunistic_yolo_frames.clear()
        self.opportunistic_target_pairs.clear()
        if not self.opportunistic_target_locked:
            self.get_logger().info(
                "single-item opportunistic target selection enabled; "
                f"priority={ordered} confirmations="
                f"{OPPORTUNISTIC_TARGET_CONFIRMATIONS}")

    def _maybe_lock_opportunistic_target_locked(self) -> None:
        """Commit once to a repeatedly associated class/marker pair."""
        if (self.opportunistic_target_locked
                or self.state != STATE_SCAN
                or self.target_marker_id is not None
                or self.now() - self.state_t0 < SCAN_SETTLE_S
                or not self.opportunistic_yolo_frames
                or not self.aruco_frames):
            return
        yolo_stamp, records = self.opportunistic_yolo_frames[-1]
        aruco_stamp, markers = min(
            self.aruco_frames,
            key=lambda frame: abs(frame[0] - yolo_stamp))
        if abs(aruco_stamp - yolo_stamp) > ARUCO_SYNC_TOLERANCE_NS:
            return

        cutoff = yolo_stamp - OPPORTUNISTIC_TARGET_WINDOW_NS
        ready = []
        for kind in self.opportunistic_target_kinds:
            detections = sorted(
                (record for record in records
                 if record.get("class") == kind),
                key=lambda item: float(item.get("conf", 0.0)),
                reverse=True)
            for detection in detections:
                marker = marker_below_yolo(detection, markers)
                if marker is None:
                    continue
                try:
                    marker_id = int(marker["id"])
                    marker_world = np.asarray(
                        marker["position_world"], dtype=float)
                except (KeyError, TypeError, ValueError):
                    continue
                if (marker_id in self.excluded_marker_ids
                        or marker_id in self.recheck_marker_skips
                        or marker_id in self.skipped_tissue_markers
                        or marker_world.shape != (3,)
                        or not np.all(np.isfinite(marker_world))):
                    continue
                identity = (kind, marker_id)
                pairs = self.opportunistic_target_pairs.setdefault(
                    identity,
                    deque(maxlen=OPPORTUNISTIC_TARGET_CONFIRMATIONS))
                while pairs and pairs[0][0] < cutoff:
                    pairs.popleft()
                pair = (yolo_stamp, aruco_stamp)
                # Several ArUco callbacks may arrive before the next YOLO
                # result.  Count the source image once, otherwise one frozen
                # product detection could satisfy the three-frame gate.
                if not pairs or pairs[-1][0] != yolo_stamp:
                    pairs.append(pair)
                if len(pairs) >= OPPORTUNISTIC_TARGET_CONFIRMATIONS:
                    ready.append((kind, marker_id))
                # Only the highest-confidence usable marker for this class is
                # allowed to collect one confirmation from this YOLO frame.
                break
        if not ready:
            return
        selected, selected_marker = min(
            ready,
            key=lambda item: (
                self.opportunistic_target_priority[item[0]], item[1]))
        previous = self.target_kind
        self._set_pregrasp_target_kind(selected)
        # The frames used to choose the class are deliberately discarded by
        # _set_pregrasp_target_kind so they cannot also satisfy localisation.
        # Give the newly committed class one complete, independent sampling
        # window at this already useful camera pose.  Without this reset, a
        # late commitment could hit the original dwell deadline after only
        # one or two fresh frames and incur an unnecessary revisit.
        self.state_t0 = self.now()
        self.opportunistic_target_locked = True
        self.opportunistic_yolo_frames.clear()
        self.opportunistic_target_pairs.clear()
        self.get_logger().info(
            f"[order-select] committed single target kind={selected} "
            f"marker={selected_marker} previous={previous} after "
            f"{OPPORTUNISTIC_TARGET_CONFIRMATIONS} synchronized "
            "YOLO/ArUco pairs; "
            "this trip will still carry exactly one item")

    def _set_pregrasp_target_kind(self, kind: str) -> None:
        """Update target-dependent grasp parameters before localisation."""
        if (self.state != STATE_SCAN or self.target_marker_id is not None):
            raise RuntimeError("target kind can only change during initial scan")
        self.target_kind = kind
        self.product_height = PRODUCT_CENTER_ABOVE_MARKER_M[kind]
        self.product_grasp_width = PRODUCT_GRASP_WIDTH_M[kind]
        self.grip_preshape_command = float(np.clip(
            (self.product_grasp_width + GRIP_PRESHAPE_CLEARANCE_M)
            / GRIPPER_MAX_OPENING_M,
            GRIP_CLOSE + GENERIC_EMPTY_GRIP_MARGIN,
            GRIP_OPEN))
        self.use_dual_tissue_grasp = kind == "zhijin"
        self.no_middle_tissue = False
        self.skipped_tissue_markers.clear()
        self.skipped_tissue_slots.clear()
        self.default_scan_poses = (
            SCAN_OVERVIEW_POSES
            if self.scan_skip_lower and kind_never_on_lower_shelf(kind)
            else SCAN_CAMERA_POSES)
        if not self.inventory_scan_hint_active:
            self.scan_poses = self.default_scan_poses
        self.scan_unlocked_markers.clear()
        self.scan_unlocked_boxes.clear()
        self.yolo_frames.clear()
        self.marker_positions.clear()
        self.depth_target_samples.clear()
        self.association_candidate_id = None
        self.association_confirmation_count = 0
        self.last_association_pair = None

    def aruco_cb(self, message: String) -> None:
        records = [record for record in decode_list(message)
                   if record.get("camera", "head") == "head"]
        if not records:
            return
        stamp_ns = stamp_from_record(records[0])
        if stamp_ns is None:
            return
        with self.lock:
            self.aruco_frames.append((stamp_ns, records))
            self._maybe_lock_opportunistic_target_locked()
            self.try_association_locked()
            self.try_recheck_locked()

    def _maybe_lock_yolo_only_target_locked(self) -> None:
        """Lock a stable fixed-grid YOLO target without requiring ArUco."""
        if (self.state not in (STATE_SCAN, STATE_REVISIT)
                or (self.state == STATE_REVISIT
                    and self.revisit_box_key is None)
                or self.target_marker_id is not None
                or self.target_world is not None
                or self.now() - self.state_t0 < SCAN_SETTLE_S):
            return
        candidates = []
        for entry in self.scan_unlocked_boxes.values():
            if entry.get("confirmations", 0) < YOLO_ONLY_TARGET_CONFIRMATIONS:
                continue
            worlds = entry.get("worlds")
            if not worlds:
                continue
            try:
                samples = np.asarray(list(worlds), dtype=float)
            except (TypeError, ValueError):
                continue
            if (samples.ndim != 2 or samples.shape[1] != 3
                    or len(samples) < YOLO_ONLY_TARGET_CONFIRMATIONS
                    or not np.all(np.isfinite(samples))):
                continue
            if (float(np.max(np.ptp(samples, axis=0)))
                    > YOLO_ONLY_TARGET_SPREAD_MAX_M):
                continue
            try:
                confidence = float(entry.get("max_conf", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < YOLO_ONLY_TARGET_CONF_MIN:
                continue
            candidates.append((
                int(entry["confirmations"]), confidence,
                np.median(samples, axis=0)))
        if not candidates:
            return

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _confirmations, confidence, median = candidates[0]
        x, y, z = (float(value) for value in median)
        # Navigation already identifies the current station.  Keep the shelf
        # fixed to that station so a noisy oblique view cannot jump an aisle.
        slot = fixed_slot_from_world(
            x, z, shelf=self._current_station_shelf())
        if slot is None:
            return
        shelf, level, column = slot
        if (self.target_kind == "zhijin" and column != "2"
                and not DUAL_TISSUE_SIDE_ROLLED_ENABLED):
            key = f"{level}|{shelf}|{column}"
            if key not in self.skipped_tissue_slots:
                self.skipped_tissue_slots.add(key)
                self.get_logger().warn(
                    "[tissue-filter] ignoring side-column tissue "
                    f"slot={key}; only the middle column is eligible")
            return
        if f"{level}|{shelf}|{column}" in self.excluded_slot_keys:
            return
        level_name = {"L3": "top", "L2": "middle", "L1": "lower"}[level]
        surface_z = SHELF_SURFACE_Z_M[level_name]
        target_z = min(
            z,
            surface_z + PRODUCT_HALF_HEIGHT_M[self.target_kind]
            + PRODUCT_CENTER_Z_TOLERANCE_M)
        column_x = (
            SCAN_X[self._scan_x_index_for_shelf(shelf)]
            + {"1": -0.22, "2": 0.00, "3": 0.22}[column])
        target_world = np.array([
            column_x,
            y + PRODUCT_HALF_DEPTH_M[self.target_kind],
            target_z,
        ], dtype=float)
        self.get_logger().warn(
            f"[yolo-only] locked kind={self.target_kind} "
            f"slot=({shelf}, {level}, {column}) conf={confidence:.3f} "
            f"world_median={np.round(median, 3)} -> target="
            f"{np.round(target_world, 3)}")
        self._commit_localised_target(
            target_world, None, "yolo_only",
            extra=f" slot=({shelf}, {level}, {column})",
            shelf=shelf)

    @staticmethod
    def _scan_x_index_for_shelf(shelf: str) -> int:
        return {
            "E": 0, "D": 1, "C": 2, "B": 3, "A": 4,
        }.get(str(shelf).upper(), 0)

    def _current_station_shelf(self) -> str:
        index = self.scan_index
        if self.scan_station_order:
            index = self.scan_station_order[
                self.scan_index % len(self.scan_station_order)]
        return ("E", "D", "C", "B", "A")[index % len(SCAN_X)]

    def _commit_localised_target(
            self, target_world: np.ndarray,
            marker_id: int | None, source: str,
            extra: str = "",
            physical_marker_id: int | None = None,
            shelf: str | None = None,
            enter_align: bool = True) -> None:
        """本地化一致后提交：目标世界坐标、抓取臂、对齐位姿、层/列。

        marker 关联路径与 YOLO-only 路径共用；marker_id 为 None 表示无码
        锁定（抓取前复核仍可用深度或可解码的 marker 精化）。
        """
        self.target_world = np.asarray(target_world, dtype=float)
        self.target_marker_id = marker_id
        self.target_physical_marker_id = physical_marker_id
        self.committed_slot = None
        slot_x = float(self.target_world[0])
        slot_z = float(self.target_world[2])

        if self.use_dual_tissue_grasp:
            self.grasp_arm = "r"
            desired_base_x = self.target_world[0]
            # 场景边缘墙壁避让：A 货架 C1（最西列）与 E 货架 C3（最东列）
            # 的双臂部署/预抓路径会扫到 x=±2.47 的墙壁（实测 A C1 左臂
            # 扫西墙 21mm、E C3 右臂扫东墙 91mm）。把对齐基座向货架中心
            # 平移，使部署路径全程留在墙内（A C1 东移 0.06m、E C3 西移
            # 0.12m 后实测干净），手臂仍可达目标两侧。
            if desired_base_x < -DUAL_TISSUE_WALL_CLEAR_X_THRESHOLD_M:
                desired_base_x += DUAL_TISSUE_WEST_WALL_SHIFT_M
            elif desired_base_x > DUAL_TISSUE_WALL_CLEAR_X_THRESHOLD_M:
                desired_base_x -= DUAL_TISSUE_EAST_WALL_SHIFT_M
        else:
            # 恢复“A 货架用左臂、其他货架用右臂”的统一臂选择：
            # 顶层不再强制左臂，避免非 A 货架也走左臂导致抓取偏左。
            if shelf is not None:
                west_column = str(shelf).upper() == "A"
            else:
                west_slot = (
                    None if marker_id is None
                    else fixed_layout_by_marker().get(int(marker_id)))
                west_column = bool(
                    west_slot and west_slot.get("shelf") == "A")
            desired_right_base_x = (
                self.target_world[0] - ARM_LATERAL_BIAS_M)
            if west_column or desired_right_base_x < NAV_X_MIN:
                self.grasp_arm = "l"
                desired_base_x = (
                    self.target_world[0] + ARM_LATERAL_BIAS_M)
            else:
                self.grasp_arm = "r"
                desired_base_x = desired_right_base_x
        self.align_base_x = float(np.clip(
            desired_base_x, NAV_X_MIN, NAV_X_MAX))
        # X 执行偏差补偿（见 GRASP_TCP_X_OFFSET_BY_ARM）：反向偏移指令
        # 目标，基座同步跟随，保持横向偏置不变。
        if (not self.use_dual_tissue_grasp
                and self.target_kind not in SPHERE_RADIUS_M):
            x_offset = GRASP_TCP_X_OFFSET_BY_ARM.get(self.grasp_arm, 0.0)
            if slot_z >= TOP_SHELF_Z_M and self.grasp_arm == "l":
                # 顶层左臂实机末端偏西约 1cm，目标点向东再移 8mm 抵消。
                x_offset += TOP_LEFT_GRASP_X_BIAS_M
            if x_offset:
                self.target_world[0] += x_offset
                self.align_base_x = float(np.clip(
                    self.align_base_x + x_offset, NAV_X_MIN, NAV_X_MAX))
        if self.target_world[2] >= TOP_SHELF_Z_M:
            self.shelf_level = "top"
        elif self.target_world[2] >= MIDDLE_SHELF_Z_MIN_M:
            self.shelf_level = "middle"
        else:
            self.shelf_level = "lower"
        self.object_geometry = (
            "sphere" if self.target_kind in SPHERE_RADIUS_M else "generic")
        self.is_top_shelf = self.shelf_level == "top"
        self.use_sphere_grasp = bool(
            self.object_geometry == "sphere"
            and self.shelf_level in ("top", "middle"))
        if self.is_top_shelf:
            self.align_base_y = float(
                self.target_world[1] - TOP_GRASP_CENTER_DISTANCE_M)
            if self.use_dual_tissue_grasp:
                # Rolled wrists and the overhead bracket need this extra elbow
                # room; the former close approach saturated both wrist joints.
                self.align_base_y += DUAL_TISSUE_ALIGN_FORWARD_M
        elif self.use_dual_tissue_grasp:
            self.align_base_y = (
                SCAN_Y + GENERIC_EXTENSION_ALIGN_FORWARD_M
                + DUAL_TISSUE_ALIGN_FORWARD_M)
        elif (self.shelf_level in ("middle", "lower")
              and self.object_geometry != "sphere"):
            self.align_base_y = (
                SCAN_Y + GENERIC_EXTENSION_ALIGN_FORWARD_M)
        else:
            self.align_base_y = SCAN_Y
        self.slide_grasp = float(np.clip(
            SLIDE_REFERENCE_COMMAND
            - (self.target_world[2] - SLIDE_REFERENCE_Z_M),
            SLIDE_MIN, SLIDE_MAX))
        self.ik_retry_forward_m = 0.0
        self.deploy_retry_count = 0
        self.tissue_rotated_90 = False
        self.tissue_rotate_stage = 0
        self.tissue_rotate_targets = {}
        self._recheck_passed = False
        committed = fixed_slot_from_world(slot_x, slot_z)
        if committed is not None:
            self.committed_slot = committed
        if enter_align:
            self.set_state(STATE_ALIGN)
        self.get_logger().info(
            f"[localised] source={source} marker={marker_id} "
            f"product_world={np.round(self.target_world, 3)} "
            f"arm={'both' if self.use_dual_tissue_grasp else self.grasp_arm} "
            f"grasp_profile={self.grasp_profile_name()} "
            f"align_y={self.align_base_y:.3f} "
            f"slot={committed}{extra}")

    def configure_direct_slot_target(
            self, shelf: str, level: str, column: str,
            marker_id: int | None = None,
            product_y: float | None = None,
            product_z: float | None = None,
            defer_align: bool = False) -> bool:
        """用固定货架槽位直接生成抓取目标，跳过“先到架中心再扫描定位”。

        该路径完全复用 ``_commit_localised_target`` 的抓取位姿计算，只是把
        YOLO/ArUco 锁定换成已确认的固定几何槽位。默认仍会进入
        ``STATE_ALIGN``；带长距离导航器的子类可用 ``defer_align`` 延迟
        该状态，抵达货架后再继续原 close-recheck / grasp 流程，因此不会
        削弱抓取前校验，也不会让精对齐超时覆盖跨场地运输。
        """
        shelf = str(shelf).upper()
        level = str(level).upper()
        column_text = str(column)
        column = column_text[-1] if column_text[-1:].isdigit() else column_text
        if (shelf not in {"A", "B", "C", "D", "E"}
                or level not in {"L1", "L2", "L3"}
                or column not in {"1", "2", "3"}):
            self.get_logger().warn(
                f"[direct-slot] invalid fixed slot "
                f"shelf={shelf} level={level} column={column}")
            return False
        if (self.target_kind == "zhijin" and column != "2"
                and not DUAL_TISSUE_SIDE_ROLLED_ENABLED):
            self.get_logger().warn(
                "[tissue-filter] direct slot rejected for tissue "
                f"column={column}; only the middle column is eligible")
            return False
        slot_key = f"{level}|{shelf}|{column}"
        if slot_key in self.excluded_slot_keys:
            self.get_logger().warn(
                f"[direct-slot] fixed slot {slot_key} is excluded")
            return False

        level_name = {"L3": "top", "L2": "middle", "L1": "lower"}[level]
        shelf_x = SCAN_X[self._scan_x_index_for_shelf(shelf)]
        column_offset = {"1": -0.22, "2": 0.00, "3": 0.22}[column]
        target_x = float(shelf_x + column_offset)
        surface_z = float(SHELF_SURFACE_Z_M[level_name])
        half_height = float(PRODUCT_HALF_HEIGHT_M.get(self.target_kind, 0.0))
        target_z = surface_z + half_height
        try:
            measured_z = float(product_z)
            max_center_z = (
                surface_z + half_height + PRODUCT_CENTER_Z_TOLERANCE_M)
            if math.isfinite(measured_z) and surface_z <= measured_z <= 1.40:
                target_z = float(min(measured_z, max_center_z))
        except (TypeError, ValueError):
            pass
        # 固定槽位映射的 L1 下界是 0.50m。若某类几何半高为 0，直接用台面
        # 高度会落在判定边界外，导致 committed_slot 缺失；留 5mm 几何余量，
        # 对夹持目标的影响远小于槽位身份丢失。
        target_z = max(target_z, surface_z + 0.005)
        # 货架平面是固定几何，所有商品中心都在这条平面上。记忆里的
        # world_y 是 YOLO/深度前表面，叠加半深后仍可能差 3~6cm；直接
        # 用固定货架平面 y 更稳，避免“爪子没对准”的纵向偏差。
        target_y = SHELF_PRODUCT_CENTER_Y_M

        self._commit_localised_target(
            np.array([target_x, target_y, target_z], dtype=float),
            marker_id,
            "memory_direct",
            extra=f" slot={slot_key}",
            shelf=shelf,
            enter_align=not defer_align)
        self.direct_slot_target_active = True
        return True

    def advance_direct_transit(self) -> None:
        """Run a deferred direct-slot transit in navigation-aware subclasses.

        The base pick controller has no obstacle-aware long-range navigator.
        Subclasses that request ``defer_align=True`` must override this hook;
        aborting here prevents an unknown state from silently publishing zero
        velocity forever.
        """
        self.get_logger().error(
            "direct-slot transit requested without a navigation handler")
        self.abort_reason = "direct transit handler unavailable"
        self.set_state(STATE_ABORT)

    @staticmethod
    def _detection_world(detection: dict) -> np.ndarray | None:
        """Return a finite 3-D world point from a YOLO record.

        优先使用框中心的同步深度点 ``world``；中心深度缺失时回退到同一
        YOLO 框的 ROI 前景点 ``front_world``。这能让无 ArUco 时更稳定地
        走 YOLO + depth 定位，而不是因中心点缺失直接放弃。
        """
        for key in ("world", "front_world"):
            value = detection.get(key)
            try:
                world = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                continue
            if world.shape == (3,) and np.all(np.isfinite(world)):
                return world
        return None

    def target_slot(self) -> tuple[str, str, str] | None:
        """返回锁定目标的固定槽位 (shelf, level, column)；未锁定返回 None。"""
        if getattr(self, "committed_slot", None) is not None:
            return tuple(self.committed_slot)
        marker_id = getattr(self, "target_marker_id", None)
        if marker_id is not None:
            marker_slot = SLOT_BY_MARKER.get(int(marker_id))
            if marker_slot is not None:
                return tuple(marker_slot)
        target_world = getattr(self, "target_world", None)
        if target_world is None:
            return None
        try:
            return fixed_slot_from_world(
                float(target_world[0]), float(target_world[2]))
        except (TypeError, ValueError, IndexError):
            return None

    def target_slot_key(self) -> str | None:
        slot = self.target_slot()
        if slot is None:
            return None
        shelf, level, column = slot
        return f"{level}|{shelf}|{column}"

    def try_association_locked(self) -> None:
        if (self.state not in (STATE_SCAN, STATE_REVISIT)
                or self.now() - self.state_t0 < SCAN_SETTLE_S):
            return
        if not self.yolo_frames or not self.aruco_frames:
            return

        yolo_stamp, detections = self.yolo_frames[-1]
        aruco_stamp, markers = min(
            self.aruco_frames, key=lambda frame: abs(frame[0] - yolo_stamp))
        if abs(aruco_stamp - yolo_stamp) > ARUCO_SYNC_TOLERANCE_NS:
            return
        association_pair = (yolo_stamp, aruco_stamp)
        if association_pair == self.last_association_pair:
            return
        self.last_association_pair = association_pair

        best_pair = None
        for detection in sorted(
                detections, key=lambda item: float(item.get("conf", 0.0)),
                reverse=True):
            marker = marker_below_yolo(detection, markers)
            if marker is not None:
                best_pair = (detection, marker)
                break
        if best_pair is None:
            return

        detection, marker = best_pair
        marker_id = int(marker["id"])
        if self.state == STATE_REVISIT:
            # 有框无码补拍（revisit_marker_id=None）接受任意码；
            # 有明确码的补拍仍只认目标码。
            if (self.revisit_marker_id is not None
                    and marker_id != self.revisit_marker_id):
                return
        if marker_id in self.excluded_marker_ids:
            return
        if marker_id in self.recheck_marker_skips:
            return
        marker_world = np.asarray(marker["position_world"], dtype=float)
        if marker_id in self.skipped_tissue_markers:
            return
        if (self.target_kind == "zhijin"
                and not DUAL_TISSUE_SIDE_ROLLED_ENABLED):
            marker_slot = fixed_slot_from_world(
                float(marker_world[0]), float(marker_world[2]))
            if marker_slot is None or marker_slot[2] != "2":
                if marker_id not in self.skipped_tissue_markers:
                    self.skipped_tissue_markers.add(marker_id)
                    slot_text = (
                        "unknown" if marker_slot is None
                        else f"{marker_slot[0]}-{marker_slot[1]}-"
                             f"{marker_slot[2]}")
                    self.get_logger().warn(
                        "[tissue-filter] ignoring non-middle-column tissue "
                        f"marker={marker_id} slot={slot_text}; only the "
                        "middle column is eligible")
                return
        physical_marker_id = None
        fixed_slot = None
        if self.tcp_diagnostic_ground_truth:
            physical_marker_id, fixed_slot, slot_distance = (
                fixed_slot_nearest_marker(marker_world))
            if (fixed_slot is None
                    or fixed_slot.get("object_kind") != self.target_kind):
                if self.now() - self.last_association_reject_log > 1.0:
                    actual_kind = (None if fixed_slot is None
                                   else fixed_slot.get("object_kind"))
                    self.get_logger().warn(
                        f"[association] ignoring nearest raw ArUco ID="
                        f"{marker_id} below YOLO {self.target_kind}: "
                        f"physical_slot={physical_marker_id} kind="
                        f"{actual_kind!r} distance={slot_distance:.3f}m")
                    self.last_association_reject_log = self.now()
                return
        if self.state == STATE_SCAN:
            entry = self.scan_unlocked_markers.setdefault(
                marker_id,
                {"confirmations": 0, "pose": None, "world": None})
            entry["confirmations"] += 1
            entry["pose"] = self.current_scan_camera_pose()[0]
            entry["world"] = marker_world.copy()
        if self.target_marker_id is None:
            association_identity = (
                (marker_id, physical_marker_id)
                if self.tcp_diagnostic_ground_truth else marker_id)
            if association_identity != self.association_candidate_id:
                self.association_candidate_id = association_identity
                self.association_confirmation_count = 1
            else:
                self.association_confirmation_count += 1
            if (self.association_confirmation_count
                    < ASSOCIATION_CONFIRMATIONS_REQUIRED):
                return
            self.target_marker_id = marker_id
            self.target_physical_marker_id = physical_marker_id
            self.committed_slot = None
            self._recheck_passed = False
            self.marker_positions.clear()
            self.depth_target_samples.clear()
            physical_text = (
                "" if physical_marker_id is None
                else f" physical_slot_ID={physical_marker_id}")
            self.get_logger().info(
                f"[association] YOLO {self.target_kind} "
                f"conf={float(detection.get('conf', 0.0)):.3f} -> "
                f"nearest non-upper raw ArUco ID={marker_id}"
                f"{physical_text} confirmed across "
                f"{self.association_confirmation_count} synchronized frames")
        if marker_id != self.target_marker_id:
            return
        self.marker_positions.append(marker_world)
        depth_world = None
        try:
            candidate = np.asarray(detection.get("world"), dtype=float)
            if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
                depth_world = candidate
        except (TypeError, ValueError):
            pass
        depth_delta_ms = detection.get("depth_delta_ms")
        if (depth_world is not None
                and isinstance(depth_delta_ms, (int, float))
                and depth_delta_ms <= DEPTH_TARGET_MAX_DELTA_MS
                and DEPTH_TARGET_Z_MIN_M <= depth_world[2] <= DEPTH_TARGET_Z_MAX_M):
            self.depth_target_samples.append(depth_world)
        if len(self.marker_positions) < MARKER_SAMPLES_REQUIRED:
            return

        samples = np.asarray(self.marker_positions, dtype=float)
        spread = float(np.max(np.ptp(samples, axis=0)))
        if spread > MARKER_SAMPLE_SPREAD_MAX_M:
            return
        marker_median = np.median(samples, axis=0)
        # Prefer the depth-measured centre when it passes quality gates; it
        # tracks a product that has been displaced from its nominal slot,
        # which the marker + fixed-offset model cannot.  Otherwise fall back
        # to the marker model exactly as before.
        depth_median = None
        depth_target = None
        depth_samples = (
            np.asarray(self.depth_target_samples, dtype=float)
            if self.depth_target_samples else np.empty((0, 3)))
        if len(depth_samples) >= DEPTH_TARGET_MIN_SAMPLES:
            depth_median = np.median(depth_samples, axis=0)
            depth_spread = float(np.max(np.ptp(depth_samples, axis=0)))
            if (depth_spread <= DEPTH_TARGET_SPREAD_MAX_M
                    and np.linalg.norm(
                        depth_median[:2] - marker_median[:2])
                    <= DEPTH_TARGET_MARKER_XY_MAX_M):
                horizontal = float(np.linalg.norm(
                    depth_median[:2] - marker_median[:2]))
                if (horizontal <= DEPTH_TARGET_IN_SLOT_XY_MAX_M
                        and self.target_kind not in SPHERE_RADIUS_M):
                    # 普通盒子：前表面低于中心，marker 模型与深度前表面平均。
                    marker_z = marker_median[2] + self.product_height
                    z_target = 0.5 * (depth_median[2] + marker_z)
                elif self.target_kind in SPHERE_RADIUS_M:
                    # 球体（苹果/橙子，wxj v2）：中心高度 = 货架板面 + 半径，
                    # 由物理几何决定。marker 实测 Z 可偏差 3cm 以上，深度前
                    # 表面采样带也可能落在球体下半部，两者都不如板面模型
                    # 可靠；深度测量只用于 X/Y。
                    model_z = marker_median[2] + self.product_height
                    sphere_level = (
                        "top" if model_z >= TOP_SHELF_Z_M
                        else ("middle" if model_z >= MIDDLE_SHELF_Z_MIN_M
                              else "lower"))
                    z_target = (
                        SHELF_SURFACE_Z_M[sphere_level]
                        + SPHERE_RADIUS_M[self.target_kind])
                else:
                    # 非球体且深度横向偏差较大：不做 marker 混合（避免错配），
                    # 直接用深度前表面 Z，维持原有行为。
                    z_target = depth_median[2]
                depth_target = np.array([
                    depth_median[0],
                    depth_median[1] + PRODUCT_HALF_DEPTH_M[self.target_kind],
                    z_target,
                ])
        if depth_target is not None:
            measured_target = depth_target
            target_source = "depth"
        else:
            if self.target_kind in SPHERE_RADIUS_M:
                # 球体：中心高度 = 板面 + 半径，规避 marker 实测 Z 噪声。
                model_z = marker_median[2] + self.product_height
                sphere_level = (
                    "top" if model_z >= TOP_SHELF_Z_M
                    else ("middle" if model_z >= MIDDLE_SHELF_Z_MIN_M
                          else "lower"))
                measured_target = np.array([
                    marker_median[0],
                    marker_median[1] + PRODUCT_BEHIND_MARKER_M,
                    SHELF_SURFACE_Z_M[sphere_level]
                    + SPHERE_RADIUS_M[self.target_kind],
                ])
            else:
                measured_target = marker_median + np.array([
                    0.0, PRODUCT_BEHIND_MARKER_M, self.product_height,
                ])
            target_source = "marker"
        if self.tcp_diagnostic_ground_truth:
            physical_marker_id, fixed_slot, slot_distance = (
                fixed_slot_nearest_marker(marker_median))
            if (fixed_slot is None
                    or fixed_slot.get("object_kind") != self.target_kind):
                self.get_logger().warn(
                    "[TCP-DIAG] nearest marker left its expected physical "
                    f"slot (distance={slot_distance:.3f}m); discarding it")
                self.target_marker_id = None
                self.target_physical_marker_id = None
                self.association_candidate_id = None
                self.association_confirmation_count = 0
                self.marker_positions.clear()
                self.depth_target_samples.clear()
                return
            self.target_physical_marker_id = physical_marker_id
            self.target_world = np.asarray(
                fixed_slot["world_position"], dtype=float)
            vision_error = measured_target - self.target_world
            self.get_logger().warn(
                f"[TCP-DIAG] replacing measured centre "
                f"{np.round(measured_target, 4)} with exact centre "
                f"{np.round(self.target_world, 4)}; measured-minus-exact="
                f"{np.round(vision_error, 4)}m")
        else:
            self.target_world = measured_target
        # 商品与 ArUco 会在场景初始化时一起随机换位，marker ID 对应的是
        # 商品身份，不再代表固定物理槽位。纸巾侧列动作必须以本次视觉定位
        # 得到的世界坐标判列，否则 SLOT_BY_MARKER 的初始布局会把侧列误判
        # 为中列。这里只修正纸巾的动作选择；其他品类保持原 marker 路径。
        if self.target_kind == "zhijin":
            physical_slot = fixed_slot_from_world(
                float(self.target_world[0]), float(self.target_world[2]))
            if physical_slot is not None:
                marker_slot = SLOT_BY_MARKER.get(int(self.target_marker_id))
                self.committed_slot = physical_slot
                self.get_logger().info(
                    "[tissue-slot] using localised physical slot "
                    f"{physical_slot}; marker={self.target_marker_id} "
                    f"initial_slot={marker_slot}")
        if self.use_dual_tissue_grasp:
            # Centre the chassis between both arms instead of biasing it for a
            # single selected arm.
            self.grasp_arm = "r"
            desired_base_x = self.target_world[0]
        else:
            # 最西侧一列（shelf A，贴西墙）固定用左臂：右臂停车点会压到
            # 西墙（x≈-2.03）导致原地旋转/抓取卡死；左臂停车点偏东侧，安全。
            west_slot = fixed_layout_by_marker().get(self.target_marker_id)
            west_column = bool(west_slot and west_slot.get("shelf") == "A")
            desired_right_base_x = (
                self.target_world[0] - ARM_LATERAL_BIAS_M)
            if west_column or desired_right_base_x < NAV_X_MIN:
                self.grasp_arm = "l"
                desired_base_x = (
                    self.target_world[0] + ARM_LATERAL_BIAS_M)
            else:
                self.grasp_arm = "r"
                desired_base_x = desired_right_base_x
        self.align_base_x = float(np.clip(
            desired_base_x, NAV_X_MIN, NAV_X_MAX))
        # X 执行偏差补偿（见 GRASP_TCP_X_OFFSET_BY_ARM）：反向偏移指令
        # 目标，基座同步跟随，保持横向偏置不变。
        if (not self.use_dual_tissue_grasp
                and self.target_kind not in SPHERE_RADIUS_M):
            x_offset = GRASP_TCP_X_OFFSET_BY_ARM.get(self.grasp_arm, 0.0)
            if x_offset:
                self.target_world[0] += x_offset
                self.align_base_x = float(np.clip(
                    self.align_base_x + x_offset, NAV_X_MIN, NAV_X_MAX))
        if self.target_world[2] >= TOP_SHELF_Z_M:
            self.shelf_level = "top"
        elif self.target_world[2] >= MIDDLE_SHELF_Z_MIN_M:
            self.shelf_level = "middle"
        else:
            self.shelf_level = "lower"
        self.object_geometry = (
            "sphere" if self.target_kind in SPHERE_RADIUS_M else "generic")
        self.is_top_shelf = self.shelf_level == "top"
        self.use_sphere_grasp = bool(
            self.object_geometry == "sphere"
            and self.shelf_level in ("top", "middle"))
        if self.is_top_shelf:
            self.align_base_y = float(
                self.target_world[1] - TOP_GRASP_CENTER_DISTANCE_M)
        elif self.use_dual_tissue_grasp:
            self.align_base_y = (
                SCAN_Y + GENERIC_EXTENSION_ALIGN_FORWARD_M
                + DUAL_TISSUE_ALIGN_FORWARD_M)
        elif (self.shelf_level in ("middle", "lower")
              and self.object_geometry != "sphere"):
            self.align_base_y = (
                SCAN_Y + GENERIC_EXTENSION_ALIGN_FORWARD_M)
        else:
            self.align_base_y = SCAN_Y
        self.slide_grasp = float(np.clip(
            SLIDE_REFERENCE_COMMAND
            - (self.target_world[2] - SLIDE_REFERENCE_Z_M),
            SLIDE_MIN, SLIDE_MAX))
        self.ik_retry_forward_m = 0.0
        self.set_state(STATE_ALIGN)
        depth_text = (
            "" if depth_median is None
            else f" depth_world={np.round(depth_median, 3)}")
        self.get_logger().info(
            f"[localised] source={target_source} marker={marker_id} "
            f"samples={len(samples)} "
            f"spread={spread:.4f}m marker_world="
            f"{np.round(marker_median, 3)}"
            f"{depth_text} product_world="
            f"{np.round(self.target_world, 3)} "
            f"arm={'both' if self.use_dual_tissue_grasp else self.grasp_arm} "
            f"grasp_profile={self.grasp_profile_name()} "
            f"align_y={self.align_base_y:.3f}")

    @staticmethod
    def _verification_poses_for_z(z: float) -> tuple:
        """Return close camera poses suitable for one shelf level."""
        z = float(z)
        if z >= TOP_SHELF_SURFACE_Z_M - 0.10:
            return (
                ("overview_high", 0.11, 0.00, -0.20),
                ("overview_mid", 0.11, 0.00, -0.45),
            )
        if z >= MIDDLE_SHELF_SURFACE_Z_M - 0.10:
            return (
                ("overview_mid", 0.11, 0.00, -0.45),
                ("overview_down", 0.11, 0.00, -0.65),
            )
        return (
            ("lower_center", 0.45, 0.00, -0.45),
            ("lower_yaw_minus", 0.45, -0.15, -0.45),
            ("lower_yaw_plus", 0.45, 0.15, -0.45),
        )

    def _start_close_recheck(self) -> None:
        """Start fresh multi-pose verification at the selected slot."""
        self.recheck_poses = self._verification_poses_for_z(
            float(self.target_world[2]))
        self.recheck_pose_index = 0
        now = self.now()
        self.recheck_started_at = now
        self.recheck_pose_started_at = now
        self.recheck_confirmation_times.clear()
        self.recheck_last_yolo_stamp = None
        self.scan_camera_ready_since = None
        with self.lock:
            self.yolo_frames.clear()
            self.aruco_frames.clear()
        self.get_logger().info(
            f"[close-recheck] starting marker={self.target_marker_id} "
            f"kind={self.target_kind} poses="
            f"{[pose[0] for pose in self.recheck_poses]}")

    def current_recheck_pose(self):
        if not self.recheck_poses:
            return None
        return self.recheck_poses[self.recheck_pose_index]

    def try_recheck_locked(self) -> None:
        """Count fresh close-view class confirmations once per YOLO frame."""
        if self.state != STATE_RECHECK or self.current_recheck_pose() is None:
            return
        # A YOLO-only target has no decoded marker but still receives the same
        # close-view depth verification before any arm motion.
        if not self.yolo_frames:
            return
        if not self.scan_camera_ready(self.current_recheck_pose()):
            return
        if (self.scan_camera_ready_since is None
                or self.now() - self.scan_camera_ready_since
                < SCAN_CAMERA_STABLE_S):
            return

        yolo_stamp, detections = self.yolo_frames[-1]
        if yolo_stamp == self.recheck_last_yolo_stamp:
            return
        self.recheck_last_yolo_stamp = yolo_stamp

        # ArUco is preferred when a synchronized close-view frame exists.  It
        # is deliberately optional: the product or shelf edge can occlude the
        # marker at close range, in which case depth-based slot verification
        # must still be able to run.
        markers = []
        if self.aruco_frames:
            aruco_stamp, candidate_markers = min(
                self.aruco_frames,
                key=lambda frame: abs(frame[0] - yolo_stamp))
            if abs(aruco_stamp - yolo_stamp) <= ARUCO_SYNC_TOLERANCE_NS:
                markers = candidate_markers

        for detection in sorted(
                detections,
                key=lambda item: float(item.get("conf", 0.0)),
                reverse=True):
            matched, source = self._recheck_detection_matches(
                detection, markers)
            if not matched:
                continue
            now = self.now()
            cutoff = now - CLOSE_RECHECK_WINDOW_S
            while (self.recheck_confirmation_times
                   and self.recheck_confirmation_times[0] < cutoff):
                self.recheck_confirmation_times.popleft()
            self.recheck_confirmation_times.append(now)
            self.get_logger().info(
                f"[close-recheck] marker={self.target_marker_id} "
                f"kind={self.target_kind} source={source} "
                f"conf={float(detection.get('conf', 0.0)):.3f} "
                f"count={len(self.recheck_confirmation_times)}/"
                f"{CLOSE_RECHECK_CONFIRMATIONS}")
            return

    def _recheck_detection_matches(
            self, detection: dict,
            markers: list[dict]) -> tuple[bool, str]:
        """Verify a matching marker or fall through to the same 3-D slot."""
        marker = marker_below_yolo(detection, markers)
        marker_id = None
        if marker is not None:
            try:
                marker_id = int(marker["id"])
            except (KeyError, TypeError, ValueError):
                marker_id = None
            if (self.target_marker_id is not None
                    and marker_id == int(self.target_marker_id)):
                return True, "aruco"

        world = self._detection_world(detection)
        try:
            target = np.asarray(self.target_world, dtype=float)
        except (TypeError, ValueError):
            return False, "depth"
        if (world is None or target.shape != (3,)
                or not np.all(np.isfinite(target))):
            return False, "depth"
        matched = (
            abs(float(world[0] - target[0])) <= CLOSE_RECHECK_XY_MAX_M
            and abs(float(world[1] - target[1])) <= CLOSE_RECHECK_XY_MAX_M
            and abs(float(world[2] - target[2])) <= CLOSE_RECHECK_Z_MAX_M)
        if marker_id is None:
            source = "depth(no-aruco)"
        elif self.target_marker_id is None:
            source = f"depth(ignore-aruco={marker_id})"
        else:
            source = (
                f"depth(ignore-aruco={marker_id},"
                f"expected={self.target_marker_id})")
        return matched, source

    def _recheck_confirmed(self) -> bool:
        cutoff = self.now() - CLOSE_RECHECK_WINDOW_S
        while (self.recheck_confirmation_times
               and self.recheck_confirmation_times[0] < cutoff):
            self.recheck_confirmation_times.popleft()
        return (len(self.recheck_confirmation_times)
                >= CLOSE_RECHECK_CONFIRMATIONS)

    def _advance_recheck_pose(self) -> bool:
        """Move to the next view; return False when all views are exhausted."""
        if self.recheck_pose_index + 1 >= len(self.recheck_poses):
            return False
        self.recheck_pose_index += 1
        self.recheck_pose_started_at = self.now()
        self.recheck_confirmation_times.clear()
        self.recheck_last_yolo_stamp = None
        self.scan_camera_ready_since = None
        with self.lock:
            self.yolo_frames.clear()
            self.aruco_frames.clear()
        self.get_logger().info(
            f"[close-recheck] trying alternate pose="
            f"{self.current_recheck_pose()[0]}")
        return True

    def _recheck_fail(self) -> None:
        """Skip this slot and resume scanning (after bounded adjacent-column
        retries for memory direct-slot targets)."""
        marker_id = self.target_marker_id
        if marker_id is not None:
            self.recheck_marker_skips.add(marker_id)
        else:
            skipped_slot = self.target_slot_key()
            if skipped_slot is not None:
                self.excluded_slot_keys.add(skipped_slot)
        self._recheck_passed = False
        self.recheck_poses = ()
        self.recheck_confirmation_times.clear()
        self.recheck_last_yolo_stamp = None
        self.scan_camera_ready_since = None
        # 方案 B：记忆直达槽位复核失败时，先试同货架相邻列（横向 0.22m 级），
        # 而不是立刻回货架中心全量扫描。列证据通常只差一格，相邻列命中即可
        # 省掉整轮扫描/补拍/重新定位/二次对齐（实测约 26s → 约 5s）。
        if (marker_id is None
                and self.direct_slot_target_active
                and self._try_adjacent_direct_slot()):
            return
        self.get_logger().warn(
            f"[close-recheck] FAILED marker={marker_id} "
            f"kind={self.target_kind}; all close-view poses exhausted; "
            "skipping this slot and resuming the shelf scan")
        self.target_marker_id = None
        self.target_physical_marker_id = None
        self.target_world = None
        self.committed_slot = None
        self.direct_slot_target_active = False
        self.association_candidate_id = None
        self.association_confirmation_count = 0
        self.marker_positions.clear()
        self.depth_target_samples.clear()
        self.last_association_pair = None
        self._restore_full_scan_after_inventory_hint()
        self.scan_index = 0
        self.scan_pose_index = 0
        self.scan_station_order = self._nearest_scan_stations()
        with self.lock:
            self.yolo_frames.clear()
            self.aruco_frames.clear()
        self.set_state(STATE_GO_SCAN)

    def _try_adjacent_direct_slot(self) -> bool:
        """复核失败后改试同货架相邻列（就近方向优先，受重试次数限制）。

        只在记忆直达槽位（``configure_direct_slot_target`` 建立的
        ``memory_direct`` 目标）复核失败时调用；命中后继续走原
        直达 + ALIGN + close-recheck 流程，全部相邻列失败才回退全量扫描。
        """
        slot = self.target_slot()
        if slot is None:
            return False
        shelf, level, column = slot
        if (self.direct_slot_adjacent_retries
                >= self.direct_slot_adjacent_max_retries):
            return False
        shelf_x = SCAN_X[self._scan_x_index_for_shelf(shelf)]
        base_x = (
            float(self.base_xy[0]) if self.base_xy is not None
            else float("nan"))
        candidates = []
        for delta in (-1, 1):
            next_column = str(int(column) + delta)
            if next_column not in {"1", "2", "3"}:
                continue
            slot_key = f"{level}|{shelf}|{next_column}"
            if slot_key in self.excluded_slot_keys:
                continue
            target_x = shelf_x + {
                "1": -0.22, "2": 0.00, "3": 0.22}[next_column]
            distance = (
                float("inf") if not math.isfinite(base_x)
                else abs(base_x - target_x))
            candidates.append((distance, next_column))
        if not candidates:
            return False
        candidates.sort(key=lambda item: item[0])
        next_column = candidates[0][1]
        self.direct_slot_adjacent_retries += 1
        self.get_logger().warn(
            f"[direct-slot-retry] recheck failed at "
            f"{shelf}-{level}-{column}; trying adjacent column "
            f"{shelf}-{level}-{next_column} "
            f"({self.direct_slot_adjacent_retries}/"
            f"{self.direct_slot_adjacent_max_retries})")
        if not self.configure_direct_slot_target(
                shelf, level, next_column):
            self.direct_slot_adjacent_retries -= 1
            return False
        return True

    def initialize_commands(self) -> None:
        self.cmd_slide = self.joints.get("slide_joint", 0.0)
        self.cmd_head = np.array([
            self.joints.get("head_yaw_joint", 0.0),
            self.joints.get("head_pitch_joint", 0.0),
        ])
        self.cmd_left_arm = np.array([
            self.joints.get(f"left_arm_joint{i + 1}", 0.0) for i in range(6)])
        self.cmd_right_arm = np.array([
            self.joints.get(f"right_arm_joint{i + 1}", 0.0) for i in range(6)])
        self.cmd_left_grip = self.joints.get(
            "left_arm_eef_gripper_joint", GRIP_OPEN)
        self.cmd_right_grip = self.joints.get(
            "right_arm_eef_gripper_joint", GRIP_OPEN)
        self.des_slide = self.cmd_slide
        self.des_head = self.cmd_head.copy()
        self.des_left_arm = self.cmd_left_arm.copy()
        self.des_right_arm = self.cmd_right_arm.copy()
        self.des_left_grip = self.cmd_left_grip
        self.des_right_grip = self.cmd_right_grip
        self.initialized = True

    def set_state(self, new_state: str) -> None:
        if new_state == self.state:
            return
        self.state = new_state
        self.state_t0 = self.now()
        self.state_monotonic_t0 = time.monotonic()
        self.nav_target = None
        self.commands_ready_since = None
        if new_state == STATE_SCAN:
            self.scan_unlocked_markers.clear()
            self.scan_unlocked_boxes.clear()
            if not self.opportunistic_target_locked:
                self.opportunistic_yolo_frames.clear()
                self.opportunistic_target_pairs.clear()
        if new_state == STATE_CLOSE and not self.use_sphere_grasp:
            self.generic_close_start_grip = self.selected_gripper_position()
            self.generic_close_stage = 1
            self.generic_close_grip_samples.clear()
        self.get_logger().info(f"state -> {new_state}")
        if new_state in (
                STATE_CLOSE, STATE_LIFT, STATE_RETREAT,
                STATE_DONE, STATE_ABORT):
            self.log_manipulation_snapshot(new_state)

    def state_elapsed_monotonic(self) -> float:
        """Wall-clock time in the active state, immune to ROS clock jumps."""
        return max(0.0, time.monotonic() - self.state_monotonic_t0)

    def _on_rotation_recovery_exhausted(self) -> None:
        """Fail a pre-grasp controller after repeated rotation stalls."""
        self.abort_reason = "rotation recovery budget exhausted"
        self.set_state(STATE_ABORT)

    def _abort_recovery_ready(self) -> bool:
        """True when the abort posture has settled enough to shut down.

        The base implementation keeps the original settle gate (0.5 s of
        converged commands or the fixed abort timeout).  Subclasses with a
        known neutral manipulation posture (e.g. IntegratedNavPickPlace)
        override this to restore arms, grippers, slide and head to the
        initial pose before the worker exits, so a failed order never leaves
        the next worker with a closed gripper or a deployed arm.
        """
        elapsed = self.now() - self.state_t0
        return bool(
            (elapsed > 0.5 and self.commands_ready())
            or elapsed >= ABORT_SHUTDOWN_TIMEOUT_S)

    def log_manipulation_snapshot(self, label: str) -> None:
        """Log measured grasp geometry without changing control decisions."""
        if self.use_dual_tissue_grasp:
            left_tcp = self.arm_tcp_world("left")
            right_tcp = self.arm_tcp_world("right")
            line_angle = (
                None if left_tcp is None or right_tcp is None
                else math.degrees(math.atan2(
                    right_tcp[1] - left_tcp[1],
                    right_tcp[0] - left_tcp[0])))
            left_grip = self.joints.get("left_arm_eef_gripper_joint")
            right_grip = self.joints.get("right_arm_eef_gripper_joint")
            self.get_logger().info(
                f"[dual-tissue-snapshot] phase={label} "
                f"left_tcp={None if left_tcp is None else np.round(left_tcp, 4)} "
                f"right_tcp={None if right_tcp is None else np.round(right_tcp, 4)} "
                f"line_angle={line_angle}deg "
                f"left_grip={left_grip} right_grip={right_grip} "
                f"base={np.round(self.base_xy, 4)}")
        actual_tcp = self.selected_tcp_world()
        contact_error = (
            None if actual_tcp is None or self.forward_contact_world is None
            else actual_tcp - self.forward_contact_world)
        product_error = (
            None if actual_tcp is None or self.target_world is None
            else actual_tcp - self.target_world)
        base_delta = (
            None if self.forward_start_base_xy is None
            else self.base_xy - self.forward_start_base_xy)
        gripper = self.selected_gripper_position()
        self.get_logger().info(
            f"[grasp-snapshot] phase={label} "
            f"tcp={None if actual_tcp is None else np.round(actual_tcp, 4)} "
            f"contact_error={None if contact_error is None else np.round(contact_error, 4)}m "
            f"product_error={None if product_error is None else np.round(product_error, 4)}m "
            f"base={np.round(self.base_xy, 4)} "
            f"base_delta={None if base_delta is None else np.round(base_delta, 4)}m "
            f"grip={gripper} preshape={self.grip_preshape_command:.3f}")

    def set_twist(self, linear: float, angular: float) -> None:
        self.des_linear = float(np.clip(linear, -0.90, 0.90))
        self.des_angular = float(np.clip(angular, -2.50, 2.50))

    def begin_manip_base_hold(self) -> None:
        """Capture the actual top/middle/lower base pose without adding a gate."""
        self.manip_base_hold_xy = self.base_xy.copy()
        self.manip_base_hold_yaw = float(self.base_yaw)
        self.manip_base_hold_last_log = 0.0
        self.get_logger().info(
            f"[{self.shelf_level}-base-hold] captured reference base="
            f"{np.round(self.manip_base_hold_xy, 4)} "
            f"yaw={self.manip_base_hold_yaw:.4f}; "
            f"linear_limit={MANIP_BASE_HOLD_LINEAR_MAX_MPS:.3f}m/s "
            f"yaw_limit={MANIP_BASE_HOLD_YAW_MAX_RADPS:.3f}rad/s")

    def _deploy_base_nudge_retry(self, reason: str) -> bool:
        """部署期 pregrasp 未收敛：向前微调基座并回到 ALIGN 重新解算重试。

        返回 True 表示已触发重试；达到上限或基座已到最前时返回 False，
        调用方应继续走原有的中止路径。
        """
        if (self.deploy_retry_count >= GENERIC_DEPLOY_RETRY_MAX
                or self.align_base_y >= GENERIC_ALIGN_Y_MAX_M - 1e-6):
            if (self.shelf_level == "top"
                    and not self.use_dual_tissue_grasp
                    and self._switch_grasp_arm_retry(reason)):
                return True
            return False
        self.deploy_retry_count += 1
        nudge = min(
            GENERIC_DEPLOY_RETRY_STEP_M,
            GENERIC_ALIGN_Y_MAX_M - self.align_base_y)
        self.align_base_y = float(np.clip(
            self.align_base_y + nudge,
            DUAL_ALIGN_Y_MIN_M, GENERIC_ALIGN_Y_MAX_M))
        self.get_logger().warn(
            f"[{reason}] pregrasp did not converge; base-nudge retry "
            f"{self.deploy_retry_count}/{GENERIC_DEPLOY_RETRY_MAX} -> "
            f"align_y={self.align_base_y:.3f}")
        self.set_state(STATE_ALIGN)
        return True

    def _switch_grasp_arm_retry(self, reason: str) -> bool:
        """顶层右/左臂预抓取不收敛时，换另一条手臂重试一次。

        顶层的 front-upright 腕部姿态与当前收臂姿态差异很大，某条手臂可能
        因为关节余量不足而一直无法收敛；换另一条手臂往往能解出可用位姿。
        只允许切换一次，避免两条手臂来回切换形成死循环。
        """
        if self._grasp_arm_switch_count >= 1:
            return False
        current_offset = GRASP_TCP_X_OFFSET_BY_ARM.get(self.grasp_arm, 0.0)
        base_slot_x = float(self.target_world[0]) - current_offset
        new_arm = "l" if self.grasp_arm == "r" else "r"
        if new_arm == "l":
            desired_base_x = base_slot_x + ARM_LATERAL_BIAS_M
        else:
            desired_base_x = base_slot_x - ARM_LATERAL_BIAS_M
        new_offset = GRASP_TCP_X_OFFSET_BY_ARM.get(new_arm, 0.0)
        if self.is_top_shelf and new_arm == "l":
            new_offset += TOP_LEFT_GRASP_X_BIAS_M
        self.grasp_arm = new_arm
        self.target_world[0] = float(base_slot_x + new_offset)
        self.align_base_x = float(np.clip(
            desired_base_x + new_offset, NAV_X_MIN, NAV_X_MAX))
        if self.is_top_shelf:
            self.align_base_y = float(
                self.target_world[1] - TOP_GRASP_CENTER_DISTANCE_M)
        self.ik_retry_forward_m = 0.0
        self.deploy_retry_count = 0
        # 目标点未变，只是换手抓取，不需要再重做一遍近距离复核。
        self._recheck_passed = True
        self._grasp_arm_switch_count += 1
        self.get_logger().warn(
            f"[{reason}] top-front pregrasp did not converge with arm="
            f"{'r' if new_arm == 'l' else 'l'}; switching to arm={new_arm} "
            f"target_x={self.target_world[0]:.3f} "
            f"align_x={self.align_base_x:.3f} "
            f"align_y={self.align_base_y:.3f}")
        self.set_state(STATE_ALIGN)
        return True

    def _proceed_to_deploy(self) -> None:
        """从停稳门控进入抓取动作。

        只负责动作选择和状态切换，不含底盘运动。原 ALIGN/RECHECK 中
        ``configure_grasp`` 之后的所有安全门控保持不变。
        """
        if self._prepare_tissue_rotation_if_needed():
            return
        grasp_status = self.configure_grasp()
        if grasp_status is True:
            if self.shelf_level in ("top", "middle", "lower"):
                self.begin_manip_base_hold()
            self.set_state(STATE_DEPLOY)
        elif grasp_status == "retry":
            self.set_state(STATE_ALIGN)
        else:
            self.set_state(STATE_ABORT)

    def _start_grasp_settle(self) -> None:
        """导航已到位，先抓取当前底盘快照并等待停稳。"""
        self.set_twist(0.0, 0.0)
        self._grasp_settle_anchor_xy = self.base_xy.copy()
        self._grasp_settle_anchor_yaw = float(self.base_yaw)
        self._grasp_settle_started_at = self.now()
        self._grasp_settle_logged = False
        self.set_state(STATE_GRASP_SETTLE)

    def _grasp_settle_tick(self) -> None:
        """底盘停稳后进入臂部动作；超时放行但不跳过抓取前动作门控。"""
        if (self._grasp_settle_anchor_xy is None
                or self._grasp_settle_anchor_yaw is None
                or self._grasp_settle_started_at is None):
            self._start_grasp_settle()
            return

        self.set_twist(0.0, 0.0)
        elapsed = max(0.0, self.now() - self._grasp_settle_started_at)
        moved = float(np.linalg.norm(
            self.base_xy - self._grasp_settle_anchor_xy))
        yaw_moved = abs(wrap_to_pi(
            self.base_yaw - self._grasp_settle_anchor_yaw))

        if (moved <= GRASP_BASE_SETTLE_MAX_XY_M
                and yaw_moved <= GRASP_BASE_SETTLE_MAX_YAW_RAD):
            if elapsed >= GRASP_BASE_SETTLE_S:
                self._grasp_settle_anchor_xy = None
                self._grasp_settle_anchor_yaw = None
                self._grasp_settle_started_at = None
                self._proceed_to_deploy()
            return

        # 底盘仍在移动：重置停稳锚点，直到静止窗口连续满足。
        if elapsed >= GRASP_BASE_SETTLE_TIMEOUT_S:
            if not self._grasp_settle_logged:
                self._grasp_settle_logged = True
                self.get_logger().warn(
                    "[grasp-settle] base did not fully settle before "
                    f"timeout; elapsed={elapsed:.2f}s moved={moved:.3f}m "
                    f"yaw={yaw_moved:.3f}rad; proceeding with arm gates")
            self._grasp_settle_anchor_xy = None
            self._grasp_settle_anchor_yaw = None
            self._grasp_settle_started_at = None
            self._proceed_to_deploy()
            return

        self._grasp_settle_anchor_xy = self.base_xy.copy()
        self._grasp_settle_anchor_yaw = float(self.base_yaw)
        self._grasp_settle_started_at = self.now()

    def apply_manip_base_hold(self) -> None:
        """Softly oppose top/middle/lower reaction forces; never block the arm."""
        active_states = (
            STATE_TISSUE_ROTATE, STATE_DEPLOY, STATE_ARM_FORWARD,
            STATE_POST_EXTEND,
            STATE_CLOSE, STATE_TRIAL_LIFT, STATE_LIFT)
        if (self.shelf_level not in ("top", "middle", "lower")
                or self.state not in active_states
                or self.manip_base_hold_xy is None
                or self.manip_base_hold_yaw is None):
            return
        if (self.shelf_level == "top"
                and self.use_dual_tissue_grasp
                and self.dual_top_extract_stage >= 8):
            # Once the fork touches the exposed lip, even a centimetre of
            # chassis correction changes its insertion depth.  A failed run
            # drove 15 mm toward the shelf during lift, pinned the fork to the
            # board and left 103 mm of TCP error.  Zero wheel command (set at
            # the start of tick) is the safe hold policy for this phase.
            return

        world_error = self.manip_base_hold_xy - self.base_xy
        heading = np.array([
            math.cos(self.base_yaw), math.sin(self.base_yaw)])
        longitudinal_error = float(np.dot(world_error, heading))
        yaw_error = wrap_to_pi(
            self.manip_base_hold_yaw - self.base_yaw)

        if longitudinal_error <= MANIP_BASE_HOLD_LINEAR_DEADBAND_M:
            linear = 0.0
        else:
            # Only oppose the observed backward recoil.  Deliberately avoid a
            # reverse correction if the base moves slightly forward: pulling
            # the chassis back while the fingers are closing would destabilise
            # a product that has already entered the gripper.
            linear = float(np.clip(
                MANIP_BASE_HOLD_LINEAR_KP * longitudinal_error,
                0.0,
                MANIP_BASE_HOLD_LINEAR_MAX_MPS))
        if abs(yaw_error) <= MANIP_BASE_HOLD_YAW_DEADBAND_RAD:
            angular = 0.0
        else:
            angular = float(np.clip(
                MANIP_BASE_HOLD_YAW_KP * yaw_error,
                -MANIP_BASE_HOLD_YAW_MAX_RADPS,
                MANIP_BASE_HOLD_YAW_MAX_RADPS))
        self.set_twist(linear, angular)

        now = self.now()
        if now - self.manip_base_hold_last_log >= MANIP_BASE_HOLD_LOG_PERIOD_S:
            self.manip_base_hold_last_log = now
            saturated = (
                abs(linear) >= MANIP_BASE_HOLD_LINEAR_MAX_MPS - 1e-6
                or abs(angular) >= MANIP_BASE_HOLD_YAW_MAX_RADPS - 1e-6)
            self.get_logger().info(
                f"[{self.shelf_level}-base-hold] state={self.state} "
                f"error_world={np.round(world_error, 4)}m "
                f"longitudinal_error={longitudinal_error:+.4f}m "
                f"yaw_error={yaw_error:+.4f}rad "
                f"command=({linear:+.3f}m/s,{angular:+.3f}rad/s) "
                f"saturated={int(saturated)}")

    def current_scan_camera_pose(self):
        return self.scan_poses[self.scan_pose_index]

    def configure_inventory_scan_hint(
            self, world_x: float, marker_z: float | None = None) -> None:
        """Prioritise shelf views measured during an earlier order.

        This only selects the first station and camera level.  Normal
        YOLO/ArUco confirmation, localisation and close recheck remain
        mandatory before a grasp.
        """
        self.scan_preferred_x = float(world_x)
        if marker_z is None or not math.isfinite(float(marker_z)):
            return
        marker_z = float(marker_z)
        if marker_z >= 1.05:
            self.scan_poses = (SCAN_CAMERA_POSES[0],)
            level = "top"
        elif marker_z >= 0.70:
            self.scan_poses = SCAN_CAMERA_POSES[1:2]
            level = "middle"
        else:
            self.scan_poses = SCAN_CAMERA_POSES[2:]
            level = "lower"
        self.inventory_scan_hint_active = True
        self.get_logger().info(
            f"using measured inventory scan hint x={world_x:.3f} "
            f"marker_z={marker_z:.3f} level={level} "
            f"poses={[pose[0] for pose in self.scan_poses]}; "
            "perception confirmation remains required")

    def _restore_full_scan_after_inventory_hint(self) -> None:
        if not self.inventory_scan_hint_active:
            return
        self.inventory_scan_hint_active = False
        self.scan_poses = self.default_scan_poses
        self.scan_pose_index = 0

    def _nearest_scan_stations(self) -> list[int]:
        """Order scan stations by travel distance from current base position.

        A confirmed cross-order inventory observation takes precedence.  If
        it later proves stale, the remaining stations are still scanned in
        increasing distance from the hinted station.  Without a hint, retain
        the existing west-first/nearest-first policies.
        """
        if self.base_xy is None:
            base = np.zeros(2, dtype=float)
        else:
            base = np.asarray(self.base_xy, dtype=float)
        if self.scan_preferred_x is not None:
            preferred = min(
                range(len(SCAN_X)),
                key=lambda index: (
                    abs(SCAN_X[index] - float(self.scan_preferred_x)),
                    index))
            return [preferred] + sorted(
                (index for index in range(len(SCAN_X))
                 if index != preferred),
                key=lambda index: (
                    abs(SCAN_X[index] - SCAN_X[preferred]), index))
        if self.scan_prefer_west_start:
            # 第一单之后的订单：从最西侧（A 货架）向东依次扫
            return sorted(
                range(len(SCAN_X)),
                key=lambda index: (SCAN_X[index], index))
        ordered = sorted(
            range(len(SCAN_X)),
            key=lambda index: (
                float(np.linalg.norm(
                    np.array([SCAN_X[index], SCAN_Y]) - base)),
                index))
        return ordered

    def current_scan_station_x(self) -> float:
        if self.scan_station_order is None:
            return SCAN_X[self.scan_index]
        return SCAN_X[self.scan_station_order[self.scan_index]]

    def scan_camera_ready(self, pose=None) -> bool:
        if pose is None:
            pose = self.current_scan_camera_pose()
        _, slide_target, yaw_target, pitch_target = pose
        slide = self.joints.get("slide_joint")
        yaw = self.joints.get("head_yaw_joint")
        pitch = self.joints.get("head_pitch_joint")
        if slide is None or yaw is None or pitch is None:
            return False
        return (
            abs(slide - slide_target) <= SCAN_CAMERA_REACHED_SLIDE_M
            and abs(yaw - yaw_target) <= SCAN_CAMERA_REACHED_HEAD_RAD
            and abs(pitch - pitch_target) <= SCAN_CAMERA_REACHED_HEAD_RAD)

    def log_scan_perception_summary(self) -> None:
        """每 1s 打一行扫描感知摘要，用于定位"完全没关联"的槽位。

        区分三种情况：YOLO 有没有输出目标框、ArUco 有没有解到码、
        两者是否同步配对（配对成功会走 association 分支）。
        """
        with self.lock:
            yolo_records = (
                list(self.yolo_frames[-1][1]) if self.yolo_frames else [])
            yolo_total = (
                self.yolo_total_frames[-1][1]
                if self.yolo_total_frames else 0)
            aruco_records = (
                list(self.aruco_frames[-1][1]) if self.aruco_frames else [])
        target_records = [
            record for record in yolo_records
            if record.get("class") == self.target_kind]
        target_confs = sorted(
            round(float(record.get("conf", 0.0)), 3)
            for record in target_records)
        marker_ids = sorted(
            {int(record["id"]) for record in aruco_records
             if record.get("id") is not None})
        self.get_logger().info(
            f"[scan-diag] pose={self.current_scan_camera_pose()[0]} "
            f"station={self.current_scan_station_x():.2f} "
            f"yolo_total={yolo_total} "
            f"yolo_all={len(yolo_records)} "
            f"yolo_{self.target_kind}={len(target_records)} "
            f"conf={target_confs[:8]} "
            f"aruco_n={len(marker_ids)} ids={marker_ids[:24]}")

    def _start_revisit(self) -> None:
        """对当前位姿"已关联但未锁定"或"有框无码"的候选做定点多角度补拍。

        marker 锁定门槛与正常扫描完全一致（2 帧确认 + 5 个 marker
        样本 + 4cm 扩散）；无码候选同样保留并继续累计原有的 4 帧、
        9cm 扩散和置信度门槛。补拍只增加采样视角，不降低任何锁定门槛。
        """
        marker_candidates = {
            marker_id: entry for marker_id, entry in
            self.scan_unlocked_markers.items()
            if self.revisit_rounds.get(marker_id, 0)
            < REVISIT_MAX_ROUNDS_PER_MARKER}
        box_candidates = {
            box_key: entry for box_key, entry in
            self.scan_unlocked_boxes.items()
            if self.revisit_rounds.get(box_key, 0)
            < REVISIT_MAX_ROUNDS_PER_MARKER}
        if not marker_candidates and not box_candidates:
            return
        # 重置方向3备选信息（每次补拍都是新的候选）
        self.revisit_box_world = None
        self.revisit_box_conf = 0.0
        self.revisit_box_confirmations = 0
        self.revisit_box_key = None
        if marker_candidates:
            if (self.target_marker_id is not None
                    and self.target_marker_id in marker_candidates):
                # 优先补拍已通过 2 帧确认但样本不足的目标
                candidate_key = self.target_marker_id
            else:
                candidate_key = max(
                    marker_candidates,
                    key=lambda mid: marker_candidates[mid]["confirmations"])
            last_pose = marker_candidates[candidate_key].get("pose")
            candidate_world = marker_candidates[candidate_key].get("world")
            filter_marker = candidate_key
            candidate_text = f"marker={candidate_key}"
        else:
            # 有框无码：YOLO 检测到目标类但没解到码，补拍时接受任意码
            candidate_key = max(
                box_candidates,
                key=lambda bk: box_candidates[bk]["confirmations"])
            self.revisit_box_key = candidate_key
            last_pose = box_candidates[candidate_key].get("pose")
            filter_marker = None
            box_world = box_candidates[candidate_key].get("world")
            self.revisit_box_world = box_world
            self.revisit_box_conf = float(
                box_candidates[candidate_key].get("max_conf", 0.0))
            self.revisit_box_confirmations = int(
                box_candidates[candidate_key].get("confirmations", 0))
            candidate_world = box_world
            candidate_text = (
                f"box={candidate_key} world="
                f"{np.round(box_world, 3).tolist()}"
                if box_world is not None else f"box={candidate_key}")
        # 把最后关联到该候选的姿态排到补拍首位：该姿态已证明能检测到货物，
        # 优先用它补样本，其余姿态作为多角度兜底。
        poses = list(self._revisit_poses_for_world(candidate_world))
        if last_pose:
            pose_by_name = {pose[0]: pose for pose in REVISIT_POSES}
            if last_pose in pose_by_name:
                poses = [pose for pose in poses if pose[0] != last_pose]
                poses.insert(0, pose_by_name[last_pose])
        self.revisit_marker_id = filter_marker
        self.revisit_poses = tuple(poses)
        self.revisit_pose_index = 0
        self.revisit_pose_t0 = self.now()
        self.revisit_pose_monotonic_t0 = time.monotonic()
        self.revisit_rounds[candidate_key] = (
            self.revisit_rounds.get(candidate_key, 0) + 1)
        self.revisit_total_rounds += 1
        # If the normal scan already confirmed this exact marker and only ran
        # out of dwell before reaching the five-sample gate, keep those good
        # samples.  The previous implementation discarded 1--4 stable samples
        # here, then spent the long first-pose revisit dwell collecting the
        # same evidence again.  Never preserve a complete-but-dispersed set:
        # an outlier would keep ptp() above the unchanged 4 cm safety gate for
        # the entire revisit because that statistic cannot shrink.
        preserve_partial_marker_samples = False
        if (filter_marker is not None
                and self.target_marker_id == filter_marker
                and 0 < len(self.marker_positions) < MARKER_SAMPLES_REQUIRED):
            partial_samples = np.asarray(
                self.marker_positions, dtype=float)
            preserve_partial_marker_samples = bool(
                partial_samples.ndim == 2
                and partial_samples.shape[1:] == (3,)
                and np.all(np.isfinite(partial_samples))
                and float(np.max(np.ptp(partial_samples, axis=0)))
                <= MARKER_SAMPLE_SPREAD_MAX_M)
        if preserve_partial_marker_samples:
            self.get_logger().info(
                f"[revisit] preserving {len(self.marker_positions)}/"
                f"{MARKER_SAMPLES_REQUIRED} stable samples for marker="
                f"{filter_marker}; localisation thresholds unchanged")
        else:
            self.marker_positions.clear()
            self.depth_target_samples.clear()
            self.association_candidate_id = None
            self.association_confirmation_count = 0
            self.last_association_pair = None
            self.target_marker_id = None
            self.target_physical_marker_id = None
        self.scan_unlocked_markers.clear()
        if self.revisit_box_key is None:
            self.scan_unlocked_boxes.clear()
        else:
            selected_box = self.scan_unlocked_boxes.get(
                self.revisit_box_key)
            self.scan_unlocked_boxes = (
                {} if selected_box is None
                else {self.revisit_box_key: selected_box})
        self.scan_camera_ready_since = None
        self.set_state(STATE_REVISIT)
        self.get_logger().info(
            f"[revisit] {candidate_text} kind={self.target_kind} "
            f"station_x={self.current_scan_station_x():.3f} "
            f"first_pose={self.revisit_poses[0][0]} "
            f"poses={len(self.revisit_poses)} "
            f"first_dwell={REVISIT_FIRST_POSE_DWELL_S:.1f}s "
            f"dwell={REVISIT_DWELL_S:.1f}s "
            f"rounds={self.revisit_total_rounds}/"
            f"{REVISIT_MAX_ROUNDS_PER_SCAN}")

    @staticmethod
    def _revisit_poses_for_world(world) -> tuple:
        """Use only camera poses for the candidate's measured shelf level."""
        try:
            z = float(np.asarray(world, dtype=float)[2])
        except (TypeError, ValueError, IndexError):
            return REVISIT_POSES
        if not math.isfinite(z):
            return REVISIT_POSES
        if z >= 1.05:
            return (SCAN_CAMERA_POSES[0],)
        if z >= 0.70:
            return SCAN_CAMERA_POSES[1:2]
        return SCAN_CAMERA_POSES[2:]

    def _revisit_fail(self) -> None:
        """补拍未锁定：放弃该槽位本轮补拍，恢复主扫描的下一个位姿。"""
        was_box_revisit = self.revisit_marker_id is None
        target_text = (
            f"marker={self.revisit_marker_id}"
            if self.revisit_marker_id is not None
            else "box(any-marker)")
        self.get_logger().warn(
            f"[revisit] FAILED {target_text} kind={self.target_kind}; "
            "resuming normal scan")
        self.revisit_marker_id = None
        self.revisit_poses = REVISIT_POSES
        self.revisit_pose_index = 0
        self.revisit_pose_t0 = 0.0
        self.marker_positions.clear()
        self.depth_target_samples.clear()
        self.association_candidate_id = None
        self.association_confirmation_count = 0
        self.last_association_pair = None
        self.target_marker_id = None
        self.target_physical_marker_id = None
        self.scan_camera_ready_since = None
        if (was_box_revisit and self.revisit_box_world is not None
                and not self.tcp_diagnostic_ground_truth
                and self._try_position_fallback()):
            self.revisit_box_world = None
            self.revisit_box_key = None
            return
        self.revisit_box_world = None
        self.revisit_box_key = None
        self._advance_scan_pose()

    def _try_position_fallback(self) -> bool:
        """方向3备选：ArUco 码无法解码时，用 YOLO 框世界坐标推断固定槽位。

        仅在 box 补拍失败后触发。固定布局中的 marker ID 只用于找到槽位
        几何和检查排除列表，不得写入 ``target_marker_id`` 冒充一次真实解码；
        后续近距离复核以 YOLO 深度为主，画面中的邻近码不再作为
        目标身份的硬门槛。
        """
        box_world = np.asarray(self.revisit_box_world, dtype=float)
        if box_world.shape != (3,) or not np.all(np.isfinite(box_world)):
            return False
        if self.revisit_box_conf < 0.85:
            self.get_logger().warn(
                "[position-fallback] skipped: box conf "
                f"{self.revisit_box_conf:.3f} < 0.85")
            return False
        if self.revisit_box_confirmations < 2:
            return False
        z = float(box_world[2])
        if not (0.40 <= z <= 1.40):
            return False
        level = (
            "L3" if z >= TOP_SHELF_Z_M
            else ("L2" if z >= MIDDLE_SHELF_Z_MIN_M else "L1"))
        slots = [slot for slot in fixed_layout_by_marker().values()
                 if slot.get("level") == level]
        if not slots:
            return False
        slot = min(
            slots, key=lambda s: abs(
                float(s["world_position"][0]) - box_world[0]))
        if abs(float(slot["world_position"][0]) - box_world[0]) > 0.08:
            self.get_logger().warn(
                "[position-fallback] skipped: no slot within 8cm of "
                f"box x={box_world[0]:.3f} level={level}")
            return False
        inferred_slot_marker_id = int(slot["aruco_id"])
        inferred_slot_key = (
            f"{slot['level']}|{slot['shelf']}|{str(slot['column'])[-1]}")
        if (self.target_kind == "zhijin"
                and str(slot["column"])[-1] != "2"
                and not DUAL_TISSUE_SIDE_ROLLED_ENABLED):
            if inferred_slot_key not in self.skipped_tissue_slots:
                self.skipped_tissue_slots.add(inferred_slot_key)
                self.get_logger().warn(
                    "[tissue-filter] position fallback rejected side-column "
                    f"tissue slot={inferred_slot_key}; only the middle "
                    "column is eligible")
            return False
        if inferred_slot_key in self.excluded_slot_keys:
            return False
        if inferred_slot_marker_id in self.excluded_marker_ids:
            return False
        if inferred_slot_marker_id in self.recheck_marker_skips:
            return False
        if inferred_slot_marker_id in self.skipped_tissue_markers:
            return False
        level_name = {"L3": "top", "L2": "middle", "L1": "lower"}[level]
        surface_z = SHELF_SURFACE_Z_M[level_name]
        max_center_z = (
            surface_z + PRODUCT_HALF_HEIGHT_M[self.target_kind]
            + PRODUCT_CENTER_Z_TOLERANCE_M)
        if z > max_center_z:
            self.get_logger().warn(
                f"[position-fallback] depth-only Z {z:.3f}m exceeds the "
                f"physical centre envelope for {self.target_kind} on "
                f"{level_name} (surface {surface_z:.3f} + half height "
                f"{PRODUCT_HALF_HEIGHT_M[self.target_kind]:.3f} + "
                f"{PRODUCT_CENTER_Z_TOLERANCE_M:.3f}); clamping to "
                f"{max_center_z:.3f}")
            z = float(max_center_z)

        self.target_world = np.array([
            float(slot["world_position"][0]),
            float(box_world[1] + PRODUCT_HALF_DEPTH_M[self.target_kind]),
            z], dtype=float)
        # 无码就是无码：不要把按世界 x 推断出的槽位码当成真实 ArUco。
        self.target_marker_id = None
        self.target_physical_marker_id = None
        self._recheck_passed = False
        self.get_logger().warn(
            f"[position-fallback] inferred_slot_marker="
            f"{inferred_slot_marker_id} undecodable; "
            f"using YOLO world {np.round(box_world, 3)} -> target "
            f"{np.round(self.target_world, 3)} kind={self.target_kind} "
            f"level={level}")
        self._commit_localised_target(
            self.target_world, None, "position_fallback",
            extra=(
                f" inferred_slot_marker={inferred_slot_marker_id}"
                f" box_world={np.round(box_world, 3)}"),
            shelf=str(slot["shelf"]))
        return True

    def _advance_scan_pose(self) -> bool:
        """推进到下一个扫描位姿/站点/周期；返回 True 表示扫描仍在继续。"""
        self.scan_pose_index += 1
        self.scan_camera_ready_since = None
        if self.scan_pose_index >= len(self.scan_poses):
            if self.inventory_scan_hint_active:
                hinted_poses = [pose[0] for pose in self.scan_poses]
                self._restore_full_scan_after_inventory_hint()
                self.get_logger().warn(
                    "measured inventory views did not confirm the target; "
                    f"hinted_poses={hinted_poses}; restoring the full scan "
                    "at the same station")
                self.set_state(STATE_GO_SCAN)
                return True
            self.scan_pose_index = 0
            self.scan_index += 1
            if self.scan_index >= len(SCAN_X):
                self.scan_index = 0
                self.scan_station_order = self._nearest_scan_stations()
                self.scan_cycles += 1
                if self.scan_cycles >= self.max_scan_cycles:
                    if (self.target_kind == "zhijin"
                            and not DUAL_TISSUE_SIDE_ROLLED_ENABLED):
                        self.no_middle_tissue = True
                        self.get_logger().error(
                            "[tissue-filter] no middle-column tissue found "
                            f"after {self.scan_cycles} shelf scan cycles; "
                            "skipping this order")
                    self.get_logger().error(
                        f"target {self.target_kind!r} was not localised "
                        f"after {self.scan_cycles} shelf scan cycles")
                    self.set_state(STATE_ABORT)
                    return False
        if self.state != STATE_ABORT:
            self.set_state(STATE_GO_SCAN)
        return self.state != STATE_ABORT

    def drive_to(self, target_xy, final_yaw: float,
                 position_tolerance: float = 0.055,
                 linear_min_mps: float | None = None,
                 linear_gain: float = NAV_LINEAR_GAIN,
                 rotate_gate_rad: float = NAV_ROTATE_GATE_RAD,
                 translate_angular_max_rps: float = (
                     NAV_TRANSLATE_ANGULAR_MAX_RADPS)) -> bool:
        delta = np.asarray(target_xy, dtype=float) - self.base_xy
        distance = float(np.linalg.norm(delta))
        if distance > position_tolerance:
            desired_yaw = math.atan2(delta[1], delta[0])
            yaw_error = wrap_to_pi(desired_yaw - self.base_yaw)
            if abs(yaw_error) > rotate_gate_rad:
                if self._rotate_with_unstick(desired_yaw):
                    return False
                self.set_twist(0.0, float(np.clip(
                    2.2 * yaw_error, -NAV_ANGULAR_MAX_RADPS,
                    NAV_ANGULAR_MAX_RADPS)))
            else:
                linear = float(np.clip(
                    linear_gain * distance,
                    NAV_LINEAR_MIN_MPS if linear_min_mps is None
                    else linear_min_mps,
                    NAV_LINEAR_MAX_MPS))
                self.set_twist(linear, float(np.clip(
                    1.8 * yaw_error, -translate_angular_max_rps,
                    translate_angular_max_rps)))
            return False
        yaw_error = wrap_to_pi(final_yaw - self.base_yaw)
        if abs(yaw_error) > NAV_YAW_DEADBAND_RAD:
            if self._rotate_with_unstick(final_yaw):
                return False
            self.set_twist(0.0, float(np.clip(
                NAV_FINAL_YAW_GAIN * yaw_error, -NAV_ANGULAR_MAX_RADPS,
                NAV_ANGULAR_MAX_RADPS)))
            return False
        self.set_twist(0.0, 0.0)
        return True

    def _rotate_with_unstick(self, target_yaw: float) -> bool:
        """原地旋转到 target_yaw；卡死（yaw 长时间无变化）时先短距倒车解除。

        返回 True 表示本 tick 正在执行“解除卡死”的倒车动作，调用方不应再
        下发旋转速度。
        """
        now = self.now()
        target_yaw = wrap_to_pi(target_yaw)
        if (self._rot_stall_target is None
                or abs(wrap_to_pi(target_yaw - self._rot_stall_target)) > 0.05):
            self._rot_stall_target = target_yaw
            self._rot_stall_anchor_yaw = float(self.base_yaw)
            self._rot_stall_anchor_t = now
            self._rot_stall_anchor_xy = self.base_xy.copy()
            self._rot_stall_unsticks = 0
            self._rot_unstick_phase = False

        if self._rot_unstick_phase:
            # 沿当前朝向倒车一小段（保持 yaw），腾出旋转空间
            heading = np.array([
                math.cos(self.base_yaw), math.sin(self.base_yaw)])
            moved_back = float(np.dot(
                self._rot_stall_anchor_xy - self.base_xy, heading))
            yaw_err = wrap_to_pi(
                self._rot_stall_anchor_yaw - self.base_yaw)
            if (moved_back >= NAV_ROT_UNSTICK_DIST_M
                    or now - self._rot_stall_anchor_t
                    >= NAV_ROT_UNSTICK_TIMEOUT_S):
                self._rot_unstick_phase = False
                self._rot_stall_anchor_yaw = float(self.base_yaw)
                self._rot_stall_anchor_t = now
                self._rot_stall_anchor_xy = self.base_xy.copy()
                self.set_twist(0.0, 0.0)
                self.get_logger().info(
                    f"[rotate-unstick] backing finished moved="
                    f"{moved_back:.3f}m; resuming rotation")
                return False
            angular = float(np.clip(2.0 * yaw_err, -0.5, 0.5))
            self.set_twist(-NAV_ROT_UNSTICK_SPEED_MPS, angular)
            return True

        yaw_changed = abs(wrap_to_pi(
            self.base_yaw - self._rot_stall_anchor_yaw))
        moved = (
            0.0 if self._rot_stall_anchor_xy is None
            else float(np.linalg.norm(
                self.base_xy - self._rot_stall_anchor_xy)))
        if now - self._rot_stall_anchor_t >= NAV_ROT_STALL_S:
            if yaw_changed < NAV_ROT_STALL_MIN_CHANGE_RAD and moved < 0.03:
                self._rot_stall_unsticks += 1
                if self._rot_stall_unsticks >= NAV_ROT_UNSTICK_MAX:
                    # Most callers are pre-grasp.  Integrated post-delivery
                    # refinement overrides the hook so an already completed
                    # order is never converted back into a failed grasp.
                    self.set_twist(0.0, 0.0)
                    self.get_logger().error(
                        "[rotate-unstick] recovery budget exhausted after "
                        f"{NAV_ROT_UNSTICK_MAX} attempts; ending this motion")
                    self._on_rotation_recovery_exhausted()
                    return True
                self.get_logger().warn(
                    f"[rotate-unstick] rotation stalled for "
                    f"{now - self._rot_stall_anchor_t:.1f}s "
                    f"(yaw_change={yaw_changed:.3f}rad); backing up "
                    f"{NAV_ROT_UNSTICK_DIST_M:.2f}m to free the base")
                self._rot_unstick_phase = True
                self._rot_stall_anchor_xy = self.base_xy.copy()
                self._rot_stall_anchor_yaw = float(self.base_yaw)
                self._rot_stall_anchor_t = now
                return True
            # 有进展：重置计时基准
            self._rot_stall_anchor_yaw = float(self.base_yaw)
            self._rot_stall_anchor_t = now
            self._rot_stall_anchor_xy = self.base_xy.copy()
        return False

    def world_to_footprint(self, world: np.ndarray) -> np.ndarray:
        delta = np.asarray(world, dtype=float) - np.array([
            self.base_xy[0], self.base_xy[1], 0.0])
        cosine, sine = math.cos(-self.base_yaw), math.sin(-self.base_yaw)
        return np.array([
            cosine * delta[0] - sine * delta[1],
            sine * delta[0] + cosine * delta[1],
            delta[2],
        ])

    def footprint_to_world(self, footprint: np.ndarray) -> np.ndarray:
        footprint = np.asarray(footprint, dtype=float)
        cosine, sine = math.cos(self.base_yaw), math.sin(self.base_yaw)
        return np.array([
            self.base_xy[0] + cosine * footprint[0] - sine * footprint[1],
            self.base_xy[1] + sine * footprint[0] + cosine * footprint[1],
            footprint[2],
        ])

    def solve_kdl_world(
            self, world: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Solve an upright endpoint pose using the measured-base footprint."""
        target = np.eye(4)
        target[:3, 3] = self.world_to_footprint(world)
        reference = np.asarray(reference, dtype=float)
        reference_with_slide = np.concatenate((
            [self.slide_grasp], reference))
        if self.grasp_arm == "r":
            solutions = self.kdl.inverse_kinematics(
                T_right=target, target_height=self.slide_grasp,
                ref_pos=reference_with_slide)
        else:
            solutions = self.kdl.inverse_kinematics(
                T_left=target, target_height=self.slide_grasp,
                ref_pos=reference_with_slide)
        if solutions is None or len(solutions) == 0:
            raise ValueError(f"KDL IK failed for {np.round(world, 4)}")
        candidates = [np.asarray(item[1:], dtype=float) for item in solutions]
        return min(
            candidates,
            key=lambda item: float(np.max(np.abs(item - reference))))

    def selected_tcp_world(self) -> np.ndarray | None:
        """Return the selected arm's measured endpoint in the world frame."""
        slide = self.joints.get("slide_joint")
        joints = self.selected_arm_positions()
        if (slide is None or not math.isfinite(slide)
                or not np.all(np.isfinite(joints))):
            return None
        left, right = self.kdl.forward_kinematics(
            np.concatenate(([slide], joints)), index=(
                "left" if self.grasp_arm == "l" else "right"))
        transform = left if self.grasp_arm == "l" else right
        return self.footprint_to_world(transform[:3, 3])

    def arm_tcp_world(self, side: str) -> np.ndarray | None:
        """Return a measured left/right TCP for dual-arm diagnostics."""
        slide = self.joints.get("slide_joint")
        prefix = "left" if side == "left" else "right"
        joints = np.array([
            self.joints.get(f"{prefix}_arm_joint{i + 1}", float("inf"))
            for i in range(6)])
        if (slide is None or not math.isfinite(float(slide))
                or not np.all(np.isfinite(joints))):
            return None
        left, right = self.kdl.forward_kinematics(
            np.concatenate(([slide], joints)), index=side)
        transform = left if side == "left" else right
        return self.footprint_to_world(transform[:3, 3])

    def arm_target_tcp_world(
            self, side: str, joints: np.ndarray) -> np.ndarray | None:
        """Return FK TCP for a commanded arm pose at the measured slide."""
        if not hasattr(self, "kdl"):
            return None
        slide = self.joints.get("slide_joint")
        joints = np.asarray(joints, dtype=float)
        if (slide is None
                or not math.isfinite(float(slide))
                or not np.all(np.isfinite(joints))):
            return None
        left, right = self.kdl.forward_kinematics(
            np.concatenate(([slide], joints)), index=side)
        transform = left if side == "left" else right
        return self.footprint_to_world(transform[:3, 3])

    def selected_gripper_position(self) -> float | None:
        side = "left" if self.grasp_arm == "l" else "right"
        value = self.joints.get(f"{side}_arm_eef_gripper_joint")
        if value is None or not math.isfinite(float(value)):
            return None
        return float(value)

    def grasp_profile_name(self) -> str:
        """Return the composed layer/geometry profile used by this target."""
        if self.use_dual_tissue_grasp:
            suffix = (
                "_side_rolled_three_point"
                if getattr(self, "dual_side_rolled", False) else "")
            return f"{self.shelf_level}_dual_tissue{suffix}"
        if self.use_sphere_grasp:
            return f"{self.shelf_level}_sphere"
        if self.shelf_level == "top":
            return "top_front"
        if self.shelf_level in ("middle", "lower"):
            return f"{self.shelf_level}_front"
        return "unselected"

    def top_grasp_tcp_z(self) -> float:
        """Return a top TCP height with enough clearance above the shelf."""
        minimum_z = (
            TOP_SHELF_SURFACE_Z_M + TOP_MIN_TCP_TARGET_CLEARANCE_M)
        return float(max(
            self.target_world[2] + GRASP_TCP_Z_RAISE_M
            + GRASP_TCP_Z_OFFSET_BY_KIND.get(self.target_kind, 0.0),
            minimum_z))

    def solve_kdl_both_world(
            self, left_world: np.ndarray, right_world: np.ndarray,
            left_reference: np.ndarray,
            right_reference: np.ndarray,
            top_wrist_rolled: bool | None = None,
            top_wrist_inward: bool | None = None,
            left_rotation: np.ndarray | None = None,
            right_rotation: np.ndarray | None = None,
            target_height: float | None = None,
            ) -> tuple[np.ndarray, np.ndarray]:
        """Solve one symmetric two-arm pose at the current slide height."""
        left_target = np.eye(4)
        right_target = np.eye(4)
        if top_wrist_rolled is None:
            top_wrist_rolled = self.dual_top_wrist_rolled
        if top_wrist_inward is None:
            top_wrist_inward = self.dual_top_wrist_inward
        if top_wrist_rolled:
            # Mirror the wrist rolls so each gripper's narrow dimension passes
            # between the shelf upright and the box.  Rotation is about the
            # footprint-frame forward axis, so the TCP approach direction and
            # all Cartesian waypoints remain unchanged.
            #
            # 中层/下层与顶层共用同一套“窄腕横探”姿态：link6 碰撞盒的 160 mm
            # 长边在未滚转时横向布置，侧列探入会被货架前立柱挡住（实测撞柱），
            # 滚转 ±90° 后 50 mm 窄边朝横向，探入姿态才能通过立柱与纸盒之间
            # 的窄通道。
            left_roll = (-DUAL_TISSUE_TOP_WRIST_ROLL_RAD
                         if top_wrist_inward
                         else DUAL_TISSUE_TOP_WRIST_ROLL_RAD)
            left_target[:3, :3] = Rotation.from_euler(
                "x", left_roll).as_matrix()
            right_target[:3, :3] = Rotation.from_euler(
                "x", -left_roll).as_matrix()
        # The top-shelf fork keeps the left wrist in its side-bracket pose but
        # deliberately returns the right endpoint to the normal orientation,
        # turning link6's 160 mm collision box from a vertical side pad into a
        # horizontal support bar.  Explicit matrices take precedence over the
        # normal mirrored-roll policy.
        if left_rotation is not None:
            left_target[:3, :3] = np.asarray(left_rotation, dtype=float)
        if right_rotation is not None:
            right_target[:3, :3] = np.asarray(right_rotation, dtype=float)
        left_target[:3, 3] = self.world_to_footprint(left_world)
        right_target[:3, 3] = self.world_to_footprint(right_world)
        ik_height = (
            self.slide_grasp if target_height is None
            else float(target_height))
        reference = np.concatenate((
            [ik_height],
            np.asarray(left_reference, dtype=float),
            np.asarray(right_reference, dtype=float)))
        solutions = self.kdl.inverse_kinematics(
            T_left=left_target,
            T_right=right_target,
            target_height=ik_height,
            ref_pos=reference)
        if solutions is None or len(solutions) == 0:
            raise ValueError(
                "dual-arm KDL IK failed for "
                f"left={np.round(left_world, 4)} "
                f"right={np.round(right_world, 4)}")
        candidates = [
            np.asarray(item[1:], dtype=float) for item in solutions]
        arm_reference = reference[1:]
        best = min(
            candidates,
            key=lambda item: float(np.max(
                np.abs(item - arm_reference))))
        return best[:6].copy(), best[6:].copy()

    def tissue_rotate_tcp_z(self) -> float:
        """预调整阶段用的 TCP 高度，与正常纸巾夹持同一套标定。"""
        surface_z = SHELF_SURFACE_Z_M[self.shelf_level]
        target_raise = (
            DUAL_TISSUE_TOP_TCP_RAISE_M
            if self.shelf_level == "top"
            else GRASP_TCP_Z_RAISE_M)
        clearance = (
            DUAL_TISSUE_LOWER_TCP_CLEARANCE_M
            if self.shelf_level == "lower"
            else DUAL_TISSUE_TCP_CLEARANCE_M)
        return float(max(
            self.target_world[2] + target_raise,
            surface_z + clearance))

    def configure_tissue_90_rotation(self) -> bool:
        """预解算“平转 90°”三个位姿：左支点+右预备 → 右前伸 → 双退回。

        左臂先贴住纸盒左侧作为支点（只横向抵住、不前后推），右臂手侧面
        对准中心偏右的前脸位置持续前伸；单侧推力绕左支点形成 CCW 力矩，
        使长边从前向横向转为前后纵深，且不会整体向后平移。
        """
        if self.target_world is None:
            return False
        anchor_span = TISSUE_ROTATE_ANCHOR_SPAN_M
        offset = TISSUE_ROTATE_RIGHT_OFFSET_M
        z = self.tissue_rotate_tcp_z()
        target_x = float(self.target_world[0])
        target_y = float(self.target_world[1])
        anchor_left = np.array([
            target_x - anchor_span, target_y, z], dtype=float)
        pre_right = np.array([
            target_x + offset,
            target_y - DUAL_TISSUE_PREGRASP_BACKOFF_M, z], dtype=float)
        push_right = pre_right.copy()
        push_right[1] += TISSUE_ROTATE_PUSH_M
        retract_left = np.array([
            target_x - anchor_span,
            target_y - DUAL_TISSUE_PREGRASP_BACKOFF_M, z], dtype=float)
        retract_right = np.array([
            target_x + offset,
            target_y - DUAL_TISSUE_PREGRASP_BACKOFF_M, z], dtype=float)
        left_ref = self.cmd_left_arm.copy()
        right_ref = self.cmd_right_arm.copy()
        targets = {}
        try:
            anchor_joints, pre_joints = self.solve_kdl_both_world(
                anchor_left, pre_right, left_ref, right_ref,
                left_rotation=np.eye(3), right_rotation=np.eye(3))
            targets[0] = (anchor_joints.copy(), pre_joints.copy())
            push_left_joints, push_joints = self.solve_kdl_both_world(
                anchor_left, push_right,
                targets[0][0], targets[0][1],
                left_rotation=np.eye(3), right_rotation=np.eye(3))
            targets[1] = (push_left_joints.copy(), push_joints.copy())
            retract_left_joints, retract_joints = self.solve_kdl_both_world(
                retract_left, retract_right,
                targets[1][0], targets[1][1],
                left_rotation=np.eye(3), right_rotation=np.eye(3))
            targets[2] = (retract_left_joints.copy(),
                          retract_joints.copy())
        except ValueError as exc:
            self.get_logger().error(
                f"[tissue-rotate] IK failed: {exc}")
            return False
        self.tissue_rotate_targets = targets
        self.get_logger().info(
            f"[tissue-rotate] 90-deg planar rotation planned; "
            f"anchor_left={np.round(anchor_left, 4)} "
            f"right_offset={offset:.3f}m push={TISSUE_ROTATE_PUSH_M:.3f}m "
            f"pre_right={np.round(pre_right, 4)} "
            f"push_right={np.round(push_right, 4)}")
        return True

    def start_tissue_rotate_stage(self, stage: int) -> None:
        """开始平转 90° 的一个段：左支点固定，右臂负责推转。"""
        labels = {
            0: "rotate_anchor_pre",
            1: "rotate_push",
            2: "rotate_retract",
        }
        paths = {
            0: DUAL_TISSUE_PREGRASP_BACKOFF_M
            + max(TISSUE_ROTATE_ANCHOR_SPAN_M,
                  TISSUE_ROTATE_RIGHT_OFFSET_M),
            1: DUAL_TISSUE_PREGRASP_BACKOFF_M + TISSUE_ROTATE_PUSH_M,
            2: DUAL_TISSUE_PREGRASP_BACKOFF_M + TISSUE_ROTATE_PUSH_M,
        }
        speeds = {
            0: DUAL_TISSUE_FORWARD_SPEED_MPS,
            1: TISSUE_ROTATE_SPEED_MPS,
            2: DUAL_TISSUE_RETREAT_SPEED_MPS,
        }
        left_target, right_target = self.tissue_rotate_targets[stage]
        self.tissue_rotate_stage = stage
        self.des_slide = self.slide_grasp
        self.start_dual_tissue_motion(
            labels[stage], left_target, right_target,
            paths[stage], speeds[stage],
            STATE_TISSUE_ROTATE,
            require_convergence=(stage != 1))

    def _prepare_tissue_rotation_if_needed(self) -> bool:
        """纸巾在锁定后先执行平面 90° 预旋转；失败则回退到原抓取流程。"""
        if not TISSUE_ROTATE_ENABLED:
            return False
        if not self.use_dual_tissue_grasp or self.tissue_rotated_90:
            return False
        # 中间列（任意层）改为“宽间距同步伸入 → 合拢夹持 → 抬起 → 取走”
        # 的双臂动作，不再先做 90° 平面推转，避免预旋转改变纸盒朝向。
        slot = self.target_slot()
        if slot is not None and slot[2] == "2":
            self.get_logger().info(
                "[tissue-rotate] middle-column direct close flow "
                "selected; skipping 90-deg pre-rotation")
            return False
        if (slot is not None and slot[2] in ("1", "3")
                and DUAL_TISSUE_SIDE_ROLLED_ENABLED):
            self.get_logger().info(
                "[tissue-rotate] side-column narrow-wrist flow selected; "
                "keeping the box square and skipping product rotation")
            return False
        if self.configure_tissue_90_rotation():
            self.tissue_rotate_stage = 0
            self.start_tissue_rotate_stage(0)
            if self.shelf_level in ("top", "middle", "lower"):
                self.begin_manip_base_hold()
            return True
        self.get_logger().warn(
            "[tissue-rotate] 90-deg rotation IK unavailable; "
            "falling back to direct dual-arm grasp")
        return False

    def configure_dual_tissue_grasp(self) -> bool:
        """Prepare a symmetric side clamp for the tissue box at any level.

        Middle columns retain the established hand-side direct clamp.  Side
        columns pre-roll both wrists outside the shelf and retain that narrow
        orientation through insertion, contact, lift and retreat.
        """
        surface_z = SHELF_SURFACE_Z_M[self.shelf_level]
        rotated = bool(getattr(self, "tissue_rotated_90", False))
        slot = self.target_slot()
        column = slot[2] if slot is not None else "2"
        self.dual_side_rolled = bool(
            DUAL_TISSUE_SIDE_ROLLED_ENABLED
            and column in ("1", "3")
            and not rotated)
        self.dual_clamp_half_span = (
            TISSUE_ROTATED_CLAMP_HALF_SPAN_M
            if rotated else DUAL_TISSUE_CLAMP_HALF_SPAN_M)
        # 侧滚腕路径加深探入（抓盒中段），中列手侧面保持原 5mm。
        self.dual_insert_forward_m = (
            DUAL_TISSUE_SIDE_ROLLED_INSERT_FORWARD_M
            if self.dual_side_rolled
            else DUAL_TISSUE_INSERT_FORWARD_M)
        self.dual_top_wrist_rolled = self.dual_side_rolled
        self.dual_top_wrist_inward = False
        self.dual_contact_push_side = (
            "right" if column == "3" else "left")
        self.dual_overhead_route = False
        self.dual_middle_extend_close = (
            column == "2" and not rotated)
        self.dual_direct_probe = not self.dual_middle_extend_close
        if getattr(self, "dual_middle_extend_close", False):
            # 中间列采用“宽间距同步前伸 → 再合拢夹持”的动作：surround
            # 阶段保持 DUAL_TISSUE_PREGRASP_HALF_SPAN_M 宽半跨度，确保
            # 闭合前双臂间距足够，随后再合拢到夹持半跨度。
            self.dual_probe_span_l = DUAL_TISSUE_PREGRASP_HALF_SPAN_M
            self.dual_probe_span_r = DUAL_TISSUE_PREGRASP_HALF_SPAN_M
        elif rotated:
            self.dual_probe_span_l = TISSUE_ROTATED_PROBE_HALF_SPAN_M
            self.dual_probe_span_r = TISSUE_ROTATED_PROBE_HALF_SPAN_M
        elif column == "1":
            self.dual_probe_span_l = DUAL_TISSUE_POST_SIDE_SPAN_M
            self.dual_probe_span_r = DUAL_TISSUE_NEIGHBOUR_PROBE_SPAN_M
        elif column == "3":
            self.dual_probe_span_l = DUAL_TISSUE_NEIGHBOUR_PROBE_SPAN_M
            self.dual_probe_span_r = DUAL_TISSUE_POST_SIDE_SPAN_M
        else:
            self.dual_probe_span_l = DUAL_TISSUE_DIRECT_PROBE_SPAN_M
            self.dual_probe_span_r = DUAL_TISSUE_DIRECT_PROBE_SPAN_M
        self.dual_surround_half_span = max(
            self.dual_probe_span_l, self.dual_probe_span_r)
        if rotated:
            self.dual_pregrasp_half_span = TISSUE_ROTATED_PROBE_HALF_SPAN_M
        elif self.dual_middle_extend_close:
            self.dual_pregrasp_half_span = DUAL_TISSUE_PREGRASP_HALF_SPAN_M
        else:
            self.dual_pregrasp_half_span = self.dual_surround_half_span
        self.dual_squeeze_m = (
            DUAL_TISSUE_SIDE_ROLLED_SQUEEZE_M
            if self.dual_side_rolled else DUAL_TISSUE_SQUEEZE_M)
        tcp_clearance = (
            DUAL_TISSUE_SIDE_ROLLED_TCP_CLEARANCE_M
            if self.dual_side_rolled
            else (DUAL_TISSUE_LOWER_TCP_CLEARANCE_M
                  if self.shelf_level == "lower"
                  else DUAL_TISSUE_TCP_CLEARANCE_M))
        target_raise = (
            DUAL_TISSUE_TOP_TCP_RAISE_M
            if self.shelf_level == "top"
            else GRASP_TCP_Z_RAISE_M)
        tcp_z = max(
            float(self.target_world[2] + target_raise),
            surface_z + tcp_clearance)
        self.dual_contact_tcp_z = float(tcp_z)
        insert_y = (
            self.target_world[1] + self.dual_insert_forward_m)

        def pair(span_l: float, span_r: float, y: float):
            left = np.array([
                self.target_world[0] - span_l, y, tcp_z], dtype=float)
            right = np.array([
                self.target_world[0] + span_r, y, tcp_z], dtype=float)
            return left, right

        if rotated:
            probe_span_l = TISSUE_ROTATED_PROBE_HALF_SPAN_M
            probe_span_r = TISSUE_ROTATED_PROBE_HALF_SPAN_M
        else:
            probe_span_l = self.dual_probe_span_l
            probe_span_r = self.dual_probe_span_r
        pre_left, pre_right = pair(
            probe_span_l, probe_span_r,
            self.target_world[1] - DUAL_TISSUE_PREGRASP_BACKOFF_M)
        surround_left, surround_right = pair(
            probe_span_l, probe_span_r, insert_y)
        clamp_left, clamp_right = pair(
            self.dual_clamp_half_span, self.dual_clamp_half_span, insert_y)
        retreat_left, retreat_right = pair(
            self.dual_clamp_half_span, self.dual_clamp_half_span,
            self.target_world[1] - DUAL_TISSUE_PREGRASP_BACKOFF_M)
        left_reference = self.cmd_left_arm.copy()
        right_reference = self.cmd_right_arm.copy()
        try:
            pre_left_joints, pre_right_joints = self.solve_kdl_both_world(
                pre_left, pre_right, left_reference, right_reference)
            if self.dual_direct_probe:
                surround_left, surround_right = pair(
                    probe_span_l, probe_span_r, insert_y)
                surround_left_joints, surround_right_joints = (
                    self.solve_kdl_both_world(
                        surround_left, surround_right,
                        pre_left_joints, pre_right_joints))
                if self.dual_side_rolled:
                    segment_delta = max(
                        float(np.max(np.abs(
                            surround_left_joints - pre_left_joints))),
                        float(np.max(np.abs(
                            surround_right_joints - pre_right_joints))))
                    if (segment_delta
                            > DUAL_TISSUE_SIDE_ROLLED_MAX_SEGMENT_JOINT_DELTA_RAD):
                        raise ValueError(
                            "side-column rolled approach changed one joint "
                            f"by {segment_delta:.3f}rad")
                self.dual_surround_pass_left_joints = None
                self.dual_surround_pass_right_joints = None
                self.dual_surround_unroll_left_joints = None
                self.dual_surround_unroll_right_joints = None
            else:
                surround_left_joints, surround_right_joints = (
                    self.solve_kdl_both_world(
                        surround_left, surround_right,
                        pre_left_joints, pre_right_joints))
            if self.dual_direct_probe:
                self.dual_surround_close_left_joints = None
                self.dual_surround_close_right_joints = None
                clamp_left_joints = None
                clamp_right_joints = None
                retreat_left_joints = None
                retreat_right_joints = None
            else:
                # 夹持/撤退保持手侧面（0°）姿态解算。
                clamp_left_joints, clamp_right_joints = (
                    self.solve_kdl_both_world(
                        clamp_left, clamp_right,
                        surround_left_joints, surround_right_joints))
                retreat_left_joints, retreat_right_joints = (
                    self.solve_kdl_both_world(
                        retreat_left, retreat_right,
                        clamp_left_joints, clamp_right_joints))
                if self.dual_middle_extend_close:
                    # 中间列保留独立的“闭合前”宽间距：surround 阶段只
                    # 同步前伸，保持宽半跨度；随后 surround_close 再合拢
                    # 到夹持跨度，避免闭合前双臂间距过小导致夹持不稳。
                    self.dual_surround_close_left_joints = (
                        clamp_left_joints.copy())
                    self.dual_surround_close_right_joints = (
                        clamp_right_joints.copy())
        except ValueError as exc:
            self.get_logger().error(f"[dual-tissue-IK] {exc}")
            if self.ik_retry_forward_m < DUAL_IK_RETRY_MAX_M:
                step = min(
                    DUAL_IK_RETRY_STEP_M,
                    DUAL_IK_RETRY_MAX_M - self.ik_retry_forward_m)
                self.ik_retry_forward_m += step
                self.align_base_y = float(np.clip(
                    self.align_base_y - step,
                    DUAL_ALIGN_Y_MIN_M, DUAL_ALIGN_Y_MAX_M))
                self.get_logger().warn(
                    f"[IK-retry] dual-tissue IK failed; backing base up "
                    f"{self.ik_retry_forward_m:.3f}m -> align_y="
                    f"{self.align_base_y:.3f}")
                return "retry"
            return False

        self.dual_surround_pass_left_joints = None
        self.dual_surround_pass_right_joints = None
        self.dual_surround_unroll_left_joints = None
        self.dual_surround_unroll_right_joints = None
        self.dual_pregrasp_left_joints = pre_left_joints
        self.dual_pregrasp_right_joints = pre_right_joints
        self.dual_surround_left_joints = surround_left_joints
        self.dual_surround_right_joints = surround_right_joints
        # Side columns retain their rolled wrist orientation; pass/unroll and
        # forward/return belong only to the disabled legacy post route.
        self.dual_surround_forward_left_joints = None
        self.dual_surround_forward_right_joints = None
        self.dual_surround_return_left_joints = None
        self.dual_surround_return_right_joints = None
        self.dual_surround_stage = 0
        self.dual_top_extract_stage = 0
        self.dual_top_fork_targets = {}
        # Direct probing fills these from measured bilateral contacts in
        # start_dual_tissue_squeeze(), before STATE_CLOSE can use them.
        self.dual_clamp_left_joints = clamp_left_joints
        self.dual_clamp_right_joints = clamp_right_joints
        self.dual_retreat_left_joints = retreat_left_joints
        self.dual_retreat_right_joints = retreat_right_joints
        self.des_left_arm = pre_left_joints.copy()
        self.des_right_arm = pre_right_joints.copy()
        self.des_left_grip = DUAL_TISSUE_GRIP_COMMAND
        self.des_right_grip = DUAL_TISSUE_GRIP_COMMAND
        self.des_slide = self.slide_grasp
        self.get_logger().info(
            f"[dual-tissue-IK] tcp_z={tcp_z:.3f} "
            f"raise={tcp_z - self.target_world[2]:.3f}m "
            f"pre_left={np.round(pre_left, 3)} "
            f"pre_right={np.round(pre_right, 3)} "
            f"surround_left={np.round(surround_left, 3)} "
            f"surround_right={np.round(surround_right, 3)} "
            f"clamp_left={np.round(clamp_left, 3)} "
            f"clamp_right={np.round(clamp_right, 3)} "
            f"retreat_left={np.round(retreat_left, 3)} "
            f"retreat_right={np.round(retreat_right, 3)} "
            f"grippers=closed({DUAL_TISSUE_GRIP_COMMAND:.2f}) "
            f"side_rolled={int(self.dual_side_rolled)} "
            f"contact_push={self.dual_contact_push_side}")
        if self.dual_direct_probe:
            self.get_logger().info(
                "[dual-tissue-IK] direct probe defers clamp/retreat IK "
                "until measured bilateral contact")
        return True

    def configure_dual_tissue_arm_lift(self) -> bool:
        """Lift via the arm joints when the slide has no upward headroom.

        On the top shelf the slide is already pinned at SLIDE_MIN to reach the
        shelf height, so the slide-based lift is a no-op and the retreat would
        drag the box across the board.  Build short, branch-continuous vertical
        and horizontal waypoints.  This prevents a redundant wrist solution
        from sweeping one loaded arm through the tissue or the opposite arm.
        """
        left_tcp = self.arm_tcp_world("left")
        right_tcp = self.arm_tcp_world("right")
        if left_tcp is None or right_tcp is None:
            self.get_logger().error(
                "[dual-tissue-arm-lift] measured TCP unavailable")
            return False
        start_z = 0.5 * (left_tcp[2] + right_tcp[2])
        inward_preload = (
            DUAL_TISSUE_TOP_MIDDLE_LIFT_INWARD_PRELOAD_M
            if getattr(self, "dual_middle_extend_close", False) else 0.0)
        # Keep the commanded squeeze preload while lifting.  Anchoring the
        # next IK samples only to the load-displaced measured TCPs relaxes the
        # clamp for one frame and can release the box.
        clamp_left_tcp = (
            None if self.dual_clamp_left_joints is None
            else self.arm_target_tcp_world(
                "left", self.dual_clamp_left_joints))
        clamp_right_tcp = (
            None if self.dual_clamp_right_joints is None
            else self.arm_target_tcp_world(
                "right", self.dual_clamp_right_joints))
        hold_left_x = float(
            (left_tcp[0] if clamp_left_tcp is None else clamp_left_tcp[0])
            + inward_preload)
        hold_right_x = float(
            (right_tcp[0] if clamp_right_tcp is None else clamp_right_tcp[0])
            - inward_preload)
        retreat_y = (
            self.target_world[1] - DUAL_TISSUE_PREGRASP_BACKOFF_M)
        left_reference = self.arm_positions("left")
        right_reference = self.arm_positions("right")
        self.dual_lift_arm_waypoints = []
        self.dual_lift_retreat_waypoints = []
        achieved_lift = 0.0

        def solve_guarded(
                left_world: np.ndarray, right_world: np.ndarray,
                left_ref: np.ndarray, right_ref: np.ndarray,
                label: str) -> tuple[np.ndarray, np.ndarray]:
            left_joints, right_joints = self.solve_kdl_both_world(
                left_world, right_world, left_ref, right_ref)
            left_delta = float(np.max(np.abs(left_joints - left_ref)))
            right_delta = float(np.max(np.abs(right_joints - right_ref)))
            largest_delta = max(left_delta, right_delta)
            if largest_delta > DUAL_TISSUE_ARM_SEGMENT_MAX_JOINT_DELTA_RAD:
                raise ValueError(
                    f"{label} selected a discontinuous IK branch: "
                    f"left_delta={left_delta:.3f}rad "
                    f"right_delta={right_delta:.3f}rad limit="
                    f"{DUAL_TISSUE_ARM_SEGMENT_MAX_JOINT_DELTA_RAD:.3f}rad")
            return left_joints, right_joints

        # Plan upward from the measured clamped pose.  If only the highest
        # waypoint changes branch, retaining a lower but >=25 mm lift is safer
        # and faster than aborting or dragging the box at shelf height.
        lift_steps = int(math.ceil(
            DUAL_TISSUE_TOP_ARM_LIFT_M
            / DUAL_TISSUE_ARM_LIFT_STEP_M))
        for index in range(1, lift_steps + 1):
            amount = min(
                DUAL_TISSUE_TOP_ARM_LIFT_M,
                index * DUAL_TISSUE_ARM_LIFT_STEP_M)
            lift_z = start_z + amount
            lift_left = np.array([hold_left_x, left_tcp[1], lift_z])
            lift_right = np.array([hold_right_x, right_tcp[1], lift_z])
            try:
                left_joints, right_joints = solve_guarded(
                    lift_left, lift_right,
                    left_reference, right_reference,
                    f"lift waypoint {index}/{lift_steps}")
            except ValueError as exc:
                if achieved_lift < DUAL_TISSUE_ARM_LIFT_MIN_CLEARANCE_M:
                    self.get_logger().error(
                        f"[dual-tissue-arm-lift] no safe lift clearance: "
                        f"{exc}; achieved={achieved_lift:.3f}m")
                    self.dual_lift_arm_waypoints = []
                    return False
                self.get_logger().warn(
                    f"[dual-tissue-arm-lift] high waypoint rejected: "
                    f"{exc}; using verified {achieved_lift:.3f}m lift")
                break
            self.dual_lift_arm_waypoints.append((
                amount, left_joints.copy(), right_joints.copy()))
            left_reference = left_joints
            right_reference = right_joints
            achieved_lift = amount

        if achieved_lift < DUAL_TISSUE_ARM_LIFT_MIN_CLEARANCE_M:
            self.get_logger().error(
                f"[dual-tissue-arm-lift] planned lift {achieved_lift:.3f}m "
                f"is below safe clearance "
                f"{DUAL_TISSUE_ARM_LIFT_MIN_CLEARANCE_M:.3f}m")
            self.dual_lift_arm_waypoints = []
            return False

        # Pre-plan the complete raised retreat before moving either loaded arm.
        # If any segment has no continuous IK solution the robot remains in the
        # symmetric clamp pose and enters the existing safe abort path.
        retreat_distance = abs(float(left_tcp[1] - retreat_y))
        retreat_steps = max(1, int(math.ceil(
            retreat_distance / DUAL_TISSUE_ARM_RETREAT_STEP_M)))
        lift_z = start_z + achieved_lift
        try:
            for index in range(1, retreat_steps + 1):
                progress = index / retreat_steps
                waypoint_y = float(
                    left_tcp[1] + progress * (retreat_y - left_tcp[1]))
                retreat_left = np.array([
                    hold_left_x, waypoint_y, lift_z])
                retreat_right = np.array([
                    hold_right_x, waypoint_y, lift_z])
                left_joints, right_joints = solve_guarded(
                    retreat_left, retreat_right,
                    left_reference, right_reference,
                    f"retreat waypoint {index}/{retreat_steps}")
                self.dual_lift_retreat_waypoints.append((
                    abs(float(left_tcp[1] - waypoint_y)),
                    left_joints.copy(), right_joints.copy()))
                left_reference = left_joints
                right_reference = right_joints
        except ValueError as exc:
            self.get_logger().error(
                f"[dual-tissue-arm-lift] raised retreat is unsafe: {exc}")
            self.dual_lift_arm_waypoints = []
            self.dual_lift_retreat_waypoints = []
            return False

        self.dual_lift_arm_stage = 0
        self.dual_lift_retreat_stage = 0
        self.dual_lift_arm_achieved_m = achieved_lift
        self.dual_lift_left_joints = (
            self.dual_lift_arm_waypoints[-1][1].copy())
        self.dual_lift_right_joints = (
            self.dual_lift_arm_waypoints[-1][2].copy())
        self.dual_lift_retreat_left_joints = (
            self.dual_lift_retreat_waypoints[-1][1].copy())
        self.dual_lift_retreat_right_joints = (
            self.dual_lift_retreat_waypoints[-1][2].copy())
        self.get_logger().info(
            f"[dual-tissue-arm-lift] slide pinned at SLIDE_MIN; "
            f"planned_lift={achieved_lift:.3f}m/"
            f"{DUAL_TISSUE_TOP_ARM_LIFT_M:.3f}m via "
            f"{len(self.dual_lift_arm_waypoints)} guarded arm segments; "
            f"retreat={retreat_distance:.3f}m via "
            f"{len(self.dual_lift_retreat_waypoints)} guarded segments "
            f"left_start={np.round(left_tcp, 4)} "
            f"right_start={np.round(right_tcp, 4)} "
            f"loaded_hold_x=({hold_left_x:.4f},{hold_right_x:.4f}) "
            f"retreat_y={retreat_y:.3f}")
        return True

    def start_dual_tissue_arm_lift_stage(self, stage: int) -> None:
        """Play all guarded vertical waypoints without stop/start cycling."""
        self.dual_lift_arm_stage = len(self.dual_lift_arm_waypoints) - 1
        self.start_dual_tissue_waypoint_motion(
            "arm_lift_continuous",
            self.dual_lift_arm_waypoints,
            DUAL_TISSUE_ARM_LIFT_SPEED_MPS,
            STATE_LIFT,
            require_convergence=True)

    def start_dual_tissue_arm_retreat_stage(self, stage: int) -> None:
        """Immediately play the complete raised retreat as one path."""
        self.dual_lift_retreat_stage = (
            len(self.dual_lift_retreat_waypoints) - 1)
        self.start_dual_tissue_waypoint_motion(
            "raised_retreat_continuous",
            self.dual_lift_retreat_waypoints,
            DUAL_TISSUE_TOP_ARM_RETREAT_SPEED_MPS,
            STATE_RETREAT,
            require_convergence=True)

    def configure_dual_tissue_top_fork(self) -> bool:
        """Build a right-wrist fork below a shelf-supported front overhang.

        The left rolled wrist remains against the tissue side as a guide.  The
        right wrist releases laterally, moves in front of the shelf, descends
        below the board surface, and returns to its normal orientation.  In
        that orientation link6's long collision box is horizontal and reaches
        under the overhang.  A small upward preload seats the box on this bar
        before the coordinated lift.
        """
        left_tcp = self.arm_tcp_world("left")
        right_tcp = self.arm_tcp_world("right")
        if left_tcp is None or right_tcp is None:
            self.get_logger().error(
                "[dual-tissue-top-fork] measured TCP unavailable")
            return False

        left_rotation = Rotation.from_euler(
            "x", -DUAL_TISSUE_TOP_WRIST_ROLL_RAD).as_matrix()
        # IK rotations are expressed in the robot footprint frame, not the
        # world frame.  With the base facing the shelf, a +90 degree endpoint
        # yaw turns link6's 160 mm bar along shelf depth.  Its rear tip then
        # reaches beneath the exposed tissue lip while its front-to-back span
        # remains outside the static shelf board.
        right_rotation = Rotation.from_euler(
            "z", math.pi / 2.0).as_matrix()
        front_y = (
            self.target_world[1]
            - DUAL_TISSUE_TOP_FORK_FRONT_BACKOFF_M)
        low_z = (
            SHELF_SURFACE_Z_M["top"]
            - DUAL_TISSUE_TOP_FORK_TCP_BELOW_SURFACE_M)

        left_hold = np.asarray(left_tcp, dtype=float).copy()
        release_right = np.asarray(right_tcp, dtype=float).copy()
        release_right[0] += DUAL_TISSUE_TOP_FORK_RELEASE_M
        front_right = release_right.copy()
        front_right[1] = front_y
        low_right = front_right.copy()
        low_right[2] = low_z
        # After the +90 degree yaw, the bar is narrow laterally and its centre
        # sits about 70 mm to the right of the TCP.  Compensate that full
        # offset so the bar, rather than the TCP, is centred below the tissue.
        # The former midpoint TCP put the physical bar at the right-front
        # corner and the preload pushed the box diagonally across the shelf.
        # This inner pose is reachable only after the wrist is below the shelf
        # surface and has changed orientation.
        centre_right = low_right.copy()
        centre_right[0] = (
            0.5 * (left_hold[0] + right_tcp[0])
            - DUAL_TISSUE_TOP_FORK_BAR_LATERAL_OFFSET_M)
        preload_right = centre_right.copy()
        preload_right[2] += DUAL_TISSUE_TOP_FORK_PRELOAD_M
        # The exposed lip is forward of the tissue centre of mass.  Lifting
        # directly from there makes the box roll backward and lose the fork.
        # First raise the fixed wrists enough to seat the front lip.  Then
        # route the left wrist behind the tissue and push it forward over the
        # stationary horizontal bar until the support is approximately below
        # its centre of mass.  Only then perform the common-slide lift.
        seat_left = left_hold.copy()
        seat_left[2] += DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M
        seat_right = preload_right.copy()
        seat_right[2] += DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M
        pusher_high_outer = seat_left.copy()
        pusher_high_outer[0] -= DUAL_TISSUE_TOP_FORK_PUSHER_RELEASE_M
        pusher_high_outer[1] = (
            self.target_world[1]
            + DUAL_TISSUE_TOP_FORK_PUSHER_REAR_CLEARANCE_M)
        pusher_high_outer[2] += DUAL_TISSUE_TOP_FORK_PUSHER_OVERHEAD_M
        pusher_high_centre = pusher_high_outer.copy()
        pusher_high_centre[0] = (
            self.target_world[0]
            + DUAL_TISSUE_TOP_FORK_PUSHER_X_LEAD_M)
        pusher_behind = pusher_high_centre.copy()
        pusher_behind[2] = (
            seat_left[2] - DUAL_TISSUE_TOP_FORK_PUSHER_LOWER_M)
        pusher_push = pusher_behind.copy()
        pusher_push[1] -= DUAL_TISSUE_TOP_FORK_PUSH_M

        targets = {}
        left_ref = self.arm_positions("left")
        right_ref = self.arm_positions("right")

        def solve(stage, left_world, right_world, *, right_unrolled=False,
                  support_slide=False, target_height=None):
            nonlocal left_ref, right_ref
            if target_height is None:
                target_height = (
                    DUAL_TISSUE_TOP_FORK_SLIDE_M
                    if support_slide else self.slide_grasp)
            left_ref, right_ref = self.solve_kdl_both_world(
                left_world, right_world, left_ref, right_ref,
                top_wrist_rolled=True,
                top_wrist_inward=True,
                left_rotation=(left_rotation if right_unrolled else None),
                right_rotation=(right_rotation if right_unrolled else None),
                target_height=target_height)
            targets[stage] = (left_ref.copy(), right_ref.copy())

        try:
            solve(2, left_hold, release_right)
            solve(3, left_hold, front_right)
            solve(4, left_hold, low_right, support_slide=True)
            solve(5, left_hold, low_right, right_unrolled=True,
                  support_slide=True)
            solve(6, left_hold, centre_right, right_unrolled=True,
                  support_slide=True)
            solve(7, left_hold, preload_right, right_unrolled=True,
                  support_slide=True)
            # Seat with fixed stage-7 joints and a short common-slide lift.
            targets[8] = tuple(item.copy() for item in targets[7])
            # Solve each left-pusher pose, but always restore the exact right
            # support joints afterwards.  Re-solving an unchanged right TCP
            # selected a redundant posture that shifted the physical link6
            # bar 46 mm sideways even though endpoint error stayed small.
            solve(
                9, pusher_high_outer, seat_right, right_unrolled=True,
                target_height=(
                    DUAL_TISSUE_TOP_FORK_SLIDE_M
                    - DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M))
            right_ref = targets[8][1].copy()
            targets[9] = (targets[9][0], right_ref.copy())
            solve(
                10, pusher_high_centre, seat_right, right_unrolled=True,
                target_height=(
                    DUAL_TISSUE_TOP_FORK_SLIDE_M
                    - DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M))
            right_ref = targets[8][1].copy()
            targets[10] = (targets[10][0], right_ref.copy())
            solve(
                11, pusher_behind, seat_right, right_unrolled=True,
                target_height=(
                    DUAL_TISSUE_TOP_FORK_SLIDE_M
                    - DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M))
            right_ref = targets[8][1].copy()
            targets[11] = (targets[11][0], right_ref.copy())
            solve(
                12, pusher_push, seat_right, right_unrolled=True,
                target_height=(
                    DUAL_TISSUE_TOP_FORK_SLIDE_M
                    - DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M))
            right_ref = targets[8][1].copy()
            targets[12] = (targets[12][0], right_ref.copy())
            # Keep the exact converged pusher/support joints during the full
            # lift.  A separately solved +Z endpoint can jump to another IK
            # branch; that previously rotated joint 4 by 1.69 rad and swept
            # the bar back into the shelf.  The remaining lift is performed
            # by the common slide, so stages 13 and 14 need no new arm IK.
            targets[13] = tuple(item.copy() for item in targets[12])
            targets[14] = tuple(item.copy() for item in targets[12])
        except ValueError as exc:
            self.get_logger().error(
                f"[dual-tissue-top-fork] IK failed: {exc}")
            return False

        self.dual_top_fork_targets = targets
        self.get_logger().info(
            "[dual-tissue-top-fork] support trajectory solved; "
            f"left_hold={np.round(left_hold, 4)} "
            f"right_start={np.round(right_tcp, 4)} "
            f"right_front={np.round(front_right, 4)} "
            f"right_low={np.round(low_right, 4)} "
            f"right_centre={np.round(centre_right, 4)} "
            f"right_preload={np.round(preload_right, 4)} "
            f"left_pusher_high={np.round(pusher_high_centre, 4)} "
            f"left_pusher_behind={np.round(pusher_behind, 4)} "
            f"left_pusher_push={np.round(pusher_push, 4)}")
        return True

    def start_dual_tissue_top_fork_stage(self, stage: int) -> None:
        """Start one pre-solved segment of the top-shelf fork sequence."""
        labels = {
            2: "top_fork_release_right",
            3: "top_fork_move_front",
            4: "top_fork_lower_right",
            5: "top_fork_unroll_right",
            6: "top_fork_centre_right",
            7: "top_fork_preload",
            8: "top_fork_seat",
            9: "top_fork_raise_pusher_behind",
            10: "top_fork_centre_pusher_overhead",
            11: "top_fork_lower_rear_pusher",
            12: "top_fork_push_over_support",
            13: "top_fork_lift",
            14: "top_fork_retreat",
        }
        paths = {
            2: DUAL_TISSUE_TOP_FORK_RELEASE_M,
            3: (DUAL_TISSUE_TOP_FORK_FRONT_BACKOFF_M
                - DUAL_TISSUE_TOP_EDGE_BACKOFF_M),
            4: 0.170,
            5: 0.080,
            6: DUAL_TISSUE_TOP_FORK_CENTRE_M,
            7: DUAL_TISSUE_TOP_FORK_PRELOAD_M,
            8: DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M,
            9: 0.285,
            10: 0.220,
            11: (DUAL_TISSUE_TOP_FORK_PUSHER_OVERHEAD_M
                 + DUAL_TISSUE_TOP_FORK_PUSHER_LOWER_M),
            12: DUAL_TISSUE_TOP_FORK_PUSH_M,
            13: DUAL_TISSUE_TOP_FORK_LIFT_M,
            # Keep the support pose long enough for the chassis to carry the
            # held box completely beyond the shelf edge.
            14: 0.075,
        }
        speeds = {
            2: DUAL_TISSUE_RETREAT_SPEED_MPS,
            3: DUAL_TISSUE_RETREAT_SPEED_MPS,
            4: DUAL_TISSUE_FORWARD_SPEED_MPS,
            5: 0.020,
            6: DUAL_TISSUE_RETREAT_SPEED_MPS,
            7: 0.010,
            8: DUAL_TISSUE_ARM_LIFT_SPEED_MPS,
            9: 0.030,
            10: 0.030,
            11: DUAL_TISSUE_RETREAT_SPEED_MPS,
            12: 0.010,
            13: DUAL_TISSUE_ARM_LIFT_SPEED_MPS,
            14: DUAL_TISSUE_RETREAT_SPEED_MPS,
        }
        left_target, right_target = self.dual_top_fork_targets[stage]
        self.dual_top_extract_stage = stage
        if 4 <= stage <= 7:
            self.des_slide = DUAL_TISSUE_TOP_FORK_SLIDE_M
        if stage in (8, 13):
            # Raise both fixed wrists with the common slide.  Holding the
            # support joints preserves the horizontal support-bar attitude
            # and eliminates the IK branch flip seen in simulation.
            self.dual_lift_use_arm = False
            self.dual_lift_settled_since = None
            self.des_left_arm = left_target.copy()
            self.des_right_arm = right_target.copy()
            lift_amount = (
                DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M
                if stage == 8 else DUAL_TISSUE_LIFT_M)
            target_slide = max(
                SLIDE_MIN, DUAL_TISSUE_TOP_FORK_SLIDE_M - lift_amount)
            self.des_slide = target_slide
            self.get_logger().info(
                f"[dual-tissue-{labels[stage]}] fixed support joints; "
                f"raising with slide from "
                f"{DUAL_TISSUE_TOP_FORK_SLIDE_M:.3f}m to "
                f"{target_slide:.3f}m")
            self.set_state(STATE_LIFT)
            return
        if stage in (9, 10, 11, 12):
            self.des_slide = max(
                SLIDE_MIN,
                DUAL_TISSUE_TOP_FORK_SLIDE_M
                - DUAL_TISSUE_TOP_FORK_SEAT_LIFT_M)
        if stage == 14:
            self.des_slide = max(
                SLIDE_MIN,
                DUAL_TISSUE_TOP_FORK_SLIDE_M - DUAL_TISSUE_LIFT_M)
        self.start_dual_tissue_motion(
            labels[stage], left_target, right_target,
            paths[stage], speeds[stage],
            STATE_LIFT if stage == 8 else STATE_RETREAT,
            # During the rolled descent the physical right wrist settles
            # about 37 mm inward because joint 5 is near its contact-limited
            # pose.  Its measured Y/Z already put it safely below the exposed
            # lip; stage 5 immediately unrolls that wrist and remains gated.
            # Do not abort the whole grasp on the harmless stage-4 X/orient-
            # ation residual.
            require_convergence=(stage in (5, 6, 7, 9, 10, 11, 12)))

    def start_dual_tissue_slide_lift(self) -> None:
        """Raise a middle-shelf tissue before any horizontal retreat."""
        self.dual_lift_use_arm = False
        self.dual_lift_settled_since = None
        target_slide = max(
            SLIDE_MIN, self.slide_grasp - DUAL_TISSUE_LIFT_M)
        self.des_slide = target_slide
        self.get_logger().info(
            f"[dual-tissue-slide-lift] raising held middle-shelf tissue "
            f"{self.slide_grasp - target_slide:.3f}m before retreat; "
            f"start_slide={self.slide_grasp:.3f} "
            f"target_slide={target_slide:.3f}")
        self.set_state(STATE_LIFT)

    def advance_dual_tissue_slide_lift(self) -> str:
        """Wait for a stable slide lift; never retreat on an unraised box."""
        now = self.now()
        lift_elapsed = now - self.state_t0
        target_slide = max(
            SLIDE_MIN, self.slide_grasp - DUAL_TISSUE_LIFT_M)
        self.des_slide = target_slide
        measured_slide = self.joints.get("slide_joint")
        reached = (
            measured_slide is not None
            and abs(measured_slide - target_slide)
            <= DUAL_TISSUE_SLIDE_LIFT_TOLERANCE_M)
        if reached:
            if self.dual_lift_settled_since is None:
                self.dual_lift_settled_since = now
            elif (now - self.dual_lift_settled_since
                  >= DUAL_TISSUE_SLIDE_LIFT_STABLE_S):
                self.get_logger().info(
                    f"[dual-tissue-slide-lift] upward motion settled; "
                    f"target_slide={target_slide:.3f} "
                    f"measured_slide={measured_slide:.3f}; "
                    "starting raised retreat")
                return "reached"
        else:
            self.dual_lift_settled_since = None

        if lift_elapsed >= DUAL_TISSUE_SLIDE_LIFT_TIMEOUT_S:
            self.get_logger().error(
                f"[dual-tissue-slide-lift] upward motion did not converge; "
                f"target_slide={target_slide:.3f} "
                f"measured_slide={measured_slide}; "
                "aborting instead of dragging the tissue across the shelf")
            return "failed"
        return "moving"

    def configure_grasp(self) -> bool:
        if self.use_dual_tissue_grasp:
            return self.configure_dual_tissue_grasp()
        if self.use_sphere_grasp:
            return self.configure_sphere_grasp()
        if self.is_top_shelf:
            return self.configure_top_grasp()
        if self.shelf_level == "lower":
            return self.configure_lower_grasp()

        pregrasp_world = self.target_world.copy()
        pregrasp_world[1] -= PREGRASP_BACKOFF_M
        pregrasp_world[2] = float(max(
            pregrasp_world[2] + GRASP_TCP_Z_RAISE_M
            + GRASP_TCP_Z_OFFSET_BY_KIND.get(self.target_kind, 0.0),
            SHELF_SURFACE_Z_M[self.shelf_level]
            + GENERIC_TCP_FINGER_CLEARANCE_BY_KIND_M.get(
                self.target_kind, GENERIC_TCP_FINGER_CLEARANCE_M)))

        nominal_contact_world = pregrasp_world.copy()
        # The 30 mm fixed overshoot can push the wrist past the widest part of
        # a small product (kouxiangtang radius is only 24.5 mm), so the jaws
        # close behind it and miss.  Cap the overshoot just inside the
        # product's half-depth; the fixed diagnostic mode stays unchanged.
        grasp_overshoot = min(
            FRONT_GRASP_OVERSHOOT_M,
            PRODUCT_HALF_DEPTH_M.get(
                self.target_kind, FRONT_GRASP_OVERSHOOT_M) - 0.004)
        nominal_contact_world[1] = self.target_world[1] + (
            0.0 if self.tcp_diagnostic_ground_truth else grasp_overshoot)
        nominal_contact_world[2] = pregrasp_world[2]
        extended_contact_world = generic_post_extend_world(
            nominal_contact_world, self.target_kind)

        reference = (self.cmd_right_arm.copy() if self.grasp_arm == "r"
                     else self.cmd_left_arm.copy())
        arm_base_rotation = (
            MMK2FIK.TMat_chest2rgt_base[:3, :3]
            if self.grasp_arm == "r"
            else MMK2FIK.TMat_chest2lft_base[:3, :3])
        # MMK2FIK expects a rotation in the selected arm-base frame.  Applying
        # the inverse mount rotation makes the endpoint orientation identity in
        # the footprint frame: gripper upright and square to the shelf, with no
        # downward-pitched "pick" pose.
        front_grasp_rotation = arm_base_rotation.T

        # Slightly different initial backoffs give IK a safe fallback.  Both
        # endpoints keep exactly the same Z and orientation, so only the arm
        # extends toward the product; the mobile base does not push forward.
        for backoff in (PREGRASP_BACKOFF_M, 0.16, 0.14, 0.12):
            start_world = pregrasp_world.copy()
            start_world[1] = self.target_world[1] - backoff
            # Solve only the two endpoints.  The 50 Hz synchronized command
            # streamer below moves continuously between them; intermediate IK
            # targets must not introduce repeated stop-and-go motion.
            worlds = (
                start_world,
                nominal_contact_world,
                extended_contact_world)
            solutions = []
            waypoint_reference = reference
            for waypoint in worlds:
                footprint = self.world_to_footprint(waypoint)
                try:
                    joints = self.ik.get_armjoint_pose_wrt_footprint(
                        footprint, front_grasp_rotation, self.grasp_arm,
                        self.slide_grasp, waypoint_reference)
                except ValueError:
                    solutions = []
                    break
                waypoint_reference = np.asarray(joints, dtype=float)
                solutions.append(waypoint_reference)
            if not solutions:
                continue

            self.pregrasp_arm_joints = solutions[0].copy()
            self.approach_arm_joints = [solutions[1].copy()]
            self.post_extend_nominal_world = (
                nominal_contact_world.copy())
            self.post_extend_target_world = (
                extended_contact_world.copy())
            self.post_extend_arm_joints = solutions[2].copy()
            self.approach_index = 0
            self.forward_contact_world = nominal_contact_world.copy()
            if self.grasp_arm == "r":
                self.des_right_arm = self.pregrasp_arm_joints.copy()
                self.des_right_grip = self.grip_preshape_command
            else:
                self.des_left_arm = self.pregrasp_arm_joints.copy()
                self.des_left_grip = self.grip_preshape_command
            self.des_slide = self.slide_grasp
            self.get_logger().info(
                f"[IK-front] arm={self.grasp_arm} "
                f"pregrasp={np.round(start_world, 3)} "
                f"nominal_close={np.round(nominal_contact_world, 3)} "
                f"extended_close={np.round(extended_contact_world, 3)} "
                f"post_extension="
                f"{np.linalg.norm(extended_contact_world - nominal_contact_world):.3f}m "
                f"z_drop={GENERIC_POST_EXTEND_Z_DROP_M_BY_KIND.get(self.target_kind, 0.0):.3f}m "
                f"direct_speed={GENERIC_DIRECT_FORWARD_SPEED_MPS:.3f}m/s "
                "fixed_trajectory=1 feedback_gates=0 "
                f"tcp_diag={int(self.tcp_diagnostic_ground_truth)} "
                f"slide={self.slide_grasp:.3f}")
            return True
        if self.ik_retry_forward_m < GENERIC_IK_RETRY_MAX_M:
            step = min(
                GENERIC_IK_RETRY_STEP_M,
                GENERIC_IK_RETRY_MAX_M - self.ik_retry_forward_m)
            self.ik_retry_forward_m += step
            self.align_base_y = float(np.clip(
                self.align_base_y + step,
                SCAN_Y, GENERIC_ALIGN_Y_MAX_M))
            self.get_logger().warn(
                f"[IK-retry] generic front IK failed; re-driving forward "
                f"{self.ik_retry_forward_m:.3f}m -> align_y="
                f"{self.align_base_y:.3f}")
            return "retry"
        self.get_logger().error(
            f"IK failed for localised target {np.round(self.target_world, 3)}")
        return False

    def configure_lower_grasp(self) -> bool:
        """Configure an isolated, level front insertion for the lower shelf."""
        nominal_contact_world = self.target_world.copy()
        nominal_contact_world[2] = float(max(
            nominal_contact_world[2] + GRASP_TCP_Z_RAISE_M
            + GRASP_TCP_Z_OFFSET_BY_KIND.get(self.target_kind, 0.0),
            SHELF_SURFACE_Z_M[self.shelf_level]
            + GENERIC_TCP_FINGER_CLEARANCE_BY_KIND_M.get(
                self.target_kind, GENERIC_TCP_FINGER_CLEARANCE_M)))
        # This is a wrist-to-inner-finger geometry transform, not a perception
        # correction, so it remains active in fixed-layout diagnostic mode.
        nominal_contact_world[1] += LOWER_GRASP_TCP_FORWARD_M
        post_extension_enabled = self.object_geometry != "sphere"
        extended_contact_world = nominal_contact_world.copy()
        if post_extension_enabled:
            extended_contact_world = generic_post_extend_world(
                nominal_contact_world, self.target_kind)

        reference = (self.cmd_right_arm.copy() if self.grasp_arm == "r"
                     else self.cmd_left_arm.copy())
        arm_base_rotation = (
            MMK2FIK.TMat_chest2rgt_base[:3, :3]
            if self.grasp_arm == "r"
            else MMK2FIK.TMat_chest2lft_base[:3, :3])
        front_grasp_rotation = arm_base_rotation.T

        for backoff in (
                LOWER_PREGRASP_BACKOFF_M, 0.14, 0.18, 0.12):
            pregrasp_world = self.target_world.copy()
            pregrasp_world[1] -= backoff
            pregrasp_world[2] = nominal_contact_world[2]
            solutions = []
            waypoint_reference = reference
            # Solve the established close point and the new 50 mm endpoint
            # with identical height and orientation.  They are played as two
            # separate monotonic trajectories; the gripper stays open at the
            # first endpoint and closes only after the second.
            waypoints = [pregrasp_world, nominal_contact_world]
            if post_extension_enabled:
                waypoints.append(extended_contact_world)
            for waypoint in waypoints:
                footprint = self.world_to_footprint(waypoint)
                try:
                    joints = self.ik.get_armjoint_pose_wrt_footprint(
                        footprint, front_grasp_rotation, self.grasp_arm,
                        self.slide_grasp, waypoint_reference)
                except ValueError:
                    solutions = []
                    break
                waypoint_reference = np.asarray(joints, dtype=float)
                solutions.append(waypoint_reference)
            if not solutions:
                continue

            self.pregrasp_arm_joints = solutions[0].copy()
            self.approach_arm_joints = [solutions[1].copy()]
            if post_extension_enabled:
                self.post_extend_nominal_world = (
                    nominal_contact_world.copy())
                self.post_extend_target_world = (
                    extended_contact_world.copy())
                self.post_extend_arm_joints = solutions[2].copy()
            else:
                self.post_extend_nominal_world = None
                self.post_extend_target_world = None
                self.post_extend_arm_joints = None
            self.approach_index = 0
            self.forward_contact_world = nominal_contact_world.copy()
            if self.grasp_arm == "r":
                self.des_right_arm = self.pregrasp_arm_joints.copy()
                self.des_right_grip = self.grip_preshape_command
            else:
                self.des_left_arm = self.pregrasp_arm_joints.copy()
                self.des_left_grip = self.grip_preshape_command
            self.des_slide = self.slide_grasp
            self.get_logger().info(
                f"[IK-lower-front] arm={self.grasp_arm} "
                f"pregrasp={np.round(pregrasp_world, 3)} "
                f"nominal_close={np.round(nominal_contact_world, 3)} "
                f"extended_close="
                f"{None if not post_extension_enabled else np.round(extended_contact_world, 3)} "
                f"post_extension="
                f"{np.linalg.norm(extended_contact_world - nominal_contact_world) if post_extension_enabled else 0.0:.3f}m "
                f"z_drop={GENERIC_POST_EXTEND_Z_DROP_M_BY_KIND.get(self.target_kind, 0.0) if post_extension_enabled else 0.0:.3f}m "
                f"tcp_forward={LOWER_GRASP_TCP_FORWARD_M:.3f}m "
                f"slide={self.slide_grasp:.3f} "
                f"direct_speed={GENERIC_DIRECT_FORWARD_SPEED_MPS:.3f}m/s "
                "fixed_trajectory=1 feedback_gates=0")
            return True

        if self.ik_retry_forward_m < GENERIC_IK_RETRY_MAX_M:
            step = min(
                GENERIC_IK_RETRY_STEP_M,
                GENERIC_IK_RETRY_MAX_M - self.ik_retry_forward_m)
            self.ik_retry_forward_m += step
            self.align_base_y = float(np.clip(
                self.align_base_y + step,
                SCAN_Y, GENERIC_ALIGN_Y_MAX_M))
            self.get_logger().warn(
                f"[IK-retry] lower-front IK failed; re-driving forward "
                f"{self.ik_retry_forward_m:.3f}m -> align_y="
                f"{self.align_base_y:.3f}")
            return "retry"
        self.get_logger().error(
            f"lower-front IK failed for target "
            f"{np.round(self.target_world, 3)} at base="
            f"{np.round(self.base_xy, 3)} slide={self.slide_grasp:.3f}")
        return False

    def configure_sphere_grasp(self) -> bool:
        """Configure sphere geometry for the selected supported shelf layer."""
        pregrasp_world = self.target_world.copy()
        grasp_tcp_z = float(self.target_world[2] + GRASP_TCP_Z_RAISE_M)
        if self.shelf_level == "top":
            grasp_tcp_z = self.top_grasp_tcp_z()
        pregrasp_world[2] = grasp_tcp_z
        pregrasp_world[1] -= SPHERE_PREGRASP_BACKOFF_M
        contact_world = self.target_world.copy()
        contact_world[2] = grasp_tcp_z
        radius = SPHERE_RADIUS_M[self.target_kind]
        # Stop just inside the near surface.  The fingers then surround part of
        # the sphere without asking the wrist TCP to pass through its centre.
        contact_world[1] -= radius - SPHERE_FINGER_ENGAGEMENT_M
        reference = (self.cmd_right_arm.copy() if self.grasp_arm == "r"
                     else self.cmd_left_arm.copy())
        try:
            pregrasp_joints = self.solve_kdl_world(
                pregrasp_world, reference)
            contact_joints = self.solve_kdl_world(
                contact_world, pregrasp_joints)
        except ValueError as exc:
            self.get_logger().error(
                f"{self.shelf_level}-sphere IK failed: {exc}")
            return False

        self.sphere_pregrasp_world = pregrasp_world.copy()
        self.sphere_contact_world = contact_world.copy()
        self.forward_contact_world = contact_world.copy()
        self.sphere_forward_reference = pregrasp_joints.copy()
        self.sphere_forward_last_progress = -1.0
        self.sphere_open_grip_reference = None
        self.sphere_creep_start_world = None
        self.sphere_creep_goal_world = None
        self.sphere_creep_started_at = None
        self.sphere_creep_progress_samples.clear()
        self.sphere_close_grip_samples.clear()
        self.sphere_trial_grip_samples.clear()
        self.sphere_trial_lift_arm_joints = None
        self.sphere_lift_arm_joints = None
        self.sphere_retreat_arm_joints = None
        self.sphere_trial_slide = None
        self.sphere_lift_slide = None
        self.sphere_slide_command = self.slide_grasp
        self.middle_sphere_slide_corrections = 0
        self.sphere_grip_verified = False
        self.pregrasp_arm_joints = pregrasp_joints.copy()
        # Keep the endpoint for the common deploy guard.  The sphere state does
        # not jump to it; it generates a continuous Cartesian line at 50 Hz.
        self.approach_arm_joints = [contact_joints.copy()]
        self.approach_index = 0
        if self.grasp_arm == "r":
            self.des_right_arm = pregrasp_joints.copy()
            self.des_right_grip = self.grip_preshape_command
        else:
            self.des_left_arm = pregrasp_joints.copy()
            self.des_left_grip = self.grip_preshape_command
        self.des_slide = self.slide_grasp
        self.get_logger().info(
            f"[IK-{self.shelf_level}-sphere] arm={self.grasp_arm} "
            f"kind={self.target_kind} "
            f"radius={radius:.3f}m "
            f"finger_engagement={SPHERE_FINGER_ENGAGEMENT_M:.3f}m "
            f"product_center_z={self.target_world[2]:.4f}m "
            f"tcp_target_z={grasp_tcp_z:.4f}m "
            f"tcp_raise={grasp_tcp_z - self.target_world[2]:.4f}m "
            f"pregrasp={np.round(pregrasp_world, 3)} "
            f"contact={np.round(contact_world, 3)} "
            f"cart_speed={SPHERE_FAST_SPEED_MPS:.3f}->"
            f"{SPHERE_TERMINAL_SPEED_MPS:.3f}m/s "
            f"slow_zone={SPHERE_TERMINAL_ZONE_M:.3f}m "
            f"tcp_tol={np.round(SPHERE_TCP_TOLERANCE_M, 3)}m "
            f"slide={self.slide_grasp:.3f}")
        return True

    def configure_top_grasp(self) -> bool:
        """Raise to a front-facing top-shelf pregrasp, then extend to close."""
        grasp_tcp_z = self.top_grasp_tcp_z()
        pregrasp_world = self.target_world.copy()
        pregrasp_world[2] = grasp_tcp_z
        pregrasp_world[1] -= TOP_PREGRASP_BACKOFF_M
        nominal_contact_world = self.target_world.copy()
        nominal_contact_world[2] = grasp_tcp_z
        nominal_contact_world[1] += TOP_GRASP_TCP_FORWARD_M
        extended_contact_world = nominal_contact_world.copy()
        extended_contact_world[1] += GENERIC_POST_CONTACT_EXTENSION_M
        top_post_extend_z_drop = GENERIC_POST_EXTEND_Z_DROP_M_BY_KIND.get(
            self.target_kind, 0.0)
        extended_contact_world[2] -= top_post_extend_z_drop

        reference = (self.cmd_right_arm.copy() if self.grasp_arm == "r"
                     else self.cmd_left_arm.copy())
        arm_base_rotation = (
            MMK2FIK.TMat_chest2rgt_base[:3, :3]
            if self.grasp_arm == "r"
            else MMK2FIK.TMat_chest2lft_base[:3, :3])
        # Identity in the footprint frame points the gripper squarely toward
        # the shelf.  Pregrasp and contact have identical Z and orientation;
        # after the high pose is reached there is no intermediate correction
        # target that could make the arm shake.  Products with an explicit
        # post-extension Z drop descend only during the final monotonic segment.
        top_grasp_rotation = arm_base_rotation.T

        solutions = []
        waypoint_reference = reference
        for waypoint in (
                pregrasp_world,
                nominal_contact_world,
                extended_contact_world):
            footprint = self.world_to_footprint(waypoint)
            try:
                joints = self.ik.get_armjoint_pose_wrt_footprint(
                    footprint, top_grasp_rotation, self.grasp_arm,
                    self.slide_grasp, waypoint_reference)
            except ValueError:
                solutions = []
                break
            waypoint_reference = np.asarray(joints, dtype=float)
            solutions.append(waypoint_reference)

        if not solutions:
            self.get_logger().error(
                f"top-shelf IK failed for target "
                f"{np.round(self.target_world, 3)} at base_y="
                f"{self.align_base_y:.3f}, slide={self.slide_grasp:.3f}")
            return False

        self.pregrasp_arm_joints = solutions[0].copy()
        self.approach_arm_joints = [solutions[1].copy()]
        self.post_extend_nominal_world = nominal_contact_world.copy()
        self.post_extend_target_world = extended_contact_world.copy()
        self.post_extend_arm_joints = solutions[2].copy()
        self.approach_index = 0
        self.forward_contact_world = nominal_contact_world.copy()
        if self.grasp_arm == "r":
            self.des_right_arm = self.pregrasp_arm_joints.copy()
            self.des_right_grip = self.grip_preshape_command
        else:
            self.des_left_arm = self.pregrasp_arm_joints.copy()
            self.des_left_grip = self.grip_preshape_command
        self.des_slide = self.slide_grasp
        self.get_logger().info(
            f"[IK-top-front] arm={self.grasp_arm} "
            f"pregrasp={np.round(pregrasp_world, 3)} "
            f"nominal_close={np.round(nominal_contact_world, 3)} "
            f"extended_close={np.round(extended_contact_world, 3)} "
            f"post_extension={GENERIC_POST_CONTACT_EXTENSION_M:.3f}m "
            f"z_drop={top_post_extend_z_drop:.3f}m "
            "wrist=front-upright "
            f"product_center_z={self.target_world[2]:.4f}m "
            f"tcp_target_z={grasp_tcp_z:.4f}m "
            f"tcp_raise={grasp_tcp_z - self.target_world[2]:.4f}m "
            f"tcp_forward={TOP_GRASP_TCP_FORWARD_M:.3f}m "
            f"direct_speed={GENERIC_DIRECT_FORWARD_SPEED_MPS:.3f}m/s "
            "fixed_trajectory=1 feedback_gates=0 "
            f"tcp_diag={int(self.tcp_diagnostic_ground_truth)} "
            f"slide={self.slide_grasp:.3f}")
        return True

    def set_selected_arm_target(self, joints: np.ndarray) -> None:
        self.commands_ready_since = None
        if self.grasp_arm == "r":
            self.des_right_arm = np.asarray(joints, dtype=float).copy()
        else:
            self.des_left_arm = np.asarray(joints, dtype=float).copy()

    def middle_sphere_pregrasp_ready(self) -> bool:
        """Verify measured TCP and compensate residual Z with the slide."""
        actual_tcp = self.selected_tcp_world()
        if actual_tcp is None:
            return False
        error = actual_tcp - self.sphere_pregrasp_world
        if np.all(
                np.abs(error)
                <= MIDDLE_SPHERE_PREGRASP_TCP_TOLERANCE_M):
            self.get_logger().info(
                f"[middle-sphere-deploy] measured pregrasp ready; "
                f"actual={np.round(actual_tcp, 4)} "
                f"error={np.round(error, 4)}m "
                f"slide={self.sphere_slide_command:.4f}")
            return True

        if (abs(error[2]) > MIDDLE_SPHERE_PREGRASP_TCP_TOLERANCE_M[2]
                and self.middle_sphere_slide_corrections
                < MIDDLE_SPHERE_SLIDE_CORRECTIONS_MAX):
            correction = float(np.clip(
                error[2],
                -MIDDLE_SPHERE_SLIDE_CORRECTION_MAX_STEP_M,
                MIDDLE_SPHERE_SLIDE_CORRECTION_MAX_STEP_M))
            corrected_slide = float(np.clip(
                self.sphere_slide_command + correction,
                SLIDE_MIN, SLIDE_MAX))
            if abs(corrected_slide - self.sphere_slide_command) > 1e-5:
                previous_slide = self.sphere_slide_command
                self.sphere_slide_command = corrected_slide
                self.des_slide = corrected_slide
                self.middle_sphere_slide_corrections += 1
                self.commands_ready_since = None
                self.get_logger().info(
                    f"[middle-sphere-deploy] correcting measured TCP Z; "
                    f"actual={np.round(actual_tcp, 4)} "
                    f"error={np.round(error, 4)}m "
                    f"slide={previous_slide:.4f}->"
                    f"{corrected_slide:.4f} "
                    f"attempt={self.middle_sphere_slide_corrections}/"
                    f"{MIDDLE_SPHERE_SLIDE_CORRECTIONS_MAX}")
        return False

    def start_arm_forward(self) -> None:
        self.approach_index = 0
        self.forward_terminal_entered_at = None
        self.forward_terminal_slow_logged = False
        self.forward_start_tcp = self.selected_tcp_world()
        self.forward_start_base_xy = self.base_xy.copy()
        self.get_logger().info(
            f"[arm-forward-start] tcp="
            f"{None if self.forward_start_tcp is None else np.round(self.forward_start_tcp, 4)} "
            f"contact={np.round(self.forward_contact_world, 4)} "
            f"base={np.round(self.forward_start_base_xy, 4)} "
            f"grip={self.selected_gripper_position()} "
            f"preshape={self.grip_preshape_command:.3f}")
        if self.use_sphere_grasp:
            self.sphere_forward_reference = self.pregrasp_arm_joints.copy()
            self.sphere_forward_last_progress = -1.0
            self.sphere_open_grip_reference = self.selected_gripper_position()
            self.sphere_creep_start_world = None
            self.sphere_creep_goal_world = None
            self.sphere_creep_started_at = None
            self.sphere_creep_progress_samples.clear()
            self.set_selected_arm_target(self.pregrasp_arm_joints)
        else:
            # Freeze one start pose and the already-solved contact endpoint.
            # The following motion is open-loop and monotonic: no measured TCP
            # is fed back into IK and no convergence gate can pause/reverse it.
            self.generic_forward_start_world = (
                self.forward_start_tcp.copy()
                if self.forward_start_tcp is not None
                else self.target_world.copy())
            self.generic_direct_start_joints = (
                self.selected_arm_positions().copy())
            try:
                # 用“当前实测关节 + 当前基座”重新解一次接触末端，而不是沿用
                # 理想 pregrasp 解出的旧端点。预抓取若有几度误差，旧端点会
                # 把误差带进前伸轨迹；重解能显著减少最终 TCP 偏差。
                self.generic_direct_contact_joints = (
                    self.solve_kdl_world(
                        self.forward_contact_world,
                        self.selected_arm_positions()).copy())
            except Exception:
                self.generic_direct_contact_joints = (
                    self.approach_arm_joints[-1].copy())
            path_length = float(np.linalg.norm(
                self.forward_contact_world - self.generic_forward_start_world))
            # Smoothstep's peak slope is 1.5 times its average.  Scaling the
            # duration by 1.5 keeps the approximate peak Cartesian speed at or
            # below the requested low speed.
            self.generic_direct_duration_s = max(
                GENERIC_DIRECT_FORWARD_MIN_DURATION_S,
                1.5 * path_length / GENERIC_DIRECT_FORWARD_SPEED_MPS)
            self.generic_direct_endpoint_ready_since = None
            self.set_selected_arm_target(self.generic_direct_start_joints)
            self.get_logger().info(
                f"[generic-direct] fixed endpoint armed; "
                f"duration={self.generic_direct_duration_s:.2f}s "
                f"settle={GENERIC_DIRECT_FORWARD_SETTLE_S:.2f}s "
                f"speed={GENERIC_DIRECT_FORWARD_SPEED_MPS:.3f}m/s "
                "feedback_gates=0 replanning=0")
        self.set_state(STATE_ARM_FORWARD)

    def start_dual_tissue_motion(
            self, label: str, left_target: np.ndarray,
            right_target: np.ndarray, path_length: float,
            speed: float, state: str,
            require_convergence: bool = False) -> None:
        """Start one fixed, synchronized dual-arm segment."""
        self.dual_motion_start_left = self.arm_positions("left")
        self.dual_motion_start_right = self.arm_positions("right")
        self.dual_motion_target_left = np.asarray(
            left_target, dtype=float).copy()
        self.dual_motion_target_right = np.asarray(
            right_target, dtype=float).copy()
        self.dual_motion_duration_s = max(
            DUAL_TISSUE_MIN_MOTION_DURATION_S,
            1.5 * path_length / speed)
        self.dual_motion_label = label
        self.dual_motion_require_convergence = require_convergence
        self.dual_motion_endpoint_ready_since = None
        self.dual_motion_path_distances = None
        self.dual_motion_path_left = None
        self.dual_motion_path_right = None
        self.des_left_arm = self.dual_motion_start_left.copy()
        self.des_right_arm = self.dual_motion_start_right.copy()
        self.commands_ready_since = None
        left_joint_delta = float(np.max(np.abs(
            self.dual_motion_target_left - self.dual_motion_start_left)))
        right_joint_delta = float(np.max(np.abs(
            self.dual_motion_target_right - self.dual_motion_start_right)))
        self.get_logger().info(
            f"[dual-tissue-{label}] fixed synchronized segment armed; "
            f"path={path_length:.3f}m "
            f"duration={self.dual_motion_duration_s:.2f}s "
            f"speed={speed:.3f}m/s "
            f"joint_delta=({left_joint_delta:.3f},"
            f"{right_joint_delta:.3f})rad "
            f"convergence_gate={int(require_convergence)} replanning=0")
        self.set_state(state)
        # The post-band dogleg re-enters STATE_ARM_FORWARD for each segment;
        # set_state skips the timestamp on an unchanged state, so reset the
        # segment timer explicitly here.
        self.state_t0 = self.now()

    def start_dual_tissue_waypoint_motion(
            self, label: str,
            waypoints: list[tuple[float, np.ndarray, np.ndarray]],
            speed: float, state: str,
            require_convergence: bool = False) -> None:
        """Play guarded IK samples as one uninterrupted loaded motion."""
        if not waypoints:
            raise ValueError(f"{label} requires at least one waypoint")
        # Preserve the previously commanded clamp pose at the handoff.  Using
        # measured joints here would momentarily remove useful position error.
        start_left = (
            self.arm_positions("left")
            if self.des_left_arm is None
            else np.asarray(self.des_left_arm, dtype=float).copy())
        start_right = (
            self.arm_positions("right")
            if self.des_right_arm is None
            else np.asarray(self.des_right_arm, dtype=float).copy())
        distances = [0.0]
        left_path = [start_left.copy()]
        right_path = [start_right.copy()]
        for distance, left_target, right_target in waypoints:
            distances.append(float(distance))
            left_path.append(np.asarray(left_target, dtype=float).copy())
            right_path.append(np.asarray(right_target, dtype=float).copy())

        total_path = max(0.001, distances[-1])
        self.dual_motion_start_left = start_left
        self.dual_motion_start_right = start_right
        self.dual_motion_target_left = left_path[-1].copy()
        self.dual_motion_target_right = right_path[-1].copy()
        self.dual_motion_path_distances = np.asarray(distances, dtype=float)
        self.dual_motion_path_left = np.asarray(left_path, dtype=float)
        self.dual_motion_path_right = np.asarray(right_path, dtype=float)
        self.dual_motion_duration_s = max(
            DUAL_TISSUE_MIN_MOTION_DURATION_S,
            1.5 * total_path / speed)
        self.dual_motion_label = label
        self.dual_motion_require_convergence = require_convergence
        self.dual_motion_endpoint_ready_since = None
        self.des_left_arm = start_left.copy()
        self.des_right_arm = start_right.copy()
        self.commands_ready_since = None
        self.get_logger().info(
            f"[dual-tissue-{label}] continuous guarded path armed; "
            f"path={total_path:.3f}m waypoints={len(waypoints)} "
            f"duration={self.dual_motion_duration_s:.2f}s "
            f"speed={speed:.3f}m/s convergence_gate="
            f"{int(require_convergence)} replanning=0")
        self.set_state(state)
        self.state_t0 = self.now()

    def start_dual_tissue_surround(self) -> None:
        """Start the side-pose insertion (middle columns direct, side columns
        route over the front post without any wrist rotation)."""
        self.forward_start_base_xy = self.base_xy.copy()
        self.dual_surround_stage = 0
        if self.dual_overhead_route:
            self.start_dual_tissue_motion(
                "surround_dogleg_out",
                self.dual_surround_pass_left_joints,
                self.dual_surround_pass_right_joints,
                max(
                    0.0,
                    DUAL_TISSUE_POST_OUTER_SPAN_M
                    - self.dual_pregrasp_half_span),
                DUAL_TISSUE_CLOSE_SPEED_MPS,
                STATE_ARM_FORWARD,
                require_convergence=True)
            return
        forward = (
            DUAL_TISSUE_PREGRASP_BACKOFF_M
            + self.dual_insert_forward_m)
        lateral = (
            self.dual_pregrasp_half_span
            - getattr(
                self, "dual_surround_half_span",
                DUAL_TISSUE_SURROUND_HALF_SPAN_M))
        probe_left = (
            self.dual_surround_pass_left_joints
            if self.dual_surround_pass_left_joints is not None
            else self.dual_surround_left_joints)
        probe_right = (
            self.dual_surround_pass_right_joints
            if self.dual_surround_pass_right_joints is not None
            else self.dual_surround_right_joints)
        nominal_path = math.hypot(forward, lateral)
        measured_paths = []
        for side, target in (
                ("left", probe_left), ("right", probe_right)):
            try:
                actual_tcp = self.arm_tcp_world(side)
                target_tcp = self.arm_target_tcp_world(side, target)
            except (AttributeError, KeyError, TypeError, ValueError):
                # Host-side geometry tests intentionally build a minimal
                # controller without live joint feedback.  Production falls
                # back the same way during a transient missing sample.
                continue
            if actual_tcp is None or target_tcp is None:
                continue
            distance = float(np.linalg.norm(target_tcp - actual_tcp))
            if math.isfinite(distance):
                measured_paths.append(distance)
        measured_path = (
            max(measured_paths) if measured_paths else nominal_path)
        effective_path = max(nominal_path, measured_path)
        if measured_path > nominal_path + 0.005:
            self.get_logger().warn(
                "[dual-tissue-surround] measured arm start is behind the "
                f"nominal pregrasp; nominal_path={nominal_path:.3f}m "
                f"measured_path={measured_path:.3f}m; extending duration")
        self.start_dual_tissue_motion(
            "surround",
            probe_left, probe_right,
            effective_path,
            DUAL_TISSUE_FORWARD_SPEED_MPS,
            STATE_ARM_FORWARD,
            require_convergence=True)

    def advance_dual_tissue_deploy(self, deploy_elapsed: float) -> None:
        """Wait for the measured symmetric pregrasp before insertion.

        A fixed dwell is not evidence that the physical arms have arrived.
        The insertion duration assumes this pregrasp as its starting pose, so
        entering it early compresses the remaining motion window and makes a
        slower arm fail the endpoint gate.  Start immediately once the old
        minimum dwell and a stable dual-arm/slide feedback gate both pass.
        """
        if (self.dual_surround_left_joints is None
                or self.dual_surround_right_joints is None):
            self.get_logger().error(
                "dual tissue approach has no solved endpoint")
            self.set_state(STATE_ABORT)
            return

        deploy_ready = self.dual_commands_ready(
            ARM_REACHED_TOLERANCE_RAD + 0.015, 0.025)
        if (deploy_elapsed >= DUAL_TISSUE_DEPLOY_DWELL_S
                and deploy_ready):
            self.get_logger().info(
                f"[dual-tissue-deploy] measured pregrasp stable after "
                f"{deploy_elapsed:.2f}s; dual_arm_error="
                f"{self.dual_arm_error():.4f}rad; starting fixed surround "
                "motion")
            self.start_dual_tissue_surround()
            return

        if deploy_elapsed < DUAL_TISSUE_DEPLOY_TIMEOUT_S:
            return

        measured_slide = self.joints.get("slide_joint")
        slide_error = (
            float("inf") if measured_slide is None
            else abs(float(measured_slide) - self.des_slide))
        self.get_logger().error(
            "[dual-tissue-deploy] measured pregrasp did not converge; "
            f"elapsed={deploy_elapsed:.2f}s "
            f"dual_arm_error={self.dual_arm_error():.4f}rad "
            f"slide_error={slide_error:.4f}m; aborting before insertion")
        self.set_state(STATE_ABORT)

    def advance_dual_tissue_surround_sequence(self) -> None:
        """After insertion, close (middle) or start the deep squeeze."""
        self.dual_surround_stage += 1
        if getattr(self, "dual_middle_extend_close", False):
            if self.dual_surround_stage == 1:
                close_left = (
                    self.dual_surround_close_left_joints
                    if self.dual_surround_close_left_joints is not None
                    else self.dual_clamp_left_joints)
                close_right = (
                    self.dual_surround_close_right_joints
                    if self.dual_surround_close_right_joints is not None
                    else self.dual_clamp_right_joints)
                if close_left is None or close_right is None:
                    self.get_logger().error(
                        "[dual-tissue-surround] middle-column close "
                        "targets are unavailable")
                    self.set_state(STATE_ABORT)
                    return
                self.start_dual_tissue_motion(
                    "surround_close",
                    close_left, close_right,
                    max(
                        0.0,
                        self.dual_pregrasp_half_span
                        - self.dual_clamp_half_span),
                    DUAL_TISSUE_CLOSE_SPEED_MPS,
                    STATE_DUAL_SQUEEZE,
                    require_convergence=True)
            else:
                self.get_logger().error(
                    "[dual-tissue-surround] unexpected middle-column "
                    f"stage={self.dual_surround_stage}")
                self.set_state(STATE_ABORT)
            return
        if self.dual_overhead_route:
            if self.dual_surround_stage == 1:
                self.start_dual_tissue_motion(
                    "surround_dogleg_forward",
                    self.dual_surround_forward_left_joints,
                    self.dual_surround_forward_right_joints,
                    DUAL_TISSUE_PREGRASP_BACKOFF_M
                    - DUAL_TISSUE_POST_Y_CLEAR_M,
                    DUAL_TISSUE_FORWARD_SPEED_MPS,
                    STATE_ARM_FORWARD,
                    require_convergence=True)
                return
            if self.dual_surround_stage == 2:
                self.start_dual_tissue_motion(
                    "surround_dogleg_in",
                    self.dual_surround_return_left_joints,
                    self.dual_surround_return_right_joints,
                    max(
                        0.0,
                        DUAL_TISSUE_POST_OUTER_SPAN_M
                        - DUAL_TISSUE_DIRECT_PROBE_SPAN_M),
                    DUAL_TISSUE_CLOSE_SPEED_MPS,
                    STATE_ARM_FORWARD,
                    require_convergence=True)
                return
            if self.dual_surround_stage == 3:
                self.start_dual_tissue_motion(
                    "surround_final",
                    self.dual_surround_left_joints,
                    self.dual_surround_right_joints,
                    max(
                        0.0,
                        self.dual_insert_forward_m
                        + DUAL_TISSUE_POST_Y_CLEAR_M),
                    DUAL_TISSUE_FORWARD_SPEED_MPS,
                    STATE_ARM_FORWARD,
                    require_convergence=True)
                return
        elif (self.dual_surround_unroll_left_joints is not None
              and self.dual_surround_stage == 1):
            self.start_dual_tissue_motion(
                "surround_unroll",
                self.dual_surround_unroll_left_joints,
                self.dual_surround_unroll_right_joints,
                DUAL_TISSUE_UNROLL_PATH_M,
                DUAL_TISSUE_UNROLL_SPEED_MPS,
                STATE_ARM_FORWARD,
                require_convergence=True)
            return
        if getattr(self, "dual_side_rolled", False):
            # Confirm both side contacts before applying the final preload.
            self.start_dual_tissue_contact_search()
            return
        if not self.start_dual_tissue_squeeze():
            self.set_state(STATE_ABORT)

    def start_dual_tissue_contact_search(self) -> None:
        """Move each arm monotonically inward until its own contact.

        侧列按立柱方向镜像：立柱侧手臂慢推，另一侧停靠，
        使纸盒始终离开立柱。squeeze 再以实测双侧位置锚定。
        """
        left_tcp = self.arm_tcp_world("left")
        right_tcp = self.arm_tcp_world("right")
        if left_tcp is None or right_tcp is None:
            self.get_logger().error(
                "[dual-tissue-contact] measured TCP unavailable")
            self.set_state(STATE_ABORT)
            return

        common_y = 0.5 * (left_tcp[1] + right_tcp[1])
        common_z = 0.5 * (left_tcp[2] + right_tcp[2])
        push_side = getattr(self, "dual_contact_push_side", "left")
        left_span = (
            DUAL_TISSUE_PARK_SPAN_M
            if push_side == "right" else DUAL_TISSUE_PUSH_SPAN_M)
        right_span = (
            DUAL_TISSUE_PUSH_SPAN_M
            if push_side == "right" else DUAL_TISSUE_PARK_SPAN_M)
        left_goal = np.array([
            self.target_world[0] - left_span, common_y, common_z])
        right_goal = np.array([
            self.target_world[0] + right_span, common_y, common_z])
        left_start_joints = self.arm_positions("left")
        right_start_joints = self.arm_positions("right")
        try:
            left_goal_joints, right_goal_joints = self.solve_kdl_both_world(
                left_goal, right_goal,
                left_start_joints, right_start_joints)
            if getattr(self, "dual_side_rolled", False):
                segment_delta = max(
                    float(np.max(np.abs(
                        left_goal_joints - left_start_joints))),
                    float(np.max(np.abs(
                        right_goal_joints - right_start_joints))))
                if (segment_delta
                        > DUAL_TISSUE_SIDE_ROLLED_MAX_SEGMENT_JOINT_DELTA_RAD):
                    raise ValueError(
                        "side-column rolled contact search changed one "
                        f"joint by {segment_delta:.3f}rad")
        except ValueError as exc:
            self.get_logger().error(
                f"[dual-tissue-contact] search IK failed: {exc}")
            self.set_state(STATE_ABORT)
            return

        left_path = max(0.0, left_goal[0] - left_tcp[0])
        right_path = max(0.0, right_tcp[0] - right_goal[0])
        self.dual_contact_start_left_joints = left_start_joints
        self.dual_contact_start_right_joints = right_start_joints
        self.dual_contact_target_left_joints = left_goal_joints
        self.dual_contact_target_right_joints = right_goal_joints
        self.dual_contact_start_left_world = left_tcp.copy()
        self.dual_contact_start_right_world = right_tcp.copy()
        self.dual_contact_goal_left_world = left_goal
        self.dual_contact_goal_right_world = right_goal
        self.dual_contact_left_duration_s = max(
            DUAL_TISSUE_MIN_MOTION_DURATION_S,
            1.5 * left_path / DUAL_TISSUE_CONTACT_SEARCH_SPEED_MPS)
        self.dual_contact_right_duration_s = max(
            DUAL_TISSUE_MIN_MOTION_DURATION_S,
            1.5 * right_path / DUAL_TISSUE_CONTACT_SEARCH_SPEED_MPS)
        self.dual_contact_duration_s = max(
            self.dual_contact_left_duration_s,
            self.dual_contact_right_duration_s)
        self.dual_left_contacted = False
        self.dual_right_contacted = False
        self.dual_left_contact_hold_joints = None
        self.dual_right_contact_hold_joints = None
        self.dual_left_contact_samples.clear()
        self.dual_right_contact_samples.clear()
        self.des_left_arm = left_start_joints.copy()
        self.des_right_arm = right_start_joints.copy()
        line_angle = math.degrees(math.atan2(
            right_tcp[1] - left_tcp[1],
            right_tcp[0] - left_tcp[0]))
        if abs(line_angle) > DUAL_TISSUE_MAX_LINE_ANGLE_DEG:
            self.get_logger().error(
                f"[dual-tissue-contact] start line angle "
                f"{line_angle:.2f}deg exceeds "
                f"{DUAL_TISSUE_MAX_LINE_ANGLE_DEG:.1f}deg; "
                "one arm is blocked or the box is rotated - aborting "
                "instead of clamping a skewed box")
            self.set_state(STATE_ABORT)
            return
        self.get_logger().info(
            f"[dual-tissue-contact] independent monotonic search armed; "
            f"duration={self.dual_contact_duration_s:.2f}s "
            f"side_durations=({self.dual_contact_left_duration_s:.2f},"
            f"{self.dual_contact_right_duration_s:.2f})s "
            f"speed={DUAL_TISSUE_CONTACT_SEARCH_SPEED_MPS:.3f}m/s "
            f"push_side={push_side} "
            f"start_line_angle={line_angle:.2f}deg "
            f"left_start={np.round(left_tcp, 4)} "
            f"right_start={np.round(right_tcp, 4)} "
            f"left_goal={np.round(left_goal, 4)} "
            f"right_goal={np.round(right_goal, 4)}")
        self.set_state(STATE_DUAL_CONTACT)

    def dual_tissue_contact_stalled(
            self, samples: deque, actual_progress: float,
            commanded_progress: float) -> bool:
        """Detect sustained lack of inward progress behind the command."""
        now = self.now()
        samples.append((now, actual_progress))
        cutoff = now - DUAL_TISSUE_CONTACT_STALL_WINDOW_S
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        if len(samples) < 2:
            return False
        span = samples[-1][0] - samples[0][0]
        travel_range = (
            max(item[1] for item in samples)
            - min(item[1] for item in samples))
        return (
            actual_progress >= DUAL_TISSUE_CONTACT_MIN_ADVANCE_M
            and commanded_progress - actual_progress
            >= DUAL_TISSUE_CONTACT_COMMAND_LEAD_M
            and span >= DUAL_TISSUE_CONTACT_STALL_MIN_SPAN_S
            and travel_range <= DUAL_TISSUE_CONTACT_STALL_RANGE_M)

    def mark_dual_tissue_contact(
            self, side: str, reason: str,
            actual_tcp: np.ndarray | None) -> None:
        """Freeze one arm at its current commanded light-contact pose."""
        if side == "left":
            if self.dual_left_contacted:
                return
            self.dual_left_contacted = True
            self.dual_left_contact_hold_joints = self.des_left_arm.copy()
        else:
            if self.dual_right_contacted:
                return
            self.dual_right_contacted = True
            self.dual_right_contact_hold_joints = self.des_right_arm.copy()
        self.get_logger().info(
            f"[dual-tissue-contact] side={side} contact={reason} "
            f"tcp={None if actual_tcp is None else np.round(actual_tcp, 4)}; "
            "holding this side while the other continues inward")

    def advance_dual_tissue_contact_search(self) -> str:
        """Advance both searches independently, never reversing either arm."""
        elapsed = self.now() - self.state_t0
        duration = max(self.dual_contact_duration_s, 1e-6)
        left_duration = max(self.dual_contact_left_duration_s, 1e-6)
        right_duration = max(self.dual_contact_right_duration_s, 1e-6)
        left_progress = float(np.clip(
            elapsed / left_duration, 0.0, 1.0))
        right_progress = float(np.clip(
            elapsed / right_duration, 0.0, 1.0))
        left_eased = (
            left_progress * left_progress * (3.0 - 2.0 * left_progress))
        right_eased = (
            right_progress * right_progress * (3.0 - 2.0 * right_progress))
        if self.dual_left_contacted:
            self.des_left_arm = self.dual_left_contact_hold_joints.copy()
        else:
            self.des_left_arm = (
                self.dual_contact_start_left_joints
                + left_eased * (self.dual_contact_target_left_joints
                                - self.dual_contact_start_left_joints))
        if self.dual_right_contacted:
            self.des_right_arm = self.dual_right_contact_hold_joints.copy()
        else:
            self.des_right_arm = (
                self.dual_contact_start_right_joints
                + right_eased * (self.dual_contact_target_right_joints
                                 - self.dual_contact_start_right_joints))

        left_tcp = self.arm_tcp_world("left")
        right_tcp = self.arm_tcp_world("right")
        if left_tcp is not None and not self.dual_left_contacted:
            actual = max(
                0.0, left_tcp[0] - self.dual_contact_start_left_world[0])
            total = max(
                0.0,
                self.dual_contact_goal_left_world[0]
                - self.dual_contact_start_left_world[0])
            commanded = left_eased * total
            # The closed gripper is used as a flat paddle; its tendon reading
            # is noisy (observed down to -0.000) and can fire "contact" after
            # only a few mm of travel, far from the box side.  Only trust the
            # stall/endpoint detectors, which require the arm to actually stop
            # against the box or reach the search endpoint.
            if self.dual_tissue_contact_stalled(
                    self.dual_left_contact_samples, actual, commanded):
                self.mark_dual_tissue_contact("left", "stall", left_tcp)
            elif actual >= max(
                    0.0,
                    total - DUAL_TISSUE_CONTACT_ENDPOINT_TOLERANCE_M):
                self.mark_dual_tissue_contact("left", "endpoint", left_tcp)
        if right_tcp is not None and not self.dual_right_contacted:
            actual = max(
                0.0,
                self.dual_contact_start_right_world[0] - right_tcp[0])
            total = max(
                0.0,
                self.dual_contact_start_right_world[0]
                - self.dual_contact_goal_right_world[0])
            commanded = right_eased * total
            if self.dual_tissue_contact_stalled(
                    self.dual_right_contact_samples, actual, commanded):
                self.mark_dual_tissue_contact("right", "stall", right_tcp)
            elif actual >= max(
                    0.0,
                    total - DUAL_TISSUE_CONTACT_ENDPOINT_TOLERANCE_M):
                self.mark_dual_tissue_contact("right", "endpoint", right_tcp)

        timed_out = (
            elapsed >= duration + DUAL_TISSUE_MOTION_SETTLE_S)
        if timed_out:
            missing = []
            if not self.dual_left_contacted:
                missing.append("left")
            if not self.dual_right_contacted:
                missing.append("right")
            if missing:
                self.get_logger().error(
                    "[dual-tissue-contact] bilateral contact was not "
                    f"confirmed before timeout; missing={missing} "
                    f"left_tcp={None if left_tcp is None else np.round(left_tcp, 4)} "
                    f"right_tcp={None if right_tcp is None else np.round(right_tcp, 4)}; "
                    "aborting instead of applying squeeze to an unbracketed box")
                return "failed"

        if self.dual_left_contacted and self.dual_right_contacted:
            left_tcp = self.arm_tcp_world("left")
            right_tcp = self.arm_tcp_world("right")
            line_angle = None if left_tcp is None or right_tcp is None else (
                math.degrees(math.atan2(
                    right_tcp[1] - left_tcp[1],
                    right_tcp[0] - left_tcp[0])))
            if (line_angle is not None
                    and abs(line_angle) > DUAL_TISSUE_MAX_LINE_ANGLE_DEG):
                self.get_logger().error(
                    f"[dual-tissue-contact] bilateral line angle "
                    f"{line_angle:.2f}deg exceeds "
                    f"{DUAL_TISSUE_MAX_LINE_ANGLE_DEG:.1f}deg; "
                    "grasp would be skewed - aborting")
                return "failed"
            self.get_logger().info(
                f"[dual-tissue-contact] bilateral contact complete; "
                f"line_angle={line_angle}deg; preparing equal squeeze")
            return "reached"
        return "moving"

    def start_dual_tissue_squeeze(self) -> bool:
        """Close both arms onto the box with a measured-anchored clamp.

        中列沿用手侧面夹持；侧列使用竖直固定侧板的滚腕三点夹持。
        两种路径都以实测双侧接触位置为锚点，不盲推视觉中心。

        20260818：直接探入后不再按名义目标中心盲目深压——右臂在从探入
        跨度合拢到 0.09 名义跨度时会被纸巾提前顶住（实测右 TCP 停在
        1.680 而目标 1.670），随后抬升阶段右臂关节被憋出 ~1.04 rad 误差。
        改为以实测左右 TCP 为锚，每侧只额外合拢 DUAL_TISSUE_SQUEEZE_M
        （0.010m），合拢量小且始终围绕真实纸盒位置，不会在闭合前把右臂
        硬顶进纸巾。
        """
        measured_left = self.arm_tcp_world("left")
        measured_right = self.arm_tcp_world("right")
        if measured_left is not None and measured_right is not None:
            # 只施加当前姿态对应的小预压。
            common_y = 0.5 * (measured_left[1] + measured_right[1])
            common_z = 0.5 * (measured_left[2] + measured_right[2])
            box_centre_x = 0.5 * (measured_left[0] + measured_right[0])
            squeeze_left = measured_left.copy()
            squeeze_left[0] += self.dual_squeeze_m
            squeeze_right = measured_right.copy()
            squeeze_right[0] -= self.dual_squeeze_m
            squeeze_left[1] = common_y
            squeeze_left[2] = common_z
            squeeze_right[1] = common_y
            squeeze_right[2] = common_z
            final_half_span = 0.5 * (squeeze_right[0] - squeeze_left[0])
        else:
            box_centre_x = float(self.target_world[0])
            common_y = float(
                self.target_world[1] + self.dual_insert_forward_m)
            common_z = float(self.dual_contact_tcp_z)
            squeeze_left = np.array([
                box_centre_x - self.dual_clamp_half_span, common_y,
                common_z])
            squeeze_right = np.array([
                box_centre_x + self.dual_clamp_half_span, common_y,
                common_z])
            final_half_span = self.dual_clamp_half_span
        clamp_left = squeeze_left.copy()
        clamp_right = squeeze_right.copy()
        left_reference = self.arm_positions("left")
        right_reference = self.arm_positions("right")
        try:
            # 手背 outward 滚转姿态下合拢/撤退。
            squeeze_left_joints, squeeze_right_joints = (
                self.solve_kdl_both_world(
                    squeeze_left, squeeze_right,
                    left_reference, right_reference))
            clamp_left_joints, clamp_right_joints = (
                self.solve_kdl_both_world(
                    clamp_left, clamp_right,
                    squeeze_left_joints, squeeze_right_joints))
            retreat_left = clamp_left.copy()
            retreat_right = clamp_right.copy()
            retreat_y = self.target_world[1] - (
                DUAL_TISSUE_TOP_EDGE_BACKOFF_M
                if self.shelf_level == "top"
                else DUAL_TISSUE_PREGRASP_BACKOFF_M)
            retreat_left[1] = retreat_y
            retreat_right[1] = retreat_y
            retreat_left_joints, retreat_right_joints = (
                self.solve_kdl_both_world(
                    retreat_left, retreat_right,
                    clamp_left_joints, clamp_right_joints))
            if getattr(self, "dual_side_rolled", False):
                squeeze_delta = max(
                    float(np.max(np.abs(
                        squeeze_left_joints - left_reference))),
                    float(np.max(np.abs(
                        squeeze_right_joints - right_reference))))
                retreat_delta = max(
                    float(np.max(np.abs(
                        retreat_left_joints - clamp_left_joints))),
                    float(np.max(np.abs(
                        retreat_right_joints - clamp_right_joints))))
                if max(squeeze_delta, retreat_delta) > (
                        DUAL_TISSUE_SIDE_ROLLED_MAX_SEGMENT_JOINT_DELTA_RAD):
                    raise ValueError(
                        "side-column rolled squeeze/retreat changed one "
                        f"joint by {max(squeeze_delta, retreat_delta):.3f}rad")
        except ValueError as exc:
            self.get_logger().error(
                f"[dual-tissue-squeeze] IK failed: {exc}")
            return False

        self.dual_clamp_left_joints = clamp_left_joints.copy()
        self.dual_clamp_right_joints = clamp_right_joints.copy()
        self.dual_retreat_left_joints = retreat_left_joints.copy()
        self.dual_retreat_right_joints = retreat_right_joints.copy()
        # 手指大面（mesh x 面）在 TCP 纸盒侧 31mm 处；大面间距与过盈量
        face_gap = 2.0 * (final_half_span - 0.031)
        interference = 0.086 - (final_half_span - 0.031)
        close_path = max(
            0.0,
            (self.dual_pregrasp_half_span
             if measured_left is None or measured_right is None
             else 0.5 * (measured_right[0] - measured_left[0]))
            - final_half_span)
        self.get_logger().info(
            f"[dual-tissue-squeeze] "
            f"{'rolled-three-point' if self.dual_side_rolled else 'hand-side'} "
            f"clamp span="
            f"{final_half_span:.3f}m/side "
            f"big_face_gap={face_gap * 1000:.0f}mm "
            f"interference≈{interference * 1000:.0f}mm/side "
            f"close_path={close_path:.3f}m "
            f"box_centre={box_centre_x:.4f} "
            f"left={np.round(squeeze_left, 4)} "
            f"right={np.round(squeeze_right, 4)} "
            f"retreat_y={retreat_y:.3f}")
        self.start_dual_tissue_motion(
            "squeeze", squeeze_left_joints, squeeze_right_joints,
            close_path,
            DUAL_TISSUE_SQUEEZE_SPEED_MPS,
            STATE_DUAL_SQUEEZE)
        return True

    def advance_dual_tissue_motion(self) -> str:
        """Play a monotonic dual-arm segment without corrective oscillation."""
        elapsed = self.now() - self.state_t0
        duration = max(self.dual_motion_duration_s, 1e-6)
        progress = float(np.clip(elapsed / duration, 0.0, 1.0))
        eased = progress * progress * (3.0 - 2.0 * progress)
        if self.dual_motion_path_distances is not None:
            path_distance = eased * self.dual_motion_path_distances[-1]
            segment = int(np.searchsorted(
                self.dual_motion_path_distances,
                path_distance, side="right") - 1)
            segment = max(0, min(
                segment, len(self.dual_motion_path_distances) - 2))
            segment_start = self.dual_motion_path_distances[segment]
            segment_end = self.dual_motion_path_distances[segment + 1]
            segment_progress = float(np.clip(
                (path_distance - segment_start)
                / max(segment_end - segment_start, 1e-6), 0.0, 1.0))
            self.des_left_arm = (
                self.dual_motion_path_left[segment]
                + segment_progress
                * (self.dual_motion_path_left[segment + 1]
                   - self.dual_motion_path_left[segment]))
            self.des_right_arm = (
                self.dual_motion_path_right[segment]
                + segment_progress
                * (self.dual_motion_path_right[segment + 1]
                   - self.dual_motion_path_right[segment]))
        else:
            self.des_left_arm = (
                self.dual_motion_start_left
                + eased * (self.dual_motion_target_left
                           - self.dual_motion_start_left))
            self.des_right_arm = (
                self.dual_motion_start_right
                + eased * (self.dual_motion_target_right
                           - self.dual_motion_start_right))
        if elapsed < duration:
            return "moving"

        settle_elapsed = elapsed - duration
        arm_error = self.dual_arm_error()
        left_tcp = self.arm_tcp_world("left")
        right_tcp = self.arm_tcp_world("right")
        left_target_tcp = self.arm_target_tcp_world(
            "left", self.dual_motion_target_left)
        right_target_tcp = self.arm_target_tcp_world(
            "right", self.dual_motion_target_right)
        left_tcp_error = (
            float("inf")
            if left_tcp is None or left_target_tcp is None
            else float(np.max(np.abs(left_tcp - left_target_tcp))))
        right_tcp_error = (
            float("inf")
            if right_tcp is None or right_target_tcp is None
            else float(np.max(np.abs(right_tcp - right_target_tcp))))
        joint_endpoint_ready = (
            arm_error <= ARM_REACHED_TOLERANCE_RAD + 0.015)
        tcp_endpoint_ready = (
            left_tcp_error <= DUAL_TISSUE_ENDPOINT_TCP_TOLERANCE_M
            and right_tcp_error <= DUAL_TISSUE_ENDPOINT_TCP_TOLERANCE_M)
        # Wrist joints can retain a small redundant-angle error even when the
        # complete gripper pose is already at the commanded Cartesian target.
        # Accept either representation, while a genuinely blocked arm still
        # misses its TCP target by several centimetres and fails this gate.
        endpoint_ready = joint_endpoint_ready or tcp_endpoint_ready
        if self.dual_motion_label.startswith("arm_lift_"):
            lift_height_ready = (
                left_tcp is not None and right_tcp is not None
                and left_target_tcp is not None
                and right_target_tcp is not None
                and abs(float(left_tcp[2] - left_target_tcp[2]))
                <= DUAL_TISSUE_ARM_LIFT_Z_TOLERANCE_M
                and abs(float(right_tcp[2] - right_target_tcp[2]))
                <= DUAL_TISSUE_ARM_LIFT_Z_TOLERANCE_M)
            endpoint_ready = endpoint_ready and lift_height_ready
        elif self.dual_motion_label.startswith("raised_retreat_"):
            # Lateral position error is useful clamp preload.  Gate the loaded
            # retreat on depth and height instead of rejecting that preload.
            endpoint_ready = (
                left_tcp is not None and right_tcp is not None
                and left_target_tcp is not None
                and right_target_tcp is not None
                and abs(float(left_tcp[1] - left_target_tcp[1]))
                <= DUAL_TISSUE_ENDPOINT_TCP_TOLERANCE_M
                and abs(float(right_tcp[1] - right_target_tcp[1]))
                <= DUAL_TISSUE_ENDPOINT_TCP_TOLERANCE_M
                and abs(float(left_tcp[2] - left_target_tcp[2]))
                <= DUAL_TISSUE_ENDPOINT_TCP_TOLERANCE_M
                and abs(float(right_tcp[2] - right_target_tcp[2]))
                <= DUAL_TISSUE_ENDPOINT_TCP_TOLERANCE_M)
        now = self.now()
        if endpoint_ready:
            if self.dual_motion_endpoint_ready_since is None:
                self.dual_motion_endpoint_ready_since = now
        else:
            self.dual_motion_endpoint_ready_since = None
        endpoint_stable = (
            self.dual_motion_endpoint_ready_since is not None
            and now - self.dual_motion_endpoint_ready_since
            >= MOTION_ENDPOINT_STABILITY_S)
        if (elapsed < duration + DUAL_TISSUE_MOTION_SETTLE_S
                and (settle_elapsed < DUAL_TISSUE_MOTION_MIN_SETTLE_S
                     or not endpoint_stable)):
            return "moving"

        if (self.dual_motion_require_convergence
                and not endpoint_stable):
            # 顶层抬升撤退：盒子已横向移出货架前缘时，残余关节误差只是
            # 夹持预压/负载的静态偏差，不代表撤退未完成。实测 0.030 速下
            # 差 0.07 rad 即被误判中止丢盒；此时双侧 TCP y 已 < clear_y，
            # 直接按完成放行，避免打开夹爪。
            retreat_clear = False
            if self.dual_motion_label.startswith("raised_retreat_"):
                clear_y = (
                    self.target_world[1]
                    - PRODUCT_BEHIND_MARKER_M
                    - GENERIC_RETREAT_CLEAR_MARGIN_M)
                retreat_clear = (
                    left_tcp is not None
                    and right_tcp is not None
                    and left_tcp[1] < clear_y
                    and right_tcp[1] < clear_y)
            if not retreat_clear:
                left_error_vector = (
                    self.arm_positions("left") - self.des_left_arm)
                right_error_vector = (
                    self.arm_positions("right") - self.des_right_arm)
                left_error = float(np.max(np.abs(left_error_vector)))
                right_error = float(np.max(np.abs(right_error_vector)))
                self.get_logger().error(
                    f"[dual-tissue-{self.dual_motion_label}] endpoint did "
                    f"not converge; elapsed={elapsed:.2f}s "
                    f"left_arm_error={left_error:.4f}rad "
                    f"right_arm_error={right_error:.4f}rad "
                    f"left_tcp_error={left_tcp_error:.4f}m "
                    f"right_tcp_error={right_tcp_error:.4f}m "
                    f"left_joint_errors={np.round(left_error_vector, 4)} "
                    f"right_joint_errors={np.round(right_error_vector, 4)} "
                    f"left_tcp={None if left_tcp is None else np.round(left_tcp, 4)} "
                    f"right_tcp={None if right_tcp is None else np.round(right_tcp, 4)}; "
                    "aborting before contact search")
                self.set_state(STATE_ABORT)
                return "failed"
            self.get_logger().warn(
                f"[dual-tissue-{self.dual_motion_label}] endpoint did "
                f"not converge but both TCPs are clear of the shelf "
                f"(left_y={left_tcp[1]:.3f} right_y={right_tcp[1]:.3f} < "
                f"{clear_y:.3f}); treating the raised retreat as complete")
        self.get_logger().info(
            f"[dual-tissue-{self.dual_motion_label}] segment complete; "
            f"elapsed={elapsed:.2f}s "
            f"settle={settle_elapsed:.2f}s "
            f"arm_error={arm_error:.4f}rad "
            f"tcp_errors=({left_tcp_error:.4f},{right_tcp_error:.4f})m "
            f"endpoint_stable={int(endpoint_stable)} "
            f"left_tcp={None if left_tcp is None else np.round(left_tcp, 4)} "
            f"right_tcp={None if right_tcp is None else np.round(right_tcp, 4)}; "
            f"convergence_gate={int(self.dual_motion_require_convergence)}")
        return "reached"

    def advance_generic_forward(self) -> tuple[str, np.ndarray | None]:
        """Play one slow fixed trajectory, then close without pose gates."""
        elapsed = self.now() - self.state_t0
        duration = max(self.generic_direct_duration_s, 1e-6)
        progress = float(np.clip(elapsed / duration, 0.0, 1.0))
        # Cubic smoothstep is monotonic with zero endpoint velocity.  It avoids
        # striking the product without any feedback correction or stop/go gate.
        eased = progress * progress * (3.0 - 2.0 * progress)
        joints = (
            self.generic_direct_start_joints
            + eased * (self.generic_direct_contact_joints
                       - self.generic_direct_start_joints))
        self.set_selected_arm_target(joints)

        if elapsed < duration:
            return "moving", None

        settle_elapsed = elapsed - duration
        arm_error = self.selected_arm_error()
        endpoint_ready = (
            arm_error <= ARM_REACHED_TOLERANCE_RAD + 0.015)
        now = self.now()
        if endpoint_ready:
            if self.generic_direct_endpoint_ready_since is None:
                self.generic_direct_endpoint_ready_since = now
        else:
            self.generic_direct_endpoint_ready_since = None
        endpoint_stable = (
            self.generic_direct_endpoint_ready_since is not None
            and now - self.generic_direct_endpoint_ready_since
            >= MOTION_ENDPOINT_STABILITY_S)
        if (elapsed < duration + GENERIC_DIRECT_FORWARD_SETTLE_S
                and (settle_elapsed < GENERIC_DIRECT_FORWARD_MIN_SETTLE_S
                     or not endpoint_stable)):
            return "moving", None

        actual_tcp = self.selected_tcp_world()
        error = (
            None if actual_tcp is None
            else actual_tcp - self.forward_contact_world)
        next_action = (
            "keeping gripper open for 50 mm continuation"
            if self.object_geometry != "sphere"
            else "closing without a TCP gate")
        self.get_logger().info(
            f"[generic-direct] fixed slow trajectory complete; "
            f"{next_action}. elapsed={elapsed:.2f}s "
            f"settle={settle_elapsed:.2f}s "
            f"arm_error={arm_error:.4f}rad "
            f"endpoint_stable={int(endpoint_stable)} "
            f"actual={None if actual_tcp is None else np.round(actual_tcp, 4)} "
            f"diagnostic_error="
            f"{None if error is None else np.round(error, 4)}m")
        return "reached", error

    def start_post_extension(self) -> None:
        """Continue beyond a profile's established close point while open."""
        if (self.post_extend_arm_joints is None
                or self.post_extend_nominal_world is None
                or self.post_extend_target_world is None):
            self.get_logger().error(
                f"{self.shelf_level} post-contact extension has no solved "
                "endpoint")
            self.set_state(STATE_ABORT)
            return
        self.post_extend_start_joints = (
            self.selected_arm_positions().copy())
        extension_length = float(np.linalg.norm(
            self.post_extend_target_world
            - self.post_extend_nominal_world))
        self.post_extend_duration_s = max(
            GENERIC_DIRECT_FORWARD_MIN_DURATION_S,
            1.5 * extension_length / GENERIC_DIRECT_FORWARD_SPEED_MPS)
        self.post_extend_endpoint_ready_since = None
        # Snap the diagnostic target to the second endpoint before entering
        # the state.  The arm command itself starts from its measured joints,
        # so there is no command discontinuity at the old close point.
        self.forward_contact_world = (
            self.post_extend_target_world.copy())
        self.set_selected_arm_target(
            self.post_extend_start_joints)
        self.get_logger().info(
            f"[{self.shelf_level}-post-extend] keeping gripper open at "
            f"nominal_close={np.round(self.post_extend_nominal_world, 4)}; "
            f"continuing to extended_close="
            f"{np.round(self.post_extend_target_world, 4)} "
            f"distance={extension_length:.3f}m "
            f"duration={self.post_extend_duration_s:.2f}s "
            f"speed={GENERIC_DIRECT_FORWARD_SPEED_MPS:.3f}m/s "
            "orientation=unchanged feedback_gates=0 replanning=0")
        self.set_state(STATE_POST_EXTEND)

    def prepare_sphere_post_extension(self) -> bool:
        """Build the sphere's 50 mm endpoint from its measured close pose.

        Physical sphere contact may end the existing creep before its nominal
        Cartesian goal.  Starting from the measured TCP guarantees that the
        requested continuation is 50 mm from the point where this run would
        previously have closed, while preserving measured X/Z and orientation.
        A one-shot IK failure falls back to closing at the established point;
        it never enters a retry/correction loop.
        """
        actual_tcp = self.selected_tcp_world()
        if actual_tcp is None:
            self.get_logger().warn(
                f"[{self.shelf_level}-sphere-post-extend] measured TCP "
                "unavailable; closing at the established sphere point")
            return False
        measured_joints = self.selected_arm_positions().copy()
        extended_world = actual_tcp.copy()
        extended_world[1] += GENERIC_POST_CONTACT_EXTENSION_M
        try:
            extended_joints = self.solve_kdl_world(
                extended_world, measured_joints)
        except ValueError as exc:
            self.get_logger().warn(
                f"[{self.shelf_level}-sphere-post-extend] one-shot endpoint "
                f"IK failed ({exc}); closing at the established sphere point")
            return False

        self.post_extend_nominal_world = actual_tcp.copy()
        self.post_extend_target_world = extended_world.copy()
        self.post_extend_arm_joints = extended_joints.copy()
        self.get_logger().info(
            f"[{self.shelf_level}-sphere-post-extend] prepared from measured "
            f"close point={np.round(actual_tcp, 4)} to "
            f"extended_close={np.round(extended_world, 4)} "
            f"distance={GENERIC_POST_CONTACT_EXTENSION_M:.3f}m; "
            "gripper remains open")
        return True

    def advance_post_extension(self) -> tuple[str, np.ndarray | None]:
        """Play a profile's 50 mm extension, then permit closure."""
        elapsed = self.now() - self.state_t0
        duration = max(self.post_extend_duration_s, 1e-6)
        progress = float(np.clip(elapsed / duration, 0.0, 1.0))
        eased = progress * progress * (3.0 - 2.0 * progress)
        joints = (
            self.post_extend_start_joints
            + eased * (self.post_extend_arm_joints
                       - self.post_extend_start_joints))
        self.set_selected_arm_target(joints)
        if elapsed < duration:
            return "moving", None

        settle_elapsed = elapsed - duration
        arm_error = self.selected_arm_error()
        endpoint_ready = (
            arm_error <= ARM_REACHED_TOLERANCE_RAD + 0.015)
        now = self.now()
        if endpoint_ready:
            if self.post_extend_endpoint_ready_since is None:
                self.post_extend_endpoint_ready_since = now
        else:
            self.post_extend_endpoint_ready_since = None
        endpoint_stable = (
            self.post_extend_endpoint_ready_since is not None
            and now - self.post_extend_endpoint_ready_since
            >= MOTION_ENDPOINT_STABILITY_S)
        if (elapsed < duration + GENERIC_DIRECT_FORWARD_SETTLE_S
                and (settle_elapsed < GENERIC_DIRECT_FORWARD_MIN_SETTLE_S
                     or not endpoint_stable)):
            return "moving", None

        actual_tcp = self.selected_tcp_world()
        error = (
            None if actual_tcp is None
            else actual_tcp - self.post_extend_target_world)
        self.get_logger().info(
            f"[{self.shelf_level}-post-extend] continuation complete; "
            f"closing "
            f"without a TCP gate. elapsed={elapsed:.2f}s "
            f"settle={settle_elapsed:.2f}s "
            f"arm_error={arm_error:.4f}rad "
            f"endpoint_stable={int(endpoint_stable)} "
            f"actual="
            f"{None if actual_tcp is None else np.round(actual_tcp, 4)} "
            f"diagnostic_error="
            f"{None if error is None else np.round(error, 4)}m")
        return "reached", error

    def configure_generic_top_lift_from_measured(self) -> bool:
        """Build a small real top lift and a raised horizontal retreat."""
        actual_tcp = self.selected_tcp_world()
        if actual_tcp is None:
            self.get_logger().error(
                "[generic-top-lift] measured TCP unavailable")
            return False
        measured_joints = self.selected_arm_positions()
        lift_world = actual_tcp.copy()
        lift_world[2] += GENERIC_TOP_LIFT_M
        retreat_world = lift_world.copy()
        retreat_world[1] = self.generic_forward_start_world[1]
        try:
            lift_joints = self.solve_kdl_world(
                lift_world, measured_joints)
            retreat_joints = self.solve_kdl_world(
                retreat_world, lift_joints)
        except ValueError as exc:
            self.get_logger().error(
                f"[generic-top-lift] IK failed: {exc}")
            return False
        self.generic_top_lift_arm_joints = lift_joints.copy()
        self.generic_top_retreat_arm_joints = retreat_joints.copy()
        self.set_selected_arm_target(lift_joints)
        self.get_logger().info(
            f"[generic-top-lift] configured from measured grasp; "
            f"start={np.round(actual_tcp, 4)} "
            f"lift={np.round(lift_world, 4)} "
            f"retreat={np.round(retreat_world, 4)} "
            f"lift_amount={GENERIC_TOP_LIFT_M:.3f}m")
        return True

    def advance_sphere_contact_creep(
            self) -> tuple[str, np.ndarray | None]:
        """Continue from first surface contact to the geometric grasp point."""
        creep_path = (
            self.sphere_creep_goal_world - self.sphere_creep_start_world)
        creep_length = float(np.linalg.norm(creep_path))
        creep_elapsed = self.now() - self.sphere_creep_started_at
        progress = min(
            1.0,
            creep_elapsed * SPHERE_CONTACT_CREEP_SPEED_MPS
            / max(creep_length, 1e-9))
        if progress > self.sphere_forward_last_progress + 1e-6:
            waypoint = self.sphere_creep_start_world + progress * creep_path
            try:
                joints = self.solve_kdl_world(
                    waypoint, self.sphere_forward_reference)
            except ValueError as exc:
                self.get_logger().error(
                    f"[sphere-creep] trajectory IK failed at "
                    f"progress={progress:.3f}: {exc}")
                return "failed", None
            self.sphere_forward_reference = joints.copy()
            self.sphere_forward_last_progress = progress
            self.set_selected_arm_target(joints)

        actual_tcp = self.selected_tcp_world()
        if actual_tcp is None:
            return "moving", None
        error = actual_tcp - self.sphere_creep_goal_world
        sphere_error = actual_tcp - self.sphere_contact_world
        creep_unit = creep_path / max(creep_length, 1e-9)
        actual_advance = float(np.dot(
            actual_tcp - self.sphere_creep_start_world, creep_unit))
        now = self.now()
        self.sphere_creep_progress_samples.append((now, actual_advance))
        cutoff = now - SPHERE_CREEP_STALL_WINDOW_S
        while (self.sphere_creep_progress_samples
               and self.sphere_creep_progress_samples[0][0] < cutoff):
            self.sphere_creep_progress_samples.popleft()
        stall_span = (
            0.0 if len(self.sphere_creep_progress_samples) < 2
            else self.sphere_creep_progress_samples[-1][0]
            - self.sphere_creep_progress_samples[0][0])
        stall_range = (
            0.0 if not self.sphere_creep_progress_samples
            else max(item[1] for item in self.sphere_creep_progress_samples)
            - min(item[1] for item in self.sphere_creep_progress_samples))
        longitudinal_stall = (
            stall_span >= SPHERE_CREEP_STALL_MIN_SPAN_S
            and stall_range <= SPHERE_CREEP_STALL_RANGE_M)

        gripper = self.selected_gripper_position()
        grip_drop = (
            0.0 if gripper is None
            or self.sphere_open_grip_reference is None
            else self.sphere_open_grip_reference - gripper)
        constrained_contact = (
            actual_advance >= SPHERE_CREEP_MIN_ADVANCE_M
            and grip_drop >= SPHERE_CREEP_CAPTURE_GRIP_DROP
            and np.all(
                np.abs(sphere_error) <= SPHERE_CREEP_CONTACT_TOLERANCE_M)
            and longitudinal_stall)
        if constrained_contact:
            # The rigid Cartesian endpoint can be unreachable once the sphere
            # constrains the fingers.  Use four independent physical signals
            # instead: meaningful entry beyond first touch, finger deflection,
            # bounded transverse/vertical error, and sustained lack of further
            # inward motion.  This avoids both premature closing and endless
            # pushing against an already-engaged product.
            self.set_selected_arm_target(self.selected_arm_positions())
            self.get_logger().info(
                f"[sphere-creep] constrained grasp contact accepted; "
                f"actual={np.round(actual_tcp, 4)} "
                f"sphere_error={np.round(sphere_error, 4)}m "
                f"advance={actual_advance:.4f}m "
                f"grip={gripper:.3f} drop={grip_drop:.3f} "
                f"stall_range={stall_range:.4f}m over "
                f"{stall_span:.2f}s; closing gripper")
            return "reached", error
        goal_tolerance = (
            MIDDLE_SPHERE_CREEP_GOAL_TOLERANCE_M
            if self.shelf_level == "middle"
            else SPHERE_TCP_TOLERANCE_M)
        if (progress >= 1.0
                and np.all(np.abs(error) <= goal_tolerance)):
            self.set_selected_arm_target(self.selected_arm_positions())
            self.get_logger().info(
                f"[sphere-creep] geometric grasp point reached; "
                f"actual={np.round(actual_tcp, 4)} "
                f"error={np.round(error, 4)}m "
                f"elapsed={creep_elapsed:.2f}s; closing gripper")
            return "reached", error
        if creep_elapsed >= SPHERE_CONTACT_CREEP_TIMEOUT_S:
            self.get_logger().error(
                f"[sphere-creep] grasp point not reached within "
                f"{creep_elapsed:.2f}s; actual={np.round(actual_tcp, 4)} "
                f"target={np.round(self.sphere_creep_goal_world, 4)} "
                f"error={np.round(error, 4)}m; aborting without pushing")
            return "failed", error
        return "moving", error

    def advance_sphere_forward(self) -> tuple[str, np.ndarray | None]:
        """Stream a fast approach, then creep after first sphere contact."""
        if self.sphere_creep_started_at is not None:
            return self.advance_sphere_contact_creep()

        path = self.sphere_contact_world - self.sphere_pregrasp_world
        path_length = float(np.linalg.norm(path))
        elapsed = self.now() - self.state_t0
        terminal_distance = min(SPHERE_TERMINAL_ZONE_M, path_length)
        fast_distance = path_length - terminal_distance
        fast_duration = fast_distance / SPHERE_FAST_SPEED_MPS
        if elapsed <= fast_duration:
            commanded_distance = elapsed * SPHERE_FAST_SPEED_MPS
        else:
            commanded_distance = (
                fast_distance
                + (elapsed - fast_duration) * SPHERE_TERMINAL_SPEED_MPS)
        progress = min(1.0, commanded_distance / max(path_length, 1e-9))
        if progress > self.sphere_forward_last_progress + 1e-6:
            waypoint = self.sphere_pregrasp_world + progress * path
            try:
                joints = self.solve_kdl_world(
                    waypoint, self.sphere_forward_reference)
            except ValueError as exc:
                self.get_logger().error(
                    f"[sphere-forward] trajectory IK failed at "
                    f"progress={progress:.3f}: {exc}")
                return "failed", None
            self.sphere_forward_reference = joints.copy()
            self.sphere_forward_last_progress = progress
            self.set_selected_arm_target(joints)

        actual_tcp = self.selected_tcp_world()
        if actual_tcp is None:
            return "moving", None
        error = actual_tcp - self.sphere_contact_world
        path_unit = path / max(path_length, 1e-9)
        actual_travel = float(np.dot(
            actual_tcp - self.sphere_pregrasp_world, path_unit))
        in_terminal_zone = actual_travel >= fast_distance

        gripper = self.selected_gripper_position()
        if gripper is not None:
            if self.sphere_open_grip_reference is None:
                self.sphere_open_grip_reference = gripper
            else:
                # Open-loop tracking can overshoot slightly above GRIP_OPEN.
                # Retain the largest observed open value so a subsequent
                # inward finger deflection remains visible as product contact.
                self.sphere_open_grip_reference = max(
                    self.sphere_open_grip_reference, gripper)
        grip_drop = (
            0.0 if gripper is None
            or self.sphere_open_grip_reference is None
            else self.sphere_open_grip_reference - gripper)
        if (in_terminal_zone
                and grip_drop >= SPHERE_OPEN_GRIP_DROP_CONTACT):
            # A single open-finger deflection proves only that the curved front
            # surface has been touched.  Freeze the old trajectory and begin a
            # fresh, measured 20 mm/s creep so the sphere reaches the useful
            # region between both fingers before they close.
            measured_joints = self.selected_arm_positions()
            self.sphere_creep_start_world = actual_tcp.copy()
            self.sphere_creep_goal_world = self.sphere_contact_world.copy()
            if self.shelf_level == "middle":
                # The middle-layer slide has already corrected height before
                # approach.  Once physical contact exists, preserve measured
                # lateral/vertical coordinates and advance only toward the
                # shelf so contact cannot redirect a 3-D correction sideways.
                self.sphere_creep_goal_world[0] = actual_tcp[0]
                self.sphere_creep_goal_world[2] = actual_tcp[2]
            self.sphere_creep_started_at = self.now()
            self.sphere_forward_reference = measured_joints.copy()
            self.sphere_forward_last_progress = -1.0
            self.sphere_creep_progress_samples.clear()
            self.set_selected_arm_target(measured_joints)
            self.get_logger().info(
                f"[sphere-forward] finger contact detected; "
                f"actual={np.round(actual_tcp, 4)} "
                f"travel={actual_travel:.4f}m "
                f"grip={gripper:.3f} "
                f"open_ref={self.sphere_open_grip_reference:.3f} "
                f"drop={grip_drop:.3f} elapsed={elapsed:.2f}s; "
                f"switching to {SPHERE_CONTACT_CREEP_SPEED_MPS:.3f}m/s "
                f"{'Y-only ' if self.shelf_level == 'middle' else ''}"
                "measured contact creep")
            return "moving", error
        if (progress >= 1.0
                and np.all(np.abs(error) <= SPHERE_TCP_TOLERANCE_M)):
            # Stop arm drive at the measured pose before the gripper starts to
            # close; no elapsed-time assumption is used for spheres.
            self.set_selected_arm_target(self.selected_arm_positions())
            self.get_logger().info(
                f"[sphere-forward] measured TCP reached contact; "
                f"actual={np.round(actual_tcp, 4)} "
                f"error={np.round(error, 4)}m grip_drop={grip_drop:.3f} "
                f"elapsed={elapsed:.2f}s")
            return "reached", error
        if elapsed >= SPHERE_FORWARD_TIMEOUT_S:
            self.get_logger().error(
                f"[sphere-forward] measured TCP did not reach contact in "
                f"{elapsed:.2f}s; actual={np.round(actual_tcp, 4)} "
                f"target={np.round(self.sphere_contact_world, 4)} "
                f"error={np.round(error, 4)}m "
                f"grip_drop={grip_drop:.3f}")
            return "failed", error
        return "moving", error

    def configure_sphere_lift_from_measured(self) -> bool:
        """Dispatch sphere retention/lift to the selected shelf layer."""
        if self.shelf_level == "top":
            return self.configure_top_sphere_lift_from_measured()
        if self.shelf_level == "middle":
            return self.configure_middle_sphere_lift_from_measured()
        self.get_logger().error(
            f"sphere lift is not implemented for layer={self.shelf_level}")
        return False

    def configure_top_sphere_lift_from_measured(self) -> bool:
        """Use arm IK because the top-shelf slide is already at its limit."""
        actual_tcp = self.selected_tcp_world()
        if actual_tcp is None:
            self.get_logger().error(
                "[sphere-lift] measured TCP unavailable")
            return False
        reference = self.selected_arm_positions()
        trial_world = actual_tcp.copy()
        trial_world[2] += SPHERE_TRIAL_LIFT_M
        lift_world = actual_tcp.copy()
        lift_world[2] += SPHERE_LIFT_M
        retreat_world = lift_world.copy()
        retreat_world[1] = self.sphere_pregrasp_world[1]
        try:
            trial_joints = self.solve_kdl_world(trial_world, reference)
            lift_joints = self.solve_kdl_world(lift_world, trial_joints)
            retreat_joints = self.solve_kdl_world(
                retreat_world, lift_joints)
        except ValueError as exc:
            self.get_logger().error(f"[sphere-lift] IK failed: {exc}")
            return False
        self.sphere_trial_lift_arm_joints = trial_joints.copy()
        self.sphere_lift_arm_joints = lift_joints.copy()
        self.sphere_retreat_arm_joints = retreat_joints.copy()
        self.sphere_trial_grip_samples.clear()
        self.set_selected_arm_target(trial_joints)
        self.get_logger().info(
            f"[top-sphere-lift] actual_contact={np.round(actual_tcp, 4)} "
            f"trial={np.round(trial_world, 4)} "
            f"lift={np.round(lift_world, 4)} "
            f"raised_retreat={np.round(retreat_world, 4)}")
        return True

    def configure_middle_sphere_lift_from_measured(self) -> bool:
        """Hold the grasp pose and use the slide for middle-shelf lifting."""
        actual_tcp = self.selected_tcp_world()
        if actual_tcp is None:
            self.get_logger().error(
                "[middle-sphere-lift] measured TCP unavailable")
            return False
        measured_joints = self.selected_arm_positions()
        self.sphere_trial_slide = max(
            SLIDE_MIN,
            self.sphere_slide_command - SPHERE_TRIAL_LIFT_M)
        self.sphere_lift_slide = max(
            SLIDE_MIN,
            self.sphere_slide_command - SPHERE_LIFT_M)
        self.sphere_trial_lift_arm_joints = measured_joints.copy()
        self.sphere_lift_arm_joints = measured_joints.copy()
        self.sphere_retreat_arm_joints = self.pregrasp_arm_joints.copy()
        self.sphere_trial_grip_samples.clear()
        self.set_selected_arm_target(measured_joints)
        self.des_slide = self.sphere_trial_slide
        self.get_logger().info(
            f"[middle-sphere-lift] actual_contact="
            f"{np.round(actual_tcp, 4)} "
            f"trial_slide={self.sphere_trial_slide:.3f} "
            f"full_slide={self.sphere_lift_slide:.3f} "
            f"raised_retreat_y={self.sphere_pregrasp_world[1]:.3f}")
        return True

    def sphere_capture_minimum(self) -> float:
        return SPHERE_GRIP_MIN_CAPTURE_POSITION[self.target_kind]

    def sphere_grip_is_stable(
            self, samples: deque, gripper: float) -> tuple[bool, float, float]:
        """Record a rolling grip window and report its span and range."""
        now = self.now()
        samples.append((now, gripper))
        cutoff = now - SPHERE_GRIP_STABILITY_WINDOW_S
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        span = 0.0 if len(samples) < 2 else samples[-1][0] - samples[0][0]
        spread = 0.0 if not samples else (
            max(item[1] for item in samples)
            - min(item[1] for item in samples))
        stable = (
            span >= SPHERE_GRIP_STABILITY_MIN_SPAN_S
            and spread <= SPHERE_GRIP_STABILITY_RANGE)
        return stable, spread, span

    @staticmethod
    def grip_range(samples: deque, window_s: float) -> float:
        """Max-minus-min of the most recent ~window_s of gripper samples."""
        n = max(2, int(round(window_s / 0.02)))
        recent = list(samples)[-n:]
        if len(recent) < 2:
            return float("inf")
        return float(max(recent) - min(recent))

    def selected_arm_positions(self) -> np.ndarray:
        return np.array([
            self.joints.get(f"{self.grasp_arm == 'l' and 'left' or 'right'}_arm_joint{i + 1}",
                            float("inf"))
            for i in range(6)])

    def arm_positions(self, side: str) -> np.ndarray:
        """Return measured joint positions for one explicitly named arm."""
        prefix = "left" if side == "left" else "right"
        return np.array([
            self.joints.get(
                f"{prefix}_arm_joint{i + 1}", float("inf"))
            for i in range(6)])

    def dual_arm_error(self) -> float:
        """Return the largest desired/measured error across both arms."""
        left_error = np.max(np.abs(
            self.arm_positions("left") - self.des_left_arm))
        right_error = np.max(np.abs(
            self.arm_positions("right") - self.des_right_arm))
        return float(max(left_error, right_error))

    def dual_commands_ready(
            self, arm_tolerance: float = ARM_REACHED_TOLERANCE_RAD,
            slide_tolerance: float = 0.025) -> bool:
        """Diagnostic settling check for both arms and the slide."""
        slide = self.joints.get("slide_joint")
        if slide is None:
            self.commands_ready_since = None
            return False
        within_tolerance = (
            abs(slide - self.des_slide) < slide_tolerance
            and self.dual_arm_error() < arm_tolerance)
        if not within_tolerance:
            self.commands_ready_since = None
            return False
        if self.commands_ready_since is None:
            self.commands_ready_since = self.now()
            return False
        return self.now() - self.commands_ready_since >= ARM_READY_SETTLE_S

    def selected_arm_error(self) -> float:
        measured = self.selected_arm_positions()
        desired = self.des_left_arm if self.grasp_arm == "l" else self.des_right_arm
        return float(np.max(np.abs(measured - desired)))

    def commands_ready(
            self,
            arm_tolerance: float = ARM_REACHED_TOLERANCE_RAD,
            slide_tolerance: float = 0.025) -> bool:
        slide = self.joints.get("slide_joint")
        if slide is None:
            self.commands_ready_since = None
            return False
        within_tolerance = (
            abs(slide - self.des_slide) < slide_tolerance
            and self.selected_arm_error() < arm_tolerance)
        if not within_tolerance:
            self.commands_ready_since = None
            return False
        if self.commands_ready_since is None:
            self.commands_ready_since = self.now()
            return False
        return self.now() - self.commands_ready_since >= ARM_READY_SETTLE_S

    @staticmethod
    def slew(current, desired, maximum_step):
        delta = np.clip(np.asarray(desired) - np.asarray(current),
                        -maximum_step, maximum_step)
        return np.asarray(current) + delta

    @staticmethod
    def synchronized_slew(current, desired, maximum_step):
        """Advance every joint by the same fraction of its remaining path.

        Per-joint clipping makes small-motion joints arrive early while the
        largest-motion joint keeps moving.  Scaling the complete delta vector
        preserves the joint-space path and makes all six commanded joints
        arrive together without reducing the previous peak step.
        """
        current = np.asarray(current, dtype=float)
        delta = np.asarray(desired, dtype=float) - current
        largest_delta = float(np.max(np.abs(delta)))
        if largest_delta <= maximum_step:
            return current + delta
        return current + delta * (maximum_step / largest_delta)

    def forward_arm_step_limit(self) -> float:
        """Smoothly reduce the selected arm command rate near every object."""
        if (self.state != STATE_ARM_FORWARD
                or self.forward_contact_world is None):
            return ARM_COMMAND_MAX_STEP_RAD
        if not self.use_sphere_grasp:
            # Generic speed is already imposed by the fixed eased trajectory.
            # Do not read TCP feedback here or introduce another terminal
            # slowdown/correction loop.
            return ARM_COMMAND_MAX_STEP_RAD
        actual_tcp = self.selected_tcp_world()
        if actual_tcp is None:
            return ARM_COMMAND_MAX_STEP_RAD
        remaining = float(np.linalg.norm(
            actual_tcp - self.forward_contact_world))
        blend = float(np.clip(
            remaining / FORWARD_TERMINAL_ZONE_M, 0.0, 1.0))
        step = (
            FORWARD_TERMINAL_ARM_STEP_RAD
            + blend * (ARM_COMMAND_MAX_STEP_RAD
                       - FORWARD_TERMINAL_ARM_STEP_RAD))
        if remaining <= FORWARD_TERMINAL_ZONE_M:
            if self.forward_terminal_entered_at is None:
                self.forward_terminal_entered_at = self.now()
            if not self.forward_terminal_slow_logged:
                self.forward_terminal_slow_logged = True
                self.get_logger().info(
                    f"[arm-forward] global terminal slowdown entered; "
                    f"remaining={remaining:.4f}m zone="
                    f"{FORWARD_TERMINAL_ZONE_M:.3f}m "
                    f"joint_step={step:.4f}->"
                    f"{FORWARD_TERMINAL_ARM_STEP_RAD:.4f}rad/tick")
        return step

    def smooth_commands(self) -> None:
        slide_step = (
            DUAL_TISSUE_LIFT_SLIDE_STEP_M
            if (self.use_dual_tissue_grasp
                and self.state == STATE_LIFT)
            else 0.006)
        self.cmd_slide = float(self.slew(
            self.cmd_slide, self.des_slide, slide_step))
        self.cmd_head = self.slew(self.cmd_head, self.des_head, 0.025)
        if self.use_dual_tissue_grasp:
            # Scale one 12-joint vector so neither arm arrives early and
            # pushes the box sideways while the other arm is still moving.
            # The first deployment is outside the shelf and must retain its
            # normal speed; all following rolled-wrist phases are limited to
            # suppress tracking overshoot beside the post and the box.
            dual_step = (
                DUAL_TISSUE_SIDE_ROLLED_MAX_STEP_RAD
                if (getattr(self, "dual_side_rolled", False)
                    and self.state != STATE_DEPLOY)
                else ARM_COMMAND_MAX_STEP_RAD)
            combined = self.synchronized_slew(
                np.concatenate((self.cmd_left_arm, self.cmd_right_arm)),
                np.concatenate((self.des_left_arm, self.des_right_arm)),
                dual_step)
            self.cmd_left_arm = combined[:6]
            self.cmd_right_arm = combined[6:]
        else:
            forward_step = self.forward_arm_step_limit()
            selected_step = (
                RETREAT_ARM_MAX_STEP_RAD
                if self.state == STATE_RETREAT
                else forward_step)
            left_step = (
                selected_step if self.grasp_arm == "l"
                else ARM_COMMAND_MAX_STEP_RAD)
            right_step = (
                selected_step if self.grasp_arm == "r"
                else ARM_COMMAND_MAX_STEP_RAD)
            self.cmd_left_arm = self.synchronized_slew(
                self.cmd_left_arm, self.des_left_arm, left_step)
            self.cmd_right_arm = self.synchronized_slew(
                self.cmd_right_arm, self.des_right_arm, right_step)
        left_grip_step = (
            GRIP_CLOSE_MAX_STEP
            if self.des_left_grip < self.cmd_left_grip
            else GRIP_OPEN_MAX_STEP)
        right_grip_step = (
            GRIP_CLOSE_MAX_STEP
            if self.des_right_grip < self.cmd_right_grip
            else GRIP_OPEN_MAX_STEP)
        self.cmd_left_grip = float(self.slew(
            self.cmd_left_grip, self.des_left_grip, left_grip_step))
        self.cmd_right_grip = float(self.slew(
            self.cmd_right_grip, self.des_right_grip, right_grip_step))

        self.cmd_linear += float(np.clip(
            self.des_linear - self.cmd_linear, -0.03, 0.03))
        self.cmd_angular += float(np.clip(
            self.des_angular - self.cmd_angular, -0.10, 0.10))

    def publish_commands(self) -> None:
        twist = Twist()
        twist.linear.x = self.cmd_linear
        twist.angular.z = self.cmd_angular
        self.cmd_vel_pub.publish(twist)
        self.slide_pub.publish(Float64MultiArray(data=[self.cmd_slide]))
        self.head_pub.publish(Float64MultiArray(data=self.cmd_head.tolist()))
        self.left_pub.publish(Float64MultiArray(
            data=self.cmd_left_arm.tolist() + [self.cmd_left_grip]))
        self.right_pub.publish(Float64MultiArray(
            data=self.cmd_right_arm.tolist() + [self.cmd_right_grip]))

    def tick(self) -> None:
        if self.base_xy is None or not self.joints:
            return
        if not self.initialized:
            self.initialize_commands()

        self.set_twist(0.0, 0.0)

        # Navigation and final base alignment are pre-grasp operations, so a
        # hard wall-clock ceiling can safely route them through STATE_ABORT.
        # Use a monotonic clock: ROS time can jump when the simulator restarts.
        state_elapsed = self.state_elapsed_monotonic()
        if (self.state == STATE_GO_SCAN
                and state_elapsed >= GO_SCAN_HARD_TIMEOUT_S):
            self.get_logger().error(
                "[go-scan] navigation/camera setup exceeded "
                f"{GO_SCAN_HARD_TIMEOUT_S:.0f}s; aborting this attempt")
            self.abort_reason = (
                f"go-scan timeout after {GO_SCAN_HARD_TIMEOUT_S:.0f}s")
            self.set_state(STATE_ABORT)
        elif (self.state == STATE_ALIGN
              and state_elapsed >= ALIGN_HARD_TIMEOUT_S):
            self.get_logger().error(
                "[align] base did not converge within "
                f"{ALIGN_HARD_TIMEOUT_S:.0f}s; aborting this attempt")
            self.abort_reason = (
                f"align timeout after {ALIGN_HARD_TIMEOUT_S:.0f}s")
            self.set_state(STATE_ABORT)

        if self.state == STATE_GO_SCAN:
            pose_name, slide_target, yaw_target, pitch_target = (
                self.current_scan_camera_pose())
            if self.scan_station_order is None:
                self.scan_station_order = self._nearest_scan_stations()
            navigation_ready = self.drive_to(
                [self.current_scan_station_x(), SCAN_Y], YAW_NORTH, 0.08)
            # Keep the camera assembly at the posture inherited when this
            # order started while crossing the delivery/obstacle corridor.
            # Lowering or pitching it during base transit is unnecessary and
            # can put the head in a poor clearance posture.  Only command the
            # selected shelf view after the base has reached and aligned at
            # the scan station; the normal settling gate below then waits for
            # the camera motion to finish before enabling the scan state.
            if navigation_ready:
                self.des_slide = slide_target
                self.des_head[:] = [yaw_target, pitch_target]
            camera_ready = navigation_ready and self.scan_camera_ready()
            if navigation_ready and camera_ready:
                if self.scan_camera_ready_since is None:
                    self.scan_camera_ready_since = self.now()
            else:
                self.scan_camera_ready_since = None
            if (self.scan_camera_ready_since is not None
                    and self.now() - self.scan_camera_ready_since
                    >= SCAN_CAMERA_STABLE_S):
                with self.lock:
                    self.yolo_frames.clear()
                    self.aruco_frames.clear()
                    self.marker_positions.clear()
                    self.depth_target_samples.clear()
                    self.association_candidate_id = None
                    self.association_confirmation_count = 0
                    self.last_association_pair = None
                    self.target_marker_id = None
                    self.target_physical_marker_id = None
                self.get_logger().info(
                    f"[scan-camera] settled pose={pose_name} "
                    f"slide={slide_target:.2f} yaw={yaw_target:+.2f} "
                    f"pitch={pitch_target:+.2f} station_x="
                    f"{self.current_scan_station_x():.3f}")
                self.set_state(STATE_SCAN)

        elif self.state == STATE_SCAN:
            _, slide_target, yaw_target, pitch_target = (
                self.current_scan_camera_pose())
            self.des_slide = slide_target
            self.des_head[:] = [yaw_target, pitch_target]
            if self.now() - self.scan_diag_last_log >= 1.0:
                self.scan_diag_last_log = self.now()
                self.log_scan_perception_summary()
            if self.now() - self.state_t0 > SCAN_DWELL_S:
                if ((self.scan_unlocked_markers or self.scan_unlocked_boxes)
                        and self.revisit_total_rounds
                        < REVISIT_MAX_ROUNDS_PER_SCAN):
                    self._start_revisit()
                if self.state != STATE_REVISIT:
                    self._advance_scan_pose()

        elif self.state == STATE_REVISIT:
            pose = self.revisit_poses[self.revisit_pose_index]
            _, slide_target, yaw_target, pitch_target = pose
            self.des_slide = slide_target
            self.des_head[:] = [yaw_target, pitch_target]
            if self.scan_camera_ready(pose):
                if self.scan_camera_ready_since is None:
                    self.scan_camera_ready_since = self.now()
                    # 驻留从相机真正到位开始计时，姿态切换耗时不计入补拍窗口
                    self.revisit_pose_t0 = self.now()
                if (self.now() - self.scan_camera_ready_since
                        >= SCAN_CAMERA_STABLE_S
                        and self.now() - self.revisit_pose_t0
                        >= (REVISIT_FIRST_POSE_DWELL_S
                            if self.revisit_pose_index == 0
                            else REVISIT_DWELL_S)):
                    self.revisit_pose_index += 1
                    self.scan_camera_ready_since = None
                    self.revisit_pose_monotonic_t0 = time.monotonic()
                    if self.revisit_pose_index >= len(self.revisit_poses):
                        self._revisit_fail()
            else:
                self.scan_camera_ready_since = None
                if (time.monotonic() - self.revisit_pose_monotonic_t0
                        >= REVISIT_POSE_COMMAND_TIMEOUT_S):
                    self.get_logger().warn(
                        "[revisit] camera pose did not converge within "
                        f"{REVISIT_POSE_COMMAND_TIMEOUT_S:.1f}s; skipping pose "
                        f"{self.revisit_pose_index + 1}/{len(self.revisit_poses)}")
                    self.revisit_pose_index += 1
                    self.revisit_pose_monotonic_t0 = time.monotonic()
                    if self.revisit_pose_index >= len(self.revisit_poses):
                        self._revisit_fail()

        elif self.state == STATE_DIRECT_TRANSIT:
            self.advance_direct_transit()

        elif self.state == STATE_ALIGN:
            align_x_tolerance = 0.025
            if self.drive_to(
                    [self.align_base_x, self.align_base_y],
                    YAW_NORTH, align_x_tolerance,
                    linear_min_mps=NAV_ALIGN_LINEAR_MIN_MPS,
                    linear_gain=NAV_ALIGN_LINEAR_GAIN,
                    rotate_gate_rad=NAV_ALIGN_ROTATE_GATE_RAD,
                    translate_angular_max_rps=(
                        NAV_ALIGN_TRANSLATE_ANGULAR_MAX_RADPS)):
                if self.close_recheck and not self._recheck_passed:
                    self.set_state(STATE_RECHECK)
                    self._start_close_recheck()
                else:
                    self._start_grasp_settle()

        elif self.state == STATE_RECHECK:
            pose = self.current_recheck_pose()
            if pose is None:
                self._recheck_fail()
                return
            _, slide_target, yaw_target, pitch_target = pose
            self.des_slide = slide_target
            self.des_head[:] = [yaw_target, pitch_target]
            if self.scan_camera_ready(pose):
                if self.scan_camera_ready_since is None:
                    self.scan_camera_ready_since = self.now()
                if (self.now() - self.scan_camera_ready_since
                        >= SCAN_CAMERA_STABLE_S
                        and self._recheck_confirmed()):
                    self._recheck_passed = True
                    self.recheck_confirmation_times.clear()
                    self.recheck_poses = ()
                    self.get_logger().info(
                        f"[close-recheck] PASS marker="
                        f"{self.target_marker_id} kind={self.target_kind}; "
                        "proceeding to grasp")
                    self._start_grasp_settle()
            else:
                self.scan_camera_ready_since = None

            if (self.state == STATE_RECHECK
                    and self.now() - self.recheck_pose_started_at
                    >= CLOSE_RECHECK_POSE_TIMEOUT_S):
                if not self._advance_recheck_pose():
                    self._recheck_fail()

        elif self.state == STATE_TISSUE_ROTATE:
            self.des_slide = self.slide_grasp
            status = self.advance_dual_tissue_motion()
            if status == "reached":
                if self.tissue_rotate_stage < 2:
                    self.start_tissue_rotate_stage(
                        self.tissue_rotate_stage + 1)
                else:
                    self.tissue_rotated_90 = True
                    self.get_logger().info(
                        "[tissue-rotate] 90-deg planar rotation complete; "
                        "starting dual-arm grasp")
                    self._proceed_to_deploy()
            elif status == "failed":
                self.set_state(STATE_ABORT)

        elif self.state == STATE_GRASP_SETTLE:
            self._grasp_settle_tick()

        elif self.state == STATE_DEPLOY:
            middle_sphere = (
                self.use_sphere_grasp and self.shelf_level == "middle")
            self.des_slide = (
                self.sphere_slide_command
                if middle_sphere else self.slide_grasp)
            if self.use_dual_tissue_grasp:
                self.des_left_grip = DUAL_TISSUE_GRIP_COMMAND
                self.des_right_grip = DUAL_TISSUE_GRIP_COMMAND
            deploy_elapsed = self.now() - self.state_t0
            deploy_ready = (
                False if self.use_dual_tissue_grasp
                else self.commands_ready(
                    MIDDLE_SPHERE_CORRECTION_ARM_TOLERANCE_RAD
                    if middle_sphere else ARM_REACHED_TOLERANCE_RAD,
                    MIDDLE_SPHERE_CORRECTION_SLIDE_TOLERANCE_M
                    if middle_sphere else 0.025))

            actual_tcp = None
            tcp_error = None
            measured_ready = deploy_ready
            soft_continue = False
            hard_continue = False
            if middle_sphere:
                # First let the normal controller settle, then use the TCP
                # measurement both to correct residual height and to decide
                # whether the precise pregrasp has been reached.  Previously
                # this correction was unreachable behind a 0.010 rad / 4 mm
                # command gate even when the visible pose was already usable.
                measured_ready = False
                if deploy_elapsed > 0.8 and deploy_ready:
                    measured_ready = self.middle_sphere_pregrasp_ready()
                actual_tcp = self.selected_tcp_world()
                if actual_tcp is not None:
                    tcp_error = actual_tcp - self.sphere_pregrasp_world
                    soft_continue = (
                        deploy_elapsed
                        >= MIDDLE_SPHERE_DEPLOY_SOFT_CONTINUE_S
                        and np.all(np.abs(tcp_error)
                                   <= MIDDLE_SPHERE_PREGRASP_SOFT_LIMIT_M))
                    hard_continue = (
                        deploy_elapsed >= MIDDLE_SPHERE_DEPLOY_TIMEOUT_S
                        and np.all(np.abs(tcp_error)
                                   <= MIDDLE_SPHERE_PREGRASP_HARD_LIMIT_M))

            continue_experiment = soft_continue or hard_continue
            deploy_gripper = self.selected_gripper_position()
            preshape_ready = (
                deploy_gripper is not None
                and abs(deploy_gripper - self.grip_preshape_command)
                <= GRIP_PRESHAPE_REACHED_TOLERANCE)
            if self.use_dual_tissue_grasp:
                self.advance_dual_tissue_deploy(deploy_elapsed)
            elif (not self.use_dual_tissue_grasp
                    and not self.use_sphere_grasp):
                arm_error = self.selected_arm_error()
                measured_slide = self.joints.get("slide_joint")
                slide_error = (
                    float("inf") if measured_slide is None
                    else abs(float(measured_slide) - self.des_slide))
                # Some simulator/controller builds report a small persistent
                # gripper tracking error even though the fingers have visibly
                # opened.  Do not let that auxiliary feedback deadlock the
                # whole order: arm and slide convergence are the safety gates,
                # while preshape readiness remains a diagnostic below.
                converged = deploy_ready
                soft_ready = (
                    deploy_elapsed >= GENERIC_DIRECT_DEPLOY_DWELL_S
                    and arm_error <= GENERIC_DEPLOY_SOFT_ARM_TOLERANCE_RAD)
                if converged or soft_ready:
                    if not self.approach_arm_joints:
                        self.get_logger().error(
                            "front approach has no contact target")
                        self.set_state(STATE_ABORT)
                        return
                    gate = "converged" if converged else "soft"
                    self.get_logger().info(
                        f"[generic-direct] pregrasp ready after "
                        f"{deploy_elapsed:.2f}s gate={gate} "
                        f"arm_error={arm_error:.4f}rad "
                        f"slide_error={slide_error:.4f}m "
                        f"grip={deploy_gripper} "
                        f"preshape_ready={int(preshape_ready)}; "
                        "starting forward motion")
                    self.start_arm_forward()
                elif deploy_elapsed >= GENERIC_DEPLOY_HARD_TIMEOUT_S:
                    if self._deploy_base_nudge_retry("generic-direct"):
                        return
                    diag_measured = np.round(self.selected_arm_positions(), 4)
                    diag_desired = np.round(
                        self.des_left_arm if self.grasp_arm == "l"
                        else self.des_right_arm, 4)
                    diag_delta = np.round(diag_measured - diag_desired, 4)
                    self.get_logger().error(
                        f"[generic-direct] pregrasp did not converge within "
                        f"{GENERIC_DEPLOY_HARD_TIMEOUT_S:.0f}s "
                        f"arm_error={arm_error:.4f}rad "
                        f"slide_error={slide_error:.4f}m "
                        f"grip={deploy_gripper} "
                        f"preshape={self.grip_preshape_command:.3f} "
                        f"measured={diag_measured.tolist()} "
                        f"desired={diag_desired.tolist()} "
                        f"delta={diag_delta.tolist()} "
                        f"align_y={self.align_base_y:.4f} "
                        f"target={np.round(self.target_world, 4).tolist()}; "
                        "aborting instead of starting a blind forward sweep")
                    self.set_state(STATE_ABORT)
            elif (self.use_sphere_grasp
                  and deploy_elapsed > 0.8 and (
                    measured_ready or continue_experiment)
                  and preshape_ready):
                if not self.approach_arm_joints:
                    self.get_logger().error(
                        "front approach has no contact target")
                    self.set_state(STATE_ABORT)
                    return
                if continue_experiment:
                    envelope = (
                        "soft" if soft_continue else "hard-experimental")
                    self.get_logger().warn(
                        f"[middle-sphere-deploy] {envelope} pregrasp "
                        f"gate bypassed; actual={np.round(actual_tcp, 4)} "
                        f"error={np.round(tcp_error, 4)}m; "
                        "continuing forward grasp instead of aborting")
                self.start_arm_forward()
            elif (self.use_sphere_grasp
                  and deploy_elapsed >= (
                    MIDDLE_SPHERE_DEPLOY_TIMEOUT_S
                    if middle_sphere else DEPLOY_TIMEOUT_S)):
                if self._deploy_base_nudge_retry("sphere"):
                    return
                self.get_logger().error(
                    f"pregrasp did not converge in time; "
                    f"actual_tcp={None if actual_tcp is None else np.round(actual_tcp, 4)} "
                    f"tcp_error={None if tcp_error is None else np.round(tcp_error, 4)}; "
                    f"grip={deploy_gripper} "
                    f"preshape={self.grip_preshape_command:.3f}; "
                    "pose is missing or outside the wide experimental "
                    "envelope; aborting")
                self.set_state(STATE_ABORT)

        elif self.state == STATE_ARM_FORWARD:
            # One endpoint and a synchronized 50 Hz command stream: there are
            # no intermediate targets at which the arm can stop and shake.
            if self.use_dual_tissue_grasp:
                if self.advance_dual_tissue_motion() == "reached":
                    # Top shelf first goes above and behind the front post;
                    # other levels complete their direct insertion here.
                    # Both then use independent bilateral contact search.
                    self.advance_dual_tissue_surround_sequence()
            elif self.use_sphere_grasp:
                sphere_status, _ = self.advance_sphere_forward()
                if sphere_status == "reached":
                    if self.prepare_sphere_post_extension():
                        self.start_post_extension()
                    else:
                        self.set_state(STATE_CLOSE)
                elif sphere_status == "failed":
                    self.set_selected_arm_target(self.pregrasp_arm_joints)
                    self.set_state(STATE_ABORT)
            else:
                generic_status, _ = self.advance_generic_forward()
                if generic_status == "reached":
                    if (self.object_geometry != "sphere"
                            and self.target_kind
                            not in GENERIC_NO_POST_EXTEND_KINDS):
                        self.start_post_extension()
                    else:
                        self.set_state(STATE_CLOSE)

        elif self.state == STATE_DUAL_CONTACT:
            search_status = self.advance_dual_tissue_contact_search()
            if search_status == "reached":
                if not self.start_dual_tissue_squeeze():
                    self.set_state(STATE_ABORT)
            elif search_status == "failed":
                self.set_state(STATE_ABORT)

        elif self.state == STATE_DUAL_SQUEEZE:
            if self.advance_dual_tissue_motion() == "reached":
                self.set_state(STATE_CLOSE)

        elif self.state == STATE_POST_EXTEND:
            extension_status, _ = self.advance_post_extension()
            if extension_status == "reached":
                self.set_state(STATE_CLOSE)

        elif (self.state == STATE_CLOSE
              and self.use_dual_tissue_grasp):
            self.des_left_arm = self.dual_clamp_left_joints.copy()
            self.des_right_arm = self.dual_clamp_right_joints.copy()
            self.des_left_grip = DUAL_TISSUE_GRIP_COMMAND
            self.des_right_grip = DUAL_TISSUE_GRIP_COMMAND
            close_elapsed = self.now() - self.state_t0
            if close_elapsed >= DUAL_TISSUE_CLAMP_DWELL_S:
                self.dual_lift_use_arm = False
                if self.shelf_level == "top":
                    # 顶层：slide 已 pin 在最高位无法再升，改用双臂同步抬升
                    # 让纸盒整体离开板面，再水平撤出。**不要先拉边**：拉边
                    # 会把纸盒拖到板缘（后部仍压在板面上），随后抬升时纸盒
                    # 绕板缘翘起、取不出来（20260818 实测卡死在此）。
                    # 20260817 153144 等 9+ 局完整成功（抓取→导航→放下）的
                    # 流程就是在板面中央直接侧夹抬升：纸盒悬空后下滑约
                    # 12mm 坐落在闭合手指底端（"夹+托"），水平撤退全程
                    # 不碰板面，携带稳定不掉。
                    if self.configure_dual_tissue_arm_lift():
                        self.dual_lift_use_arm = True
                        self.dual_top_extract_stage = 0
                        self.get_logger().info(
                            f"[dual-tissue-clamp] lateral squeeze held for "
                            f"{close_elapsed:.2f}s; top shelf has no overhead "
                            "board, performing the established arm lift")
                        self.start_dual_tissue_arm_lift_stage(0)
                    else:
                        self.get_logger().error(
                            "[dual-tissue] top-shelf arm lift IK failed; "
                            "refusing a same-height retreat across the board")
                        self.set_state(STATE_ABORT)
                        return
                elif self.shelf_level in ("middle", "lower"):
                    # 中层/下层：先 slide 抬升 60mm 让纸盒整体离开板面
                    # （纸盒悬空后被闭合手指夹托着），再水平撤退——全程
                    # 不拖板面。下层 slide 也有足够行程。
                    self.get_logger().info(
                        f"[dual-tissue-clamp] lateral squeeze held for "
                        f"{close_elapsed:.2f}s; lifting "
                        f"{self.shelf_level}-shelf tissue "
                        "before horizontal retreat")
                    self.start_dual_tissue_slide_lift()
                else:
                    self.get_logger().error(
                        "[dual-tissue] unsupported shelf level "
                        f"{self.shelf_level} for the tissue clamp")
                    self.set_state(STATE_ABORT)
                    return

        elif self.state == STATE_CLOSE:
            close_command = (
                GRIP_CLOSE
                if self.use_sphere_grasp
                else GRIP_CLOSE_BY_CLASS.get(
                    self.target_kind, GENERIC_GRIP_CLOSE))
            close_elapsed = self.now() - self.state_t0
            if self.use_sphere_grasp:
                if self.grasp_arm == "r":
                    self.des_right_grip = close_command
                else:
                    self.des_left_grip = close_command
                gripper = self.selected_gripper_position()
                capture_minimum = self.sphere_capture_minimum()
                if gripper is None:
                    self.get_logger().error(
                        "[sphere-grip] gripper feedback unavailable; "
                        "cannot verify capture")
                    self.set_selected_arm_target(self.pregrasp_arm_joints)
                    self.set_state(STATE_ABORT)
                elif (close_elapsed >= SPHERE_CLOSE_SAMPLE_AFTER_S
                      and gripper <= capture_minimum):
                    self.get_logger().error(
                        f"[sphere-grip] capture failed while closing: "
                        f"measured={gripper:.3f} <= "
                        f"minimum={capture_minimum:.3f}; retracting")
                    self.set_selected_arm_target(self.pregrasp_arm_joints)
                    self.set_state(STATE_ABORT)
                elif close_elapsed >= SPHERE_CLOSE_SAMPLE_AFTER_S:
                    stable, spread, span = self.sphere_grip_is_stable(
                        self.sphere_close_grip_samples, gripper)
                    if stable:
                        self.sphere_grip_verified = True
                        self.get_logger().info(
                            f"[sphere-grip] stable capture verified: "
                            f"position={gripper:.3f} "
                            f"range={spread:.3f} over {span:.2f}s")
                        if self.shelf_level == "middle":
                            # A same-height withdrawal repeatedly let oranges
                            # roll out within the first 3--8 cm.  Preserve the
                            # required LIFT stage, but use only the existing
                            # 10 mm slide trial before retreat; this clears the
                            # shelf surface without approaching the board
                            # above.  Full height restoration still happens
                            # only after the TCP is outside the shelf.
                            if self.configure_middle_sphere_lift_from_measured():
                                self.set_state(STATE_TRIAL_LIFT)
                            else:
                                self.set_state(STATE_ABORT)
                        elif self.configure_sphere_lift_from_measured():
                            self.set_state(STATE_TRIAL_LIFT)
                        else:
                            self.set_state(STATE_ABORT)
                    elif close_elapsed >= SPHERE_CLOSE_TIMEOUT_S:
                        self.get_logger().error(
                            f"[sphere-grip] grip did not stabilise within "
                            f"{close_elapsed:.2f}s; position={gripper:.3f} "
                            f"range={spread:.3f} over {span:.2f}s; aborting")
                        self.set_selected_arm_target(
                            self.pregrasp_arm_joints)
                        self.set_state(STATE_ABORT)
            else:
                # Two-stage close for cylindrical/boxed goods: first squeeze to
                # a gentle intermediate opening and let the product settle
                # centred between the jaws, then close fully.  A single
                # continuous close can eject a cylinder sideways when the jaw
                # contact is slightly asymmetric (observed with maidong).
                gripper_now = self.selected_gripper_position()
                if gripper_now is not None:
                    self.generic_close_grip_samples.append(gripper_now)
                if self.generic_close_stage == 1:
                    stage1 = max(float(close_command), GENERIC_GRIP_STAGE1)
                    stage1_settled = (
                        gripper_now is not None
                        and close_elapsed >= GENERIC_CLOSE_STAGE1_MIN_S
                        and self.grip_range(
                            self.generic_close_grip_samples,
                            GENERIC_CLOSE_STAGE1_STABLE_WINDOW_S)
                        < GENERIC_CLOSE_STAGE1_STABLE_RANGE)
                    if (stage1_settled
                            or close_elapsed >= GENERIC_CLOSE_STAGE1_DWELL_S):
                        self.generic_close_stage = 2
                        self.generic_close_grip_samples.clear()
                    des_grip = stage1
                else:
                    des_grip = close_command
                if self.grasp_arm == "r":
                    self.des_right_grip = des_grip
                else:
                    self.des_left_grip = des_grip
                if (gripper_now is not None
                        and self.now() - self.last_generic_close_log > 0.5):
                    self.last_generic_close_log = self.now()
                    self.get_logger().info(
                        f"[generic-close-diag] elapsed="
                        f"{close_elapsed:.2f}s grip={gripper_now:.3f} "
                        f"des={des_grip:.3f} "
                        f"stage={self.generic_close_stage} "
                        f"tcp={np.round(self.selected_tcp_world(), 4)}")
                close_settled = (
                    self.generic_close_stage == 2
                    and close_elapsed >= GENERIC_CLOSE_MIN_S
                    and gripper_now is not None
                    and self.grip_range(
                        self.generic_close_grip_samples,
                        GENERIC_CLOSE_STABLE_WINDOW_S)
                    < GENERIC_CLOSE_STABLE_RANGE)
                if close_elapsed >= GENERIC_CLOSE_DWELL_S or close_settled:
                    gripper = gripper_now
                    message = (
                        f"[generic-grip] close settled; "
                        f"start={self.generic_close_start_grip} "
                        f"measured={gripper} command={close_command:.3f} "
                        f"elapsed={close_elapsed:.2f}s "
                        f"stage={self.generic_close_stage} "
                        f"close_step={GRIP_CLOSE_MAX_STEP:.3f}rad/tick")
                    if (gripper is not None
                            and gripper
                            <= close_command + GENERIC_EMPTY_GRIP_MARGIN):
                        self.get_logger().error(
                            message + "; empty-grasp signature observed; "
                            "grasp lost during close - aborting instead of "
                            "reporting a false success")
                        self.set_selected_arm_target(
                            self.pregrasp_arm_joints)
                        self.set_state(STATE_ABORT)
                    else:
                        self.get_logger().info(message)
                        if (self.is_top_shelf
                                and not
                                self.configure_generic_top_lift_from_measured()):
                            self.set_selected_arm_target(
                                self.pregrasp_arm_joints)
                            self.set_state(STATE_ABORT)
                        else:
                            self.set_state(STATE_LIFT)

        elif self.state == STATE_TRIAL_LIFT:
            self.des_slide = (
                self.slide_grasp if self.shelf_level == "top"
                else self.sphere_trial_slide)
            gripper = self.selected_gripper_position()
            capture_minimum = self.sphere_capture_minimum()
            trial_elapsed = self.now() - self.state_t0
            if gripper is None or gripper <= capture_minimum:
                self.get_logger().error(
                    f"[sphere-trial-lift] product slipped during 10mm test; "
                    f"measured_grip={gripper} minimum={capture_minimum:.3f}")
                self.set_selected_arm_target(self.pregrasp_arm_joints)
                self.set_state(STATE_ABORT)
            elif self.commands_ready():
                stable, spread, span = self.sphere_grip_is_stable(
                    self.sphere_trial_grip_samples, gripper)
                if stable:
                    if self.shelf_level == "top":
                        self.get_logger().info(
                            f"[sphere-trial-lift] "
                            f"{SPHERE_TRIAL_LIFT_M:.3f}m test passed; "
                            f"grip={gripper:.3f} range={spread:.3f} over "
                            f"{span:.2f}s; top shelf has no overhead board, "
                            "continuing full lift")
                        self.set_selected_arm_target(
                            self.sphere_lift_arm_joints)
                        self.set_state(STATE_LIFT)
                    else:
                        # Keep only the 10 mm capture test (wxj v2)。中层
                        # 上方有货架板，完整抬升会把球顶进上层板；在试抬
                        # 高度直接水平收回，运输高度恢复留给外层流程在
                        # TCP 离开货架后进行。
                        self.des_slide = self.sphere_trial_slide
                        self.set_selected_arm_target(
                            self.sphere_retreat_arm_joints)
                        self.get_logger().info(
                            f"[sphere-trial-lift] "
                            f"{SPHERE_TRIAL_LIFT_M:.3f}m capture test passed; "
                            f"grip={gripper:.3f} range={spread:.3f} over "
                            f"{span:.2f}s; retracting before full height "
                            "restoration")
                        self.set_state(STATE_RETREAT)
            if (self.state == STATE_TRIAL_LIFT
                    and trial_elapsed >= SPHERE_TRIAL_LIFT_TIMEOUT_S):
                self.get_logger().error(
                    "[sphere-trial-lift] grip/arm did not stabilise; aborting")
                self.set_selected_arm_target(self.pregrasp_arm_joints)
                self.set_state(STATE_ABORT)

        elif (self.state == STATE_LIFT
              and self.use_dual_tissue_grasp):
            self.des_left_grip = DUAL_TISSUE_GRIP_COMMAND
            self.des_right_grip = DUAL_TISSUE_GRIP_COMMAND
            lift_elapsed = self.now() - self.state_t0
            if self.dual_lift_use_arm:
                # 顶层双臂抬升（slide 已 pin 在最高位，只能靠臂关节）。
                self.des_slide = self.slide_grasp
                if self.advance_dual_tissue_motion() == "reached":
                    next_stage = self.dual_lift_arm_stage + 1
                    if next_stage < len(self.dual_lift_arm_waypoints):
                        self.start_dual_tissue_arm_lift_stage(next_stage)
                    else:
                        self.get_logger().info(
                            f"[dual-tissue-arm-lift] raised "
                            f"{self.dual_lift_arm_achieved_m:.3f}m via "
                            f"{len(self.dual_lift_arm_waypoints)} slow "
                            "branch-checked segments; retracting "
                            "horizontally at the raised height")
                        self.start_dual_tissue_arm_retreat_stage(0)
            else:
                # 中层/下层 slide 抬升：纸盒整体离开板面后被手指夹托着。
                self.des_left_arm = self.dual_clamp_left_joints.copy()
                self.des_right_arm = self.dual_clamp_right_joints.copy()
                lift_status = self.advance_dual_tissue_slide_lift()
                if lift_status == "reached":
                    self.start_dual_tissue_motion(
                        "retreat",
                        self.dual_retreat_left_joints,
                        self.dual_retreat_right_joints,
                        DUAL_TISSUE_PREGRASP_BACKOFF_M
                        + self.dual_insert_forward_m,
                        DUAL_TISSUE_RETREAT_SPEED_MPS,
                        STATE_RETREAT)
                elif lift_status == "failed":
                    self.set_state(STATE_ABORT)

        elif self.state == STATE_LIFT:
            if self.use_sphere_grasp:
                self.des_slide = (
                    self.slide_grasp if self.shelf_level == "top"
                    else self.sphere_lift_slide)
                gripper = self.selected_gripper_position()
                capture_minimum = self.sphere_capture_minimum()
                if gripper is None or gripper <= capture_minimum:
                    self.get_logger().error(
                        f"[sphere-lift] product lost during full lift; "
                        f"measured_grip={gripper} "
                        f"minimum={capture_minimum:.3f}")
                    self.set_state(STATE_ABORT)
                elif (self.now() - self.state_t0 > 0.25
                        and self.commands_ready()):
                    self.get_logger().info(
                        f"[{self.shelf_level}-sphere-lift] raised "
                        f"{SPHERE_LIFT_M:.3f}m using "
                        f"{'arm' if self.shelf_level == 'top' else 'slide'}; "
                        "retracting horizontally at raised height")
                    self.set_selected_arm_target(
                        self.sphere_retreat_arm_joints)
                    self.set_state(STATE_RETREAT)
                elif self.now() - self.state_t0 >= SPHERE_LIFT_TIMEOUT_S:
                    self.get_logger().error(
                        f"[{self.shelf_level}-sphere-lift] "
                        "lift motion did not converge; aborting")
                    self.set_state(STATE_ABORT)
            else:
                lift_elapsed = self.state_elapsed_monotonic()
                if (self.is_top_shelf
                        and self.generic_top_lift_arm_joints is not None):
                    self.des_slide = self.slide_grasp
                    if (lift_elapsed > 0.25
                            and self.commands_ready(
                                ARM_REACHED_TOLERANCE_RAD + 0.015)):
                        self.get_logger().info(
                            f"[generic-top-lift] raised "
                            f"{GENERIC_TOP_LIFT_M:.3f}m; "
                            "retracting horizontally at raised height")
                        self.set_selected_arm_target(
                            self.generic_top_retreat_arm_joints)
                        self.set_state(STATE_RETREAT)
                    elif lift_elapsed >= GENERIC_TOP_LIFT_TIMEOUT_S:
                        self.get_logger().warn(
                            "[generic-top-lift] lift did not fully settle; "
                            "continuing raised retreat instead of aborting")
                        self.set_selected_arm_target(
                            self.generic_top_retreat_arm_joints)
                        self.set_state(STATE_RETREAT)
                elif self.shelf_level == "lower":
                    # Lower goods use their own small slide lift.  A fixed
                    # dwell keeps this branch independent of the middle-layer
                    # convergence gates and avoids dragging the product over
                    # the lower front rail at its original height.
                    self.des_slide = max(
                        SLIDE_MIN, self.slide_grasp - LOWER_LIFT_M)
                    if lift_elapsed >= LOWER_LIFT_DWELL_S:
                        measured_slide = self.joints.get("slide_joint")
                        self.get_logger().info(
                            f"[lower-front-lift] commanded "
                            f"{LOWER_LIFT_M:.3f}m upward slide motion; "
                            f"target_slide={self.des_slide:.3f} "
                            f"measured_slide={measured_slide}; "
                            "retracting horizontally")
                        self.set_selected_arm_target(
                            self.pregrasp_arm_joints)
                        self.set_state(STATE_RETREAT)
                else:
                    # Middle-layer generic lift: wait generously for normal
                    # convergence, then degrade to a guarded horizontal exit.
                    self.des_slide = max(
                        SLIDE_MIN, self.slide_grasp - LIFT_AMOUNT_M)
                    slide = self.joints.get(
                        "slide_joint", self.des_slide + 1.0)
                    if (lift_elapsed > 0.5
                            and abs(slide - self.des_slide) < 0.025):
                        # Pull the grasped product horizontally clear of the
                        # shelf at the lifted height.
                        self.set_selected_arm_target(
                            self.pregrasp_arm_joints)
                        self.set_state(STATE_RETREAT)
                    elif lift_elapsed >= GENERIC_MIDDLE_LIFT_TIMEOUT_S:
                        # The item is already grasped.  A slightly incomplete
                        # lift is safer to resolve by withdrawing horizontally
                        # at the measured height than by waiting forever or
                        # entering STATE_ABORT, whose recovery opens the grip.
                        error = abs(float(slide) - self.des_slide)
                        self.get_logger().warn(
                            "[generic-middle-lift] slide did not reach the "
                            f"normal 0.025m tolerance within "
                            f"{GENERIC_MIDDLE_LIFT_TIMEOUT_S:.1f}s "
                            f"(error={error:.3f}m); continuing with a guarded "
                            "horizontal retreat at the measured height")
                        self.set_selected_arm_target(
                            self.pregrasp_arm_joints)
                        self.set_state(STATE_RETREAT)

        elif (self.state == STATE_RETREAT
              and self.use_dual_tissue_grasp):
            self.des_left_grip = DUAL_TISSUE_GRIP_COMMAND
            self.des_right_grip = DUAL_TISSUE_GRIP_COMMAND
            # 旧手侧面姿态需横移避柱；窄腕姿态保持直退，
            # 避免带货重新扫向立柱。
            slot = self.target_slot()
            if (slot is not None and slot[2] in ("1", "3")
                    and not getattr(self, "dual_side_rolled", False)
                    and self.now() - self.state_t0
                    < TISSUE_POST_RETREAT_SHIFT_S):
                lateral = (
                    TISSUE_POST_RETREAT_LATERAL_MPS
                    if slot[2] == "1"
                    else -TISSUE_POST_RETREAT_LATERAL_MPS)
                self.set_twist(lateral, 0.0)
            if self.advance_dual_tissue_motion() == "reached":
                if (self.shelf_level == "top"
                        and self.dual_lift_use_arm
                        and self.dual_lift_retreat_waypoints):
                    next_stage = self.dual_lift_retreat_stage + 1
                    if next_stage < len(self.dual_lift_retreat_waypoints):
                        self.start_dual_tissue_arm_retreat_stage(next_stage)
                        return
                if (self.shelf_level == "top"
                        and self.dual_top_extract_stage == 1):
                    # 旧流程残留的 stage-1 分支：纸盒在抬升前被拉到板缘。
                    # 该流程已被 STATE_CLOSE 的直接抬升取代（stage 恒为 0），
                    # 此处仅作防御性兜底。
                    if self.configure_dual_tissue_arm_lift():
                        self.dual_lift_use_arm = True
                        self.dual_top_extract_stage = 0
                        self.start_dual_tissue_arm_lift_stage(0)
                    else:
                        self.get_logger().error(
                            "[dual-tissue-arm-lift] IK failed; aborting")
                        self.set_state(STATE_ABORT)
                else:
                    self.set_state(STATE_DONE)
                    self.get_logger().info(
                        f"SUCCESS: removed {self.target_kind} from shelf; "
                        f"holding it between both arms (ArUco "
                        f"ID={self.target_marker_id})")

        elif self.state == STATE_RETREAT:
            gripper = self.selected_gripper_position()
            if self.use_sphere_grasp:
                capture_minimum = self.sphere_capture_minimum()
                if gripper is None or gripper <= capture_minimum:
                    self.get_logger().error(
                        f"[sphere-grip] product lost during lift/retreat; "
                        f"measured_grip={gripper} "
                        f"minimum={capture_minimum:.3f}")
                    self.set_state(STATE_ABORT)
            elif (self.now() - self.state_t0 >= GENERIC_RETREAT_TIMEOUT_S):
                actual_tcp = self.selected_tcp_world()
                clear_y = (
                    self.target_world[1]
                    - PRODUCT_BEHIND_MARKER_M
                    - GENERIC_RETREAT_CLEAR_MARGIN_M)
                if (actual_tcp is not None
                        and actual_tcp[1] < clear_y):
                    self.get_logger().warn(
                        f"[generic-retreat] arm did not settle within "
                        f"{GENERIC_RETREAT_TIMEOUT_S:.0f}s "
                        f"(arm_error={self.selected_arm_error():.4f}); "
                        f"TCP clear of the shelf "
                        f"(y={actual_tcp[1]:.3f} < {clear_y:.3f}); "
                        "treating as removed")
                    self.set_state(STATE_DONE)
                else:
                    self.get_logger().error(
                        f"[generic-retreat] timeout but TCP not clear of "
                        f"the shelf (y="
                        f"{None if actual_tcp is None else round(float(actual_tcp[1]), 3)} "
                        f"< {clear_y:.3f}); aborting")
                    self.set_state(STATE_ABORT)
            elif (not self.use_sphere_grasp
                    and self.shelf_level != "lower"
                    and self.now() - self.state_t0 >= 1.0):
                # The arm streams back to the pregrasp joints; once the
                # measured TCP is clearly south of the shelf front rail the
                # removal is already complete.  The old code waited for joint
                # convergence (which base drift can prevent, measured 8 s) and
                # only then fell back to this same TCP-clear check.
                actual_tcp = self.selected_tcp_world()
                clear_y = (
                    self.target_world[1]
                    - PRODUCT_BEHIND_MARKER_M
                    - GENERIC_RETREAT_CLEAR_MARGIN_M)
                if (actual_tcp is not None
                        and actual_tcp[1] < clear_y):
                    self.get_logger().warn(
                        f"[generic-retreat] TCP clear of the shelf "
                        f"(y={actual_tcp[1]:.3f} < {clear_y:.3f}) after "
                        f"{self.now() - self.state_t0:.2f}s; "
                        "treating as removed")
                    self.set_state(STATE_DONE)
                    self.get_logger().info(
                        f"SUCCESS: removed {self.target_kind} from shelf; "
                        f"holding it with {self.grasp_arm} arm (ArUco "
                        f"ID={self.target_marker_id})")
            lower_generic_done = (
                not self.use_sphere_grasp
                and self.shelf_level == "lower"
                and self.now() - self.state_t0 >= LOWER_RETREAT_DWELL_S)
            established_profile_done = (
                self.state == STATE_RETREAT
                and not (
                    not self.use_sphere_grasp
                    and self.shelf_level == "lower")
                and self.now() - self.state_t0 > 0.5
                and self.commands_ready())
            if (self.state == STATE_RETREAT
                    and (lower_generic_done or established_profile_done)):
                self.set_state(STATE_DONE)
                self.get_logger().info(
                    f"SUCCESS: removed {self.target_kind} from shelf; "
                    f"holding it with {self.grasp_arm} arm (ArUco "
                    f"ID={self.target_marker_id})")

        elif self.state == STATE_ABORT:
            if self._abort_recovery_ready():
                self.get_logger().error(
                    "abort motion settled; shutting down client cleanly")
                rclpy.shutdown()
                return

        self.apply_manip_base_hold()
        self.smooth_commands()
        self.publish_commands()

        if self.now() - self.last_status_log > 1.0:
            target_text = ("unlocked" if self.target_world is None
                           else np.array2string(
                               self.target_world, precision=3,
                               suppress_small=True))
            profile = self.grasp_profile_name()
            sphere_feedback = ""
            if self.use_sphere_grasp:
                tcp = self.selected_tcp_world()
                grip = self.selected_gripper_position()
                tcp_text = ("unavailable" if tcp is None
                            else np.array2string(
                                tcp, precision=3, suppress_small=True))
                grip_text = "unavailable" if grip is None else f"{grip:.3f}"
                sphere_feedback = f" tcp={tcp_text} grip={grip_text}"
            self.get_logger().info(
                f"state={self.state} base=({self.base_xy[0]:.2f},"
                f"{self.base_xy[1]:.2f}) target={target_text} "
                f"marker={self.target_marker_id} "
                f"profile={profile} "
                f"tcp_diag={int(self.tcp_diagnostic_ground_truth)}"
                f"{sphere_feedback}")
            self.last_status_log = self.now()


class MainThreadResultViewer(Node):
    """Receive annotated YOLO images; render them only from the main thread.

    Calling OpenCV HighGUI from a MultiThreadedExecutor callback can freeze the
    GTK window and, more importantly, block all subsequent YOLO callbacks.  The
    subscription callback therefore only stores the newest frame.  ``show`` is
    invoked by ``main`` on Python's main thread.
    """

    WINDOW_NAME = "YOLO goods + shelf-pick status"

    def __init__(self, controller: ShelfPickController):
        super().__init__("yolo_shelf_pick_result_viewer")
        self.controller = controller
        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_camera = "unknown"
        self.latest_aruco = []
        self.create_subscription(
            Image, "/kele/result_image", self.image_cb, 5)
        self.create_subscription(
            String, "/aruco/head/detections", self.aruco_cb, 10)

    def image_cb(self, message: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        frame_id = message.header.frame_id.lower()
        camera_name = next(
            (name for name in ("head", "left", "right")
             if name in frame_id),
            "unknown")
        with self.frame_lock:
            self.latest_frame = frame.copy()
            self.latest_frame_camera = camera_name

    def aruco_cb(self, message: String) -> None:
        records = decode_list(message)
        with self.frame_lock:
            self.latest_aruco = records

    def show(self) -> int:
        with self.frame_lock:
            frame = (None if self.latest_frame is None
                     else self.latest_frame.copy())
            frame_camera = self.latest_frame_camera
            aruco_records = list(self.latest_aruco)
        if frame is None:
            # HighGUI still needs event pumping before the first ROS frame.
            return cv2.waitKey(20) & 0xFF

        target_marker = self.controller.target_marker_id
        show_head_aruco = (
            frame_camera == "head"
            or frame_camera == "unknown")
        for record in aruco_records if show_head_aruco else ():
            try:
                marker_id = int(record["id"])
                corners = np.rint(np.asarray(
                    record["corners"], dtype=float)).astype(np.int32).reshape(4, 2)
            except (KeyError, TypeError, ValueError):
                continue
            color = ((0, 255, 255) if marker_id == target_marker
                     else (255, 0, 255))
            cv2.polylines(
                frame, [corners.reshape(-1, 1, 2)], True,
                color, 3, cv2.LINE_AA)
            label_x = max(0, int(corners[:, 0].min()))
            label_y = max(18, int(corners[:, 1].min()) - 7)
            cv2.putText(
                frame, f"ArUco ID={marker_id}", (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

        marker = "none" if target_marker is None else str(target_marker)
        grasp_profile = self.controller.grasp_profile_name()
        status = (f"state={self.controller.state}  "
                  f"view={frame_camera}  "
                  f"target={self.controller.target_kind}  "
                  f"profile={grasp_profile}  "
                  f"lower_aruco={marker}")
        text_y = frame.shape[0] - 10
        cv2.rectangle(
            frame, (0, frame.shape[0] - 34),
            (frame.shape[1] - 1, frame.shape[0] - 1), (20, 20, 20), -1)
        cv2.putText(
            frame, status, (8, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.WINDOW_NAME, frame)
        return cv2.waitKey(20) & 0xFF


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO coarse search + lower-ArUco localisation + shelf pick")
    parser.add_argument(
        "--target-kind", required=True,
        choices=sorted(PRODUCT_CENTER_ABOVE_MARKER_M),
        help="exact goods class to remove from the shelf")
    parser.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS),
        help="multi-class Ultralytics checkpoint (default: repository best.pt)")
    parser.add_argument(
        "--confidence", type=float, default=0.45,
        help="minimum YOLO confidence (default: 0.45)")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--show", action="store_true", help="show the YOLO result window")
    parser.add_argument(
        "--max-scan-cycles", type=int, default=3,
        help="abort after this many complete shelf scans (default: 3)")
    parser.add_argument(
        "--tcp-diagnostic-ground-truth", action="store_true",
        help=("fixed-layout diagnostic: keep YOLO/ArUco target selection but "
              "use the exact scene product centre; the fixed gripper TCP "
              "transform remains active; requires SUPERMARKET_RANDOMIZE=0"))
    parser.add_argument(
        "--scan-skip-lower", action="store_true",
        help=("fixed-layout speedup: skip the three lower-shelf camera poses "
              "when the requested kind never sits on the lower shelf in "
              "retail_competition_layout.json; keep off when the server runs "
              "with SUPERMARKET_RANDOMIZE=1"))
    parser.add_argument(
        "--no-close-recheck", action="store_true",
        help="disable close-range class verification before grasping")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if args.max_scan_cycles < 1:
        parser.error("--max-scan-cycles must be >= 1")
    return args


def main() -> None:
    from run_log import start_run_log
    start_run_log("shelf_pick")
    args = parse_args()
    weights = str(Path(args.weights).expanduser().resolve())
    if not Path(weights).is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    rclpy.init()
    nodes = []
    spin_thread = None
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        # Never call HighGUI from KeleDetectNode's executor callback.  When
        # --show is requested, MainThreadResultViewer handles the same
        # /kele/result_image stream from the process main thread instead.
        yolo_node = KeleDetectNode(
            backend="yolo", pub_res_img=True, device=args.device,
            weights=weights, target_kind=args.target_kind,
            confidence=args.confidence, show=False,
            camera_names=("head",))
        aruco_node = ArucoDetectNode(
            "head", marker_size=MARKER_SIZE_M, publish_tf=False,
            publish_result_image=True)
        controller = ShelfPickController(
            args.target_kind, args.max_scan_cycles,
            args.tcp_diagnostic_ground_truth, args.scan_skip_lower,
            close_recheck=not args.no_close_recheck)
        nodes = [yolo_node, aruco_node, controller]
        viewer = None
        if args.show:
            viewer = MainThreadResultViewer(controller)
            nodes.append(viewer)
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
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        executor.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        for node in nodes:
            node.destroy_node()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
