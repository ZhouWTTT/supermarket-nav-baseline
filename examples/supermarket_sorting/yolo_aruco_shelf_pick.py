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
    "heweidao": 0.00,
    "shupian": 0.040,
    "zhijin": 0.043,
    "maidong": 0.034,
    "kele": 0.0315,
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
YOLO_ONLY_TARGET_CONFIRMATIONS = 4
YOLO_ONLY_TARGET_SPREAD_MAX_M = 0.09
YOLO_ONLY_TARGET_CONF_MIN = 0.80
# Horizontal depth-to-marker distance below which the product is considered
# still in its nominal slot.  In that case the depth-measured Z is biased high
# by the downward camera angle and the marker-derived Z is biased low by the
# marker pose refinement; averaging the two cancels most of both biases.
# A displaced product (distance above this threshold) keeps the measured Z.
DEPTH_TARGET_IN_SLOT_XY_MAX_M = 0.10

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
SCAN_CAMERA_POSES = (
    ("overview_high", 0.11, 0.00, -0.20),
    # 中间层改用下降相机（slide 0.60，相机 z≈0.79，pitch +0.16 微仰）：
    # L2 货物（z≈0.89-0.96）在相机上方 8~14°，+0.16 把 L2 拉到画面中上部，
    # 正面视角比高位俯视更稳；带 ±0.15 偏航覆盖边缘列。
    ("middle_center", 0.60, 0.00, 0.16),
    ("middle_yaw_minus", 0.60, -0.15, 0.16),
    ("middle_yaw_plus", 0.60, 0.15, 0.16),
    # 下层三档 slide 0.45 -> 0.60：相机再降 15cm（相机 z≈0.94 -> ≈0.79），
    # 更接近 L1 货物高度（z≈0.54-0.60），视角更正面，减少前缘遮挡对
    # 下层货物和码牌的干扰。slide 关节范围 -0.04~0.87，0.60 安全。
    ("lower_center", 0.60, 0.00, -0.45),
    ("lower_yaw_minus", 0.60, -0.15, -0.45),
    ("lower_yaw_plus", 0.60, 0.15, -0.45),
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
SCAN_SETTLE_S = 0.15
SCAN_DWELL_S = 0.6
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
SCAN_OVERVIEW_POSES = SCAN_CAMERA_POSES[:3]


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
# drive_to cruise/rotation limits.  The old profile rotated in place whenever
# the heading error exceeded 0.18 rad and then crept at min(0.36, distance);
# measured ALIGN phases took 9--20 s for base moves under 0.35 m.  Translate
# while correcting heading once the error is moderate, keep a minimum approach
# speed, and accept the final heading with a deadband so odom yaw noise cannot
# stall the phase.
NAV_LINEAR_MAX_MPS = 0.90
NAV_LINEAR_MIN_MPS = 0.10
# 货架对齐的最后一段使用更低的进给下限：对齐容差只有 2.5cm，高速停车
# 过冲会偏移抓取位姿。最后 50mm 保持低速，不做加速。
NAV_ALIGN_LINEAR_MIN_MPS = 0.10
NAV_ROTATE_GATE_RAD = 0.45
NAV_ANGULAR_MAX_RADPS = 1.50
NAV_TRANSLATE_ANGULAR_MAX_RADPS = 1.00
NAV_YAW_DEADBAND_RAD = 0.035
# 原地旋转卡死恢复：旋转指令发出后若 yaw 长时间无变化（被西墙/货架顶住），
# 先短距离倒车解除卡死再继续旋转，避免在西侧货架贴墙处永久卡住。
NAV_ROT_STALL_S = 2.5            # 旋转无进展判定时间（秒）
NAV_ROT_STALL_MIN_CHANGE_RAD = 0.03
NAV_ROT_UNSTICK_DIST_M = 0.15    # 解除卡死时的倒车距离
NAV_ROT_UNSTICK_SPEED_MPS = 0.08
NAV_ROT_UNSTICK_TIMEOUT_S = 4.0
NAV_ROT_UNSTICK_MAX = 3          # 最多解除次数，超过后重置继续尝试

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

# A tissue box is 172 mm wide, more than twice one gripper's 80 mm opening.
# In the fixed layout every tissue target is on the middle shelf, so a separate
# symmetric two-arm profile uses the closed grippers as padded side supports:
# reach around both sides, move inward together, lift with the slide, and
# retreat together while maintaining the lateral squeeze.
DUAL_TISSUE_PREGRASP_BACKOFF_M = 0.160
# 探入深度从 +2cm 加大到 +3cm：侧夹/环绕位置更深入纸盒两侧，夹得更稳。
DUAL_TISSUE_INSERT_FORWARD_M = 0.030
# Keep the complete gripper bodies clear during insertion, not merely their
# TCP centres.  The 140 mm surround half-span leaves 54 mm around a nominal
# 86 mm tissue half-width, so the measured ~20 mm lateral vision error does
# not turn one arm into an early frontal contact.
DUAL_TISSUE_PREGRASP_HALF_SPAN_M = 0.150
# "直接探入"的初始双臂间距（单侧半跨度，wxj v2 值）：总间距 28cm（单侧
# 14cm），合拢行程约 5cm，两侧基本同时夹住盒壁，又保留足够的横向视觉
# 误差余量。全列统一使用。
DUAL_TISSUE_DIRECT_PROBE_SPAN_M = 0.140
DUAL_TISSUE_SURROUND_HALF_SPAN_M = 0.140
# Half-span of the final side clamp.  The closed grippers approach the 172 mm
# box (half-width 86 mm) from the sides; 90 mm 半跨度让钳爪在保持阶段真正
# 压进盒壁（每侧约 4mm 过盈 + 手指接触面），夹持更稳。再小会顶到盒角。
DUAL_TISSUE_CLAMP_HALF_SPAN_M = 0.090
# Raise only the tissue TCP by 15 mm to keep both gripper bodies off the shelf.
# 初始探入高度再降 4cm（+5.5cm -> +1.5cm）：宽跨度直探时夹爪更贴盒壁底部。
DUAL_TISSUE_TCP_CLEARANCE_M = 0.015
# On the lower shelf the arms reach further down and their solved joints sag
# below the target Z by 10-25 mm; the left arm then drags on the board and
# stalls 50-70 mm short, skewing the contact line by ~12 deg.  Use a larger
# clearance on the lower shelf so the fingers stay above the board.  FK
# diagnosis showed the closed grippers ended ~25-45 mm ABOVE the tissue box
# top (wrist bodies resting on the top corners, box never lifted).  Lower the
# contact height so the wrist/fingers engage the box side below its top: with
# a 0.10 clearance the commanded TCP is ~0.599 m, the sim-real ~0.574 m, below
# the 0.587 m box top and above the 0.499 m board.
# 下层纸巾初始探入高度降低 4cm：从比板面高 13cm 降到 9cm，夹爪更贴盒壁。
DUAL_TISSUE_LOWER_TCP_CLEARANCE_M = 0.09
# Only the dual-tissue profile starts closer to the shelf.  Previous trials
# measured roughly 46--49 mm of base rollback during two-arm insertion, so a
# 40 mm forward offset restores reach margin without affecting other goods.
DUAL_TISSUE_ALIGN_FORWARD_M = 0.040
# 夹爪完全闭合（0 为最紧），配合侧夹预压增大，避免纸盒从指间滑脱。
DUAL_TISSUE_GRIP_COMMAND = 0.0
DUAL_TISSUE_FORWARD_SPEED_MPS = 0.036
DUAL_TISSUE_CONTACT_SEARCH_HALF_SPAN_M = 0.045
DUAL_TISSUE_CONTACT_SEARCH_SPEED_MPS = 0.015
# Minimum inward travel before a grip/stall contact signal is accepted.  The
# measured travel to the box side varies widely with the box's lateral offset
# (18-70 mm observed), so keep the bar low; the measured-centre anchor below
# is what makes the final clamp symmetric.
DUAL_TISSUE_CONTACT_MIN_ADVANCE_M = 0.012
DUAL_TISSUE_CONTACT_COMMAND_LEAD_M = 0.004
DUAL_TISSUE_CONTACT_STALL_WINDOW_S = 0.55
DUAL_TISSUE_CONTACT_STALL_MIN_SPAN_S = 0.40
DUAL_TISSUE_CONTACT_STALL_RANGE_M = 0.0015
DUAL_TISSUE_CONTACT_ENDPOINT_TOLERANCE_M = 0.003
# A closed unloaded gripper tracks 0.08 in prior runs; side contact in the
# successful-but-unstable run forced both measured joints down near 0.012.
DUAL_TISSUE_GRIP_CONTACT_MAX = 0.045
# 双臂闭合时的单侧预压量（米，wxj v2 值）：从实测接触位置再向内压入的
# 量。加大后两侧钳爪对纸盒的夹紧力更强，纸盒不易在抬/撤过程中滑动。
DUAL_TISSUE_SQUEEZE_M = 0.030
DUAL_TISSUE_SQUEEZE_SPEED_MPS = 0.012
DUAL_TISSUE_RETREAT_SPEED_MPS = 0.018
DUAL_TISSUE_MIN_MOTION_DURATION_S = 2.0
DUAL_TISSUE_MOTION_SETTLE_S = 0.75
DUAL_TISSUE_MOTION_MIN_SETTLE_S = 0.20
DUAL_TISSUE_DEPLOY_DWELL_S = 2.0
DUAL_TISSUE_CLAMP_DWELL_S = 4.0
DUAL_TISSUE_LIFT_M = 0.060
DUAL_TISSUE_LIFT_DWELL_S = 2.5
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
# 所有货物抓取姿态的目标 Z 统一抬升 1cm：让指尖略高于测得的货物中心，
# 避免指尖落在货架导轨/前缘高度（D 架苹果指尖撞导轨问题），对深度测量
# 的偏低误差更宽容。对球体（苹果/橙子）与普通货物、各层货架一致生效。
GRASP_TCP_Z_RAISE_M = 0.010
# 按货物类型的抓取高度额外偏移（米）：负值降低。可乐、核桃味刀调低 1cm。
GRASP_TCP_Z_OFFSET_BY_KIND = {"kele": -0.010, "heweidao": -0.010}
# KDL/仿真运动学的 X 方向执行偏差（wxj v2 实测）：右臂 TCP 比指令偏东约
# 1cm、左臂偏西约 0.5cm。反向补偿到指令目标 X，让实际接触点落在真实
# 目标上。
GRASP_TCP_X_OFFSET_BY_ARM = {"r": -0.010, "l": 0.005}
GRIPPER_MAX_OPENING_M = 0.080
GRIP_PRESHAPE_CLEARANCE_M = 0.012
GRIP_PRESHAPE_REACHED_TOLERANCE = 0.04

STATE_GO_SCAN = "go_scan"
STATE_SCAN = "scan"
STATE_REVISIT = "revisit"
STATE_ALIGN = "align"
STATE_RECHECK = "recheck"
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
        self.grasp_arm = "r"
        self.align_base_x = None
        self.align_base_y = SCAN_Y
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
        self.dual_lift_settled_since = None
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
        self.dual_surround_stage = 0
        self.dual_clamp_half_span = DUAL_TISSUE_CLAMP_HALF_SPAN_M
        self.dual_insert_forward_m = DUAL_TISSUE_INSERT_FORWARD_M
        # 中间列纸巾"直接探入"：不走宽环绕，双臂以夹持跨度直接前探后压紧
        self.dual_direct_probe = False
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
        self.dual_motion_endpoint_ready_since = None
        self.dual_contact_start_left_joints = None
        self.dual_contact_start_right_joints = None
        self.dual_contact_target_left_joints = None
        self.dual_contact_target_right_joints = None
        self.dual_contact_start_left_world = None
        self.dual_contact_start_right_world = None
        self.dual_contact_goal_left_world = None
        self.dual_contact_goal_right_world = None
        self.dual_contact_duration_s = 0.0
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
        self.last_odom_time = self.now()

    def joint_cb(self, message: JointState) -> None:
        self.joints = {
            name: float(message.position[index])
            for index, name in enumerate(message.name)
            if index < len(message.position)
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
                    world = record.get("world")
                    slot = None
                    if (isinstance(world, (list, tuple))
                            and len(world) == 3):
                        try:
                            # Stable fixed-grid grouping prevents one product
                            # from becoming several rounded-coordinate boxes.
                            slot = fixed_slot_from_world(
                                float(world[0]), float(world[2]))
                        except (TypeError, ValueError):
                            slot = None
                    if slot is not None:
                        box_key = slot
                    elif (isinstance(world, (list, tuple))
                          and len(world) == 3
                          and all(isinstance(v, (int, float))
                                  for v in world)):
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
                    if isinstance(world, (list, tuple)) and len(world) == 3:
                        try:
                            world_array = np.asarray(world, dtype=float)
                            entry["world"] = world_array
                            entry["worlds"].append(world_array)
                        except (TypeError, ValueError):
                            pass
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
        self._commit_yolo_only_target(target_world, slot)

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

    def _commit_yolo_only_target(
            self, target_world: np.ndarray,
            slot: tuple[str, str, str]) -> None:
        """Configure the existing grasp state machine from a YOLO-only slot."""
        self.target_world = np.asarray(target_world, dtype=float)
        self.target_marker_id = None
        self.target_physical_marker_id = None
        self.committed_slot = tuple(slot)
        self._recheck_passed = False

        if self.use_dual_tissue_grasp:
            self.grasp_arm = "r"
            desired_base_x = self.target_world[0]
        else:
            desired_right_base_x = (
                self.target_world[0] - ARM_LATERAL_BIAS_M)
            if slot[0] == "A" or desired_right_base_x < NAV_X_MIN:
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
        self.get_logger().info(
            f"[localised] source=yolo_only marker=None "
            f"product_world={np.round(self.target_world, 3)} "
            f"arm={'both' if self.use_dual_tissue_grasp else self.grasp_arm} "
            f"grasp_profile={self.grasp_profile_name()} "
            f"align_y={self.align_base_y:.3f} slot={slot}")

    def target_slot(self) -> tuple[str, str, str] | None:
        """Return the fixed matrix slot committed for the current target."""
        if self.committed_slot is not None:
            return tuple(self.committed_slot)
        if self.target_marker_id is not None:
            marker_slot = SLOT_BY_MARKER.get(int(self.target_marker_id))
            if marker_slot is not None:
                return tuple(marker_slot)
        if self.target_world is None:
            return None
        try:
            return fixed_slot_from_world(
                float(self.target_world[0]), float(self.target_world[2]))
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

        try:
            world = np.asarray(detection.get("world"), dtype=float)
            target = np.asarray(self.target_world, dtype=float)
        except (TypeError, ValueError):
            return False, "depth"
        if (world.shape != (3,) or target.shape != (3,)
                or not np.all(np.isfinite(world))
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
        """Skip this slot and resume scanning."""
        marker_id = self.target_marker_id
        if marker_id is not None:
            self.recheck_marker_skips.add(marker_id)
        else:
            skipped_slot = self.target_slot_key()
            if skipped_slot is not None:
                self.excluded_slot_keys.add(skipped_slot)
        self.get_logger().warn(
            f"[close-recheck] FAILED marker={marker_id} "
            f"kind={self.target_kind}; all close-view poses exhausted; "
            "skipping this slot and resuming the shelf scan")
        self._recheck_passed = False
        self.recheck_poses = ()
        self.recheck_confirmation_times.clear()
        self.recheck_last_yolo_stamp = None
        self.scan_camera_ready_since = None
        self.target_marker_id = None
        self.target_physical_marker_id = None
        self.target_world = None
        self.committed_slot = None
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

    def apply_manip_base_hold(self) -> None:
        """Softly oppose top/middle/lower reaction forces; never block the arm."""
        active_states = (
            STATE_DEPLOY, STATE_ARM_FORWARD, STATE_POST_EXTEND,
            STATE_CLOSE, STATE_TRIAL_LIFT, STATE_LIFT)
        if (self.shelf_level not in ("top", "middle", "lower")
                or self.state not in active_states
                or self.manip_base_hold_xy is None
                or self.manip_base_hold_yaw is None):
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
            self.scan_poses = SCAN_CAMERA_POSES[1:4]
            level = "middle"
        else:
            self.scan_poses = SCAN_CAMERA_POSES[4:]
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
            return SCAN_CAMERA_POSES[1:4]
        return SCAN_CAMERA_POSES[4:]

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

        固定格只提供几何，不伪造一次真实的 marker 解码。后续仍执行
        与正常路径相同的近距离 YOLO 深度复核。
        """
        entry = self.scan_unlocked_boxes.get(self.revisit_box_key)
        if not isinstance(entry, dict):
            return False
        try:
            confirmations = int(entry.get("confirmations", 0))
            confidence = float(entry.get("max_conf", 0.0))
            samples = np.asarray(list(entry.get("worlds", ())), dtype=float)
        except (TypeError, ValueError):
            return False
        if (confirmations < YOLO_ONLY_TARGET_CONFIRMATIONS
                or samples.ndim != 2 or samples.shape[1:] != (3,)
                or len(samples) < YOLO_ONLY_TARGET_CONFIRMATIONS
                or not np.all(np.isfinite(samples))):
            return False
        spread = float(np.max(np.ptp(samples, axis=0)))
        if spread > YOLO_ONLY_TARGET_SPREAD_MAX_M:
            self.get_logger().warn(
                "[position-fallback] skipped: YOLO-only sample spread "
                f"{spread:.3f}m > {YOLO_ONLY_TARGET_SPREAD_MAX_M:.3f}m")
            return False
        if confidence < YOLO_ONLY_TARGET_CONF_MIN:
            self.get_logger().warn(
                "[position-fallback] skipped: box conf "
                f"{confidence:.3f} < {YOLO_ONLY_TARGET_CONF_MIN:.3f}")
            return False
        box_world = np.median(samples, axis=0)
        slot = fixed_slot_from_world(
            float(box_world[0]), float(box_world[2]),
            shelf=self._current_station_shelf())
        if slot is None:
            return False
        shelf, level, column = slot
        slot_key = f"{level}|{shelf}|{column}"
        if slot_key in self.excluded_slot_keys:
            return False
        level_name = {"L3": "top", "L2": "middle", "L1": "lower"}[level]
        z = min(
            float(box_world[2]),
            SHELF_SURFACE_Z_M[level_name]
            + PRODUCT_HALF_HEIGHT_M[self.target_kind]
            + PRODUCT_CENTER_Z_TOLERANCE_M)
        column_x = (
            SCAN_X[self._scan_x_index_for_shelf(shelf)]
            + {"1": -0.22, "2": 0.00, "3": 0.22}[column])
        target_world = np.array([
            column_x,
            float(box_world[1] + PRODUCT_HALF_DEPTH_M[self.target_kind]),
            z], dtype=float)
        self.get_logger().warn(
            f"[position-fallback] ArUco undecodable; "
            f"using YOLO world {np.round(box_world, 3)} -> target "
            f"{np.round(target_world, 3)} kind={self.target_kind} "
            f"slot={slot}")
        self._commit_yolo_only_target(target_world, slot)
        return self.target_world is not None and self.state == STATE_ALIGN

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
                 linear_min_mps: float | None = None) -> bool:
        delta = np.asarray(target_xy, dtype=float) - self.base_xy
        distance = float(np.linalg.norm(delta))
        if distance > position_tolerance:
            desired_yaw = math.atan2(delta[1], delta[0])
            yaw_error = wrap_to_pi(desired_yaw - self.base_yaw)
            if abs(yaw_error) > NAV_ROTATE_GATE_RAD:
                if self._rotate_with_unstick(desired_yaw):
                    return False
                self.set_twist(0.0, float(np.clip(
                    2.2 * yaw_error, -NAV_ANGULAR_MAX_RADPS,
                    NAV_ANGULAR_MAX_RADPS)))
            else:
                linear = float(np.clip(
                    1.2 * distance,
                    NAV_LINEAR_MIN_MPS if linear_min_mps is None
                    else linear_min_mps,
                    NAV_LINEAR_MAX_MPS))
                self.set_twist(linear, float(np.clip(
                    1.8 * yaw_error, -NAV_TRANSLATE_ANGULAR_MAX_RADPS,
                    NAV_TRANSLATE_ANGULAR_MAX_RADPS)))
            return False
        yaw_error = wrap_to_pi(final_yaw - self.base_yaw)
        if abs(yaw_error) > NAV_YAW_DEADBAND_RAD:
            if self._rotate_with_unstick(final_yaw):
                return False
            self.set_twist(0.0, float(np.clip(
                2.0 * yaw_error, -NAV_ANGULAR_MAX_RADPS,
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
                if self._rot_stall_unsticks > NAV_ROT_UNSTICK_MAX:
                    # 多次解除仍卡死：重置继续尝试旋转，由上层超时/重试兜底
                    self._rot_stall_unsticks = 0
                    self._rot_stall_anchor_yaw = float(self.base_yaw)
                    self._rot_stall_anchor_t = now
                    self._rot_stall_anchor_xy = self.base_xy.copy()
                    self.get_logger().warn(
                        "[rotate-unstick] still stuck after multiple "
                        "attempts; continuing rotation")
                    return False
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

    def selected_gripper_position(self) -> float | None:
        side = "left" if self.grasp_arm == "l" else "right"
        value = self.joints.get(f"{side}_arm_eef_gripper_joint")
        if value is None or not math.isfinite(float(value)):
            return None
        return float(value)

    def grasp_profile_name(self) -> str:
        """Return the composed layer/geometry profile used by this target."""
        if self.use_dual_tissue_grasp:
            return f"{self.shelf_level}_dual_tissue"
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
            right_reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve one symmetric two-arm pose at the current slide height."""
        left_target = np.eye(4)
        right_target = np.eye(4)
        left_target[:3, 3] = self.world_to_footprint(left_world)
        right_target[:3, 3] = self.world_to_footprint(right_world)
        reference = np.concatenate((
            [self.slide_grasp],
            np.asarray(left_reference, dtype=float),
            np.asarray(right_reference, dtype=float)))
        solutions = self.kdl.inverse_kinematics(
            T_left=left_target,
            T_right=right_target,
            target_height=self.slide_grasp,
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

    def configure_dual_tissue_grasp(self) -> bool:
        """Prepare a symmetric side clamp for the tissue box at any level.

        全部列统一使用"直接探入"动作（wxj v2 策略）：张开双臂宽跨度前探 →
        到位后合拢到夹持跨度压紧纸盒两侧 → 保持夹持收回。直线前探不横向
        扫掠，天然避开货架前立柱，无需再绕柱，也不再跳过墙侧列。
        """
        surface_z = SHELF_SURFACE_Z_M[self.shelf_level]
        self.dual_clamp_half_span = DUAL_TISSUE_CLAMP_HALF_SPAN_M
        self.dual_insert_forward_m = DUAL_TISSUE_INSERT_FORWARD_M
        self.dual_direct_probe = True
        self.dual_pregrasp_half_span = DUAL_TISSUE_DIRECT_PROBE_SPAN_M
        self.dual_squeeze_m = DUAL_TISSUE_SQUEEZE_M
        tcp_clearance = (
            DUAL_TISSUE_LOWER_TCP_CLEARANCE_M
            if self.shelf_level == "lower"
            else DUAL_TISSUE_TCP_CLEARANCE_M)
        tcp_z = max(
            float(self.target_world[2] + GRASP_TCP_Z_RAISE_M),
            surface_z + tcp_clearance)
        self.dual_contact_tcp_z = float(tcp_z)
        insert_y = (
            self.target_world[1] + self.dual_insert_forward_m)

        def pair(half_span: float, y: float):
            left = np.array([
                self.target_world[0] - half_span, y, tcp_z], dtype=float)
            right = np.array([
                self.target_world[0] + half_span, y, tcp_z], dtype=float)
            return left, right

        pre_left, pre_right = pair(
            self.dual_pregrasp_half_span,
            self.target_world[1] - DUAL_TISSUE_PREGRASP_BACKOFF_M)
        surround_left, surround_right = pair(
            self.dual_pregrasp_half_span, insert_y)
        clamp_left, clamp_right = pair(
            self.dual_clamp_half_span, insert_y)
        retreat_left, retreat_right = pair(
            self.dual_clamp_half_span,
            self.target_world[1] - DUAL_TISSUE_PREGRASP_BACKOFF_M)
        left_reference = self.cmd_left_arm.copy()
        right_reference = self.cmd_right_arm.copy()
        try:
            pre_left_joints, pre_right_joints = self.solve_kdl_both_world(
                pre_left, pre_right, left_reference, right_reference)
            # 直接探入：先以宽跨度直探到 insert_y，再闭合到夹持跨度。
            surround_left_joints, surround_right_joints = (
                self.solve_kdl_both_world(
                    surround_left, surround_right,
                    pre_left_joints, pre_right_joints))
            close_left, close_right = pair(
                self.dual_clamp_half_span, insert_y)
            self.dual_surround_close_left_joints, (
                self.dual_surround_close_right_joints) = (
                self.solve_kdl_both_world(
                    close_left, close_right,
                    surround_left_joints, surround_right_joints))
            clamp_left_joints, clamp_right_joints = (
                self.solve_kdl_both_world(
                    clamp_left, clamp_right,
                    surround_left_joints, surround_right_joints))
            retreat_left_joints, retreat_right_joints = (
                self.solve_kdl_both_world(
                    retreat_left, retreat_right,
                    clamp_left_joints, clamp_right_joints))
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

        self.dual_pregrasp_left_joints = pre_left_joints
        self.dual_pregrasp_right_joints = pre_right_joints
        self.dual_surround_left_joints = surround_left_joints
        self.dual_surround_right_joints = surround_right_joints
        # 绕柱段已废弃（v2 全列直接探入）：pass/forward/return 关节始终
        # 为 None，状态机只走前探→合拢→压紧→收回流程。
        self.dual_surround_pass_left_joints = None
        self.dual_surround_pass_right_joints = None
        self.dual_surround_forward_left_joints = None
        self.dual_surround_forward_right_joints = None
        self.dual_surround_return_left_joints = None
        self.dual_surround_return_right_joints = None
        self.dual_surround_stage = 0
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
            f"grippers=closed({DUAL_TISSUE_GRIP_COMMAND:.2f})")
        return True

    def configure_dual_tissue_arm_lift(self) -> bool:
        """Lift via the arm joints when the slide has no upward headroom.

        On the top shelf the slide is already pinned at SLIDE_MIN to reach the
        shelf height, so the slide-based lift is a no-op and the retreat would
        drag the box across the board.  Instead solve a +DUAL_TISSUE_LIFT_M
        TCP pose plus a raised horizontal retreat for both arms.
        """
        left_tcp = self.arm_tcp_world("left")
        right_tcp = self.arm_tcp_world("right")
        if left_tcp is None or right_tcp is None:
            self.get_logger().error(
                "[dual-tissue-arm-lift] measured TCP unavailable")
            return False
        lift_z = 0.5 * (left_tcp[2] + right_tcp[2]) + DUAL_TISSUE_LIFT_M
        lift_left = np.array([left_tcp[0], left_tcp[1], lift_z])
        lift_right = np.array([right_tcp[0], right_tcp[1], lift_z])
        retreat_y = (
            self.target_world[1] - DUAL_TISSUE_PREGRASP_BACKOFF_M)
        retreat_left = np.array([left_tcp[0], retreat_y, lift_z])
        retreat_right = np.array([right_tcp[0], retreat_y, lift_z])
        left_reference = self.arm_positions("left")
        right_reference = self.arm_positions("right")
        try:
            lift_left_joints, lift_right_joints = (
                self.solve_kdl_both_world(
                    lift_left, lift_right,
                    left_reference, right_reference))
            retreat_left_joints, retreat_right_joints = (
                self.solve_kdl_both_world(
                    retreat_left, retreat_right,
                    lift_left_joints, lift_right_joints))
        except ValueError as exc:
            self.get_logger().error(
                f"[dual-tissue-arm-lift] IK failed: {exc}")
            return False
        self.dual_lift_left_joints = lift_left_joints.copy()
        self.dual_lift_right_joints = lift_right_joints.copy()
        self.dual_lift_retreat_left_joints = retreat_left_joints.copy()
        self.dual_lift_retreat_right_joints = retreat_right_joints.copy()
        self.get_logger().info(
            f"[dual-tissue-arm-lift] slide pinned at SLIDE_MIN; "
            f"lift={DUAL_TISSUE_LIFT_M:.3f}m via arm joints "
            f"left_start={np.round(left_tcp, 4)} "
            f"right_start={np.round(right_tcp, 4)} "
            f"lift_left={np.round(lift_left, 4)} "
            f"lift_right={np.round(lift_right, 4)} "
            f"retreat_y={retreat_y:.3f}")
        return True

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
            + GENERIC_TCP_FINGER_CLEARANCE_M))

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
        extended_contact_world = nominal_contact_world.copy()
        extended_contact_world[1] += (
            GENERIC_POST_EXTEND_M_BY_KIND.get(
                self.target_kind, GENERIC_POST_CONTACT_EXTENSION_M))

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
                f"{GENERIC_POST_CONTACT_EXTENSION_M:.3f}m "
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
            + GENERIC_TCP_FINGER_CLEARANCE_M))
        # This is a wrist-to-inner-finger geometry transform, not a perception
        # correction, so it remains active in fixed-layout diagnostic mode.
        nominal_contact_world[1] += LOWER_GRASP_TCP_FORWARD_M
        post_extension_enabled = self.object_geometry != "sphere"
        extended_contact_world = nominal_contact_world.copy()
        if post_extension_enabled:
            extended_contact_world[1] += (
                GENERIC_POST_EXTEND_M_BY_KIND.get(
                    self.target_kind, GENERIC_POST_CONTACT_EXTENSION_M))

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
                f"{GENERIC_POST_CONTACT_EXTENSION_M if post_extension_enabled else 0.0:.3f}m "
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
        """Raise to a front-facing top-shelf pregrasp, then extend horizontally."""
        grasp_tcp_z = self.top_grasp_tcp_z()
        pregrasp_world = self.target_world.copy()
        pregrasp_world[2] = grasp_tcp_z
        pregrasp_world[1] -= TOP_PREGRASP_BACKOFF_M
        nominal_contact_world = self.target_world.copy()
        nominal_contact_world[2] = grasp_tcp_z
        nominal_contact_world[1] += TOP_GRASP_TCP_FORWARD_M
        extended_contact_world = nominal_contact_world.copy()
        extended_contact_world[1] += GENERIC_POST_CONTACT_EXTENSION_M

        reference = (self.cmd_right_arm.copy() if self.grasp_arm == "r"
                     else self.cmd_left_arm.copy())
        arm_base_rotation = (
            MMK2FIK.TMat_chest2rgt_base[:3, :3]
            if self.grasp_arm == "r"
            else MMK2FIK.TMat_chest2lft_base[:3, :3])
        # Identity in the footprint frame points the gripper squarely toward
        # the shelf.  Pregrasp and contact have identical Z and orientation;
        # after the high pose is reached there is no downward component and no
        # intermediate correction target that could make the arm shake.
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
            speed: float, state: str) -> None:
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
        self.dual_motion_endpoint_ready_since = None
        self.des_left_arm = self.dual_motion_start_left.copy()
        self.des_right_arm = self.dual_motion_start_right.copy()
        self.commands_ready_since = None
        self.get_logger().info(
            f"[dual-tissue-{label}] fixed synchronized segment armed; "
            f"path={path_length:.3f}m "
            f"duration={self.dual_motion_duration_s:.2f}s "
            f"speed={speed:.3f}m/s feedback_gates=0 replanning=0")
        self.set_state(state)
        # The post-band dogleg re-enters STATE_ARM_FORWARD for each segment;
        # set_state skips the timestamp on an unchanged state, so reset the
        # segment timer explicitly here.
        self.state_t0 = self.now()

    def start_dual_tissue_surround(self) -> None:
        """Move both open grippers straight forward around the tissue sides.

        全列统一直接探入（v2 策略）：宽跨度直线前探，无狗腿/合拢中间段。
        """
        self.forward_start_base_xy = self.base_xy.copy()
        forward = (
            DUAL_TISSUE_PREGRASP_BACKOFF_M
            + self.dual_insert_forward_m)
        lateral = (
            self.dual_pregrasp_half_span
            - DUAL_TISSUE_SURROUND_HALF_SPAN_M)
        self.start_dual_tissue_motion(
            "surround",
            self.dual_surround_left_joints,
            self.dual_surround_right_joints,
            math.hypot(forward, lateral),
            DUAL_TISSUE_FORWARD_SPEED_MPS,
            STATE_ARM_FORWARD)

    def start_dual_tissue_contact_search(self) -> None:
        """Move each arm monotonically inward until its own contact."""
        left_tcp = self.arm_tcp_world("left")
        right_tcp = self.arm_tcp_world("right")
        if left_tcp is None or right_tcp is None:
            self.get_logger().error(
                "[dual-tissue-contact] measured TCP unavailable")
            self.set_state(STATE_ABORT)
            return

        common_y = 0.5 * (left_tcp[1] + right_tcp[1])
        common_z = 0.5 * (left_tcp[2] + right_tcp[2])
        left_goal = np.array([
            self.target_world[0] - DUAL_TISSUE_CONTACT_SEARCH_HALF_SPAN_M,
            common_y, common_z])
        right_goal = np.array([
            self.target_world[0] + DUAL_TISSUE_CONTACT_SEARCH_HALF_SPAN_M,
            common_y, common_z])
        left_start_joints = self.arm_positions("left")
        right_start_joints = self.arm_positions("right")
        try:
            left_goal_joints, right_goal_joints = self.solve_kdl_both_world(
                left_goal, right_goal,
                left_start_joints, right_start_joints)
        except ValueError as exc:
            self.get_logger().error(
                f"[dual-tissue-contact] search IK failed: {exc}")
            self.set_state(STATE_ABORT)
            return

        left_path = max(0.0, left_goal[0] - left_tcp[0])
        right_path = max(0.0, right_tcp[0] - right_goal[0])
        path_length = max(left_path, right_path)
        self.dual_contact_start_left_joints = left_start_joints
        self.dual_contact_start_right_joints = right_start_joints
        self.dual_contact_target_left_joints = left_goal_joints
        self.dual_contact_target_right_joints = right_goal_joints
        self.dual_contact_start_left_world = left_tcp.copy()
        self.dual_contact_start_right_world = right_tcp.copy()
        self.dual_contact_goal_left_world = left_goal
        self.dual_contact_goal_right_world = right_goal
        self.dual_contact_duration_s = max(
            DUAL_TISSUE_MIN_MOTION_DURATION_S,
            1.5 * path_length / DUAL_TISSUE_CONTACT_SEARCH_SPEED_MPS)
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
            f"speed={DUAL_TISSUE_CONTACT_SEARCH_SPEED_MPS:.3f}m/s "
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
        progress = float(np.clip(elapsed / duration, 0.0, 1.0))
        eased = progress * progress * (3.0 - 2.0 * progress)
        if self.dual_left_contacted:
            self.des_left_arm = self.dual_left_contact_hold_joints.copy()
        else:
            self.des_left_arm = (
                self.dual_contact_start_left_joints
                + eased * (self.dual_contact_target_left_joints
                           - self.dual_contact_start_left_joints))
        if self.dual_right_contacted:
            self.des_right_arm = self.dual_right_contact_hold_joints.copy()
        else:
            self.des_right_arm = (
                self.dual_contact_start_right_joints
                + eased * (self.dual_contact_target_right_joints
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
            commanded = eased * total
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
            commanded = eased * total
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
            if not self.dual_left_contacted:
                self.get_logger().warn(
                    "[dual-tissue-contact] left contact was not detected; "
                    "using its final monotonic endpoint")
                self.mark_dual_tissue_contact("left", "timeout", left_tcp)
            if not self.dual_right_contacted:
                self.get_logger().warn(
                    "[dual-tissue-contact] right contact was not detected; "
                    "using its final monotonic endpoint")
                self.mark_dual_tissue_contact("right", "timeout", right_tcp)

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
        """Add equal inward preload and preserve it through retreat."""
        left_tcp = self.arm_tcp_world("left")
        right_tcp = self.arm_tcp_world("right")
        if left_tcp is None or right_tcp is None:
            self.get_logger().error(
                "[dual-tissue-squeeze] measured TCP unavailable")
            return False
        common_y = 0.5 * (left_tcp[1] + right_tcp[1])
        common_z = 0.5 * (left_tcp[2] + right_tcp[2])
        # Anchor the clamp on the measured box centre -- the midpoint of the
        # two bilateral contact TCPs brackets the real box.  Solving the clamp
        # around the assumed target centre instead made one gripper press into
        # a displaced box while the other floated, producing the crooked grab.
        box_centre_x = 0.5 * (left_tcp[0] + right_tcp[0])
        clamp_left = np.array([
            box_centre_x - self.dual_clamp_half_span,
            common_y, common_z])
        clamp_right = np.array([
            box_centre_x + self.dual_clamp_half_span,
            common_y, common_z])
        squeeze_left = np.array([
            left_tcp[0] + self.dual_squeeze_m, common_y, common_z])
        squeeze_right = np.array([
            right_tcp[0] - self.dual_squeeze_m, common_y, common_z])
        left_reference = self.arm_positions("left")
        right_reference = self.arm_positions("right")
        try:
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
            retreat_y = (
                self.target_world[1] - DUAL_TISSUE_PREGRASP_BACKOFF_M)
            retreat_left[1] = retreat_y
            retreat_right[1] = retreat_y
            retreat_left_joints, retreat_right_joints = (
                self.solve_kdl_both_world(
                    retreat_left, retreat_right,
                    clamp_left_joints, clamp_right_joints))
        except ValueError as exc:
            self.get_logger().error(
                f"[dual-tissue-squeeze] IK failed: {exc}")
            return False

        self.dual_clamp_left_joints = clamp_left_joints.copy()
        self.dual_clamp_right_joints = clamp_right_joints.copy()
        self.dual_retreat_left_joints = retreat_left_joints.copy()
        self.dual_retreat_right_joints = retreat_right_joints.copy()
        self.get_logger().info(
            f"[dual-tissue-squeeze] equal preload="
            f"{self.dual_squeeze_m:.3f}m/side "
            f"box_centre={box_centre_x:.4f} "
            f"left={np.round(squeeze_left, 4)} "
            f"right={np.round(squeeze_right, 4)} "
            f"retreat_y={retreat_y:.3f}")
        self.start_dual_tissue_motion(
            "squeeze", squeeze_left_joints, squeeze_right_joints,
            self.dual_squeeze_m,
            DUAL_TISSUE_SQUEEZE_SPEED_MPS,
            STATE_DUAL_SQUEEZE)
        return True

    def advance_dual_tissue_motion(self) -> str:
        """Play a monotonic dual-arm segment without corrective oscillation."""
        elapsed = self.now() - self.state_t0
        duration = max(self.dual_motion_duration_s, 1e-6)
        progress = float(np.clip(elapsed / duration, 0.0, 1.0))
        eased = progress * progress * (3.0 - 2.0 * progress)
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
        endpoint_ready = (
            arm_error <= ARM_REACHED_TOLERANCE_RAD + 0.015)
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

        left_tcp = self.arm_tcp_world("left")
        right_tcp = self.arm_tcp_world("right")
        self.get_logger().info(
            f"[dual-tissue-{self.dual_motion_label}] segment complete; "
            f"elapsed={elapsed:.2f}s "
            f"settle={settle_elapsed:.2f}s "
            f"arm_error={arm_error:.4f}rad "
            f"endpoint_stable={int(endpoint_stable)} "
            f"left_tcp={None if left_tcp is None else np.round(left_tcp, 4)} "
            f"right_tcp={None if right_tcp is None else np.round(right_tcp, 4)}; "
            "continuing without a TCP convergence gate")
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
        """Continue 50 mm beyond a profile's established close point."""
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
            f"[{self.shelf_level}-post-extend] 50 mm continuation complete; "
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
            combined = self.synchronized_slew(
                np.concatenate((self.cmd_left_arm, self.cmd_right_arm)),
                np.concatenate((self.des_left_arm, self.des_right_arm)),
                ARM_COMMAND_MAX_STEP_RAD)
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
                    if self.revisit_pose_index >= len(self.revisit_poses):
                        self._revisit_fail()
            else:
                self.scan_camera_ready_since = None

        elif self.state == STATE_ALIGN:
            align_x_tolerance = 0.025
            if self.drive_to(
                    [self.align_base_x, self.align_base_y],
                    YAW_NORTH, align_x_tolerance,
                    linear_min_mps=NAV_ALIGN_LINEAR_MIN_MPS):
                if self.close_recheck and not self._recheck_passed:
                    self.set_state(STATE_RECHECK)
                    self._start_close_recheck()
                else:
                    grasp_status = self.configure_grasp()
                    if grasp_status is True:
                        if self.shelf_level in ("top", "middle", "lower"):
                            self.begin_manip_base_hold()
                        self.set_state(STATE_DEPLOY)
                    elif grasp_status != "retry":
                        self.set_state(STATE_ABORT)

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
                    grasp_status = self.configure_grasp()
                    if grasp_status is True:
                        if self.shelf_level in ("top", "middle", "lower"):
                            self.begin_manip_base_hold()
                        self.set_state(STATE_DEPLOY)
                    elif grasp_status == "retry":
                        self.set_state(STATE_ALIGN)
                    else:
                        self.set_state(STATE_ABORT)
            else:
                self.scan_camera_ready_since = None

            if (self.state == STATE_RECHECK
                    and self.now() - self.recheck_pose_started_at
                    >= CLOSE_RECHECK_POSE_TIMEOUT_S):
                if not self._advance_recheck_pose():
                    self._recheck_fail()

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
            deploy_ready = self.commands_ready(
                MIDDLE_SPHERE_CORRECTION_ARM_TOLERANCE_RAD
                if middle_sphere else ARM_REACHED_TOLERANCE_RAD,
                MIDDLE_SPHERE_CORRECTION_SLIDE_TOLERANCE_M
                if middle_sphere else 0.025)

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
            if (self.use_dual_tissue_grasp
                    and deploy_elapsed >= DUAL_TISSUE_DEPLOY_DWELL_S):
                if (self.dual_surround_left_joints is None
                        or self.dual_surround_right_joints is None):
                    self.get_logger().error(
                        "dual tissue approach has no solved endpoint")
                    self.set_state(STATE_ABORT)
                    return
                self.get_logger().info(
                    f"[dual-tissue-deploy] dwell complete after "
                    f"{deploy_elapsed:.2f}s; both closed grippers are at "
                    "the symmetric pregrasp; starting fixed surround motion")
                self.start_dual_tissue_surround()
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
                    if self.dual_surround_stage == 0:
                        # 直接探入（全列统一）：宽跨度前探完成。改用双臂
                        # 独立接触搜索：每只手臂向内推进到"自己"碰到盒壁
                        # 即停（stall 检测），另一只继续跟进，避免固定合拢
                        # 行程下左右臂先后撞盒、夹持不对称。两侧都接触后
                        # 再用实测中点对称压紧。
                        self.dual_surround_stage = 1
                        self.start_dual_tissue_contact_search()
                    else:
                        # 双臂均已接触盒壁，按实测中点对称压紧。
                        if not self.start_dual_tissue_squeeze():
                            self.set_state(STATE_ABORT)
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
                    # Top shelf: the slide is already pinned at its upper
                    # limit, so the slide lift would be a no-op.  Lift via the
                    # arm joints so the box clears the board before retreat.
                    # There is no shelf above this layer.
                    if self.configure_dual_tissue_arm_lift():
                        self.dual_lift_use_arm = True
                    else:
                        self.get_logger().error(
                            "[dual-tissue] top-shelf arm lift IK failed; "
                            "refusing a same-height retreat across the board")
                        self.set_state(STATE_ABORT)
                        return
                    self.dual_lift_settled_since = None
                    self.get_logger().info(
                        f"[dual-tissue-clamp] lateral squeeze held for "
                        f"{close_elapsed:.2f}s; top shelf has no overhead "
                        "board, performing the established arm lift")
                    self.set_state(STATE_LIFT)
                else:
                    # Match wxj: middle/lower tissue leaves the shelf at its
                    # measured grasp height before transport-height recovery.
                    self.get_logger().info(
                        f"[dual-tissue-clamp] lateral squeeze held for "
                        f"{close_elapsed:.2f}s; retracting at grasp height "
                        "before restoring transport height")
                    self.start_dual_tissue_motion(
                        "retreat_at_grasp_height",
                        self.dual_retreat_left_joints,
                        self.dual_retreat_right_joints,
                        DUAL_TISSUE_PREGRASP_BACKOFF_M
                        + self.dual_insert_forward_m,
                        DUAL_TISSUE_RETREAT_SPEED_MPS,
                        STATE_RETREAT)

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
                            # Match wxj: a middle-shelf sphere withdraws at
                            # the exact grasp height; top spheres retain the
                            # established trial/full arm-lift sequence.
                            self.des_slide = self.sphere_slide_command
                            self.sphere_retreat_arm_joints = (
                                self.pregrasp_arm_joints.copy())
                            self.set_selected_arm_target(
                                self.sphere_retreat_arm_joints)
                            self.get_logger().info(
                                "[middle-sphere-retreat] retracting at "
                                "grasp height before any vertical motion")
                            self.set_state(STATE_RETREAT)
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
                self.des_left_arm = self.dual_lift_left_joints.copy()
                self.des_right_arm = self.dual_lift_right_joints.copy()
                self.des_slide = self.slide_grasp
                if self.dual_arm_error() < ARM_REACHED_TOLERANCE_RAD + 0.015:
                    if self.dual_lift_settled_since is None:
                        self.dual_lift_settled_since = self.now()
                else:
                    self.dual_lift_settled_since = None
                if (self.dual_lift_settled_since is not None
                        and self.now() - self.dual_lift_settled_since
                        >= 0.25):
                    self.get_logger().info(
                        f"[dual-tissue-arm-lift] raised "
                        f"{DUAL_TISSUE_LIFT_M:.3f}m via the arm joints; "
                        "retracting horizontally at the raised height")
                    self.start_dual_tissue_motion(
                        "retreat",
                        self.dual_lift_retreat_left_joints,
                        self.dual_lift_retreat_right_joints,
                        DUAL_TISSUE_PREGRASP_BACKOFF_M
                        + self.dual_insert_forward_m,
                        DUAL_TISSUE_RETREAT_SPEED_MPS,
                        STATE_RETREAT)
                elif lift_elapsed >= DUAL_TISSUE_LIFT_DWELL_S + 3.0:
                    self.get_logger().warn(
                        "[dual-tissue-arm-lift] lift did not fully settle; "
                        "continuing the raised retreat")
                    self.start_dual_tissue_motion(
                        "retreat",
                        self.dual_lift_retreat_left_joints,
                        self.dual_lift_retreat_right_joints,
                        DUAL_TISSUE_PREGRASP_BACKOFF_M
                        + self.dual_insert_forward_m,
                        DUAL_TISSUE_RETREAT_SPEED_MPS,
                        STATE_RETREAT)
            else:
                self.des_left_arm = self.dual_clamp_left_joints.copy()
                self.des_right_arm = self.dual_clamp_right_joints.copy()
                self.des_slide = max(
                    SLIDE_MIN, self.slide_grasp - DUAL_TISSUE_LIFT_M)
                if lift_elapsed >= DUAL_TISSUE_LIFT_DWELL_S:
                    measured_slide = self.joints.get("slide_joint")
                    self.get_logger().info(
                        f"[dual-tissue-lift] commanded "
                        f"{DUAL_TISSUE_LIFT_M:.3f}m upward slide motion; "
                        f"target_slide={self.des_slide:.3f} "
                        f"measured_slide={measured_slide}; "
                        "starting slow synchronized retreat at the raised "
                        "height")
                    self.start_dual_tissue_motion(
                        "retreat",
                        self.dual_retreat_left_joints,
                        self.dual_retreat_right_joints,
                        DUAL_TISSUE_PREGRASP_BACKOFF_M
                        + self.dual_insert_forward_m,
                        DUAL_TISSUE_RETREAT_SPEED_MPS,
                        STATE_RETREAT)

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
                lift_elapsed = self.now() - self.state_t0
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
                    # Established middle-layer generic lift remains unchanged.
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

        elif (self.state == STATE_RETREAT
              and self.use_dual_tissue_grasp):
            self.des_left_grip = DUAL_TISSUE_GRIP_COMMAND
            self.des_right_grip = DUAL_TISSUE_GRIP_COMMAND
            if self.advance_dual_tissue_motion() == "reached":
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
